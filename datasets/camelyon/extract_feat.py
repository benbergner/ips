#!/usr/bin/env python
import os
import h5py
from pathlib import Path
import argparse
import yaml
from tqdm import tqdm
import pandas as pd
import torch

from torch.utils.data import DataLoader
from pretraining.model.byol_model import BYOLModel
from datasets.camelyon.cam_dataset import CAMELYON16Dataset, RandomPatchSampler

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(
    description="Extract SSL features from foreground patches for each slide"
)

parser.add_argument('--train', dest='is_train', action='store_true')
parser.add_argument('--test', dest='is_train', action='store_false')
parser.set_defaults(is_train=True)

parser.add_argument(
    "--lvl",
    type=int,
    default=0,
    help="Choose the magnification level (0: highest) for feature extraction"
)
parser.add_argument(
    "--otsu_lvl",
    type=int,
    default=0,
    help="Choose the magnification level for otsu threshold"
)
parser.add_argument(
    "--tile_size",
    type=int,
    default=256,
    help="Choose the tile size"
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=256,
    help="Choose the batch size"
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=16,
    help="Number of workers for data loading"
)
parser.add_argument(
    "data_dir",
    help="The directory where the CAMELYON16 dataset is located"
)
parser.add_argument(
    "bounds_dir",
    help="The directory where the bounds file is located"
)
parser.add_argument(
    "coords_dir",
    help="The directory where the coords file is located"
)
parser.add_argument(
    "model_dir",
    help="The directory where the model checkpoint is located"
)
parser.add_argument(
    "feat_save_dir",
    help="The directory where the features shall be located"
)

args = parser.parse_args()

# Get arguments
train = args.is_train
lvl = args.lvl
otsu_lvl = args.otsu_lvl
tile_size = args.tile_size
batch_size = args.batch_size
num_workers = args.num_workers
data_dir = args.data_dir
bounds_dir = args.bounds_dir
coords_dir = args.coords_dir
model_dir = args.model_dir
feat_save_dir = args.feat_save_dir

"""
model_load_path_byol = 'models/camelyon16/byol/resnet50_' + str(n_epoch_load) + '.pth.tar'

specifier_train = which_dataset + '_lvl' + str(level) + '_otsu_lvl' + str(otsu_level) + '_size' + str(tile_size) + '_thresh' + str(poi_threshold) + '_' + add_str + '_train.pkl'
specifier_test = which_dataset + '_lvl' + str(level) + '_otsu_lvl' + str(otsu_level) + '_size' + str(tile_size) + '_thresh' + str(poi_threshold) + '_' + add_str + '_test.pkl'
dset_path16 = "/dhc/dsets/CAMELYON/CAMELYON16/"
print("specifier_train: ", specifier_train)

coords_path_train = os.path.join("data", "camelyon", "data_info_files", "coords_" + specifier_train)
bounds_path_train = os.path.join("data", "camelyon", "data_info_files", "bounds_" + specifier_train)
coords_path_test = os.path.join("data", "camelyon", "data_info_files", "coords_" + specifier_test)
bounds_path_test = os.path.join("data", "camelyon", "data_info_files", "bounds_" + specifier_test)

if train:
    #feat_save_path = 'data/camelyon/data_info_files/resnet50_cam16_' + str(level) + '_' + which_backbone + str(n_epoch_load) + '_train.hdf5'
    feat_save_path = 'data/camelyon/data_info_files/' + which_backbone + '_cam16_' + str(level) + '_thresh' + str(poi_threshold) + '_' + which_backbone + str(n_epoch_load) + '_' + add_str + '_centercr_train.hdf5'
else:
    #feat_save_path = 'data/camelyon/data_info_files/resnet50_cam16_' + str(level) + '_' + which_backbone + str(n_epoch_load) + '_test.hdf5'
    feat_save_path = 'data/camelyon/data_info_files/' + which_backbone + '_cam16_' + str(level) + '_thresh' + str(poi_threshold) + '_' + which_backbone + str(n_epoch_load) + '_' + add_str + '_centercr_test.hdf5'
"""

bounds_df = pd.read_pickle(bounds_dir)
coords_df = pd.read_pickle(coords_dir)
sampler = RandomPatchSampler(bounds_df, batch_size=batch_size, patch_shuffle=False, slide_shuffle=False)
dataset = CAMELYON16Dataset(data_dir, coords_df, lvl, tile_size)
dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

h5file = h5py.File(feat_save_dir, "w")

# Load pre-trained model
with open(Path('pretraining/config/train_config.yaml'), 'r') as f:
    config = yaml.safe_load(f)
net = BYOLModel(config)
checkpoint = torch.load(model_dir, map_location=device)
loaded_dict = checkpoint['model']
prefix = 'module.'
n_clip = len(prefix)
adapted_dict = {k[n_clip:]: v for k, v in loaded_dict.items()
            if k.startswith(prefix)}
net.load_state_dict(adapted_dict, strict=True)
net = net.online_network.encoder
net.to(device)

net.eval()
with torch.no_grad():
    current_slide = None
    feature_list = []
    pos_idx_list = []
    num_processed = 0
    for data in dataloader:
        # Get data
        patches = data['patch'].to(device)
        pos_idx = data['pos_id'].to(device)
        data_idx = data['data_id'].to(device)
        slide_names = [name for name in data['slide_name'] if name]
        if len(slide_names) > 0:
            slide_label = data['label'].max() # either 0 or 1, but not negative
            slide_name = slide_names[0] # slide names are same after filtering

        # Check if new slide
        is_new_slide = slide_name != current_slide
        if is_new_slide:
            feature_list = []
            pos_idx_list = []
            current_slide = slide_name

        # Extract patches from batch (removes empty elements)
        stopper_idx = torch.where(data_idx < 0, 1, 0).nonzero()
        num_neg_idx = stopper_idx.shape[0]
        if num_neg_idx > 0:
            stop_id = stopper_idx[0]
            patches = patches[:stop_id]
            pos_idx = pos_idx[:stop_id]

        # Extract features
        if patches.shape[0] > 0:
            features = net(patches)
            b, n_feat = features.shape[:2]
            #TODO: is this needed?
            features = features.view(b, n_feat)

            feature_list.append(features)
            pos_idx_list.append(pos_idx)

        is_last_patch = data_idx[-1] == RandomPatchSampler.SLIDE_END_TOKEN#-2
        if is_last_patch:
            num_processed += 1
            print("Nr. slides processed: ", num_processed)

            features_np = torch.cat(feature_list, 0).cpu().numpy()
            pos_idx_np = torch.cat(pos_idx_list, 0).cpu().numpy()
            
            # Save as HDF5 file
            slide_grp = h5file.create_group(slide_name)
            slide_grp.create_dataset('img', data=features_np, compression="gzip", compression_opts=9)
            slide_grp.create_dataset('pos', data=pos_idx_np, compression="gzip", compression_opts=9)
            slide_grp.attrs['label'] = slide_label

h5file.close()
print("Stored features successfully!")
