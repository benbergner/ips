#!/usr/bin/env python
import os
import argparse
from tqdm import tqdm
import pandas as pd
import pickle
import multiprocessing as mp

from datamodel import SlideManager
from cam_methods import split_slide

parser = argparse.ArgumentParser(
    description="Compute foreground coordinates for each slide"
)

parser.add_argument('--train', dest='is_train', action='store_true')
parser.add_argument('--test', dest='is_train', action='store_false')
parser.set_defaults(is_train=True)

parser.add_argument(
    "--lvl",
    type=int,
    default=0,
    help="Choose the magnification level (0: highest) for foreground computation"
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
    help="Choose the tile size."
)
parser.add_argument(
    "--fg_perc_thresh",
    type=float,
    default=0.01,
    help="Minimum percentage of foreground pixels so that tile is considered foreground."
)
parser.add_argument(
    "--overlap",
    type=int,
    default=0,
    help="Overlap between tiles."
)
parser.add_argument(
    "--n_worker",
    type=int,
    default=16,
    help="Number of processes to spawn to parallelize computations"
)
parser.add_argument(
    "data_dir",
    help="The directory where the CAMELYON16 dataset is located"
)
parser.add_argument(
    "otsu_fname",
    help="The name of the file that holds Otsu thresholds."
)
parser.add_argument(
    "out_dir",
    help="Directory where foreground coordinates shall be stored. " + \
    "Filenames will be coords_{train/test}.pkl and bounds_{train/test}.pkl"
)

args = parser.parse_args()

# Get arguments
train = args.is_train
lvl = args.lvl
otsu_lvl = args.otsu_lvl
tile_size = args.tile_size
fg_perc_thresh = args.fg_perc_thresh
overlap = args.overlap
n_worker = args.n_worker
data_dir = args.data_dir
otsu_fname = args.otsu_fname
out_dir = args.out_dir

subset = 'train' if train else 'test'
bounds_path = os.path.join(out_dir, 'bounds_' + subset + '.pkl')
coords_path = os.path.join(out_dir, 'coords_' + subset + '.pkl')

def get_foreground_coords(name):
    slide = slide_man.get_slide(name)
    otsu_threshold = slide.get_otsu_threshold(otsu_lvl)
    tile_iter = split_slide(slide, lvl, otsu_threshold, fg_perc_thresh, tile_size, overlap)
    
    x_vals, y_vals = [], []
    for _, bounds in tile_iter:
        x, y = bounds[0]
        x_vals.append(x)
        y_vals.append(y)
    names = [name] * len(x_vals)
    print("Finished slide: ", name)

    return x_vals, y_vals, names

slide_man = SlideManager(data_dir=data_dir, otsu_fname=otsu_fname)
slide_names = slide_man.get_slide_names_subset(train=train)

# Computing of foreground coordinates can take a long time, thus parallelize
pool = mp.Pool(n_worker)
fg_coords = list(tqdm(
    pool.imap(get_foreground_coords, slide_names), total=len(slide_names)
))

# Create lists to be populated
start_idx, end_idx = [], []
all_idx, pos_idx = [], []
x_vals, y_vals = [], []
names = []

start = 0
for slide_id, slide_coords in enumerate(fg_coords):
    for patch_id, (x, y, name) in enumerate(zip(*slide_coords)):
        x_vals.append(x)
        y_vals.append(y)

        all_idx.append(start + patch_id)
        pos_idx.append(patch_id)
        
        names.append(name)
    
    print("Finished slide #", slide_id)

    end = start + patch_id
    
    start_idx.append(start)
    end_idx.append(end)

    start = end + 1

# Create dataframes
lvls = [lvl] * len(start_idx)
bounds_df = pd.DataFrame(
    {
        'level': lvls,
        'names': slide_names,
        'start_id': start_idx,
        'end_id': end_idx
    }
)
coords_df = pd.DataFrame(
    {
        'id': all_idx,
        'pos_id': pos_idx,
        'name': names,
        'x': x_vals,
        'y': y_vals
    }
)

# Save dataframes
bounds_file = open(bounds_path, 'wb')
pickle.dump(bounds_df, bounds_file)
bounds_file.close()

coords_file = open(coords_path, 'wb')
pickle.dump(coords_df, coords_file)
coords_file.close()

print("Done storing foreground coordinates.")