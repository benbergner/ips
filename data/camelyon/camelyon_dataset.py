import os
import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

from .datamodel import SlideManager
from .cam_methods import remove_alpha_channel

class PatchSampler(Sampler):

    FILL_TOKEN = -1
    SLIDE_END_TOKEN = -2

    def __init__(self, bounds, num_samples=None, batch_size=1):
        self.bounds = bounds
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_slides = self.bounds.shape[0]
    
    def __len__(self):
        return self.num_samples

    def __iter__(self):

        slide_idx = list(range(self.num_slides))
        self.all_patch_idx = []
        for slide_id in slide_idx:

            row = self.bounds.iloc[slide_id]
            start_id = row['start_id']
            end_id = row['end_id']

            patch_idx = list(range(start_id, end_id+1))
            num_patches = len(patch_idx)

            # Add tokens to fill up batch
            remainder = (num_patches + 1) % self.batch_size # +1 extra patch
            num_to_add = self.batch_size - remainder# if remainder else 0
            patch_idx = patch_idx + [self.FILL_TOKEN] * num_to_add

            # Add token to identify end of slide
            patch_idx.append(self.SLIDE_END_TOKEN)
            self.all_patch_idx.extend(patch_idx)

        return iter(self.all_patch_idx)


class CamelyonImages(Dataset):

    def __init__(self, data_dir, coords_df, lvl, tile_size):

        self.slide_man = SlideManager(data_dir=data_dir)
        self.coords_df = coords_df

        self.lvl = lvl
        self.tile_size = tile_size

        transform_list = [
            transforms.Lambda(lambda x: remove_alpha_channel(np.asarray(x))),
            transforms.ToPILImage(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        self.transform = transforms.Compose(transform_list)

        self.current_slide_name = None
        self.current_slide = None
    
    def __len__(self):
        return len(self.coords_df)

    def __getitem__(self, i):

        data = {}
        is_empty = i < 0
        if not is_empty:
            row = self.coords_df.iloc[i]
            slide_name, x, y, pos_id = row[['name', 'x', 'y', 'pos_id']]

            if slide_name != self.current_slide_name:
                slide = self.slide_man.get_slide(slide_name)
                
                self.current_slide_name = slide_name
                self.current_slide = slide
            else:
                slide = self.current_slide

            patch = slide.read_region((x, y), self.lvl, (self.tile_size, self.tile_size))
            data['patch'] = self.transform(patch)
            data['label'] = int(slide.has_tumor)
            data['pos_id'] = pos_id
            data['slide_name'] = slide_name
        else:
            data['patch'] = torch.zeros((3, 224, 224))
            data['label'] = -1
            data['pos_id'] = 9999 #TODO: why not None or not fill at all?
            data['slide_name'] = '' #TODO: same here, why not None or leave out?
        data['data_id'] = i
        return data


class CamelyonFeatures(Dataset):

    def open_hdf5(self):
        self.dataset = h5py.File(self.data_dir, 'r')

    def select_slides(self):
        h5_data = h5py.File(self.data_dir, 'r')
        self.slide_names = list(h5_data.keys())

        #self.slide_names = [slide_name for slide_name in self.slide_names]#if 'tumor' in slide_name
        self.data_len = len(self.slide_names)

        h5_data.close()

    def __init__(self, conf, train=True):

        self.tasks = conf.tasks

        # TODO: switch fname pattern also in feature extraction phase
        filename = 'feat_{}_nopretr_480ep.hdf5'.format('train' if train else 'test') #'camelyon_{}.hdf5'.format('train' if train else 'test')
        #filename = 'byol_cam16_0_thresh0.01_byol500_centercr_{}.hdf5'.format('train' if train else 'test')
        self.data_dir = os.path.join(conf.data_dir, filename)

        self.select_slides()
    
    def __len__(self):
        return self.data_len

    def __getitem__(self, i):
        
        if not hasattr(self, 'dataset'):
            self.open_hdf5()

        slide_name = self.slide_names[i]

        slide = self.dataset[slide_name]
        patches = slide['img'][:]
        label = slide.attrs['label']

        data_dict = {'input': patches}
        for task in self.tasks.values():
            data_dict[task['name']] = label

        return data_dict