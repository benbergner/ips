import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

from .datamodel import Slide, SlideManager
from .cam_methods import remove_alpha_channel


#TODO: is sampler only used for patch extraction? If yes, remove shuffling options and random in name
class RandomPatchSampler(Sampler):

    FILL_TOKEN = -1
    SLIDE_END_TOKEN = -2

    def __init__(self, bounds, num_samples=None, patch_shuffle=False, slide_shuffle=False, batch_size=1):
        self.bounds = bounds
        self.num_samples = num_samples
        self.patch_shuffle = patch_shuffle
        self.slide_shuffle = slide_shuffle
        self.batch_size = batch_size
        self.num_slides = self.bounds.shape[0]

        #print("data head: ", self.bounds.head(10))
        #print("num_slides: ", self.num_slides)
    
    def __len__(self):
        return self.num_samples

    def __iter__(self):

        slide_idx = list(range(self.num_slides))
        if self.slide_shuffle:
            random.shuffle(slide_idx)

        self.all_patch_idx = []
        for slide_id in slide_idx:

            row = self.bounds.iloc[slide_id]
            start_id = row['start_id']
            end_id = row['end_id']

            patch_idx = list(range(start_id, end_id+1))
            num_patches = len(patch_idx)
            if self.patch_shuffle:
                random.shuffle(patch_idx)

            # Add tokens to fill up batch
            remainder = (num_patches + 1) % self.batch_size # +1 extra patch
            num_to_add = self.batch_size - remainder# if remainder else 0
            patch_idx = patch_idx + [self.FILL_TOKEN] * num_to_add
            if self.patch_shuffle:
                random.shuffle(patch_idx)

            # Add token to identify end of slide
            patch_idx.append(self.SLIDE_END_TOKEN)
            self.all_patch_idx.extend(patch_idx)

        return iter(self.all_patch_idx)


class CAMELYON16Dataset(Dataset):

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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
                #thresh = self.mgr.otsu_thresholds[slide_name]
                slide = self.slide_man.get_slide(slide_name)# slide_paths[slide_name]
                """
                if slide_name in self.mgr.annotation_paths:
                    annotation_path = self.mgr.annotation_paths[slide_name]
                else:
                    annotation_path = None
                
                if slide_name in self.mgr.stages:
                    stage = self.mgr.stages[slide_name]
                else:
                    stage = None
                """
                #slide = Slide(slide_name, slide_path,
                #          otsu_thresholds=thresh, annotation_filename=annotation_path, stage=stage)
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
    
    #TODO: is this needed?
    """
    def get_bounds_of_slide(self, name):
        slide = self.mgr.get_slide(name)

        xy = self.coords_df[self.coords_df['name'] == name][['x', 'y']].values
        x = xy[:,0]
        y = xy[:,1]

        downsample = slide.level_downsamples[self.lvl]
        tile_size0 = int(self.tile_size * downsample + 0.5)

        bounds = ((x, y), (tile_size0, tile_size0))

        return bounds
    """