#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pretraining.data.byol_transform import MultiViewDataInjector, get_transform
from datasets.camelyon.datamodel import Slide, SlideManager
from datasets.camelyon.cam_methods import remove_alpha_channel

class CAMELYONLoader():
    def __init__(self, config):
        self.num_replicas = config['world_size']
        self.rank = config['rank']
        self.distributed = config['distributed']
        self.resize_size = config['data']['resize_size']
        self.data_workers = config['data']['data_workers']
        self.dual_views = config['data']['dual_views']
        self.config = config

    def get_loader(self, stage, batch_size):
        dataset = self.get_dataset(stage)
        if self.distributed and stage in ('train', 'ft'):
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=self.num_replicas, rank=self.rank, shuffle=True)
        else:
            self.train_sampler = None

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=(self.train_sampler is None and stage not in ('val', 'test')),
            num_workers=self.data_workers,
            pin_memory=True,
            sampler=self.train_sampler,
            drop_last=True
        )
        return data_loader

    def get_dataset(self, stage):
        transform1 = get_transform(stage, resize_size=self.resize_size)
        if self.dual_views:
            transform2 = get_transform(stage, resize_size=self.resize_size, gb_prob=0.1, solarize_prob=0.2)
            transform = MultiViewDataInjector([transform1, transform2])
        else:
            transform = transform1
        dataset = CAMELYON_Dataset(config=self.config, transform=transform)
        return dataset

    def set_epoch(self, epoch):
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

class CAMELYON_Dataset(Dataset):

    def __init__(self, config, transform):
        self.dset_path = config['data']['dset_path']
        self.coords_path = config['data']['coords_path']
        self.level = config['data']['level']
        self.tile_size = config['data']['tile_size']

        self.slide_man = SlideManager(data_dir=self.dset_path)
        self.coords_df = pd.read_pickle(self.coords_path)
        print("len dataset: ", len(self.coords_df))
    
        self.transform = transform

    def __len__(self):
        return len(self.coords_df)

    def __getitem__(self, i):

        patch_info = self.coords_df.iloc[i]
        name, x, y = patch_info[['name', 'x', 'y']]
        
        slide_path = self.slide_man.slide_paths[name]
        slide = Slide(name, slide_path)      

        patch = slide.read_region((x, y), self.level, (self.tile_size, self.tile_size))
        patch = remove_alpha_channel(np.asarray(patch))

        patch = self.transform(patch)

        return patch, 0