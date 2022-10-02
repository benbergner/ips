import os
import json
import numpy as np
import torch

class MEGAMNIST(torch.utils.data.Dataset):
    """ Loads the Megapixel MNIST dataset """

    def __init__(self, dataset_dir, patch_size, patch_stride, train=True):
        with open(os.path.join(dataset_dir, "parameters.json")) as f:
            self.parameters = json.load(f)

        self.patch_size = patch_size
        self.patch_stride = patch_stride

        filename = "train.npy" if train else "test.npy"
        W = self.parameters["width"]
        H = self.parameters["height"]

        self._img_shape = (H, W, 1)
        self._data = np.load(os.path.join(dataset_dir, filename), allow_pickle=True)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError()

        patch_size = self.patch_size
        patch_stride = self.patch_stride

        # Placeholders
        img = np.zeros(self._img_shape, dtype=np.float32).ravel()

        # Fill the sparse representations
        data = self._data[i]
        img[data[1][0]] = data[1][1]

        # Reshape to final shape        
        img = img.reshape(self._img_shape)
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)

        # Extract patches
        patches = img.unfold(
            1, patch_size[0], patch_stride[0]
        ).unfold(
            2, patch_size[1], patch_stride[0]
        ).permute(1, 2, 0, 3, 4)
        
        #_, _, c, h, w = patches.shape
        patches = patches.reshape(-1, *patches.shape[2:])

        label = np.argmax(data[2])
        #if len(data) > 3:
        max_label = data[3]
        top_label = data[4]
        multi_label = data[5]

            return patches, label, max_label, top_label, multi_label
        else:
            return patches, label