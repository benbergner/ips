# Adapted from https://github.com/idiap/attention-sampling

import os
import argparse
import json

import numpy as np
from keras.datasets import mnist

class MegapixelMNIST:
    """
    Class to create an artificial megapixel mnist dataset
    """

    class Sample(object):
        def __init__(self, dataset, idxs, positions, noise_positions,
                     noise_patterns):
            self._dataset = dataset
            self._idxs = idxs
            self._positions = positions
            self._noise_positions = noise_positions
            self._noise_patterns = noise_patterns

            self._img = None

        @property
        def noise_positions_and_patterns(self):
            return zip(self._noise_positions, self._noise_patterns)

        def _get_slice(self, pos, s=28, scale=1, offset=(0, 0)):
            pos = (int(pos[0]*scale-offset[0]), int(pos[1]*scale-offset[1]))
            s = int(s)
            return (
                slice(max(0, pos[0]), max(0, pos[0]+s)),
                slice(max(0, pos[1]), max(0, pos[1]+s)),
                0
            )

        def create_img(self):
            if self._img is None:
                size = self._dataset._H, self._dataset._W
                img = np.zeros(size + (1,), dtype=np.uint8)
                
                if self._dataset._should_add_noise:
                    for p, i in self.noise_positions_and_patterns:
                        img[self._get_slice(p)] = \
                            255*self._dataset._noise[i]
                
                for p, i in zip(self._positions, self._idxs):
                    img[self._get_slice(p)] = \
                        255*self._dataset._images[i]
                
                self._img = img
            return self._img

    def __init__(self, N=5000, W=1500, H=1500, train=True,
                 noise=True, n_noise=50, seed=0):
        # Load the images
        x, y = mnist.load_data()[0 if train else 1]
        x = x.astype(np.float32) / 255.

        self._W, self._H = W, H
        self._images = x

        # Generate the dataset
        try:
            random_state = np.random.get_state()
            np.random.seed(seed + int(train))
            self._nums, self._targets, self._digits, self._max_targets = self._get_numbers(N, y)
            self._pos = self._get_positions(N, W, H)
            self._top_targets = self._get_top_targets()

            self._noise, self._noise_positions, self._noise_patterns = \
                self._create_noise(N, W, H, n_noise)
        finally:
            np.random.set_state(random_state)

        # Boolean whether to add noise
        self._should_add_noise = noise

    def _create_noise(self, N, W, H, n_noise):
        """
        Create some random scribble noise of straight lines
        """
        angles = np.tan(np.random.rand(n_noise)*np.pi/2.5)
        A = np.zeros((n_noise, 28, 28))
        for i in range(n_noise):
            m = min(27.49, 27.49/angles[i])
            x = np.linspace(0, m, 56)
            y = angles[i]*x
            A[i, np.round(x).astype(int), np.round(y).astype(int)] = 1.
        B = np.array(A)
        np.random.shuffle(B)
        flip_x = np.random.rand(n_noise) < 0.33
        flip_y = np.random.rand(n_noise) < 0.33
        B[flip_x] = np.flip(B[flip_x], 2)
        B[flip_y] = np.flip(B[flip_y], 2)
        noise = ((A + B) > 0).astype(float)
        noise *= np.random.rand(n_noise, 28, 28)*0.2 + 0.8
        noise = noise.astype(np.float32)

        # Randomly assign noise to all images
        positions = (np.random.rand(N, n_noise, 2)*[H-56, W-56] + 28).astype(int)
        patterns = (np.random.rand(N, n_noise)*n_noise).astype(int)

        return noise, positions, patterns

    def _get_numbers(self, N, y):
        """
        Method to get numbers from the dataset

        Parameters:
        N (int): Number of samples (megapixel images) to create
        y (numpy array): Labels of standard mnist images

        Returns:
        numpy array: Array of indexes of the selected samples
        numpy array: Array of targets for the selected samples (task majority)
        numpy array: Array of digits for the selected samples (task multilabel)
        numpy array: Array of maximum digits for the selected samples (task max)
        """
        # Initialize empty lists for the output arrays
        nums = []
        targets = []
        max_targets = []
        all_digits = []
        # Get all indexes of the dataset
        all_idxs = np.arange(len(y))
        # Loop over the required number of samples
        for _ in range(N):
            # Get a random digit
            target = int(np.random.rand()*10)
            # Get three indexes where the target digit is present
            positive_idxs = np.random.choice(all_idxs[y == target], 3)
            # Get two negative indexes where the target digit is not present
            neg_idxs = np.random.choice(all_idxs[y != target], 2)

            # Concatenate the positive and negative indexes
            pos_neg_idxs = np.concatenate([positive_idxs, neg_idxs])
            # Get the digits from the concatenated indexes
            digits = y[pos_neg_idxs]
            # Get the maximum digit from the digits array
            max_target = np.max(digits)

            # Append the outputs to their respective lists
            nums.append(pos_neg_idxs)
            targets.append(target)
            all_digits.append(digits)
            max_targets.append(max_target)

        # Convert the lists to numpy arrays and return
        return np.array(nums), np.array(targets), np.array(all_digits), np.array(max_targets)

    def _get_positions(self, N, W, H):
        """
        Generates random positions of 5 digits in an image
        with size (H, W)

        Arguments:
            N: number of images to generate positions for
            W: width of the image
            H: height of the image

        Returns:
            np.array with shape (N, 5, 2) containing positions of 5 digits
        """
        def overlap(positions, pos):
            """
            Check if the new position 'pos' overlaps with
            any of the existing positions in 'positions'
            """
            if len(positions) == 0:
                return False
            distances = np.abs(
                np.asarray(positions) - np.asarray(pos)[np.newaxis]
            )
            axis_overlap = distances < 28
            return np.logical_and(axis_overlap[:, 0], axis_overlap[:, 1]).any()

        positions = []
        for _ in range(N):
            position = []
            for _ in range(5):
                while True:
                    pos = np.round(np.random.rand(2)*[H-28, W-28]).astype(int)
                    if not overlap(position, pos):
                        break
                position.append(pos)
            positions.append(position)

        return np.array(positions)
    
    def _get_top_targets(self):
        """
        Get digit that is topmost in each image
        """
        # `pos` is a numpy array with shape (n_img, digits, height and width)
        pos_height = self._pos[:,:,0] 
        
        # Get the index of the digit with the minimum height, i.e. top-most
        top_pos_idx = np.argmin(pos_height, axis=-1)
        
        # Get the digit with the minimum height for each image
        N = self._digits.shape[0]
        top_targets = self._digits[np.arange(N), top_pos_idx]

        return top_targets

    def __len__(self):
        return len(self._nums)

    def __getitem__(self, i):
        if len(self) <= i:
            raise IndexError()
        # Create a new sample
        sample = self.Sample(
            self,
            self._nums[i],
            self._pos[i],
            self._noise_positions[i],
            self._noise_patterns[i]
        )
        x = sample.create_img().astype(np.float32) / 255
        # Obtain labels for all tasks
        y = self._targets[i]
        y_max = self._max_targets[i]
        y_top = self._top_targets[i]
        y_multi = np.eye(10)[self._digits[i]].sum(0).clip(0,1)

        return x, y, y_max, y_top, y_multi


def sparsify(dataset):
    """
    Store non-zero values and their indixes only to save memory
    """
    def to_sparse(x):
        x = x.ravel()
        indices = np.where(x != 0)
        values = x[indices]
        return (indices, values)

    print("Sparsifying dataset")
    data = []
    for i, (x, y_maj, y_max, y_top, y_multi) in enumerate(dataset):
        print(
            "\u001b[1000DProcessing {:5d} /  {:5d}".format(i+1, len(dataset)),
            end="",
            flush=True
        )

        data.append({
            'input': to_sparse(x),
            'majority': y_maj,
            'max': y_max,
            'top': y_top,
            'multi': y_multi
        })
    print()
    return data

def main(argv):
    parser = argparse.ArgumentParser(
        description="Create the Megapixel MNIST dataset"
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=5000,
        help="How many images to create for training set"
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=1000,
        help="How many images to create for test set"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1500,
        help="Set the width for the image"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1500,
        help="Set the height for the image"
    )
    parser.add_argument(
        "--no_noise",
        action="store_false",
        dest="noise",
        help="Do not use noise in the dataset"
    )
    parser.add_argument(
        "--n_noise",
        type=int,
        default=50,
        help="Set the number of noise patterns per image"
    )
    parser.add_argument(
        "--dataset_seed",
        type=int,
        default=0,
        help="Choose the random seed for the dataset"
    )
    parser.add_argument(
        "output_directory",
        help="The directory to save the dataset into"
    )

    args = parser.parse_args(argv)

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    with open(os.path.join(args.output_directory, "parameters.json"), "w") as f:
        json.dump(
            {
                "n_train": args.n_train,
                "n_test": args.n_test,
                "width": args.width,
                "height": args.height,
                "noise": args.noise,
                "n_noise": args.n_noise,
                "seed": args.dataset_seed
            },
            f,
            indent=4
        )
    
    # Write the training set
    training = MegapixelMNIST(
        N=args.n_train,
        train=True,
        W=args.width,
        H=args.height,
        noise=args.noise,
        n_noise=args.n_noise,
        seed=args.dataset_seed
    )
    data = sparsify(training)
    np.save(os.path.join(args.output_directory, "train.npy"), data)

    # Write the test set
    test = MegapixelMNIST(
        N=args.n_test,
        train=False,
        W=args.width,
        H=args.height,
        noise=args.noise,
        n_noise=args.n_noise,
        seed=args.dataset_seed
    )
    data = sparsify(test)
    np.save(os.path.join(args.output_directory, "test.npy"), data)

# Usage example: python make_mnist.py --width 1500 --height 1500 dsets/megapixel_mnist_1500
if __name__ == "__main__":
    main(None)
