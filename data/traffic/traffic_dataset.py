import os
from os import path
import sys
import hashlib
from functools import partial
from collections import namedtuple
import urllib.request
import zipfile
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

### Adapted from: https://github.com/sara-nl/attention-sampling-pytorch

def check_file(filepath, md5sum):
    """Check a file against an md5 hash value.
    Returns
    -------
        True if the file exists and has the given md5 sum False otherwise
    """

    try:
        md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(partial(f.read, 4096), b""):
                md5.update(chunk)
        return md5.hexdigest() == md5sum
    except FileNotFoundError:
        return False

def ensure_dataset_exists(directory, tries=1, progress_file=sys.stderr):
    """Ensure that the dataset is downloaded and is correct.
    Correctness is checked only against the annotations files.
    """

    set1_url = ("http://www.isy.liu.se/cvl/research/trafficSigns"
                "/swedishSignsSummer/Set1/Set1Part0.zip")
    set1_annotations_url = ("http://www.isy.liu.se/cvl/research/trafficSigns"
                            "/swedishSignsSummer/Set1/annotations.txt")
    set1_annotations_md5 = "9106a905a86209c95dc9b51d12f520d6"
    set2_url = ("http://www.isy.liu.se/cvl/research/trafficSigns"
                "/swedishSignsSummer/Set2/Set2Part0.zip")
    set2_annotations_url = ("http://www.isy.liu.se/cvl/research/trafficSigns"
                            "/swedishSignsSummer/Set2/annotations.txt")
    set2_annotations_md5 = "09debbc67f6cd89c1e2a2688ad1d03ca"

    integrity = (
        check_file(
            path.join(directory, "Set1", "annotations.txt"),
            set1_annotations_md5
        ) and check_file(
            path.join(directory, "Set2", "annotations.txt"),
            set2_annotations_md5
        )
    )

    if integrity:
        return

    if tries <= 0:
        raise RuntimeError(("Cannot download dataset or dataset download "
                            "is corrupted"))

    print("Downloading Set1", file=progress_file)
    download_file(set1_url, path.join(directory, "Set1.zip"),
                  progress_file=progress_file)
    print("Extracting...", file=progress_file)
    with zipfile.ZipFile(path.join(directory, "Set1.zip")) as archive:
        archive.extractall(path.join(directory, "Set1"))
    print("Getting annotation file", file=progress_file)
    download_file(
        set1_annotations_url,
        path.join(directory, "Set1", "annotations.txt"),
        progress_file=progress_file
    )
    print("Downloading Set2", file=progress_file)
    download_file(set2_url, path.join(directory, "Set2.zip"),
                  progress_file=progress_file)
    print("Extracting...", file=progress_file)
    with zipfile.ZipFile(path.join(directory, "Set2.zip")) as archive:
        archive.extractall(path.join(directory, "Set2"))
    print("Getting annotation file", file=progress_file)
    download_file(
        set2_annotations_url,
        path.join(directory, "Set2", "annotations.txt"),
        progress_file=progress_file
    )

    return ensure_dataset_exists(
        directory,
        tries=tries - 1,
        progress_file=progress_file
    )

def download_file(url, destination, progress_file=sys.stderr):
    """Download a file with progress."""
    
    response = urllib.request.urlopen(url)
    n_bytes = response.headers.get("Content-Length")
    if n_bytes == "":
        n_bytes = 0
    else:
        n_bytes = int(n_bytes)

    message = "\rReceived {} / {}"
    cnt = 0
    with open(destination, "wb") as dst:
        while True:
            print(message.format(cnt, n_bytes), file=progress_file,
                  end="", flush=True)
            data = response.read(65535)
            if len(data) == 0:
                break
            dst.write(data)
            cnt += len(data)
    print(file=progress_file)

class Sign(namedtuple("Sign", ["visibility", "bbox", "type", "name"])):
    """A sign object. Useful for making ground truth images as well as making
    the dataset."""

    @property
    def x_min(self):
        
        return self.bbox[2]

    @property
    def x_max(self):
        
        return self.bbox[0]

    @property
    def y_min(self):
        
        return self.bbox[3]

    @property
    def y_max(self):
        
        return self.bbox[1]

    @property
    def area(self):

        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    @property
    def center(self):

        return [
            (self.y_max - self.y_min) / 2 + self.y_min,
            (self.x_max - self.x_min) / 2 + self.x_min
        ]

    @property
    def visibility_index(self):

        visibilities = ["VISIBLE", "BLURRED", "SIDE_ROAD", "OCCLUDED"]
        return visibilities.index(self.visibility)

    def pixels(self, scale, size):

        return zip(*(
            (i, j)
            for i in range(round(self.y_min * scale), round(self.y_max * scale) + 1)
            for j in range(round(self.x_min * scale), round(self.x_max * scale) + 1)
            if i < round(size[0] * scale) and j < round(size[1] * scale)
        ))

    def __lt__(self, other):

        if not isinstance(other, Sign):
            raise ValueError("Signs can only be compared to signs")

        if self.visibility_index != other.visibility_index:
            return self.visibility_index < other.visibility_index

        return self.area > other.area


class STS:
    """The STS class reads the annotations and creates the corresponding
    Sign objects."""

    def __init__(self, directory, train=True, seed=0):

        cwd = os.getcwd().replace('dataset', '')
        directory = path.join(cwd, directory)
        ensure_dataset_exists(directory)

        self._directory = directory
        self._inner = "Set{}".format(1 + ((seed + 1 + int(train)) % 2))
        self._data = self._load_signs(self._directory, self._inner)

    def _load_files(self, directory, inner):

        files = set()
        with open(path.join(directory, inner, "annotations.txt")) as f:
            for l in f:
                files.add(l.split(":", 1)[0])

        return sorted(files)

    def _read_bbox(self, parts):

        def _float(x):

            try:
                return float(x)
            except ValueError:
                if len(x) > 0:
                    return _float(x[:-1])
                raise

        return [_float(x) for x in parts]

    def _load_signs(self, directory, inner):

        with open(path.join(directory, inner, "annotations.txt")) as f:
            lines = [l.strip() for l in f]
        keys, values = zip(*(l.split(":", 1) for l in lines))
        all_signs = []
        for v in values:
            signs = []
            for sign in v.split(";"):
                if sign == [""] or sign == "":
                    continue
                parts = [s.strip() for s in sign.split(",")]
                if parts[0] == "MISC_SIGNS":
                    continue
                signs.append(Sign(
                    visibility=parts[0],
                    bbox=self._read_bbox(parts[1:5]),
                    type=parts[5],
                    name=parts[6]
                ))
            all_signs.append(signs)
        images = [path.join(directory, inner, f) for f in keys]

        return list(zip(images, all_signs))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]

class TrafficSigns(Dataset):
    """ Loads images from the traffic signs dataset as
    a filtered version of the STS dataset.
    Arguments
    ---------
        directory: str, The directory that the dataset already is or is going
                   to be downloaded in
        train: bool, Select the training or testing sets
        seed: int, The prng seed for the dataset
    """

    LIMITS = ["50_SIGN", "70_SIGN", "80_SIGN"]
    CLASSES = ["EMPTY", *LIMITS]
    IMG_SIZE = (1200, 1600)

    def __init__(self, conf, train=True):

        self.patch_size = conf.patch_size
        self.patch_stride = conf.patch_stride
        self.tasks = conf.tasks
        
        self._data = self._filter(STS(conf.data_dir, train, conf.seed))
        
        transform_list = [
            transforms.Resize([*self.IMG_SIZE])
        ]

        if train:
            transform_list += [
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                transforms.RandomAffine(degrees=0, translate=(100 / self.IMG_SIZE[1], 100 / self.IMG_SIZE[0])),
            ]
        
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
           
        self.transform = transforms.Compose(transform_list)

    def _filter(self, data):

        filtered = []
        for image, signs in data:
            signs, acceptable = self._acceptable(signs)
            if acceptable:
                if not signs:
                    filtered.append((image, 0))
                else:
                    filtered.append((image, self.CLASSES.index(signs[0].name)))
        return filtered

    def _acceptable(self, signs):

        # Keep it as empty
        if not signs:
            return signs, True

        # Filter just the speed limits and sort them wrt visibility
        signs = sorted(s for s in signs if s.name in self.LIMITS)

        # No speed limit but many other signs
        if not signs:
            return None, False

        # Not visible sign so skip
        if signs[0].visibility != "VISIBLE":
            return None, False

        return signs, True

    def __len__(self):

        return len(self._data)

    def __getitem__(self, i):

        patch_size = self.patch_size
        patch_stride = self.patch_stride

        img, category = self._data[i]
        img = Image.open(img)
        img = self.transform(img)

        # Extract patches
        patches = img.unfold(
            1, patch_size[0], patch_stride[0]
        ).unfold(
            2, patch_size[1], patch_stride[1]
        ).permute(1, 2, 0, 3, 4)

        patches = patches.reshape(-1, *patches.shape[2:])

        data_dict = {'input': patches}
        for task in self.tasks.values():
            data_dict[task['name']] = category

        return data_dict