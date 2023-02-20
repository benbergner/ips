import csv
import argparse
import multiprocessing as mp

from datamodel import SlideManager, Slide
from data.camelyon.cam_methods import get_otsu_threshold

parser = argparse.ArgumentParser(
    description="Compute Otsu thresholds from WSIs"
)
parser.add_argument(
    "--lvl",
    type=int,
    default=0,
    help="Choose the magnification level (0: highest) from which to compute thresholds"
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
    help="The directory where to store otsu thresholds."
)

args = parser.parse_args()

# Get arguments
lvl = args.lvl
n_worker = args.n_worker
data_dir = args.data_dir
otsu_fname = args.otsu_fname

# Create slide manager to access all slides
slide_man = SlideManager(data_dir=data_dir, otsu_fname=otsu_fname)

# Multiprocessing function
def get_slide_threshold(name):
    """
    Obtains a slide and computes the otsu threshold
    """
    slide_path = slide_man.slide_paths[name]
    slide = Slide(name, slide_path)

    threshold = get_otsu_threshold(slide, level=lvl, step_size=1000)

    del slide
    return name, lvl, threshold

# Calculating the Otsu threshold can take a long time
# depending on the magnification level, thus parallelize.
pool = mp.Pool(n_worker)
slide_thresholds = list(
    pool.map(get_slide_threshold, slide_man.slide_names)
)

# Write thresholds to file
f = open(out_dir, "w")

writer = csv.writer(f)
header = ['name', 'level', 'threshold']
writer.writerow(header)

for slide_name, level, threshold in slide_thresholds:
    writer.writerow([slide_name, level, threshold])

f.close()
print("Done saving thresholds!")