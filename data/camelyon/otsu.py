# write a csv file for both training and test set which stores the otsu thesholds of each image.
import os
import sys
import csv
import argparse
import multiprocessing as mp

from datamodel import SlideManager, Slide
from camelyon_methods import get_otsu_threshold

def main(argv):

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
        "out_dir",
        help="The directory incl. filename where the thresholds should be saved"
    )

    args = parser.parse_args(argv)

    lvl = args.lvl
    n_worker = args.n_worker
    data_dir = args.data_dir
    out_dir = args.out_dir

#dset_path = '/dhc/dsets/CAMELYON/CAMELYON16/'
mgr = SlideManager(data_dir=data_dir)

#print("mgr.slide_names: ", mgr.slide_names)
#print("len slide_names: ", len(mgr.slide_names))
#sys.exit()

#fpath = os.path.join(dset_path, "otsu_thresholds.csv")
#out_path = "otsu_thresholds.csv"

def get_slide_threshold(name):
    slide_path = mgr.slide_paths[name]
    slide = Slide(name, slide_path)

    threshold = get_otsu_threshold(slide, level=0, step_size=1000, verbose=True)

    #data = [name, level, threshold]
    del slide
    return name, level, threshold

pool = mp.Pool(15)
slide_thresholds = list(
    pool.map(get_slide_threshold, mgr.slide_names)#total=len(mgr.slide_names)
)

f = open(fpath, "w")

writer = csv.writer(f)
header = ['name', 'level', 'threshold']
writer.writerow(header)

for slide_name, level, threshold in slide_thresholds:
    writer.writerow([slide_name, level, threshold])

f.close()
print("Done saving thresholds!")

if __name__ == "__main__":
    main(None)