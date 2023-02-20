import math
import numpy as np
import datetime

from skimage.draw import polygon as ski_polygon
from skimage.measure import label as ski_label

from data.camelyon.datamodel import Slide
from data.camelyon.cam_utils import ProgressBar

def remove_alpha_channel(image: np.ndarray) -> np.ndarray:
    """
    Remove the alpha channel of an image.

    Parameters
    ----------
    image : np.ndarray
        RGBA image as numpy array with W×H×C dimensions.

    Returns
    -------
    np.ndarray
        RGB image as numpy array
    """
    if len(image.shape) == 3 and image.shape[2] == 4:
        return image[::, ::, 0:3:]
    else:
        return image

def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB color image to a custom gray scale for HE-stained WSI

    Parameters
    ----------
    rgb : np.ndarray
        Color image.

    Returns
    -------
    np.ndarray
        Gray scale image as float64 array.
    """
    gray = 1.0 * rgb[::, ::, 0] + rgb[::, ::, 2] - (
        (1.0 * rgb[::, ::, 0] + rgb[::, ::, 1] + rgb[::, ::, 2])
        / 1.5)
    gray[gray < 0] = 0
    gray[gray > 255] = 255
    return gray

def create_otsu_mask_by_threshold(image: np.ndarray, threshold) -> np.ndarray:
    """
    Create a binary mask separating fore and background based on the otsu threshold.

    Parameters
    ----------
    image : np.ndarray
        Gray scale image as array W×H dimensions.

    threshold : float
        Upper Otsu threshold value.

    Returns
    -------
    np.ndarray
        The generated binary masks has value 1 in foreground areas and 0s everywhere
        else (background)
    """
    otsu_mask = image > threshold
    otsu_mask2 = image > threshold * 0.25

    otsu_mask2_labeled = ski_label(otsu_mask2)
    for i in range(1, otsu_mask2_labeled.max()):
        if otsu_mask[otsu_mask2_labeled == i].sum() == 0:
            otsu_mask2_labeled[otsu_mask2_labeled == i] = 0
    otsu_mask3 = otsu_mask2_labeled
    otsu_mask3[otsu_mask3 > 0] = 1

    return otsu_mask3.astype(np.uint8)

def _otsu_by_hist(hist, bin_centers) -> float:
    """
    Return threshold value based on Otsu's method using an images histogram.

    Based on skimage's threshold_otsu method without histogram generation.

    Parameters
    ----------
    hist : np.ndarray
        Histogram of a gray scale input image.

    bin_centers: np.ndarray
        Centers of the histogram's bins.

    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.

    References
    ----------
    Wikipedia, http://en.wikipedia.org/wiki/Otsu's_Method

    See Also
    --------
    skimage.filters.threshold_otsu
    """
    hist = hist.astype(float)

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of `weight1`/`mean1` should pair with zero values in
    # `weight2`/`mean2`, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold

def add_dict(left, right):
    """
    Merge two dictionaries by adding common items.

    Parameters
    ----------
    left: dict
        Left dictionary.

    right
        Right dictionary

    Returns
    -------
    dict
        Resulting dictionary
    """
    return {k: left.get(k, 0) + right.get(k, 0) for k in left.keys() | right.keys()}

def get_otsu_threshold(slide: Slide, level=0, step_size=1000) -> float:
    """
    Calculate the otsu threshold by reading in the slide in chunks.

    To avoid memory overflows the slide image will be loaded in by chunks of the size
    $slide width × `step_size`$. A histogram will be generated of these chunks that will
    be used to calculate the otsu threshold based on skimage's `threshold_otsu` function.

    Parameters
    ----------
    slide : Slide
        Whole slide image slide

    level : int
        Level/layer of the `slide` to be used. Use of level ≠ 0 is not advised, see notes.

    step_size : int
        Each chunk loaded will have the size $slide-width × `step_size`$ on the level 0
        slide. For higher levels the step will be downsampled accordingly (e.g.: with a
        `step_size` of 1000 and `level` of 1 and a downsample factor of 2 the actual size
        of each chunk is $level-1-slide width × 500$.

    Returns
    -------
    otsu_threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.
    """

    size = slide.level_dimensions[0]
    downsample = slide.level_downsamples[level]

    # dictionary with all unique values and counts of the whole slide
    slide_count_dict = {}
    for i, y in enumerate(range(0, size[1], step_size)):

        # check if next step exceeds the image height and adjust it if needed
        cur_step = step_size if size[1] - y > step_size else size[1] - y

        # read in the image and transform to gray scale
        start, cut_size = (0, y), (int(size[0] / downsample), int(cur_step / downsample))
        a_img_cut = np.asarray(slide.read_region(start, level, cut_size))
        a_img_cut = rgb2gray(a_img_cut)

        # get unique values and their count
        chunk_count_dict = dict(zip(*np.unique(a_img_cut, return_counts=True)))

        # add those values and count to the dictionary
        slide_count_dict = add_dict(slide_count_dict, chunk_count_dict)

    # transform dictionary back to a arrays and calculate otsu threshold
    unique_values, counts = tuple(np.asarray(x) for x in zip(*slide_count_dict.items()))
    threshold = _otsu_by_hist(counts, unique_values)

    return threshold

def create_tumor_mask(slide: Slide, level, bounds=None):
    """Create a tumor mask for a slide or slide section.

    If `bounds` is given the tumor mask of only the section of the slide will be
    calculated.


    Parameters
    ----------
    slide : Slide
        Tissue slide.

    level : int
        Slide layer.

    bounds : tuple, optional
        Boundaries of a section as: ((x, y), (width, height))
        Where x and y are coordinates of the top left corner of the slide section on
        layer 0 and width and height the dimensions of the section on the specific
        layer `level`.  (Default: None)


    Returns
    -------
    tumor_mask : np.ndarray
        Binary tumor mask of the specified section. Healthy tissue is represented by 0,
        cancerous by 1.
    """
    if bounds is None:
        start_pos = (0, 0)
        size = slide.level_dimensions[level]
    else:
        start_pos, size = bounds

    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    downsample = slide.level_downsamples[level]

    for i, annotation in enumerate(slide.annotations):
        c_values, r_values = list(zip(*annotation.polygon))
        r = np.array(r_values, dtype=np.float32)
        r -= start_pos[1]
        r /= downsample
        r = np.array(r + 0.5, dtype=np.int32)

        c = np.array(c_values, dtype=np.float32)
        c -= start_pos[0]
        c /= downsample
        c = np.array(c + 0.5, dtype=np.int32)

        rr, cc = ski_polygon(r, c, shape=mask.shape)
        mask[rr, cc] = 1

    return mask

def split_slide(slide: Slide, lvl, otsu_threshold,
                fg_perc_thresh, tile_size, overlap):
    """
    Create tiles from a slide.

    Iterator over the slide in `tile_size`×`tile_size` Tiles. For every tile an otsu mask
    is created and summed up. Only tiles with sums over the percental threshold
    `fg_perc_thresh` will be yield.

    Parameters
    ----------
    slide : Slide
        Input Slide.

    lvl : int
        Layer to produce tiles from.

    otsu_threshold : float
        Otsu threshold of the whole slide on layer `level`.

    fg_perc_thresh : float, optional
        Minimum percentage, 0 to 1, of pixels with tissue per tile. (Default 0.01; 1%)

    tile_size : int
        Pixel size of one side of a square tile the image will be split into.
        (Default: 256)

    overlap : int, optional
        Count of pixel overlapping between two tiles. (Default: 30)

    Yields
    -------
    image_tile : np.ndarray
        Array of shape (`tile_size`, `tile_size`).

    bounds : tuple
        Tile boundaries on layer 0: ((x, y), (width, height))
    """
    if tile_size <= overlap:
        raise ValueError("Overlap has to be smaller than the tile size.")
    if overlap < 0:
        raise ValueError("Overlap can not be negative.")
    if otsu_threshold < 0:
        raise ValueError("Otsu threshold can not be negative.")
    if not 0.0 <= fg_perc_thresh <= 1.0:
        raise ValueError("Foreground threshold has to be between 0 and 1")

    width0, height0 = slide.level_dimensions[0]
    downsample = slide.level_downsamples[lvl]

    # Tile size on level 0
    tile_size0 = int(tile_size * downsample + 0.5)
    overlap0 = int(overlap * downsample + 0.5)

    # Minimum number of foreground pixels to be considered as foreground
    min_fg_count = tile_size ** 2 * fg_perc_thresh

    # We take patches into account if they belong to the foreground or to tumor tissue.
    # Once `num_pos_tiles_threshold` tumor patches have been found, do not verify whether patches belong
    # to tumor tissue anymore to save time.
    num_pos_tiles = 0
    num_pos_tiles_threshold = 100
    
    skip_pos_mask_calc = False
    # Loop through WSI rows and columns
    for y in range(0, height0, tile_size0 - overlap0):

        # Compute number of tumor pixels in row
        if skip_pos_mask_calc:
            n_tumor_pixels_row = 0
        else:
            if slide.has_tumor:
                mask_row = create_tumor_mask(slide, lvl, ((0, y), (width0, tile_size)))
                n_tumor_pixels_row = np.sum(mask_row)
            else:
                n_tumor_pixels_row = 0
            
        for x in range(0, width0, tile_size0 - overlap0):

            # Only proceed if tumor exists in row
            if n_tumor_pixels_row > 0:
                if lvl != 0:
                    mask_this = create_tumor_mask(slide, lvl, ((x, y), (tile_size, tile_size)))
                    pos_count = np.sum(mask_this)
                if lvl == 0:
                    pos_count = np.sum(mask_row[:,x:(x+tile_size)])

                if pos_count > 0:
                    num_pos_tiles +=1
                    if num_pos_tiles > num_pos_tiles_threshold:
                        skip_pos_mask_calc = True

            else:
                pos_count = 0

            tile = np.asarray(slide.read_region((x, y), lvl, (tile_size, tile_size)))
            otsu_mask = create_otsu_mask_by_threshold(rgb2gray(tile), otsu_threshold)

            fg_count = np.sum(otsu_mask)
            if fg_count >= min_fg_count or pos_count > 0:
                yield remove_alpha_channel(tile), ((x, y), (tile_size0, tile_size0))