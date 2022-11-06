

def _otsu_by_hist(hist, bin_centers) -> float:
    """Return threshold value based on Otsu's method using an images histogram.

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

def get_otsu_threshold(slide: Slide, level=0, step_size=1000) -> float:
    """Calculate the otsu threshold by reading in the slide in chunks.

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