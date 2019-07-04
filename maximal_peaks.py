# from https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array
import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
import scipy.ndimage
from libtiff import TIFF

import matplotlib.pyplot as plot

def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    n_size = 2
    neighborhood = generate_binary_structure(n_size,n_size)

    #apply the local maximum filter; all pixel of maximal value
    #in their neighborhood are set to 1
    max_filt= maximum_filter(image, footprint=neighborhood)
    local_max = max_filt==image
    #local_max is a mask that contains the peaks we are
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to
    #successfully subtract it form local_max, otherwise a line will
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks,
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks

green_path = 'images/raw/20190514/20190514 kb fusion 01at1.tif'
red_path = 'images/raw/20190514/20190514 kb fusion 01bt1.tif'
green = TIFF.open(green_path).read_image() / 65535.
red = TIFF.open(red_path).read_image() / 65535.

green_avg = green.mean(axis=2)
red_avg = red.mean(axis=2)

#applying the detection and plotting results
green_peaks = detect_peaks(green_avg)
red_peaks = detect_peaks(red_avg)
# plot.subplot(4,2,(2*i+1))
plot.imshow(green_avg)
# plot.subplot(4,2,(2*i+2) )
plot.imshow(green_peaks)
plot.show()

