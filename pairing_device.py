import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy.signal
import scipy.misc
from scipy import ndimage
import imageio
import math

def pad_fft(im, fft_size=2048):
    vert, horiz = (fft_size - im.shape[0]) / 2., (fft_size - im.shape[1]) / 2.
    # im = cv2.copyMakeBorder(im, math.floor(vert), math.ceil(vert),
    #     math.floor(horiz), math.ceil(horiz), cv2.BORDER_REPLICATE)
    return cv2.copyMakeBorder(im, math.floor(vert), math.ceil(vert),
        math.floor(horiz), math.ceil(horiz), cv2.BORDER_CONSTANT, im, 255.)


def phase_corr():
    single = imageio.imread("images/single-device-mask.jpg", as_gray=True)
    grid = imageio.imread("images/raw/01 BF-before.mask.tif")
    re = scipy.signal.fftconvolve(single, grid)

    cv2.imshow('re', np.log(re / np.max(re)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

    single = pad_fft(single) / 255.
    grid = pad_fft(grid) / 255.

    height, width = grid.shape
    hanw = cv2.createHanningWindow(grid.shape, cv2.CV_64F)

    # Windowing and FFT
    G_a = np.fft.fft2(single * hanw)
    G_b = np.fft.fft2(grid * hanw)
    conj_b = np.ma.conjugate(G_b)
    R = G_a * conj_b
    R /= np.absolute(R)
    r = np.fft.fftshift(np.fft.ifft2(R).real)
    r = 255. * r / np.max(r)

    cv2.imshow('r', np.log(r))
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def create_single_device():
    image = cv2.imread("images/raw/01 BF-before.tif")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, cols = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -0.35, 1)
    image = cv2.warpAffine(image, M, (cols, rows))

    image = cv2.bilateralFilter(image, 9, 75, 75)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                  cv2.THRESH_BINARY, 11, 2)
    cv2.imshow('img',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('images/raw/01 BF-before.mask.tif', image)


# draws a hemocytometer at 10x magnification for registration purposes
def draw_pairing_device(size, bg = (128,128,128)):
    image = np.full((size, size, 3), bg, dtype="uint8")
    # draw filters
    # draw traps
    # size = 2*pad + offset + n * side
    # offsets = [(pad, pad), (pad, offset + pad), (offset+ pad, pad), (offset+ pad, offset+ pad)]
    # DEBUG:
    # reference = cv2.imread("Z:/hemocytometer-cropped-cleaned.jpg")[:size,:size,:]
    # alpha = 0.8
    # cv2.addWeighted(image, alpha, reference, 1 - alpha, 0, image)

# create_single_device()

phase_corr()