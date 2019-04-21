import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import itertools

# draws a hemocytometer at 10x magnification for registration purposes
def draw_pairing_device(bg = (128,128,128)):
    image = np.full((size, size, 3), bg, dtype="uint8")

    # draw filters


    # draw traps


    size = 2*pad + offset + n * side
    offsets = [(pad, pad), (pad, offset + pad), (offset+ pad, pad), (offset+ pad, offset+ pad)]




    # DEBUG:
    # reference = cv2.imread("Z:/hemocytometer-cropped-cleaned.jpg")[:size,:size,:]
    # alpha = 0.8
    # cv2.addWeighted(image, alpha, reference, 1 - alpha, 0, image)
    return
