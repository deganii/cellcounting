# Counts cells captured in an Incyto hemocytometer and delivers various statistics
# TODO: add viability for trypan stains
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt

# image = cv2.imread("Y:/idegani/20190404/20190401 b cell trapt1.tif")
# image = cv2.imread("Y:\idegani/20190403 Tip filter/20190402 tip filter post stiched.jpg")
# reference = cv2.imread("Z:/hemocytometer-cropped-cleaned.jpg")
# reference = imutils.translate(reference, -8, -8)
# overlay = reference.copy()
# overlay = np.zeros(reference.shape, dtype = "uint8")
#
# n, sidex, sidey, thickness, offset = 4, 383, 383, 6, 8
# offsets = [(0,0),(0, 3092), (3092, 0), (3092, 3092)]
#
# # draw the grids of interest for registration purposes
# # at 10x, the grid is approximately
# for i in range(n):
#   for j in range(n):
#     for offset_x, offset_y in offsets:
#       cv2.rectangle(overlay,
#         (offset_x+i*sidex, offset_y+j*sidey),
#         (offset_x+(i+1)*sidex, offset_y+(j+1)*sidey),
#         (0,0,255), thickness)

# alpha = 0.8
# cv2.addWeighted(overlay, alpha, reference, 1 - alpha, 0, reference)

#

# image = imutils.resize(image, width=1024)
# ratio = image.shape[0] / float(resized.shape[0])
# plt_image = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)
# imgplot = plt.imshow(plt_image)

# cv2.imshow('reference',imutils.resize(reference, width=1024))

# cv2.imshow('reference',reference)
# cv2.imshow('reference1',imutils.translate(reference, 0, -3000))
# cv2.imshow('reference2',imutils.translate(reference, -3000, 0))
# cv2.imshow('reference3',imutils.translate(reference, -3000, -3000))
# cv2.waitKey(0)
# # cv2.waitKey(0)
# cv2.destroyAllWindows()


# convert the resized image to grayscale, blur it slightly,
# and threshold it
image = cv2.imread("images/pairing-device-reference.jpg")
# image = cv2.imread("images/hemocytometer-reference.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = cv2.GaussianBlur(image, (9, 9), 0)


# edges = cv2.Canny(blurred,50,100)
# cv2.imshow('edges',edges)
# cv2.waitKey(0)

image = cv2.bilateralFilter(image, 9, 75, 75)
# image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)[1]

# image = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY)[1]
#
image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
# image = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
#

# image = cv2.bilateralFilter(image, 9, 75, 75)
# kernel = np.ones((5,5),np.uint8)
# image = cv2.erode(image,kernel,iterations = 1)
# image = cv2.dilate(image,kernel,iterations = 1)
# image = cv2.medianBlur(image, 9, 0)

# cv2.imshow('thresh',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imwrite('images/pairing-device-bilateral-mask.jpg', image)

cv2.imwrite('images/presentation.jpg', image)

# find contours in the thresholded image and initialize the
# shape detector
# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
#                         cv2.CHAIN_APPROX_SIMPLE)
#
# cnts = imutils.grab_contours(cnts)
#
# for cnt in cnts:
#
#     approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
#     print( len(approx))
#     if len(approx)==4:
#         print( "square")
#         cv2.drawContours(image,[cnt],0,(0,0,255),-1)
#     # elif len(approx) > 15:
#     #     print ("circle")
#     #     cv2.drawContours(image,[cnt],0,(0,255,255),-1)
#
# cv2.imshow('img',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

