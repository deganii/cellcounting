#
import cv2
import numpy as np
import math
import imutils
from hemocytometer import draw_hemocytometer
import imregpoc
import utils
from libtiff import TIFF

#
# # load an image as grayscale, resize to n x n for
# # spatial fourier transform, padding/centering if necessary
def load(im_path, resize=2048):
    im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    if im.shape[0] > im.shape[1]:
        im = imutils.resize(im, height=resize)
    else:
        im = imutils.resize(im, width=resize)

    if im.shape[0] != im.shape[1]:
        vert, horiz = (resize - im.shape[0]) / 2., (resize - im.shape[1]) / 2.
        im = cv2.copyMakeBorder(im, math.floor(vert), math.ceil(vert),
            math.floor(horiz), math.ceil(horiz), cv2.BORDER_REPLICATE)
        # im = cv2.copyMakeBorder(im, math.floor(vert), math.ceil(vert),
        #     math.floor(horiz), math.ceil(horiz), cv2.BORDER_CONSTANT, im, cv2.mean(im))
    # cv2.imshow('im',im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return im
#
# # aligns image and returns a tuple of:
# # (aligned_image, superimposed registration, transform params, warp_matrix)
# # Note: must downsample to keep runtime reasonable (i.e. to 2048x2048)
# def align_phase(ref_path, im_path, resize=2048, alpha = 0.5):
#     ref, im = load(ref_path), load(im_path)
#     result = imregpoc.imregpoc(ref, im)
#     warp_matrix = result.getPerspective()
#     transform = result.getParam()
#     print("Warp Matrix: ", warp_matrix)
#     print("[x translation, y translation, rotation, scaling] = ", transform)
#     aligned = cv2.warpPerspective(im, warp_matrix, ref.shape,
#       flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
#     reg = cv2.addWeighted(ref, alpha, aligned, 1 - alpha, 0)
#     return aligned, reg, transform, warp_matrix
#
# def align_hemocytometer_phase(im, resize=2048):
#     return align_phase('images/hemocytometer-reference.jpg', im, resize=resize)
#
# def align_device_phase(im, resize=2048):
#     return align_phase('images/pairing-device-reference.jpg', im, resize=resize)
#
#
#
#
#
# # aligned, reg, transform, warp_matrix = align_device_phase("test/trap3-cropped.jpg", resize=1024)
#
#
#
# aligned, reg, transform, warp_matrix = align_hemocytometer_phase("test/trap3-cropped.jpg", resize=1024)
#
# cv2.imshow('aligned',imutils.resize(aligned, width=1024))
# cv2.imshow('reg',imutils.resize(reg, width=1024))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#


#     result = imregpoc.imregpoc(hemo_drawn, hemo_actual)
#


# hemo_actual =  imutils.resize(cv2.imread("Z:/20190315/ImageJ_Simple_Stitch/15umC-2.tif"), width=resize)



# hemo_actual =  imutils.resize(cv2.imread("Z:/20190315/ImageJ_Simple_Stitch/BeforeA.tif"), width=resize)





# hemo_actual = utils.resize_square(hemo_actual, resize)
# hemo_actual = cv2.threshold(hemo_actual, np.mean(hemo_actual) + 30, 255, cv2.THRESH_BINARY)[1]
# avg_color = np.mean(hemo_actual)

# hemo_actual = utils.load_apply_threshold("test/10umC.jpg")
# hemo_ref = cv2.imread("images/hemocytometer-mask.jpg", 0)

hemo_actual = utils.load_apply_threshold("test/trap3.jpg")
hemo_ref = cv2.imread("images/pairing-device-mask.jpg", 0)


# hemo_actual = cv2.imread("test/trap3.jpg", 0)
# hemo_ref = cv2.imread("images/pairing-device-reference.jpg", 0)




hemo_actual = cv2.resize(hemo_actual, (0,0), fx=0.5, fy=0.5)
hemo_ref = cv2.resize(hemo_ref, (0,0), fx=0.5, fy=0.5)

hemo_actual_big = hemo_actual.copy()
hemo_ref_big = hemo_ref.copy()

# (win_h, win_w) = hemo_actual.shape
# win = cv2.createHanningWindow((win_w, win_h), cv2.CV_32F)
# hemo_actual = (-1.*hemo_actual) * win * -1.

# hemo_ref = utils.resize_square(hemo_ref, resize)
max_side = np.max([np.amax(hemo_actual.shape), np.amax(hemo_ref.shape)])
print("Max side: ", max_side)
hemo_actual = utils.pad_square(hemo_actual, max_side)
hemo_ref = utils.pad_square(hemo_ref, max_side)

resize = 2048
hemo_actual = cv2.resize(hemo_actual, (resize,resize), cv2.INTER_CUBIC)
hemo_ref = cv2.resize(hemo_ref, (resize,resize), cv2.INTER_CUBIC)


# hemo_drawn = imutils.resize(draw_hemocytometer(bg=(avg_color,avg_color,avg_color)), width=resize)
# hemo_drawn = cv2.cvtColor(hemo_drawn, cv2.COLOR_BGR2GRAY)

# hemo_actual = imutils.resize(cv2.imread("Z:/hemocytometer-cropped-cleaned.jpg"), width=resize)

# height, width = hemo_actual.shape
# if(height != resize or width != resize):
#     top, bottom, left, right = 0,resize - height,0,resize - width
#     hemo_actual = cv2.copyMakeBorder(hemo_actual,top, bottom, left, right, cv2.BORDER_REPLICATE)



# cv2.imshow('hemo_actual',imutils.resize(hemo_actual,1024))
# cv2.imshow('hemo_ref',imutils.resize(hemo_ref, 1024))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

result = imregpoc.imregpoc(hemo_ref,hemo_actual)
# utils.render_alignment(hemo_ref,
#     hemo_actual, result.getPerspective())

utils.render_alignment(hemo_ref_big,
    hemo_actual_big, result.getPerspective())

# warp_matrix = result.getPerspective()
# print("Warp Matrix: ", warp_matrix)
#
np.set_printoptions(precision=3, suppress=True)
print("[x translation, y translation, rotation, scaling] = ", np.array(result.getParam()))
#
#
# sz = hemo_ref.shape
# # cv2.imshow('hemo',imutils.resize(result.stitching(), width=1024))
# # ret = result.stitching()
#
# im_aligned = cv2.warpPerspective(hemo_actual, warp_matrix,
#         (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
#
#
# im_aligned = cv2.cvtColor(im_aligned, cv2.COLOR_GRAY2BGR)
# im_aligned[:,:,0] = 255
# im_aligned[:,:,2] = 255
#
# hemo_ref = cv2.cvtColor(hemo_ref, cv2.COLOR_GRAY2BGR)
#
#
# alpha = 0.3
# cv2.addWeighted(hemo_ref, alpha, im_aligned, 1 - alpha, 0, im_aligned)
#
# cv2.imshow('im_aligned',imutils.resize(im_aligned, 1024))
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()