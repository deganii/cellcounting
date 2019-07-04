import cv2
import imutils
import numpy as np

# Apply a consistent adaptive threshold across all images
def load_apply_threshold(im_path, resize = None):
    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # im = cv2.GaussianBlur(image, (9, 9), 0)
    im = cv2.bilateralFilter(im, 9, 75, 75)
    return cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                  cv2.THRESH_BINARY, 11, 2)

# pad to a square image of side "size" pixels
def pad_square(im, size):
    height, width = im.shape
    if height != size or width != size:
        delta_w = size - width
        delta_h = size - height
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        return cv2.copyMakeBorder(im, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=255)
    return im

# resize to a square, padding where necessary
def resize_square(im, size):
    height, width = im.shape
    if (height != size and width != size):
        if (height > width):
            im = imutils.resize(im, height=size)
        else:
            im = imutils.resize(im, width=size)
        height, width = im.shape

    delta_w = size - width
    delta_h = size - height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    # top, bottom, left, right = 0, size - height, 0, size - width
    # im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_REPLICATE)

    return cv2.copyMakeBorder(im,top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)


# show an image aligned to its reference
def render_alignment(ref, im, warp_matrix, save='test/fed.jpg'):
    print("Warp Matrix: ", warp_matrix)
    # print("[x translation, y translation, rotation, scaling] = ", result.getParam())

    sz = ref.shape
    im_aligned = cv2.warpPerspective(im, warp_matrix,
        (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)


    M = np.float32([[1.0035, 0, 10], [0, 1.0035, 3]])
    rows, cols = sz
    im_aligned = cv2.warpAffine(im_aligned, M, (cols, rows))

    im_aligned = cv2.cvtColor(im_aligned, cv2.COLOR_GRAY2BGR)
    im_aligned[:, :, 0] = 255
    im_aligned[:, :, 2] = 255

    ref = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)

    alpha = 0.5
    im_aligned = cv2.addWeighted(ref, alpha, im_aligned, 1 - alpha, 0, im_aligned)
    if save is not None:
        cv2.imwrite(save, im_aligned)
        cv2.imwrite('test/ref.jpg', ref)
        cv2.imwrite('test/im.jpg', im)

    cv2.imshow('im_aligned', imutils.resize(im_aligned, 1024))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)[1]

# image = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY)[1]
#
# image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY,11,2)
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

# cv2.imwrite('images/presentation.jpg', image)
