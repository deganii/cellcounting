# align two images via Enhanced Correlation Coefficient (ECC) Maximization
# Credit :
# https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
import cv2
import numpy as np
import imutils
from hemocytometer import draw_hemocytometer

# Read the images to be aligned
# im1 = cv2.imread("images/image1.jpg");
# im2 = cv2.imread("images/image2.jpg");

# WARP MODES:
# MOTION_AFFINE
# MOTION_EUCLIDEAN
# MOTION_HOMOGRAPHY
# MOTION_TRANSLATION


def align_ecc(ref, im, warp_mode=cv2.MOTION_HOMOGRAPHY):

    # Convert images to grayscale
    # im1_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    # im2_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im1_gray = ref
    im2_gray = im

    # Find size of image1
    sz = ref.shape

    # Define the motion model

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000;

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # Use warpPerspective for Homography
        im_aligned = cv2.warpPerspective(im, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        # Use warpAffine for Translation, Euclidean and Affine
        im_aligned = cv2.warpAffine(im, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

    # overlay them
    alpha = 0.5
    cv2.addWeighted(im_aligned, alpha, ref, 1 - alpha, 0, im_aligned)

    # return aligned image
    return im_aligned, warp_matrix


# hemo_drawn = draw_hemocytometer()
# hemo_actual = cv2.imread("Z:/hemocytometer-cropped-cleaned.jpg")

# must downsample to a small dimension to keep runtime reasonable
resize = 256

hemo_drawn = imutils.resize(draw_hemocytometer(bg=(0,0,0)), width=resize)
hemo_drawn = cv2.cvtColor(hemo_drawn, cv2.COLOR_BGR2GRAY)

# hemo_actual = imutils.resize(cv2.imread("Z:/hemocytometer-cropped-cleaned.jpg"), width=resize)

hemo_actual = cv2.imread("Z:/20190315/ImageJ_Simple_Stitch/15umC-2.tif")
hemo_actual = cv2.cvtColor(hemo_actual, cv2.COLOR_BGR2GRAY)
hemo_actual = cv2.threshold(hemo_actual, np.mean(hemo_actual) + 30, 255, cv2.THRESH_BINARY)[1]
hemo_actual = imutils.resize(hemo_actual, width=resize)

cv2.imshow('hemo_actual',imutils.resize(hemo_actual, width=256))
cv2.imshow('hemo_drawn',imutils.resize(hemo_drawn, width=256))
cv2.waitKey(0)
cv2.destroyAllWindows()

aligned, warp = align_ecc(hemo_drawn, hemo_actual,cv2.MOTION_EUCLIDEAN)

print("Warp Matrix: ", warp)
cv2.imshow('hemo',imutils.resize(aligned, width=1024))
cv2.waitKey(0)
cv2.destroyAllWindows()
