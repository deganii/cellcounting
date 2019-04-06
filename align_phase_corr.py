
import cv2
import numpy as np
import imutils
from hemocytometer import draw_hemocytometer
import imregpoc

# must downsample to a small dimension to keep runtime reasonable
resize = 2048

hemo_actual =  imutils.resize(cv2.imread("Z:/20190315/ImageJ_Simple_Stitch/15umC-2.tif"), width=resize)
hemo_actual = cv2.cvtColor(hemo_actual, cv2.COLOR_BGR2GRAY)
# hemo_actual = cv2.threshold(hemo_actual, np.mean(hemo_actual) + 30, 255, cv2.THRESH_BINARY)[1]
avg_color = np.mean(hemo_actual)

hemo_drawn = imutils.resize(draw_hemocytometer(bg=(avg_color,avg_color,avg_color)), width=resize)
hemo_drawn = cv2.cvtColor(hemo_drawn, cv2.COLOR_BGR2GRAY)

# hemo_actual = imutils.resize(cv2.imread("Z:/hemocytometer-cropped-cleaned.jpg"), width=resize)

height, width = hemo_actual.shape
if(height != resize or width != resize):
    top, bottom, left, right = 0,resize - height,0,resize - width
    hemo_actual = cv2.copyMakeBorder(hemo_actual,top, bottom, left, right, cv2.BORDER_REPLICATE)

# cv2.imshow('hemo_actual',hemo_actual)
# cv2.imshow('hemo_drawn',hemo_drawn)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

result = imregpoc.imregpoc(hemo_drawn,hemo_actual)
warp_matrix = result.getPerspective()
print("Warp Matrix: ", warp_matrix)

print("[x translation, y translation, rotation, scaling] = ", result.getParam())


sz = hemo_drawn.shape
# cv2.imshow('hemo',imutils.resize(result.stitching(), width=1024))
# ret = result.stitching()

im_aligned = cv2.warpPerspective(hemo_actual, warp_matrix,
        (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
alpha = 0.3
cv2.addWeighted(hemo_drawn, alpha, im_aligned, 1 - alpha, 0, im_aligned)

cv2.imshow('im_aligned',im_aligned)

cv2.waitKey(0)
cv2.destroyAllWindows()