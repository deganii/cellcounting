import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt
from utils import load_apply_threshold
MIN_MATCH_COUNT = 10

# Note: works with opencv-contrib-python  3.2.0.7
img = cv2.imread('images/hemocytometer-mask.jpg')
# cv2.imwrite('images/hemocytometer-mask1024.jpg', imutils.resize(img, height=1024))
img = imutils.resize(img, height=1024)
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
# kp = sift.detect(gray,None)
# img=cv2.drawKeypoints(gray,kp,img)
# cv2.imwrite('images/hemocytometer-sift.jpg',img)

test = load_apply_threshold('test/10umC.jpg')
# mask and threshold

test = imutils.resize(test, height=1024)
# test = cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)
# sift2 = cv2.xfeatures2d.SIFT_create()
# kp2 = sift2.detect(test,None)


kp1, des1 = sift.detectAndCompute(img,None)
kp2, des2 = sift.detectAndCompute(test,None)


FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)



matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)


if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(test,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img,kp1,test,kp2,good,None,**draw_params)

cv2.imwrite('test/sift_match.jpg', img3)
# plt.imshow(img3, 'gray'),plt.show()
