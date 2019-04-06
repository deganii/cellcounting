
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import itertools

# draws a hemocytometer at 10x magnification for registration purposes
def draw_hemocytometer(bg = (128,128,128)):
    n, side, thickness, offset, color, pad = 4, 383, 6, 3092, (255, 255, 255), 8
    size, l_size = 2*pad + offset + n * side, offset + n * side
    offsets = [(pad, pad), (pad, offset + pad), (offset+ pad, pad), (offset+ pad, offset+ pad)]

    image = np.full((size, size, 3), bg, dtype="uint8")

    # draw the enclosing rectangle
    cv2.rectangle(image,(pad,pad), (pad+l_size,pad+l_size), color, thickness)

    # draw the 4 4x4 corner grids
    for i, j in itertools.product(range(n), range(n)):
      for offset_x, offset_y in offsets:
        cv2.rectangle(image,
          (offset_x + i * side, offset_y + j * side),
          (offset_x + (i + 1) * side, offset_y + (j + 1) * side),
          color, thickness)
      if i == j:
        for offset_x, offset_y in offsets:
            cv2.line(image,
              (offset_x + i * side, pad),
              (offset_x + i * side, l_size),
              color, thickness)
            cv2.line(image,
              (pad, offset_x + i * side),
              (l_size, offset_x + i * side),
              color, thickness)

    # add vertical lines
    r_offset, r_side, n_lines = 9,77.25, 21
    for i in range(n_lines):
       cv2.line(image, (pad+int(side*4 + r_offset + r_side*i), pad),
                (pad+int(side*4 + r_offset + r_side*i), l_size + pad),
         color, thickness)
       if i % 4 == 0:
            cv2.line(image, (pad+int(side * 4 + 2*r_offset + r_side * i), pad),
                     (pad+int(side * 4 + 2*r_offset + r_side * i), l_size + pad),
                     color, thickness)
            cv2.line(image, (pad+int(side * 4 + r_side * i), pad),
                     (pad+int(side * 4 + r_side * i), l_size + pad),
                     color, thickness)

    # add horizontal lines
    for i in range(n_lines):
       cv2.line(image, (pad, pad+int(side*4 + r_offset + r_side*i)),
                (l_size + pad, pad+int(side*4 + r_offset + r_side*i)),
         color, thickness)
       if i % 4 == 0:
            cv2.line(image, (pad, pad+int(side * 4 + 2*r_offset + r_side * i)),
                     (l_size + pad, pad+int(side * 4 + 2*r_offset + r_side * i)),
                     color, thickness)
            cv2.line(image, (pad, pad+int(side * 4 + r_side * i)),
                     (l_size + pad, pad+int(side * 4 + r_side * i)),
                     color, thickness)

    # DEBUG:
    # reference = cv2.imread("Z:/hemocytometer-cropped-cleaned.jpg")[:size,:size,:]
    # alpha = 0.8
    # cv2.addWeighted(image, alpha, reference, 1 - alpha, 0, image)

    return image


# cv2.imshow('hemo',draw_hemocytometer())
# cv2.imshow('hemo',imutils.resize(draw_hemocytometer(), width=1024))
# cv2.imshow('hemo',imutils.translate(reference, 0, -3000))
# cv2.imshow('hemo',imutils.translate(reference, -3000, 0))
# cv2.imshow('hemo',imutils.translate(reference, -3000, -3000))
# cv2.waitKey(0)
# cv2.destroyAllWindows()


