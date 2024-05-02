import cv2
import numpy as np
import matplotlib.pyplot as plt


def equalize_hsv(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = cv2.equalizeHist(hsv[:, :, 2])
    hsv[:, :, 2] = v
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def equalize_bgr(img_bgr):
    b = cv2.equalizeHist(img_bgr[:, :, 0])
    img_bgr[:, :, 0] = b

    g = cv2.equalizeHist(img_bgr[:, :, 1])
    img_bgr[:, :, 1] = g

    r = cv2.equalizeHist(img_bgr[:, :, 2])
    img_bgr[:, :, 2] = r

    return img_bgr


chairs = cv2.imread("images/chairs_oscura.jpg")

## Equalizar histograma en HSV
chairs_eq_hsv = equalize_hsv(chairs)

hsv_orig = cv2.cvtColor(chairs, cv2.COLOR_BGR2HSV)
hsv_eq = cv2.cvtColor(chairs_eq_hsv, cv2.COLOR_BGR2HSV)

hist_orig = cv2.calcHist(hsv_orig, [2], None, [256], [0, 256])
hist_eq = cv2.calcHist(hsv_eq, [2], None, [256], [0, 256])

# plt.figure()
#
# plt.subplot(121)
# plt.title("Histograma antes de equalizarlo")
# plt.bar(range(256), np.squeeze(hist_orig))
#
# plt.subplot(122)
# plt.title("Histograma en equalizado")
# plt.bar(range(256), np.squeeze(hist_eq))
#
# plt.show()


## Equalizar histograma en BGR
chairs_eq_bgr = equalize_bgr(chairs.copy())
hist_orig_bgr = np.empty((3, 256), dtype=np.uint8)
hist_eq_bgr = np.empty((3, 256), dtype=np.uint8)

for c in range(3):
    hist_orig_bgr[c, :] = cv2.calcHist(chairs, [c], None, [256], [0, 256])[:, 0]
    hist_eq_bgr[c, :] = cv2.calcHist(chairs_eq_bgr, [c], None, [256], [0, 256])[:, 0]


# plt.figure()
#
# for i in range(3):
#     plt.subplot(2, 3, i + 1)
#     plt.bar(range(256), np.squeeze(hist_orig_bgr[i, :]))
#
#     plt.subplot(2, 3, i + 4)
#     plt.bar(range(256), np.squeeze(hist_eq_bgr[i, :]))
#
# plt.show()


cv2.namedWindow("Chairs", cv2.WINDOW_NORMAL)
cv2.namedWindow("Chairs Equalized HSV", cv2.WINDOW_NORMAL)
cv2.namedWindow("Chairs Equalized BGR", cv2.WINDOW_NORMAL)


cv2.imshow("Chairs", chairs)
cv2.imshow("Chairs Equalized HSV", chairs_eq_hsv)
cv2.imshow("Chairs Equalized BGR", chairs_eq_bgr)

while True:
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
