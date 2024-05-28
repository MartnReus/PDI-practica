import cv2
import pytesseract
from matplotlib import pyplot as plt

max_lowThreshold = 100
window_name = 'Edge Map'
title_trackbar = 'Min Threshold:'
ratio = 3
kernel_size = 3


# Load image from webcam
image = cv2.imread("images/texto1.jpg",cv2.IMREAD_COLOR)
image = cv2.imread("images/textos/chart.JPG")
# image = image[0:2400,800:2800,:]

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# gray = cv2.GaussianBlur(gray, (5, 5), 0)
# detected_edges = cv2.Canny(gray, 50, 150, apertureSize=3)
# mask = detected_edges != 0

# canny = gray * mask


def CannyThreshold(val):
    low_threshold = val
    img_blur = cv2.blur(gray, (3,3))
    detected_edges = cv2.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    mask = detected_edges != 0
    dst = image  * (mask[:,:,None].astype(image.dtype))

    # tessdata_dir_config = r'--tessdata-dir "/usr/share/tessdata/"'
    # text = pytesseract.image_to_string(dst,config=tessdata_dir_config,lang='spa')
    # print("Recognized Text:", text)

    cv2.imshow(window_name, dst)
    return dst


init_th = 82
# lines = cv2.HoughLines()
cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
cv2.createTrackbar(title_trackbar, window_name , init_th, max_lowThreshold, CannyThreshold)
dst = CannyThreshold(init_th)
cv2.waitKey()

