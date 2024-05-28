import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('images/textos/desk.JPG')
# image = image[0:2400,800:2800,:]

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

document_contour = np.empty([2,2])
# Loop over the contours to find the document
for contour in contours:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    if len(approx) == 4:
        document_contour = approx
        break

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

# print(document_contour.flatten('F')[0:4])
# print(document_contour.flatten('F')[4:8])
x = document_contour.flatten('F')[0:4]
y = document_contour.flatten('F')[4:8]
warped = four_point_transform(image, document_contour.reshape(4, 2))

# plt.figure()
# plt.imshow(image[:,:,[2,1,0]])
# plt.plot(x,y,"*")
# plt.show()

warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
cleaned = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


def rotate_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find edges using Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Determine the angle of rotation
    _, _, angle = cv2.minAreaRect(largest_contour)
    
    # Rotate the image
    rows, cols = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    
    return rotated_image

# rotated_image = rotate_image(image)
#


def applySobel(image):
    blurred = cv2.GaussianBlur(image,(5,5),0)
    grad_x = cv2.Sobel(blurred,cv2.CV_16S,1,0,ksize=3)
    grad_y = cv2.Sobel(blurred,cv2.CV_16S,0,1,ksize=3)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0)
    
    return grad

grad = applySobel(image)

kernel = np.ones((5,5),np.uint8)
img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel,iterations=3)


plt.figure()
plt.imshow(img[:,:,[2,1,0]])
plt.show()
