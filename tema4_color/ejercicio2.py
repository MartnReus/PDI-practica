import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("images/rio.jpg")
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hist = cv2.calcHist(image_hsv, [2], None, [256], [0, 255])

plt.figure()
plt.bar(range(256), np.squeeze(hist))
plt.show()


def nothing(x):
    pass


cv2.namedWindow("Imagen")
cv2.createTrackbar("Lower V", "Imagen", 0, 255, nothing)
cv2.createTrackbar("Upper V", "Imagen", 10, 255, nothing)

while True:
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_v = cv2.getTrackbarPos("Lower V", "Imagen")
    upper_v = cv2.getTrackbarPos("Upper V", "Imagen")

    lower_bound = np.array([0, 0, lower_v])
    upper_bound = np.array([255, 255, upper_v])
    mask = cv2.inRange(image_hsv, lower_bound, upper_bound)

    new_hsv = (30, 255, 255)
    image_hsv[mask == 255, :] = new_hsv
    # el poder hacer [mask==255] es un feature de NumPy que se llama boolean indexing

    painted_image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow("Imagen", painted_image)

    k = cv2.waitKey(1) & 0xFF  # ESC para salir
    if k == 27:
        break


cv2.destroyAllWindows()
