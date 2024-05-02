import cv2
import matplotlib.pyplot as plt


def img2complement(img_rgb):
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_mod = img_hsv.copy()

    for y in range(img_hsv.shape[0]):
        for x in range(img_hsv.shape[1]):
            img_mod[y, x, 0] = (img_mod[y, x, 0] + 90) % 180

    img_mod = cv2.cvtColor(img_mod, cv2.COLOR_HSV2RGB)
    return img_mod


rosas = cv2.imread("images/rosas.jpg")
rosas = cv2.cvtColor(rosas, cv2.COLOR_BGR2RGB)

rosas_mod = img2complement(rosas)

plt.figure()

plt.subplot(121)
plt.imshow(rosas)
plt.title("Imagen original")

plt.subplot(122)
plt.imshow(rosas_mod)
plt.title("Imagen modificada")

plt.show()
