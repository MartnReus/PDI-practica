import cv2
import matplotlib.pyplot as plt
import os
import sys

PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(PATH, "../utils"))
from aux import perfiles_intensidad_rgb, perfiles_intensidad_hsv

patron = cv2.imread("images/patron.tif")
# patron = cv2.imread("images/rosas.jpg")
patron = cv2.cvtColor(patron, cv2.COLOR_BGR2RGB)

patron_hsv = cv2.cvtColor(patron, cv2.COLOR_RGB2HSV)
patron_mod = patron_hsv.copy()

for y in range(patron_hsv.shape[0]):
    for x in range(patron_hsv.shape[1]):
        patron_mod[y, x, 0] = (120 - patron_mod[y, x, 0]) % 180
        patron_mod[y, x, 1] = 255
        patron_mod[y, x, 2] = 255

patron_mod = cv2.cvtColor(patron_mod, cv2.COLOR_HSV2RGB)

plt.figure()

plt.subplot(121)
plt.imshow(patron)
plt.title("Imagen original")

plt.subplot(122)
plt.imshow(patron_mod)
plt.title("Imagen modificada")

plt.show()

perfiles_intensidad_rgb(patron_mod, 10, dim=0)
perfiles_intensidad_hsv(patron_mod, 10, dim=0)
