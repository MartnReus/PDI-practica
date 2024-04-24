import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(PATH, "../utils"))

from aux import quantize_image, bitplane_slices

# path of the current file

earth_path = "images/earth.bmp"
cameraman_path = "images/cameraman.tif"
deforestacion_path = "images/Deforestacion.png"

earth_path = os.path.join(PATH, "..", earth_path)
cameraman_path = os.path.join(PATH, "..", cameraman_path)
deforestacion_path = os.path.join(PATH, "..", deforestacion_path)


img_earth = cv2.imread(earth_path, cv2.IMREAD_GRAYSCALE)
img_cameraman = cv2.imread(cameraman_path, cv2.IMREAD_GRAYSCALE)
img_deforestacion = cv2.imread(deforestacion_path, cv2.IMREAD_GRAYSCALE)

bit_planes = bitplane_slices(img_earth)
sum_planes = bit_planes[0, :, :]
for i in range(1, 3):
    sum_planes += bit_planes[i, :, :]

cameraman_quantized = quantize_image(img_cameraman, 2)

fig, ax = plt.subplots(2, 2)
x, y = (100, 0)
# print("Shape deforestacion: ", img_deforestacion.shape)
# print("Shape cameraman: ", cameraman_quantized.shape)

img_deforestacion_roi = img_deforestacion[
    y : y + cameraman_quantized.shape[0], x : x + cameraman_quantized.shape[1]
].copy()

img_deforestacion_roi = img_deforestacion_roi.astype(np.uint8) + cameraman_quantized
img_deforestacion_mod = img_deforestacion.copy()
img_deforestacion_mod[
    y : y + cameraman_quantized.shape[0], x : x + cameraman_quantized.shape[1]
] = img_deforestacion_roi

bitplanes_def = bitplane_slices(img_deforestacion)
sum_planes_def = (
    bitplanes_def[0, :, :]  # + bitplanes_def[1, :, :] + bitplanes_def[2, :, :]
)

bitplanes_def_mod = bitplane_slices(img_deforestacion_mod)
sum_planes_def_mod = (
    bitplanes_def_mod[
        0, :, :
    ]  # + bitplanes_def_mod[1, :, :] + bitplanes_def_mod[2, :, :]
)

ax[0, 0].imshow(img_deforestacion, cmap="gray")
ax[0, 1].imshow(img_deforestacion_mod, cmap="gray")
ax[1, 0].imshow(sum_planes_def, cmap="gray")
ax[1, 1].imshow(sum_planes_def_mod, cmap="gray")

plt.show()
