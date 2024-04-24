import cv2
import numpy as np


def kmeans():
    print("importado")


def quantize_image(image, levels):
    quantized = np.floor_divide(image, 256 / (2**levels))
    return quantized


def bitplane_slices(image):
    img_bin = []
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            img_bin.append(np.binary_repr(image[y, x], width=8))

    bit_planes = np.empty((8, image.shape[0], image.shape[1]))
    for i in range(8):
        bit_planes[i, :, :] = (
            np.array([int(plane[7 - i]) for plane in img_bin], dtype=np.uint8)
            * pow(2, 7 - i)
        ).reshape(image.shape[0], image.shape[1])

    return bit_planes.astype(np.uint8)
