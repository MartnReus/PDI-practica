import cv2
import numpy as np
import matplotlib.pyplot as plt


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


def perfiles_intensidad_rgb(img_rgb, fix_line, dim=0):
    if dim is None or dim == 0:
        r = img_rgb[fix_line, :, 0]
        g = img_rgb[fix_line, :, 1]
        b = img_rgb[fix_line, :, 2]
    else:
        r = img_rgb[:, fix_line, 0]
        g = img_rgb[:, fix_line, 1]
        b = img_rgb[:, fix_line, 2]

    plt.figure()
    plt.title("Perfiles de intensidad RGB")

    plt.plot(r, "r")
    plt.plot(g, "g")
    plt.plot(b, "b")

    plt.show()


def perfiles_intensidad_hsv(img_rgb, fix_line, dim=0):
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    if dim is None or dim == 0:
        h = img_hsv[fix_line, :, 0]
        s = img_hsv[fix_line, :, 1]
        v = img_hsv[fix_line, :, 2]
    else:
        h = img_hsv[:, fix_line, 0]
        s = img_hsv[:, fix_line, 1]
        v = img_hsv[:, fix_line, 2]

    plt.figure()
    plt.title("Perfiles de intensidad HSV")

    plt.plot(h)
    plt.plot(s)
    plt.plot(v)
    plt.legend(["Hue", "Saturation", "Value"])

    plt.show()
