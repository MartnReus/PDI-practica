import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('images/cameraman.tif', cv2.IMREAD_GRAYSCALE)

image_dft = cv2.dft(np.double(image),flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shifted = np.fft.fftshift(image_dft)
image_mag = np.log(1 + cv2.magnitude(dft_shifted[:, :, 0], dft_shifted[:, :, 1]))


plt.figure()

plt.subplot(121)
plt.imshow(image, cmap='gray')

plt.subplot(122)
plt.imshow(image_mag, cmap='gray')

plt.show()
