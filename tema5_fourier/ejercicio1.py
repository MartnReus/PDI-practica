import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

horizontal = np.zeros([256, 256], dtype=np.uint8)

def generate_line(size,vertical=True, line_width=5,padding=0):
    image = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    image[padding:image.shape[1]-padding, center - line_width // 2: center + line_width // 2 + 1] = 255
    # image[padding:image.shape[1]-padding,25 - line_width // 1: 25 + line_width // 2 + 1] = 255
    # image[padding:image.shape[1]-padding,255-25 - line_width // 2: 255-25 + line_width // 2 + 1] = 255
    if (vertical):
        return image
    return image.T

def generate_gaussian_image(width, height, max_intensity=255):
    # Create a black background image
    img = np.zeros((height, width), dtype=np.uint8)

    # Calculate the center coordinates
    center_x = width // 2
    center_y = height // 2

    # Create a meshgrid of coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Calculate distance from the center for each pixel
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    # Calculate Gaussian intensity
    sigma = max(width, height) / 50  # Adjust the sigma value to control the spread of the Gaussian
    intensity = max_intensity * np.exp(-0.5 * (distance / sigma)**2)

    # Ensure intensity values are within the valid range
    intensity[intensity > max_intensity] = max_intensity

    # Set the intensity values in the image
    img = intensity.astype(np.uint8)

    return img

vertical = generate_line(256,line_width=3,padding=8)
vertical = generate_gaussian_image(256,256)
cv2.imwrite('images/vertical.jpg', vertical)
horizontal = generate_line(256,vertical=False,line_width=3,padding=50)

vertical_dft = cv2.dft(np.double(vertical),flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shifted = np.fft.fftshift(vertical_dft)
vertical_mag = np.log(1+ cv2.magnitude(dft_shifted[:, :, 0], dft_shifted[:, :, 1]))

horizontal_dft = cv2.dft(np.double(horizontal),flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shifted = np.fft.fftshift(horizontal_dft)
horizontal_mag = np.log(1+ cv2.magnitude(dft_shifted[:, :, 0], dft_shifted[:, :, 1]))

plt.figure()

plt.subplot(121)
plt.imshow(vertical, cmap='gray')

plt.subplot(122)
plt.imshow(vertical_mag, cmap='gray')

# plt.subplot(221)
# plt.imshow(vertical, cmap='gray')
#
# plt.subplot(222)
# plt.imshow(vertical_mag, cmap='gray')

# plt.subplot(223)
# plt.imshow(horizontal, cmap='gray')
#
# plt.subplot(224)
# plt.imshow(horizontal_mag, cmap='gray')

plt.show()
