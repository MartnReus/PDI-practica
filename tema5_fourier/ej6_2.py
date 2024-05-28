import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def image2fourier(img):
    image_dft = cv2.dft(img.astype(np.float32),flags= cv2.DFT_COMPLEX_OUTPUT)
    # print(image_dft.shape)
    dft_shifted = np.fft.fftshift(image_dft)
    mag,phase = cv2.cartToPolar(dft_shifted[:, :, 0], dft_shifted[:, :, 1])
    mag = np.log(np.add(np.ones_like(mag),mag))
    return [mag,phase]

def fourier2image(mag, phase,unaltered_mag=False):
    if (not(unaltered_mag)):
        mag = np.exp(mag) - np.ones_like(mag)
    
    real, imag = cv2.polarToCart(mag, phase)
    combined = cv2.merge((real, imag))
    
    dft_ishift = np.fft.ifftshift(combined)
    img_back = cv2.idft(dft_ishift,flags=cv2.DFT_SCALE | cv2.DFT_COMPLEX_INPUT)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    
    img_back = img_back.astype(np.uint8)
    
    return img_back

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

# def rotate_image(image,angle):
#     if angle == 0:
#         return image
#     # rows, cols = image.shape
#     # cy, cx = rows // 2, cols // 2
#     # rotated_image = cv2.warpAffine(image, cv2.getRotationMatrix2D((cx, cy), -angle, 1), (cols, rows))
#     # return rotated_image
#     # Get the dimensions of the image
#     (h, w) = image.shape[:2]
#
#     # Get the center of the image
#     center = (w // 2, h // 2)
#
#     # Compute the rotation matrix
#     M = cv2.getRotationMatrix2D(center, -angle, 1.0)
#
#     # Perform the actual rotation
#     rotated = cv2.warpAffine(image, M, (w, h))
#
#     return rotated



parrafo0 = cv2.imread('images/parrafo0.jpg', cv2.IMREAD_GRAYSCALE)
parrafo1 = cv2.imread('images/parrafo1.jpg', cv2.IMREAD_GRAYSCALE)
# parrafo1 = cv2.GaussianBlur(parrafo1,(5,5),0)

parrafo0_mag,parrafo0_pha = image2fourier(parrafo0)
parrafo1_mag,parrafo1_pha = image2fourier(parrafo1)
_,parrafo0_mag_th = cv2.threshold(parrafo0_mag,parrafo0_mag.mean()*1.5,255,cv2.THRESH_BINARY)
_,parrafo1_mag_th = cv2.threshold(parrafo1_mag,parrafo1_mag.mean()*1.5,255,cv2.THRESH_BINARY)

non_zero_coords = np.column_stack(np.where(parrafo0_mag_th > 0))
min_y, min_x = non_zero_coords[np.argmin(non_zero_coords[:, 0] * parrafo0_mag_th.shape[1] + non_zero_coords[:, 1])]
max_y, max_x = non_zero_coords[np.argmax(non_zero_coords[:, 0] * parrafo0_mag_th.shape[1] + non_zero_coords[:, 1])]
print(f"Min: ({min_x},{min_y})")
print(f"Max: ({max_x},{max_y})")


m1 = min_y/min_x
m2 = max_y/max_x
tan = (m2-m1)/(1+m1*m2)
angle0 = np.degrees(np.arctan(tan))
print(angle0)
parrafo0_rot = rotate_image(parrafo0,angle0)

non_zero_coords = np.column_stack(np.where(parrafo1_mag_th > 0))
min_y, min_x = non_zero_coords[np.argmin(non_zero_coords[:, 0] * parrafo1_mag_th.shape[1] + non_zero_coords[:, 1])]
max_y, max_x = non_zero_coords[np.argmax(non_zero_coords[:, 0] * parrafo1_mag_th.shape[1] + non_zero_coords[:, 1])]
print(f"Min: ({min_x},{min_y})")
print(f"Max: ({max_x},{max_y})")

cx = parrafo1.shape[1]//2
cy = parrafo1.shape[0]//2

max_y = max_y - cy
min_y = min_y - cy

max_x = max_x - cx
min_x = min_x - cx

m1 = 0
m2 = (max_y-min_y)/(max_x-min_x)
tan = (m2-m1)/(1+m1*m2)
angle1 = np.degrees(np.arctan(tan))

print(angle1)

# parrafo1_rot_mag = rotate_image(parrafo1_mag,angle1)
# parrafo1_mag_th = rotate_image(parrafo1_mag_th,angle1)
# parrafo1_rot = fourier2image(parrafo1_rot_mag,parrafo1_pha)
parrafo1_rot = rotate_image(parrafo1,angle0)
# parrafo1_rot = rotate_image(parrafo1,30)
# parrafo1_rot = ndimage.rotate(parrafo1,-angle1)
parrafo1_rot_mag,parrafo1_rot_pha = image2fourier(parrafo1_rot)
# _,parrafo1_mag_th_rot = cv2.threshold(parrafo1_rot_mag,parrafo1_rot_mag.mean()*1.5,255,cv2.THRESH_BINARY)

plt.figure()

plt.subplot(221)
plt.imshow(parrafo0,cmap='gray')
# plt.imshow(parrafo0_rot,cmap='gray')

plt.subplot(222)
# plt.imshow(parrafo1,cmap='gray')
plt.imshow(parrafo1_rot,cmap='gray')

plt.subplot(223)
plt.imshow(parrafo0_mag_th,cmap='gray')
# plt.imshow(parrafo0_mag,cmap='gray')

plt.subplot(224)
# plt.imshow(parrafo1_mag,cmap='gray')
# plt.imshow(parrafo1_rot_mag,cmap='gray')
plt.imshow(parrafo1_mag_th,cmap='gray')
# plt.imshow(parrafo1_mag_th_rot,cmap='gray')
# plt.imshow(parrafo1_pha,cmap='gray')

plt.show()
