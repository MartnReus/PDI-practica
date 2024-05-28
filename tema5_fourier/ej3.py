import cv2
import numpy as np
import matplotlib.pyplot as plt

def image2fourier(img):
    image_dft = cv2.dft(img.astype(np.float32),flags= cv2.DFT_COMPLEX_OUTPUT)
    print(image_dft.shape)
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

# Image in spatial domanin
# Filter in frequency domain
def filter(img,filter,is_hp=False):
    mag,phase = image2fourier(img)
    print(mag[0:5,0:5])
    filtered_dft = mag*filter
    print(filtered_dft[0:5,0:5])
    filtered_img = fourier2image(filtered_dft,phase)
    # if is_hp:
    #     filtered_img = filtered_img.astype(np.uint8) + 127
    return filtered_img.astype(np.uint8)
    

def ideal_lp_filter(cutoff,shape):
    if shape[0]%2 != 0 or shape[1]%2 != 0:
        raise Exception("shape must be even on both axis")

    rows, cols = shape
    crow, ccol = rows // 2 , cols // 2  # Center of the image
    mask = np.zeros((rows, cols), np.float32)
    a = np.array([crow,ccol])
    for i in range(rows):
        for j in range(cols):
            b = np.array([i,j])
            if np.linalg.norm(a-b,2) <= cutoff:
                mask[i, j] = 1
    return mask.astype(np.uint8)


def butterworth_lp_filter(cutoff,order,shape):
    if shape[0]%2 != 0 or shape[1]%2 != 0:
        raise Exception("shape must be even on both axis")

    rows, cols = shape
    crow, ccol = rows // 2 , cols // 2  # Center of the image
    mask = np.zeros((rows, cols), np.float32)
    a = np.array([crow,ccol])
    # a = np.ones_like(mask) * a
    # den = 1 + (np.linalg.norm(a-mask,2)/cutoff)**(2*order)
    # mask = 1/den
    for i in range(rows):
        for j in range(cols):
            b = np.array([i,j])
            den = 1 + (np.linalg.norm(a-b,2)/cutoff)**(2*order)
            mask[i,j] = 1/den
    return mask

def gaussian_lp_filter(cutoff,shape):
    if shape[0]%2 != 0 or shape[1]%2 != 0:
        raise Exception("shape must be even on both axis")

    rows, cols = shape
    crow, ccol = rows // 2 , cols // 2  # Center of the image
    mask = np.zeros((rows, cols), np.float32)
    a = np.array([crow,ccol])
    for i in range(rows):
        for j in range(cols):
            b = np.array([i,j])
            # den = 1 + (np.linalg.norm(a-b,2)/cutoff)**(2*order)
            mask[i,j] = np.exp(-np.linalg.norm(a-b,2)**2/(2*cutoff**2))
    return mask

def ideal_hp_filter(cutoff,shape):
    if shape[0]%2 != 0 or shape[1]%2 != 0:
        raise Exception("shape must be even on both axis")

    rows, cols = shape
    crow, ccol = rows // 2 , cols // 2  # Center of the image
    mask = np.zeros((rows, cols), np.uint8)
    a = np.array([crow,ccol])
    for i in range(rows):
        for j in range(cols):
            b = np.array([i,j])
            if np.linalg.norm(a-b,2) >= cutoff:
                mask[i, j] = 1
    return mask.astype(np.uint8)


def butterworth_hp_filter(cutoff,order,shape):
    if shape[0]%2 != 0 or shape[1]%2 != 0:
        raise Exception("shape must be even on both axis")

    rows, cols = shape
    crow, ccol = rows // 2 , cols // 2  # Center of the image
    mask = np.zeros((rows, cols), np.float32)
    a = np.array([crow,ccol])
    # a = np.ones_like(mask) * a
    # den = 1 + (np.linalg.norm(a-mask,2)/cutoff)**(2*order)
    # mask = 1/den
    for i in range(rows):
        for j in range(cols):
            b = np.array([i,j])
            den = 1 + (np.linalg.norm(a-b,2)/cutoff)**(-2*order)
            mask[i,j] = 1/den
    return mask

def gaussian_hp_filter(cutoff,shape):
    if shape[0]%2 != 0 or shape[1]%2 != 0:
        raise Exception("shape must be even on both axis")

    rows, cols = shape
    crow, ccol = rows // 2 , cols // 2  # Center of the image
    mask = np.zeros((rows, cols), np.float32)
    a = np.array([crow,ccol])
    for i in range(rows):
        for j in range(cols):
            b = np.array([i,j])
            # den = 1 + (np.linalg.norm(a-b,2)/cutoff)**(2*order)
            mask[i,j] = 1 - np.exp(-np.linalg.norm(a-b,2)**2/(2*cutoff**2))
    return mask


camaleon = cv2.imread('images/camaleon.tif', cv2.IMREAD_GRAYSCALE)
camaleon  = cv2.copyMakeBorder(camaleon,0,1,0,0,cv2.BORDER_REPLICATE)
print(camaleon.shape)

# lp_filter1 = ideal_lp_filter(60,camaleon.shape) 
# lp_filter2 = butterworth_lp_filter(80,5,camaleon.shape) 
# lp_filter3 = gaussian_lp_filter(80,camaleon.shape) 
# filtered_image_ideal = filter(camaleon,lp_filter1)
# filtered_image_butterworth = filter(camaleon,lp_filter2)
# filtered_image_gaussian = filter(camaleon,lp_filter3)

hp_filter1 = ideal_hp_filter(20,camaleon.shape) 
hp_filter2 = butterworth_hp_filter(20,5,camaleon.shape) 
hp_filter3 = gaussian_hp_filter(20,camaleon.shape) 

hp_filter1 = np.ones_like(hp_filter1) + hp_filter1
# print(hp_filter1[0:10,0:10])

filtered_image_ideal = filter(camaleon,hp_filter1,is_hp=True)
# filtered_image_butterworth = filter(camaleon,hp_filter2,is_hp=True)
# filtered_image_gaussian = filter(camaleon,hp_filter3,is_hp=True)

# plt.figure()
# plt.subplot(231)
# plt.imshow(filtered_image_ideal,cmap="gray")
# plt.tight_layout()
#
# plt.subplot(234)
# plt.imshow(lp_filter1,cmap="gray")
# plt.tight_layout()
#
# plt.subplot(232)
# plt.imshow(filtered_image_butterworth,cmap="gray")
# plt.tight_layout()
#
# plt.subplot(235)
# plt.imshow(lp_filter2,cmap="gray")
# plt.tight_layout()
#
#
# plt.subplot(233)
# plt.imshow(filtered_image_gaussian,cmap="gray")
# plt.tight_layout()
#
# plt.subplot(236)
# plt.imshow(lp_filter3,cmap="gray")
# plt.tight_layout()
#
# plt.show()
    

plt.figure()
plt.subplot(231)
plt.imshow(filtered_image_ideal,cmap="gray")
plt.tight_layout()

plt.subplot(234)
plt.imshow(hp_filter1,cmap="gray")
plt.tight_layout()

plt.subplot(232)
plt.imshow(filtered_image_butterworth,cmap="gray")
plt.tight_layout()

plt.subplot(235)
plt.imshow(hp_filter2,cmap="gray")
plt.tight_layout()


plt.subplot(233)
plt.imshow(filtered_image_gaussian,cmap="gray")
plt.tight_layout()

plt.subplot(236)
plt.imshow(hp_filter3,cmap="gray")
plt.tight_layout()

plt.show()


