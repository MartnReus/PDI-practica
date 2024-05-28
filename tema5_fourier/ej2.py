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


## 1. Imagen con magnitud y fase cero

# image = cv2.imread("images/iguazu.jpg",cv2.IMREAD_GRAYSCALE)
# image_fourier = image2fourier(image)
# no_mag = fourier2image(np.zeros_like(image_fourier[0]),image_fourier[1],True)
# no_phase = fourier2image(image_fourier[0],np.zeros_like(image_fourier[1]))
# # no_mag = no_mag.astype(np.uint8) + np.ones_like(no_mag.astype(np.uint8))*128


## 2. Experimento de Openheim
puente = cv2.imread("images/puente.jpg",cv2.IMREAD_GRAYSCALE)
ferrari = cv2.imread("images/ferrari-c.png",cv2.IMREAD_GRAYSCALE)

mag_puente,phase_puente = image2fourier(puente)
mag_ferrari,phase_ferrari = image2fourier(ferrari)

puente_mag_f = fourier2image(mag_ferrari,phase_puente)
puente_pha_f = fourier2image(mag_puente,phase_ferrari)

ferrari_mag_p = fourier2image(mag_puente,phase_ferrari)
ferrari_pha_p = fourier2image(mag_ferrari,phase_puente)

plt.figure()

plt.subplot(321)
plt.title("Puente original")
plt.imshow(puente, cmap='gray')
plt.tight_layout()

plt.subplot(322)
plt.title("Ferrari original")
plt.imshow(ferrari, cmap='gray')
plt.tight_layout()

plt.subplot(323)
plt.title("Puente con magnitud de ferrari")
plt.imshow(puente_mag_f, cmap='gray')
plt.tight_layout()

plt.subplot(324)
plt.title("Ferrari con magnitud de puente")
plt.imshow(ferrari_mag_p, cmap='gray')
plt.tight_layout()

plt.subplot(325)
plt.title("Puente con fase de ferrari")
plt.imshow(puente_pha_f, cmap='gray')
plt.tight_layout()

plt.subplot(326)
plt.title("Ferrari con fase de puente")
plt.imshow(ferrari_pha_p, cmap='gray')
plt.tight_layout()

plt.show()


