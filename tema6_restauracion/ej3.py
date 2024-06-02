import cv2
import matplotlib.pyplot as plt
import numpy as np

sangre = cv2.imread('images/sangre.jpg')


def add_impulsive_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    noisy_image = noisy_image / 255.0

    num_salt = np.ceil(salt_prob * image[:, :, 0].size)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image[:, :, 0].shape]
    noisy_image[coords[0], coords[1], :] = 1

    num_pepper = np.ceil(pepper_prob * image[:, :, 0].size)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image[:, :, 0].shape]
    noisy_image[coords[0], coords[1], :] = 0

    noisy_image_uint8 = (noisy_image * 255).astype(np.uint8)
    return noisy_image_uint8

def add_gaussian_noise(image, mean, var):
    noisy_image = np.copy(image).astype('float64')
    noise = np.random.normal(mean, var, image.shape)
    noisy_image += noise
    noisy_image = np.clip(noisy_image,0,255).astype('uint8')

    return noisy_image


# noisy_sangre = add_impulsive_noise(sangre, 0.05, 0.05)
noisy_sangre = add_gaussian_noise(sangre, 0, 20)

sangre_hsv = cv2.cvtColor(sangre,cv2.COLOR_BGR2HSV)
noisy_sangre_hsv = cv2.cvtColor(noisy_sangre,cv2.COLOR_BGR2HSV)


hist_sangre = cv2.calcHist(sangre_hsv, [2], None, [256], [0, 256])
hist_sangre_noisy = cv2.calcHist(noisy_sangre_hsv, [2], None, [256], [0, 256])


sangre = cv2.cvtColor(sangre,cv2.COLOR_BGR2RGB)
noisy_sangre = cv2.cvtColor(noisy_sangre,cv2.COLOR_BGR2RGB)

plt.figure()

plt.subplot(221)
plt.imshow(sangre)
plt.axis('off')

plt.subplot(222)
plt.imshow(noisy_sangre)
plt.axis('off')

plt.subplot(223)
plt.bar(range(256), np.squeeze(hist_sangre))

plt.subplot(224)
plt.bar(range(256), np.squeeze(hist_sangre_noisy))

plt.show()
