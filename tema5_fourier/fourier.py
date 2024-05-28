import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

# image = cv2.imread('images/cameraman.tif', cv2.IMREAD_GRAYSCALE)
image = cv2.imread('images/textos/desk.JPG', cv2.IMREAD_GRAYSCALE)


blurred = cv2.GaussianBlur(image,(5,5),0)
grad_x = cv2.Sobel(blurred,cv2.CV_16S,1,0,ksize=3)
grad_y = cv2.Sobel(blurred,cv2.CV_16S,0,1,ksize=3)

abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)
grad = cv2.addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0)


def image2fourier(img):
    image_dft = cv2.dft(np.double(img),flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(image_dft)
    image_mag = np.log(1 + cv2.magnitude(dft_shifted[:, :, 0], dft_shifted[:, :, 1]))
    return image_mag


def get_dominant_angle(image):
    # Compute the FFT of the image
    fft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    fft_shifted = np.fft.fftshift(fft)

    # Compute the phase spectrum
    magnitude, phase = cv2.cartToPolar(fft_shifted[:,:,0], fft_shifted[:,:,1])

    # Define a range of angles to search
    angles = np.arange(-90, 91, 1)

    best_angle = None
    best_corr = -1

    image_f = image.copy().astype(np.float32)
    for angle in angles:
        # Rotate the image
        rotated_image = rotate_image(image_f, angle)

        # Compute phase correlation
        corr = cv2.phaseCorrelate(image_f, rotated_image)

        # Check if current correlation is better
        # print(corr)
        # print(best_corr)
        if corr[1] > best_corr:
            best_corr = corr[1]
            best_angle = angle

    # print(best_angle)

    return best_angle

def rotate_image_to_align_axes(image):

    # magnitude_spectrum = image2fourier(image)
    # Compute Fourier transform
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    # Find dominant angle of rotation
    rows, cols = image.shape
       # Find dominant angle of rotation
    rows, cols = image.shape
    cy, cx = rows // 2, cols // 2
    cy_mag, cx_mag = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
    _, angle = cv2.cartToPolar(cx_mag - cx, cy_mag - cy)

    # print(angle)
    # Rotate image back
    angle_degrees = np.degrees(angle)
    rotated_image = image
    if angle_degrees != 0:
        rotated_image = cv2.warpAffine(image, cv2.getRotationMatrix2D((cx, cy), -angle_degrees, 1), (cols, rows))

    return rotated_image


def rotate_image(image, angle):
    # Get image center
    image_center = tuple(np.array(image.shape[1::-1]) / 2)

    # Define rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    # Perform the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    return rotated_image

image_fourier = image2fourier(image)
grad_fourier = image2fourier(grad)
grad_fourier_hori = rotate_image(grad_fourier,-90)
# angle = get_dominant_angle(grad)

_,grad_fourier_th = cv2.threshold(grad_fourier_hori,grad_fourier.mean()*1.5,255,cv2.THRESH_BINARY)

cdst = cv2.cvtColor(grad_fourier_th.copy().astype(np.uint8), cv2.COLOR_GRAY2RGB)

lines = cv2.HoughLines(grad_fourier_th.astype(np.uint8),1,np.pi/180,20)

if lines is not None:
    angles = []
    rhos = []
    # print("rho: ", lines)
    for i in range(0, len(lines)):
        rhos.append(lines[i][0][0])
        angles.append(lines[i][0][1])
    mean_rho = np.mean(rhos)
    mean_angle = np.median(angles)
    x = mean_rho*math.cos(mean_angle)
    y = mean_rho*math.sin(mean_angle)
    print("Mean rho: ", x)
    print("Mean angle: ", np.degrees(y))
    rho = mean_rho
    theta = mean_angle
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    size = np.linalg.norm(grad_fourier_th.shape) 
    pt1 = (int(x0 + size*(-b)), int(y0 + size *(a)))
    pt2 = (int(x0 - size*(-b)), int(y0 - size *(a)))
    cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    cdst = rotate_image(cdst, np.degrees(mean_angle))

grad_rotated = rotate_image(grad, np.degrees(mean_angle))
image_rotated = rotate_image(image, np.degrees(mean_angle))
grad_fourier_r = image2fourier(grad_rotated)

kernel = np.ones((5,5),np.uint8)
morph = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel,iterations=3)
morph_fourier = image2fourier(morph)

plt.figure()

plt.subplot(231)
plt.imshow(image, cmap='gray')

plt.subplot(234)
plt.imshow(image_fourier, cmap='gray')

plt.subplot(232)
plt.imshow(grad, cmap='gray')

plt.subplot(235)
plt.imshow(grad_fourier, cmap='gray')

plt.subplot(233)
plt.imshow(image_rotated, cmap='gray')

plt.subplot(236)
plt.imshow(grad_fourier_r, cmap='gray')

plt.figure()
plt.imshow(cdst)

plt.show()
