import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_rotation_angle(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 200, 200, 200, 200, cv2.BORDER_REPLICATE)
    
    # Perform Fourier Transform
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    
    magnitude_spectrum_uint8 = np.uint8(magnitude_spectrum)

    
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)

    x_mag = magnitude_spectrum_uint8[center[1],center[0]-1]
    y_mag = magnitude_spectrum_uint8[center[1]-1,center[0]]

    print(f"Mag en x: {x_mag} -  Mag en y: {y_mag}")


    top_row = magnitude_spectrum_uint8[0,:]
    max_top = np.argmax(top_row)
    max_left = np.argmax(magnitude_spectrum_uint8[:,0])
    print(f'Max top: {max_top} Max left: {max_left}')
    x_p = (max_left,0)
    y_p = (0,max_top+w//2)

    print(f'Center: {center}')
    print(f'x axis: {x_p}')
    print(f'y axis: {y_p}')
    # Display the magnitude spectrum
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum')
    plt.show()
    # edges = cv2.Canny(magnitude_spectrum_uint8, 50, 150, apertureSize=3)
    # lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    #
    # # Find the most prominent line
    # if lines is not None:
    #     for rho, theta in lines[0]:
    #         angle = np.degrees(theta) - 90
    #         return angle
    return 0

def correct_rotation(image, angle):
    # Get the image size
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Compute the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    print(M)
    
    # Perform the rotation
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

# Load image

image_path = 'images/dataset/sampleDatasetC2/input_sample/00001.jpg'
# image = cv2.imread('images/parrafo1.jpg')
image = cv2.imread(image_path)

# Detect rotation angle
angle = detect_rotation_angle(image)
print(f"Detected rotation angle: {angle} degrees")

# Correct the rotation
corrected_image = correct_rotation(image, -angle)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
corrected_image = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB)
# Display the original and corrected images
plt.figure()

plt.subplot(121)
plt.imshow(image)
plt.title('Original Image')

plt.subplot(122)
plt.imshow(corrected_image)
plt.title('Corrected Image')

plt.show()

