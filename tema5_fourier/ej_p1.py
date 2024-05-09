import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_grayscale_histogram(image_path):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Calculate the histogram
    hist = cv2.calcHist([img], [0], None, [256], [0,256])
    
    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.title('Grayscale Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.bar(np.arange(256), hist.flatten(), color='black', width=1)
    plt.xlim([0, 256])
    plt.show()

# image = cv2.imread('images/p1_sucio.png', cv2.IMREAD_GRAYSCALE)
image = plot_grayscale_histogram('images/p1_sucio.png')


