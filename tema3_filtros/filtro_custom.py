import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load image
building = cv2.imread("images/cameraman.tif",cv2.IMREAD_GRAYSCALE)
kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
kernel = np.transpose(kernel)

building = cv2.filter2D(building,-1,kernel)


plt.figure()
plt.imshow(building,cmap="gray")
plt.show()
