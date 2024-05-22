import cv2
import pytesseract
from matplotlib import pyplot as plt

# Load image from webcam
image = cv2.imread("images/texto.jpeg",cv2.IMREAD_COLOR)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# Thresholding
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Dilation and erosion
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
binary = cv2.dilate(binary, kernel, iterations=1)
binary = cv2.erode(binary, kernel, iterations=1)

# Deskewing
coords = cv2.findNonZero(binary)
angle = cv2.minAreaRect(coords)[-1]
if angle < -45:
    angle = -(90 + angle)
else:
    angle = -angle
(h, w) = binary.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# Resizing
resized = cv2.resize(rotated, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# Use OCR to extract text
tessdata_dir_config = r'--tessdata-dir "/usr/share/tessdata/"'
text = pytesseract.image_to_string(resized,config=tessdata_dir_config,lang='spa')
plt.figure()
plt.imshow(resized,cmap='gray')
plt.show()

print("Recognized Text:", text)
