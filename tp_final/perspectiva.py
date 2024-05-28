import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    print(f's: ${s} \ndiff: ${diff}')
    
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def plot_contours(image, contours, contour_color=(0, 255, 0), thickness=2):
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, -1, contour_color, thickness)
    return image_with_contours


def plot_bounding_box(image, box, box_color=(0, 0, 255), thickness=2):
    image_with_box = image.copy()
    box = np.intp(box)
    cv2.drawContours(image_with_box, [box], -1, box_color, thickness)
    return image_with_box

def applySobel(image):
    blurred = cv2.GaussianBlur(image,(5,5),0)
    grad_x = cv2.Sobel(blurred,cv2.CV_16S,1,0,ksize=5)
    grad_y = cv2.Sobel(blurred,cv2.CV_16S,0,1,ksize=5)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0)
    
    return grad

def scan_document(image_path, thr=50,thr2=500):
    image = cv2.imread(image_path)

    # kernel = np.ones((3,3),np.uint8)
    # image = cv2.erode(image,kernel)
    # borderType = cv2.BORDER_CONSTANT
    # top = int(0.02 * image.shape[0]) # shape[0] = rows
    # bottom = top
    # left = int(0.02 * image.shape[1]) # shape[1] = cols
    # right = left
    #
    # image = cv2.copyMakeBorder(image, top, bottom, left, right, borderType)
    # image = cv2.copyMakeBorder(image, top, bottom, 2, 2, borderType,None,[0,0,0])

    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = cv2.resize(image, (int(image.shape[1] / ratio), 500))
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    
    kernel = np.ones((5,5),np.uint8)
    morphed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel,iterations=3)

    edged = cv2.Canny(morphed, thr, thr2)
    # edged = applySobel(gray)
    plt.figure()
    plt.subplot(121)
    plt.imshow(morphed,cmap='gray')

    plt.subplot(122)
    plt.imshow(edged,cmap='gray')
    plt.show()
    
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

    # Plot all detected contours
    image_with_all_contours = plot_contours(image, contours)
    cv2.imshow('All Contours', image_with_all_contours)

    # print(contours)
    screenCnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        print("approx polygon: ",len(approx)) 
        if len(approx) > 1 :
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            hull = cv2.convexHull(approx)

            approx_hull = cv2.approxPolyDP(hull, 0.02 * peri, True)
            print("approx hull: ",len(approx_hull))
            if len(approx_hull) == 4:
                bounding_box = approx_hull
                screenCnt = approx_hull
                break

            # bounding_box = hull
            # bounding_box = box
            # screenCnt = box
            break
    if screenCnt is None:
        raise ValueError("No document-like contour found")

      # Plot the bounding box
    image_with_bounding_box = plot_bounding_box(image, bounding_box)
    cv2.imshow('Bounding Box', image_with_bounding_box)


    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

    warped = cv2.GaussianBlur(warped, (3, 3), 0)


    return warped


# for thr in range(20, 70, 10):
#     for thr2 in range(100, 500, 100):
#         try:
#             scanned_image = scan_document("images/textos/chart.JPG",thr,thr2)
#             window_name = "Imagen Rotada"
#             cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
#             cv2.imshow(window_name, scanned_image)
#             cv2.waitKey(0)
#         except: 
#             continue

# scanned_image = scan_document("images/textos/chart.JPG",20,100)
scanned_image = scan_document("images/texto1.jpg",30,200)

print("\n------ Sin procesar ------\n")

# scanned_image = scan_document("images/textos/chart.JPG",20,100)
orig_image = cv2.imread("images/texto1.jpg")
tessdata_dir_config = r'--tessdata-dir "/usr/share/tessdata/"'
text_orig = pytesseract.image_to_string(orig_image,config=tessdata_dir_config,lang='spa')

print(text_orig)

print("\n------ Procesada ------\n")
tessdata_dir_config = r'--tessdata-dir "/usr/share/tessdata/"'
text_processed = pytesseract.image_to_string(scanned_image,config=tessdata_dir_config,lang='spa')
print(text_processed)

# scanned_image = scan_document("images/texto.jpeg",40,100)
window_name = "Imagen Rotada"
cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
cv2.imshow(window_name, scanned_image)
cv2.waitKey(0)

cv2.destroyAllWindows()

