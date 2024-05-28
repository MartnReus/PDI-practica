import cv2
from cv2.gapi import dilate
import numpy as np
import pytesseract
import matplotlib.pyplot as plt


class Escaner:

    frames = np.array([])

    def __init__(self):
        pass


    def show_scaned_image(self,path):
        image = cv2.imread(path)
        thr = 100
        thr2 = 700

        # Rotar la imagen para probar con otro lado
        # timg = np.transpose(image.copy())
        # timg = np.flip(timg,[1])
        # timg = np.transpose(timg)
        # image = np.flip(timg,[0])

        scanned_image = self.process_frame(image,fix_position=True,thr2=thr2,thr=thr)
        while True:
            print("Trying again")
            thr -= 5
            thr2 -=25
            scanned_image = self.process_frame(image,fix_position=True,thr2=thr2,thr=thr)
            if type(scanned_image) is not int: break

        # Convert to grayscale
        gray_image = cv2.cvtColor(scanned_image, cv2.COLOR_BGR2GRAY)

        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_image = binary_image
        
        dilated_image = cv2.medianBlur(binary_image, 7)

        # grab the dimensions of the image and calculate the center of the
# image


        f = np.fft.fft2(dilated_image)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        # magnitude_spectrum_uint8 = np.uint8(magnitude_spectrum)
        (h, w) = dilated_image.shape[:2]
        center = (w // 2, h // 2)

        x_mag = magnitude_spectrum[center[1],center[0]-5:center[0]]
        y_mag = magnitude_spectrum[center[1]-5:center[1],center[0]]

        x_mag_avg = np.mean(x_mag)
        y_mag_avg = np.mean(y_mag)

        print(f"Mag en x: {x_mag_avg} -  Mag en y: {y_mag_avg}")
        # plt.imshow(magnitude_spectrum, cmap='gray')
        # plt.title('Magnitude Spectrum')
        # plt.show()

        if x_mag_avg > y_mag_avg:
            timg = np.transpose(dilated_image.copy())
            dilated_image = np.flip(timg,[1])

        # dilated_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel,iterations=1)
        # dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
        # dilated_image = cv2.erode(binary_image, kernel, iterations=1)


        window_name = "Imagen Rotada"
        cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, dilated_image)
        cv2.waitKey(0)

        cv2.destroyAllWindows()
        
        recognized_text = ""
        tessdata_dir_config = r'--tessdata-dir "/usr/share/tessdata/"'
        recognized_text = pytesseract.image_to_string(dilated_image,config=tessdata_dir_config,lang='spa')

        return recognized_text


    def load_video(self, path,save_path):
        print("Loading video...")
        cap = cv2.VideoCapture(path)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter(save_path, fourcc, 20.58, (1920, 1080))

        while cap.isOpened():
            ret, frame = cap.read()

            if not(ret):
                print("Video loaded!")
                break

            processed_frame = self.process_frame(frame)
            # processed_frame = frame
            out.write(processed_frame)


        cap.release()
        out.release()

    def process_frame(self,frame,thr=30,thr2=200, fix_position=False):
        image = frame

        ratio = image.shape[0] / 500.0
        orig = image.copy()
        image = cv2.resize(image, (int(image.shape[1] / ratio), 500))
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        

        
        kernel = np.ones((5,5),np.uint8)
        morphed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel,iterations=3)

        edged = cv2.Canny(morphed, thr, thr2)
        
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]


        bounding_box = None
        screenCnt = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # print("approx polygon: ",len(approx)) 
            if len(approx) > 1 :
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                hull = cv2.convexHull(approx)

                approx_hull = cv2.approxPolyDP(hull, 0.02 * peri, True)
                # print("approx hull: ",len(approx_hull))
                if len(approx_hull) == 4:
                    bounding_box = approx_hull
                    screenCnt = approx_hull
                    break

                # bounding_box = hull
                # bounding_box = box
                # screenCnt = box
                break

        # Plot the bounding box
        if bounding_box is not None:
            image_with_bounding_box = self.plot_bounding_box(image, bounding_box)
        else:
            image_with_bounding_box = image

        if not(fix_position):
            bounded_image = cv2.resize(image_with_bounding_box,(orig.shape[1],orig.shape[0]))
            return bounded_image

        # Return 1 -> document not detected
        if screenCnt is None:
            return 1

        warped = self.four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
        warped = cv2.GaussianBlur(warped, (3, 3), 0)


        return warped

    def order_points(self,pts):
        rect = np.zeros((4, 2), dtype = "float32")
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        print(f"Puntos a orderar: ${pts}")
        print(f's: ${s} \ndiff: ${diff}')
        print(f"Rect: ${rect}")
        
        return rect

    def four_point_transform(self,image, pts):
        rect = self.order_points(pts)
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


    def save_video(self, path):
        print(f'Saving video in {path}...')
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter(path, fourcc, 20.0, (640, 480))
        for frame in self.frames:
            out.write(frame)
        out.release()
        print("Video saved!")


    def plot_bounding_box(self,image, box, box_color=(0, 0, 255), thickness=2):
        image_with_box = image.copy()
        box = np.intp(box)
        cv2.drawContours(image_with_box, [box], -1, box_color, thickness)
        return image_with_box


