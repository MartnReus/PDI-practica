from core.escaner import Escaner
import cv2
import concurrent.futures
import matplotlib.pyplot as plt
import time
import numpy as np


def run_ocr(ocr_model,image,esc,idx,label):
    print(f"Processing {idx} - Running {ocr_model} on {label}")
    text = esc.ocr_read(image,ocr=ocr_model)
    print(f"Processing {idx} - Finished {ocr_model} on {label}")
    # text = f'{label}: \n{text}'
    return text

def get_metrics(output_text,real_text,esc):
    results = dict()
    results["cer"]= esc.metrics(output_text,real_text,metric='cer').item()
    results["wer"]= esc.metrics(output_text,real_text,metric='wer').item()
    
    return results


def main():
    esc = Escaner()

    # image_path = 'images/textos/desk.JPG'
    # image_path = 'images/texto1.jpg'
    # image_path = 'images/dataset/sampleDatasetC2/input_sample/00001.jpg'
    image_path = 'images/dataset/sampleDatasetC2/input_sample/00099.jpg'

    # output_text_path = 'images/dataset/sampleDatasetC2/input_sample_groundtruth/00099.txt'

    prueba1_text_path = 'images/esp/Prueba-Español.txt'
    prueba2_text_path = 'images/esp/Prueba-Español2.txt'

    real_text = txt2string(prueba1_text_path,verbose=False)


    processed_image = preprocess_image(image_path,morph=False,esc=esc)
    fixed_image = processed_image


    original_image = cv2.imread(image_path)

    timg = cv2.transpose(original_image.copy())
    original_image = np.flip(timg,[1])

    original_image = cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB) 
    fixed_image = cv2.cvtColor(fixed_image,cv2.COLOR_GRAY2RGB) 


    image_paths = [
        "Prueba-español1-1.jpg",
        "Prueba-español1-2.jpg",
        "Prueba-español1-3.jpg",
    ]

    image_list = [original_image, fixed_image]
    # runs = [
    #     ['tesseract', image_list[0],"Original"],
    #     ['tesseract', image_list[1],"Processed"],
    #     # ['doctr', image_list[0],"Original"],
    #     # ['doctr', image_list[1],"Processed"],
    #     # ['easyocr', image_list[0],"Original"],
    #     # ['easyocr', image_list[1],"Processed"],
    # ]


    print("Leyendo imagenes con OCR...")
    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        futures = [executor.submit(run_ocr,run[0],run[1],esc,idx,run[2]) for idx,run in enumerate(runs)]
        concurrent.futures.wait(futures)


    data = [future.result() for future in futures]
    end = time.time()

    # Guardar la salida en un archivo
    with open('ocr_output_original.txt','w') as f:
        f.write(data[0])

    with open('ocr_output_processed.txt','w') as f:
        f.write(data[1])
        
    print(f"Tiempo de procesamiento de OCRs: {end-start} s")


    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        futures = [executor.submit(get_metrics,text,real_text,esc) for text in data]
        concurrent.futures.wait(futures)

    results = [future.result() for future in futures]
    end = time.time()
    print(f"Tiempo de calculo de metricas: {end-start} s")

    for i,result in enumerate(results):
        print(f"CER - {runs[i][0]} - {runs[i][2]}: {100.0*result['cer']:.2f}%")
        # print(f"WER - {runs[i][0]} - {runs[i][2]}: {100.0*result['wer']:.2f}%")

    
    # original_image = cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB) 
    # fixed_image = cv2.cvtColor(fixed_image,cv2.COLOR_BGR2RGB) 
    cv2.imwrite('processed.png',fixed_image)

    plt.figure()
    plt.subplot(121)
    plt.imshow(original_image)
    plt.axis('off')  # Hide the axis

    plt.subplot(122)
    plt.imshow(fixed_image, cmap='gray')
    plt.axis('off')  # Hide the axis
    plt.show()


    # orig_text_tess = esc.ocr_read(original_image,ocr='tesseract')
    # proc_text_tess = esc.ocr_read(fixed_image,ocr='tesseract')
    # output_text_easyocr = esc.ocr_read(fixed_image,ocr='easyocr')
    #
    # cer_tess = esc.metrics(output_text_tes,real_text)
    # cer_easyocr = esc.metrics(output_text_easyocr,real_text)
    #
    # print("EasyOCR: ", output_text_easyocr)
    # print(f"CER Tesseract: {100.0*cer_tess.item()}%")
    # print(f"CER EasyOCR:  {100.0*cer_easyocr.item()}%")
    #
    #

def txt2string(txt_path,verbose=False):
    if verbose:
        print("Leyendo salida real desde archivo...")

    real_text = ''
    with open(txt_path) as f:
        real_text = f.read()

    return real_text

def preprocess_image(path,morph,esc):

    print("\nPreprocesando imagen...")
    start = time.time()
    fixed_image = esc.show_scaned_image(path,morph=morph)

    end = time.time()
    print(f"Tiempo de preprocesamiento de la imagen: {end-start} s")

    return fixed_image


if __name__ == '__main__':
    main()
