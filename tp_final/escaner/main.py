from core.escaner import Escaner

def main():
    video_path = 'images/dataset/sampleDatasetC1/input_sample/background00/letter001.avi'
    video_path = 'images/dataset/sampleDatasetC1/input_sample/background00/magazine001.avi'
    esc = Escaner()
    # esc.load_video(video_path,'output.avi')

    image_path = 'images/textos/desk.JPG'
    image_path = 'images/texto1.jpg'
    # image_path = 'images/dataset/sampleDatasetC2/input_sample/00001.jpg'

    text = esc.show_scaned_image(image_path)

    print(text)

if __name__ == '__main__':
    main()
