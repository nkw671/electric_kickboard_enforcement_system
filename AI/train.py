from ultralytics import YOLO
def main():

    model = YOLO('../temp/yolo26n.pt')


    model.train(
        data='C:/Kickboard_project/training_data/data.yaml',
        epochs=50,
        imgsz=640,
        device=0,
        workers=0
    )

if __name__ == '__main__':
    main()