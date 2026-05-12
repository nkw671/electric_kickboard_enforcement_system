from ultralytics import YOLO
def main():

    model = YOLO('../temp/yolo26n.pt')


    model.train(
        data='/kickboard.v1-test1.yolo26/data.yaml',
        epochs=200,
        imgsz=640,
        device=0,
        workers=0
    )

if __name__ == '__main__':
    main()