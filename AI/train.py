from ultralytics import YOLO
def main():

    model = YOLO('../temp/yolo26n.pt')


    model.train(
        data='C:\Projects\Kickboard_project\AI\new_train\kickboard-detection(add-more-helmet-data)-1/data.yaml',
        epochs=200,
        imgsz=800,
        device=0,
        workers=0
    )

if __name__ == '__main__':
    main()