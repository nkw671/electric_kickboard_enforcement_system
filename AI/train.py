from torch.cuda import device
from ultralytics import YOLO
def main():





    model = YOLO("yolo26s.pt")  # 기존 학습 모델에서 파인튜닝

    model.train(
        data=r'C:\Projects\Kickboard_project\AI\new_train\moreData\data.yaml',
        epochs=100,
        imgsz=800,
        device=0,
        workers=0,
        optimizer='SGD',
        lr0=0.01,       # 파인튜닝은 낮은 학습률로 기존 가중치 보존
        weight_decay=0.001,
        cls=1.2,
        box=8.5,
        cos_lr=True
    )

    # 경로 앞에 r을 붙여서 윈도우 역슬래시(\) 경로 오작동을 방지합니다.
    #model = YOLO(r"C:\ai\stable-diffusion-webui\runs\detect\train27\weights\last.pt")

    # resume=True를 주면 train27 환경 그대로 이어서 학습이 시작됩니다.

    #model.train(resume=True)
if __name__ == '__main__':
    main()