from torch.cuda import device
from ultralytics import YOLO
def main():





    model = YOLO('../temp/yolo26n.pt')

    model.train(
        data=r'C:\Projects\Kickboard_project\AI\new_train\kickboard+helmet+background_v2-1\data.yaml',
        epochs=200,
        imgsz=800,
        device=0,
        workers=0,
        optimizer='SGD',
        lr0=0.001,
        weight_decay=0.001,
        # 아래 인자들은 공식 문서에 명시된 하이퍼파라미터입니다.
        cls=1.2,  # 분류(헬멧 유무)에 더 집중
        dfl=2.0,  # 박스 경계 정교화 강화
        box=8.5,  # 위치 예측 가중치 상향
        cos_lr=True  # 학습률 부드럽게 감소
    )

    # 경로 앞에 r을 붙여서 윈도우 역슬래시(\) 경로 오작동을 방지합니다.
    #model = YOLO(r"C:\ai\stable-diffusion-webui\runs\detect\train27\weights\last.pt")

    # resume=True를 주면 train27 환경 그대로 이어서 학습이 시작됩니다.

    #model.train(resume=True)
if __name__ == '__main__':
    main()