from ultralytics import YOLO
def main():
    # 1. 모델 로드 (이미 가지고 계신 yolo26n.pt 사용)
    model = YOLO('yolo26n.pt')

    # 2. 학습 실행 (모든 코드는 이 if __name__ == '__main__': 블록 안에 있어야 합니다)
    model.train(
        data='C:/Kickboard_project/training_data/data.yaml',
        epochs=50,
        imgsz=640,
        device=0,      # 이제 GPU가 잘 잡히니 0으로 쓰시면 됩니다!
        workers=0      # 에러가 계속나면 workers를 0으로 설정해 보세요.
    )

if __name__ == '__main__':
    main()