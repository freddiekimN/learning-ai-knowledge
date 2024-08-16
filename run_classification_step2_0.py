
import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms
import matplotlib.pyplot as plt

# 사전 학습된 Faster R-CNN 모델 로드
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 이미지 전처리
transform = transforms.Compose([transforms.ToTensor()])

# 동영상 파일 열기
cap = cv2.VideoCapture('image_total.mp4')

# 총 프레임 수 얻기
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total number of frames: {total_frames}")

# 프레임 속도(FPS) 얻기
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Frame Rate (FPS): {fps}")

results = []
images = []
frame_count = 0
while True:
    ret, image = cap.read()

    # 더 이상 프레임이 없을 경우 종료
    if not ret:
        break

    if frame_count % 10 == 0:

        # Transform 정의 (PIL Image -> Tensor)
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            predictions = model(image_tensor)
            images.append(image)
            results.append(predictions[0])

    frame_count += 1
    
#결과값 저장
import pickle
with open('results.pkl', 'wb') as f:
    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(images, f, protocol=pickle.HIGHEST_PROTOCOL)
    
