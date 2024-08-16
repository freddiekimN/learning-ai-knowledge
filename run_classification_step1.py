
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

image = cv2.imread('image_23.jpg')

# Transform 정의 (PIL Image -> Tensor)
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    predictions = model(image_tensor)
    
#결과값 저장
import pickle
with open('results.pkl', 'wb') as f:
	pickle.dump(predictions[0], f, protocol=pickle.HIGHEST_PROTOCOL)	

#결과값 읽기 
with open('results.pkl', 'rb') as f:
	predictions = pickle.load(f)
    
    
plt.figure(figsize=(30, 15))    
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.axis('off')


score_threshold=0.5
coco_classes = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]
target_classes = [0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17]

# 결과 값 출력  
for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):  
    if score > score_threshold:
        x1, y1, x2, y2 = box
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='red', facecolor='none', linewidth=2))
        plt.text(x1, y1, f'{score:.2f}:{coco_classes[label]}', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))

plt.show()

    
# # plt.cla()
# # # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# score_threshold=0.5

# for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
#     # if score > score_threshold and label in target_classes:
#     if score > score_threshold:
#         x1, y1, x2, y2 = box
        
#         # 사각형의 네 변을 그리기 위해 좌표 설정
#         x_values = [x1, x2, x2, x1, x1]
#         y_values = [y1, y1, y2, y2, y1]

#         # plt.plot을 사용하여 사각형 그리기
#         # plt.plot(x_values, y_values, color='red', linewidth=2)
        
#         plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='red', facecolor='none', linewidth=2))
            
#         # plt.text(x1, y1, f'{score:.3f}', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
#         # 라벨 숫자 그리기
#         plt.text(x1, y1, f'{score:.3f}:{coco_classes[label]}', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
        

# plt.pause(1)  # Pause to update the plot    
# plt.show()