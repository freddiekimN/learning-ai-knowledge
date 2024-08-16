import cv2
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import pickle
import numpy as np

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


#결과값 읽기 
with open('results.pkl', 'rb') as f:
    results = pickle.load(f)
    images = pickle.load(f)

import os

width = 1600
height = 800

video = cv2.VideoWriter('image_total_wPred.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

enable_plot = False    
plt.figure(figsize=(30, 15))    
for image_bgr, predictions in zip(images,results):

    plt.clf()  # 현재 화면을 지우고 갱신할 수 있도록 합니다.
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # BGR을 RGB로 변환
    # image_rgb = image_bgr[..., ::-1]
    plt.imshow(image_rgb)
    plt.axis('off')

    # 결과 값 출력  
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):  
        if score > score_threshold and label in target_classes:
            x1, y1, x2, y2 = box
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='red', facecolor='none', linewidth=2))
            plt.text(x1, y1, f'{score:.2f}:{coco_classes[label]}', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))

    plt.pause(0.1)
    plt.draw()  # 현재의 플롯을 그립니다.
    
    image = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))

    resized_frame_rgb = cv2.resize(image, (width, height))
    resized_frame_bgr = resized_frame_rgb[..., ::-1]
    video.write(resized_frame_bgr)
    

# 동영상 작성 종료
video.release()    