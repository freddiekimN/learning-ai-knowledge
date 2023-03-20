
import torch
from torchsummary import summary as summary
import subprocess
import os

os.chdir('./yolov5/')

# img = "https://ultralytics.com/images/zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list
# # img = "https://www.youtube.com/watch?v=la7J3PLAXGg"
# # img = "Apgujeong-dong_Night_Street_in_Korea.mp4"
# cmd = f"python3 detect.py --weight yolov5s.pt --source {img}"
# print(cmd.split(' '))
# subprocess.run(cmd.split(' '))

# img = "/home/joo/Workspace/study/learning-ai-knowledge/yolov5/data/datasets/coco128/images/train2017/000000000061.jpg"
# cmd = f"python3 detect.py --weight yolov5s.pt --source {img}"
# print(cmd.split(' '))
# subprocess.run(cmd.split(' '))


# cmd = "python val.py --weights yolov5s.pt --data coco128.yaml --img 640 --task study"
import yolov5.val

cmd = "python train.py --data coco128.yaml --weights yolov5s.pt --img 640"
print(cmd.split(' '))
subprocess.run(cmd.split(' '))