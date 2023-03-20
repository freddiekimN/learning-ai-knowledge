
import torch
from torchsummary import summary as summary
import subprocess
import os

os.chdir('./yolov5/')

cmd = "python train.py --data coco128.yaml --weights yolov5s.pt --img 640 --epochs 3"
print(cmd.split(' '))
subprocess.run(cmd.split(' '))

