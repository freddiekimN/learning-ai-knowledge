
import torch
from torchsummary import summary as summary
import subprocess
import os

os.chdir('./yolov5/')

cmd = "python val.py --weights yolov5s.pt --data coco128.yaml --img 640"
print(cmd.split(' '))
subprocess.run(cmd.split(' '))

