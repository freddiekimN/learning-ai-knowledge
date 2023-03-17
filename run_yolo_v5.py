
import torch
from torchsummary import summary as summary
import subprocess
import os

os.chdir('./yolov5/')

img = "https://ultralytics.com/images/zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list
# img = "https://www.youtube.com/watch?v=la7J3PLAXGg"
# img = "Apgujeong-dong_Night_Street_in_Korea.mp4"
cmd = f"python3 detect.py --weight yolov5s.pt --source {img}"
print(cmd.split(' '))
subprocess.run(cmd.split(' '))