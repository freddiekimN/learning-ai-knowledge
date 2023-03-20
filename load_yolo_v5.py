import torch
from torchsummary import summary as summary
# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

# Images
img = "https://ultralytics.com/images/zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
# results.print()  # or .show(), .save(), .crop(), .pandas(), etc.

results.show()


