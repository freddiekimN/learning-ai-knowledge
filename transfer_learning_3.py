# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
from torchsummary import summary as summary
from imshow import imshow
from visualize_model import visualize_model
from train_model import train_model
from train_model_main import train_model_main

plt.ion()   # interactive mode

# 학습을 위한 데이터 증가(Augmentation)와 일반화하기
# 단지 검증을 위한 일반화하기
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '../data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

# 모델 학습하기
model_ft = models.resnet18(pretrained=True)
model_ft = train_model_main(model_ft,True,use_gpu,dataloaders,class_names,dataset_sizes)

model_conv = models.resnet18(pretrained=True)
model_conv = train_model_main(model_conv,False,use_gpu,dataloaders,class_names,dataset_sizes)

# model_152_ft = models.resnet152(pretrained=True)
# model_152_ft = train_model_main(model_152_ft,True,use_gpu,dataloaders,class_names,dataset_sizes)

# model_152_conv = models.resnet152(pretrained=True)
# model_152_conv = train_model_main(model_152_conv,False,use_gpu,dataloaders,class_names,dataset_sizes)

summary(model_conv,(3,244,244))
# summary(model_152_conv,(3,244,244))