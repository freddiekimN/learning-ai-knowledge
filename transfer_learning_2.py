import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import Kitti
import torchvision

# Kitti 데이터셋을 불러올 때 적용할 전처리 함수입니다.
# 이 예제에서는 이미지를 Resize하고 Normalize합니다.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Kitti 데이터셋을 불러옵니다.
# 이 예제에서는 train set만 사용합니다.
kitti_train = Kitti(root='../data/', train=True, transform=transform)

# DataLoader를 사용하여 배치 단위로 데이터를 로드합니다.
train_loader = DataLoader(kitti_train, batch_size=1, shuffle=True)

# ResNet-18 모델을 불러옵니다.
# 이 모델은 imagenet 데이터셋으로 사전 학습된 모델입니다.
model = torchvision.models.resnet18(pretrained=True)
# model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)

# ResNet-18 모델의 마지막 레이어를 새로운 작업에 맞게 변경합니다.
# 이 예제에서는 Kitti 데이터셋에 있는 클래스(자동차와 보행자)를 구분하는 이진 분류 문제를 풉니다.
# 따라서 마지막 레이어의 출력 크기를 2로 변경합니다.
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# 모델의 파라미터를 GPU로 옮깁니다.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 손실 함수와 옵티마이저를 정의합니다.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    running_loss = 0.0
    for ndex, (images, labels) in enumerate(train_loader):
        # inputs, labels = images
        # inputs, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        target_tensor=torch.as_tensor([1,1]).long()

        outputs = model(images)
        loss = criterion(outputs, target_tensor)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print('[Epoch %d] loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))
