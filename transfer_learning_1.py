import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import Kitti
from torchvision.transforms import transforms


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = Kitti(root='../data', transform=train_transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


model = torchvision.models.resnet18(pretrained=True)

# ResNet18의 마지막 FC 레이어를 Kitti 데이터셋에 맞게 수정합니다.
# ResNet-18 모델의 마지막 레이어를 새로운 작업에 맞게 변경합니다.
# 이 예제에서는 Kitti 데이터셋에 있는 클래스(자동차와 보행자)를 구분하는 이진 분류 문제를 풉니다.
# 따라서 마지막 레이어의 출력 크기를 2로 변경합니다.
model.fc = nn.Linear(in_features=512, out_features=2)

# GPU를 사용할 경우
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print('[Epoch %d] loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))
