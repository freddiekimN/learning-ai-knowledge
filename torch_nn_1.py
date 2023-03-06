from pathlib import Path
import requests
import wget

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)
FILENAME = "mnist.pkl.gz"


# 파일 다운 받기 
# URL = "https://github.com/pytorch/tutorials/tree/main/_static/"

# if not (PATH / FILENAME).exists():
#      wget.download(URL + FILENAME,out=PATH.as_posix())
# 나의 경우는 잘되지않음. 그래서 git clone https://github.com/pytorch/tutorials/ 으로 받은 다음 
# _static에서 가져옴.
        
### 2 section      
# 이 데이터셋은 numpy 배열 포맷이고, 데이터를 직렬화하기 위한 python 전용 포맷 pickle 을 이용하여 저장되어 있습니다   
import numpy as np
import pickle
import gzip
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
    
from matplotlib import pyplot
import numpy as np

# 각 이미지는 28 x 28 형태 이고, 784 (=28x28) 크기를 가진 하나의 행으로 저장되어 있습니다. 하나를 살펴 봅시다; 먼저 우리는 이 이미지를 2d로 재구성해야 합니다.
pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
print(x_train.shape)

print(y_train[0])
print(y_train.shape)
# PyTorch는 numpy 배열 보다는 torch.tensor 를 사용하므로, 우리는 데이터를 변환해야 합니다.    
import torch

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())

#Xavier initialisation 기법을 이용하여 가중치를 초기화 합니다. (1/sqrt(n)을 곱해주는 것을 통해서 초기화).
#28x28 = 784 임.
#0~9까지의 숫자는 10개의 클래스 이고 해당 클래스를 softmax 함수로 구분하려는 모델임.
#즉 
import math

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

def log_softmax(x):
    print(x)
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias)

bs = 64  # 배치 크기

xb = x_train[0:bs]  # x로부터 미니배치(mini-batch) 추출
preds = model(xb)  # 예측
preds[0], preds.shape
print(preds[0], preds.shape)

def nll(input, target):
    print(input[range(target.shape[0]), target])
    # 64
    # range(target.shape[0]) 의 의미는 0 ~ 63 임.
    # target의 의미는 0 ~ 63에 해당하는 정답 숫자임. 즉 index 를 의미함.
    return -input[range(target.shape[0]), target].mean()

loss_func = nll

yb = y_train[0:bs]
print(loss_func(preds, yb))

def accuracy(out, yb):
    print(out.shape)
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

print(accuracy(preds, yb))

lr = 0.5  # 학습률(learning rate)
epochs = 2  # 훈련에 사용할 에폭(epoch) 수

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        #         set_trace()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()