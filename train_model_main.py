from visualize_model import visualize_model
import torchvision
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from train_model import train_model

def train_model_main(model_conv,use_grad,use_gpu,dataloaders,class_names,dataset_sizes):
    # model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = use_grad
        
    # print('before modification')
    # summary(model_conv,(3, 244 , 244 ))    

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    if use_gpu:
        model_conv = model_conv.cuda()

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opoosed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    model_conv = train_model(model_conv, criterion, optimizer_conv,
                            exp_lr_scheduler, dataloaders,use_gpu, dataset_sizes, num_epochs=25)

    visualize_model(model_conv,dataloaders,use_gpu,class_names)
    
    return model_conv
    
    
