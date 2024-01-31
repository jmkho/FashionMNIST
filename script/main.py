import torch
import torchvision 
from torchvision import datasets 
from torchvision import models 
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import nn

import torch.nn.functional as F
from torch.nn import init

import matplotlib.pyplot as plt
import os
from timeit import default_timer as timer 
from tqdm.auto import tqdm
from pathlib import Path
from torchinfo import summary
torch.manual_seed(42)

from helper_functions import accuracy_fn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Load Dataset
train_dataset = datasets.FashionMNIST(root='data', download=True, train=True, transform=ToTensor(), target_transform=None)
test_dataset = datasets.FashionMNIST(root='data', download=True, train=False, transform=ToTensor())

class_names = train_dataset.classes

# 2. Create Dataloader 
BATCH_SIZE = 32
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_features_batch, train_labels_batch = next(iter(train_dataloader))

# @@. Transform data to get the same input output of the model


# 3. Build a Baseline Model
# consists of two nn.Linear() layers. Because of image data, we'll need to start with nn.Flatten() -> compresses the dimensions of a tensor into a single vector.
# nn modules function as a model (can forward pass)

class FashionMNISTModelv0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )
    
    def forward(self, x):
        return self.layer_stack(x)

# 6. Building a Non-Linear Model
# adding nn.Relu() for non-linearity
class FashionMNISTModelv1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)

# 7. Building another CNN model
# utilizing nn.Conv2d, nn.MaxPool2d

# MobileNetV3 - Large 
class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out 

class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class Block(nn.Module):
    # expand + depthwise + pointwise 
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear

        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear

        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()

        if stride == 1 and in_size != out_size: 
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size)
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.se != None:
            out = self.se(out)

        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV3_Large(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3_Large, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(3, 40, 240, 80, hswish(), None, 2),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 480, 112, hswish(), SeModule(112), 1),
            Block(3, 112, 672, 112, hswish(), SeModule(112), 1),
            Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
            Block(5, 160, 672, 160, hswish(), SeModule(160), 2),
            Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
        )

        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = hswish()

        self.linear3 = nn.Linear(960, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()

        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None: 
                    init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        # out = F.avg_pool2d(out, 7)    # before this pooling the image has a 1x1 size already. So when it got pooled, the image reduces to 0x0.
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out

# MobileNetV3 - Large 

# 3.2. time counting forward pass on GPU and CPU
def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Training time on {device}: {total_time:.3f} seconds")
    return total_time

# 3.3. training loop and train a model on batches of data
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          device: torch.device = device):
    train_time_start = timer()

    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)
    epochs = 5

    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n------------------------")

        train_loss = 0
        model.train()
        for batch, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            train_loss += loss 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 400 == 0:
                print(f"Looked at {batch * len(x)} / {len(train_dataloader.dataset)} samples")
        
        train_loss /= len(train_dataloader)

        test_loss, test_acc = 0, 0
        model.eval()
        with torch.inference_mode():
            for x, y in test_dataloader: 
                x, y = x.to(device), y.to(device)
                test_pred = model(x)
                test_loss += loss_fn(test_pred, y)
                test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)

        print(f"Train loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.5f}")

    train_time_end = timer()
    print_train_time(start=train_time_start, end=train_time_end, device=str(next(model.parameters()).device))

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device: torch.device = device):
    """
    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """

    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for x, y in data_loader: 
            x, y = x.to(device), y.to(device)
            y_pred = model(x)

            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        
        loss /= len(data_loader)
        acc /= len(data_loader)
    
    return {"model_name": model.__class__.__name__,     #only works when model was created with a class 
            "model_loss": loss.item(),
            "model_acc": acc}

def main():
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input(device)
    # v0 - baseline 
    # model = FashionMNISTModelv0(input_shape=784, hidden_units=10, output_shape=len(class_names))
    # v1 - with nonlinearity 
    # model = FashionMNISTModelv1(input_shape=784, hidden_units=10, output_shape=len(class_names))
    # v2 - MobileNetV3 Large
    model = MobileNetV3_Large()
    model.to(device)

    # 3.1. Setup optimizer and loss function
    loss_fn = nn.CrossEntropyLoss()

    # If you need model summary 
    # summary(model=model,
    #         input_size=(32,1,16,16),
    #         col_names=['input_size', 'output_size', 'num_params', 'trainable'],
    #         col_width=20,
    #         row_settings=['var_names'])
    # input("Printed model summary")

    # if model_mode == 'train':
    train(model=model, 
          train_dataloader=train_dataloader, 
          test_dataloader=test_dataloader,
          loss_fn=loss_fn)

# elif model_mode == 'test':
    test = eval_model(model=model, 
                      data_loader=test_dataloader,
                      loss_fn=loss_fn,
                      accuracy_fn=accuracy_fn)
        
    print(test)

    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_NAME = 'FashionMNIST_MobileNetV3_model.pth'
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)


if __name__ == '__main__':
    # main('test')
    main()