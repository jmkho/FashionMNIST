import torch
import torchvision 
from torchvision import datasets 
from torchvision import models 
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import nn

import matplotlib.pyplot as plt
from timeit import default_timer as timer 
from tqdm.auto import tqdm
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

# 3.2. time counting forward pass on GPU and CPU
def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Training time on {device}: {total_time:.3f} seconds")
    return total_time

# 3.3. training loop and train a model on batches of data
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader,
          loss: torch.nn.Module):
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)
    train_time_start_on_cpu = timer()
    epochs = 3 

    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n------------------------")

        train_loss = 0
        model.train()
        for batch, (x, y) in enumerate(train_dataloader):
            y_pred = model_0(x)
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
                test_pred = model(x)
                test_loss += loss_fn(test_pred, y)
                test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)

        print(f"Train loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.5f}")

    train_time_end_on_cpu = timer()
    total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu, end=train_time_end_on_cpu, device=str(next(model.parameters()).device))

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

def main(model_mode):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input(device)
    # v0
    # model = FashionMNISTModelv0(input_shape=784, hidden_units=10, output_shape=len(class_names))
    # v1
    model = FashionMNISTModelv1(input_shape=784, hidden_units=10, output_shape=len(class_names))
    model.to(device)

    # 3.1. Setup optimizer and loss function
    loss_fn = nn.CrossEntropyLoss()

    if model_mode == 'train':
        train(model=model, 
              train_dataloader=train_dataloader, 
              test_dataloader=test_dataloader,
              loss=loss_fn)
    
    elif model_mode == 'test':
        test = eval_model(model=model, 
                          data_loader=test_dataloader,
                          loss_fn=loss_fn,
                          accuracy_fn=accuracy_fn)
        
        print(test)


if __name__ == '__main__':
    main('test')