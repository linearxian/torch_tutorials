import torch
from torch import nn
from torch._C import device
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# 1 - download dataset
# 2 - create data loader
# 3 - build model
# 4 - train
# 5 - save trained model



BATCH_SIZE = 128



class FeedForwardNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        self.softmax = nn.Softmax(dim=1)


    def forward(self, input_data):
        flattend_data = self.flatten(input_data)
        logits = self.dense_layers(flattend_data)
        predictions = self.softmax(logits)
        return predictions


def download_mnist_datasets():
    train_data = datasets.MNIST(
        root="data",
        download=True,
        train=True,
        transform=ToTensor()
    )
    validation_data = datasets.MNIST(
        root="data",
        download=True,
        train=False,
        transform=ToTensor()
    )
    return train_data, validation_data


def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    pass


def train():
    pass


if __name__ == "__main__":
    # download MNIST Dataset
    train_data, _ = download_mnist_datasets()
    print("MNIST dataset downloaded")

    # create a data loader for the train set
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)

    # build model
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device")
    feed_forward_net = FeedForwardNet().to(device)