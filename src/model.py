import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        
        ## input : 3 x 224 x 224
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),  ## conv1: [3x224x224] -> [8x224x224]
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=2, stride=2))  ## maxpool: [8 x 224 x 224] -> [8 x 112 x 112]
        self.dropout = nn.Dropout(p=dropout)
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),  ## conv2: [8 x 112 x 112] -> [16 x 112 x 112]
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2))  ## maxpool: [16 x 112 x 112] -> [16 x 56 x 56]
        self.dropout = nn.Dropout(p=dropout)
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  ## conv2: [16 x 56 x 56] -> [32 x 56 x 56]
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2))  ## maxpool: [32 x 56 x 56] -> [32 x 28 x 28]
        self.dropout = nn.Dropout(p=dropout)
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  ## conv2: [32 x 28 x 28] -> [64 x 28 x 28]
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2))  ## maxpool: [64 x 28 x 28] -> [64 x 14 x 14]
        self.dropout = nn.Dropout(p=dropout)
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  ## conv2: [64 x 14 x 14] -> [128 x 14 x 14]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2))  ## maxpool: [128 x 14 x 14] -> [128 x 7 x 7]
        self.dropout = nn.Dropout(p=dropout)

        self.fc = nn.Sequential(
            nn.Flatten(),

            nn.Linear(128 * 7 * 7, 512, bias=True),  ## linear: [128 x 7 x 7] -> 512
            nn.ReLU(),
            nn.Dropout(p=dropout),
            
            nn.Linear(512, 256, bias=True),  ## linear: 512 -> 256
            nn.ReLU(),
            nn.Dropout(p=dropout),
            
            nn.Linear(256, 128, bias=True),  ## linear: 256 -> 128
            nn.ReLU(),
            nn.Dropout(p=dropout),
            
            nn.Linear(128, 64, bias=True),  ## linear: 128 -> 64
            nn.ReLU(),
            nn.Dropout(p=dropout),
            
            nn.Linear(64, num_classes)  ## classifier: 64 -> num_classes
#             nn.LogSoftmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        x = self.layer1(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.dropout(x)
        
        x = self.layer3(x)
        x = self.dropout(x)

        x = self.layer4(x)
        x = self.dropout(x)
        
        x = self.layer5(x)
        x = self.dropout(x)

        x = self.fc(x)
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
