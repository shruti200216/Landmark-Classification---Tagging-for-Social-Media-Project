import torch
import torch.nn as nn

# Define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
        super().__init__()
        
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),  # Example values, adjust as needed
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Add more convolutional layers as needed
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # Adjust the output size if needed
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(16 * 7 * 7, num_classes),  # Adjust the input size if needed
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process the input tensor through the
        # feature extractor, the pooling, and the final linear
        # layers (if appropriate for the architecture chosen)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
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