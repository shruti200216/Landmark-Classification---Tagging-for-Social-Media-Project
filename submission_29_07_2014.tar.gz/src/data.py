# src/data.py

import pytest
import torch
from pathlib import Path
from src.data_utils import visualize_one_batch, get_data_loaders

# Your other imports and code remain the same...

######################################################################################
#                                     TESTS
######################################################################################

@pytest.fixture(scope="session")
def data_loaders():
    return get_data_loaders(batch_size=2, num_workers=0)

def test_data_loaders_keys(data_loaders):
    assert (
        set(data_loaders.keys()) == {"train", "valid", "test"}
    ), "The keys of the data_loaders dictionary should be train, valid and test"


def test_data_loaders_output_type(data_loaders):
    # Test the data loaders
    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    assert isinstance(images, torch.Tensor), "images should be a Tensor"
    assert isinstance(labels, torch.Tensor), "labels should be a Tensor"
    assert (
        images[0].shape[-1] == 224
    ), "The tensors returned by your dataloaders should be 224x224. Did you forget to resize and/or crop?"


def test_data_loaders_output_shape(data_loaders):
    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    assert (
        len(images) == 2
    ), f"Expected a batch of size 2, got size {len(images)}"
    assert (
        len(labels) == 2
    ), f"Expected a labels tensor of size 2, got size {len(labels)}"


def test_visualize_one_batch(data_loaders):
    # Create a train_data object using the first data loader
    train_data = data_loaders["train"].dataset

    # Call visualize_one_batch with train_data
    visualize_one_batch(data_loaders, train_data, max_n=2)