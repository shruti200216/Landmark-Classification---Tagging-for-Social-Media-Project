# src/data_utils.py

import torch
import torch.utils.data
from torchvision import datasets, transforms
from pathlib import Path
import math
import matplotlib.pyplot as plt

from .helpers import compute_mean_and_std, get_data_location

def get_data_loaders(
    batch_size: int = 32, valid_size: float = 0.2, num_workers: int = 1, limit: int = -1
):
    """
    Create and return the train, validation, and test data loaders.

    :param batch_size: size of the mini-batches
    :param valid_size: fraction of the dataset to use for validation (e.g., 0.2 for 20%)
    :param num_workers: number of workers to use in the data loaders
    :param limit: maximum number of data points to consider
    :return: a dictionary with keys 'train', 'valid', and 'test' containing the respective data loaders
    """
    data_loaders = {"train": None, "valid": None, "test": None}
    base_path = Path(get_data_location())
    
    # Compute mean and std of the dataset
    mean, std = compute_mean_and_std()
    print(f"Dataset mean: {mean}, std: {std}")

    # Define data transforms
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        "valid": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
    }

    # Create train and validation datasets
    train_data = datasets.ImageFolder(base_path / "train", transform=data_transforms["train"])
    valid_data = datasets.ImageFolder(base_path / "train", transform=data_transforms["valid"])

    # Obtain training and validation indices
    n_tot = len(train_data)
    indices = torch.randperm(n_tot)
    if limit > 0:
        indices = indices[:limit]
        n_tot = limit
    split = int(math.ceil(valid_size * n_tot))
    train_idx, valid_idx = indices[split:], indices[:split]

    # Define samplers for training and validation
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

    # Prepare data loaders
    data_loaders["train"] = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
    )
    data_loaders["valid"] = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers
    )

    # Create the test data loader
    test_data = datasets.ImageFolder(base_path / "test", transform=data_transforms["test"])
    data_loaders["test"] = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return data_loaders

def visualize_one_batch(data_loaders, train_data, max_n: int = 5):
    """
    Visualize one batch of data.

    :param data_loaders: dictionary containing data loaders
    :param train_data: training dataset
    :param max_n: maximum number of images to show
    """
    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    # Undo normalization for visualization
    mean, std = compute_mean_and_std()
    invTrans = transforms.Compose([
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / s for s in std]),
        transforms.Normalize(mean=[-m for m in mean], std=[1.0, 1.0, 1.0]),
    ])
    images = invTrans(images)

    class_names = train_data.classes

    images = torch.permute(images, (0, 2, 3, 1)).clip(0, 1)

    fig = plt.figure(figsize=(25, 4))
    for idx in range(max_n):
        ax = fig.add_subplot(1, max_n, idx + 1, xticks=[], yticks=[])
        ax.imshow(images[idx])
        ax.set_title(class_names[labels[idx].item()])