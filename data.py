#Loads data
from typing import *

import torch
import numpy as np
import os
from multiprocessing import pool

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import Subset, DataLoader

from const import *
from log_cfg import logger

def load_partial_dataset() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads the dataset partially

    Returns:
        train_loader (DataLoader): The training data loader
        test_loader (DataLoader): The testing data loader
        val_loader (DataLoader): The validation data loader
    """
    # Create a transform to convert the images to tensors
    transform = ToTensor()

    # Load the full training set & reduce it
    logger.info(f"Loading full training set")
    full_train_dataset = ImageFolder(os.path.join(DATASET_PATH, 'images/train'), transform=transform)
    reduced_train_dataset = reduce_dataset(full_train_dataset, SAMPLES)

    # Load the testing set
    test_dataset = ImageFolder(os.path.join(DATASET_PATH, 'images/test'), transform=transform)

    # Load the validation set
    val_dataset = ImageFolder(os.path.join(DATASET_PATH, 'images/val'), transform=transform)

    # Create a data loaders
    train_loader = DataLoader(reduced_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    # Empty data loaders for now
    test_loader = None
    val_loader = None

    return train_loader, test_loader, val_loader

def load_full_dataset() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads the dataset fully

    Returns:
        train_loader (DataLoader): The training data loader
        test_loader (DataLoader): The testing data loader
        val_loader (DataLoader): The validation data loader
    """
    pass


# function to reduce the dataset size
def reduce_dataset(dataset: torch.utils.data.Dataset, samples_per_class: int) -> torch.utils.data.Dataset:
    """
    Reduces the dataset size while maintaining class proportionality

    Args:
        dataset (torch.utils.data.Dataset): The dataset to reduce
        samples_per_class (int): The number of samples to keep per class

    Returns:
        reduced_dataset (torch.utils.data.Dataset): The reduced dataset
    """
    # Initialize a dictionary to count the number of samples per class
    class_counts = {}
    
    # Create a list of indices for each class
    class_indices = [[] for _ in range(len(dataset.classes))]

    # Iterate through the dataset to collect class indices
    for idx, (image, label) in enumerate(dataset):
        class_counts[label] = class_counts.get(label, 0) + 1
        class_indices[label].append(idx)
        # only print if the index is a multiple of 5000
        if idx % 5000 == 0:
            logger.info(f"Progress: {idx}/{len(dataset)}")

    # Initialize a list to store the selected indices
    selected_indices = []

    # Iterate through class indices and select samples_per_class for each class
    for indices in class_indices:
        selected_indices.extend(indices[:samples_per_class])

    # Create a new Subset with the selected indices
    reduced_dataset = Subset(dataset, selected_indices)

    return reduced_dataset

"""
The images are at `DATASET_PATH/images` containing a 1081 classes (directories) with images contained.
`DATASET_PATH/plantnet300K_metadata.json` contains the metadata of the images.
"""

# Test
if __name__ == '__main__':
    train_loader, test_loader, val_loader = load_partial_dataset()
    logger.debug(train_loader)
    logger.debug(test_loader)
    logger.debug(val_loader)