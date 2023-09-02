#Loads data
from typing import *

import torch
import numpy as np
import os
import multiprocessing as mp

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split

from const import *
from log_cfg import logger

def load_dataset(partial=True) -> Tuple[DataLoader, DataLoader, DataLoader]:
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
    if partial:
        reduced_train_dataset = reduce_dataset(full_train_dataset, SAMPLES)
    else:
        reduced_train_dataset = full_train_dataset

    # Load the testing set
    logger.info(f"Loading testing set")
    test_dataset = ImageFolder(os.path.join(DATASET_PATH, 'images/test'), transform=transform)

    # Load the validation set
    logger.info(f"Loading validation set")
    val_dataset = ImageFolder(os.path.join(DATASET_PATH, 'images/val'), transform=transform)

    # Create a data loaders
    train_loader = DataLoader(reduced_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    return train_loader, test_loader, val_loader


# function to reduce the dataset size
def reduce_dataset(dataset: torch.utils.data.Dataset, samples_per_class: int) -> torch.utils.data.Subset:
    """
    Reduces the dataset size to `samples_per_class` samples per class

    Args:
        dataset (torch.utils.data.Dataset): The dataset to reduce
        samples_per_class (int): The number of samples per class

    Returns:
        reduced_dataset (torch.utils.data.Subset): The reduced dataset
    """
    logger.info(f"Reducing dataset to {samples_per_class} samples per class")
    
    # stratified sampling
    train_indices, _ = train_test_split(range(len(dataset)), train_size=samples_per_class, stratify=dataset.targets)

    # create a subset of the dataset
    reduced_dataset = Subset(dataset, train_indices)

    return reduced_dataset

"""
The images are at `DATASET_PATH/images` containing a 1081 classes (directories) with images contained.
`DATASET_PATH/plantnet300K_metadata.json` contains the metadata of the images.
"""

# Test
if __name__ == '__main__':
    train_loader, test_loader, val_loader = load_dataset()
    logger.debug(train_loader)
    logger.debug(test_loader)
    logger.debug(val_loader)