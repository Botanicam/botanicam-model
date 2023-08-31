#Loads data
from typing import *

import torch
import numpy as np
from torch.utils.data import DataLoader

from const import *

def load_partial_dataset() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads the dataset partially

    Returns:
        train_loader (DataLoader): The training data loader
        test_loader (DataLoader): The testing data loader
        val_loader (DataLoader): The validation data loader
    """
    pass

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
def reduce_dataset(dataset: torch.utils.data.Dataset, samples: int) -> torch.utils.data.Dataset:
    """
    Reduces the dataset size
    As the dataset is already split into train, test and validation sets, we can just reduce the train set


    Args:
        dataset (torch.utils.data.Dataset): The dataset to reduce
        samples (int): The number of samples to reduce to

    Returns:
        dataset (torch.utils.data.Dataset): The reduced dataset
    """
    # First, get the class weights
    class_weights = get_class_weights(dataset)


# function to get the class weights
def get_class_weights(dataset: torch.utils.data.Dataset) -> torch.Tensor:
    """
    Gets the class weights

    Args:
        dataset (torch.utils.data.Dataset): The dataset to get the class weights from

    Returns:
        class_weights (torch.Tensor): The class weights
    """
    pass

"""
The images are at `DATASET_PATH/images` containing a 1081 classes (directories) with images contained.
`DATASET_PATH/plantnet300K_metadata.json` contains the metadata of the images.
"""