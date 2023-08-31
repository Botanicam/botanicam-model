#Loads data
from typing import *

import torch
import numpy as np
import os
import random

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import Subset, DataLoader
from torch

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
    transform = ToTensor()

    # Load full training set
    full_train_dataset = ImageFolder(os.path.join(dataset_path, 'images/train'), transform=transform)

    # Get class weights 
    ### : ifaz go make get_class_weights()! ###
    #class_weights = get_class_weights(full_train_dataset)

    # Create a sampler to select samples based on class weights
    sampler = torch.utils.data.sampler.WeightedRandomSampler(class_weights, samples, replacement=False)

    # Create a subset of the training dataset using the sampler
    reduced_train_dataset = Subset(full_train_dataset, sampler=sampler)

    return reduced_train_dataset


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
