from typing import *

import torch
import os

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor
from torch.utils.data import Subset, DataLoader
from torchvision.transforms import Resize

from const import *
from log_cfg import logger

def load_dataset(partial : bool = PARTIAL_LOAD) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads the dataset partially

    Returns:
        train_loader (DataLoader): The training data loader
        test_loader (DataLoader): The testing data loader
        val_loader (DataLoader): The validation data loader
    """
    # Create a transform to convert the images to tensors
    transform = transforms.Compose([
        Resize((224, 224)),
        ToTensor()
    ])

    # Define the root directory of the dataset
    # DATASET CONST
    dataset_root = DATASET_PATH

    
    # Load the full training set
    logger.info(f"Loading full training set")
    full_train_dataset = ImageFolder(os.path.join(dataset_root, 'images/train'), transform=transform)

    
    # Load the full testing set
    logger.info(f"Loading full testing set")
    full_test_dataset = ImageFolder(os.path.join(dataset_root, 'images/test'), transform=transform)

    # Load the full validation set
    logger.info(f"Loading full validation set")
    full_val_dataset = ImageFolder(os.path.join(dataset_root, 'images/val'), transform=transform)

    # Reduce the datasets if partial is True
    if partial:
        reduced_train_dataset = reduce_dataset(full_train_dataset, SAMPLES)
        reduced_val_dataset = reduce_dataset(full_val_dataset, SAMPLES)
    else:
        reduced_train_dataset = full_train_dataset
        reduced_val_dataset = full_val_dataset

    # Create data loaders
    train_loader = DataLoader(reduced_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(full_test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(reduced_val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

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
    train_indices, _ = torch.utils.data.random_split(range(len(dataset)), [samples_per_class, len(dataset) - samples_per_class])

    # create a subset of the dataset
    reduced_dataset = Subset(dataset, train_indices)

    return reduced_dataset

"""
The images are at `DATASET_PATH/images` containing a 1081 classes (directories) with images contained.
`DATASET_PATH/plantnet300K_metadata.json` contains the metadata of the images.
"""
# Test
if __name__ == '__main__':
    from tqdm import tqdm
    import shutil

    train_loader, test_loader, val_loader = load_dataset(partial=PARTIAL_LOAD)

    # serialize the data loaders
    torch.save(train_loader, 'train_loader.pth')
    torch.save(test_loader, 'test_loader.pth')
    torch.save(val_loader, 'val_loader.pth')

    # Check basic statistics about the datasets
    num_classes = NUM_CLASSES
    num_train_samples = len(train_loader.dataset)
    num_test_samples = len(test_loader.dataset)
    num_val_samples = len(val_loader.dataset)
    
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Number of training samples: {num_train_samples}")
    logger.info(f"Number of testing samples: {num_test_samples}")
    logger.info(f"Number of validation samples: {num_val_samples}")
    
    # Iterate through a few batches to check data loader behavior
    for batch_idx, (data, targets) in enumerate(train_loader):
       #  if batch_idx < 2:  # Print information for the first 2 batches
        logger.info(f"Batch {batch_idx}:")
        logger.info(f"  Data shape: {data.shape}")
        logger.info(f"  Targets shape: {targets.shape}")
        
    # Visualize a few samples (you can use matplotlib or another library for this)
    import matplotlib.pyplot as plt
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        try:
            # if batch_idx < 2:  # Visualize the first 2 batches
            for i in range(data.size(0)):  # Visualize individual samples in the batch
                sample_image = data[i].permute(1, 2, 0)  # Rearrange channels for visualization
                plt.imshow(sample_image)
                plt.title(f"Class: {train_loader.dataset.classes[targets[i]]}")
                plt.show()
        except Exception as e:
            logger.error(e)
            break