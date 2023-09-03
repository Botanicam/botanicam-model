from typing import *

import torch
import os

from sklearn.utils import check_random_state
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision.transforms import Resize

from const import *
from log_cfg import logger

def load_dataset(partial=True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads the dataset partially
    
    Args:
        partial (Bool): whether or not the dataset should be loaded fully or reduced

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

    # Load the full training set & reduce it
    logger.info(f"Loading full training set")
    # CHANGE ADDRESS
    full_train_dataset = ImageFolder(os.path.join(DATASET_PATH, 'images/train'), transform=transform)
    full_test_dataset = ImageFolder(os.path.join(DATASET_PATH, 'images/test'), transform=transform)
    full_val_dataset = ImageFolder(os.path.join(DATASET_PATH, 'images/val'), transform=transform)

    if partial:
        reduced_train_dataset = reduce_dataset(full_train_dataset, SAMPLES)
        reduced_test_dataset = reduce_dataset(full_test_dataset, SAMPLES)
        reduced_val_dataset = reduce_dataset(full_val_dataset, SAMPLES)

    else:
        reduced_train_dataset = full_train_dataset
        reduced_test_dataset = full_test_dataset
        reduced_val_dataset = full_val_dataset

    # Load the testing set
    logger.info(f"Loading testing set")
    test_dataset = ImageFolder(os.path.join(DATASET_PATH, 'images/test'), transform=transform)

    # Load the validation set
    logger.info(f"Loading validation set")
    val_dataset = ImageFolder(os.path.join(DATASET_PATH, 'images/val'), transform=transform)

    # Create a data loaders
    train_loader = DataLoader(reduced_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(reduced_test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(reduced_val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    return train_loader, test_loader, val_loader


def reduce_dataset(dataset: torch.utils.data.Dataset, samples_per_class: int, random_state=None) -> torch.utils.data.Subset:
    """
    Reduces the dataset size to `samples_per_class` samples per class

    Args:
        dataset (torch.utils.data.Dataset): The dataset to reduce
        samples_per_class (int): The number of samples per class
        random_state: Optional random state for reproducible results

    Returns:
        reduced_dataset (torch.utils.data.Subset): The reduced dataset
    """
    logger.info(f"Reducing dataset to {samples_per_class} samples per class")
    
    # Initialize a random state for shuffling
    rng = check_random_state(random_state)

    # Convert the list of targets to a PyTorch tensor
    targets_tensor = torch.tensor(dataset.targets)

    # Calculate the number of classes and samples per class
    num_classes = len(torch.unique(targets_tensor))
    samples_per_class = min(samples_per_class, len(dataset) // num_classes)

    # Initialize lists to store selected indices
    selected_indices = []

    # Iterate through each class
    for class_label in range(num_classes):
        # Find indices of samples belonging to the current class
        class_indices = [idx for idx, label in enumerate(targets_tensor) if label == class_label]

        # If the class has fewer samples than samples_per_class, include all samples
        if len(class_indices) <= samples_per_class:
            selected_indices.extend(class_indices)
        else:
            # Randomly select samples_per_class indices from the class
            selected_indices.extend(rng.choice(class_indices, samples_per_class, replace=False))

    # Shuffle the selected indices
    rng.shuffle(selected_indices)

    # Create a subset of the dataset using the selected indices
    reduced_dataset = Subset(dataset, selected_indices)

    return reduced_dataset

"""
The images are at `DATASET_PATH/images` containing a 1081 classes (directories) with images contained.
`DATASET_PATH/plantnet300K_metadata.json` contains the metadata of the images.
"""

# Test
if __name__ == '__main__':
    train_loader, test_loader, val_loader = load_dataset(partial=True)
    
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
        if batch_idx < 2:  # Print information for the first 2 batches
            logger.info(f"Batch {batch_idx}:")
            logger.info(f"  Data shape: {data.shape}")
            logger.info(f"  Targets shape: {targets.shape}")
        
    # Visualize a few samples (you can use matplotlib or another library for this)
    import matplotlib.pyplot as plt
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        try:
            if batch_idx < 2:  # Visualize the first 2 batches
                for i in range(data.size(0)):  # Visualize individual samples in the batch
                    sample_image = data[i].permute(1, 2, 0)  # Rearrange channels for visualization
                    plt.imshow(sample_image)
                    plt.title(f"Class: {train_loader.dataset.classes[targets[i]]}")
                    plt.show()
        except Exception as e:
            logger.error(e)
            break
