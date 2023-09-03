from typing import *
import argparse

from torch.utils.data import DataLoader

from const import *
from log_cfg import logger
import data
import model


def load_dataset(partial: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads the dataset

    Args:
        partial (bool): Whether to load the dataset partially or not
                        If True, only 1% of the dataset is loaded,
                        else the whole dataset is loaded

    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]: Train, test and validation data loaders
    """
    logger.debug(f"Loading {'partially' if partial else 'fully'}")
    # Load the dataset
    train_loader, test_loader, val_loader = data.load_dataset(partial)

    return train_loader, test_loader, val_loader


def train(train_loader: DataLoader, val_loader: DataLoader, partial: bool = True) -> model.BotanicamModel:
    """
    Trains the model

    Args:
        partial (bool): Whether to train the model partially or not
                        If True, only 1% of the dataset is used for training,
                        else the whole dataset is used
    """
    logger.debug(f"Training {'partially' if partial else 'fully'}")

    # Load the model
    m = model.BotanicamModel()

    # Train!
    m.train(train_loader, val_loader)

    # Save the model
    m.save(path="model_save.pth")
    
    return m

def load_model(path="model_save.pth"):
    """
    Loads the model

    Returns:
        BotanicamModel: Model
    """
    # Load the model
    m = model.BotanicamModel()

    # Load the saved model
    m.load(path="model_save.pth")

    return m


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train the model")

    parser.add_argument("--partial", action="store_true", help="Whether to train the model partially or not")
    # load and filename
    parser.add_argument("--load", action="store_true", help="Whether to load the model or not")
    parser.add_argument("--filename", type=str, help="Filename of the model to load")

    args = parser.parse_args()

    # Load the dataset
    train_loader, test_loader, val_loader = load_dataset(args.partial)

    if args.load:
        # Load the model
        m = load_model(args.filename)
    else:
        # Train the model
        m = train(
            train_loader=train_loader,
            val_loader=val_loader,
            partial=args.partial
        )
    
    # Test
    accuracy = m.test(test_loader=test_loader)

    # Print history if the model was trained
    if not args.load:
        m.plot_training()