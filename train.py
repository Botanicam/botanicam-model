from typing import *
import argparse

from const import *
from log_cfg import logger
import data
import model


def train(partial: bool = True):
    """
    Trains the model

    Args:
        partial (bool): Whether to train the model partially or not
                        If True, only 1% of the dataset is used for training,
                        else the whole dataset is used
    """
    logger.debug(f"Training {'partially' if partial else 'fully'}")
    # Load the dataset
    train_loader, test_loader, val_loader = data.load_dataset(partial)

    # Load the model
    m = model.BotanicamModel()

    # Train!
    m.train(train_loader, val_loader)

    # Save the model
    m.save(path="model_save.pth")
    m.plot_training()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--partial", action="store_true", help="Whether to train the model partially or not")
    args = parser.parse_args()

    # Train the model
    train(args.partial)