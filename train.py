from typing import Tuple
import argparse
from model import BotanicamModel
from torch.utils.data import DataLoader


from const import *
from log_cfg import logger
import data

def load_dataset(partial: bool = PARTIAL_LOAD) -> Tuple[DataLoader, DataLoader, DataLoader]:
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



if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train the model")

    parser.add_argument("--partial", action="store_true", help="Whether to train the model partially or not")
    # load and filename
    parser.add_argument("--load", action="store_true", help="Whether to load the model or not")
    parser.add_argument("--filename", type=str, help="Filename of the model to load")

    # ENTER EPOCH TO RESUME LOADING BY python train.py --resume checkpoint/checkpoint_epoch_{HIGHEST CHECKPOINT NUMBER IN THE FOLDER}.pth in dir
    parser.add_argument("--resume", type=str, help="Path to checkpoint file to resume training from") 
    

    # resume training from epoch checkpoint
    args = parser.parse_args()

    # Load the dataset
    train_loader, test_loader, val_loader = load_dataset(args.partial)

    # Initialise model
    m = BotanicamModel()
    if args.load:
        # Load the model
        m.load(args.filename)
    else:

        if args.resume:
            # Train model or resume from a checkpoint
            m.resume_training(
                checkpoint_path=args.resume, 
                train_loader=train_loader, 
                val_loader=val_loader)
        else:
            # Train from scratch
            m.train(
                train_loader=train_loader,
                val_loader=val_loader,
            )
    
    # Test
    accuracy = m.test(test_loader=test_loader)

    # Print history if the model was trained
    if not args.load:
        m.plot_training()
