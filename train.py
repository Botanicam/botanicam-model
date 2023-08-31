from typing import *
import torch
from tqdm import tqdm
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

import epoch
from const import *
import data


def train(partial: bool = True):
    """
    Trains the model

    Args:
        partial (bool): Whether to train the model partially or not
                        If True, only 1% of the dataset is used for training,
                        else the whole dataset is used
    """
    pass