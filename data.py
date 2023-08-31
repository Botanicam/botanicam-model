# loads data

from typing import *

import torch
import numpy as np
from torch.utils.data import DataLoader

from const import *

"""
The images are at `DATASET_PATH/images` containing a 1081 classes (directories) with images contained.
`DATASET_PATH/plantnet300K_metadata.json` contains the metadata of the images.
"""