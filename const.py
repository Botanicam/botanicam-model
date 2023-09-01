# Number of epochs
EPOCHS: int = 10

# Batch size
BATCH_SIZE: int = 32

# Learning rate
LR: float = 0.001

# Number of classes
NUM_CLASSES: int = 1081 # ref https://openreview.net/forum?id=eLYinD0TtIt

# Number of workers for dataloader
NUM_WORKERS: int = 2 # ref https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

# Path to the dataset
DATASET_PATH: str = 'plantnet_300K'

# Samples for iterative training
SAMPLES: int = 5000

# Number of workers (i.e. threads)
NUM_WORKERS: int = 2