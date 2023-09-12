# Number of epochs
EPOCHS: int = 30

# Batch size
BATCH_SIZE: int = 32

# Learning rate
LR: float = 0.01

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

# Save model path
MODEL_PATH: str = 'checkpoints'

# Convergence threshold for model training improvement
CONVERGENCE_THRESHOLD: int = 3

MAX_NON_IMPROVEMENT_EPOCHS: int = 5

# Whether or not to use full or partial dataset
PARTIAL_LOAD: bool = False

# Checkpoint interval
SAVE_EVERY_N_EPOCHS: int = 1