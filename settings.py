import os
import torch

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

NET_NAME = "net768x2-queen-buckets"
CHECKPOINT_TO_LOAD = None # set to a .pt file to resume training, else set to None

# Set to input buckets map if using input buckets
INPUT_BUCKETS_MAP = [
#   A  B  C  D  E  F  G  H
    1, 1, 1, 1, 2, 2, 2, 2, # 0
    1, 1, 1, 1, 2, 2, 2, 2, # 1
    1, 1, 1, 1, 2, 2, 2, 2, # 2
    1, 1, 1, 1, 2, 2, 2, 2, # 3
    3, 3, 3, 3, 4, 4, 4, 4, # 4
    3, 3, 3, 3, 4, 4, 4, 4, # 5
    3, 3, 3, 3, 4, 4, 4, 4, # 6 
    3, 3, 3, 3, 4, 4, 4, 4, # 7
    0
]

# Uncomment this line if not using input buckets
#INPUT_BUCKETS_MAP = [0 for _ in range(65)]

HIDDEN_SIZE = 1024 # The final hidden layer is twice as big
FACTORIZER = False # Set to False if no input buckets
INPUT_BUCKETS = max(INPUT_BUCKETS_MAP) + 1 + FACTORIZER
OUTPUT_BUCKETS = 1 # Set to 1 if no output buckets

# 1 superbatch = 100M positions
START_SUPERBATCH = 1 # set to 1 if not resuming training
END_SUPERBATCH = 400
SAVE_INTERVAL = 40 # save net every SAVE_INTERVAL superbatches

DATA_FILE_NAME = "2B.bin" # .bin
BATCH_SIZE = 16384
THREADS = 12

# Adjust LR if resuming training
LR = 0.001
LR_DROP_INTERVAL = 1
LR_MULTIPLIER = 0.99

SCALE = 400
WDL = 0.0
MAX_WEIGHT_BIAS = 2.0

assert NET_NAME != ""
if CHECKPOINT_TO_LOAD: assert os.path.exists(CHECKPOINT_TO_LOAD)
assert HIDDEN_SIZE > 0
if not FACTORIZER: assert INPUT_BUCKETS >= 1
if FACTORIZER: assert INPUT_BUCKETS - FACTORIZER > 1
assert OUTPUT_BUCKETS >= 1
if not CHECKPOINT_TO_LOAD: assert START_SUPERBATCH == 1
assert END_SUPERBATCH >= START_SUPERBATCH
assert SAVE_INTERVAL > 0
assert os.path.exists(DATA_FILE_NAME)
assert BATCH_SIZE > 0
assert THREADS > 0
assert LR > 0.0 and LR_DROP_INTERVAL > 0 and LR_MULTIPLIER > 0.0
assert SCALE > 0
assert WDL >= 0.0 and WDL <= 1.0
assert MAX_WEIGHT_BIAS > 0.0