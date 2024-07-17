import os
import torch

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

NET_NAME = "net768x2"
CHECKPOINT_TO_LOAD = None # set to a .pt file to resume training, else set to None

HIDDEN_SIZE = 1024 # The final hidden layer is twice as big
OUTPUT_BUCKETS = 1 # Set to 1 for no output buckets

# 1 superbatch = 100M positions
START_SUPERBATCH = 1 # set to 1 if not resuming training
END_SUPERBATCH = 400
SAVE_INTERVAL = 100 # save net every SAVE_INTERVAL superbatches

DATA_FILE_NAME = "dataloader/2B.bin" # .bin
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