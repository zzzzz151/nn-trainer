import ctypes
import numpy as np
import torch
import math
import time
import sys
import os
import warnings
from batch import Batch
from model import PerspectiveNet

warnings.filterwarnings("ignore")

NET_NAME = "net1024"
NET_TO_LOAD = None # set to a .pt file to resume training, else set to None or ""
HIDDEN_SIZE = 1024

# 1 superbatch = 100M positions
START_SUPERBATCH = 1 # set to 1 if not resuming training
END_SUPERBATCH = 400
SAVE_INTERVAL = 100 # save net every SAVE_INTERVAL superbatches

DATA_FILE_NAME = "starzix2B.bf" # bulletformat
BATCH_SIZE = 16384
THREADS = 12

# Adjust LR if resuming training
LR = 0.001
LR_DROP_INTERVAL = 1
LR_MULTIPLIER = 0.99

SCALE = 400
WDL = 0.0
WEIGHT_BIAS_MAX = 2.0

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":
    assert NET_NAME != ""
    if NET_TO_LOAD != None and NET_TO_LOAD != "": assert os.path.exists(NET_TO_LOAD)
    assert HIDDEN_SIZE > 0
    assert(START_SUPERBATCH == 1 if NET_TO_LOAD == None or NET_TO_LOAD == "" else START_SUPERBATCH > 1)
    assert END_SUPERBATCH >= START_SUPERBATCH
    assert SAVE_INTERVAL > 0
    assert os.path.exists(DATA_FILE_NAME)
    assert BATCH_SIZE > 0
    assert THREADS > 0
    assert LR > 0.0 and LR_DROP_INTERVAL > 0 and LR_MULTIPLIER > 0.0
    assert SCALE > 0
    assert WDL >= 0.0 and WDL <= 1.0
    assert WEIGHT_BIAS_MAX > 0.0

    assert os.path.exists("./dataloader.dll") or os.path.exists("./dataloader.so")
    dataloader = ctypes.CDLL("./dataloader.dll" if os.path.exists("./dataloader.dll") else "./dataloader.so")

    # define dataloader functions
    dataloader.init.restype = None # void
    dataloader.init.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.c_uint64]
    dataloader.numDataEntries.restype = ctypes.c_uint64
    dataloader.nextBatch.restype = ctypes.POINTER(Batch)

    dataloader.init(
        ctypes.c_char_p(DATA_FILE_NAME.encode('utf-8')),
        BATCH_SIZE, 
        THREADS
    )

    print("Device:", "CPU" if device == torch.device("cpu") else torch.cuda.get_device_name(0))
    print("Net name:", NET_NAME)
    print("Net arch: (768 -> {})x2 -> 1".format(HIDDEN_SIZE))
    print("Superbatches: {} to {} (save net every {})".format(START_SUPERBATCH, END_SUPERBATCH, SAVE_INTERVAL))
    print("Data entries: {} ({})".format(dataloader.numDataEntries(), DATA_FILE_NAME))
    print("Batch size:", BATCH_SIZE)
    print("Threads:", THREADS)
    print("LR: start {:.20f} multiply by {} every {} superbatches".format(LR, LR_MULTIPLIER, LR_DROP_INTERVAL))
    print("Scale:", SCALE)
    print("WDL:", WDL)
    print("Weight/bias clamp: [{}, {}]".format(-WEIGHT_BIAS_MAX, WEIGHT_BIAS_MAX))
    print()

    SCALE = float(SCALE)

    # 1 superbatch = 100M positions
    BATCHES_PER_SUPERBATCH = math.ceil(100_000_000.0 / float(BATCH_SIZE))

    if not os.path.exists("nets"):
        os.makedirs("nets")
    
    net = PerspectiveNet(HIDDEN_SIZE, WEIGHT_BIAS_MAX).to(device)

    # Load net if resuming training
    if NET_TO_LOAD != None and NET_TO_LOAD != "":
        net.load_state_dict(torch.load(NET_TO_LOAD))
        print("Loaded net", NET_TO_LOAD)
        print()

    #optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    
    for superbatch_num in range(START_SUPERBATCH, END_SUPERBATCH + 1):
        superbatch_start_time = time.time()
        superbatch_total_loss = 0

        # Drop learning rate
        if superbatch_num > START_SUPERBATCH and (superbatch_num - START_SUPERBATCH) % LR_DROP_INTERVAL == 0:
            LR *= LR_MULTIPLIER
            for param_group in optimizer.param_groups:
                param_group['lr'] = LR
            print("LR dropped to {:.20f}".format(LR))

        for batch_num in range(1, BATCHES_PER_SUPERBATCH + 1):
            batch = dataloader.nextBatch().contents

            optimizer.zero_grad()
            
            stm_features_dense_tensor, nstm_features_dense_tensor = batch.features_dense_tensors()
            prediction = net.forward(stm_features_dense_tensor, nstm_features_dense_tensor)

            expected = torch.sigmoid(batch.stm_scores_tensor() / SCALE) * (1.0 - WDL)
            expected += batch.stm_results_tensor() * WDL
            expected = expected.to(device)

            loss = torch.mean((torch.sigmoid(prediction) - expected) ** 2)
            loss.backward()

            optimizer.step()
            net.clamp_weights_biases()

            superbatch_total_loss += loss.item()

            # Log every N batches
            if batch_num == 1 or batch_num == BATCHES_PER_SUPERBATCH or batch_num % 32 == 0:
                positions_seen_this_superbatch = batch_num * batch.batch_size
                positions_per_sec = positions_seen_this_superbatch / (time.time() - superbatch_start_time)

                log = "\rSuperbatch {}/{}, batch {}/{}, superbatch train loss {:.4f}, {} positions/s".format(
                    superbatch_num,
                    END_SUPERBATCH, 
                    batch_num,
                    BATCHES_PER_SUPERBATCH, 
                    superbatch_total_loss / batch_num,
                    round(positions_per_sec)
                )

                if batch_num == BATCHES_PER_SUPERBATCH:
                    print(log)
                else:
                    sys.stdout.write(log)
                    sys.stdout.flush()  

        # Save net as .pt (pytorch file)
        if (superbatch_num - START_SUPERBATCH + 1) % SAVE_INTERVAL == 0 or superbatch_num == END_SUPERBATCH:
            pt_file_name = "nets/{}-{}.pt".format(NET_NAME, superbatch_num)
            torch.save(net.state_dict(), pt_file_name)
            print("Net saved", pt_file_name)

    print("Start pos eval:", int(net.eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") * SCALE))
    print("e2e4 eval:", int(net.eval("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1") * SCALE))

















