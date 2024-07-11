import ctypes
import numpy as np
import torch
import math
import time
import sys
import os
import warnings
from batch import Batch
from model import PerspectiveNet768x2

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

NET_NAME = "net768x2"
CHECKPOINT_TO_LOAD = None # set to a .pt file to resume training, else set to None or ""

net = PerspectiveNet768x2(hidden_size=1024).to(device)

# 1 superbatch = 100M positions
START_SUPERBATCH = 1 # set to 1 if not resuming training
END_SUPERBATCH = 400
SAVE_INTERVAL = 40 # save net every SAVE_INTERVAL superbatches

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

if __name__ == "__main__":
    assert NET_NAME != ""
    if CHECKPOINT_TO_LOAD != None and CHECKPOINT_TO_LOAD != "": assert os.path.exists(CHECKPOINT_TO_LOAD)
    assert(START_SUPERBATCH == 1 if CHECKPOINT_TO_LOAD == None or CHECKPOINT_TO_LOAD == "" else START_SUPERBATCH > 1)
    assert END_SUPERBATCH >= START_SUPERBATCH
    assert SAVE_INTERVAL > 0
    assert os.path.exists(DATA_FILE_NAME)
    assert BATCH_SIZE > 0
    assert THREADS > 0
    assert LR > 0.0 and LR_DROP_INTERVAL > 0 and LR_MULTIPLIER > 0.0
    assert SCALE > 0
    assert WDL >= 0.0 and WDL <= 1.0
    assert MAX_WEIGHT_BIAS > 0.0

    # load dataloader exe/binary
    dll = "./dataloader/dataloader.dll"
    so = "./dataloader/dataloader.so"
    assert os.path.exists(dll) or os.path.exists(so)
    dataloader = ctypes.CDLL(dll if os.path.exists(dll) else so)

    # define dataloader functions
    dataloader.init.restype = None # void
    dataloader.init.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.c_uint64]
    dataloader.numDataEntries.restype = ctypes.c_uint64
    dataloader.nextBatch.restype = ctypes.POINTER(Batch)

    # init dataloader
    dataloader.init(
        ctypes.c_char_p(DATA_FILE_NAME.encode('utf-8')),
        BATCH_SIZE, 
        THREADS,
    )

    print("Device:", "CPU" if device == torch.device("cpu") else torch.cuda.get_device_name(0))
    print("Net name:", NET_NAME)
    print("Net arch: (768x2 -> {})x2 -> 1".format(net.HIDDEN_SIZE))
    print("Superbatches: {} to {} (save net every {})".format(START_SUPERBATCH, END_SUPERBATCH, SAVE_INTERVAL))
    print("Data entries: {} ({})".format(dataloader.numDataEntries(), DATA_FILE_NAME))
    print("Batch size:", BATCH_SIZE)
    print("Threads:", THREADS)
    print("LR: start {:.20f} multiply by {} every {} superbatches".format(LR, LR_MULTIPLIER, LR_DROP_INTERVAL))
    print("Scale:", SCALE)
    print("WDL:", WDL)
    print("Weights/biases clipping: [{}, {}]".format(-MAX_WEIGHT_BIAS, MAX_WEIGHT_BIAS))
    print()

    # 1 superbatch = 100M positions
    BATCHES_PER_SUPERBATCH = math.ceil(100_000_000.0 / float(BATCH_SIZE))

    #optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)

    scaler = torch.cuda.amp.GradScaler()

    # Load checkpoint if resuming training
    if CHECKPOINT_TO_LOAD != None and CHECKPOINT_TO_LOAD != "":
        print("Resuming training from checkpoint", CHECKPOINT_TO_LOAD)

        checkpoint = torch.load(CHECKPOINT_TO_LOAD, 
            map_location = lambda storage, 
            loc: storage.cuda(torch.cuda.current_device()),
            weights_only=False)

        net.load_state_dict(checkpoint["model"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        assert(len(optimizer.param_groups) == 1 and "lr" in optimizer.param_groups[0])
        optimizer.param_groups[0]["lr"] = LR

        assert optimizer.param_groups[0]["weight_decay"] == 0.01

        scaler.load_state_dict(checkpoint["scaler"])

    net = torch.compile(net)

    for superbatch_num in range(START_SUPERBATCH, END_SUPERBATCH + 1):
        superbatch_start_time = time.time()
        superbatch_total_loss = 0.0
        superbatch_total_scaled_loss = 0.0

        # Drop learning rate
        if superbatch_num > START_SUPERBATCH and (superbatch_num - START_SUPERBATCH) % LR_DROP_INTERVAL == 0:
            LR *= LR_MULTIPLIER
            assert(len(optimizer.param_groups) == 1 and "lr" in optimizer.param_groups[0])
            optimizer.param_groups[0]["lr"] = LR
            print("LR dropped to {:.20f}".format(LR))

        for batch_num in range(1, BATCHES_PER_SUPERBATCH + 1):
            batch = dataloader.nextBatch().contents

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = net.forward(batch.features_dense_tensor(), batch.to_tensor("is_white_stm"))
                assert prediction.dtype is torch.bfloat16

                expected = torch.sigmoid(batch.to_tensor("stm_scores") / float(SCALE)) * (1.0 - WDL)
                expected += batch.to_tensor("stm_results") * WDL
                assert expected.dtype is torch.float32

                loss = torch.pow(torch.abs(torch.sigmoid(prediction) - expected), 2.5).mean()
                assert loss.dtype is torch.float32

            superbatch_total_loss += loss.item()

            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()            
            superbatch_total_scaled_loss += scaled_loss.item()

            scaler.step(optimizer)
            scaler.update()

            net.clamp_weights_biases(MAX_WEIGHT_BIAS)

            # Log every N batches
            if batch_num == 1 or batch_num == BATCHES_PER_SUPERBATCH or batch_num % 32 == 0:
                positions_seen_this_superbatch = batch_num * batch.batch_size
                positions_per_sec = positions_seen_this_superbatch / (time.time() - superbatch_start_time)

                log = "\rSuperbatch {}/{}, batch {}/{}, superbatch train loss {:.4f} ({} scaled), {} positions/s".format(
                    superbatch_num,
                    END_SUPERBATCH, 
                    batch_num,
                    BATCHES_PER_SUPERBATCH, 
                    superbatch_total_loss / batch_num,
                    round(superbatch_total_scaled_loss / batch_num),
                    round(positions_per_sec)
                )

                if batch_num == BATCHES_PER_SUPERBATCH:
                    print(log)
                else:
                    sys.stdout.write(log)
                    sys.stdout.flush()  

        # Save checkpoint as .pt (pytorch file)
        if (superbatch_num - START_SUPERBATCH + 1) % SAVE_INTERVAL == 0 or superbatch_num == END_SUPERBATCH:

            checkpoint = {"model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict()
            }

            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")

            pt_file_name = "checkpoints/{}-{}.pt".format(NET_NAME, superbatch_num)
            torch.save(checkpoint, pt_file_name)
            print("Checkpoint saved", pt_file_name)

