from settings import *
from batch import *
from model import PerspectiveNet768x2
import ctypes
import numpy as np
import torch
import math
import time
import sys
import os
import warnings

if __name__ == "__main__":
    NUM_DATA_ENTRIES = int(os.path.getsize(DATA_FILE_NAME) / 32)

    print("Device:", "CPU" if device == torch.device("cpu") else torch.cuda.get_device_name(0))
    print("Net name:", NET_NAME)
    print("Net arch: (768x2x{} -> {})x2 -> {}, horizontally mirrored".format(INPUT_BUCKETS, HIDDEN_SIZE, OUTPUT_BUCKETS))
    print("Superbatches: {} to {} (save net every {})".format(START_SUPERBATCH, END_SUPERBATCH, SAVE_INTERVAL))
    print("Data entries: {} ({})".format(NUM_DATA_ENTRIES, DATA_FILE_NAME))
    print("Batch size:", BATCH_SIZE)
    print("Threads:", THREADS)
    print("LR: start {:.20f} multiply by {} every {} superbatches".format(LR, LR_MULTIPLIER, LR_DROP_INTERVAL))
    print("Scale:", SCALE)
    print("WDL:", WDL)
    print("Weights/biases clipping: [{}, {}]".format(-MAX_WEIGHT_BIAS, MAX_WEIGHT_BIAS))
    print()

    if CHECKPOINT_TO_LOAD: assert START_SUPERBATCH > 1

    # load dataloader
    assert os.path.exists("./dataloader.dll") or os.path.exists("./dataloader.so")
    dataloader = ctypes.CDLL("./dataloader.dll" if os.path.exists("./dataloader.dll") else "./dataloader.so")

    # define dataloader functions
    dataloader.init.restype = None # void
    dataloader.init.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint8, ctypes.c_uint8]
    dataloader.nextBatch.restype = ctypes.POINTER(Batch)

    # init dataloader
    dataloader.init(
        ctypes.c_char_p(DATA_FILE_NAME.encode('utf-8')),
        BATCH_SIZE, 
        THREADS,
        OUTPUT_BUCKETS
    )

    net = PerspectiveNet768x2().to(device)

    # 1 superbatch = 100M positions
    BATCHES_PER_SUPERBATCH = math.ceil(100_000_000.0 / float(BATCH_SIZE))

    #optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)

    scaler = torch.cuda.amp.GradScaler()

    net = torch.compile(net)

    # Load checkpoint if resuming training
    if CHECKPOINT_TO_LOAD:
        print("Resuming training from checkpoint", CHECKPOINT_TO_LOAD)

        checkpoint = torch.load(CHECKPOINT_TO_LOAD, 
            map_location = lambda storage, 
            loc: storage.cuda(torch.cuda.current_device()),
            weights_only=False)

        net.load_state_dict(checkpoint["model"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        assert(len(optimizer.param_groups) == 1 and "lr" in optimizer.param_groups[0])
        optimizer.param_groups[0]["lr"] = LR

        scaler.load_state_dict(checkpoint["scaler"])

    for superbatch_num in range(START_SUPERBATCH, END_SUPERBATCH + 1):
        superbatch_start_time = time.time()
        superbatch_total_loss = 0.0

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
                prediction = net.forward(
                    batch.features_dense_tensor(True), 
                    batch.features_dense_tensor(False),
                    to_tensor(batch.is_white_stm),
                    to_tensor(batch.output_buckets)
                )

                assert prediction.dtype is torch.bfloat16

                expected = torch.sigmoid(to_tensor(batch.stm_scores) / float(SCALE)) * (1.0 - WDL)
                expected += to_tensor(batch.stm_results) * WDL
                assert expected.dtype is torch.float32

                loss = torch.pow(torch.abs(torch.sigmoid(prediction) - expected), 2.5).mean()
                assert loss.dtype is torch.float32

            superbatch_total_loss += loss.item()

            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()            

            scaler.step(optimizer)
            scaler.update()

            net.clamp_weights_biases()

            # Log every N batches
            if batch_num == 1 or batch_num == BATCHES_PER_SUPERBATCH or batch_num % 32 == 0:
                positions_seen_this_superbatch = batch_num * BATCH_SIZE
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

        # Save checkpoint as .pt (pytorch file)
        if (superbatch_num - START_SUPERBATCH + 1) % SAVE_INTERVAL == 0 or superbatch_num == END_SUPERBATCH:
            checkpoint = {
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict()
            }

            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")

            pt_file_name = "checkpoints/{}-{}.pt".format(NET_NAME, superbatch_num)
            torch.save(checkpoint, pt_file_name)
            print("Checkpoint saved", pt_file_name)

