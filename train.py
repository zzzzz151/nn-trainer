import ctypes
import numpy as np
import torch
import math
import time
import sys
from batch import Batch
from model import PerspectiveNet

SUPERBATCHES = 0 # 1 superbatch = 100M positions
HIDDEN_SIZE = 32
LR = 0.001
LR_DROP_INTERVAL = 7
LR_MULTIPLIER = 0.1
SCALE = 400.0
WDL = 0.3
WEIGHT_BIAS_MAX = 1.98
QA = 255.0
QB = 64.0

if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", "CPU" if device == torch.device("cpu") else torch.cuda.get_device_name(0))

    dataloader = ctypes.CDLL("./dataloader/dataloader.so")

    # define dataloader functions return types
    dataloader.init.restype = None # void
    dataloader.loadNextBatch.restype = None # void
    dataloader.batchPtr.restype = ctypes.POINTER(Batch)

    dataloader.init()
    batch = dataloader.batchPtr().contents

    # 1 superbatch = 100M positions
    BATCHES_PER_SUPERBATCH = math.ceil(100_000_000.0 / float(batch.batch_size))

    print("Superbatches:", SUPERBATCHES)
    print("Hidden layer size:", HIDDEN_SIZE)
    print("LR: start {} multiply by {} every {} superbatches".format(LR, LR_MULTIPLIER, LR_DROP_INTERVAL))
    print("Scale:", SCALE)
    print("WDL:", WDL)
    print("Weight/bias clamp: [{}, {}]".format(-WEIGHT_BIAS_MAX, WEIGHT_BIAS_MAX))
    print("QA, QB: {}, {}".format(QA, QB))
    print()

    net = PerspectiveNet(HIDDEN_SIZE, WEIGHT_BIAS_MAX).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    for superbatch_idx in range(SUPERBATCHES):
        superbatch_start_time = time.time()
        superbatch_total_loss = 0

        # Drop learning rate
        if superbatch_idx > 0 and superbatch_idx % LR_DROP_INTERVAL == 0:
            LR *= LR_MULTIPLIER
            for param_group in optimizer.param_groups:
                param_group['lr'] = LR
            print("Dropped LR to {:f}".format(LR))

        for batch_idx in range(BATCHES_PER_SUPERBATCH):
            dataloader.loadNextBatch()

            stm_features_sparse_tensor, nstm_features_sparse_tensor = batch.features_sparse_tensors(device)

            prediction = net.forward(
                stm_features_sparse_tensor.to_dense(), 
                nstm_features_sparse_tensor.to_dense()
            )

            expected = torch.sigmoid(batch.stm_scores_tensor() / SCALE).to(device) 
            if WDL > 0.0: 
                expected *= (1.0 - WDL) 
                expected += batch.stm_results_tensor().to(device) * WDL

            loss = torch.mean((prediction - expected) ** 2)
            loss.backward()

            optimizer.step()
            net.clamp_weights_biases()

            superbatch_total_loss += loss.item()

            # Log every N batches
            if batch_idx == 0 or batch_idx == BATCHES_PER_SUPERBATCH - 1 or (batch_idx+1) % 16 == 0:
                positions_seen_this_superbatch = (batch_idx+1) * batch.batch_size
                positions_per_sec = positions_seen_this_superbatch / (time.time() - superbatch_start_time)

                log = "\rSuperbatch {}/{}, batch {}/{}, superbatch train loss {:.4f}, {} positions/s".format(
                    superbatch_idx+1, 
                    SUPERBATCHES, 
                    batch_idx+1, 
                    BATCHES_PER_SUPERBATCH, 
                    superbatch_total_loss / (batch_idx+1),
                    round(positions_per_sec)
                )

                if batch_idx == BATCHES_PER_SUPERBATCH - 1:
                    print(log)
                else:
                    sys.stdout.write(log)
                    sys.stdout.flush()  

    print("Start pos eval:", int(net.eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", device) * SCALE))

    net.save_quantized("net.nnue", QA, QB)

















