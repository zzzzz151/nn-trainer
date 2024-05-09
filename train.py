import ctypes
import numpy as np
import torch
import math
import time
import sys
import os
import json
from json import JSONEncoder
from batch import Batch
from model import PerspectiveNet

HIDDEN_SIZE = 32
SUPERBATCHES = 100
POSITIONS_PER_SUPERBATCH = 100_000_000
LR = 0.001
LR_DROP_INTERVAL = 50
LR_MULTIPLIER = 0.1
SCALE = 400.0
WDL = 0.3
QA = 255
QB = 64
NETS_FOLDER = "nets"

if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", "CPU" if device == torch.device("cpu") else torch.cuda.get_device_name(0))
    
    print("Hidden layer size:", HIDDEN_SIZE)

    dataloader = ctypes.CDLL("./dataloader/dataloader.so")

    # define dataloader functions return types
    dataloader.init.restype = None # void
    dataloader.loadNextBatch.restype = None # void
    dataloader.batchPtr.restype = ctypes.POINTER(Batch)

    dataloader.init()
    batch = dataloader.batchPtr().contents

    assert POSITIONS_PER_SUPERBATCH >= batch.batch_size
    BATCHES_PER_SUPERBATCH = math.ceil(float(POSITIONS_PER_SUPERBATCH) / float(batch.batch_size))

    print("Superbatches:", SUPERBATCHES)
    print("Positions per superbatch:", POSITIONS_PER_SUPERBATCH)
    print("Batches per superbatch:", BATCHES_PER_SUPERBATCH)
    print("LR:", LR)
    print("LR drop interval:", LR_DROP_INTERVAL)
    print("LR drop multiplier:", LR_MULTIPLIER)
    print("WDL:", WDL)
    print("Scale:", SCALE)
    print("QA:", QA)
    print("QB", QB)
    print()

    net = PerspectiveNet(HIDDEN_SIZE).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    for superbatch_idx in range(SUPERBATCHES):
        print("Start pos eval:", int(net.start_pos_eval(device) * SCALE))

        superbatch_start_time = time.time()
        superbatch_total_loss = 0

        # Drop learning rate
        if superbatch_idx > 0 and superbatch_idx % LR_DROP_INTERVAL == 0:
            LR *= LR_MULTIPLIER
            for param_group in optimizer.param_groups:
                param_group['lr'] = LR
            print("Dropped LR to {:f}".format(LR))

        for batch_idx in range(BATCHES_PER_SUPERBATCH):

            #print("Starting superbatch {}/{} batch {}/{}".format(
            #    superbatch_idx+1, SUPERBATCHES, batch_idx+1, BATCHES_PER_SUPERBATCH))

            dataloader.loadNextBatch()

            stm_features_tensor, nstm_features_tensor = batch.features_tensors(device)
            prediction = net.forward(stm_features_tensor, nstm_features_tensor)

            expected = torch.sigmoid(batch.stm_scores_tensor() / SCALE).to(device) * (1.0 - WDL) 
            expected += batch.stm_results_tensor().to(device) * WDL

            loss = torch.mean((prediction - expected) ** 2)
            loss.backward()

            optimizer.step()
            net.clamp_weights()

            superbatch_total_loss += loss.item()

            if batch_idx == 0 or batch_idx == BATCHES_PER_SUPERBATCH - 1 or (batch_idx+1) % 8 == 0:
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

    # Save net

    # Ensure float32 weights and biases
    for param_name, param in net.named_parameters():
        param.data = param.data.float()

    if not os.path.exists(NETS_FOLDER):
        os.makedirs(NETS_FOLDER)

    class EncodeTensor(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().detach().numpy().tolist()
            return super(EncodeTensor, self).default(obj)

    with open(NETS_FOLDER + "/unquantized.json", 'w') as json_file:
        json.dump(net.state_dict(), json_file, cls=EncodeTensor)















