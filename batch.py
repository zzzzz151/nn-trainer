from settings import *
import ctypes
import numpy as np
import torch

class Batch(ctypes.Structure):
    _fields_ = [
        ('num_active_features', ctypes.c_uint),
        ('active_features_white_stm', ctypes.POINTER(ctypes.c_int)),
        ('active_features_black_stm', ctypes.POINTER(ctypes.c_int)),
        ('is_white_stm', ctypes.POINTER(ctypes.c_bool)),
        ('stm_scores', ctypes.POINTER(ctypes.c_float)),
        ('stm_results', ctypes.POINTER(ctypes.c_float)),
        ('output_buckets', ctypes.POINTER(ctypes.c_uint8))
    ]

    def features_dense_tensor(self, white_stm: bool):
        features_tensor = torch.zeros(BATCH_SIZE, 768 * INPUT_BUCKETS, device=device)

        arr = np.ctypeslib.as_array(
            self.active_features_white_stm if white_stm else self.active_features_black_stm,
            shape=(BATCH_SIZE, 64 if FACTORIZER else 32)
        )

        return torch.from_numpy(arr).to(device)

def to_tensor(x):
    arr = np.ctypeslib.as_array(x, shape=(BATCH_SIZE, 1))
    return torch.from_numpy(arr).to(device)

