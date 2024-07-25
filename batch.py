from settings import *
import ctypes
import numpy as np
import torch

class Batch(ctypes.Structure):
    _fields_ = [
        ('num_active_features', ctypes.c_uint32),
        ('active_features_white_stm', ctypes.POINTER(ctypes.c_int16)),
        ('active_features_black_stm', ctypes.POINTER(ctypes.c_int16)),
        ('is_white_stm', ctypes.POINTER(ctypes.c_bool)),
        ('stm_scores', ctypes.POINTER(ctypes.c_float)),
        ('stm_results', ctypes.POINTER(ctypes.c_float)),
        ('output_buckets', ctypes.POINTER(ctypes.c_uint8))
    ]

    def features_dense_tensor(self, white_stm: bool):
        features_tensor = torch.zeros(BATCH_SIZE, 768, device=device)

        arr = np.ctypeslib.as_array(
            self.active_features_white_stm if white_stm else self.active_features_black_stm,
            shape=(self.num_active_features, 2)
        )

        indices_tensor = torch.from_numpy(arr).int()

        features_tensor[indices_tensor[:, 0], indices_tensor[:, 1]] = 1

        return features_tensor

def to_tensor(x):
    arr = np.ctypeslib.as_array(x, shape=(BATCH_SIZE, 1))
    return torch.from_numpy(arr).to(device)

