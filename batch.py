import ctypes
import numpy as np
import torch

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class Batch(ctypes.Structure):
    _fields_ = [
        ('batch_size', ctypes.c_uint32),
        ('num_active_features', ctypes.c_uint32),
        ('active_features', ctypes.POINTER(ctypes.c_int16)),
        ('is_white_stm', ctypes.POINTER(ctypes.c_bool)),
        ('stm_scores', ctypes.POINTER(ctypes.c_float)),
        ('stm_results', ctypes.POINTER(ctypes.c_float))
    ]

    def features_dense_tensor(self):
        features_tensor = torch.zeros(self.batch_size, 768, device=device)

        indices_tensor = torch.from_numpy(
                np.ctypeslib.as_array(self.active_features, shape=(self.num_active_features, 2))
            ).int()

        features_tensor[indices_tensor[:, 0], indices_tensor[:, 1]] = 1

        return features_tensor

    def is_white_stm_tensor(self):
        return torch.from_numpy(
                np.ctypeslib.as_array(self.is_white_stm, shape=(self.batch_size, 1))
            ).to(device)

    def stm_scores_tensor(self):
        return torch.from_numpy(
                np.ctypeslib.as_array(self.stm_scores, shape=(self.batch_size, 1))
            ).to(device)

    def stm_results_tensor(self):
        return torch.from_numpy(
                np.ctypeslib.as_array(self.stm_results, shape=(self.batch_size, 1))
            ).to(device)

