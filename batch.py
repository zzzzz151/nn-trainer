import ctypes
import numpy as np
import torch

class Batch(ctypes.Structure):
    _fields_ = [
        ('batch_size', ctypes.c_uint32),
        ('num_active_features', ctypes.c_uint32),
        ('stm_features', ctypes.POINTER(ctypes.c_int16)),
        ('nstm_features', ctypes.POINTER(ctypes.c_int16)),
        ('stm_scores', ctypes.POINTER(ctypes.c_float)),
        ('stm_results', ctypes.POINTER(ctypes.c_float))
    ]

    def stm_scores_tensor(self):
        return torch.from_numpy(
            np.ctypeslib.as_array(self.stm_scores, shape=(self.batch_size, 1))
        )

    def stm_results_tensor(self):
        return torch.from_numpy(
            np.ctypeslib.as_array(self.stm_results, shape=(self.batch_size, 1))
        )

    def features_sparse_tensors(self, device):
        stm_features_indices_tensor = torch.transpose(
            torch.from_numpy(
                np.ctypeslib.as_array(self.stm_features, shape=(self.num_active_features, 2))
            ), 
            0, 
            1
        )

        nstm_features_indices_tensor = torch.transpose(
            torch.from_numpy(
                np.ctypeslib.as_array(self.nstm_features, shape=(self.num_active_features, 2))
            ), 
            0, 
            1
        )

        ones = torch.ones(self.num_active_features)

        stm_features_sparse_tensor = torch.sparse_coo_tensor(
            stm_features_indices_tensor.long(),
            ones,
            (self.batch_size, 768),
            check_invariants=False,
            is_coalesced=True,
            device=device
        )

        nstm_features_sparse_tensor = torch.sparse_coo_tensor(
            nstm_features_indices_tensor.long(),
            ones,
            (self.batch_size, 768),
            check_invariants=False,
            is_coalesced=True,
            device=device
        )

        return stm_features_sparse_tensor, nstm_features_sparse_tensor