import torch
import numpy as np
import struct

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class SCReLU(torch.nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return torch.pow(torch.clamp(x, 0, 1), 2)

class PerspectiveNet(torch.nn.Module):
    def __init__(self, hidden_size, weight_bias_max):
        super().__init__()
        self.conn1 = torch.nn.Linear(768, hidden_size)
        self.conn2 = torch.nn.Linear(hidden_size * 2, 1)
        self.screlu = SCReLU()
        self.weight_bias_max = weight_bias_max

        # Random weights and biases
        torch.manual_seed(42)
        with torch.no_grad():
            self.conn1.weight.uniform_(-self.weight_bias_max, self.weight_bias_max)
            self.conn1.bias.uniform_(-self.weight_bias_max, self.weight_bias_max)
            self.conn2.weight.uniform_(-self.weight_bias_max, self.weight_bias_max)
            self.conn2.bias.uniform_(-self.weight_bias_max, self.weight_bias_max)

    # The arguments should be dense tensors and not sparse tensors, as the former are way faster
    def forward(self, stm_features_tensor, nstm_features_tensor):
        stm_hidden = self.conn1(stm_features_tensor)
        nstm_hidden = self.conn1(nstm_features_tensor)

        hidden_layer = torch.cat((stm_hidden, nstm_hidden), dim = len(stm_features_tensor.size()) - 1)
        hidden_layer = self.screlu(hidden_layer)

        return self.conn2(hidden_layer)

    def clamp_weights_biases(self):
        self.conn1.weight.data.clamp_(-self.weight_bias_max, self.weight_bias_max)
        self.conn1.bias.data.clamp_(-self.weight_bias_max, self.weight_bias_max)
        self.conn2.weight.data.clamp_(-self.weight_bias_max, self.weight_bias_max)
        self.conn2.bias.data.clamp_(-self.weight_bias_max, self.weight_bias_max)

    def eval(self, fen):
        fen = fen.split(" ")
        stm = 1 if fen[-5] == "b" else 0

        stm_features_dense_tensor = torch.zeros(768, device=device)
        nstm_features_dense_tensor = torch.zeros(768, device=device)

        for rank_idx, rank in enumerate(fen[0].split('/')):
            file_idx = 0
            for char in rank:
                if char.isdigit():
                    file_idx += int(char)
                else:
                    sq = 8 * (7 - rank_idx) + file_idx
                    piece_type = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}[char.lower()]

                    is_black_piece = char.islower() 
                    piece_color = 1 if is_black_piece else 0

                    feature1 = piece_color * 384 + piece_type * 64 + sq
                    feature2 = (1 - piece_color) * 384 + piece_type * 64 + (sq ^ 56)

                    if stm == 0:
                        stm_features_dense_tensor[feature1] = 1
                        nstm_features_dense_tensor[feature2] = 1
                    else:
                        stm_features_dense_tensor[feature2] = 1
                        nstm_features_dense_tensor[feature1] = 1
                    
                    file_idx += 1

        return self.forward(stm_features_dense_tensor, nstm_features_dense_tensor)

    def save_quantized(self, file_name, QA, QB):
        QA = float(QA)
        QB = float(QB)

        # Extract weights and biases
        weights1 = self.conn1.weight.detach().cpu().numpy()
        bias1 = self.conn1.bias.detach().cpu().numpy()
        weights2 = self.conn2.weight.detach().cpu().numpy()
        bias2 = self.conn2.bias.detach().cpu().numpy()

        # Quantize weights and biases
        weights1_quantized = np.round(weights1 * QA).T.astype(np.int16)
        bias1_quantized = np.round(bias1 * QA).astype(np.int16)
        weights2_quantized = np.round(weights2 * QB).astype(np.int16)
        bias2_quantized = np.round(bias2 * QA * QB).astype(np.int16)

        # Flatten to 1D lists
        weights1_1d = weights1_quantized.flatten().tolist()
        bias1_1d = bias1_quantized.flatten().tolist()
        weights2_1d = weights2_quantized.flatten().tolist()
        bias2_1d = bias2_quantized.flatten().tolist()

        # Save to binary file
        with open(file_name, "wb") as bin_file:
            bin_file.write(struct.pack('<' + 'h' * len(weights1_1d), *weights1_1d))
            bin_file.write(struct.pack('<' + 'h' * len(bias1_1d), *bias1_1d))
            bin_file.write(struct.pack('<' + 'h' * len(weights2_1d), *weights2_1d))
            bin_file.write(struct.pack('<' + 'h' * len(bias2_1d), *bias2_1d))
