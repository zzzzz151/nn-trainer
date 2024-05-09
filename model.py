import torch
from batch import Batch

WEIGHT_MAX = 1.98

class PerspectiveNet(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.conn1 = torch.nn.Linear(768, hidden_size)
        self.conn2 = torch.nn.Linear(hidden_size * 2, 1)

        # Random weights and biases
        torch.manual_seed(42)
        with torch.no_grad():
            self.conn1.weight.uniform_(-WEIGHT_MAX, WEIGHT_MAX)
            self.conn2.weight.uniform_(-WEIGHT_MAX, WEIGHT_MAX)
            self.conn1.bias.uniform_(-WEIGHT_MAX, WEIGHT_MAX)
            self.conn2.bias.uniform_(-WEIGHT_MAX, WEIGHT_MAX)

    def forward(self, stm_features_tensor, nstm_features_tensor):
        stm_hidden = self.conn1(stm_features_tensor.to_dense())
        nstm_hidden = self.conn1(nstm_features_tensor.to_dense())

        hidden_layer = torch.cat((stm_hidden, nstm_hidden), dim=1)
        hidden_layer = torch.pow(torch.clamp(hidden_layer, 0, 1), 2) # screlu activation

        return torch.sigmoid(self.conn2(hidden_layer))

    def clamp_weights(self):
        self.conn1.weight.data.clamp_(-WEIGHT_MAX, WEIGHT_MAX)
        self.conn2.weight.data.clamp_(-WEIGHT_MAX, WEIGHT_MAX)

    def start_pos_eval(self, device):
        dense_stm_tensor = torch.zeros(768, device=device)
        dense_nstm_tensor = torch.zeros(768, device=device)

        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

        for rank_idx, rank in enumerate(fen.split('/')):
            file_idx = 0
            for char in rank:
                if char.isdigit():
                    file_idx += int(char)
                else:
                    sq = 8 * (7 - rank_idx) + file_idx
                    piece_type = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}[char.lower()]

                    is_black_piece = char.islower() 
                    piece_color = 1 if is_black_piece else 0
                    
                    dense_stm_tensor[piece_color * 384 + piece_type * 64 + sq] = 1
                    dense_nstm_tensor[(1 - piece_color) * 384 + piece_type * 64 + (sq ^ 56)] = 1
                    
                    file_idx += 1

        stm_hidden = self.conn1(dense_stm_tensor)
        nstm_hidden = self.conn1(dense_nstm_tensor)

        hidden_layer = torch.cat((stm_hidden, nstm_hidden), dim=0)
        hidden_layer = torch.pow(torch.clamp(hidden_layer, 0, 1), 2) # screlu activation

        return torch.sigmoid(self.conn2(hidden_layer))
