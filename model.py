import torch

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class PerspectiveNet768x2(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()

        self.HIDDEN_SIZE = hidden_size
        self.features_to_hidden_white_stm = torch.nn.Linear(768, hidden_size)
        self.features_to_hidden_black_stm = torch.nn.Linear(768, hidden_size)
        self.hidden_to_out = torch.nn.Linear(hidden_size * 2, 1)

        # Random weights and biases
        torch.manual_seed(42)
        with torch.no_grad():
            self.features_to_hidden_white_stm.weight.uniform_(-0.1, 0.1)
            self.features_to_hidden_white_stm.bias.uniform_(-0.1, 0.1)

            self.features_to_hidden_black_stm.weight.uniform_(-0.1, 0.1)
            self.features_to_hidden_black_stm.bias.uniform_(-0.1, 0.1)

            self.hidden_to_out.weight.uniform_(-0.1, 0.1)
            self.hidden_to_out.bias.uniform_(-0.1, 0.1)

    # The arguments should be dense tensors and not sparse tensors, as the former are way faster
    def forward(self, features_tensor: torch.Tensor, is_white_stm_tensor: torch.Tensor):
        white_hidden = self.features_to_hidden_white_stm(features_tensor)
        black_hidden = self.features_to_hidden_black_stm(features_tensor)

        # stm accumulator first
        dim = len(features_tensor.size()) - 1
        hidden_layer = is_white_stm_tensor * torch.cat([white_hidden, black_hidden], dim=dim)
        hidden_layer += ~is_white_stm_tensor * torch.cat([black_hidden, white_hidden], dim=dim)

        # SCReLU activation
        hidden_layer = torch.pow(torch.clamp(hidden_layer, 0, 1), 2) 

        return self.hidden_to_out(hidden_layer)

    def clamp_weights_biases(self, maximum: float):
        assert maximum > 0.0

        self.features_to_hidden_white_stm.weight.data.clamp_(-maximum, maximum)
        self.features_to_hidden_white_stm.bias.data.clamp_(-maximum, maximum)

        self.features_to_hidden_black_stm.weight.data.clamp_(-maximum, maximum)
        self.features_to_hidden_black_stm.bias.data.clamp_(-maximum, maximum)

        self.hidden_to_out.weight.data.clamp_(-maximum, maximum)
        self.hidden_to_out.bias.data.clamp_(-maximum, maximum)

    def eval(self, fen: str):
        fen = fen.strip()
        fen_split_spaces = fen.split(" ")
        features_tensor = torch.zeros(768, device=device)

        for rank_idx, rank in enumerate(fen_split_spaces[0].split('/')):
            file_idx = 0
            for char in rank:
                if char.isdigit():
                    file_idx += int(char)
                else:
                    square = 8 * (7 - rank_idx) + file_idx
                    is_black_piece = char.islower()
                    piece_type = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}[char.lower()]

                    features_tensor[is_black_piece * 384 + piece_type * 64 + square] = 1
                    
                    file_idx += 1
        
        assert fen_split_spaces[-5] in ["w", "b"]
        is_white_stm_tensor = torch.tensor(True if fen_split_spaces[-5] == "w" else False, device=device)

        return float(self.forward(features_tensor, is_white_stm_tensor))


