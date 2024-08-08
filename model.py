from settings import *
import torch

class PerspectiveNet768x2(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.features_to_hidden_white_stm = torch.nn.Linear(768 * INPUT_BUCKETS, HIDDEN_SIZE)
        self.features_to_hidden_black_stm = torch.nn.Linear(768 * INPUT_BUCKETS, HIDDEN_SIZE)
        self.hidden_to_out = torch.nn.Linear(HIDDEN_SIZE * 2, OUTPUT_BUCKETS)

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
    def forward(self, 
        features_white_stm_tensor, 
        features_black_stm_tensor, 
        is_white_stm_tensor, 
        output_buckets_tensor):

        white_hidden = self.features_to_hidden_white_stm(features_white_stm_tensor)
        black_hidden = self.features_to_hidden_black_stm(features_black_stm_tensor)

        assert len(features_white_stm_tensor.size()) == len(features_black_stm_tensor.size())
        dim = len(features_white_stm_tensor.size()) - 1

        # stm accumulator first
        hidden_layer  =  is_white_stm_tensor * torch.cat([white_hidden, black_hidden], dim=dim)
        hidden_layer += ~is_white_stm_tensor * torch.cat([black_hidden, white_hidden], dim=dim)

        # SCReLU activation
        hidden_layer = torch.pow(torch.clamp(hidden_layer, 0, 1), 2) 

        if OUTPUT_BUCKETS > 1:
            return torch.gather(self.hidden_to_out(hidden_layer), dim, output_buckets_tensor.long())

        return self.hidden_to_out(hidden_layer)

    def clamp_weights_biases(self):
        self.features_to_hidden_white_stm.weight.data.clamp_(-MAX_WEIGHT_BIAS, MAX_WEIGHT_BIAS)
        self.features_to_hidden_white_stm.bias.data.clamp_(-MAX_WEIGHT_BIAS, MAX_WEIGHT_BIAS)

        self.features_to_hidden_black_stm.weight.data.clamp_(-MAX_WEIGHT_BIAS, MAX_WEIGHT_BIAS)
        self.features_to_hidden_black_stm.bias.data.clamp_(-MAX_WEIGHT_BIAS, MAX_WEIGHT_BIAS)

        self.hidden_to_out.weight.data.clamp_(-MAX_WEIGHT_BIAS, MAX_WEIGHT_BIAS)
        self.hidden_to_out.bias.data.clamp_(-MAX_WEIGHT_BIAS, MAX_WEIGHT_BIAS)

    def eval(self, fen: str):
        fen = fen.strip()
        fen_split_spaces = fen.split(" ")

        stm = fen_split_spaces[-5]
        assert stm in ["w", "b"]

        # [is_black_piece]
        kings_squares = [None, None] 
        queens_squares = [64, 64]

        WHITE = 0
        BLACK = 1

        for rank_idx, rank in enumerate(fen_split_spaces[0].split('/')):
            file_idx = 0
            for char in rank:
                if char.isdigit():
                    file_idx += int(char)
                else:
                    square = 8 * (7 - rank_idx) + file_idx
                    this_xor = 7 if square % 8 > 3 else 0

                    if char == 'K':
                        kings_squares[WHITE] = square
                    elif char == 'k':
                        kings_squares[BLACK] = square
                    elif char == 'Q':
                        if queens_squares[WHITE] != 64:
                            queens_squares[WHITE] = 64
                        else:
                            queens_squares[WHITE] = square
                    elif char == 'q':
                        if queens_squares[BLACK] != 64:
                            queens_squares[BLACK] = 64
                        else:
                            queens_squares[BLACK] = square

                    file_idx += 1

        assert None not in kings_squares

        def feature(color: int, is_black_piece: int, piece_type: int, square: int):
            enemy_queen_square = queens_squares[1 - color]

            if kings_squares[color] % 8 > 3:
                square ^= 7

                if enemy_queen_square != 64:
                    enemy_queen_square ^= 7

            return INPUT_BUCKETS_MAP[enemy_queen_square] * 768 + is_black_piece * 384 + piece_type * 64 + square

        features_white_stm_tensor = torch.zeros(768 * INPUT_BUCKETS, device=device)
        features_black_stm_tensor = torch.zeros(768 * INPUT_BUCKETS, device=device)
        num_pieces = 0

        for rank_idx, rank in enumerate(fen_split_spaces[0].split('/')):
            file_idx = 0
            for char in rank:
                if char.isdigit():
                    file_idx += int(char)
                else:
                    square = 8 * (7 - rank_idx) + file_idx
                    is_black_piece = char.islower()
                    piece_type = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}[char.lower()]

                    features_white_stm_tensor[feature(WHITE, is_black_piece, piece_type, square)] = 1
                    features_black_stm_tensor[feature(BLACK, is_black_piece, piece_type, square)] = 1

                    num_pieces += 1
                    file_idx += 1
        
        assert num_pieces >= 2 and num_pieces <= 32
        
        output_bucket_idx = int((num_pieces - 1) / (32 / OUTPUT_BUCKETS))

        return float(self.forward(
            features_white_stm_tensor,
            features_black_stm_tensor, 
            torch.tensor(stm == "w", device=device), 
            torch.tensor(output_bucket_idx, device=device)
        ))


