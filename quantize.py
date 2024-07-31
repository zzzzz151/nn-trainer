from settings import *
from model import *
import torch
import numpy as np
import struct
import warnings

QA = 255
QB = 64

if __name__ == "__main__":
    print(CHECKPOINT_TO_LOAD)
    print("Net arch: (768x2 -> {})x2 -> {}".format(HIDDEN_SIZE, OUTPUT_BUCKETS))
    print("QA, QB: {}, {}".format(QA, QB))

    assert CHECKPOINT_TO_LOAD != None and os.path.exists(CHECKPOINT_TO_LOAD)

    net = PerspectiveNet768x2().to(device)
    net = torch.compile(net)

    checkpoint = torch.load(CHECKPOINT_TO_LOAD, weights_only=False)
    net.load_state_dict(checkpoint["model"])

    def quantized(x, quant_factor):
        return np.round(x.detach().cpu().numpy() * float(quant_factor)).astype(np.int16)

    out_file_name = CHECKPOINT_TO_LOAD[:-3] + ".bin"

    # Save to binary file
    with open(out_file_name, "wb") as bin_file:
        # Feature weights
        bin_file.write(np.stack(
            (quantized(net.features_to_hidden_white_stm.weight.data.T, QA), 
            quantized(net.features_to_hidden_black_stm.weight.data.T, QA)),
            axis=0)
            .tobytes()
        )

        # Hidden biases
        bin_file.write(np.stack(
            (quantized(net.features_to_hidden_white_stm.bias.data, QA), 
            quantized(net.features_to_hidden_black_stm.bias.data, QA)), 
            axis=0)
            .tobytes()
        )

        # Output weights
        bin_file.write(
            quantized(net.hidden_to_out.weight.data, QB)
            .reshape((OUTPUT_BUCKETS, 2, HIDDEN_SIZE))
            .tobytes()
        )

        # Output biases
        bin_file.write(quantized(net.hidden_to_out.bias.data, QA * QB).tobytes())

    print(out_file_name)

    # Load quantized net into pytorch

    def dequantized(x, quant_factor):
        return x.astype(np.float32) / float(quant_factor)

    net = PerspectiveNet768x2().to(device)

    with open(out_file_name, "rb") as bin_file:
        # Feature weights

        feature_weights = np.frombuffer(bin_file.read(2 * 768 * HIDDEN_SIZE * 2), dtype=np.int16).reshape(2, 768, HIDDEN_SIZE)

        net.features_to_hidden_white_stm.weight.data = torch.tensor(
            dequantized(feature_weights[0].T, QA), dtype=torch.float32, device=device
        )

        net.features_to_hidden_black_stm.weight.data = torch.tensor(
            dequantized(feature_weights[1].T, QA), dtype=torch.float32, device=device
        )
        
        # Hidden biases

        hidden_biases = np.frombuffer(bin_file.read(2 * HIDDEN_SIZE * 2), dtype=np.int16).reshape(2, HIDDEN_SIZE)

        net.features_to_hidden_white_stm.bias.data = torch.tensor(
            dequantized(hidden_biases[0], QA), dtype=torch.float32, device=device
        )
            
        net.features_to_hidden_black_stm.bias.data = torch.tensor(
            dequantized(hidden_biases[1], QA), dtype=torch.float32, device=device
        )
        
        # Output weights
        output_weights = np.frombuffer(bin_file.read(OUTPUT_BUCKETS * 2 * HIDDEN_SIZE * 2), dtype=np.int16)
        output_weights = dequantized(output_weights.reshape(OUTPUT_BUCKETS, HIDDEN_SIZE * 2), QB)
        net.hidden_to_out.weight.data = torch.tensor(output_weights, dtype=torch.float32, device=device)
        
        # Output biases
        output_biases = np.frombuffer(bin_file.read(OUTPUT_BUCKETS * 2), dtype=np.int16)
        net.hidden_to_out.bias.data = torch.tensor(dequantized(output_biases, QA * QB), dtype=torch.float32, device=device)

    # Print some evals with quantized net
    
    def eval_quantized(fen: str):
        return int(net.eval(fen) * float(SCALE))

    print()
    print("Start pos eval:", eval_quantized("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"))
    print("e2e4 eval:", eval_quantized("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"))
    print("e2e4 g8f6 eval:", eval_quantized("rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2"))
    print("Bongcloud eval:", eval_quantized("rnbkqbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBKQBNR w KQkq - 0 1"))

    # Moves in "r3k2b/6P1/8/1pP5/8/8/4P3/4K2R w Kq b6 0 1"
    FENS = [
        "r3k2b/6P1/1P6/8/8/8/4P3/4K2R b Kq - 0 1",
        "r3k2b/6P1/8/1pP5/8/4P3/8/4K2R b Kq - 0 1",
        "r3k2b/6P1/8/1pP5/4P3/8/8/4K2R b Kq e3 0 1",
        "r3k2b/6P1/2P5/1p6/8/8/4P3/4K2R b Kq - 0 1",
        "r3k2Q/8/8/1pP5/8/8/4P3/4K2R b Kq - 0 1",
        "r3k2R/8/8/1pP5/8/8/4P3/4K2R b Kq - 0 1",
        "r3k2B/8/8/1pP5/8/8/4P3/4K2R b Kq - 0 1",
        "r3k2N/8/8/1pP5/8/8/4P3/4K2R b Kq - 0 1",
        "r3k1Qb/8/8/1pP5/8/8/4P3/4K2R b Kq - 0 1",
        "r3k1Rb/8/8/1pP5/8/8/4P3/4K2R b Kq - 0 1",
        "r3k1Bb/8/8/1pP5/8/8/4P3/4K2R b Kq - 0 1",
        "r3k1Nb/8/8/1pP5/8/8/4P3/4K2R b Kq - 0 1",
        "r3k2b/6P1/8/1pP5/8/8/4P3/3K3R b q - 1 1",
        "r3k2b/6P1/8/1pP5/8/8/4P3/5K1R b q - 1 1",
        "r3k2b/6P1/8/1pP5/8/8/3KP3/7R b q - 1 1",
        "r3k2b/6P1/8/1pP5/8/8/4PK2/7R b q - 1 1",
        "r3k2b/6P1/8/1pP5/8/8/4P3/5RK1 b q - 1 1",
        "r3k2b/6P1/8/1pP5/8/8/4P3/4KR2 b q - 1 1",
        "r3k2b/6P1/8/1pP5/8/8/4P3/4K1R1 b q - 1 1",
        "r3k2b/6P1/8/1pP5/8/8/4P2R/4K3 b q - 1 1",
        "r3k2b/6P1/8/1pP5/8/7R/4P3/4K3 b q - 1 1",
        "r3k2b/6P1/8/1pP5/7R/8/4P3/4K3 b q - 1 1",
        "r3k2b/6P1/8/1pP4R/8/8/4P3/4K3 b q - 1 1",
        "r3k2b/6P1/7R/1pP5/8/8/4P3/4K3 b q - 1 1",
        "r3k2b/6PR/8/1pP5/8/8/4P3/4K3 b q - 1 1",
        "r3k2R/6P1/8/1pP5/8/8/4P3/4K3 b q - 0 1"
    ]

    for fen in FENS:
        print("{ " + "\"{}\", {}".format(fen, eval_quantized(fen)) + " },")




