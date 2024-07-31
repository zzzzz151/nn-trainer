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
    print("Net arch: (768x2x{} -> {})x2 -> {}, horizontally mirrored".format(INPUT_BUCKETS, HIDDEN_SIZE, OUTPUT_BUCKETS))
    print("QA, QB: {}, {}".format(QA, QB))

    assert CHECKPOINT_TO_LOAD != None and os.path.exists(CHECKPOINT_TO_LOAD)

    net = PerspectiveNet768x2()
    net = torch.compile(net)

    checkpoint = torch.load(CHECKPOINT_TO_LOAD, weights_only=False)
    net.load_state_dict(checkpoint["model"])

    # Write quantized weights and biases to binary file

    out_file_name = CHECKPOINT_TO_LOAD[:-3] + ".bin"
    out_file = open(out_file_name, 'wb')

    def write_quantized(weights_or_biases, quantization_factor):
        quantized = np.round(weights_or_biases.detach().numpy() * quantization_factor)
        flattened = quantized.astype(np.int16).flatten().tolist()
        out_file.write(struct.pack('<' + 'h' * len(flattened), *flattened))

    # Write features weights
    write_quantized(net.features_to_hidden_white_stm.weight.T, QA)
    write_quantized(net.features_to_hidden_black_stm.weight.T, QA)

    # Write features biases
    write_quantized(net.features_to_hidden_white_stm.bias, QA)
    write_quantized(net.features_to_hidden_black_stm.bias, QA)

    # Write output weights and output biases
    write_quantized(net.hidden_to_out.weight, QB)
    write_quantized(net.hidden_to_out.bias, QA * QB)

    out_file.close()
    print(out_file_name)

    # Load quantized weights and biases to pytorch net

    net = PerspectiveNet768x2().to(device)

    with open(out_file_name, "rb") as bin_file:
        # Read quantized features weights
        features_weights_white_stm = struct.unpack(f'<{768 * INPUT_BUCKETS * HIDDEN_SIZE}h', bin_file.read(768 * INPUT_BUCKETS * HIDDEN_SIZE * 2))
        features_weights_black_stm = struct.unpack(f'<{768 * INPUT_BUCKETS * HIDDEN_SIZE}h', bin_file.read(768 * INPUT_BUCKETS * HIDDEN_SIZE * 2))

        # Read quantized features biases
        features_biases_white_stm = struct.unpack(f'<{HIDDEN_SIZE}h', bin_file.read(HIDDEN_SIZE * 2))
        features_biases_black_stm = struct.unpack(f'<{HIDDEN_SIZE}h', bin_file.read(HIDDEN_SIZE * 2))

        # Read quantized output weights and output biases
        output_weights = struct.unpack(f'<{2 * HIDDEN_SIZE}h', bin_file.read(2 * HIDDEN_SIZE * 2))
        output_bias = struct.unpack('<1h', bin_file.read(1 * 2))
    
    # Move quantized features weights to the net

    net.features_to_hidden_white_stm.weight.data = torch.tensor(
        np.array(features_weights_white_stm).reshape(768 * INPUT_BUCKETS, HIDDEN_SIZE).T / QA, dtype=torch.float32, device=device)

    net.features_to_hidden_black_stm.weight.data = torch.tensor(
        np.array(features_weights_black_stm).reshape(768 * INPUT_BUCKETS, HIDDEN_SIZE).T / QA, dtype=torch.float32, device=device)

    # Move quantized features biases to the net

    net.features_to_hidden_white_stm.bias.data = torch.tensor(
        np.array(features_biases_white_stm) / QA, dtype=torch.float32, device=device)

    net.features_to_hidden_black_stm.bias.data = torch.tensor(
        np.array(features_biases_black_stm) / QA, dtype=torch.float32, device=device)

    # Move quantized output weights and biases to the net
    
    net.hidden_to_out.weight.data = torch.tensor(
        np.array(output_weights).reshape(1, 2 * HIDDEN_SIZE) / QB, dtype=torch.float32, device=device)

    net.hidden_to_out.bias.data = torch.tensor(
        np.array(output_bias) / (QA * QB), dtype=torch.float32, device=device)

    # Print some evals with quantized net
    
    def eval_quantized(fen: str):
        return int(net.eval(fen) * float(SCALE))

    print()
    print("Start pos eval:", eval_quantized("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"))
    print("e2e4 eval:", eval_quantized("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"))
    print("e2e4 g8f6 eval:", eval_quantized("rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2"))
    print("Bongcloud eval:", eval_quantized("rnbkqbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBKQBNR w KQkq - 0 1"))

    FENS = [
        # Moves in "r3k2b/6P1/8/1pP5/8/8/4P3/4K2R w Kq b6 0 1"
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
        "r3k2R/6P1/8/1pP5/8/8/4P3/4K3 b q - 0 1",

        # Some more positions
        "4k3/7n/5q2/8/8/6R1/8/4K3 w - - 0 1",
        "4k3/7n/1q6/8/8/6R1/8/4K3 w - - 0 1",
        "4k3/7n/1q6/8/8/6R1/8/2K5 w - - 0 1",
        "4k3/7n/5q2/8/8/6R1/8/2K5 w - - 0 1",
        "q3k3/7n/5q2/8/8/6R1/8/2K5 w - - 0 1",
        "4k3/7n/8/2q5/8/6R1/7q/5K2 w - - 0 1",
        "4k3/7n/8/8/8/6R1/8/5K2 w - - 0 1",
        "4k3/7n/8/8/8/6R1/8/1K6 w - - 0 1"
    ]

    for fen in FENS:
        print("{ " + "\"{}\", {}".format(fen, eval_quantized(fen)) + " },")




