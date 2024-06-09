import torch
import numpy as np
import struct
import warnings
from model import *

warnings.filterwarnings("ignore")
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

PT_FILE_NAME = "nets/net768x2-400.pt"
net = PerspectiveNet768x2(hidden_size=1024).to(device)

QA = 255
QB = 64

if __name__ == "__main__":
    print(PT_FILE_NAME)
    print("Net arch: (768 -> {})x2 -> 1".format(net.HIDDEN_SIZE))
    print("QA, QB: {}, {}".format(QA, QB))

    QA = float(QA)
    QB = float(QB)

    net.load_state_dict(torch.load(PT_FILE_NAME))

    # Extract weights and biases
    weights1 = net.features_to_hidden_white_stm.weight.detach().cpu().numpy()
    weights2 = net.features_to_hidden_black_stm.weight.detach().cpu().numpy()
    biases1 = net.features_to_hidden_white_stm.bias.detach().cpu().numpy()
    biases2 = net.features_to_hidden_black_stm.bias.detach().cpu().numpy()
    weights3 = net.hidden_to_out.weight.detach().cpu().numpy()
    bias3 = net.hidden_to_out.bias.detach().cpu().numpy()

    # Quantize weights and biases
    weights1_quantized = np.round(weights1 * QA).T.astype(np.int16)
    weights2_quantized = np.round(weights2 * QA).T.astype(np.int16)
    biases1_quantized = np.round(biases1 * QA).astype(np.int16)
    biases2_quantized = np.round(biases2 * QA).astype(np.int16)
    weights3_quantized = np.round(weights3 * QB).astype(np.int16)
    bias3_quantized = np.round(bias3 * QA * QB).astype(np.int16)

    # Flatten to 1D lists
    weights1_1d = weights1_quantized.flatten().tolist()
    weights2_1d = weights2_quantized.flatten().tolist()
    biases1_1d = biases1_quantized.flatten().tolist()
    biases2_1d = biases2_quantized.flatten().tolist()
    weights3_1d = weights3_quantized.flatten().tolist()
    bias3_1d = bias3_quantized.flatten().tolist()

    out_file_name = PT_FILE_NAME[:-3] + ".bin"

    # Save to binary file
    with open(out_file_name, "wb") as bin_file:
        bin_file.write(struct.pack('<' + 'h' * len(weights1_1d), *weights1_1d))
        bin_file.write(struct.pack('<' + 'h' * len(weights2_1d), *weights2_1d))
        bin_file.write(struct.pack('<' + 'h' * len(biases1_1d), *biases1_1d))
        bin_file.write(struct.pack('<' + 'h' * len(biases2_1d), *biases2_1d))
        bin_file.write(struct.pack('<' + 'h' * len(weights3_1d), *weights3_1d))
        bin_file.write(struct.pack('<' + 'h' * len(bias3_1d), *bias3_1d))

    print(out_file_name)

    # Print eval quantized

    with open(out_file_name, "rb") as bin_file:
        weights1 = struct.unpack(f'<{768 * net.HIDDEN_SIZE}h', bin_file.read(768 * net.HIDDEN_SIZE * 2))
        weights2 = struct.unpack(f'<{768 * net.HIDDEN_SIZE}h', bin_file.read(768 * net.HIDDEN_SIZE * 2))
        biases1 = struct.unpack(f'<{net.HIDDEN_SIZE}h', bin_file.read(net.HIDDEN_SIZE * 2))
        biases2 = struct.unpack(f'<{net.HIDDEN_SIZE}h', bin_file.read(net.HIDDEN_SIZE * 2))
        weights3 = struct.unpack(f'<{2 * net.HIDDEN_SIZE}h', bin_file.read(2 * net.HIDDEN_SIZE * 2))
        bias3 = struct.unpack('<1h', bin_file.read(1 * 2))

    net.features_to_hidden_white_stm.weight.data = torch.tensor(np.array(weights1).reshape(768, net.HIDDEN_SIZE).T / QA, dtype=torch.float32, device=device)
    net.features_to_hidden_black_stm.weight.data = torch.tensor(np.array(weights2).reshape(768, net.HIDDEN_SIZE).T / QA, dtype=torch.float32, device=device)
    net.features_to_hidden_white_stm.bias.data = torch.tensor(np.array(biases1) / QA, dtype=torch.float32, device=device)
    net.features_to_hidden_black_stm.bias.data = torch.tensor(np.array(biases2) / QA, dtype=torch.float32, device=device)
    net.hidden_to_out.weight.data = torch.tensor(np.array(weights3).reshape(1, 2 * net.HIDDEN_SIZE) / QB, dtype=torch.float32, device=device)
    net.hidden_to_out.bias.data = torch.tensor(np.array(bias3) / (QA * QB), dtype=torch.float32, device=device)

    # Print some evals

    print()
    print("Start pos eval:", int(net.eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") * 400))
    print("e2e4 eval:", int(net.eval("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1") * 400))
    print("e2e4 g8f6 eval:", int(net.eval("rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2") * 400))

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
        print("{}: {}".format(fen, int(net.eval(fen) * 400)))




