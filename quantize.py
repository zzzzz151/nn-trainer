import torch
import numpy as np
import struct
import warnings
from model import PerspectiveNet

warnings.filterwarnings("ignore")

PT_FILE_NAME = "nets/net1024-400.pt"
HIDDEN_SIZE = 1024
QA = 255
QB = 64

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":
    print(PT_FILE_NAME)
    print("Hidden layer size:", HIDDEN_SIZE)
    print("QA, QB: {}, {}".format(QA, QB))

    QA = float(QA)
    QB = float(QB)

    net = PerspectiveNet(HIDDEN_SIZE, 0).to(device)
    net.load_state_dict(torch.load(PT_FILE_NAME))

    # Extract weights and biases
    weights1 = net.conn1.weight.detach().cpu().numpy()
    bias1 = net.conn1.bias.detach().cpu().numpy()
    weights2 = net.conn2.weight.detach().cpu().numpy()
    bias2 = net.conn2.bias.detach().cpu().numpy()

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

    out_file_name = PT_FILE_NAME[:-3] + ".nnue"

    # Save to binary file
    with open(out_file_name, "wb") as bin_file:
        bin_file.write(struct.pack('<' + 'h' * len(weights1_1d), *weights1_1d))
        bin_file.write(struct.pack('<' + 'h' * len(bias1_1d), *bias1_1d))
        bin_file.write(struct.pack('<' + 'h' * len(weights2_1d), *weights2_1d))
        bin_file.write(struct.pack('<' + 'h' * len(bias2_1d), *bias2_1d))

    print(out_file_name)

    # Print eval quantized

    with open(out_file_name, "rb") as bin_file:
        weights1 = struct.unpack(f'<{768 * HIDDEN_SIZE}h', bin_file.read(768 * HIDDEN_SIZE * 2))
        bias1 = struct.unpack(f'<{HIDDEN_SIZE}h', bin_file.read(HIDDEN_SIZE * 2))
        weights2 = struct.unpack(f'<{2 * HIDDEN_SIZE}h', bin_file.read(2 * HIDDEN_SIZE * 2))
        bias2 = struct.unpack('<1h', bin_file.read(1 * 2))

    net.conn1.weight.data = torch.tensor(np.array(weights1).reshape(768, HIDDEN_SIZE).T / QA, dtype=torch.float32, device=device)
    net.conn1.bias.data = torch.tensor(np.array(bias1) / QA, dtype=torch.float32, device=device)
    net.conn2.weight.data = torch.tensor(np.array(weights2).reshape(1, 2 * HIDDEN_SIZE) / QB, dtype=torch.float32, device=device)
    net.conn2.bias.data = torch.tensor(np.array(bias2) / (QA * QB), dtype=torch.float32, device=device)

    print("Start pos eval:", int(net.eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") * 400.0))
    print("e2e4 eval:", int(net.eval("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1") * 400.0))



