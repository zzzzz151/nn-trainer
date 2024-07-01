# NN trainer for [Starzix](https://github.com/zzzzz151/Starzix) engine

Have clang++ installed and run `make` to compile data converter and data loader

Convert txt data in the format `fen | score | result`, where `score` is from white pov and `result` is `0.0` if white lost, `0.5` if draw, `1.0` if white won, to binary format used by the trainer:

`dataloader/convert[.exe] dataloader/data.txt`

Set training parameters in train.py and run it

To quantize a net, set parameters in quantize.py and run it

