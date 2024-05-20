# NN trainer for [Starzix](https://github.com/zzzzz151/Starzix) engine

# Usage

Compile dataloader:

- Linux: ```clang++ -shared -O3 -std=c++20 -march=native -fPIC -Wunused -Wall -Wextra dataloader.cpp -o dataloader.so```

- Windows: ```clang++ -shared -O3 -std=c++20 -march=native -Wunused -Wall -Wextra dataloader.cpp -o dataloader.dll```

Set parameters in train.py and run train.py

A trained net can be quantized by setting the parameters in quantize.py and running quantize.py
