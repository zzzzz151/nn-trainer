# NN trainer for [Starzix](https://github.com/zzzzz151/Starzix) engine

Compile data converter:

```clang++ -O3 -std=c++20 -march=native -Wunused -Wall -Wextra dataloader/convert.cpp -o dataloader/convert[.exe]```

Convert txt data to binary format:

```./dataloader/convert[.exe] dataloader/data.txt```

Compile dataloader:

- Linux: ```clang++ -shared -O3 -std=c++20 -march=native -fPIC -Wunused -Wall -Wextra dataloader/dataloader.cpp -o dataloader/dataloader.so```

- Windows: ```clang++ -shared -O3 -std=c++20 -march=native -Wunused -Wall -Wextra dataloader/dataloader.cpp -o dataloader/dataloader.dll```

Set training parameters in train.py and run it

To quantize a net, set parameters in quantize.py and run it

