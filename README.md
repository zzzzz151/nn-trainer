# nn-trainer

# Compile dataloader

**Linux**

```clang++ -shared -O3 -std=c++20 -march=x86-64-v3 -fPIC -Wunused -Wall -Wextra dataloader/dataloader.cpp -o dataloader/dataloader.so```

**Windows**

```clang++ -shared -O3 -std=c++20 -march=x86-64-v3 -Wunused -Wall -Wextra dataloader/dataloader.cpp -o dataloader/dataloader.dll```