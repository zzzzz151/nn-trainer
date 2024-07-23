CXXFLAGS_CONVERTER = -std=c++20 -march=native -O3 -Wunused -Wall -Wextra
CXXFLAGS_DATALOADER = $(CXXFLAGS_CONVERTER) -shared

ifeq ($(OS), Windows_NT)
	CLANG_PLUS_PLUS_18 = $(shell where clang++-18 > NUL 2>&1)
	CONVERTER_SUFFIX = .exe
	DATALOADER_SUFFIX = .dll
else
	CLANG_PLUS_PLUS_18 = $(shell command -v clang++-18 2>/dev/null)
	CONVERTER_SUFFIX =
	DATALOADER_SUFFIX = .so
	CXXFLAGS_DATALOADER += -fPIC
endif

ifeq ($(strip $(CLANG_PLUS_PLUS_18)),)
    COMPILER = clang++
else
    COMPILER = clang++-18
endif

all:
	$(COMPILER) $(CXXFLAGS_DATALOADER) dataloader.cpp -o dataloader$(DATALOADER_SUFFIX)