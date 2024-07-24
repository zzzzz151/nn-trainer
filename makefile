CXX ?= clang++
CXXFLAGS = -std=c++20 -march=native -O3 -Wunused -Wall -Wextra -shared
SUFFIX = .dll

ifneq ($(OS), Windows_NT)
	SUFFIX = .so
	CXXFLAGS += -fPIC
endif

all:
	$(CXX) $(CXXFLAGS) dataloader.cpp -o dataloader$(SUFFIX)