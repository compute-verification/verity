# P2's GPU decode at 16,448 bits (the efficient-cryptography campaign's tuned build of record: two parts, the cross term as
# lo*hi doubled, TPI 32, 64-thread blocks; 24.16 GB/s in r20260928-065407-16f0), with the shared key (M2, as measured
# there) and with fresh per-block keys read from device memory (M1: verity_pous.schemes.p2's key schedule).
# CGBN = NVlabs/CGBN at CGBN.commit; the build needs libgmp-dev (CGBN's host pass includes gmp.h).
NVCC ?= nvcc
CGBN ?= CGBN
BIN ?= bin
FLAGS = -O3 -arch=sm_89 -std=c++14 -DP2_BITS=16448 -DP2_TPI=32 -DXMP_WMAD -DP2_PARTS=2 -DP2_SYM -DP2_TPB=64

all: $(BIN)/p2dec_16448_32_s2 $(BIN)/p2dec_16448_32_s2_keys

$(BIN)/p2dec_16448_32_s2: p2dec.cu
	@mkdir -p $(BIN); $(NVCC) $(FLAGS) -I $(CGBN)/include -o $@ $< -Xptxas -v 2> $@.ptxas

$(BIN)/p2dec_16448_32_s2_keys: p2dec.cu
	@mkdir -p $(BIN); $(NVCC) $(FLAGS) -DP2_KEYS -I $(CGBN)/include -o $@ $< -Xptxas -v 2> $@.ptxas
