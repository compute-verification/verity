# P2 v1's GPU library (p2v1.cu): the tuned decode of ../p2_gpu/p2dec.cu with per-block v1 keys, and encode by two roots.
# CGBN = NVlabs/CGBN at ../p2_gpu/CGBN.commit; CGBN's host pass includes gmp.h (libgmp-dev's, or INC="-I <dir of a gmp.h>",
# such as gmpy2's wheel's, where nothing may be installed on the system).
# ARCH is the device's: sm_89 for the RTX 4090 and the L40S, sm_120 for the RTX PRO 6000.
# FAST selects the decode: p2sq.cuh's Karatsuba squaring and p2keys4.cuh's v4 key expansion (counter-mode SHA-512,
# reference.keys) by default; v1's SHAKE256 keys (-DP2_KEYS, p2keys.cuh) are retired, and gpu.Kernels refuses them.
NVCC ?= nvcc
CGBN ?= CGBN
BIN ?= bin
ARCH ?= sm_89
INC ?=
FAST ?= -DP2_FAST -DP2_KEYS4 -DP2SQ_KARA
FLAGS = -O3 -arch=$(ARCH) -std=c++17 -DP2_BITS=16448 -DP2_TPI=32 -DXMP_WMAD -DP2_PARTS=2 -DP2_SYM -DP2_TPB=64 -DP2_LIBRARY $(FAST)

all: $(BIN)/libp2v1.so

$(BIN)/libp2v1.so: p2v1.cu ../p2_gpu/p2dec.cu p2sq.cuh p2keys.cuh p2keys4.cuh ../../../verity/ml/kernels/sha512.cuh Makefile
	@mkdir -p $(BIN); $(NVCC) $(FLAGS) -Xcompiler -fPIC -shared $(INC) -I $(CGBN)/include -o $@ $< -Xptxas -v 2> $@.ptxas
