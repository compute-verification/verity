"""P2 v1 (the POUS store's `docs/efficient-crypto/p2-spec-draft.md`, frozen 29 Sep 2026) in the POUS-only harness.

- `reference.py`: the scheme on the CPU, pinned by `vectors.json` (the P2 kit's `p2-test-vectors/v1`).
- `p2v1.cu`, `Makefile`: the GPU library (decode with v1 keys on `../p2_gpu/p2dec.cu`'s tuned squaring, encode by two roots).
- `gpu.py`: the library through ctypes, with the keys expanded on the device by SHAKE256 with the v1 framing.
- `harness.py`: P2's arm of the harness (`bench.py --impl p2-v1-gpu`); `memory_challenger.py`: its memory challenger process, k = 105.
"""
