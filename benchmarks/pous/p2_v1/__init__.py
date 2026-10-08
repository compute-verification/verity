"""P2 (`verity/core/protocols/pous/p2/PROTOCOL.md`, specification v4) in the POUS-only harness.

- `reference.py`: the scheme on the CPU, pinned by `vectors.json`.
- `wire.py`: `p2-audit-wire/v4`'s frames and its reference verifier, pinned by `audit-vectors.json` and
  `audit-negative-vectors.json`; `vectors.py` writes all three files (and the package's keys fixture) in one command.
- `p2v1.cu`, `Makefile`: the GPU library (decode on `../p2_gpu/p2dec.cu`'s tuned squaring, encode by two roots).
- `gpu.py`: the library through ctypes, with the keys expanded on the device.
- `harness.py`: P2's arm of the harness (`bench.py --impl p2-v1-gpu`); `memory_challenger.py`: its memory challenger process, k = 105.
"""
