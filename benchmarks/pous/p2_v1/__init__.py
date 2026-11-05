"""P2 v4 (`verity/core/protocols/pous/README.md`, "P2") in the POUS-only harness.

- The scheme on the CPU, its audit wire `p2-audit-wire/v4` and the vectors that pin both are PoUS's
  (`verity.core.protocols.pous.p2`), which the arm is gated against.
- `p2v1.cu`, `Makefile`: the GPU library (decode on `../p2_gpu/p2dec.cu`'s tuned squaring, encode by two roots).
- `gpu.py`: the library through ctypes, with the keys expanded on the device.
- `harness.py`: P2's arm of the harness (`bench.py --impl p2-v1-gpu`); `memory_challenger.py`: its memory challenger process, k = 105.
- `store.py`: the encoded weights store's layout (the MQA16K INT8 weights block-aligned in W; `store.py layout` writes
  its JSON) and the reference `decode_layer` a decode-to-staging kernel matches; `store-toy/` is its toy fixture.
"""
