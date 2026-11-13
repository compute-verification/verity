"""P2 v4 (`verity/protocols/accounting/pous/verifier/README.md`, "P2") in the POUS-only harness.

- `harness.py`: P2's arm of the harness (`bench.py --impl p2-v1-gpu`); `memory_challenger.py`: its memory challenger process, k = 105.

The scheme's reference, wire and vectors are the verifier's (`verity/protocols/accounting/pous/verifier/p2_v1/`: `reference.py`,
`wire.py`, `vectors.py`). The GPU library, its wrapper, the encoded weights store and the store's load are P2's prover
(`verity/protocols/accounting/pous/prover/p2/`: the package `p2_store` with `p2v1.cu`, `gpu.py`, `store.py` and `load.py`, the
vLLM integration's decode at use, on `p2_gpu/p2dec.cu`).
"""
