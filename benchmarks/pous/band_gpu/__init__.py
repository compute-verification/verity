"""The band and dense schemes' GPU kernels and the POUS-only harness that measures them (`bench.py --impl band-gpu`).

- `pous_native.cu`, `native.py`: the batched SHAKE256 and Feistel kernels and the audit responder, JIT-built by torch.
- `codec.py`, `dense.py`: the GPU twins (`dense.encode`, `dense.decode_segments`, `dense.decode_blocks`) and the kernels'
  dispatch switches; moved from the vLLM tree's `engine/pous`, with no vLLM dependency.
- `harness.py`: encode, `C` resident, the synthetic decode loop, the timed audits, the step floor.
- `memory_challenger.py`: the memory challenger process of the timed audit (the responder's wire client and `verity.core.protocols.pous.protocol.TimedAudit`).
"""
