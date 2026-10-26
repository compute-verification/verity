"""One band-fill wipe of a GPU's memory under `Pous.Guarantees.SecureErasureMeetsFamily`, its memory challenger on another
machine.

- `prover.py`: on the GPU node. Fills the device with the band's `C` under the memory challenger's fresh salt (`W = 0`, so
  `C_v = key_v`), with `pous_native.cu`'s `band_chain` kernel, and serves its blocks with the native responder; or, as the
  cheat, keeps foreign bytes in every 19th block and recomputes a challenged one from its kept parents.
- `challenger.py`: on the auditor's machine. Draws the salt, runs the memory challenger's signed sessions
  (`verity.core.protocols.pous.memory_challenger`) over TCP to the prover, and judges them against `C`'s blocks recomputed by
  `verity.core.protocols.pous`'s reference encoder.
- `wire.py`: what the two share. `run.sh`: both as research runs.
"""
