# pearl_c4: Pearl-C on FP4 operands, experimental

Pearl-C4 (`pearl-c-nvfp4-v0`, with `pearl-c-mxfp4-v0` as a contrast line) is PoUW's Pearl-C with the FP8 product
replaced by the block-scaled FP4 one of sm_120 (`mma.sync kind::mxf4nvf4`): the noise enters inside one FP4 atom per
element, one noisy cast forms each operand, and the BF16 clean-up is checked as in Pearl-C. It is a protocol being
tried: no demo lists it, and its guarantee rests on a premise that is stated and not proved.

## What is here

- `prover/kernels/`: the sm_120a kernels (`pearl_c4.cu`), their build (`build.sh`, which refuses a cubin unless the
  offline gate `sass.py` passes), the card-side runner (`run.sh` and `dev.py`, which `build.sh` ships beside the cubin),
  the numpy twin of one tile (`twin.py`) and the device check's fixture (`fixture.py`, built from the twin).
- `prover/f1_fast.py`: a vectorised evaluation of the scheme's F1′ item, which a caller may pass to the scheme in place
  of its pure-Python rule.
- `tests/`: the twin, the fixture, the gate and `f1_fast` against the scalar reference, as a suite of their own.

## Where the rest is

- The scheme's scalar reference (`verity/protocols/accounting/pouw/prover/schemes/pearl_c4.py`, `pearl_c4_c_L.py`),
  its tile check as a circuit (`prover/circuit/pc4.py`), its pinned vectors and their tests stay in PoUW's package:
  PoUW's own circuits load `pc4`, which loads the scheme, and PoUW loads nothing of `experimental/`.
- The shipping, the reference replay of a device run and the replay on a model's captures are measurements, under
  `benchmarks/pouw/pearl_c4/`.
- Its Lean is in PoUW's areas of `security/`: `Definitions/Pouw/PearlC/` (the files named `*Fp4*`, `Row4Circuit`,
  `Tile4Circuit`), `Definitions/Pouw/Assumptions/PearlC/TTOutFp4*` and `Proofs/Pouw/PearlC/` (the files named `*Fp4*`,
  `Row4Words`, `Tile4Circuit`).
