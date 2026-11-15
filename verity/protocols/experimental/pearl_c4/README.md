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
- `prover/pearl_c4.py`, `prover/pearl_c4_c_L.py`: the scheme's scalar reference, a PoUW scheme. Importing this package
  registers it with PoUW's registry (`schemes.register`) under `pearl-c-nvfp4-v0`, which raises
  `NotImplementedError("compute-accounting/pearl-c4-experimental")` until then: PoUW loads nothing of `experimental/`.
- `prover/pc4.py`: its tile check as a circuit, on PoUW's shared tile-check parts (`circuit.tile_units`), and
  `prover/pc4_rowk.py` its row units cut along k on `circuit.rowk`'s cut, each with its circuit-check sidecar.
- `tests/`: the twin, the fixture, the gate and `f1_fast` against the scalar reference, the scheme's pinned vectors
  (`pearl_c4_vectors.py`, `pearl_c4_regions_vectors.py`, their JSON in `tests/vectors/`) and the scheme's and circuit's
  tests, as a suite of their own. They extend PoUW's tests, whose helpers they import by name (`conftest.py`).

## Where the rest is

- PoUW keeps what its own circuits share with `pc4`: the row schemas and domain strings of `circuit.leaves` (pc4-head,
  pc4-carry, nvfp4-codes), the word primitives of `circuit.words` and their Boolean versions, which are registered
  Definitions the Lean row model is held to, and `circuit.rowk`'s generic ports.
- The shipping (`benchmarks/pouw/pearl_c4/ship.sh`), the reference replay of a device run (`replay.py` beside it) and
  the replay on a model's captures (`real.py`, `real_table.py`) are measurements, and stay under `benchmarks/`.
- Its Lean is in PoUW's areas of `security/`: `Definitions/Pouw/PearlC/` (the files named `*Fp4*`, `Row4Circuit`,
  `Tile4Circuit`), `Definitions/Pouw/Assumptions/PearlC/TTOutFp4*` and `Proofs/Pouw/PearlC/` (the files named `*Fp4*`,
  `Row4Words`, `Tile4Circuit`).
