# pearl_c4: Pearl-C on FP4 operands, experimental

Pearl-C4 (`pearl-c-nvfp4-v0`, with `pearl-c-mxfp4-v0` as a contrast line) is PoUW's Pearl-C with the FP8 product
replaced by the block-scaled FP4 one of sm_120 (`mma.sync kind::mxf4nvf4`): the noise enters inside one FP4 atom per
element, one noisy cast forms each operand, and the BF16 clean-up is checked as in Pearl-C. It is research only: it
has no kernel, no application lists it, and its guarantee rests on a premise that is stated and not proved. Its tile
check stays because `benchmarks/pouw/hidden_zk/stage.py` stages its proof blocks for Flock, which the ZK server's
controls (`verity/protocols/proof/flock/prover/agreement/zk_server_controls.sh`) and Flock's tests prove.

## What is here

- `prover/pearl_c4.py`, `prover/pearl_c4_c_L.py`: the scheme's scalar reference, a PoUW scheme. Importing this package
  registers its builder with PoUW's (`schemes.register`, family `pearl-c4`); PoUW's kernel schemes name
  `pearl-c-nvfp4-v0` and raise `NotImplementedError` naming this package until then: PoUW loads nothing of
  `experimental/`.
- `prover/pc4.py`: its tile check as a circuit, on PoUW's shared tile-check parts (`circuit.tile_units`), and
  `prover/pc4_rowk.py` its row proof blocks cut along k on `circuit.rowk`'s cut, each with its circuit-check sidecar.
- `tests/`: the scheme's pinned vectors
  (`pearl_c4_vectors.py`, `pearl_c4_regions_vectors.py`, their JSON in `tests/vectors/`) and the scheme's and circuit's
  tests, as a suite of their own. They extend PoUW's tests, whose helpers they import by name (`conftest.py`).

## Where the rest is

- PoUW keeps what its own circuits share with `pc4`: the row schemas and domain strings of `circuit.leaves` (pc4-head,
  pc4-carry, nvfp4-codes), the word primitives of `circuit.words` and their Boolean versions, which are registered
  Definitions the Lean row model is held to, and `circuit.rowk`'s generic ports.
- Its Lean is in PoUW's areas of `security/`: `Definitions/Pouw/PearlC/` (the files named `*Fp4*`, `Row4Circuit`,
  `Tile4Circuit`), `Definitions/Pouw/Assumptions/PearlC/TTOutFp4*` and `Proofs/Pouw/PearlC/` (the files named `*Fp4*`,
  `Row4Words`, `Tile4Circuit`).
