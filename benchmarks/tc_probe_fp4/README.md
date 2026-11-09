# tc_probe_fp4: Blackwell's FP4 and FP8 tensor-core instructions against their models

The Blackwell counterpart of `benchmarks/tc_probe`. Each probe runs an instruction on its GPU, raw bits in and out, and
compares every output word with the instruction's pinned model in `verity/ml/circuits/silicon` (`BLACKWELL_SM120_NVF4`,
`BLACKWELL_SM120_MXF4`, `GroupSum`). Where no model is pinned yet, a capture decides between named candidates. Every
mismatch is recorded, and nothing is tuned to fit. Everything runs on sm_120a (RTX 5090, RTX PRO 6000) unless its file
says sm_100a (B200).

## Probes

- `probe.py`: one `mma.sync` per tile (`mma_fp4.cu`, its rows in `rows_fp4.cuh`) or one `tcgen05.mma` (`tcgen05_fp4.cu`),
  compared word for word with the model. Its docstring lists the instructions (`--instruction`): `nvf4`, `mxf4`, `bf16`,
  the K = 32 kinds (`f8f6f4` and its mixed and scaled forms), the `tcgen05_*` rows, the conversions, the 2:4 sparse rows
  and the Lean conformance vectors. `--sweep` runs the families, `--replay-autoproof` the reproduction gate, and
  `--sass-check` confirms each tile kernel holds exactly one MMA.
- `cast.py` (`cvt_fp4.cu`): the FP4 path's conversions against exact references.
- `k32.py`: the K = 32 E2M1 rows' encodings, and the candidate models a capture decides between.
- `sparse.py` and `meta_rule.py`: the 2:4 sparse FP4 rows against candidate models, and what each metadata nibble gathers.
- `lean_vectors.py`: named NVFP4 conformance vectors from a generated Lean vector file, captured on the card.
- `f8_sm100.py` and `f8_sm100.sh` (`tcgen05_f8.cu`): the B200's `tcgen05.mma` FP8 step, generated on the CPU, captured
  under one GPU lease and scored on the CPU. `f8_model.py` is `GroupSum` on whole arrays of words, for the scoring.
- `model5.py`: autoproof's model5, the control `probe.py` keeps. The probe refuted it on sm_120.
- `stage0_sm120.py`: which FP4-path mnemonics ptxas accepts for sm_120a, and what each lowers to. It is record-only.

## Records and tests

- `tool.py`: the store declaration (`tc_probe_fp4`), which says which parameters key a result.
- `tests/`: the CPU side, which builds and runs no kernel. It covers the K = 32 candidates against `GroupSum`, the cast
  references against brute force, the sparse rows' mapping, the fake-silicon dry runs and the store declaration.
  `tests/test_f8_model.py` holds `f8_model.step` to `GroupSum.step` word for word.

From the repository root after `uv sync --all-packages`, on a GPU of the instruction's architecture with nvcc (and
`cuobjdump` for `--sass-check`):

~~~text
uv run python benchmarks/tc_probe_fp4/probe.py --instruction nvf4 --out DIR --sass-check --sweep --n-random 4096
OUT=DIR bash benchmarks/tc_probe_fp4/f8_sm100.sh                                           # on a B200
uv run tools/verity/check/suites.py benchmarks/tc_probe_fp4                                # the CPU suite
~~~
