# tc_probe: tensor-core instructions against their models

Verity's circuits for GPU arithmetic use each tensor-core instruction's exact semantics (`verity/ml/circuits/silicon`), so a
served model's circuit computes the words the GPU computes, bit for bit. Honest serving passes only where those models match
the silicon. These probes check that on the hardware: each runs an instruction on its GPU and compares every output word
with the instruction's pinned model.

- `tc_probe.py`: one `mma.sync`, `wgmma` or `tcgen05.mma` per tile (`mma_tiles.cu`, `wgmma_tiles.cu`, `tcgen05_tiles.cu`),
  with arbitrary accumulators, compared word for word with the model. Its docstring lists the instructions (sm_80 to
  sm_120a; BF16, E4M3 and E5M2), the finite-sweep families and the replays (golden records, frozen input sets).
  `--sass-check` confirms each tile kernel holds the expected MMA-class instructions; `--compile-only` does that without a
  GPU.
- `instances_hw.py`: a target's frozen input sets chained on silicon, the K-steps run with the FP32 accumulator carried, and
  every hardware word compared with the set's expected word.
- `canary.py`: the fixed small tile set (`tc_probe.py --canary`) to rerun after a driver or toolkit change, or on a second
  device of the SKU.
- `trust.py`: each silicon relation's trust level (UNPROBED, HYPOTHESIS, PINNED, TRUSTED, TRUSTED_STABLE), computed from the
  evidence store and the discrepancy log (`verity/ml/circuits/silicon/discrepancy_log.json`), never set by hand, and the
  dossier that shows why.
- `tool.py`: the store declarations, which say which parameters key a result.

From the repository root after `uv sync --all-packages`, on a GPU of the instruction's architecture with nvcc (and
`cuobjdump` for `--sass-check`):

~~~text
uv run python benchmarks/tc_probe/tc_probe.py --instruction sm89.mma.m16n8k32.e4m3 --out DIR --sweep --sass-check
uv run python benchmarks/tc_probe/tc_probe.py --instruction sm89.mma.m16n8k32.e4m3 --out DIR --canary
uv run python benchmarks/tc_probe/instances_hw.py --instances fp8-ada-synthetic/vu-k1536-fp8 --out DIR
uv run python benchmarks/tc_probe/trust.py --all [--root STORE] [--format md|json]
~~~

`tc_probe.py` writes `probe_results.json` (the device's identity, the layout check, per-family counts per model and the first
mismatches), `capture_sample.json` (golden-format records sampled per family) and `tiles_<family>.npz` (the full capture). In
a recorded run (`RESEARCH_RUN_ID` set) it also writes `result.json` (kind `tc-evidence/v1`) and prints the labels `trust.py`
reads. `instances_hw.py` writes `probe_results.json` and `capture_<tier>.npz`. `trust.py` prints a dossier per subject, or
writes them with `--out`.

Status: current. `benchmarks/tc_probe_fp4` is the Blackwell FP4 and FP8 counterpart. `tests/` covers the trust levels, the
input sets and the store declarations without a GPU, and replays a handful of instances where nvcc and a GPU are present.
