# served_commit: design

What someone working on these arms needs beyond the [README](README.md) and each file's docstring or header.

## The arms and scripts

- `arm.py`: the arms' base, `ServedCommitArm`, which commits under `rows-sha512`. `prof_arm.py`'s rows arms and the
  harness's `commit` arms measure the Commit. Its docstring has what a run does, and which runs its gate accepts without a
  root (declared warm-ups, named in the verdict).
- `served_arm.py` and `served_arm.sh`: the served arm, the Commit at 4 KiB rows against compiled and eager stock in one run
  (the specs `*-rtx-served-arm`). With `PIN_BUILD` (a Build of record's `build.json`) every arm compiles the Build's
  kernels and `out/pin_check.txt` says so, which a counted compiled ratio needs. `tests/benchmarks/served_commit/test_served_arm.py` tests both over
  stand-ins for the arms they build on.
- `run.sh`: `verity bench served_commit`, one spec's sweep in one process; `SPEC` names the spec.
- `tp_decode.py` and `tp_decode.sh`: one Commit arm against stock in one TP engine, eager (GLM-5.3 TP8).
- `hash_prof.py` and `hash_prof.sh`: one step's rows-sha512 hashing as the windowed worker calls it, at T tokens a step, in
  the worker's form (held leaves folded many windows at once, levels copied on a copy stream) and the per-window form before
  it, with the kernels' knobs, under nsys and ncu.
- The specs: the served arm's, `benchmarks/harness/specs/*-rtx-served-arm*.toml` (one RTX PRO 6000 each, on the width sweep).
  `run.sh` and `prof.sh` take any served spec as `SPEC`, with its arms, sweep and reps unless their arguments name others.

The acquisition under CUDA graphs is `adapters/vllm/verity_vllm/acquire/sources/served_compiled_source.py`.

## Interface

- **Provides:** the arms the shared harness runs (`arm.py`, `served_arm.py`, `prof_arm.py`), the tensor-parallel decode
  step (`tp_decode.py`) and the scripts that launch and profile them.
- **Loaded by:** the harness, through the specs that name `served_commit.served_arm`'s arms, and `benchmarks/kernels/moe_align`,
  which imports `tp_decode`.
- **Loads:** the vLLM integration's `commit`, `pipeline`, `engine`, `acquire`, `correspondence`, `query` and `protocols`
  packages; `verity.circuits`, `verity_vllm.definitions`, `verity.commitments` and `verity.commitments.cuda`;
  `benchmarks/harness` (`verity_bench`); torch and vLLM.
