# Pearl-C on the H100 and H200: design

The runner, the build's gate, the other jobs and who reads them, beside [the README](README.md).

## The runner, `run.py`

`run.py` runs on one H100 or H200 from a tree built off the card, through the CUDA driver API with the standard library
alone. `h200.sh` is the command of the `research run` that starts it and the GLM-5.3 jobs after it under one lease; its
header has the full command and its switches.

- `build.sh` compiles the cubin and refuses it unless `sass_gate.py` passes: no ptxas warning or serialized wgmma, no
  spill or stack frame, every FP8 atom and peel step the intended warpgroup instruction in the stated counts, no other
  MMA form. It records the harness's `.FTZ` and MUFU gate and needs no uv. `ship.sh` runs it, then writes `fixture.py`'s
  inputs and expected buffers, `run.py` and the tree's `MANIFEST.sha256`.
- `run.py` refuses any GPU but the expected one, then runs every kernel on the fixture and compares each buffer byte
  for byte, the GEMMs at all three tiles and on all, one and three CTAs, over poisoned outputs. Only a variant whose
  buffers all match is timed, phase by phase. It exits 3 on a refused identity and 4 on a failed check.
- `h200.sh` writes every step's exit code to `$OUT/rc.txt` and each job's output under `$OUT` (`passes/`, `timing-h3/`,
  `whole/`), and exits non-zero when a step failed.

## The other jobs

Each is started by `h200.sh` or by its own command, which its docstring gives.

| Script | Runs on | What it does |
| --- | --- | --- |
| `glm53.py` | a CPU | GLM-5.3's linears as one rank serves them, each with the kernels' domain verdict; the other scripts read their shapes from it |
| `llama32_1b.py` | a CPU | Llama-3.2-1B's linears as one H200 serves them, each with the kernels' domain verdict, as `glm53.py` has GLM-5.3's |
| `glm53_timing.py` | the card | the served call against the stock FP8 block GEMMs, interleaved rep by rep, at decode's and prefill's rows |
| `glm53_passes.py capture` | the card | the gate and three retained passes (prefill, decode and a control) over poisoned outputs |
| `glm53_passes.py verify`, `full` | CPUs | the reference replay of those passes: the sampled tiles, or every tile and every served row |
| `glm53_profile.py` | the card | where a served call's time goes, step by step, under Nsight Systems |
| `glm53_slices.py` | a CPU, with a local snapshot of the checkpoint | whether every row of each rank's slice of the down projections passes the per-row rules |
| `grouped_check.py` | the card | the grouped call (`grouped.py`) against the same rows called unit by unit: the same bytes, then both timed |
| `overhead_arm.py` | the card | the harness arm `verity bench overhead` loads for Pearl+; `overhead_arm.json` names the cubin it pins |

`hash_prefill.cuh` is A's commitment past decode's rows, with its host build `hash_prefill_host.cpp` for byte tests;
`acc.h` is the generated accumulator operand lists.

## Who reads it

The vLLM adapter loads `run.py` and `grouped.py` by path as the served kernel on these cards, takes a unit's keys from
the scheme (`pearl_c.keys_of`, which `fixture.py` re-exports), and records both files' sha256 in every pass, so a change
to either is a new variant, which serves only once it has passed its own gate. `verity bench overhead` imports
`overhead_arm.py`'s `PearlPlus` and `epsilon`, its `__all__`, by the names its `SCHEMES` holds. `../../../../../../../../benchmarks/pouw/pearlc_k256_gate.py` reads `glm53_slices.py`'s samples; `../../../../../../../../benchmarks/pouw/pearl_c_vllm/` builds and loads this
tree; the tests in `../../../../../../../../tests/benchmarks/pouw/` run `run.py` and `grouped.py` on the stand-in driver and the hashing on its host builds.
`run.py` and `grouped.py` read the environment themselves, because they run outside the tree; `../../../../../../../../benchmarks/pouw/settings.py` declares
each variable. Nothing here is a verifier: a replay's ACCEPT is a development check, never the protocol's verdict.
