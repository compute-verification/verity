# facet_break: try to beat every facet of the machine model

The machine model (`../machine_model/`) bounds what a program can issue on sm_120 by a polytope of facets: row r says
Σ_k c_kr·x_k ≤ 1 for the program's warp instructions x_k of census class k a SMSP-cycle. ε is proved conditional on every
facet being a true ceiling. This directory is the standing search for programs that beat one: it generates MMA + lane
programs around each facet's vertices, times them with opcode_probe's gates, and loads every measured program (its own and
any other probe's or tensor_kinds' records) on every facet.

| File | What |
|---|---|
| `loads.py` | A program's load on every row at an epoch (`loads`), every violation (`summary`, load > 1: rate rows carry the 2% margin), the results table (`table`) and the refit a beaten row needs (`refit`: the upper hull of the measured points at the margin, `eps1.facets`). IMMA.16832 loads every row its own kind has no line on through the generic `MMA` lines (they were measured with it); every other form only through its own (`--generic`: through `MMA` too, advisory) |
| `mix.cuh` | The kernels: a MMA form and up to two lane ops in one warp's loop (`FB_MIX`), in separate warps (`FB_SPEC`), the lanes alone (`FB_ALONE`), the MMA alone (`FB_MMA`), over `../opcode_probe/probe.cuh` |
| `generate.py` | The seeded grid: the targeted part (each lane class beside IMMA.16832 at the vertices of Π's (x_MMA, x_X) section and around them, two-lane pairs, warps a SMSP, each form alone, the JI3n family) and `--n` sampled programs; `--emit` writes the probe source with each program's exact SASS counts, `--only` keeps the gate's passes |
| `run.sh` | One batch under `research run`: generate, build, gate, drop the refusals by name, rebuild in chunks of 64, time under one `gpu-lease`, then `loads.py table` and `summary` into `$OUT` |

~~~bash
research run --on vy-nebius-2 --project verity --campaign pouw --source . --cwd source --custody-ttl 12h \
  --env GPU_LEASE_WHO=facet-break -- bash benchmarks/pouw/facet_break/run.sh --seed 2 --n 300
python benchmarks/pouw/facet_break/loads.py summary ~/.research/runs/*/out/rows.jsonl ~/.research/runs/*/out/tensor_kinds.jsonl
python benchmarks/pouw/facet_break/loads.py refit RUN/out/rows.jsonl --row R1
~~~

A load counts only the instructions whose class is certain (`loads.class_of`; NOP, MOV and the loop's own uniform
instructions are left out), so it is a lower bound and a reported violation is real. A violation goes in the lane's note
with its row, load, program, run id and SASS, and its refit line.
