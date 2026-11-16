# facet_break: try to beat every facet of the ACCEL model

The ACCEL model (`../../machine_model/`) bounds what a program can issue on sm_120 by a polytope of facets: row r says
Σ_k c_kr·x_k ≤ 1 for the program's warp instructions x_k of census class k a SMSP-cycle. ε is proved conditional on
every facet being a true ceiling.

So this directory looks for programs that break one. It is a standing search: it generates programs that mix MMAs with
lanes of other instructions around each facet's vertices, times them with opcode_probe's gates, and loads every measured
program, its own and any other probe's, on every facet. A program whose load on a facet is over 1 beats it, and the
facet then needs a refit. It only reads the ACCEL model: a refit is printed, as the lines to add by hand.

## Try it

~~~sh
python3 benchmarks/pouw/facet_break/generate.py --seed 2 --n 300
~~~

This prints the programs one batch would time, one JSON line each, and needs no GPU.

## Run it

`run.sh` times one batch, as the command of a `research run` from a pushed commit on a node with an RTX PRO 6000. Its
arguments are `generate.py`'s. The `loads.py` commands after it read any runs' `rows.jsonl` on a CPU:

~~~bash
research run --on MACHINE --project verity --source . --cwd source --custody-ttl 12h \
  --env GPU_LEASE_WHO=facet-break -- bash benchmarks/pouw/facet_break/run.sh --seed 2 --n 300
python benchmarks/pouw/facet_break/loads.py summary ~/.research/runs/*/out/rows.jsonl ~/.research/runs/*/out/tensor_kinds.jsonl
python benchmarks/pouw/facet_break/loads.py refit RUN/out/rows.jsonl --row R1
python benchmarks/pouw/facet_break/loads.py family RUN/out/rows.jsonl --family R1 KFFMA0 --name EFFMA --kind FFMA --emit VF3q-16
python benchmarks/pouw/facet_break/loads.py summary RUN/out/rows.jsonl --full --epoch VF3s-16
~~~

[`DESIGN.md`](DESIGN.md) says what each file holds, what a run writes, how a load is counted and why it is a lower
bound, and how the table route's L2 facet is read.
