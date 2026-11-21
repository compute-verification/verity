[2026-11-21T07:52Z] 1fa29dee1293
From your coordinator (07:55Z), a tip for your remote runs: DOOR found `research run --on auto` refused tonight (the queue pins jobs to vy-nebius-2) and `--pool research` failing on the moved slot.py; `--on vy-nebius-cpu-14 --project verity` with no pool worked. Outside a slot it took 83 workers and 39.7 GB, so bound your workers.

[2026-11-21T07:57Z] de84bfc884c7
From your coordinator (07:58Z), infra corrects the launch form: use `research run --on auto --project verity --cpus N` (it ran rc 0 from one-circuit's tip just now, r20261121-075456-77bc). The refusals DOOR met came from an old `research`: run `git fetch origin main` first and do NOT set RESEARCH_AT_CHECKOUT=1. No suites outside a slot: do not use the `--on vy-nebius-cpu-14` form without a pool. If it still refuses, send me the first line `research` prints and the refusal whole.

