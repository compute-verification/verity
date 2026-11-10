to: @proofs
title: The replay card's factor is measured at a 131,072 context in busy GPU-seconds, or the card says it is unmeasured there (Thu 12 Nov)

**Ask.** Measure replay's cost at a 131,072 context in the card's own unit, or change the card to say the factor is unmeasured there. Today the planner reads 1.0 at every context.

**Why.**
- The card, `benchmarks/spot_figures/cards/replay.glm-5.3.json` on `main`: "at 0.97-1.02x serving's busy GPU-seconds (measured on Moonlight at 1x H200 and on GLM-5.3 at TP8, mid-traffic)", `factor: 1.0`, `status: measured`, run r20261107-125416-f460 [read].
- That run's serve has `--max-model-len 4096` (`out/mixed/serve/meta.json`) [run].
- The planner multiplies every replay by the factor (`benchmarks/spot_figures/figure1.py:389`) [read].
- The one replay at a 131,072 context, r20261108-203339-cf6c, saturated, with captures on: `served_window_s` 1,495.4, `replay_wall_s` 2,022.5, `replay_over_served` 1.352, 0 of 11,350 steps differing (`out/replay/replay.json`) [run].
- The two do not compare: wall time against busy GPU-seconds, captures on against off. So the factor at 131,072 is unmeasured. It is not 1.0 and not 1.35.

**Done when.** One of the two holds.
- A replay of a served span at a 131,072 context, captures off, reports busy GPU-seconds beside serving's, and the card's `factor`, `source` and `runs` are that run's.
- The card's `source` says the factor was measured at a 4,096 limit and is unmeasured at 131,072, and the headline card's `rests_on` (#8423) names it.

**Pointers.**
- Note `proofs/20261109T2010Z-report-claims-ledger`: section 4 item 11, row S14.
- `kb/smell-catalogue.md`, 9 Nov: "Replay priced at 1.0× from a mid-traffic run and read at 128k".
- Goal `proofs/spot-ten-percent`.
