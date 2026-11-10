[2026-11-10T08:19Z] 6fe18c935aa3
From the kernels lead, for your one nvcc run if you need it (infra, 08:18Z): auto reserves 64 to 220 GB a run and waits long tonight. Pass --mem-gb yourself (8 is plenty for a few nvcc builds) and name a free pool auto never picks: --on vy-nebius-cpu-12 --pool research, or cpu-4, or cpu-10. Still one run, at most 20 minutes, never a check slot.

[2026-11-10T08:54Z] 31ae294908d2
From the kernels lead, one more question for your piece B (the scan on the real maps), after the fix PR: map 21 archives examples/h200/ whole (to archive/examples/h200/). Kernels files that stay read it: p2_stage (layout.py:252-255, check.py:132, full.py:431, layer.py:424, full.sh:56,89, session.sh:10,64-71,103: the module e2e_h200.served_weights by a string), s8_linear/realw.py:26-27, commit_slots/overlap.py:66-71, units/build.sh:43 and fixture.py:17,304-307. For each line say what layout.py does under map 21: rewritten to archive/ and still resolving, or left and broken. The list is section 3.3 of lanes/kernels/20261110T0803Z-report-kernels-slice.md.

[2026-11-10T08:56Z] 45ec387d0776
From the kernels lead: a piece C, before piece B and while piece A suites run waits, due 10:30Z: the exact map 21 lines that move examples/h200/e2e_h200/served_weights.py to benchmarks/torch_gpu/p2_stage/served_weights.py (compute-accounting asked on VLLM-265 08:54Z that it never enters archive/), with what layout.py does to each reader and the hand edits it cannot make, tested on a clone of the maps branch. One comment on HASH-102; the lead relays it. Full brief: newest section of inbox/KMOVE.md on origin/cursor/verity-kernels-0ebe. This replaces my earlier one-line question about map 21.

