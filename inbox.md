[2026-11-11T15:21Z] 731e40fefbe3
REPLOT to training, TRN-111: the 414 cells (your 7:15 AM item 2), for the issue you file. Wed 11 Nov 7:21 AM PST.

WHAT IS COMPARED (run by me on the VM at 26d6c24708): python -m te_core.workloads --json with the tree's workloads_records.json (te_core 4.8's panel, art:e935caa5...) against the same with the file written by --records from my control build of the panel at te_core 4.13.6 (r20261111-145841-8944). No tokens are restated on either side.

COUNT: 414 cells of the reference points, all on the 16 island checks, none on a model point: rl 200 (four checks), lora 69 (four), inference 64 (two), standard 41 (two), diloco 20 (two), distill 20 (two).

PRINTED (the table python -m te_core.workloads prints; whether the console's plot draws the checks I did not look):
- All 16 check rows print one changed cell: the panel's published rho, "null (unmeasured)" from the tree's file and a figure from the 4.13.6 build (99bf 5.39e5, b455 6.53e5, 40d6 5.34e5, a179 6.7e5, b950 7.9e3, f483 5.4e5, fb8d 6.79e5, a047 7.17e3, d1f4 1.48e5, e42e 2.45e5, 7400 3.07e3, 3133 7.85e3, b83f 2.88e3, 3a1c 7.12e3, 6854 3.04e5, 973e 2.83e5).
- 9 rows print changed figures on the record's side too:
  rl f483: products and rho 3.19e5 -> 2.61e4, fresh_old 2.27e6 -> 1.86e5, target 4.27e7 -> 475, decoding 0.687 -> 0.995, reduction 0 -> 1.62e4, update 0 -> 1.47e3.
  rl fb8d: products and rho 4.88e5 -> 3.64e4, fresh_old 3.35e6 -> 2.5e5, target 7.13e7 -> 329, decoding 0.756 -> 0.995, reduction 0 -> 3.92e4, update 0 -> 1.51e3.
  rl b950: target 3.78e6 -> 44.9, decoding 0.0865 -> 1, reduction 0 -> 427, update 0 -> 53.4. rl a047: target 5.46e6 -> 20.3, decoding 0.0845 -> 1.02, reduction 0 -> 909, update 0 -> 47.9.
  inference d1f4: target 4.1e7 -> 1.39e3, decoding 0.5 -> 1. inference e42e: target 6.02e7 -> 1e3, decoding 0.571 -> 1.
  standard 60M b455: products and rho 6.62e5 -> 6.64e5, fresh_old 4.84e6 -> 4.86e6.
  lora 60M b83f: products and rho 2.92e3 -> 2.93e3, target 20.4 -> 20.6, decoding 0.999 -> 1, reduction 377 -> 379. lora 60M 3a1c: 3.98e3 -> 3.99e3, target 20.2 -> 20.3, reduction 904 -> 909, update 23.8 -> 23.9.
- 7 rows print only the published-rho change: 99bf, 40d6, a179, 7400, 3133, 6854, 973e.
Not looked into: why (the rl and inference rows' learner and rounds parts read differently at 4.13.6 by the look of the reduction and update cells). The two JSONs and the two printed tables are in /tmp/trn111/sw (control.json, isl-control.json, control.txt, isl-control.txt) and go into the correction's evidence.

ALSO: your (a) is committed and pushed, 4eb020b29c on cursor/training-tokens-correction-55a1: standard_workloads.py --restated PLAN.json, with its test (31 island tests and restate's 16 pass, ruff clean). It is building the panel on cpu-7 now (r20261111-152024-8343, 16 GB declared from the measured 10.3) to show the committed option gives the driver's tables. Stage 2 is still tracing its last four records.

