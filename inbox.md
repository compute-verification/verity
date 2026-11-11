[2026-11-11T15:13Z] d2e768a4f912
REPLOT to training, TRN-111: the standard-training panel (your 12:47 item 3) and the two sentences on one name (your 13:03). Wed 11 Nov 7:13 AM PST.

1. THE PANEL: ONE ROW FALLS. training/standard-workloads built twice by the published checkout's own builder (696528f7d1, te_core 4.13.6, tree 7de182a661a4), 18 rows each; run by me on cpu-7: control r20261111-145841-8944, restated r20261111-150820-0379.
- standard 20M, 99bf: rho_own 467,967 -> 467,458; rho_default = rho_console 539,339 -> 538,753; rho_served 10,282.2 -> 10,271.0; each -0.11%. With them its kinds cell (fresh_products 539,300 -> 538,800; fresh_old 4.678e6 -> 4.673e6) and its stored inputs' fresh_per_c. The row as built today is HIGH by 0.11%; it is the move of its search row 99bf@2301.
- lora 20M, 7400: no printed rho moves (2,660.48; 3,066.25; 36.4776). Its stored inputs' w_gates 1.38979e11 -> 1.3897e11 and alpha_ratio 0.999934 -> 0.999867.
- The other 16 rows and the note are identical.
What "restated" is here: the two records by the plan (99bf: step 0 and the 2,296 two-forward steps state 131,072, its five one-forward steps keep 65,536; 7400: every step 32,768 -> 65,536), and the island's four stored traces of them read against the restated count (step_tokens, token_factor 2 -> 1, the products block's tokens). Those are the three fields a trace of a restated record differs in: checked by me on 99bf@2301's two traces in the corrected build. It is not a retrace.

2. te_core.workloads' reference points (python -m te_core.workloads --json at 26d6c24708, run by me on the VM, under a second each):
- THE 1.9e7 AT 1B DOES NOT MOVE: 1.88543e7 in all four builds (as published; the pilot's literals restated; the island's records from the control build; from the restated build). It is the model's: standard at 999,397,440 parameters, its own session c 1.0322e16, k 65,536, binding fresh_products. No model cell moves in any build (default 586,013; FP8 439,752; 7B 1.16106e6).
- The three pilot checks, the record's side only (28 cells): w_gates and alpha_ratio restate (a1c2@946 2.31908e11 -> 2.31786e11, 0.995234 -> 0.994712; ee8f@694 2.32623e11 -> 2.32457e11, 0.992069 -> 0.991362; 286b@1720 1.66764e11 -> 1.66570e11, 0.998823 -> 0.997662), and the record's rho beside the model RISES 0.05%, 0.07%, 0.12% (591,276 -> 591,588; 585,667 -> 586,086; 61.98 -> 62.05). 286b@1720: the published te_core refuses its row (data-dependent), as it is and restated; I computed its cells without the refusal (formalism.complete), which gives the literal's cells exactly on the record as it is.
- The island's two checks (16 cells): 99bf's published rho 539,339 -> 538,753; 7400's record rho 3,175.53 -> 3,175.75.
- Apart from tokens, not Thursday's, not looked into: the tree's workloads_records.json is te_core 4.8's panel, and the control build's records at 4.13.6 differ from it in 414 cells of the reference points (rl 200, lora 69, inference 64, standard 41, diloco 20, distill 20).

3. SO A CELL MOVES and by your order it goes in Thursday's pull request. How, for your yes or no:
(a) the island's builder gets one option that reads a plan's runs restated and their stored traces against the restated count, with a test; the built panel goes to the store, not the tree.
(b) te_core's own files (the three literals in workloads.py, workloads_records.json, workloads.json) I would NOT change on Thursday: a change under te_core renames its tree, and the tables' rebuild reads its trace layers by that tree's name. I list the cells they would take in the body.
If I hear nothing I do (a) and (b) as written.

4. MEMORY, mine: the control job peaked at 10.3 GB against the 8 I declared (15:03Z, cpu-7). I had not measured the builder before declaring; it holds 16 records and their traces. The rerun declared 16 and peaked at 10.3. Stage 2 is at 101 of its 128.

5. TWO TENSORS UNDER ONE NAME, read by me in the counter at 024bbb42b6a7 (flops.py 473-490, 626-672) and in the three opened records (e950, 0254, 01fd):
A new block's gain is recorded as a second value of theta's gain of the same name (sixteen names list two digests a step; twenty new gains under sixteen names, so at most sixteen of them are listed), and until its value first differs from theta's the counter gives the ops that read it theta's role, not a new weight's; an op's price reads its kind and dtype, never its role or name, so |C| is the same. No printed cell reads it: the tables take a trained weight's uses from the trace's product rows by their shapes (a gain is no product's operand, and the capture names nothing by equality with theta), and the one reader of a step's weight names, the lowest trained block of a --cut run (rotations.lowest_trained), is block 0 on all three with or without the new gains, theta's own eight blocks each having a trained matrix.

Stage 2 (the one command, r20261111-141655-2ec8): 377 of 381 traced at 15:11Z; the four still tracing are the four that Thursday leaves out. Its comparison follows when it ends.

