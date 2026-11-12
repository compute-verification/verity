[2026-11-12T16:46Z] 75f906f6aa27
grade to @training, Thu 12 Nov 08:45 AM PST (16:45Z): the node read (your 8:21 AM) and the seventh product (your 8:33 AM).

FIRST, THE ONE THAT CHANGES YOUR SENTENCE. The stored counts behind the published tables already hold the seventh product. "Six of seven are in |C|" is true of today's counter (`flops.Counter._attention`) and of a trace; it is false of the records the tables divide by.
- RAN (VM, on job 2's asif.json): every fused backward row of the 809 records, 24,125 rows and 47,729,601 calls, is counted at 2.5 times its forward call: 5 x n·L·E·S a call. The records were written by te_core 2 to 3.20, which counted a fused row by torch's own formula, and that formula counts the scores again. The same 5 in the five records job 2 could not read (13 rows) and the 16 standard-workloads records (26 rows).
- RAN (VM): a published |C| is that count as recorded. r20261103-063939-2c0c@6000 recomputed from its record's op log with te_core.gates 4.13.6 gives 1.457459012373823e19; the table prints 1.4574590123738229e19. Over the search table 1,482 of 1,497 rows equal their record's log to 2e-16 (job 2's `log_gates_before`); the other 15 are the five unread runs.
- So job 2's as-if did two things at attention, not one: 16 to 32 bits, and the backward from five products to the four (or fewer) the calls ask for. Its factors hold both.

THE FIGURE (a re-pricing of stored counts and stored traces; no step traced again). Slowdown with the seventh counted over the printed slowdown:
| table | rows read | least | median | largest | why |
|---|---|---|---|---|---|
| search | 1,497 of 1,497 | 1 | 1 | 1 | the count holds it |
| overlay | 449 of 458 | 1 | 1 | 1 | the count holds it |
| page | 188 of 197 | 1 | 1 | 1 | the count holds it |
| standard-workloads | 16 of 18 | 1 | 1 | 1 | the count holds it |
| size-scaling, attack rows | 12 of 12 | 1 | 1 | 1 | the count holds it |
| size-scaling, standard training | 45 of 45 | 0.9762 | 0.9802 | 0.9868 | a traced step: four backward products |
Not done: 9 rows of the overlay and the same 9 of the page take |C| from the finals reruns' records, which I do not hold (INFERRED five, same counter, same days); standard-workloads' two modelled rows; size-scaling's 33 rows that name no run. The traced rows by start (seventh = the forward's score products' MACs at 1,480.125): s-t20 0.97615, m-t20 0.97859, l16-t20 0.98017, xl-t20 0.98517, 1B 0.98680; C/(C + seventh), and on these rows the slowdown is 1/C (checked on the as-if steps build: the product moves by under 4 parts in 10^4). Your 0.97 to 0.99 is these rows. It is also what the library's table and a job counted today would move by: neither holds the seventh.

1. THE EFFICIENT ROWS. RAN on vy-nebius-cpu-7, r20261112-163437-901a (814 records read: the 809 and the five; peak 4 GB of 48 declared; K read, nothing written).
- Corrections to my 8:18 AM message. 20 runs, not 19: r20261103-062249-c1a1 (one of the five) holds them too, so the search table has 42 rows of 14 runs, not 39 of 13. And "25 of 6,001 steps" was wrong: it is TWO steps. Totals with the five: efficient 8,304 forward calls in 20 runs, 1,325 backward in 10; flash 83,281,225 and 51,172,212; cudnn 0.
- Every efficient forward row lists three operands, q, k and v, all [b, 8, 1024, 64] (b 16; 8 and 2 in three runs). Every backward row lists eight: five of that shape, [b, 8, 1024], two scalars. No fourth tensor: not (a). Head dim 64: not (c). Lengths 1024 and 1024: not (d). In three runs the same step holds flash rows of the very same shapes.
- One word a row: (b), INFERRED by elimination among your four. The record does not tell it from a program that turned flash off for those passes: no row names a dtype, and no field of the record's head names autocast, a dtype or a mask.
- Which pass. The 14 runs with search rows: step 0 holds 150 forward calls (25 passes of the 6 layers, batch 16) with 150 backward (3 runs), 125 backward (7 runs: five of the six layers a pass) or none (1 run), beside the step's own flash pass; the last step holds 6 forward calls, one pass, no backward. Three runs hold 54 forward calls in step 0 alone. The other six: 54 or 432 forward calls in step 0 (3 runs), 2,694 in the last of 9 steps (1), 42 a step over steps 8 to 45 (2, start s50). All train.py, island seed-mezo. A calibration before the first update and an evaluation after the last is my reading of the shapes, INFERRED; the program's source would say, and K's repository does not hold it at the record's commit (the islands' programs were sent as tars; 0 of 12 found by hash).
- Autocast from the program's arguments: not read. The job kept, apart, the tokens of `script_args` whose text holds one of my words (autocast, amp, dtype, float32, mask and so on) with the token after each; 8 of the 20 runs have one, and "amp" also matches "sample". I have not opened that file: say if I may, it is the program's own arguments, not a run's command.

2. THE COUNT'S WIDTH. 16 bits, and from no field: all 814 records' op-log rows have six fields and name no dtype (RAN), so te_core.gates prices every GEMM-like row, fused rows among them, by `ASSUMED` (gemm in bf16) (READ, gates.py 4.13.6). If the efficient rows are (b), at 32 bits (12,637 against 1,480.125 a MAC; RAN on the records):
- |C| of the 14 runs rises 0.004% to 0.698%; largest r20261103-063939-2c0c and -062249-c1a1 (0.698%), then -075914-ae6d (0.372%), -055523-9aca (0.176%), six runs at 0.088%. Over your 0.5% in two runs.
- Three runs print a slowdown above 1 and it falls by the same factor (slowdown is 1/C there, RAN on job 2's two builds): 86,240.8 to 85,642.7 (2c0c), 5,090.97 to 5,086.49 (-092604-410b), 86,577.0 to 86,553.1 (r20261104-165055-b721). The other 11 runs' rows print 1 and stay 1.
- A larger thing follows from (b), INFERRED: a float32 pass has float32 linears too, and the count priced them at 16 as well. Pricing every forward and backward op of those passes at 32 bits is a bound of 3.15% on |C| for 2c0c and c1a1, 1.6% for ae6d and 9aca, 0.40% for the six, under 0.13% for the rest: 86,240.8 would be about 83,605.

3. bmm, baddbmm, matmul over the 814. 71 runs hold bmm rows (4,020,247 calls); none holds a baddbmm or matmul row. None is attention's: no row has a fused call's batch times heads with its 1024 and 64. 3,362,760 calls are the optimizer's, in 14 hybrid runs, on stacked weight matrices ([6, 12 or 24, 512, 512 or 1408]); 644,480 are backward rows of five runs, four of them quant ([32, 4, 1024] by [32, 1024, 512] and the like); 13,007 are forward, in 53 runs ([8, 512, 4096] by [8, 4096, 512] and one run's [8192, 1, 512] by [8192, 512, 256]). Every one of the 814 has flash rows.

4. The four size-scaling runs and frontier-fit's three: none is in the scorer's state (no index line). READ: the four are the runs that traced standard training's steps (capture 4.12.9, the math path by construction, no card kernel); their rows are the last line of the table above. Frontier-fit's r20261102-064629-0b45 kept a config, a curve and a result and no op log: its compute is a formula (flops a token), so no kernel is named anywhere; torch 2.13.0+cu129 on the card. I did not open the other two.

5. The five unread: READ in job 2's unread.json, each has a record and the rewrite refused it. Four (island wildcard) hold a flash backward row whose k and v are [32, 8, 64, 64] beside q [32, 8, 1024, 64], a cross-attention shape the rewrite does not take; c1a1 holds two backward rows of one shape where its trace asks one call for v's gradient alone. Their kernels are in the census: flash, and c1a1's efficient rows.

NOT CHECKED: the finals reruns' records; torch's selection in the runs' own torch (2.13.0 by one config; your read is main's); the program's source; `script_args`.

SLIPS, mine, two: a key filter on a run's job.json matched `execution` and printed its first 80 characters, which begin a command field (an env assignment naming a lock path, no secret); and printing a frontier-fit config I printed the start of its `args` (a data path). I read no run JSON by pattern again: named keys only.

Files: /cursor/stores/self/job4/node/ (kern.py, kern.sh, runs.json, out/census.json, out/records/), /cursor/stores/self/job4/census/efficient-runs.json; the run's out on the store.

The change: edits made, tests next; nothing pushed yet. Fri 10 AM PST stands.

