[2026-11-11T08:22Z] 9cd443f82899
From onecalc (TRN-101), Wed 11 Nov 12:21 AM PST. DONE for tonight: draft PR #8854 is complete at head a4ed47cb711 (branch cursor/training-one-calculator-f980, pushed, five commits over 95239b187833), its body updated, NOT marked. Everything below I ran unless it says read.

SUITES at a4ed47cb711, node run r20261111-080220-1c31 (vy-nebius-cpu-4): suites.py on the benchmark and its grader, then suites.py --changed 95239b187833 --quick (the four suites the change touches). benchmarks/training_slowdown 290 passed; its grader 28 passed, 3 skipped; repository 280 passed; verity-graph 181 passed; none failed. The first run (r20261111-074032-7344, at e17b03de155) had the benchmark's 292 tests pass and its read guard fail: the one-door test reads the nested grader's files. Commit a4ed47cb711 declares them (the grader as an input of the suite, the directory as that test's declared read). rho's own suite passed 602 in that first run; nothing under verity/ is in the diff.

TWO COMMITS since the draft opened:
- cb81fe29c7f: ruling 3. products_of, he_of_products, products_per_c, rhos, fit_k, from_json go, and Measured.to_json with from_json (its round trips were its only readers), with two tests. No reader on origin/cursor/training-replot-55a1 (8898d6232018) or the library's branch (7dba4eaacc98, again at 91e731be6dd) outside the package's own tests, so no name stays. It changes no number: at a4ed47cb711 the regrade's 458 rows equal e17b03de155's and the stored point's print and slowdown.json are byte-equal.
- a4ed47cb711: the pyproject fix above.

THE TABLE OF MOVED FIGURES (unchanged since my last line; full table moved.md and each cell's work moved.json in art:7be38c8170ec4dbc3ec3806a6e5814d42579e695a679506288ca7a403d53d3ef; the body of #8854 has these and the per-start tables). 294 of 458 rows regraded with no GPU, all 289 that publish a slowdown; a ratio is after over before:

| cell | figures | least | median | greatest | over 0.57% | over 10% | down | the subset that binds changes on |
|---|---|---|---|---|---|---|---|---|
| the row's own policy (`rho_own`) | 289 | ×1.002783 | ×1.297281 | ×1.528686 | 263 | 208 | 0 | 25: 25 rows, the weight gradients alone → M |
| the console's (`rho_default`) | 289 | ×1.002783 | ×1.297281 | ×1.528686 | 263 | 208 | 0 | 25: 25 rows, the weight gradients alone → M |
| the served one (`rho_served`) | 289 | ×1.004739 | ×1.157626 | ×1.642929 | 263 | 198 | 0 | 186: 102 rows, the weight gradients alone → M; 81 rows, M without its short contractions → M; 3 rows, the weight gradients alone → M without its short contractions |

Thresholds, exactly: 234 rows move more than the stored point's x1.0057077577, 29 are at it, 26 under (x1.002783, the 36M start's quant rows); with the threshold at 0.57% it is 263 in each cell. None goes down. The first conjunct tested at whole m: 867 of 867 cells.

Least S per start and gain, before -> after (rows that reach their gain):
|---|---|---|---|---|---|---|---|
| 238M · 17.2 tok/p | 1.1 | 4 | 2,081,582 | ref-optim/fullmuon-fe-wsd 82a6@1804 | 2,106,254 | the same | ×1.0119 |
| 36M · 10.8 tok/p | 1.1 | 82 | 168,517 | shallow/strideshare 1b6b@1260 | 169,479 | the same | ×1.0057 |
| 36M · 10.8 tok/p | 2 | 12 | 422,277 | ref-data/grow-muon 6111@2217 | 604,009 | the same | ×1.4304 |
| 36M · 10.8 tok/p | 11 | 1 | 404,167 | hybrid/grow-wide 4551@49926 | 474,428 | the same | ×1.1738 |
| 36M · 108 tok/p | 1.1 | 4 | 467,434 | ref-optim/fullmuon-wsd ff7f@2736 | 645,350 | the same | ×1.3806 |
| 45M · 1.71 tok/p | 1.1 | 7 | 58,490 | wildcard/topmlp bce4@466 | 82,888 | the same | ×1.4171 |
| 45M · 1.71 tok/p | 2 | 3 | 244,577 | ref-optim/freeze-embed 44df@1112 | 373,295 | the same | ×1.5263 |
| 45M · 3.42 tok/p | 1.1 | 71 | 68,702 | lowrank-restart/dgrad bf6c@819 | 72,478 | the same | ×1.0550 |
| 45M · 3.42 tok/p | 2 | 28 | 363,692 | ref-optim/freeze-embed f483@3339 | 403,867 | hybrid/lateqat ac47@3901 | ×1.1105 |
| 81M · 13.9 tok/p | 1.1 | 23 | 553,827 | shallow/depthshare 1a68@3240 | 556,988 | the same | ×1.0057 |
| 81M · 13.9 tok/p | 2 | 8 | 825,918 | hybrid/grow-depth d64c@7326 | 1,048,957 | the same | ×1.2701 |
| 81M · 139 tok/p | 1.1 | 2 | 836,962 | ref-optim/fullmuon-wsd 0360@9606 | 1,078,451 | the same | ×1.2885 |
| 95M · 2.38 tok/p | 1.1 | 5 | 78,634 | wildcard/topmlp 2c21@314 | 96,565 | the same | ×1.2280 |
| 95M · 2.38 tok/p | 2 | 2 | 891,176 | ref-optim/freeze-embed 8ee3@1221 | 1,134,947 | the same | ×1.2735 |
| 95M · 4.76 tok/p | 1.1 | 36 | 185,215 | lowrank/lora-4bit d0c2@2828 | 186,868 | the same | ×1.0089 |


The stored point (I ran the grade at the base and at the head): the bound 128,563.819 -> 129,297.630; S 168,516.7 -> 169,478.6; the run at S 5.230e23 -> 5.260e23 AND gates; its share of one datacenter's compute (12,500 locations) for the print's window, a week, a month, a year 0.0309, 3.06e-05, 7.15e-06, 5.88e-07 -> 0.0311, 3.08e-05, 7.19e-06, 5.91e-07; of the world's (46,870,000 locations) 8.24e-06, 8.17e-09, 1.91e-09, 1.57e-10 -> 8.28e-06, 8.22e-09, 1.92e-09, 1.58e-10. Proved given (H), (E) and the statement's conditions on the work, which no record checks: the count. Measured, not proved: p. Scaled: every share (a fleet's size from the census).

YOUR FIVE POINTS are in the body: the first lines; both thresholds; Nci.Cut.SecurityProofs.HE and HEShared by lock name with open = [] (read in security/lean-audit.json at origin/main 8201dacff26) and the two tests of test_min_cut.py that hold the count to the Lean; the paragraph on the rows where M now wins (25 and 186); the 28 rotation rows.

HAND-OFF NOTE, synced: lanes/training/20261111T0818Z-handoff-trn101-one-calculator.md. It has PR 2's plan. New in it: what the grade needs of the trace pricer. By a walk of top-level names (an upper bound), of 3,373 lines of definitions in flops.py and capture/ the grade reaches at most 1,600, the runner alone 1,644 more, and 129 neither. The pricer is not a third calculation: it writes the product rows and re-prices a record's raw rows. What PR 2 can delete with its tests: capture/run.py whole, capture_step, capture_fn, rows_on_dial, TORCH (no non-test caller since products_of went; only docstrings name them on the two branches).

ONE THING FOR YOU: /opt/cursor/artifacts on this VM is a link into your store (/cursor/stores/self/artifacts), which you told me not to write, and it holds a code-tour-author-handoff.md another worker wrote at 08:06Z. I wrote nothing there. My hand-off for the code tour and the logs are in /tmp/onecalc/artifacts/. Copy them if you want them uploaded.

Told library the pushed head and the names that went. TRN-101 has two comments. I end here: done for PR 1 as a draft; marking is yours.

