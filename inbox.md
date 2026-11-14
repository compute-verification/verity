[2026-11-14T05:06Z] 3eeab8ba4615
KEY, Fri 13 Nov, 9:06 PM PST. #9502's head is b90c0adf1db92a789539e7d1ab2727ca77441733 on cursor/trn147-form-reads-formats-10f7, pushed as a fast-forward from 14ca2e3c12 (and that one from 9ddabfeafb). The network suite is green at it. #9502 is still a draft.

The head
RAN: a merge of 5a6d7a8d04bf (#9474's head) into 14ca2e3c12 gives b90c0adf1db9, with parents 14ca2e3c12 and 5a6d7a8d04bf. There was one conflict, in network `tests/test_nci_record_check.py`, where my `_more` and #9474's `_as_written` are added at the same place. I kept both, `_more` first, and the two asserts are as #9474 has them, so there is no second copy. The tree is byte-identical to ee4dff6f3972, where the test file had passed (`5 passed in 30.71s`).
RAN: `git merge-base --is-ancestor` with the remote branch before the push said yes, and the push printed `14ca2e3c127..b90c0adf1db`. Nothing was forced.

Restack (from today's main, 6953e38e55)
RAN: `tools/move/restack.py cursor/trn147-form-reads-formats-10f7 --dry-run --kept --json`, r20261114-023135-0a96, status `conflicted`. It crossed 28 of the 29 moves and stopped at the last merge (step `moved`) on one file, `test_nci_record_difftest.py`. It skipped no move.
RAN: the same command with `--onto e4a8682800d6` (the last move), r20261114-030917-8bcd, status `clean`. It crossed all 29 moves, skipped none, and gave result 7fba7ab9a49a.

Merges and conflicts
RAN: merge of main (6953e38e55): one conflict, in the module docstring of `test_nci_record_difftest.py`. I kept both sides: my sentence on `formats`, then main's paragraph on `_interface`'s ValueError refusals.
RAN: merge of #9474's 2d8e9bac34c7: no conflict. #9474's eight files are as its head has them. The merge driver reordered the lock, which is equal as JSON.
RAN: merge of #9547's dd6555ae618a: two files, each with two additions at one place, both kept. In the difftest test's docstring, row formats are the fourteenth test and the rows rule is the fifteenth. In the vectors' `finish`, its docstring merges and the case's `block_rows` sits beside my `ran`, `theta` and `_width`.
RAN: the lock entry `runs.NciCheck.RecordDifftestMain` was regenerated with the script's own command, r20261114-034225-b929: 687 cases (mine had 687, #9547's 670).
RAN: merge of #9474's 5a6d7a8d04bf, described under "The head".

Tests
RAN 1, retired names (`research.queue`'s `map_retires`, `own_text`, `Retired.named` over the 29 maps, on the lines the head adds over main): 0 hits for 26 maps. `terms_map` has 2 hits, `period_map` 1 and `period_net_map` 1, all `window`/`windows`, in 14 added lines across 6 files. All of them are code main itself spells (the log's `window` key, `window.registered_roots`, and the module `verity.commitments.windows.interface`). The move's scripts left them, and I kept them.
RAN 2, diff --stat against main + 5a6d7a8d04bf + dd6555ae618a (merge-tree, tree fd7c152f99cc): `11 files changed, 538 insertions(+), 93 deletions(-)`, exactly the 11 files and none outside. Against the earlier main + 2d8e9bac + dd6555ae (840da2df89) it was the same: 11 files, +538/−93.
RAN 3, `verity lean try` at 14ca2e3c12, each lean exit 0, with the proof files unchanged since:
  RecordInputs.lean: 14 theorems, probe clean (r20261114-034526-125d).
  RecordRead.lean: 16 theorems, clean (r20261114-034525-c91b).
  RecordTemplates.lean: 16 theorems, clean (r20261114-034526-b039).
  NciCheck/Record.lean (r20261114-034525-191f) and NciCheck/RecordText.lean (r20261114-034525-ce1a): no theorems.
RAN 4, the nci package's Lean check at 14ca2e3c12 (r20261114-034526-e382): `AUDIT: PASS  1779 declarations in 18 modules; axioms propext, Classical.choice, Quot.sound`, and its record difftest run printed "687 cases compared, 0 differences". No `security/Proofs` replay and no records run.
INFERRED 4: test 4 holds at b90c0adf1db9. That merge changed only `test_nci_record_check.py`, a Python test the Lean check doesn't read.
RAN 5, quick suites at 14ca2e3c12, `suites.py --changed 6953e38e55a4 --quick` (r20261114-034532-d03d): `59 suites: 58 passed (0 cached), 1 failed; 119.1 min of test time`.
  The one red was the network suite at that head, before this merge: `97 passed, 2 failed`. The failures were `test_nci_record_check.py::test_the_script_reads_a_runs_files_as_the_case_they_make` and `::test_each_option_makes_the_case_it_says`. They are #9474 against #9547: #9547 adds `block_rows` to every made case, and #9474's script writes none. Main + 2d8e9bac + dd6555ae alone fails the same two (r20261114-041421-5673: `1 suites: 0 passed (0 cached), 1 failed; 7.5 min of test time`, network `94 passed, 2 failed`).
RAN 5, after the merge, the network suite at b90c0adf1db9 (`suites.py verity/protocols/accounting/certifier/verifier/network --fresh`, r20261114-044749-fd6f): `passed  verity/protocols/accounting/certifier/verifier/network 112 passed in 833 s; 112 outcomes recorded`, and its last line is `1 suites: 1 passed (0 cached), 0 failed; 13.9 min of test time`.
RAN 5, every other suite's line from r20261114-034532-d03d, all passed. Below are 57 lines, plus `benchmarks/circuit_properties` (117 passed in 17 s), whose package name the VM's output redaction garbles:
  passed verity/protocols/policy/inference_only/verifier 4 passed in 1 s; 4 outcomes recorded
  passed repository 361 passed in 396 s; 361 outcomes recorded
  passed verity/protocols/accounting/certifier/verifier 1563 passed, 1 skipped in 524 s; 1563 outcomes recorded
  passed benchmarks/one_stage 1105 passed in 538 s; 1105 outcomes recorded
  passed verity/protocols/proofs/spot/verifier 916 passed in 83 s; 916 outcomes recorded
  passed verity-pouw-benchmarks 1437 passed, 15 skipped in 673 s; 1437 outcomes recorded
  passed benchmarks/network_e2e/encodings 603 passed in 93 s; 603 outcomes recorded
  passed benchmarks/training_slowdown 711 passed in 312 s; 711 outcomes recorded
  passed verity-network-traces 721 passed in 190 s; 721 outcomes recorded
  passed benchmarks/remote_session 279 passed in 183 s; 279 outcomes recorded
  passed verity-bench 631 passed in 79 s; 631 outcomes recorded
  passed verity-core-firewall 50 passed in 63 s; 50 outcomes recorded
  passed verity/protocols/compose/device/live_challenger 664 passed in 107 s; 664 outcomes recorded
  passed benchmarks/prove_path 171 passed in 59 s; 171 outcomes recorded
  passed verity-core-composition 21 passed in 82 s; 21 outcomes recorded
  passed benchmarks/training_slowdown/grader 25 passed, 3 skipped in 31 s; 25 outcomes recorded
  passed verity-core-sampled-proofs 318 passed, 2 skipped in 813 s; 318 outcomes recorded
  passed verity-graph 220 passed in 21 s; 220 outcomes recorded
  passed benchmarks/sampling_plan 61 passed in 38 s; 61 outcomes recorded
  passed memory-challenger-lean 27 passed in 39 s; 27 outcomes recorded
  passed verity/protocols/tests 321 passed in 11 s; 321 outcomes recorded
  passed verity-netsim-workload 35 passed in 10 s; 35 outcomes recorded
  passed verity-network-e2e 70 passed in 15 s; 70 outcomes recorded
  passed benchmarks/verifier_cost 10 passed in 20 s; 10 outcomes recorded
  passed verity-tc-probe 43 passed, 1 skipped in 8 s; 43 outcomes recorded
  passed demos/lifecycle 66 passed in 8 s; 66 outcomes recorded
  passed verity-datacenter 49 passed in 59 s; 49 outcomes recorded
  passed core-identity 5 passed in 7 s; 5 outcomes recorded
  passed verity-commit-slots 61 passed, 18 skipped in 10 s; 61 outcomes recorded
  passed benchmarks/inference_only 21 passed in 7 s; 21 outcomes recorded
  passed verity/protocols/accounting/poub/device/live_verifier 203 passed in 2 s; 203 outcomes recorded
  passed benchmarks/hash/dither 68 passed in 4 s; 68 outcomes recorded
  passed verity/protocols/accounting/pouw/verifier/lean/commit 6 passed in 4 s; 6 outcomes recorded
  passed verity-proof-throughput 48 passed in 5 s; 48 outcomes recorded
  passed verity-cli 137 passed in 6 s; 137 outcomes recorded
  passed verity-certifier-timing 29 passed in 4 s; 29 outcomes recorded
  passed benchmarks/certifier_term 15 passed in 4 s; 15 outcomes recorded
  passed verity/circuits/private_circuits 19 passed in 2 s; 19 outcomes recorded
  passed verity-sha512-rows 8 passed, 4 skipped in 3 s; 8 outcomes recorded
  passed benchmarks/pouw/pearlplus_lean 3 passed in 3 s; 3 outcomes recorded
  passed benchmarks/one_stage/interfaces 93 passed in 2 s; 93 outcomes recorded
  passed verity-commitments-benchmarks 12 passed in 4 s; 12 outcomes recorded
  passed benchmarks/unit_lowering 3 passed in 1 s; 3 outcomes recorded
  passed verity-network-sim 21 passed in 2 s; 21 outcomes recorded
  passed benchmarks/two_stage 9 passed, 6 skipped in 1 s; 9 outcomes recorded
  passed benchmarks/served_units 48 passed in 6 s; 48 outcomes recorded
  passed verity-vllm 17243 passed, 480 skipped in 895 s on 4 workers; 17243 per-test passes kept
  passed verity-flock 1477 passed, 30 skipped in 458 s on 2 workers; 1477 per-test passes kept
  passed research 5120 passed, 3 skipped in 243 s on 7 workers; 5120 per-test passes kept
  passed verity/protocols/accounting/pouw/verifier 1990 passed in 148 s on 7 workers; 1990 per-test passes kept
  passed verity/circuits/ops 949 passed, 2 skipped in 132 s on 3 workers; 949 per-test passes kept
  passed verity/definitions 1036 passed in 75 s on 7 workers; 1036 per-test passes kept
  passed verity-numerical 660 passed, 6 skipped in 44 s on 7 workers; 660 per-test passes kept
  passed verity/commitments 600 passed in 35 s on 4 workers; 600 per-test passes kept
  passed verity/circuits 1120 passed in 21 s on 7 workers; 1120 per-test passes kept
  passed verity-core-evaluate 36 passed in 17 s on 7 workers; 36 per-test passes kept
  passed verity/commitments/windows 102 passed, 1 skipped in 4 s on 7 workers; 102 per-test passes kept
RAN 6, the served pair at 14ca2e3c12 (r20261114-034525-11dd), on r20261112-101803-fbc8's files: the same eight as in r20261112-233108-21af, by SHA-512 prefix. I restored program.json and circuit.json from the evidence-store artifact and checked their SHA-256.
  As served: check ACCEPT (blocks 4, gates 1,437,386,324, fresh 337,786,912), count ACCEPT (bool gates 11,044,318,495,127).
  With `--unregister smallest`: check accepts (fresh 338,704,416), count refuses `fresh`.
  Last line: `2 cases, 0 verdict(s) where the compiled command and the reference differ`.
INFERRED 6: these verdicts hold at b90c0adf1db9. The merge didn't touch the Lean, the reference or the served script.
READ 7, the PR body (`gh pr view 9502`): isDraft true, head b90c0adf1db9, 10010 bytes. It is in today's layout, says it holds #9474's head (5a6d7a8d04bf) and #9547's head (dd6555ae618a) and why, and gives what ran with run ids and last lines and what didn't run (the `security/Proofs` replay). It carries `Lane training`, `Refs TRN-147` and `Goal training/check-reads-served-record`.

#9547 and the form
READ: #9547 changes nothing the form accepts or refuses. It touches neither Record.lean, RecordText.lean nor the reference's `check_detail`. Its `RecordRows.evaluate` starts from `Record.read`, whose `finish` holds my `formats` stage. Its `block_rows` calls `expand`, so it refuses `formats` and `inputs` where the form does. The vectors build `rows_limits` under `_without(refused)`, which handles `formats`.

Not run: the `security/Proofs` package's Lean check (its replay); the three proof files are checked by `verity lean try` alone. A fresh full test 5 at b90c0adf1db9 didn't run either; only the network suite reran, as you asked.
Pushed nothing to cursor/nci-record-check-6111, didn't touch #9644, and have merged nothing from main since 6953e38e55. When you say the word, I merge main once, rerun tests 4 and 5, and send the head.
Handoff note: 20261114T0440Z-handoff-KEY (to be updated to this head right after this message).

