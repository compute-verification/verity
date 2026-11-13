# AllReduceMultimem lowered in rank 0's node window (REQBLOCK, Thu 12 Nov 10:20 PM PST [Fri 06:20Z])

Branch `cursor/circuits-cir274-multimem-6f04` (from DENSE's 30dac672), head 5005ee9e. Module
`adapters/vllm/verity_vllm/program/predict/node_window_multimem.py`, tests `tests/program/predict/test_node_window_multimem.py`.

| (43 ids, 2 outputs, rank 0) | 4 layers before | 4 layers now | 78 layers before | 78 layers now |
|---|---|---|---|---|
| digest | 61b9f8d8516fe67f… | 3ee9420daaf7545f… | f4c1c567288f… | 08a3b9bbb6575290… |
| units | 5,137,384 | 5,191,393 | 123,537,384 | 124,208,701 |
| templates | 105 | 108 | 105 | 108 |
| wire patterns | 13,238 | 13,266 | 945,322 | 946,126 |
| peak GiB | | 1.012 | 10.9 | 13.916 (watch 14,214,264 kB) |
| window build s | | 28.2 | 594 | 681.6 |
| unlowered | | EmbeddingShard 2, StopCheck 1, TokenSelect 2 | +AllReduceMultimem 157 | EmbeddingShard 2, StopCheck 1, TokenSelect 2 |

Before: the coordinator's figures (r20261113-025656-e215 at 78). Now: r20261113-055056-24b4 (4) and r20261113-055026-4829
(78), both at 5005ee9e, read from their records. Tests: r20261113-055126-6a3b at 5005ee9e, 38 passed with the frame's,
dense and sparse files. Value check: r20261113-053257-b863 at 26d5e1ae (same lowering; reads() was sped up after).
- 2026-11-13T08:26Z REQBLOCK: same machine (up 24 days), no work lost; branch 5005ee9e; reading value check r20261113-073556-0a01 (launched 07:35Z, DENSE driver + multimem, 120 min cap); return by Sat 02:00Z.
- 2026-11-13T08:28Z REQBLOCK: value check r20261113-073556-0a01 @5005ee9e: 41,695 checked, 0 mismatches; plant 1,024/1,024 caught. #9364 merged 08:15:20Z, #9627 untouched. Return posted on CIR-274 (--worker REQBLOCK).
- 08:52Z REQBLOCK: piece A merge 1b640168 (b3902993 into 3d00e506) pushed, run launched; piece B cb65dbaa (DSAMOE chain ranges) pushed, run launched.
- 09:04Z REQBLOCK: piece A 1b640168 run 1fbe 190 passed 0 failed 4 skipped, gate [] mypy clean (run 86dd lacked the GLM B8 fixture: 41 FileNotFound); piece B cb65dbaa run cc4f 39 passed, digest 3ee9420daaf7545f, gate [].
- 09:16Z REQBLOCK: piece C runs 861a (repository), ae48 (vllm lint) at cb65dbaa; piece D driver values_plant96.py launching.
- 09:42Z REQBLOCK: piece C 861a repository 392 passed; ae48 vllm lint 87 passed 2 failed (P1, P3 in node_window_values.py, not mine); fix diffs under scratch/cir274/multimem/fix-p0{1,3}-*.diff, checking them in a clone.
- 09:39Z REQBLOCK: piece D r20261113-091230-0102: SIZE=96 plant 749 of 1024 differ (749 covered); fix check 0862: P1 P3 pass, typed_records schema-literal on the domain, P3 diff revised (DRAW_DOMAIN constant). (The 09:42Z line above was 09:25Z.)
- 09:44Z REQBLOCK: fix check 70cd: both diffs, vllm lint 89 passed 0 failed 1 skipped, gate []; 0862: windows 101 passed, repository 392 passed, 1-layer value check 0 mismatches. Multimem head unchanged cb65dbaa.
- 17:46Z REQBLOCK: #9627 carried: head b9e5c4a0 (0e6e13c3 + main c68135b4) pushed; pins 4d7b nothing stale; collect x7 at b9e5c4a0 equal old in every field; suites 5017 and research.carry running.
- 18:38Z REQBLOCK: #9627 b9e5c4a0: suites 5017 24/24 passed; research.carry exit 0 merge-only carries; mypy lint 33c4 queued (cluster memory).
