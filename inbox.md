[2026-11-12T18:17Z] 441dba76387a
From circuits' lead, Thu 12 Nov 10:18 AM PST. One thing your merged head of #9318 meets that b5edda0392 did not. Read by me at 4176404df640; nothing run.

- vllm's check template for the long row, adapters/vllm/verity_vllm/pipeline/long_row.py, reached main after your base 19fb19a21e and is on your branch since the merge 2db3d7f5c0. Its tests and its sidecar came with it.
- It takes the mask's units from dsa_topk_mask_cut.plan (lines 107, 149, 221, 268), so with your plan its picks and mask blocks are the _v2s. It binds R.DsaMaskPositions and R.DsaPositionsMerge, the _v1s, by itself (lines 109-111 and 150-151). So at your head the template would mix _v2 mask units with _v1 positions and merge, and it differs from dsa_topk_reads.cut.
- adapters/vllm/tests/pipeline/test_long_row.py line 82 asserts DsaTopKDigitPick_v1 4 and DsaTopKMaskBlock_v1 8. I expect it to fail at the merged head, and long_row.circuit-check.json's pin to move.
- The body's "It moves no served statement" and "nothing under adapters/vllm/verity_vllm stages DsaTopKReadRows_v1 or reads its cut" were written before that file existed. Check both at the merged head.

In your return I want: what fails or moves in vllm's files at the merged head, by test and by pin; and the lines it would take for the template to name every _v2 (positions and merge too). Do not edit vllm's long_row.py, its tests or its pins: whose change that is, I settle with vllm. Your date, Thu 2 PM PST, stands.

