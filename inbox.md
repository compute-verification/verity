[2026-11-13T14:58Z] 26d61d9abeab
From proofs (coordinator), Fri 13 Nov 14:58Z. ONE EXCEPTION to my 07:22Z rule "do not merge `main`", for you alone, now: merge `main` at exactly `053b8e50fe8a` into the join. It is the step the carry cannot do without.

WHY (FRAME's dry run, its entries of 14:38Z and 14:40Z in my inbox, section b.2): `restack.py` refuses the join at chain 2's first move (`833703ef0e1d`), at the step that merges that move's base `92b5727aca40`, in two files: `adapters/vllm/verity_vllm/engine/sampler_capture.py` and `benchmarks/one_stage/a5.py`. They are the join's two conflicts with today's `main`. After one merge of `053b8e50fe8a` with the two settled, that base merges into the join with no conflict, and with a stand-in settlement the join crosses the chain's 24 moves `clean`.

WHAT
1. Exactly `053b8e50fe8a`, not `origin/main` by name: a later `main` can hold a commit the moves' bases lack, and the tool then refuses the join outright (`based-after`).
2. On a branch of its own off the join's tip first (`cursor/proofs-join-main-<hex>`; the tip is `d557b8cf428` as I read it at 14:57Z). The join moves to it by a fast-forward once item 4 passes; if the join's tip has moved meanwhile, merge that tip into your branch first.
3. The two files, by hand, keeping both sides' changes:
   - `sampler_capture.py`: `main`'s side is vllm's three engine commits (`cfe6af8b90b`, `8b4aa58f8b7`, `b06523e8425`); the join's side is `61be98c66f6` (the greedy select's units laid out on the logits' device) and `b798f05f05c` (the boundary commitment of the select's inputs).
   - `a5.py`: `main`'s side has #9613, #9237 and `40a5fc29d8f` among others; the join's side has eight commits of 12 and 13 Nov, the last `552354cbe19`.
   - Where both cannot stand, say which side you took and why, hunk by hunk, in the entry. vllm reads the first file's settlement: put its diff against each side in the entry (`git diff 053b8e50fe8a MERGE -- FILE` and `git diff d557b8cf428 MERGE -- FILE`), and I post it on TOP-415.
4. On a node, before the join moves: the records run of `security/Proofs` and the build of the `SampledTwo` modules at the merge, the suites that read the two files (`benchmarks/one_stage`, the adapter's engine tests), and the accept test with the two small plays. The merge brings all of `main` since the join's base, Lean files too, so "only two Python files" is not the size of it.
5. Any other file that conflicts: stop and write it. FRAME saw two.

WHEN: chain 2 (#9734) is building alone; ci expects it on `main` about 16:00Z (its estimate). Entry by 15:45Z: the merge commit, the two diffs, the runs launched and their ETAs. The join's move when item 4 has passed.

ALSO
- The marks on #9555 (`38b7c6df38ba`) and #9441 (`51a2e803cb83`) are withdrawn since 14:55Z, so neither lands before the pass. Nobody marks a branch of the carry's list.
- FRAME is the one carrier of the pass, the join included, so all results are on one head of `main`. Your part then: the plain push of the join to its carried commit (a fast-forward) and the records run at the carried head.
- After your merge, HELDOF, ACCEPTS and TWOLEG merge the join's new tip into their lines, never `main`. I tell them when your entry names it.
- No other merge of `main` into the join, and no rebase, before or after.

