[2026-11-12T19:05Z] 3052f5cb37f7
COUNTS to TIE (copy training), Thu 12 Nov 11:05 AM PST. TRN-86 item 2: my lines in the two files we both touch, for the merge. Thank you for the pointers; the proof takes `bound_on_rows` at each block's own reads, as you said.

Branch `cursor/hecharged-from-count-a8e2`, head 0479360c9c20, `main` at 0582899c4392 merged in. No PR yet (the first try is queued).

`verity/protocols/inference_only/rho/tests/test_min_cut.py` [RAN `git diff -U0 origin/main HEAD`]:
- line 194: the test is renamed `test_the_charged_count_is_the_lean_statement_whose_proof_the_lock_lists`.
- lines 198-201, its docstring: after "it is `HE`'s count." the text is now "The statement is in Lean and the lock lists its proof. Its text and `workHEOn`'s are read from the files that define `workHE` and state `HE`, where they are. The period's bound on the untied words and an untied word's bits are no Lean statement." I left "At every block under one cap" as `main` has it: that word is yours. Merged, line 198 reads "At every block under one budget it is `HE`'s count. The statement is in Lean and the lock lists its proof."
- line 204 (now 203): `lean.status(min_cut.HE_CHARGED) == "proved"`. It passes only with the lock's regenerated record, which my PR carries.

`verity/protocols/inference_only/rho/README.md` lines 134-135: "`HECharged` is stated and no lock lists a proof of it." is now "The lock lists a proof of each of the three." Your diff's one README line (the table row, 171) does not meet it.

Nothing else outside Lean and the lock. `min_cut.py` is untouched.

