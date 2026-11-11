[2026-11-11T21:13Z] 3355f4af8500
COUNTS to training, copy FORM, TRN-130, Wed 11 Nov 1:15 PM PST. BACK AFTER THE CUT. THE BRANCH IS WHERE IT WAS, 6d9709d399, AND IT HOLDS c1ac6b770a. FORM'S FILES ALL PASS OVER THE NEW CODE: none of the four proof files and neither of lean's two `Record.lean` breaks. No draft yet: the suites run ending 62a9 is still running, 46 of 49 passed, none failed.

THE BRANCH (RAN: `git fetch`, `git merge-base --is-ancestor`): `cursor/training-digest-form-string-a8e2` on origin at 6d9709d399, as I left it; c1ac6b770a, the head of `cursor/record-form-final-3b1c`, is an ancestor. #8676's own branch `cursor/inference-only-record-form-0063` is still at 83102b1b53, so the draft's diff will show more than my seven files; the body's first lines say so. A dry merge of the branch with main at 75ef75838f has no conflict.

WHAT EACH RUN SAYS NOW (RAN; each read from its node after the VM's reset):
- r20261111-182236-62a9, suites at 6c1cf5e7a5, `--changed 2177450071`: running at 1:13 PM PST. 46 of 49 passed, 0 failed. Since my 12:05 PM, among others: "passed  verity-flock 2049 passed, 66 skipped in 4655 s", "passed  verity/protocols/pouw/verifier 1944 passed in 713 s", "passed  verity/definitions/ops 985 passed, 2 skipped in 244 s", "passed  verity/definitions 691 passed, 2 skipped in 507 s", "passed  verity/circuits 1120 passed in 46 s". Left: verity/commitments, verity/commitments/windows, verity-core-evaluate. Its slot's limit is 2:22 PM PST; if the limit ends it first, I say which suites had no verdict and run those alone.
- r20261111-200340-e106, at 6d9709d399, `RecordRead.lean`: "done rc=0"; "lean exit 0 in 128 s"; "axiom probe: none of its 16 theorems uses an axiom beyond propext, Classical.choice and Quot.sound".
- r20261111-202817-d418, at 6d9709d399, one run that tries seven files in one slot: "done rc=0", each file's exit printed.
  - `RecordInputs.lean`: "lean exit 0 in 318 s", 14 theorems on the three axioms.
  - `RecordKinds.lean`: "lean exit 0 in 40 s", 4 theorems.
  - `RecordTemplates.lean`: "lean exit 0 in 192 s", 16 theorems.
  - `security/Properties/Nci/Record.lean` (lean's): "lean exit 0 in 14 s", "no theorems in it".
  - `security/Proofs/Properties/Nci/Record.lean` (lean's): "lean exit 0 in 20 s", 16 theorems.
  - lean's `OpaqueStop.lean`, an untracked copy made inside the job, its sha256 equal to the file at fe030990ef: "lean exit 1", "5 of its 5 theorems failed". As on lean's own run: they ask the kernel about `Json`'s `==` on two strings, which this change does not touch.
  - lean's `Opaque.lean`, the same way: "lean exit 1", "1 of its 2 theorems failed ... canonicalEntry_raw". The error is "Tactic `rfl` failed" on a goal that starts `if DefinitionDigest.isComposite e = true then`. INFERRED: its hypothesis about `==` rewrites nothing there, so the reading where the comparison answers no is gone from `canonicalEntry`.
- r20261111-202256-e2f0: "failed 143 CANCELLED_MANUAL". Mine: a queued try of one file that I cancelled at 12:28 PM PST and replaced by d418. No verdict of Lean's.
- r20261111-182100-c7e3 (records at 6c1cf5e7a5), ...be9f and ...6fd6 (the two tries at 6c1cf5e7a5): done, as my earlier messages have them.

SO, lean's question, now RAN on a node and not only on my VM: their failing probe still fails where it asks about `Json`'s `==` itself, and the digests no longer call it; `isComposite_iff` is the statement a proof uses instead.

LAUNCHED AFTER THE CUT, all at 6d9709d399, none run before:
- r20261111-210810-bafc, vy-nebius-cpu-3: the records, `lean_changed.py --records verity/protocols/inference_only/verifier/nci/lean --update`. Waiting for a slot.
- r20261111-211025-fece, vy-nebius-cpu-7: `suites.py --changed c1ac6b770a`, limit six hours. Waiting for a slot.
- r20261111-211132-6702, vy-nebius-cpu-3: `check.py --build verity/protocols/inference_only/verifier/nci/lean`. One more than you asked. READ (`test_nci_record_difftest.py`'s docstring): the record difftest's Lean half, 670 cases through `pinOf` and so through `closureDigest`, runs only in that package's Lean check, not in a suite and not in the records run. It is item 3's difftest at the merged head, with the kernel replay.

NEXT: the draft the minute 62a9 ends green, at 6d9709d399, the three runs above in its body as running; its number to you; a comment on TRN-130. If a suite fails: no draft, the failing line first.

NOT YET RUN: nothing that is not in flight. Not yet read: 62a9's last three suites; bafc, fece, 6702.

