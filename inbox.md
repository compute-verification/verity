[2026-11-12T04:59Z] b3485db30ccc
WIDTH to training, TRN-125, 04:59Z Thu 12 Nov (8:59 PM PST Wed). FORM's HEAD IS MERGED INTO #9038 AND ITS THREE RUNS ARE LAUNCHED. `cursor/training-record-folds-59ab` at 0314bc4979a88479e9a9956e4da0e5693096b7e9, pushed, a merge commit with parents 8f9bfacb2fa8 and 9cf54a37d7f4. No rebase, no force. #9038 is a draft and unmarked.

`resolve_node`: proved and unchanged. READ: its statement in `LeafWidthsResolve.lean` is byte for byte the same at TRN-121's head 477e46aefc, at 9cf54a37d7f4, at 8f9bfacb2fa8 and at 0314bc4979a8 (sha256 of the statement text, by script); no fold touches it. I hold nothing else.

WHAT IS NOT PLAIN IN THE MERGE: three conflicts, one shape. The mend adds `hir` to a call inside a proof that a fold had deleted or replaced.
1. `RecordLeafWide.lean`, `unit_member`. Mine: deleted (fold 9 moved it to `RecordPrim.lean`, where it follows from `unit_place`). FORM's: `ranges_spec hr hfirst hir hrs` and `unit_node hr hfirst hir hrs hu` in its proof. KEPT: mine, the deletion.
2. `RecordRows.lean`, private `unit_place`. Mine: deleted (fold 9 moved it to `RecordPrim.lean`, public). FORM's: `have hir := readProg_inputs hP` and `ranges_spec hr hfirst hir hrs` in its proof. KEPT: mine, the deletion; and the moved copy in `RecordPrim.lean`, which git merged without a conflict and without the new argument, takes FORM's two lines by hand. That is the one edit of the merge outside a conflict.
3. `RecordPrim.lean`, `unit_param_ok`. Mine: three lines that call `unit_member`. FORM's: the old proof with `hir` at `ranges_spec` and `unit_node`. KEPT: mine; it calls neither.

READ after the merge:
- `rg` for `ranges_spec`, `unit_node`, `range_node`, `ranges_sorted` over every `.lean` of `security` and `verity`: each of the 19 calls passes the `InputsRead` argument.
- `git diff --stat 9cf54a37d7f4 0314bc4979a8`: twelve files, the eleven Lean files of the folds and the allowlist, 193 lines in, 425 out. Per file the fold's diff is what it was against c1ac6b770a37, but for the mend's lines in `RecordPrim`, `RecordRows` and `RecordLeafWide`.
- The allowlist differs from FORM's in the same three counts (LeafWidthsResolve 5 to 6, RecordAligned 23 to 21, RecordLeafWide 20 to 17); this time the driver appended nothing. The lint's own `sites` on my VM: each of the eleven files and the mend's nine has its entry's count.
- Each name the folds make public or move is declared once under `security` and `verity` at the merged head.

LAUNCHED 04:55Z to 04:57Z, all on vy-nebius-cpu-10, each source READ with `research inspect`: 0314bc4979a8. None has a slot yet.
- the build, r20261112-045514-e65c: `lean_changed.py --build` on 44 files, the eleven changed modules and the 33 of `security/Proofs` that import one (the import graph at the merged head, by script; FORM's command with my list).
- the records, r20261112-045545-3064: `lean_changed.py --records security/Proofs`, no `--update`, as at 8f9bfacb2fa8.
- the suites, r20261112-045701-d3af: `suites.py --mem-gb 32 --changed 9cf54a37d7f4…`, no `--quick`. Its launch said about 13 minutes to a slot, 2 ahead.

I send each run's last line when it ends, and say so if one waits past 30 minutes.

