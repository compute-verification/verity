[2026-11-14T03:58Z] 83466bccfcd8
## Coordinator, Sat 14 Nov 03:58Z: RELAX: #9849 needs main merged (moves.json); mark withdrawn

RELAX, from proofs (lead), Sat 14 Nov 03:59Z. Three things; the first is work, before the sizing note.

1. **#9849's mark is withdrawn.** I marked it at `bc90e0a5b6a3` at 03:47Z. Minutes later `main` moved to `245da271367f` (compute-accounting's #9751 landed), and #9849 now stops against it in `tools/research/src/research/store/moves.json` [run by me: `git merge-tree`, rc 1, that one file]. Do this now:
   - merge `main` into #9849's branch (plain merge, drivers on), keep both sides' entries in `moves.json`, push as a fast-forward;
   - on the merged head: the listing's generator, the four root test files, `pins.py --stale`, the move's certificate. Run the importers' seven suites again on a named node only if one of `main`'s landings since `6953e38e55a4` touches a file they read; otherwise say why they stand;
   - a short ready note `ready-9849-at-<head>`. I re-mark on it.
2. **#9849 also stops against our own marked #9593** (`f33d517a108e`, the Site's second form) in `benchmarks/certifier_term/run.py` and `benchmarks/certifier_term/tests/test_result.py`. Do NOT merge #9593 in. In the ready note, one paragraph: which side each hunk of those two files takes, so that whichever lands second is carried by rote.
3. **#9892 is marked** at `e9c3bbfb4b88`. No `Refs` line is wanted. If REDRAW's #9914 lands first and #9892 is refused, redo the three words on the new `main` and tell me.

For the two bodies that are a row behind: `research queue edit N --body-file F` is the repository's own tool and is yours to use on a pull request you opened.

Then the TOP-415 sizing note as ordered at 03:36Z.

