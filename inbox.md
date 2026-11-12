[2026-11-12T18:38Z] 6443416cbdfc
COUNTS to training, Thu 12 Nov 10:38 AM PST (18:38Z). #9038 moved after my review: head 760582c0b3a9 (18:29Z), base `main`; #8676 merged at 18:10Z (RAN `gh pr view` both). My review of 0314bc4979a8 stands at the new head, with the changes below.

What the three new commits are (RAN `git` on my VM at 760582c0b3a9):

- c9ddac7c1c merges `main` at 30d6fcc92f. The one file the merge holds differently from both parents is `tests/lint/allowlists/repo_retired_terms.json` (`git diff-tree --cc`). `main` changed none of the eleven Lean files since 9cf54a37d7f4.
- 91c3b066ee, the allowlist: against 30d6fcc92f it differs in three counts and nothing else (`LeafWidthsResolve.lean` 5 to 6, `RecordAligned.lean` 23 to 21, `RecordLeafWide.lean` 20 to 17), and each of the eleven files' count by the lint's own `sites` equals its entry.
- 760582c0b3 re-wraps `RecordPrim.lean:9-14`, the header's first paragraph: my SHOULD FIX 3, done; prose unchanged.
- The PR's diff against `main` is the reviewed one: twelve files, +198 -431; ten files' patches are identical to the reviewed diff's (`git patch-id` of each), `RecordPrim.lean` differs by that re-wrap, the allowlist as above.

SHOULD FIX at the new head, both in the PR body (READ just now, still the old text):

1. Its first two paragraphs ("Stacked on #8676 …", "`git diff 9cf54a37d7f4 0314bc4979a8` is the whole change", "425 lines out and 193 in") and "What I ran" name the old head and the stack: say 760582c0b3a9 on `main`, and that its runs are at 0314bc4979a8, under a merge of `main` that touched none of its Lean files.
2. Item 5, as in my review (`getElem!_push_size` at `Array Defn`, instances of `BodiesOk`'s two).

Checks: nothing has run at 760582c0b3a9 that the body names. The lean-proofs skill takes a clean merge of `main` over a passing head as needing no new Lean run; this merge is clean but for the allowlist, which the lint reads and I ran. What only a build at the new head shows is whether `main`'s other changes since be9a3d566905 moved a lemma these proofs use; the mark's own preflight builds the changed files.

READY TO MARK at 760582c0b3a9: yes, once the body is rewritten for it. No MUST FIX.

Still running: r20261112-173627-c485, the whole Lean check of `security/Proofs` with its kernel replay, at 0314bc4979a8. Its last line follows.

