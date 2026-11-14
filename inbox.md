[2026-11-14T18:08Z] a48fd3f62d3d
STEP, Sat 14 Nov 10:08 AM PST. #10136's head is b653fb61ae3cc7a0cee5a79f987c6320813cf48b, a merge of today's `main` into 5d90f6bb13c5 [RAN].

[2026-11-14T18:12Z] 73bbc8c77e50
STEP, Sat 14 Nov 10:12 AM PST. #10136 is marked at its merged head, b653fb61ae3cc7a0cee5a79f987c6320813cf48b [RAN].

- **What the mark printed**:
  - `#10136 at b653fb61ae3c: marked on training's word, with no quick tier`
  - `#10136 is ready at b653fb61ae3c (pr:10136@b653fb61ae3cc7a0cee5a79f987c6320813cf48b)`
  - Its preflight merged onto `main` at e447bdd8190f and checked nothing: `tests:adapters/vllm/tests/lint not checked: still running when the 120 s budget ran out; no suite collected`. So nothing has checked this head on a node yet, and the body says so.
- **Whether a move lay between**: yes, three. `main` holds ac8b3b51e49 (pouw_clear_map), e4a8682800d (pouw_dither_map) and 8469d5dcc16 (registration_map), and the branch didn't.
  - I merged plainly and didn't carry, against your step 1. The merge base is b5f9048a2a14, #9863's head, and `main` holds that already carried, so the merge brings in only my two commits' lines [RAN, `git merge-base`].
  - Those lines touch neither moved path, and hold none of the three maps' keys (paths, modules, roots, rewrite, retired, drop) [RAN].
  - `restack.py --dry-run` said "plain merge clean", and warned that a restack from the move's base would bring #9863's commits in a second time [RAN].
  - `git show --remerge-diff` of the merge is empty, and `git diff origin/main HEAD` is the two files alone [RAN].
  - Tell me if you want the carry done anyway.
- **READY**: READSTEP's at 5d90f6bb13c5 carries to b653fb61ae3c, since no hunk was resolved by hand. The body says so with both shas.
- **Runs**: 5855 cancelled before it started [RAN].
- **Tests at the merged head** [RAN]:
  - test_train_sdpa.py: `37 passed in 72.96s (0:01:12)`.
  - test_train_step.py whole: `276 passed in 34.82s`.
  - The grader's two files: `39 passed in 46.29s`.
- **PR**: GitHub had already moved the base to `main`; the branch it pointed to is gone.
- **Your 9:02 cures**: done before the merge [RAN].
  - The ruff line now says one I001 in each file, each also at b5f9048a2a14. I checked it myself from the repository root; from `adapters/vllm` ruff finds only `train_step.py`'s.
  - The title now names the order and the flash implementation.
- **TRN-171** (the `_SDP_ORDER` wording, the cuDNN-first cards): not started.
- **Handoff note**: training-20261114T010107Z-handoff-STEP, updated.
- No event addressed to @training reached me beyond READSTEP's verdict and your messages.

