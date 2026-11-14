[2026-11-14T09:12Z] 051ca8699ce1
NAMES2, Sat 14 Nov 1:12 AM PST. The second cure is pushed.

**The new hand commit is dfb06d585e92be6cf301afec44bb165997037c72 on `cursor/training-rename-rehearsal-6c-a32b`, whose head is 208d51aada5dd9c3fab337697de94838abb25e9d; READMAPS reads the delta from 0f7a6f11.**

- RAN: made as before. The `older_check` group's script line ran on b9fb40b9, then the same five cherry-picks. `git diff --stat 016c1313 208d51aa`: the README only, 7 insertions and 4 deletions. `tests/test_retired_terms.py`, `tests/lint/test_repo_retired_terms.py`, `tests/test_doc_paths.py`: 50 passed, 7 warnings, none naming the README.
- RAN: the paragraph takes READMAPS' three cures as written: the refusal codes `input-in-iu`, `gate-in-no-iu` and `empty-iu`; `baseline_difftest/` is "the θ difftest's directory, whose bytes the lock pins", tied to no toy; "baseline" is "also the θ difftest's name in its printed messages". No other file changes; line 305 stays.
- RAN: the chain. Step 2 started 09:02Z and is past map 7 of 17. The chain's `older_check` commit is made from `hand/hand_edits.py`, which must not change while the script runs, so the chain will first carry 0f7a6f11's text. Before step 4's push I put this text into `hand_edits.py` and remake that one commit and the five after it on the local chain branch, the way I made 6c. Nothing is pushed until then, so no force-push.
- RAN, to close the sweep (case-insensitive `git grep -w` over the check's directory for each older name in GLOSSARY's rows):
  - The record's kind `unregistered` (`Record.lean:147`, its reference) is one kind of fresh bits, a parameter's read outside θ. It is not the older name for fresh input, so it stays unlisted.
- **Question, recommended answer "leave it, unlisted": `theta_difftest_vectors.py:330`, a docstring, says "`verity/nci/v1` would answer …". No other file holds that id; the protocol is `verity/inference-only/v0`. It is the same on `main`. You ruled no file but the README changes, so I leave it, as with line 305, unless you say otherwise.**

