[2026-11-14T09:15Z] 5af61ecb8e6e
training to NAMES2, Sat 14 Nov 1:15 AM PST. Read: your 1:12 AM message.

**READMAPS has the delta 0f7a6f11..dfb06d58 since 1:14 AM. `theta_difftest_vectors.py:330`: leave it, unlisted, as you recommend.** `unregistered` stays unlisted too.

Your way for the chain is right: carry 0f7a6f11's text through the script, then remake that commit and the five after it with the READY text before step 4's push. If READMAPS' verdict is not in when step 3's reading ends, push with dfb06d58's text and say so in the draft's body.

[2026-11-14T09:16Z] 67fe70acc311
**dfb06d58 holds all three cures as written, each true at 208d51aa, and changes nothing else.**

READMAPS to NAMES2 and @training, Sat 14 Nov. Read on `cursor/training-rename-rehearsal-6c-a32b` at 208d51aada5dd9c3fab337697de94838abb25e9d.

READY dfb06d585e92be6cf301afec44bb165997037c72

## MUST FIX (0)

## SHOULD FIX (0)

## NOTE

1. The three cures (RAN).
   - **The refusal codes.** `input-in-iu`, `gate-in-no-iu` and `empty-iu` are written as strings in `Check.lean`, `Desc.lean`, `inference_only_reference.py`, `difftest_vectors.py` and `desc_difftest_vectors.py`.
   - **`baseline_difftest/`.** It is tied to no toy now. It is `ThetaDifftestMain`'s directory (`lean-audit.json:16-21`), and the lock pins its three files (`:25-27`). Its only writer is `theta_difftest_vectors.py:365-369`.
   - **"baseline difftest".** It is the θ difftest's printed name, at `ThetaDifftest.lean:49` and `:54`.
2. Nothing else changed (RAN).
   - `git diff 0f7a6f11 dfb06d58` changes only the three clauses, and every other clause of the paragraph is word for word the same.
   - `git diff --stat 9bb03080 208d51aa` is this README alone, and `016c1313..208d51aa` is the README alone, 7 insertions and 4 deletions.
   - dfb06d58's parent is b9fb40b9.
   - The five cherry-picks have the same patch-ids and messages as the commits I read: 73dd2be0 = 1f69e8fa, dc616a65 = a26c12c0, bc619951 = a4535709, 53b11159 = 3bd4ebf4 and 208d51aa = 016c1313.
3. The three tests pass at the head: `tests/test_retired_terms.py`, `tests/lint/test_repo_retired_terms.py` and `tests/test_doc_paths.py` give 50 passed and 7 bookkeeping warnings, the same count as before, none naming this README (RAN).
4. Your three rulings make no sentence of the paragraph false: `desc_difftest_vectors.py:305`, `theta_difftest_vectors.py:330`'s `verity/nci/v1`, and the kind `unregistered`. The paragraph says "a few older names stay" and claims no complete list (READ, INFERRED).

## For the lead to read

1. `verity/protocols/policy/inference_only/verifier/README.md:130-137`, the paragraph.

No node run launched. No events, PR comments or run outputs reached me this turn.

