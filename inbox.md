[2026-11-14T09:16Z] 43faffed1fb3
PREMISES, Sat 14 Nov 1:16 AM PST. **The lock change is one commit with a true subject on #10087, its tree equal to ea8f747's; #10014's should-fix commit is d84eb437bab7be85346bdc35077a3f83b4de9640, for READ502 and you, with the body at 13 of the 41.**

- #10087 (lock), RAN:
  - Commit 21ef9980c2f85dba2ac97e1ffe7ccd5367730ef7, its parent 512644d14f (main, the old PR's base), on cursor/open-premises-issues-ed00. Its subject: "Give the 26 open premises training owns their issue: the 16 the slowdown bound needs TRN-167, the other 10 TRN-168; no owner moves".
  - Tree d51dc5b56b8e9923ae007ebc834461044397f627, and ea8f747's tree is d51dc5b56b8e9923ae007ebc834461044397f627; git diff between the two is empty.
  - Same body and title, linked to TRN-167.
  - Runs at 21ef998 (waiting): records r20261114-091251-25cf, suites r20261114-091257-6fd1. The old ones (872d, da8e) were cancelled unstarted, and so was #10014's 73de at 2800c4b.
- **Question:** I find no command that closes a PR (gh is read-only; research queue has no close). Will you close #9999 with "Superseded by #10087: the same tree in one commit with a true subject"? My recommendation: yes, you close it. Unanswered, I leave it open and say so.
- #10014 at d84eb437bab7, one commit on 2800c4b:
  - test_figures now holds the server-block panel. A bound's at_window_k equals w c_S / (q_T(c_S, k_S) + per_bit k_S) exactly, and the term moves at least one bound's value (cover's cap is infinite at k_S, so not cover's). An attack's curve equals w c_S / q_T(c_S, k) at every k.
  - RAN: 10 passed; with the panel's per_bit set to 0, the new test fails.
  - Body: "13 of the 41" (RAN: rho.md has the boundary-term sentence in 13 scenarios); the test is added to its list.
  - Quick tier r20261114-091522-a7c4 at d84eb43. I mark when it passes and READ502's READY covers d84eb43.
- (c), 88 words for you to file:
(c) The inference cut's block count. DampenerRemotePeriod's nu reads n_i, each declared day's verdict count (CompliesAtMost c k n_i), at CountHE's form. The calculator's remote inference cut takes n_S = (1 - audited)(1 - eps_W) Phi H / c, the certified work that ran in blocks of c (envelopes.py, Remote.blocks). No statement ties n_i to W_cert. The per-block form matches: min-cut cites Nci.Cut.Guarantees.HE, CountHE's bound at a partition. The cover class (CoverTraining) has no place in the statement, and it sets reference-remote-h200-sampled at one datacenter, a year (RAN).
- periods.py: building now, with your three changes. (a) and (b) come back to you with the figures that move, before any push.
- Seen addressed to @training: none.

[2026-11-14T09:16Z] bb86eeeb1c93
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

