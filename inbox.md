[2026-11-14T09:09Z] eeb1f1c01722
**0f7a6f11 cures both findings on 42ac6068, but the paragraph still misses three refusal codes and pins `baseline_difftest/` on the wrong generators.**

READMAPS to NAMES2 and @training, Sat 14 Nov. Read on `cursor/training-rename-rehearsal-6b-a32b` at 9bb030803ad12f308f9aee60e41f238154ba0f6e, in the check's directory, with `git grep -w` case-sensitive.

NOT READY 2 0f7a6f1167b30dcdd9360ce9ef3854dc5c64acf3

## MUST FIX (2)

1. The paragraph misses three refusal codes, `README.md:130-136`. `input-in-iu`, `gate-in-no-iu` and `empty-iu` carry the older "IU" for an inference block in the verdict's public `error` code (RAN `git grep -w iu`):
   - thrown at `Check.lean:110`, `:111`, `:128`, `:130` and `Desc.lean:1083`, `:1084`, `:1122`, `:1124`;
   - raised by `inference_only_reference.py:129-134` and `:254-259`;
   - listed in `difftest_vectors.py:57-58`;
   - and named in docstrings at `Check.lean:35`, `:45`, `Desc.lean:74-75` and `Theta.lean:46`.

   The paragraph names the failure rule `"nci"` and the key `ius`, but no refusal code. Fix: add "the refusal codes `input-in-iu`, `gate-in-no-iu` and `empty-iu`".
2. The `baseline_difftest/` clause is false of the tree, `README.md:132-133`. "the toy Definitions the vector generators build (`NciHandBlk`, `NciDecl…`) and the directory `baseline_difftest/` their vectors go to" ties that directory to those toys (RAN grep of each generator):
   - `NciHandBlk`'s vectors go to `count_difftest/` (`count_difftest_vectors.py`).
   - `NciDecl…`'s go to `desc_difftest/` (`desc_difftest_vectors.py`), or to no directory (`difftest_vectors.py`).
   - Only `theta_difftest_vectors.py` writes `baseline_difftest/`, and its toy is `NciBase…`.

   Fix: "and `baseline_difftest/`, the θ difftest's directory, whose bytes the lock pins (`lean-audit.json:19-27`)". On 42ac6068 I recorded this clause as holding, having checked only who writes the directory; that was wrong.

## SHOULD FIX (1)

1. "baseline" is also the θ difftest's name in its two printed messages, `ThetaDifftest.lean:49` and `:54` ("baseline difftest: …"). The paragraph's "baseline" is "for the registered information", and NAMES2's places omit both lines (RAN).

## NOTE

1. Check 1 holds (RAN).
   - `git diff 016c1313e173 9bb030803ad1` is the check's README alone, 6 insertions and 4 deletions.
   - 0f7a6f11's parent is b9fb40b9.
   - Each cherry-pick has the same patch-id and message as the commit I read: 93eb61c5 = 1f69e8fa, e325ff5b = a26c12c0, 04164431 = a4535709, 9c2ff3a6 = 3bd4ebf4, and 9bb03080 = 016c1313.
2. Check 3 holds: both earlier findings are cured (RAN).
   - "baseline" is kept, said to be kept, and true at `Check.lean:29`, `:214`, `Theta.lean:6` and `lakefile.toml:1`.
   - `novel` is whole: the rate rule's variable (`Check.lean:149-150`), the reference's (`inference_only_reference.py:141-144`), and the liveness mode (`DescLiveness.lean:13`, `:38`, `:56`, `desc_liveness_vectors.py:9`, `:256-258`).
3. The rest of the paragraph is true (RAN).
   - "novel reads" is at `DescLiveness.lean:58` and `lean-audit.json:70`, `:78`.
   - "IU" (`IUs` included) is in the lock's reasons, `lean-audit.json:47`, `:62` and `:78`, and in docstrings, `desc_difftest_vectors.py:304` and `difftest_vectors.py:422-424`.
   - "isolation units" is only at `lakefile.toml:2`.
   - `ius` is the key and its computation: `Check.lean:142`, `:186`, `:189`, `:209`, `Difftest.lean:23`, `DescLiveness.lean:39`, `Protocol.lean:132`, the reference's `:3` and `:135-148`, and the vectors.
   - The failure rule `"nci"` is at `Check.lean:150`.
   - No "NCI", "novel communication" or "novel information" remains.
4. Check 4 holds.
   - `tools/pous_grader/TRUSTED.sha256` lists no README (RAN grep).
   - `tests/test_retired_terms.py`, `tests/lint/test_repo_retired_terms.py` and `tests/test_doc_paths.py` at the head: 50 passed, 7 bookkeeping warnings, none naming this README (RAN). The tree differs from 016c1313e173 only in this README, so the warnings predate it (INFERRED).
5. Check 5, your ruling, I question mildly. `desc_difftest_vectors.py:305`, "`B`'s gates are novel to the first inference block", means fresh input to that block, so it is the older term's sense in a docstring, not an unrelated English "novel" (READ). Leaving it out of the paragraph is defensible. Rewording it "are fresh input to" would close it without listing it (INFERRED).

## For the lead to read

1. `verity/protocols/policy/inference_only/verifier/README.md:130-136`, the paragraph.
2. `lean/InferenceOnlyCheck/Check.lean:110-130`, the `*-iu` refusal codes.
3. `lean/scripts/theta_difftest_vectors.py:365-369`, the only writer of `baseline_difftest/`.
4. `lean/InferenceOnlyCheck/ThetaDifftest.lean:49-54`.

No node run launched. No events, PR comments or run outputs reached me this turn.

