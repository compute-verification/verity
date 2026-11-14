[2026-11-14T08:41Z] 238b91e1c239
**Four of the five hand commits on `cursor/training-rename-rehearsal-6-9af3` are right. The check README's Names paragraph (42ac6068) is false of the tree in two places, and a4535709 does more than the constant.**

READMAPS, Sat 14 Nov. Each commit's own diff was read against the tree at 016c1313e173 (READ). Its base fb8db33486f8 is an ancestor, and all five commits are in it (RAN).

READY b9fb40b93402f0d19975f67c21ec0e7bdcf752f9
NOT READY 1 42ac606835b6c8daee54886e4deed3516e7bb445
READY 1f69e8facb88256f842b58ba50b0b35e1699bd92
READY a26c12c0ae7308d1b014f8f278dff62aefc02221
READY a4535709e5ac456014fb0661fa88c49712fa4f4b (with its SHOULD FIX)

## MUST FIX (1)

1. 42ac6068, `verity/protocols/policy/inference_only/verifier/README.md:130-134`: the paragraph lists the older names that stay, and two of its claims are false of 016c1313e173.
   - `novel` is placed as "a variable and a mode of the liveness difftest". It is also the check's own variable in its rate rule: `let (size, novel) := r[u]!` at `InferenceOnlyCheck/Check.lean:149-150`, and `novel` in `scripts/inference_only_reference.py:141-144` (READ, RAN `git grep -w novel`).
   - "baseline" for the registered information is dropped from the list but stays in the check. It is in `Check.lean:29`, in a refusal message at `Check.lean:214` ("the baseline holds only"), in `Theta.lean:6` and in `lakefile.toml:1` ("its baseline θ"), which a4535709 rewrites and keeps (RAN `git grep -w baseline`).
   - The fix is to restore "baseline", and to say `novel` is the rate rule's variable, its reference's, and the liveness difftest's mode.

## SHOULD FIX (1)

1. a4535709 touches 9 files, not 2 (READ).
   - Beyond `run.py` and `inference_only_cost.py`, it changes:
     - the comments of `certifier/verifier/lean/lakefile.toml` and `inference_only/verifier/lean/lakefile.toml`;
     - the certifier's `lean-audit.json` `why` text;
     - the error message in `inference_only_reference.py`;
     - a comment in Flock's prover `pyproject.toml`;
     - test locals: `test_merge.py`'s `NCI` becomes `INFERENCE_ONLY`, and `test_parties.py`'s becomes `POLICY`.
   - Its subject says that wider sweep, so the brief's one-line description is narrower than the commit.
   - Each change is a word for word rename of an `NCI` name or word, and nothing reads the old message (RAN grep).
   - `test_mains.py` 13 passed, `test_parties.py` 17 passed and `test_merge.py` 82 passed at 016c1313e173 (RAN).
   - For the lead: whether a lock's `why` and two lakefiles belong in this commit.

## NOTE

1. b9fb40b9 is true (READ). The Lean's `Slowdown*` names are the slowdown: the Glossary reads `SlowdownRemote` as the slowdown lower bound (`GLOSSARY.md:91`). The same README still says "overhead" for `SlowdownRemote` at lines 27, 395 and 396, outside this commit's lines (RAN grep). The retired phrase is "overhead lower bound", which those lines don't use (READ `tests/test_retired_terms.py:150`).
2. The rest of 42ac6068 holds (RAN grep):
   - `benchmarks/nci` exists, and `"nci"` is the failure rule (`Check.lean:150`, quoted in `security/Properties/InferenceOnly.lean:250`).
   - `NciHandBlk` and `NciDecl…` are among the generators' toy names, with `NciShared…`, `NciWalk…` and `NciBase…` beside them, by example.
   - `baseline_difftest/` is written by `theta_difftest_vectors.py`, and its files are pinned in the lock (`lean-audit.json:19-26`).
   - "IU" stays in the lock's reasons and in docstrings.
3. 1f69e8fa holds: `InferenceOnlyCheck.Properties.Expands` is `security/Properties/InferenceOnly/Desc.lean:47`, in that namespace (lines 26-61), and no `NciDesc` is left in any Lean file (RAN).
4. a26c12c0 does what its line says (RAN, with `OUT` and `RESEARCH_RUN_DIR` unset):
   - **Set** (`--no-policy`): exit 2, "NCI_COST_FLAGS is set: the variable is INFERENCE_ONLY_COST_FLAGS".
   - **Set but empty:** the same, exit 2. That is what its message says, since `${NCI_COST_FLAGS+set}` counts empty as set.
   - **Unset:** passes the guard and stops at `OUT`, exit 1.
   - The guard sits before the build, and the tree stayed clean.
   - The header's `--on vy-nebius-cpu-2` names a machine, but it is an unchanged context line (READ).
5. No commit writes a run, PR, issue or agent id, a date or a machine name, and none brings back a retired word (READ the five diffs).

Received and not acted on: an @training broadcast (08:24Z, the model rule). @training's 08:32Z message on #9968's fix head is to me, and I read it after this.

