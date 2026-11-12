[2026-11-12T22:15Z] 9200948703cf
SORT, TRN-146: carry of #8731. What it is for: SPOT round 2 holds an interior's history tables to round 1's record. An interior whose copies name a table that is not a port of the boundary registration's record, at that port's last root and domain, is refused with `interior-table`, in both the Lean verifier (`Spot.Run.tablesServed`) and the stream (`service.tables_served`). READ (the PR body).

- Branch `cursor/carry-8731-dc90`, head e0c95da403838c4744dcc24c325aca4979850e09: #8731's head with `main` 30d6fcc92ff7 merged in. RAN (`git ls-remote`; `git merge-base --is-ancestor` on the PR head).
- How it crossed the moves (RAN). Restack refuses this branch as "based-after": it holds #8448's pre-move head 1113b8a667, and #8448 landed after the first two moves. So I followed #8448's own crossing:
  - I merged #8448's branch at 46568eebd0, which holds the first move's base.
  - For the first two moves (6581bcf197, c1ca25a247), I called restack's own `restack()` with its `based_after` guard lifted for that step only. Each step ran the move's script on the branch and merged #8448's restacked commit (9174e2624e, then ddf92e9ff6, after merging a2655f4dce) with the move as merge base, so #8448's lines come in once.
  - #8448 landed before the third move's base, so after merging that base (33faab2be81c), restack crossed 70395628ec normally.
- Conflicts I resolved (no proof or statement judgement; RAN):
  - `Spot/OneStage.lean`: `Params` takes both main's `tied` and this branch's `tables`.
  - `Spot/Run.lean`: the docstring and definitions keep main's `interiorWindowOk` (`interior-window`, R1–R8) and this branch's `tablesServed` (`interior-table`). In `interiorSteps` the order is `interior-window` first, then `population-order`, then `interior-table`; `service.py`'s `register_interiors` has the same order.
  - `Spot/Pieces.lean`'s table and `register_interiors`' docstring: both sides (main's `across` and `replay_returns`; this branch's `tables_served`).
  - `test_interior_table.py`: #8448's later commits build no program at import, and `main` added a variable to `_opt_in`. #8731's forged-port test now calls `_program()` in place of `PROGRAM`, carries the module's opt-in (`VERITY_VLLM_INTERIOR_TABLE`) and `slow` marks, and runs on a stream of its own.
- The diff against `main` is 8 files: the three Spot Lean files, `service.py`, `tests/test_history_difftest.py`, `adapters/vllm/tests/pipeline/test_interior_table.py`, and the two locks. RAN (`git diff --stat`).
- Quick suites `r20261112-204006-fd1c` on 810e994414, rc 0, last line: `67 suites: 67 passed (0 cached), 0 failed; 169.5 min of test time`. RAN.
- The opt-in tests ran under `VERITY_VLLM_INTERIOR_TABLE=1` as `r20261112-204034-6645`, rc 0, with both selected tests passing: the new forged-port test and `test_a_table_round_1_did_not_register_is_refused`. RAN.
- `verity lean try …/Spot/Pieces.lean --siblings --on auto`, run `r20261112-194728-1fd9`: `lean exit 0 in 531 s`, no sibling clash, and the axiom probe says "no theorems in it". RAN (fetched stdout.log).
- Records runs `r20261112-194517-77c3` (security) and `r20261112-194557-7845` (security/Properties) both ended with every `AUDIT … PASS` (axioms propext, Classical.choice and Quot.sound only). Both kept the same records, committed in e0c95da403. In each lock the only changes are the hash of `Spot.OneStage.paramsParsed` and the digest of `Spot.OneStage`'s reads. No guarantee's statement or premises change. RAN (a JSON comparison of the leaves of the kept records against the branch's locks).
- The Spot package's own records run, `r20261112-194433-1760`, is still running. Its audit lists 0 guarantees, so I expect no record from it; I'll report it if it keeps one.
- Question: none for a statement. Reviewers should read the union of `interior-window` and `interior-table` in `Run.lean` and `service.py`.

