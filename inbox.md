[2026-11-14T08:29Z] 955fb0c57285
READSTEP, Sat 14 Nov, 12:29 AM PST. Re-read of #10012, delta `35a3745..5be9974`.

**The delta is the cure as written and nothing else: the verdict is one function, its boundary is held by a test, and the `>=` mutation fails that test by name.**

The verdict is for 5be997485a91e22dbaf9dfd4fbd446ec1a95aa82, the branch's head when I sent this (RAN, `git ls-remote`).

## The delta

It is one commit, 5be997485a9, whose parent is 35a3745 (RAN, `git log`). It touches two files (RAN, `git diff`):
- `benchmarks/inference_only/report.py:47-49` adds `verdict(value)`: `"within target" if value >= TARGET else "misses"`. The runner's call at `:392-393` is now `verdict=verdict(doc["value"])`.
- `tests/test_inference_only.py:140-143` adds `test_the_metrics_verdict_holds_its_value_to_the_goals_target`, parametrized over (1000, `within target`), (999.999, `misses`) and (1, `misses`).

## One comparison

No second comparison against the target survives (RAN):
- `grep` over `benchmarks/inference_only/*.py` outside tests finds `TARGET` only at `:37` (its definition), `:49` (`verdict`) and `:392` (`target=TARGET`).
- `git grep` of the whole tree for the metric's id, `report.TARGET`, `report.verdict` and `[<>]=? 1000` in `benchmarks/inference_only` and `research`'s `metrics.py` finds no other comparison.

## On the VM at the head (RAN)

- `pytest tests/test_inference_only.py`: "24 passed in 6.71s", the earlier 21 and the 3 new cases.
- The mutation `value >= TARGET` to `value > TARGET` at `report.py:49` gives "1 failed, 23 passed". The one failure is `test_the_metrics_verdict_holds_its_value_to_the_goals_target[1000-within target]`. After the revert the worktree is clean.
- `ratchet_gate.py 512644d14fb 5be9974`: "1 change(s), 0 failed".

## The body

It is true at this head (READ, `gh pr view`, updated 08:26Z):
- It names `report.verdict(value)` and the new test's three cases.
- It keeps `Goal training/small-model-slowdown-1000` and `Refs TRN-158`, and names no machine.
- My first report's other findings stand as they were, all NOTEs.

## Runs

The body names none, and I saw no run at this tree.

READY 5be997485a91e22dbaf9dfd4fbd446ec1a95aa82

MUST FIX: none. SHOULD FIX: none. NOTE: none new.

No other event reached my conversation.

