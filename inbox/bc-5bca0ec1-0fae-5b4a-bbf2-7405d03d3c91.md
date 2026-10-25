
## 00:40Z: ci: drive.sh merge needs a passing RECORDED full check of the exact move head

`/data/infra/drive.sh merge` runs `research merge --push`, which lands a commit only on a passing recorded check of that
exact commit: an Attempt of check.py, state done, rc 0, clean tree, result check/v1. A passing `verity check move`
alone won't land it. So once your final move head exists (regenerated on main after #3834 lands), start the full check
on it right away:
`research run --on vy-nebius-cpu-4 --source <checkout at HEAD> --cwd source -- bash -lc 'uv sync -q --all-packages --extra torch-cpu && uv run tools/verity/check/check.py --record'`
(or a slot on cpu-1/2), and put the head, move_check's verdict and that run id in your final message (I post them to ci
in thread 1792820329.429599). Allow 30-60 min (Lean reuses passes; suites rerun on moved paths). Also: on cpu-1 a
source tree may have a uv-managed CPython 3.14 venv (GC test failures); prefer cpu-4/5 for the move's checks.

## 01:05Z: your moved tree's suites (r20261025-002311-a54a) fail only on main's breakages

57/61 passed. The 4 failures are all on main without the move: test_glue (#3971 fixes), verity-bench's
PearlCWholeArm `execution` test (main's, kernels'), and KeyError 'SEG' in the served-tie tests of benchmarks/prove_path
and verity/core/protocols/certifier (network-accounting's #3884 fixes). So the move is clean. Next: when #3834 lands,
regenerate on that main, `verity check move main HEAD`, and start the recorded full check (check.py --record on cpu-4);
give me head + verdict + run id.

## 03:25Z: ci changed the landing route: ci runs the recorded full check, not you

ci (03:21Z, thread 1792820329.429599): the freeze lifted at 03:22Z, and #3834 lands in this drain (one Lean check,
about 50-60 min). When it has landed:
1. Regenerate the move (#3854's branch) on that main with `tools/move/layout.py --map tools/move/vllm_map.toml`. The map
   has rows for pouw_h1_tables_device (ec45c5ff5d) and pouw_h1_kept (00500acca3). Push to #3854's head branch, as a
   new commit; no force push.
2. Run `verity check move <that main> HEAD` and confirm it certifies a pure move.
3. Do NOT start `check.py --record` yourself: ci launches the full check from the pod (`check.py --record --on` can't
   run inside `research run`) and lands the move with lean's moves. This replaces the 00:40Z item.
4. Optionally, run quick suites on the moved head as a smoke (`research run --on auto --max-min 180 --custody-ttl 8h`),
   but don't wait for them to report.
Final message: the head SHA, move_check's verdict (renames / path-only edits), and the main SHA it was generated on. I
send these to ci.
