
## 00:40Z: ci: drive.sh merge needs a passing RECORDED full check of the exact move head

`/data/infra/drive.sh merge` runs `research merge --push`, which lands a commit only on a passing recorded check of that
exact commit: an Attempt of check.py, state done, rc 0, clean tree, result check/v1. A passing `verity check move`
alone won't land it. So once your final move head exists (regenerated on main after #3834 lands), start the full check
on it right away:
`research run --on vy-nebius-cpu-4 --source <checkout at HEAD> --cwd source -- bash -lc 'uv sync -q --all-packages --extra torch-cpu && uv run tools/verity/check/check.py --record'`
(or a slot on cpu-1/2), and put the head, move_check's verdict and that run id in your final message (I post them to ci
in thread 1792820329.429599). Allow 30-60 min (Lean reuses passes; suites rerun on moved paths). Also: on cpu-1 a
source tree may have a uv-managed CPython 3.14 venv (GC test failures); prefer cpu-4/5 for the move's checks.
