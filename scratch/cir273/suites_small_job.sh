# repository and verity/definitions only, run with `research run --on auto --project verity --source CHECKOUT --cwd source
# --mem-gb 32 --timeout 90m -- bash -lc "$(cat scratch/cir273/suites_small_job.sh)"` (the tests held to 24 GB).
uv sync -q --all-packages --extra torch-cpu </dev/null && git rev-parse HEAD > "$OUT/head.txt"
.venv/bin/python tools/verity/check/suites.py --synced verity/definitions repository --fresh --timeouts --mem-gb 24 \
  --json "$OUT/suites.json" > "$OUT/suites.log" 2>&1
rc=$?
tail -12 "$OUT/suites.log"
exit $rc
