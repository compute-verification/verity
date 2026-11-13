# the three suites, run with `research run --on auto --project verity --cwd source --wait --mem-gb 64 --timeout 120m --
# bash -lc "$(cat scratch/cir273/suites_job.sh)"` from the branch's checkout (a wait past 20 min exits 79; the run goes on).
uv sync -q --all-packages --extra torch-cpu </dev/null && git rev-parse HEAD > "$OUT/head.txt"
.venv/bin/python tools/verity/check/suites.py --synced verity/definitions verity-vllm repository --fresh --timeouts --mem-gb 48 \
  --json "$OUT/suites.json" > "$OUT/suites.log" 2>&1
rc=$?
tail -12 "$OUT/suites.log"
exit $rc
