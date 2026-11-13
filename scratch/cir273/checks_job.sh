# pins --stale, the changed suites (--quick) against the head's main, repository whole and the ratchet gate, run with
# `research run --on auto --project verity --source CHECKOUT --cwd source --mem-gb 64 --timeout 120m -- bash -lc "$(cat
# scratch/cir273/checks_job.sh)"` (the tests held to 48 GB).  Under `uv run`, so the venv's python3 is
# first on PATH: test_lean_verifier.py runs bare `python3` (r20261113-212217-c645 ran without it, and 7 tests lacked pytest).  B is the commit of main the head holds.
B=${B:-af9b13440693b1e06249f1c4352ea725205f03f1}
echo "head $(git rev-parse HEAD)" | tee "$OUT/head.txt"
git cat-file -e "$B^{commit}" || { echo "the base $B is not in this tree"; exit 3; }
uv sync -q --all-packages --extra torch-cpu </dev/null || exit 2
python3 tools/verity/check/pins.py --stale > "$OUT/pins_stale.txt" 2>&1; d=$?; cat "$OUT/pins_stale.txt"
uv run tools/verity/check/suites.py --synced --changed "$B" --quick --mem-gb 48 --json "$OUT/suites.json" \
  --logs "$OUT/logs" > "$OUT/suites.log" 2>&1; a=$?; tail -8 "$OUT/suites.log"
mkdir -p "$OUT/repository"
uv run tools/verity/check/suites.py --synced repository --mem-gb 48 --json "$OUT/repository/suites.json" \
  --logs "$OUT/repository/logs" > "$OUT/repository/suites.log" 2>&1; b=$?; tail -4 "$OUT/repository/suites.log"
uv run python adapters/vllm/tools/ratchet_gate.py "$B" HEAD --json > "$OUT/ratchet_gate.json" 2> "$OUT/ratchet_gate.err"; c=$?
python3 -c "import json,sys; print('ratchet failures', json.load(open(sys.argv[1]))['failures'])" "$OUT/ratchet_gate.json"
echo "pins --stale rc $d; changed-quick rc $a; repository rc $b; ratchet gate rc $c" | tee "$OUT/rcs.txt"
[ $a = 0 ] && [ $b = 0 ] && [ $c = 0 ] && [ $d = 0 ]
