set -uo pipefail
mkdir -p "$OUT"
echo "COVER tree $(git rev-parse --short=12 HEAD 2>/dev/null || echo '?') on $(hostname) from $(date -u +%H:%M:%SZ)"
export PYTHONPATH="$PWD:$PWD/benchmarks/numerical/python:$PWD/verity/protocols/flock/prover/python:$PWD/adapters/vllm:$PWD/tools/verity/circuit_check/src${PYTHONPATH:+:$PYTHONPATH}"
/usr/bin/time -v .venv/bin/python scratch/cir271/cover_list.py "$OUT" "${COVER_GUARD_GB:-24}" > "$OUT/cover_list.log" 2> "$OUT/cover_list.err"
rc=$?
grep '^COVER' "$OUT/cover_list.log" | cut -c1-300
tail -n 25 "$OUT/cover_list.err" | cut -c1-300
echo "COVER rc=$rc at $(date -u +%H:%M:%SZ)"
exit $rc
