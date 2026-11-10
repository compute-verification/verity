#!/usr/bin/env bash
# vllm-twostage (VLLM-250): advice_c.py over a kept (c) serve's private dir, on the serving node's CPU, under research run
# with --cwd source from the tree the serve ran (twostage_c.sh advice RUN_C):
#   research run --on vy-spot-glm53 --project verity --source TREE --cwd source --timeout 20m --mem-gb 8 \
#     --send advice_job.sh --send advice_c.py --env RUN_C=RUN -- bash -c 'bash $RESEARCH_RUN_DIR/inputs/advice_job.sh'
# No GPU (CUDA_VISIBLE_DEVICES empty), 4 threads, niced, and an address-space cap of MEM_GB (default 8) GB: --mem-gb with
# --on only reserves. Reads PRIV=SERVE_ROOT/RUN_C (default /workspace/research/vllm-twostage) and writes each window's
# advice and its own log under PRIV/advice-runs/<this run>/ (0700); OUT gets only advice/w<k>/registration.json, the
# binding alone. Stdout: advice_c.py's own lines (window labels and verdicts), and on a failure the exception's class and
# the code lines it passed through, never its message, which can name a request.
set -u
D=${RESEARCH_RUN_DIR:?run this under research run}; OUT=$D/out; mkdir -p "$OUT"
P=${SERVE_ROOT:-/workspace/research/vllm-twostage}/${RUN_C:?RUN_C: the serve run whose private dir is read}
A=$P/out/boundary-rows
[ -d "$A/server/boundary" ] || { echo "advice: $RUN_C kept no boundary-rows arm"; exit 2; }
W=$P/advice-runs/$(basename "$D")
(umask 077; mkdir -p "$W") || { echo "advice: the private dir takes no advice-runs/"; exit 2; }
[ -d /workspace/cp/venv312 ] && v=/workspace/cp/venv312 || v=/workspace/venv312
v=${VENV:-$v}
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH=$PWD/integrations/vllm:$PWD:$D/inputs
ulimit -v $(( ${MEM_GB:-8} * 1024 * 1024 ))
# the imports first, before anything private is read, so a missing module says so here
nice -n 10 "$v/bin/python" -c 'import advice_c' || { echo "advice: advice_c.py does not import under $v"; exit 2; }
echo "advice: from $(date -u +%T), tree $(git rev-parse --short HEAD 2>/dev/null || echo shipped), no GPU, ${MEM_GB:-8} GB cap"
(umask 077; nice -n 10 "$v/bin/python" "$D/inputs/advice_c.py" "$A" "$W" "$OUT" --controls > "$W/advice.log" 2>&1); a=$?
grep "^advice-c:" "$W/advice.log"
if [ $a != 0 ] && ! grep -q "^advice-c: \(the cut does not\|no Commit record\|control\)" "$W/advice.log"; then
  echo "advice: failed in $(grep -oE '^[A-Za-z_.]+(Error|Exception|Exit|Interrupt)\b' "$W/advice.log" | tail -n 1), through:"
  grep -E '^  File "' "$W/advice.log" | tail -n 4 | sed -E 's/^  File "([^"]*)", line ([0-9]+), in (.*)$/  \1:\2 \3/'
fi
keep=(! -path "./advice/*/registration.json")
stray=$(cd "$OUT" && find . -type f "${keep[@]}" | wc -l)
if [ "$stray" -gt 0 ]; then
  (umask 077; mkdir -p "$W/stray") && (cd "$OUT" && find . -type f "${keep[@]}" -print0 | xargs -0 -r cp --parents -t "$W/stray" \
    && find . -type f "${keep[@]}" -delete)
  echo "advice: moved $stray other file(s) from OUT to the private dir"
fi
echo "advice: exit $a at $(date -u +%T) (its log: the private dir's advice-runs/$(basename "$D")/advice.log)"
exit $a
