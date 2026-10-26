#!/usr/bin/env bash
# k27_peaks.py on a CPU machine (PRF-172: the peak memory of one drawn unit of each k_log 27 subcircuit of M3's draw under
# C-Flock --zk), on hidden_zk's builds (its job.sh, sourced: C-Flock's flock-circuit and the Lean flock-verify, cached under
# $RS_CACHE) and the workspace's own environment:
#   research run --on vy-nebius-cpu-1 --pool research --cpus 32 --mem-gb 200 --cwd source -- \
#     bash -c 'RS_KEYS=chunk:8192:1:cut64 RS_MEM_GB=200 RS_CACHE=/workspace/research/cache/pouw-hidden-zk bash benchmarks/one_stage/k27_peaks.sh'
# Each key of RS_KEYS (';'-separated) is staged, proved under --zk, verified by the Lean verifier and verified again with one bit of
# its proof flipped, under `ulimit -v` of 95% of RS_MEM_GB. Records: $OUT/<key>/k27.json, k27.log and each process's times/*.time.
set -uo pipefail
uv sync -q --all-packages --extra torch-cpu || exit 9
export HZ_CONFIGS=k27 HZ_MEM_GB=${RS_MEM_GB:-200} HZ_PYTHON=$PWD/.venv/bin/python
[ -n "${RS_CACHE:-}" ] && export HZ_CACHE=$RS_CACHE
source benchmarks/pouw/hidden_zk/job.sh
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm
setup
rc=0
IFS=';' read -ra KEYS <<< "${RS_KEYS:?}"
for k in "${KEYS[@]}"; do
  tag=${k//[:+]/_}
  mkdir -p "$OUT/$tag"
  log "k27 $k"
  ( ulimit -v $(( MEM_GB * 95 / 100 * 1024 * 1024 )); "$PY" benchmarks/one_stage/k27_peaks.py --key "$k" --out "$W/$tag" --bin "$BIN" \
      --lean "$LEAN" --timed "$J/timed.py" ) > "$OUT/$tag/k27.log" 2>&1
  r=$?
  grep '^K27 ' "$OUT/$tag/k27.log" | cut -c1-2000 | tee -a "$OUT/job.log"
  [ $r = 0 ] || { log "k27 $k: rc $r"; tail -30 "$OUT/$tag/k27.log" | tee -a "$OUT/job.log"; rc=1; }
  (cd "$W/$tag" && find . -type f \( -name '*.json' -o -name '*.log' -o -name '*.time' \) -print0 | xargs -0 -r -I{} cp --parents {} "$OUT/$tag/")
  rm -rf "$W/$tag"
done
log "k27: done, rc $rc"
exit "$rc"
