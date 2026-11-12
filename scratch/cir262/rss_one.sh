#!/usr/bin/env bash
# circuits FP8READ: the memory one token's evaluation of FlashMlaV32Decode_v1 takes, on the cores of the calendar row (no GPU,
# no lease): it sizes the workers of the off-card job. CAP is a saved capture's out/ directory on this machine.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run this under research run}; OUT=${OUT:-$D/out}; mkdir -p "$OUT"
CORES=${CORES:?CORES}; CAP=${CAP:?CAP}
[ -d "$CAP/capture" ] || { echo "rss: no capture at $CAP"; exit 2; }
[ -d /workspace/venv312 ] && venv=/workspace/venv312 || venv=/workspace/cp/venv312
PY=$venv/bin/python
ROOT=$PWD
cd adapters/vllm || exit 2
if ! "$PY" -c "import verity.definitions.registry.flashmla_dsv32" 2> /dev/null; then
  pp=$(cd "$ROOT" && PYTHONPATH=tools/research/src "$PY" -m research env --source . 2> /dev/null | sed -n 's/^export PYTHONPATH=//p')
  [ -n "$pp" ] && export PYTHONPATH="$pp"
fi
export PYTHONHASHSEED=0
for ct in "d1 0" "d67 0"; do
  set -- $ct
  taskset -c "$CORES" "$PY" - "$CAP/capture" "$1" "$2" <<'EOF' | tee -a "$OUT/rss.txt"
import resource, sys
from tests.properties import glm53_flashmla_fp8_capture_gpu as X
r = X.definition_token({"dir": sys.argv[1], "case": sys.argv[2], "token": int(sys.argv[3])})
print("rss:", sys.argv[2], sys.argv[3], {k: r.get(k) for k in ("words", "differ", "error")},
      "maxrss_MiB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, flush=True)
EOF
done
