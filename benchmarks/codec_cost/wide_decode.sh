#!/usr/bin/env bash
# The wide-word codec (benchmarks/codec_tensor: wt.cu, bench.py) at 1,536 to 4,096 bits, plain and masked: gates against
# wide.py, then the read rate, compute-only rates and the standalone store decode at k = 1, 2 (cwd = the source tree).
# 3,072 and 4,096 are not wide.py's fields, so a scratch copy of codec_tensor gets p = 2^3072 - 2165 and 2^4096 - 2549
# (both 3 mod 4) and the tree is left as it is. Builds run outside the lease, the bench under one gpu-lease. $OUT gets
# codec-tensor.json and bench.log.
#
#   research run --on vy-nebius-2 --queue --gpu-host --preemptible --cpus 8 --mem-gb 32 --cwd source \
#     -- bash benchmarks/codec_cost/wide_decode.sh
set -uxo pipefail
O=${OUT:?}
SRC=$PWD
ENV_DIR=${ENV_DIR:-/workspace/pouw/fill-out/pous/work/pous-bench}
PY=$ENV_DIR/gpu312-torch2.14.0-cu130/bin/python
[ -x "$PY" ] || { echo "no torch venv at $PY"; exit 3; }
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0} PATH=/usr/local/cuda-13.0/bin:$(dirname "$PY"):$PATH
B=$(mktemp -d /tmp/codec-wide.XXXXXX); trap 'rm -rf "$B"' EXIT
mkdir -p "$B/benchmarks"
cp -r benchmarks/codec_tensor "$B/benchmarks/"
ln -s "$SRC/benchmarks/word_codec" "$B/benchmarks/word_codec"
ln -s "$SRC/verity" "$B/verity"
CT=$B/benchmarks/codec_tensor
python3 - "$CT" << 'EOF'
import sys
from pathlib import Path
ct = Path(sys.argv[1])
edits = {
    "wide.py": [('Field("p2048", 2048, 1557))}',
                 'Field("p2048", 2048, 1557), Field("p3072", 3072, 2165), Field("p4096", 4096, 2549))}'),
                ('"p2048": 32}', '"p2048": 32, "p3072": 22, "p4096": 16}')],
    "wt.cu": [('#else\n#error "WT_FIELD is one of wide.FIELDS"',
               '#elif WT_FIELD == 3072\nconstexpr int NBITS = 3072, TW = 22;\nconstexpr u32 CF = 2165;\n'
               '#elif WT_FIELD == 4096\nconstexpr int NBITS = 4096, TW = 16;\nconstexpr u32 CF = 2549;\n'
               '#else\n#error "WT_FIELD is one of wide.FIELDS"')],
}
for name, pairs in edits.items():
    p = ct / name
    s = p.read_text()
    for old, new in pairs:
        assert s.count(old) == 1, (name, old)
        s = s.replace(old, new)
    p.write_text(s)
sys.path.insert(0, str(ct))
import wide
wide.check_fields()
EOF
FLAGS=(-O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v)
FIELDS=${FIELDS:-"1536 2048 3072 4096"}
declare -A PIDS=()
BUILDS=()
for n in $FIELDS; do
  for m in 0 1; do
    timeout "${BUILD_S:-1500}" nvcc "${FLAGS[@]}" -DWT_FIELD="$n" -DWT_MASK=$m -o "$B/wt-$n-m$m.so" "$CT/wt.cu" \
      > "$O/wt-$n-m$m.ptxas" 2>&1 & PIDS[$n-m$m]=$!
    BUILDS+=("$n-m$m")
  done
done
LIBS=()
for b in "${BUILDS[@]}"; do
  if wait "${PIDS[$b]}"; then LIBS+=(--lib "$B/wt-$b.so"); else echo "build $b rc $?" >> "$O/errors.txt"; fi
done
[ ${#LIBS[@]} -gt 0 ] || exit 4
sha256sum "$B"/*.so > "$O/libs.sha256"
export O PY CT LIBS_ARG="${LIBS[*]}"
LEASE=(gpu-lease 1 --wait --preemptible --mem-gb 16 --max-min "${MAXMIN:-20}" --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  q() { nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,temperature.gpu --format=csv,noheader \
          -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1; }
  q
  # shellcheck disable=SC2086
  "$PY" "$CT/bench.py" $LIBS_ARG --out "$O" --phases rates,decode --store-gib 1 --gate-words 64; rc=$?
  q
  exit $rc'
