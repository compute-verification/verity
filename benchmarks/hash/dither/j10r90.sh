#!/usr/bin/env bash
# J10-R's per-call-noise units (h1u.cu's kh_units_multi_j10r and _sm; PouwH1Tile_v5 / _v6's receipt) on the H200, sm_90a
# the served RTX arms built for sm_90a with nothing else changed, and the sm_90a arm:
#   kr        the receipts-only base (job.sh's kr: H1U_KSPLIT = 4, H1K_AUTO, H1K_RONLY)
#   kra       A: kr + H1U_JCTA, H1U_WARPS = 4, H1U_JX, H1U_JBW
#   kraq      AQ: kra + H1U_PF, H1U_JPI, H1U_JCS = 2 (H1U_JXS stays off: it stalls under PREQ on the RTX)
#   kraqwxsc  the sm_90a arm: kra + H1U_PF, H1U_JPI, H1U_JW, H1U_JXS, H1U_JWS, H1U_JPC (no JCS: the column tiles and
#             W' once a call, X and W' in fragment order, their prep's stores coalesced)
#
#   build  (a CPU node) job.sh's build of the four at sm_90a with --no-compress ($OUT/lib/sm_90a), then the SASS gate on
#          each: sass-gate.json on sass_pins.json as it stands (the verdict, the script's exit), and sass-gate-sm90pin.json
#          on a scratch copy of the gate whose 13.0.1-sm_90a entry is pinned as jobs/pin_target.sh pins it (ieee_pin.cu's
#          pinned builds at sm_90a, sass_gate.py --pin) and given the numeric run that pinned it on the H200,
#          j10r90.json's sm90_pin (ieee_pin.cu, ieee_pin.py, pin_target.sh and the gate as they are now), an entry that
#          never landed: what the gate would flag once it does
#   check  (an H200, under a GPU lease) job.sh's j10r90: unitsbench.py's J10-R gates at its own shapes, then j10r90.py
#          check at the frozen shape
#   time   (an H200, under a GPU lease) job.sh's j10r90t: j10r90.py time; with an s8-ship.tar sent
#          (s8_linear/build.sh's ship plus its wgmma.py and configs.py), wgmma.linear as a second plain arm
#
#   OUT=B bash benchmarks/hash/dither/j10r90.sh build                     (a CPU host, up to 30 minutes)
#   (or on the H200's host, outside any lease; where the host has no /workspace/cp/venv312, VERITY_BENCH_VENV names a
#   torch venv for check and time)
#   OUT=R LIBS=B/lib/sm_90a GPU_LEASE_WHO=WHO bash benchmarks/hash/dither/j10r90.sh check   (the H200, up to 30 minutes)
# then on a CPU host with check's j10r90-check.json and .npz: j10r90.py verify and j10r_units.py anchor.
# Env: NVCC (default CUDA 13.0's), LEASE (default "--max-min 20"), HB_WHO (default $GPU_LEASE_WHO, else dither/ and the run's id), BENCH_ARGS.
set -uo pipefail
WHAT=${1:?usage: j10r90.sh build|check|time}
HERE=$(cd "$(dirname "$0")" && pwd)
PIN_RUN=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["sm90_pin"]["run"])' "$HERE/j10r90.json") || exit 2
ROOT=$(pwd -P)
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
export NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
BASE=-DH1U_KSPLIT=4,-DH1K_AUTO=1,-DH1K_RONLY=1
A=$BASE,-DH1U_JCTA=1,-DH1U_WARPS=4,-DH1U_JX=1,-DH1U_JBW=1
Q=$A,-DH1U_PF=1,-DH1U_JPI=1
mkdir -p "$OUT"
case $WHAT in
  build)
    BUILDS="" VARIANTS="kr=$BASE kra=$A kraq=$Q,-DH1U_JCS=2 kraqwxsc=$Q,-DH1U_JW=1,-DH1U_JXS=1,-DH1U_JWS=1,-DH1U_JPC=1" \
      NVCC_EXTRA=--no-compress \
      bash "$HERE/job.sh" sm_90a build || exit 2
    LIB=$OUT/lib/sm_90a
    H=$ROOT/benchmarks/pouw/harness
    export POUW_CUDA_BIN=$(dirname "$NVCC") POUW_SASS_GATE_CACHE=none
    python3 "$H/sass_gate.py" "$LIB"/h1u_kr*.so --json "$OUT/sass-gate.json" > "$OUT/sass-gate.txt" 2>&1
    rc=$?
    echo "sass gate (sass_pins.json as it stands): rc $rc"; cat "$OUT/sass-gate.txt"
    W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
    cp "$H/sass_gate.py" "$H/sass_pins.json" "$H/ieee_pin.py" "$W/"
    python3 - "$W" 13.0.1-sm_90a > "$W/builds" <<'EOF'
import sys
sys.path.insert(0, sys.argv[1])
import ieee_pin as I
for i, flags in enumerate(I.PINNED):
    print(i, " ".join(I.nvcc_args(I.arch_of(sys.argv[2])) + flags))
EOF
    while read -r i args; do
      # shellcheck disable=SC2086
      (cd "$W" && "$NVCC" $args "$H/ieee_pin.cu" -o "$W/pin$i") || exit 2
    done < "$W/builds"
    python3 "$W/sass_gate.py" "$W"/pin? --pin --toolkit 13.0.1-sm_90a > "$OUT/sm90pin.txt" 2>&1 || { cat "$OUT/sm90pin.txt"; exit 2; }
    python3 - "$W/sass_pins.json" "$HERE/j10r90.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
d["toolkits"]["13.0.1-sm_90a"]["numeric"] = json.load(open(sys.argv[2]))["sm90_pin"]
json.dump(d, open(sys.argv[1], "w"), indent=1)
EOF
    cp "$W/sass_pins.json" "$OUT/sass_pins.sm90pin.json"
    python3 "$W/sass_gate.py" "$LIB"/h1u_kr*.so --json "$OUT/sass-gate-sm90pin.json" > "$OUT/sass-gate-sm90pin.txt" 2>&1
    echo "sass gate under the sm_90a pin (diagnostic): rc $?"; cat "$OUT/sass-gate-sm90pin.txt"
    exit $rc ;;
  check|time)
    old=/workspace/research/runs/$PIN_RUN/out/sass_pins.json
    [ -f "$old" ] && cp "$old" "$OUT/sass_pins.$PIN_RUN.json"
    s8=${LIBS:-${RESEARCH_RUN_DIR:-.}/inputs}/s8-ship.tar
    if [ "$WHAT" = time ] && [ -f "$s8" ]; then
      mkdir -p "$OUT/s8_linear" && tar -C "$OUT/s8_linear" -xf "$s8" || exit 2
      export BENCH_ARGS="--wgmma $OUT/s8_linear ${BENCH_ARGS:-}"
    fi
    HB_WHO=${HB_WHO:-${GPU_LEASE_WHO:-dither/${RESEARCH_RUN_ID:-local}}} LEASE=${LEASE:---max-min 20} \
      exec bash "$HERE/job.sh" sm_90a "$([ "$WHAT" = check ] && echo j10r90 || echo j10r90t)" ;;
  *) echo "j10r90.sh: no $WHAT (build, check or time)" >&2; exit 2 ;;
esac
