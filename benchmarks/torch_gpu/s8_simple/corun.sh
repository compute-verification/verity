#!/usr/bin/env bash
# crosscheck.py and corun.py on one leased GPU of an H200 node (cwd = a source tree with benchmarks/torch_gpu/sha512_rows
# and, for wgmma, s8_linear beside s8_simple). Before the lease, with no GPU: a Python with torch (the jobs' venv, else
# check.sh's torch 2.13.0+cu129 in scratch), the commit engine's build (nvcc), KP2-S8's ship (build.sh's, sent as
# s8-ship.tar) held to its MANIFEST, and for crosscheck the served weights (check.py weights, from the sent weights.py).
# Then one lease of at most LEASE_MIN (default 10) minutes: crosscheck.py if STEPS names it, then corun.py once for each
# of GEMMS (default "simple"), each with corun.py's options from the command line.
#
#   tar -C SHIP_DIR -cf /tmp/s8-ship.tar .
#   research run --on vy-dc-h200 --project verity --campaign e2e-demo --source . --cwd source \
#     --env GPU_LEASE_WHO=kernels-ks8simple --env STEPS="crosscheck corun" --env GEMMS="wgmma-tuned simple" \
#     --send /tmp/s8-ship.tar --send weights.py -- bash benchmarks/torch_gpu/s8_simple/corun.sh [corun.py's options]
#
# Exit: 0 every step passed (crosscheck: every word the same; corun: every co-run's words the alone runs' and the kernels'
# checks), 1 one didn't, 3 the venv, the build, the ship or the weights failed, else the lease's code.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run under research run}; OUT=$D/out; mkdir -p "$OUT"
HERE=benchmarks/torch_gpu/s8_simple ROWS=benchmarks/torch_gpu/sha512_rows
STEPS=${STEPS:-corun} GEMMS=${GEMMS:-simple}
TMP=/workspace/tmp; [ -w "$TMP" ] || TMP=${TMPDIR:-/tmp}
S=$(mktemp -d "$TMP/tmp-s8corun.XXXXXX"); trap 'rm -rf "$S"' EXIT
fail() { cat "$2" 2>/dev/null | tail -20; echo "$1 3" | tee "$OUT/status.txt"; exit 3; }
for v in /workspace/jobs/venv312 /workspace/cp/venv312; do [ -z "${PY:-}" ] && [ -x "$v/bin/python3" ] && PY=$v/bin/python3; done
PY=${PY:-python3}
if ! "$PY" -c 'import torch, triton, pynvml' 2>/dev/null; then
  export UV_CACHE_DIR=${UV_CACHE_DIR:-/workspace/cache/uv}
  # uv asks an --extra-index-url before the --index-url: torch and its CUDA wheels from cu129, the rest from PyPI
  { { uv venv -q -p 3.12 "$S/venv" || uv venv -q "$S/venv"; } && uv pip install -q -p "$S/venv/bin/python3" --index-url https://pypi.org/simple \
      --extra-index-url https://download.pytorch.org/whl/cu129 torch==2.13.0+cu129 numpy nvidia-ml-py; } > "$OUT/venv.txt" 2>&1 \
    || fail venv "$OUT/venv.txt"
  PY=$S/venv/bin/python3
fi
TV=$($PY -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null)
for c in "/usr/local/cuda-$TV" /usr/local/cuda /usr/local/cuda-13.0; do [ -x "$c/bin/nvcc" ] && { CUDA=$c; break; }; done
export PATH=${CUDA:-/usr/local/cuda}/bin:$PATH PYTHONPATH=$PWD/integrations/torch_loop:$PWD PYTHONDONTWRITEBYTECODE=1
export SHA512_ROWS_BUILD=${SHA512_ROWS_BUILD:-$HOME/.cache/verity/sha512_rows} TRITON_CACHE_DIR=$S/triton
git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}" > "$OUT/commit.txt"
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
case $ARCH in 90|100|120) ARCH=${ARCH}a ;; esac
PYTHONPATH=$PYTHONPATH:$PWD/$ROWS "$PY" -c "import engine as E; print(E.device_lib('$ARCH').path)" > "$OUT/build.txt" 2>&1 \
  || fail build "$OUT/build.txt"
if [ -f "$D/inputs/s8-ship.tar" ]; then
  mkdir -p "$S/ship" && tar -C "$S/ship" -xf "$D/inputs/s8-ship.tar" && (cd "$S/ship" && sha256sum -c --quiet MANIFEST.sha256) \
    > "$OUT/ship.txt" 2>&1 || fail ship "$OUT/ship.txt"
  cp "$S/ship/source.txt" "$S/ship/toolchain.txt" "$S/ship/MANIFEST.sha256" "$OUT/" 2>/dev/null
  export S8_WGMMA_CUBIN=$S/ship/s8_wgmma.cubin
fi
if [[ " $STEPS " == *" crosscheck "* ]]; then
  WM=(); [ -f "$D/inputs/weights.py" ] && WM=(--weights-module "$D/inputs/weights.py")
  mkdir -p "$S/w" && "$PY" "$HERE/check.py" weights --scratch "$S/w" --sets served "${WM[@]}" > "$OUT/weights.txt" 2>&1 \
    || fail weights "$OUT/weights.txt"
  cp "$S/w/weights.json" "$OUT/"
fi
gpu-lease status > "$OUT/leases-before.txt" 2>&1
echo "corun.sh: asking for one GPU at $(date -u +%H:%M:%SZ) for: $STEPS (gemms: $GEMMS)"
export OUT HERE PY S STEPS GEMMS
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${LEASE_MIN:-10}" -- bash -c '
  echo "corun.sh: leased $GPU_LEASE_UUID at $(date -u +%H:%M:%SZ)"
  q=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,power.draw,power.limit,temperature.gpu
  nvidia-smi --query-gpu=$q --format=csv -i "$GPU_LEASE_UUID" > "$OUT/gpu.txt" 2>&1
  rc=0
  for s in $STEPS; do
    case $s in
      crosscheck) "$PY" "$HERE/crosscheck.py" --scratch "$S/w" --out "$OUT/crosscheck" > "$OUT/crosscheck.out" 2>&1; r=$? ;;
      corun) r=0; for g in $GEMMS; do "$PY" "$HERE/corun.py" --out "$OUT" --gemm "$g" "$@" > "$OUT/corun-$g.out" 2>&1 || r=$?; done ;;
      *) echo "corun.sh: no step $s"; r=2 ;;
    esac
    echo "corun.sh: $s rc $r at $(date -u +%H:%M:%SZ)"; [ "$r" = 0 ] || rc=$r
  done
  nvidia-smi --query-gpu=$q --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$OUT/gpu.txt" 2>&1
  exit $rc' _ "$@"
rc=$?
echo "corun.sh: lease ended ($rc) at $(date -u +%H:%M:%SZ)"
echo "corun $rc" | tee "$OUT/status.txt"
cat "$OUT"/crosscheck/summary.txt "$OUT"/summary-*.txt 2>/dev/null
exit $rc
