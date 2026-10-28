#!/usr/bin/env bash
# commit_gpu on one leased GPU, sm_120 unless GENCODE names another (cwd = the source tree's root). Outside the lease:
# builds (by default the fastest measured form; BUILDS names others, a nomask tag a timing-only build without M_key y, a
# merged tag one without h's two compressions, the timing of a one-hash leaf format, and a quad tag one with a 4-ary tree)
# and reference.py's cases on the CPU. Under one lease: the selftest, each build's bit-exactness check against the
# reference (a build is timed only if its check passed), the raw compression rate, the e2e slice's o_proj operand A (8192
# x 4096 int8, 2^20 leaves) committed alone and beside cuBLASLt's int8 o_proj GEMM, and the leaf-size sweep over the same
# operand as raw int8 rows, as sha512/row/v2 rows and (from 1 KiB) as sha512/row-seg/v1 rows; then o_proj's operand at
# each prefill chunk (TILE_M, default 2048 8192 tokens) as e2e leaves and as tile leaves of R rows x 32 values (Decision
# 46; TILE_R as row/v2, TILESEG_R as row-seg/v1), beside the chunk's GEMM. Every latency is measured from the end of a
# stand-in producer kernel, the operand's last write. Clocks are read, never set.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --env GPU_LEASE_WHO=LANE \
#     -- bash benchmarks/pouw/commit_gpu/gpu.sh
# CHECK=quick|full (default full); REPS (default 9); BUILDS (default a0r0l; tags a<SHA_ADD>r<SHA_ROT>[l][nomask][merged][quad], l for
# SHA_ROLL=1);
# SWEEP and SEG_SWEEP (int8 values a leaf); TILE_M, TILE_R, TILESEG_R; GENCODE (default arch=compute_120a,code=sm_120a; H200:
# arch=compute_90,code=sm_90); PY, the Python that runs reference.py (default python3); LEASE, the command that leases the GPU
# (default gpu-lease 1 --wait --max-min 20 --; empty when the whole script already runs under a lease).
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/pouw/commit_gpu PY=${PY:-python3} PYTHONPATH=. GENCODE=${GENCODE:-arch=compute_120a,code=sm_120a}
W=$(mktemp -d /dev/shm/commit-gpu.XXXXXX)
export W
trap 'rm -rf "$W"' EXIT
git rev-parse HEAD > "$O/commit.txt" 2>&1
nproc > "$O/nproc.txt"

build() {   # build TAG: a<SHA_ADD>r<SHA_ROT>[l][nomask][merged][quad] (l: SHA_ROLL=1, else 0)
  local t=$1 add=${1:1:1} rot flags="-DSHA_ROLL=0"
  rot=${t#a?r}; rot=${rot%%[mnq]*}
  case $rot in *l) rot=${rot%l}; flags="-DSHA_ROLL=1" ;; esac
  case $t in *nomask*) flags="$flags -DCOMMIT_GPU_DIAG_NOMASK" ;; esac
  case $t in *merged*) flags="$flags -DCOMMIT_GPU_DIAG_MERGED" ;; esac
  case $t in *quad*) flags="$flags -DCOMMIT_GPU_DIAG_QUAD" ;; esac
  nvcc -O3 -std=c++17 -gencode "$GENCODE" -Xptxas -v -DSHA_ADD="$add" -DSHA_ROT="$rot" $flags \
    -o "$W/commit_gpu_$t" $D/commit_gpu.cu -lcublasLt -lnvidia-ml > "$O/build-$t.log" 2>&1
  echo "build $t rc $?" >> "$O/rc.txt"
}
export BUILDS=${BUILDS:-a0r0l}
for t in $BUILDS; do build "$t" & done
$PY $D/reference.py prepare "$W/cases" "${CHECK:-full}" > "$O/prepare.jsonl" 2> "$O/prepare.err"
echo "prepare rc $?" >> "$O/rc.txt"
export SWEEP=${SWEEP:-32 64 128 256 512 1024 2048 4096}
$PY $D/reference.py params "$W/p-e2e.bin" 1048576 1024
for L in $SWEEP; do $PY $D/reference.py params "$W/p-rows-$L.bin" $((33554432 / L)) $((8 * L)); done
export SEG_SWEEP=${SEG_SWEEP:-1024 2048 4096 8192 16384}
for L in $SEG_SWEEP; do $PY $D/reference.py params "$W/p-seg-$L.bin" $((33554432 / L)) $((8 * L)) --seg; done
export TILE_M=${TILE_M:-2048 8192} TILE_R=${TILE_R:-16 32 64 128} TILESEG_R=${TILESEG_R:-64 128}
for M in $TILE_M; do
  $PY $D/reference.py params "$W/p-e2e-$M.bin" $((128 * M)) 1024
  for R in $TILE_R; do $PY $D/reference.py params "$W/p-tile-$M-$R.bin" $(((M + R - 1) / R * 128)) $((256 * R)); done
  for R in $TILESEG_R; do $PY $D/reference.py params "$W/p-tileseg-$M-$R.bin" $(((M + R - 1) / R * 128)) $((256 * R)) --seg; done
done
wait
cat "$O/rc.txt"

export REPS=${REPS:-9}
LEASE=${LEASE-gpu-lease 1 --wait --max-min 20 --}
# shellcheck disable=SC2016,SC2086
$LEASE bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,power.limit --format=csv -i "$GPU_LEASE_UUID" \
    > "$O/gpu.txt" 2>&1
  timed=""
  for t in $BUILDS; do
    [ -x "$W/commit_gpu_$t" ] || continue
    case $t in
      *nomask*|*merged*|*quad*) timed="$timed $t" ;;
      *) "$W/commit_gpu_$t" selftest >> "$O/selftest.jsonl"; s=$?
         $PY $D/reference.py check "$W/commit_gpu_$t" "$W/cases" "$t" > "$O/check-$t.jsonl" 2> "$O/check-$t.err"; c=$?
         echo "selftest $t rc $s; check $t rc $c" >> "$O/rc.txt"
         [ $s = 0 ] && [ $c = 0 ] && timed="$timed $t" ;;
    esac
  done
  echo "timed:$timed" >> "$O/rc.txt"
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu --format=csv -lms 500 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  for t in $timed; do "$W/commit_gpu_$t" sha 20 > "$O/sha-$t.jsonl" 2>> "$O/bench.err"; done
  for t in $timed; do
    "$W/commit_gpu_$t" bench "$W/p-e2e.bin" e2e 1048576 1024 "$REPS" 8192 4096 4096 > "$O/bench-e2e-$t.jsonl" 2>> "$O/bench.err"
    echo "bench e2e $t rc $?" >> "$O/rc.txt"
  done
  best=$($PY $D/summary.py best "$O")
  echo "best: $best" >> "$O/rc.txt"
  for L in $SWEEP; do
    [ -n "$best" ] || break
    "$W/commit_gpu_$best" bench "$W/p-rows-$L.bin" rows $((33554432 / L)) $((8 * L)) "$REPS" 8192 4096 4096 \
      >> "$O/bench-sweep-$best.jsonl" 2>> "$O/bench.err"
    echo "bench rows L=$L $best rc $?" >> "$O/rc.txt"
  done
  for L in $SEG_SWEEP; do
    [ -n "$best" ] || break
    "$W/commit_gpu_$best" bench "$W/p-seg-$L.bin" seg $((33554432 / L)) $((8 * L)) "$REPS" 8192 4096 4096 \
      >> "$O/bench-seg-$best.jsonl" 2>> "$O/bench.err"
    echo "bench seg L=$L $best rc $?" >> "$O/rc.txt"
  done
  # o_proj'"'"'s operand (M x 4096) at each prefill chunk M: e2e leaves, then tile leaves of R rows x 32 values, beside the
  # M x 4096 x 4096 GEMM; by the best build and by each timing-only format build (merged, quad)
  tile_tags=$best
  for t in $timed; do case $t in *merged*|*quad*) tile_tags="$tile_tags $t" ;; esac; done
  for t in $tile_tags; do
    for M in $TILE_M; do
      "$W/commit_gpu_$t" bench "$W/p-e2e-$M.bin" e2e $((128 * M)) 1024 "$REPS" "$M" 4096 4096 >> "$O/bench-tile-$t.jsonl" 2>> "$O/bench.err"
      echo "bench e2e M=$M $t rc $?" >> "$O/rc.txt"
      for R in $TILE_R; do
        "$W/commit_gpu_$t" bench "$W/p-tile-$M-$R.bin" tile "$M" 4096 "$R" "$REPS" "$M" 4096 4096 >> "$O/bench-tile-$t.jsonl" 2>> "$O/bench.err"
        echo "bench tile M=$M R=$R $t rc $?" >> "$O/rc.txt"
      done
      for R in $TILESEG_R; do
        "$W/commit_gpu_$t" bench "$W/p-tileseg-$M-$R.bin" tileseg "$M" 4096 "$R" "$REPS" "$M" 4096 4096 >> "$O/bench-tile-$t.jsonl" 2>> "$O/bench.err"
        echo "bench tileseg M=$M R=$R $t rc $?" >> "$O/rc.txt"
      done
    done
  done
  kill $smi
  # the kernels'"'"' launch shapes (registers a thread, shared memory, blocks): can a commit kernel sit beside a GEMM CTA?
  N=$(command -v nsys || ls /usr/local/cuda-13.0/bin/nsys 2>/dev/null)
  if [ -n "$best" ] && [ -n "$N" ]; then
    "$N" profile -o "$W/trace" -t cuda --force-overwrite true "$W/commit_gpu_$best" bench "$W/p-rows-4096.bin" rows 8192 32768 2 \
      8192 4096 4096 > /dev/null 2> "$O/nsys.err"
    "$N" stats -r cuda_gpu_trace -f csv -o "$O/nsys" "$W/trace.nsys-rep" >> "$O/nsys.err" 2>&1
    echo "nsys rc $?" >> "$O/rc.txt"
  fi'
echo "lease rc $?" >> "$O/rc.txt"
cat "$O/rc.txt"
for f in "$O"/check-*.jsonl; do [ -e "$f" ] && tail -1 "$f"; done
$PY $D/summary.py "$O"
