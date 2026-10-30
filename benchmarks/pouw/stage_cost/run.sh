#!/usr/bin/env bash
# stage.cu on one leased GPU (cwd = the source tree): build and prep.py's constants outside the lease; under one lease the
# gate, then prefill at each PREFILL chunk, the deferred pass alone at each DEFERRED_T, and beside the decode step at each
# BESIDE_T. Nothing is timed unless the gate passes. Clocks are read, never set. Results in $OUT/*.jsonl.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source --env GPU_LEASE_WHO=LANE \
#     -- bash benchmarks/pouw/stage_cost/run.sh
# env: PREFILL (2048 8192), GATE_ROWS (50 100 300 512 1001 2500: row counts the gate checks besides PREFILL's, small and
# ragged trees; never timed), DEFERRED_T, BESIDE_T (1024 4096 16384 32768), DEC_B, DEC_L (32 4096), FILL_SMS (92), DRAW_US
# (300), PARTS (gate prefill deferred beside), HASHES (fr sz: stage.cu's commitment paths for prefill), MAXMIN (25),
# PIPE_GREEN (1; 0 skips prefill's pipe_green arms), ARMS (r1 r2: prefill's round-1 arms, the proved forward's), CFGS (the
# proved forward's arms by name, comma-separated; all by default), FR_ROT (0: the commitment's rotations on the FMA pipe,
# stage.cu's bits)
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/pouw/stage_cost
export B; B=$(mktemp -d /tmp/stage-cost.XXXXXX); trap 'rm -rf "$B"' EXIT
export PREFILL=${PREFILL:-2048 8192} GATE_ROWS=${GATE_ROWS-50 100 300 512 1001 2500} DEFERRED_T=${DEFERRED_T:-256 512 1024 2048 4096 8192 16384 32768}
export BESIDE_T=${BESIDE_T:-1024 4096 16384 32768} DEC_B=${DEC_B:-32} DEC_L=${DEC_L:-4096} FILL_SMS=${FILL_SMS:-92}
export DRAW_US=${DRAW_US:-300} PARTS=${PARTS:-gate prefill deferred beside} HASHES=${HASHES:-fr sz}
git rev-parse HEAD > "$O/commit.txt" 2>&1
nvcc --version | tail -2 > "$O/nvcc.txt"
lscpu > "$O/lscpu.txt" 2>&1
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -DFR_ROT="${FR_ROT:-0}" -Xptxas -v -o "$B/stage" $D/stage.cu \
  -lcublasLt -lnvidia-ml -ldl > "$O/build.log" 2>&1 || { grep -E "error" "$O/build.log" | head -40; tail -20 "$O/build.log"; exit 4; }
sha256sum "$B/stage" | cut -c1-64 > "$O/stage.sha256"
"$B/stage" pooltest 32 > "$O/pooltest.jsonl" 2>&1 || { cat "$O/pooltest.jsonl"; exit 7; }
# shellcheck disable=SC2086
export ALL_ROWS; ALL_ROWS=$(printf '%s\n' $PREFILL $GATE_ROWS | sort -nu | paste -sd' ')
python3 $D/prep.py "$B/c" --rows "${ALL_ROWS// /,}" > "$O/prep.log" 2>&1 || { tail -20 "$O/prep.log"; exit 6; }
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-25}" -- bash -c '
  q() { nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,power.limit,clocks_event_reasons.active \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1; }
  has() { case " $PARTS " in *" $1 "*) return 0;; esac; return 1; }
  q
  "$B/stage" gate "$B/c" $ALL_ROWS > "$O/gate.jsonl" 2>> "$O/errors.txt"; g=$?
  echo "gate rc $g" >> "$O/rc.txt"
  [ "$g" -eq 0 ] || exit 5
  if has prefill; then
    "$B/stage" shalat > "$O/shalat.jsonl" 2>> "$O/errors.txt" || echo "rc $? shalat" >> "$O/errors.txt"
    for h in $HASHES; do for M in $PREFILL; do
      HASH=$h "$B/stage" prefill "$B/c" "$M" 7 >> "$O/prefill.jsonl" 2>> "$O/errors.txt" || echo "rc $? prefill $h $M" >> "$O/errors.txt"
    done; done
    q
  fi
  if has deferred; then
    for T in $DEFERRED_T; do for c in seq500 b32; do
      "$B/stage" deferred "$T" "$c" 5 >> "$O/deferred.jsonl" 2>> "$O/errors.txt" || echo "rc $? deferred $T $c" >> "$O/errors.txt"
    done; done
    q
  fi
  if has beside; then
    for T in $BESIDE_T; do for c in b32 seq500; do for f in gemm form; do
      "$B/stage" beside "$T" "$c" "$DEC_B" "$DEC_L" "$FILL_SMS" "$f" 21 >> "$O/beside.jsonl" 2>> "$O/errors.txt" ||
        echo "rc $? beside $T $c $f" >> "$O/errors.txt"
    done; done; q; done
  fi
'
r=$?
echo "lease rc $r" >> "$O/rc.txt"
cat "$O/gate.jsonl" "$O/rc.txt"
grep -v "^device" "$O/errors.txt" | tail -5 || true
exit "$r"
