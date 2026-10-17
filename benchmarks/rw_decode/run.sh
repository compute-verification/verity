#!/usr/bin/env bash
# mont.cu on one GPU (cwd = the source tree): for each width in $SIZES and form in $FORMS, Montgomery squarings mod a
# Williams modulus and the fused RW-SMS decode, each gated against rw.py (256 words at K = 1 and 50; 128 records the
# encoder made with the factors) before it is timed; a read and a copy of 2 GiB beside them. Results in $OUT.
#
#   research run --queue --gpus 1 --cpus 8 --mem-gb 24 --max-min 45 --gpu-model rtx-pro-6000 --project verity \
#     --source . --cwd source --campaign stacking-17oct -- bash benchmarks/rw_decode/run.sh
set -uxo pipefail
export O=${OUT:-${RESEARCH_RUN_DIR:?}/out}; mkdir -p "$O"
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/rw_decode
B=$(mktemp -d /tmp/rw-decode.XXXXXX); trap 'rm -rf "$B"' EXIT
SIZES=${SIZES:-"1024 1536 2048 3072"}; FORMS=${FORMS:-"0 1 2 3 4 5"}; K=${K:-2000}; MB=${MB:-1024}; JOBS=${JOBS:-8}
nvcc --version > "$O/nvcc.txt"
for b in $SIZES; do for f in $FORMS; do echo "$b $f"; done; done \
  | xargs -P "$JOBS" -n 2 sh -c 'timeout 900 nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xptxas -v \
      -DMB_BITS=$0 -DMB_FORM=$1 -o '"$B"'/mont$0-f$1 '"$D"'/mont.cu > '"$O"'/build-$0-f$1.log 2>&1 \
      || echo "build $0 form $1 rc $?" >> '"$O"'/errors.txt' &
for b in $SIZES; do for g in 0 1 4; do
  (python3 $D/rw.py keygen "$b" "$g" $((b * 10 + g)) > "$O/key-$b-$g.json" \
    && python3 $D/rw.py params "$O/key-$b-$g.json" > "$B/p-$b-$g.txt" \
    && python3 $D/rw.py encode "$O/key-$b-$g.json" 128 "$b" > "$B/r-$b-$g.txt") \
    || echo "keygen/encode $b $g failed" >> "$O/errors.txt" &
done; done
wait
nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw --format=csv,noheader \
  -i "${CUDA_VISIBLE_DEVICES:-0}" > "$O/gpu.txt" 2>&1
first=$(ls "$B" | grep '^mont' | head -1)
"$B/$first" clock > "$O/clock-before.json" 2>&1
"$B/$first" bw 2048 10 > "$O/bw.json" 2>&1
for b in $SIZES; do for f in $FORMS; do
  x="$B/mont$b-f$f"; g=$((f >= 4 ? 4 : f % 2)); P="$B/p-$b-$g.txt"; KEY="$O/key-$b-$g.json"; ok=1
  [ -x "$x" ] || { echo "no binary for $b form $f" >> "$O/errors.txt"; continue; }
  for k in 1 50; do
    "$x" chain-gate "$f" "$k" "$P" 256 > "$B/cg.txt" 2>> "$O/errors.txt" || ok=0
    python3 $D/rw.py check-chain "$KEY" "$k" < "$B/cg.txt" > "$O/gate-chain-$b-f$f-k$k.json" || ok=0
  done
  "$x" decode-gate "$f" "$P" "$B/r-$b-$g.txt" > "$B/dg.txt" 2>> "$O/errors.txt" || ok=0
  python3 $D/rw.py check-decode "$KEY" "$B/r-$b-$g.txt" < "$B/dg.txt" > "$O/gate-decode-$b-f$f.json" || ok=0
  if [ $ok = 1 ]; then
    "$x" chain-time "$f" "$K" "$P" 128 4 5 >> "$O/chain.jsonl" 2>> "$O/errors.txt" || echo "chain-time $b $f rc $?" >> "$O/errors.txt"
    "$x" decode-time "$f" "$P" "$MB" 20 >> "$O/decode.jsonl" 2>> "$O/errors.txt" || echo "decode-time $b $f rc $?" >> "$O/errors.txt"
  else
    echo "bits $b form $f failed its gate: not timed" >> "$O/errors.txt"
  fi
done; done
"$B/$first" clock > "$O/clock-after.json" 2>&1
nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}" >> "$O/gpu.txt" 2>&1
grep -h -E "registers|spill" "$O"/build-*.log > "$O/ptxas.txt" 2>/dev/null || true
[ ! -s "$O/errors.txt" ]
