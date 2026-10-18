#!/usr/bin/env bash
# Red-team re-run of sharaw's prefill claim on one leased GPU (cwd = this branch's root): sharaw's tree fetched at its pushed
# commit, its sharaw.cu built unedited into sharaw_gate.cu (plain and SR_INLINE=1, as sharaw's gpu.sh builds it) and as its own
# binary. Outside the lease: the builds, the parameter files (reference.py params --ks-seed 11, as gpu.sh) and the reference's
# levels at 3-byte positions (sharaw_ref3.py prepare). Under the lease: sharaw's selftest; its check of forms 22 (v1_cg8) and
# 17 (lf0_cg) against verity.ml.commitments at n = 65,537 / 70,001 / 131,075; the strict gate and controls at the decode size
# on each decode winner; at the prefill size the gate, the controls, raw / commit_gpu / the winner interleaved (REPS) with the
# SM clock of each call, and the gate after timing.
#
#   research run --on vy-dc-h200 --project verity --campaign hash-loop --source . --cwd source \
#     --env GPU_LEASE_WHO=kernels-redteam -- bash benchmarks/hash/redteam/sharaw.sh
# SHARAW_COMMIT (default: cursor/hash-sha-raw-16d5's head 57bb0c0f7), REPS (7), LEASE_ARGS (--preemptible on vy-nebius-2)
set -uo pipefail
O=${OUT:?}
export PATH=/usr/local/cuda/bin:$PATH
R=$(pwd)/benchmarks/hash/redteam
SHA=${SHARAW_COMMIT:-57bb0c0f70178c2096dbe21fba04f4045ca8f7d9}
W=$(mktemp -d)
export W O
trap 'rm -rf "$W"' EXIT
git rev-parse HEAD > "$O/probe_commit.txt"
echo "$SHA" > "$O/sharaw_commit.txt"
S="$W/src"
mkdir -p "$S" && git -C "$S" init -q && git -C "$S" remote add origin https://github.com/compute-verification/verity.git
if ! git -C "$S" fetch -q --depth=1 --filter=blob:none origin "$SHA" 2> "$O/fetch.err"; then
  git -C "$S" fetch -q --depth=1 origin "$SHA" 2>> "$O/fetch.err" || { echo "fetch failed" >> "$O/rc.txt"; exit 1; }
fi
git -C "$S" sparse-checkout set --no-cone /benchmarks/hash/sharaw /benchmarks/pouw/commit_gpu /verity >> "$O/fetch.err" 2>&1
git -C "$S" checkout -q FETCH_HEAD 2>> "$O/fetch.err"
[ "$(git -C "$S" rev-parse HEAD)" = "$SHA" ] || { echo "checkout is not $SHA" >> "$O/rc.txt"; exit 1; }
sha256sum "$S/benchmarks/hash/sharaw/sharaw.cu" "$S/benchmarks/pouw/commit_gpu/commit_gpu.cuh" > "$O/sources.sha256"
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | head -1 | tr -d .)
GEN="-gencode arch=compute_${cc}a,code=sm_${cc}a"
echo "arch $cc" > "$O/rc.txt"
SRC="$S/benchmarks/hash/sharaw/sharaw.cu"
nvcc -O3 -std=c++17 $GEN -Xptxas -v -DSHARAW_CU="\"$SRC\"" -o "$W/gate" "$R/sharaw_gate.cu" -lnvidia-ml -ldl > "$O/build-gate.log" 2>&1 &
nvcc -O3 -std=c++17 $GEN -Xptxas -v -DSR_INLINE=1 -DSHARAW_CU="\"$SRC\"" -o "$W/gate_inl" "$R/sharaw_gate.cu" -lnvidia-ml -ldl > "$O/build-gate-inl.log" 2>&1 &
nvcc -O3 -std=c++17 $GEN -o "$W/sharaw" "$SRC" -lnvidia-ml -ldl > "$O/build.log" 2>&1 &
nvcc -O3 -std=c++17 $GEN -DSR_INLINE=1 -o "$W/sharaw_inl" "$SRC" -lnvidia-ml -ldl > "$O/build-inl.log" 2>&1 &
export DECODE_N=${DECODE_N:-34737} PREFILL_N=${PREFILL_N:-3293653}
(cd "$S" && for n in $DECODE_N $PREFILL_N; do
  PYTHONPATH=. python3 benchmarks/pouw/commit_gpu/reference.py params "$W/p-$n.bin" "$n" 32768 --ks-seed 11
done) 2> "$O/params.err"
echo "params rc $?" >> "$O/rc.txt"
(cd "$S" && PYTHONPATH=. python3 "$R/sharaw_ref3.py" prepare "$W/ref3") > "$O/ref3-prepare.jsonl" 2> "$O/ref3-prepare.err"
echo "ref3 prepare rc $?" >> "$O/rc.txt"
wait
for b in gate gate_inl sharaw sharaw_inl; do [ -x "$W/$b" ]; echo "build $b rc $?" >> "$O/rc.txt"; done
grep -h -E "spill|registers" "$O"/build-gate*.log | sort | uniq -c | sort -rn | head -40 > "$O/ptxas-summary.txt"
cat "$O/rc.txt"
export S REPS=${REPS:-7}
# shellcheck disable=SC2016,SC2086
gpu-lease 1 --wait --max-min 14 ${LEASE_ARGS:-} -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,power.limit,memory.used --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active --format=csv -lms 200 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  "$W/sharaw" selftest > "$O/selftest.jsonl"; echo "selftest rc $?" >> "$O/rc.txt"
  (cd "$S" && PYTHONPATH=. python3 '"$R"'/sharaw_ref3.py check "$W/sharaw" "$W/ref3" 22 17) > "$O/ref3-check.jsonl" 2> "$O/ref3-check.err"
  echo "ref3 check plain rc $?" >> "$O/rc.txt"
  (cd "$S" && PYTHONPATH=. python3 '"$R"'/sharaw_ref3.py check "$W/sharaw_inl" "$W/ref3" 17 0) > "$O/ref3-check-inl.jsonl" 2> "$O/ref3-check-inl.err"
  echo "ref3 check inl rc $?" >> "$O/rc.txt"
  "$W/gate" "$W/p-$DECODE_N.bin" $DECODE_N $REPS v4_128,v1n > "$O/gate-decode.jsonl" 2> "$O/gate-decode.err"
  echo "gate decode plain rc $?" >> "$O/rc.txt"
  "$W/gate_inl" "$W/p-$DECODE_N.bin" $DECODE_N $REPS sharaw_v1 > "$O/gate-decode-inl.jsonl" 2> "$O/gate-decode-inl.err"
  echo "gate decode inl rc $?" >> "$O/rc.txt"
  "$W/gate" "$W/p-$PREFILL_N.bin" $PREFILL_N $REPS v1_cg8 v1_cg8 > "$O/prefill.jsonl" 2> "$O/prefill.err"
  echo "prefill plain rc $?" >> "$O/rc.txt"
  "$W/gate_inl" "$W/p-$PREFILL_N.bin" $PREFILL_N $REPS lf0_cg lf0_cg > "$O/prefill-inl.jsonl" 2> "$O/prefill-inl.err"
  echo "prefill inl rc $?" >> "$O/rc.txt"
  kill $smi'
echo "lease rc $?" >> "$O/rc.txt"
cat "$O/rc.txt"
tail -1 "$O"/ref3-check*.jsonl
grep -h -E '"control"|"gate"|"arm"' "$O"/gate-*.jsonl "$O"/prefill*.jsonl | cut -c1-220
