#!/usr/bin/env bash
# kernels-redteam on PR #2629 at its head, unedited (the job's cwd is its tree):
#   off the lease  the served ship (pearl_c_sm120/build.sh, as w4_row0.sh builds it) and pd.cubin with run.sh's flags; the
#                  ship's pd_commit/pd_tiles 851 and 979 SASS against pd.cubin's, addresses stripped; host_test.py and
#                  tests/test_pearl_c_hash_decode.py
#   one GPU        bench.py at m16 = 32 over Llama-3.1-8B's four linears with the ship as Pearl-C's cubin: its check (every
#                  pd output poisoned and held to the reference launches and Python), then REPS reps of its timing
# Launch:
#   research run --on vy-nebius-2 --project verity --campaign hash-loop --source PR_TREE --cwd source \
#     --send benchmarks/hash/redteam/pd2629.sh --env GPU_LEASE_WHO=kernels-redteam -- bash -c 'exec bash "$RESEARCH_RUN_DIR/inputs/pd2629.sh"'
set -uo pipefail
: "${OUT:?run under research run, which sets OUT}"
D=$PWD/benchmarks/hash/pouw_decode; K=$PWD/benchmarks/pouw/pearl_c_sm120
CUDA=${CUDA:-/usr/local/cuda}
SCR=$(mktemp -d); trap 'rm -rf "$SCR"' EXIT
git rev-parse HEAD > "$OUT/head.txt" 2>/dev/null || true
FLAGS=(-O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -fmad=false -Xptxas -v -cubin)
CUDA=$CUDA bash "$K/build.sh" "$SCR/ship" > "$OUT/build.log" 2>&1 & p1=$!
"$CUDA/bin/nvcc" "${FLAGS[@]}" -o "$SCR/pd.cubin" "$D/pd.cu" 2> "$OUT/ptxas_pd.txt" & p2=$!
( g++ -O2 -std=c++20 -pthread -shared -fPIC -Wno-unknown-pragmas -o "$SCR/pd_host.so" "$K/hash_decode_host.cpp" &&
  PYTHONPATH=$PWD timeout 1500 python3 "$D/host_test.py" "$SCR/pd_host.so" > "$OUT/host_test.txt" 2>&1
  echo "host_test rc $?" >> "$OUT/host_test.txt" ) & p3=$!
( PYTHONPATH=$PWD timeout 1500 python3 -m pytest -q -p no:cacheprovider benchmarks/pouw/tests/test_pearl_c_hash_decode.py \
    > "$OUT/pytest_hash_decode.txt" 2>&1; echo "pytest rc $?" >> "$OUT/pytest_hash_decode.txt" ) & p4=$!
wait "$p1" || { echo "ship build failed"; tail -20 "$OUT/build.log"; exit 11; }
wait "$p2" || { echo "pd.cubin build failed"; tail -20 "$OUT/ptxas_pd.txt"; exit 12; }
(cd "$SCR" && sha256sum ship/pearl_c_sm120.cubin pd.cubin) | tee "$OUT/cubins.sha256"
norm() { "$CUDA/bin/cuobjdump" -sass -fun "$1" "$2" | sed -E 's#/\*[0-9a-f]{4}\*/##; s#/\* 0x[0-9a-f]+ \*/##; s#c\[0x3\]\[0x[0-9a-f]+\]#c[0x3][_]#g' |
  grep -E '^\s+[@A-Z]' | sed -E 's/\s+/ /g'; }
for f in pd_commit_851 pd_commit_979 pd_tiles_851 pd_tiles_979; do
  norm "$f" "$SCR/ship/pearl_c_sm120.cubin" > "$SCR/ship_$f.sass"; norm "$f" "$SCR/pd.cubin" > "$SCR/pd_$f.sass"
  echo "$f ship $(wc -l < "$SCR/ship_$f.sass") pd $(wc -l < "$SCR/pd_$f.sass") differing $(diff "$SCR/ship_$f.sass" "$SCR/pd_$f.sass" | grep -c '^[<>]')"
done | tee "$OUT/sass_compare.txt"
export PD_SIDE=851,979 SCR D REPS=${REPS:-3}
gpu-lease 1 --wait --preemptible --max-min 15 -- bash -c '
  set -euo pipefail
  q=index,uuid,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active
  nvidia-smi --query-gpu=$q --format=csv > "$OUT/gpu.csv"
  python3 "$D/bench.py" "$SCR/ship/pearl_c_sm120.cubin" "$SCR/pd.cubin" --out "$OUT" --reps "$REPS" --rows 32 --shapes qkv,o,gate_up,down
  nvidia-smi --query-gpu=$q --format=csv > "$OUT/gpu_end.csv"
'
rc=$?
wait "$p3"; wait "$p4"
tail -2 "$OUT/host_test.txt"; tail -3 "$OUT/pytest_hash_decode.txt"
echo "bench rc $rc"
exit "$rc"
