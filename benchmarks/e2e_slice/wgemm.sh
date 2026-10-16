#!/usr/bin/env bash
# EGEMM's edited GEMM with checkpointed receipts (wgemm.cu) against plain int8 at o_proj (m = 8192, n = k = 4096), on one
# leased sm_120 GPU (cwd = the source tree's root). Builds wgemm.cu outside the lease (its local-memory ops counted), the
# venv by build.sh; lease 1 runs the device gates (256 x 256 x 4096 and o_proj, poisoned buffers, unswapped and swapped)
# and dumps x, W, X, W' and the receipts; wgate.py recomputes them with derive.win_receipts and splits the swapped arms'
# operands as the Dither statement reads them; only if both pass, lease 2 times the per-call pairs IL
# (ARM:BASE, wgemm il) and the calls with their commitments PIPE (ARM:BASE:S, wgemm pipe: X's tile tree and the receipts'
# tree by COMMIT-GPU's committer beside the next call), each under every schedule in PIPE_CFGS. Clocks are read, never set.
# Under $OUT: gate.jsonl, wgate.jsonl, il.jsonl, pipe.jsonl, il.md, rc.txt.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source --env GPU_LEASE_WHO=E2E \
#     -- bash benchmarks/e2e_slice/wgemm.sh
# IL (the pairs; none: skip), IL_REPS (41), PIPE (none: skip), PIPE_REPS (15), PIPE_CFGS: rK (the GEMM's persistent grid on
# SMs - K, the commitments' stream at the highest priority), rKg (the same with the GEMM's stream highest and the
# commitments' lowest, each call's commitments released with the next call's GEMM, so they get only the K SMs the GEMM
# leaves while it runs), tiles[g] (a CTA a tile, not persistent), greenK (the
# commitments in a green context of K SMs, the GEMMs in one of the rest), old (r0 on the committer's default path, not its
# small one).
set -uo pipefail
export O=${OUT:?} D=benchmarks/e2e_slice
export PATH=/usr/local/cuda-13.0/bin:$PATH
export BIN=$(mktemp -d /tmp/e2e-wgemm.XXXXXX)
trap 'rm -rf "$BIN"' EXIT
git rev-parse HEAD > "$O/commit.txt" 2>&1
nvcc -O3 -std=c++17 --no-compress -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$BIN/wgemm" $D/wgemm.cu $D/cgpipe.cu \
  -lcuda -lcublasLt -lnvidia-ml > "$O/build.log" 2>&1
rc=$?
echo "build rc $rc" | tee -a "$O/rc.txt"
[ "$rc" = 0 ] || { tail -40 "$O/build.log"; exit 4; }
cuobjdump -sass "$BIN/wgemm" > "$BIN/wgemm.sass" 2>/dev/null
awk '/Function :/{f=$3} /[[:space:]](LDL|STL)[.[:space:]]/{n[f]++} END{for (f in n) print n[f], f}' "$BIN/wgemm.sass" | sort -k2 \
  > "$O/sass_local_mem_ops.txt" 2>&1
grep -A2 -E 'Compiling entry function .*(w_l|e_l2q|e_l3qm|e_plain)' "$O/build.log" | grep -E 'entry|registers|spill' > "$O/ptxas.txt"
bash $D/build.sh > "$O/venv.log" 2>&1 || { tail -20 "$O/venv.log"; exit 2; }
PY=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["python"])' "$O/builds.json")
export PYTHONPATH=$PWD
export W="$BIN/gates"
mkdir -p "$W/small" "$W/o"
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 6 -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.limit \
    --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  "$BIN/wgemm" gate "$W/small" 256 256 4096 0x1234 >> "$O/gate.jsonl" 2>> "$O/gate.err"; echo "gate small rc $?" >> "$O/rc.txt"
  "$BIN/wgemm" gate "$W/o" 8192 4096 4096 0x5678 >> "$O/gate.jsonl" 2>> "$O/gate.err"; echo "gate o rc $?" >> "$O/rc.txt"'
echo "lease 1 rc $?" >> "$O/rc.txt"
$PY $D/wgate.py "$W/small" "$W/o" > "$O/wgate.jsonl" 2> "$O/wgate.err"
echo "wgate rc $?" | tee -a "$O/rc.txt"
if grep -q 'rc [1-9]' "$O/rc.txt"; then
  echo "a gate failed: nothing timed"; cat "$O/rc.txt" "$O/gate.jsonl" "$O/wgate.jsonl"; tail -20 "$O/gate.err" "$O/wgate.err"; exit 5
fi
export IL_REPS=${IL_REPS:-41} PIPE_REPS=${PIPE_REPS:-15}
export IL=${IL:-"edited:w_l3qm_s8:plain sedited:w_l3qm_s8:plain edited:w_l3qm_s16:plain sedited:w_l3qm_s16:plain
  w_l3qm_s16:plain s:w_l3qm_s16:plain editedf:w_l3qm_s16:plainq seditedf:w_l3qm_s16:plainq"}
[ "$IL" = none ] && IL=""
# PIPE: ARM:BASE:S, the call with its commitments (wgemm pipe), COMMIT-GPU's params for 16,384-bit rows
export PIPE=${PIPE:-"edited:w_l3qm_s16:plain:16 sedited:w_l3qm_s16:plain:16 editedf:w_l3qm_s16:plainq:16"}
[ "$PIPE" = none ] && PIPE=""
export PIPE_CFGS=${PIPE_CFGS:-"old r0 r1 r17 r30 r41 r60 r1g r17g r30g r41g r60g tiles tilesg green16 green24 green30 green40
  green48 green60"}
# a PIPE_CFGS entry as wgemm pipe's environment
cfg_env() {
  local c=$1 e="CG_SMALL=1"
  [ "$c" = old ] && { echo "CG_SMALL=0"; return; }
  [[ $c == *g ]] && { e="$e WG_PRIO=gemm"; c=${c%g}; }
  case $c in tiles) e="$e WG_GRID=tiles" ;; green*) e="$e WG_GREEN=${c#green}" ;; r*) e="$e WG_RESERVE=${c#r}" ;; esac
  echo "$e"
}
export -f cfg_env
for s in 4 8 16 32; do PYTHONPATH=. python3 benchmarks/commit_gpu/reference.py params "$BIN/p-r$s.bin" $((262144 / s)) 16384 || exit 3; done
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 20 -- bash -c '
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu --format=csv -lms 500 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  for p in $IL; do
    base=${p##*:}; arm=${p%:*}
    "$BIN/wgemm" il "$IL_REPS" "$arm" "$base" 8192 4096 4096 >> "$O/il.jsonl" 2>> "$O/il.err"; echo "il $p rc $?" >> "$O/rc.txt"
  done
  for c in $PIPE_CFGS; do
    for p in $PIPE; do
      s=${p##*:}; r=${p%:*}; base=${r##*:}; arm=${r%:*}
      env $(cfg_env "$c") "$BIN/wgemm" pipe "$PIPE_REPS" "$arm" "$base" 8192 4096 4096 "$BIN/p-r$s.bin" "$s" \
        >> "$O/pipe.jsonl" 2>> "$O/pipe.err"
      echo "pipe $c $p rc $?" >> "$O/rc.txt"
    done
  done
  kill $smi'
echo "lease 2 rc $?" >> "$O/rc.txt"
cat "$O/rc.txt"
python3 - "$O/il.jsonl" "$O/pipe.jsonl" <<'EOF' | tee "$O/il.md"
import json, os, sys
il, pipe = sys.argv[1:]
if os.path.exists(il):
    print("| arm | base | ms | base ms | ratio | MHz |\n|---|---|---|---|---|---|")
    for l in open(il):
        r = json.loads(l)
        print(f"| {r['il']} | {r['base']} | {r['ms']:.4f} | {r['base_ms']:.4f} | {r['ratio']:.3f} | {r['mhz']:.0f} |")
if os.path.exists(pipe):
    print("\n| arm | S | reserved SMs | grid | first | committer | base ms | arm ms | commits alone ms | serial ms | pipe ms"
          " | arm / base | serial / base | pipe / base | MHz |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for l in open(pipe):
        r = json.loads(l)
        res = "tiles" if r["reserve"] < 0 else f"green {r['commit_sms']}" if r.get("green") else r["reserve"]
        print(f"| {r['pipe']} | {r['S']} | {res} | {r['grid']} | {r['prio']} | {'small' if r['small'] else 'default'} |"
              f" {r['base_ms']:.4f} | {r['arm_ms']:.4f} | {r['commit_ms']:.4f} | {r['serial_ms']:.4f} | {r['pipe_ms']:.4f} |"
              f" {r['arm_ratio']:.3f} | {r['serial_ratio']:.3f} | {r['pipe_ratio']:.3f} | {r['mhz']:.0f} |")
EOF
