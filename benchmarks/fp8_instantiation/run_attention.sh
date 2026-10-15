#!/usr/bin/env bash
# attention.py on one leased GPU of vy-nebius-2 (cwd = the clone's root): the selftest, then every method in one process
# over unsloth/Llama-3.1-8B @ 3f0d51f8 at context D4A_CTX (2048: ppl.py's window-0 tokens; longer: D4A_IDS, a sent ids file of
# WikiText-2's test window 0 that must begin with those 2048).  Clocks are read, never set.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd clone --custody-ttl 12h \
#     --env GPU_LEASE_WHO=... --env D4A_CTX=2048 --env D4A_PLAN=grid|frontier|limbs|limbs2|seeds|both [--env D4A_METHODS="..."] \
#     [--env D4A_IDS=ids8192.npy --send ids8192.npy] --send benchmarks/fp8_instantiation/run_attention.sh \
#     -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/run_attention.sh"'
set -uo pipefail
export O=${RESEARCH_RUN_DIR:?}/out
mkdir -p "$O"
export PY=/workspace/jobs/venv312/bin/python
export SNAP=/workspace/hf/hub/models--unsloth--Llama-3.1-8B/snapshots/3f0d51f8e5640f98f1a96ea9044a0e55c0a83814
export IDS=/workspace/pouw/fp8-security/llama8b-w0/ids.npy
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
export CTX=${D4A_CTX:-2048}
# D4A_IDS (an ids file sent with the run, e.g. window 0 at 8192) replaces IDS, whose 2048 tokens it must begin with.
PREFIX=()
if [ -n "${D4A_IDS:-}" ]; then PREFIX=(--prefix "$IDS"); IDS=$RESEARCH_RUN_DIR/inputs/$D4A_IDS; fi
export IDS
grid_plan() {
  m="bf16 ref ref-rot"
  for g in int8 e4m3; do for r in "" -rot; do for p in prow pfix; do for v in vchan vhead; do
    m+=" $g-$p-$v-none$r"; done; done; done; done
  for r in "" -rot; do for a in pm1 pm13 pm57 u1 u2 u4 u8 u16; do for p in prow pfix; do for v in vchan vhead; do
    m+=" int8-$p-$v-$a$r"; done; done; done; done
  for r in "" -rot; do for a in pm1 u1 u2 pm13 u4 u8 pm57; do for p in prow pfix; do m+=" e4m3-$p-vchan-$a$r"; done; done; done
  for r in "" -rot; do for a in u1 u4 u16; do for w in qk pv; do m+=" int8-prow-vchan-$a-$w$r"; done; done; done
  echo "$m"
}
# Each operand's noise alone (P on int8 per row, V per channel), the E4M3 alternative for Q K^T, and noise on all four
# operands with P's sparse.
frontier_plan() {
  m="bf16 ref int8-none int8-none-rot e4m3qk-none e4m3qk-none-rot"
  for r in "" -rot; do
    for a in pm1 pm13 pm57 u1 u2 u4 u8 u16; do m+=" int8-qk=$a$r int8-v=$a$r"; done
    for a in pm1 pm13 pm57 u1 u2 u4 u8; do m+=" e4m3qk-qk=$a$r"; done
  done
  for a in d1 d2 d4 d6 d8 d10 d12 d14 pm1 u1; do m+=" int8-p=$a"; done
  for q in u1 u2 u4 u8; do for v in u1 u4 u16; do for p in d6 d8 d10 d12 d14; do m+=" int8-qk=$q-v=$v-p=$p-rot"; done; done; done
  echo "$m"
}
# P on two int8 limbs (`prow2`), its noise on the low limb: P alone, all four operands alike, and mixed alphabets.
limbs_plan() {
  m="bf16 ref int8-none-rot int8-prow2-none int8-prow2-none-rot"
  for a in pm1 pm13 pm57 u1 u2 u4 u8 u16; do m+=" int8-prow2-p=$a"; for r in "" -rot; do m+=" int8-prow2-$a$r"; done; done
  for q in u1 u2 u4 u8; do for v in u1 u4 u16; do for p in u4 u16; do m+=" int8-prow2-qk=$q-v=$v-p=$p-rot"; done; done; done
  echo "$m"
}
# Every operand on two limbs (P, V; then Q and K too), the same alphabet on all four low limbs; seeds S.
limbs2_plan() {
  m="int8-prow2-vchan2-none int8-prow2-vchan2-none-rot int8-prow2-vchan2-qk2-none int8-prow2-vchan2-qk2-none-rot"
  for s in ${SEEDS:-s0}; do for a in pm1 pm13 pm57 u1 u2 u4 u8 u16; do
    m+=" int8-prow2-vchan2-$a-rot-$s int8-prow2-vchan2-qk2-$a-$s int8-prow2-vchan2-qk2-$a-rot-$s"; done; done
  echo "$m"
}
# The frontier points of the frontier and limbs plans at seeds S (s1 s2: s0 ran there), and mixed points at s0 too.
seeds_plan() {
  old="int8-qk=u1-rot int8-qk=u2-rot int8-qk=u4-rot int8-qk=pm13-rot int8-qk=u8-rot int8-qk=pm1 int8-qk=u2 int8-qk=u4
    int8-v=u1 int8-v=pm1 int8-v=u2 int8-v=pm1-rot int8-v=u2-rot int8-p=d8 int8-p=d6 int8-p=d4
    int8-qk=u1-v=u1-p=d14-rot int8-qk=u4-v=u1-p=d8-rot int8-qk=u4-v=u1-p=d6-rot int8-qk=u2-v=u1-p=d10-rot
    int8-prow2-p=u16 int8-prow2-p=u8 int8-prow2-pm1-rot int8-prow2-u1 int8-prow2-u1-rot int8-prow2-u2-rot int8-prow2-pm13-rot
    int8-prow2-qk=u1-v=u1-p=u4-rot int8-prow2-qk=u2-v=u1-p=u16-rot int8-prow2-qk=u4-v=u1-p=u4-rot int8-prow2-qk=u4-v=u1-p=u16-rot"
  new="int8-qk=u1-v=u1-p=d8 int8-qk=pm1-v=u1-p=d8 int8-qk=u2-v=u1-p=d8 int8-qk=u1-v=u1-p=d10 int8-qk=u1-v=pm1-p=d8
    int8-qk=u2-v=pm1-p=d6 int8-qk=u2-v=u1-p=d6 int8-qk=u4-v=u2-p=d4-rot int8-qk=u2-v=u2-p=d6-rot
    int8-prow2-qk=u1-v=u1-p=u16 int8-prow2-qk=pm1-v=u1-p=u16 int8-prow2-qk=u2-v=u1-p=u16 int8-prow2-qk=u2-v=pm1-p=u16
    int8-prow2-qk=u2-v=u2-p=u16-rot int8-prow2-qk=u4-v=u2-p=u16-rot"
  m=""
  for s in ${SEEDS:-s1 s2}; do for x in $old; do m+=" $x-$s"; done; done
  for s in s0 ${SEEDS:-s1 s2}; do for x in $new; do m+=" $x-$s"; done; done
  echo "$m"
}
case "${D4A_PLAN:-}" in
  limbs2) D4A_METHODS=$(limbs2_plan) ;;
  seeds) D4A_METHODS="$(seeds_plan) $(SEEDS="s0 s1 s2" limbs2_plan)" ;;
  grid) D4A_METHODS=$(grid_plan) ;;
  frontier) D4A_METHODS=$(frontier_plan) ;;
  limbs) D4A_METHODS=$(limbs_plan) ;;
  both) D4A_METHODS="$(frontier_plan) $(grid_plan | sed 's/bf16 ref ref-rot//')" ;;
esac
[ -n "${D4A_METHODS:-}" ] || { echo "set D4A_PLAN or D4A_METHODS" >&2; exit 2; }
export D4A_METHODS PREFIX_ARGS="${PREFIX[*]}"
LEASE=(gpu-lease 1 --wait --max-min "${D4A_MAX_MIN:-60}" --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  S=benchmarks/fp8_instantiation/attention.py
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,power.limit --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  "$PY" "$S" --selftest > "$O/selftest.json" 2> "$O/selftest_stderr.txt" || exit 3
  "$PY" "$S" "$SNAP" "$IDS" "$O" --ctx "$CTX" $PREFIX_ARGS $D4A_METHODS > "$O/results.jsonl" 2> "$O/stderr.txt"
  r=$?
  nvidia-smi --query-gpu=index,clocks.sm,clocks.mem --format=csv -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  "$PY" "$S" --table "$O/results.jsonl" > "$O/table.md" 2>> "$O/stderr.txt"
  exit $r'
rc=$?
echo "attention $rc" | tee "$O/status.txt"
tail -3 "$O/selftest_stderr.txt" "$O/stderr.txt" 2>/dev/null
exit $rc
