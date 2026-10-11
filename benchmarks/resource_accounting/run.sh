#!/usr/bin/env bash
# The matmul chain's recorded run on node 2, from the run tree's root:
#   research run --on vy-nebius-2 --project verity --campaign resource-accounting --source <tree> --cwd source \
#     --env GPU_LEASE_WHO=<id> [--env SHAPES=o,qkv,gate_up,down BATCHES=... ARMS=... REPS=5 LEASE_MIN=30] -- \
#     bash benchmarks/resource_accounting/run.sh
# In order, the first failure ends the run:
#  1. on the CPUs, outside the lease: GPU 1's ship (pearl_c_sm120/build.sh, its SASS gate) into scratch, unless ARMS has no
#     Pearl-C arm; the ship's MANIFEST, SASS and toolchain records and the cubin's digest go to $OUT/ship;
#  2. under `gpu-lease 1 --wait --preemptible --max-min $LEASE_MIN` (30 at most): GPU 1's device check on the cubin (run.py,
#     v1 only, no timing), then chain.py into $OUT/chain.json, its budget two minutes under the lease's;
#  3. on the CPUs, after the lease: the retained pass (verify_run.py --fast, which must ACCEPT) and its no-write control
#     (which must REJECT), from scratch; their verify.json files go to $OUT/verify.
# POUS=1 (one shape; --cwd clone, since the memory challenger's Lean verdict builds in the tree) adds the PoUS arm: before the
# lease, pod.sh PREPARE=1 builds that verdict; under it, once the shape's points are measured alone, chain.py starts
# pod.sh IMPL=p2-v3-mc-gpu (P2's timed audit: the GPU responder and the memory challenger, signed with the PoUS lanes' LMS key
# through pod.sh's own --mc-lms-key, under their WORK) with no load of its own, and runs the chain beside its honest audit,
# off the memory challenger's and responder's CPUs.
# The harness's outputs go to $OUT/pous. POUS_WS, POUS_AUDITS and POUS_SEGMENTS pick its weight set, audits and segments;
# POUS_IDLE=1 is the control, the chain idle through the audit's window.
# Scratch (the ship's check fixture, the retained passes) is a mktemp -d under /workspace/tmp, deleted on exit.
set -euo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:?run this under research run, or give OUT}/out}; mkdir -p "$OUT"
WORK=${WORK:-/workspace/pouw/mvp-e2e}; PY=$WORK/venv312/bin/python; CUDA=${CUDA:-/usr/local/cuda-13.0}
S=$(mktemp -d /workspace/tmp/ra-matmul.XXXXXX); trap 'rm -rf "$S"' EXIT
SHAPES=${SHAPES:-o,qkv,gate_up,down,cube8192}; BATCHES=${BATCHES:-1,8,32,128,512,2048}; LENGTHS=${LENGTHS:-32}
ARMS=${ARMS:-native-bf16,native-fp8,pearl-c,pearl-c-deferred,pearl-c-nohash}; REPS=${REPS:-5}
LEASE_MIN=${LEASE_MIN:-30}; VERIFY_AT=${VERIFY_AT:-o:32}
[ "$LEASE_MIN" -le 30 ] || { echo "RUN_FAIL_LEASE: LEASE_MIN=$LEASE_MIN is over 30 minutes" >&2; exit 2; }
stamp() { echo "ra-matmul: $* at $(date -u +%H:%M:%SZ)"; }
export PYTHONPATH=$PWD:$PWD/catalog:$PWD/kernels:$PWD/experimental:$PWD/integrations/vllm:$PWD/benchmarks/pouw
HEAD=$(git rev-parse HEAD); echo "tree $HEAD" | tee "$OUT/tree.txt"

PEARL=; case ",$ARMS," in *,pearl-c*) PEARL=1 ;; esac
KERNEL=(); CHECK=
if [ -n "$PEARL" ]; then
  stamp "ship build"
  CUDA=$CUDA nice bash benchmarks/pouw/pearl_c_sm120/build.sh "$S/ship" > "$OUT/build.log" 2>&1 \
    || { stamp "RUN_FAIL_BUILD rc=$?"; tail -20 "$OUT/build.log"; exit 11; }
  mkdir -p "$OUT/ship" && cp "$S/ship/MANIFEST.sha256" "$S/ship/sass.json" "$S/ship/toolchain.txt" "$S/ship/source.txt" "$OUT/ship/"
  (cd "$S/ship" && sha256sum pearl_c_sm120.cubin) | tee "$OUT/ship/cubin.sha256"
  cmp -s "$S/ship/run.py" benchmarks/pouw/pearl_c_sm120/run.py || { stamp "RUN_FAIL_SHIP: the ship's run.py is not the tree's"; exit 12; }
  KERNEL=(--kernel benchmarks/pouw/pearl_c_sm120 --cubin "$S/ship/pearl_c_sm120.cubin" --retain "$S/passes" --verify-at "$VERIFY_AT"
          --coins-dir "$OUT/coins")
  CHECK=$S/ship/check/check.json
fi
if [ -n "${POUS:-}" ]; then
  case $SHAPES in *,*) stamp "RUN_FAIL_POUS: POUS=1 takes one shape, not $SHAPES"; exit 2 ;; esac
  [ "$(basename "$PWD")" != "${RESEARCH_SOURCE_SHA:-}" ] || { stamp "RUN_FAIL_POUS: POUS=1 builds Lean in the tree: --cwd clone"; exit 2; }
  POUS_WORK=${POUS_WORK:-/workspace/pouw/fill-out/pous/work}; POUS_WS=${POUS_WS:-llama3.1-8b/layer0+head}
  POUS_AUDITS=${POUS_AUDITS:-20}; POUS_SEGMENTS=${POUS_SEGMENTS:-8}
  mkdir -p "$OUT/pous"
  stamp "PoUS prepare"
  RESEARCH_RUN_DIR=$OUT/pous nice bash benchmarks/pous/pod.sh PREPARE=1 IMPL=p2-v3-mc-gpu WORK="$POUS_WORK" \
    > "$OUT/pous-prepare.log" 2>&1 || { stamp "RUN_FAIL_POUS_PREPARE"; tail -20 "$OUT/pous-prepare.log"; exit 13; }
  # pod.sh's own choice of the memory challenger's and responder's CPUs (record_host), named so the chain leaves them
  read -r VCPU RCPU <<< "$(lscpu -p=CPU,CORE | grep -v '^#' | awk -F, '!seen[$2]++ {print $1}' | sed -n '3p;4p' | tr '\n' ' ')"
  [ -z "${POUS_IDLE:-}" ] || KERNEL+=(--beside-idle)
  KERNEL+=(--beside "$OUT/pous/bench" --beside-log "$OUT/pous.log" --beside-cpus "$VCPU,$RCPU" --beside-cmd
           "bash benchmarks/pous/pod.sh RESEARCH_RUN_DIR=$OUT/pous IMPL=p2-v3-mc-gpu WORK=$POUS_WORK WS=$POUS_WS AUDITS=$POUS_AUDITS MAX_SEGMENTS=$POUS_SEGMENTS LOAD_TOKENS=0 TOKENS=1 WORKERS=8 VCPU=$VCPU RCPU=$RCPU")
fi

stamp "chain under gpu-lease 1 (at most $LEASE_MIN min)"
export OUT PY CHECK SHAPES BATCHES LENGTHS ARMS REPS LEASE_MIN
set +e
gpu-lease 1 --wait --preemptible --max-min "$LEASE_MIN" -- bash -c '
  set -euo pipefail
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} CUDA_DEVICE_ORDER=PCI_BUS_ID
  UUID=${GPU_LEASE_UUID%%,*}; [ -n "$UUID" ] || { echo "RUN_FAIL_IDENTITY: no GPU_LEASE_UUID" >&2; exit 3; }
  nvidia-smi --query-gpu=uuid,name,compute_cap,clocks.sm,clocks.mem,driver_version --format=csv,noheader -i "$UUID" > "$OUT/identity.csv"
  if [ -n "$CHECK" ]; then
    PEARLC_EXPECT_UUID=$UUID SHAPES="" VARIANTS=sm120 "$PY" "$(dirname "$(dirname "$CHECK")")/run.py" "$CHECK" > "$OUT/gpu1-check.jsonl" \
      || { echo "RUN_FAIL_GPU1_CHECK: run.py exited $?" >&2; exit 4; }
  fi
  exec timeout $((LEASE_MIN * 60 - 30)) "$PY" benchmarks/resource_accounting/chain.py --out "$OUT/chain.json" --shapes "$SHAPES" \
    --batches "$BATCHES" --lengths "$LENGTHS" --arms "$ARMS" --reps "$REPS" --budget-min $((LEASE_MIN - 3)) "${@}"
' _ "${KERNEL[@]}" 2>&1 | tee "$OUT/chain.log"
rc=${PIPESTATUS[0]}
set -e
stamp "lease ended, chain exited $rc"

if [ -n "$PEARL" ] && [ -f "$OUT/chain.json" ]; then
  mkdir -p "$OUT/verify"
  "$PY" - "$OUT/chain.json" > "$S/verify.env" <<'EOF'
import json, shlex, sys
v = json.load(open(sys.argv[1])).get("verify") or {}
for k in ("pass", "control", "coins"):
    print(f"V_{k.upper()}={shlex.quote(v.get(k) or '')}")
print(f"V_OK={1 if v.get('ok') else ''}")
EOF
  . "$S/verify.env"
  if [ -n "$V_PASS" ] && [ -n "$V_CONTROL" ]; then
    stamp "verify after the lease"
    vrc=0
    nice "$PY" benchmarks/pouw/pearl_c_vllm/verify_run.py "$V_PASS" --coins "$V_COINS" --fast --jobs 16 | tee "$OUT/verify/pass.txt" || vrc=31
    cp "$(dirname "$V_PASS")/verify.json" "$OUT/verify/pass.json" 2>/dev/null || true
    nice "$PY" benchmarks/pouw/pearl_c_vllm/verify_run.py "$V_CONTROL" --coins "$V_COINS" --fast --jobs 16 --expect-reject \
      | tee "$OUT/verify/control.txt" || vrc=32
    cp "$(dirname "$V_CONTROL")/verify.json" "$OUT/verify/control.json" 2>/dev/null || true
    [ -n "$V_OK" ] || vrc=33
    stamp "verify rc=$vrc"
    [ "$rc" != 0 ] || rc=$vrc
  else
    stamp "no retained pass to verify"
  fi
fi
exit "$rc"
