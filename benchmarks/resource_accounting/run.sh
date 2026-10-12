#!/usr/bin/env bash
# The matmul chain's recorded run on node 2, from the run tree's root:
#   research run --on vy-nebius-2 --project verity --campaign resource-accounting --source <tree> --cwd source \
#     --env GPU_LEASE_WHO=<id> [--env SHAPES=o,qkv,gate_up,down BATCHES=... ARMS=... REPS=5 LEASE_MIN=30] -- \
#     bash benchmarks/resource_accounting/run.sh
# In order, the first failure ends the run:
#  1. on the CPUs, outside the lease: GPU 1's ship (pearl_c_sm120/build.sh, its SASS gate) into scratch, unless ARMS has no
#     Pearl-C arm; the ship's MANIFEST, SASS and toolchain records and the cubin's digest go to $OUT/ship;
#  2. under `gpu-lease $LEASE --max-min $LEASE_MIN` (30 at most; LEASE defaults to `1 --wait --preemptible`), on the leased GPU
#     CHAIN_GPU names (an index; default the first leased), with every host thread under `taskset -c $HOST_CPUS` when that is
#     given: GPU 1's device check on the cubin (run.py, v1 only, no timing), then chain.py into $OUT/chain.json, its budget
#     three minutes under the lease's;
#  3. on the CPUs, after the lease: the retained pass (verify_run.py --fast, which must ACCEPT) and its no-write control
#     (which must REJECT), from scratch; their verify.json files go to $OUT/verify.
# PHASE=prepare, gpu or verify runs one step alone (default all three), over SCRATCH=<dir>, a scratch the caller owns and keeps
# between them, where step 1's builds are made once and reused; LEG=<name> keeps that leg's retained passes apart there.
# LEASE=held runs step 2 under the caller's own lease (window.sh: several legs under one timed lease).
# POUS=1 (one shape; --cwd clone, since the memory challenger's Lean verdict builds in the tree) adds the PoUS arm: before the
# lease, pod.sh PREPARE=1 builds that verdict; under it, once the shape's points are measured alone, chain.py starts
# pod.sh IMPL=p2-v3-mc-gpu (P2's timed audit: the GPU responder and the memory challenger, signed with the PoUS lanes' LMS key
# through pod.sh's own --mc-lms-key, under their WORK) with no load of its own, and runs the chain beside its honest audit,
# off the memory challenger's and responder's CPUs (VCPU, RCPU: the third and fourth physical cores, of HOST_CPUS when given).
# The harness's outputs go to $OUT/pous. POUS_WS, POUS_AUDITS and POUS_SEGMENTS pick its weight set, audits and segments;
# POUS_IDLE=1 is the control, the chain idle through the audit's window.
# JOINT=<rounds> (one shape, native arms only, --cwd clone; chain.py --rounds: separate, inproc, joint, joint-v1) runs the
# chain in the PoUS harness's process instead: before the lease, pod.sh PREPARE=1 and, for the joint and joint-v1 rounds, the
# bandwidth tagger's libbwtag.so (benchmarks/bandwidth/tagger.cu, refused on a register spill) into scratch; under it, pod.sh
# IMPL=p2-v3-mc-gpu BENCH_HOST='chain.py ...' BANDWIDTH_SALT=<the GPU> (the store under the bandwidth audit's device-bound
# salt, which its verifier checks), which measures the shape's points alone in the harness's venv and then runs each round
# with the chain as the honest audit's load. POUS_AUDITS (default 100 here), POUS_WS and POUS_SEGMENTS as for POUS=1; the
# outputs go to $OUT/pous, each round's harness under $OUT/pous/bench/<round>. PROBE=1 adds chain.py --probe, in either mode,
# and SIDECAR=1 chain.py --sidecar (an idle CUDA context in a second process through the run; not with PROBE).
# SMI_MS=<ms> (JOINT) samples the leased GPU's clocks, power, temperatures and clock event reasons every <ms> through the
# lease into $OUT/smi.csv (nvidia-smi, no CUDA context), on the wall time joint.py's anchor maps its passes to.
# DECAY=1 runs probe.py decay instead (the tag pass before and after GEMM, copy and L2 bursts, no PoUS), DECAY=steps probe.py
# steps (the tag pass after each step, the L1/shared split's preferences among them): the tagger into scratch before the
# lease, the probe under it into $OUT/decay.json or $OUT/steps.json, and $OUT/smi.csv.
# TUNED=<build dir|art:id> (ARMS with native-bf16-tuned or native-fp8-tuned) is the PoUW harness's library for the tuned
# arms (tuned.py), a build.sh output such as `pouw-harness-build/v1`: its jobs/env.sh gives the cuBLASLt and CUDA runtime
# 12.9 wheels it links (sonames .so.12, which torch's .so.13 never resolve to) on LD_LIBRARY_PATH, and POUW_HARNESS_LIB, for
# chain.py in either mode; its build.json goes to $OUT/tuned-build.json.
# Scratch (the ship's check fixture, the retained passes) is a mktemp -d under /workspace/tmp, deleted on exit, unless SCRATCH.
set -euo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:?run this under research run, or give OUT}/out}; mkdir -p "$OUT"
WORK=${WORK:-/workspace/pouw/mvp-e2e}; PY=$WORK/venv312/bin/python; CUDA=${CUDA:-/usr/local/cuda-13.0}
PHASE=${PHASE:-all}
case $PHASE in all|prepare|gpu|verify) ;; *) echo "RUN_FAIL_PHASE: PHASE is all, prepare, gpu or verify, not $PHASE" >&2; exit 2 ;; esac
if [ -n "${SCRATCH:-}" ]; then S=$SCRATCH; mkdir -p "$S"
elif [ "$PHASE" = all ]; then S=$(mktemp -d /workspace/tmp/ra-matmul.XXXXXX); trap 'rm -rf "$S"' EXIT
else echo "RUN_FAIL_PHASE: PHASE=$PHASE needs SCRATCH, the scratch its other steps share" >&2; exit 2; fi
SHAPES=${SHAPES:-o,qkv,gate_up,down,cube8192}; BATCHES=${BATCHES:-1,8,32,128,512,2048}; LENGTHS=${LENGTHS:-32}
ARMS=${ARMS:-native-bf16,native-fp8,pearl-c,pearl-c-deferred,pearl-c-nohash}; REPS=${REPS:-5}
LEASE_MIN=${LEASE_MIN:-30}; VERIFY_AT=${VERIFY_AT:-o:32}; LEASE=${LEASE:-1 --wait --preemptible}
[ "$LEASE_MIN" -le 30 ] || { echo "RUN_FAIL_LEASE: LEASE_MIN=$LEASE_MIN is over 30 minutes" >&2; exit 2; }
stamp() { echo "ra-matmul: $* at $(date -u +%H:%M:%SZ)"; }
on() { case $PHASE in all|"$1") return 0 ;; *) return 1 ;; esac; }
export PYTHONPATH=$PWD:$PWD/catalog:$PWD/kernels:$PWD/experimental:$PWD/integrations/vllm:$PWD/benchmarks/pouw
HEAD=$(git rev-parse HEAD); echo "tree $HEAD" | tee "$OUT/tree.txt"
PASSES=$S/passes${LEG:+/$LEG}

# CMD under the lease (the caller's with LEASE=held), its host threads under HOST_CPUS
lease() {
  local pin=(); [ -z "${HOST_CPUS:-}" ] || pin=(taskset -c "$HOST_CPUS")
  if [ "$LEASE" = held ]; then
    [ -n "${GPU_LEASE_UUID:-}" ] || { stamp "RUN_FAIL_LEASE: LEASE=held outside a lease (no GPU_LEASE_UUID)"; return 2; }
    "${pin[@]}" "$@"
  else
    # shellcheck disable=SC2086
    gpu-lease $LEASE --max-min "$LEASE_MIN" -- "${pin[@]}" "$@"
  fi
}
# inside the lease: CHAIN_GPU (else the first leased) as the one GPU visible, its UUID as UUID and GPU_LEASE_UUID, its identity
export PICK='G=${CHAIN_GPU:-${CUDA_VISIBLE_DEVICES%%,*}}; IFS=, read -ra D <<< "$CUDA_VISIBLE_DEVICES"; IFS=, read -ra U <<< "${GPU_LEASE_UUID:-}"
  UUID=; for k in "${!D[@]}"; do [ "${D[$k]}" != "$G" ] || UUID=${U[$k]:-}; done
  [ -n "$UUID" ] || { echo "RUN_FAIL_IDENTITY: GPU $G is not one of the leased ${CUDA_VISIBLE_DEVICES:-}" >&2; exit 3; }
  export CUDA_VISIBLE_DEVICES=$G CUDA_DEVICE_ORDER=PCI_BUS_ID GPU_LEASE_UUID=$UUID
  nvidia-smi --query-gpu=index,uuid,name,compute_cap,clocks.sm,clocks.mem,driver_version --format=csv,noheader -i "$UUID" > "$OUT/identity.csv"'
# the memory challenger's and responder's CPUs: as pod.sh's record_host picks them, from HOST_CPUS when given
read -r VCPU RCPU <<< "$(
  allow=; [ -z "${HOST_CPUS:-}" ] || allow=$(for r in ${HOST_CPUS//,/ }; do seq "${r%-*}" "${r#*-}"; done | tr '\n' ' ')
  lscpu -p=CPU,CORE | grep -v '^#' | awk -F, -v allow=" $allow " '(allow == "  " || index(allow, " " $1 " ")) && !seen[$2]++ {print $1}' \
    | sed -n '3p;4p' | tr '\n' ' ')"
[ -n "$VCPU" ] && [ -n "$RCPU" ] || { stamp "RUN_FAIL_CPUS: no two physical cores in HOST_CPUS=${HOST_CPUS:-}"; exit 2; }

if on prepare || on gpu; then
  case ",$ARMS," in *-tuned,*)
    [ -n "${TUNED:-}" ] || { stamp "RUN_FAIL_TUNED: a tuned arm needs TUNED=<build dir|art:id>"; exit 2; }
    stamp "the PoUW harness's library for the tuned arms"
    TUNED_ENV=$(bash -c '. benchmarks/pouw/harness/jobs/env.sh "$1" >&2 && printf "%s\n%s\n" "$POUW_HARNESS_LIB" "$LD_LIBRARY_PATH"' \
      _ "$TUNED" 2> "$OUT/tuned-env.log") || { stamp "RUN_FAIL_TUNED_ENV"; tail -20 "$OUT/tuned-env.log"; exit 15; }
    POUW_HARNESS_LIB=$(sed -n 1p <<< "$TUNED_ENV"); LD_LIBRARY_PATH=$(sed -n 2p <<< "$TUNED_ENV")
    export POUW_HARNESS_LIB LD_LIBRARY_PATH
    cp "${POUW_HARNESS_LIB%/*}/build.json" "$OUT/tuned-build.json"
    ;;
  esac
fi

# pod.sh PREPARE=1 once per scratch (its marker), so legs sharing a scratch build the verdict once
pous_prepare() {
  if [ -f "$S/pous-prepared" ]; then stamp "PoUS prepared in this scratch already"; return 0; fi
  stamp "PoUS prepare"
  RESEARCH_RUN_DIR=$OUT/pous nice bash benchmarks/pous/pod.sh PREPARE=1 IMPL=p2-v3-mc-gpu WORK="$POUS_WORK" \
    > "$OUT/pous-prepare.log" 2>&1 || { stamp "RUN_FAIL_POUS_PREPARE"; tail -20 "$OUT/pous-prepare.log"; exit 13; }
  touch "$S/pous-prepared"
}
prepared() { [ -e "$1" ] || { stamp "RUN_FAIL_PHASE: no $1: PHASE=prepare first, over the same SCRATCH"; exit 2; }; }
# the bandwidth tagger's libbwtag.so once per scratch, refused on a register spill; its build log goes to $OUT
build_tagger() {
  if [ ! -f "$S/libbwtag.so" ]; then
    stamp "tagger build"
    { "$CUDA/bin/nvcc" --version; "$CUDA/bin/nvcc" -O3 -std=c++17 -arch=sm_120 -Xptxas -v -Xcompiler -fPIC -shared \
        benchmarks/bandwidth/tagger.cu -o "$S/libbwtag.so.part"; } > "$S/tagger-build.log" 2>&1 \
      || { cp "$S/tagger-build.log" "$OUT/"; stamp "RUN_FAIL_TAGGER_BUILD"; tail -20 "$S/tagger-build.log"; exit 14; }
    if grep -E "[1-9][0-9]* bytes spill" "$S/tagger-build.log"; then
      cp "$S/tagger-build.log" "$OUT/"; stamp "RUN_FAIL_TAGGER_BUILD: a kernel spills"; exit 14
    fi
    sha256sum benchmarks/bandwidth/tagger.cu "$S/libbwtag.so.part" >> "$S/tagger-build.log"
    mv "$S/libbwtag.so.part" "$S/libbwtag.so"
  fi
  cp "$S/tagger-build.log" "$OUT/tagger-build.log"
}

if [ -n "${DECAY:-}" ]; then
  if on prepare; then build_tagger; fi
  on gpu || exit 0
  prepared "$S/libbwtag.so"
  stamp "the decay probe under gpu-lease $LEASE (at most $LEASE_MIN min)"
  export OUT PY S DECAY
  set +e
  lease bash -c '
    set -euo pipefail
    eval "$PICK"
    MODE=decay; [ "$DECAY" != steps ] || MODE=steps
    exec timeout $((LEASE_MIN * 60 - 30)) "$PY" benchmarks/resource_accounting/probe.py "$MODE" --lib "$S/libbwtag.so" \
      --out "$OUT/$MODE.json" --smi-csv "$OUT/smi.csv"
  ' 2>&1 | tee "$OUT/chain.log"
  rc=${PIPESTATUS[0]}
  set -e
  stamp "lease ended, the probe exited $rc"
  exit "$rc"
fi

if [ -n "${JOINT:-}" ]; then
  [ -z "${POUS:-}" ] || { stamp "RUN_FAIL_JOINT: JOINT and POUS are two modes"; exit 2; }
  case $SHAPES in *,*) stamp "RUN_FAIL_JOINT: JOINT takes one shape, not $SHAPES"; exit 2 ;; esac
  case ",$ARMS," in *,pearl-c*) stamp "RUN_FAIL_JOINT: JOINT runs native arms only, not $ARMS"; exit 2 ;; esac
  [ "$(basename "$PWD")" != "${RESEARCH_SOURCE_SHA:-}" ] || { stamp "RUN_FAIL_JOINT: JOINT builds Lean in the tree: --cwd clone"; exit 2; }
  POUS_WORK=${POUS_WORK:-/workspace/pouw/fill-out/pous/work}; POUS_WS=${POUS_WS:-llama3.1-8b/layer0+head}
  POUS_AUDITS=${POUS_AUDITS:-100}; POUS_SEGMENTS=${POUS_SEGMENTS:-8}
  mkdir -p "$OUT/pous"
  TAGGER=
  case ",$JOINT," in *,joint,*|*,joint-v1,*) TAGGER="--tagger $S/libbwtag.so" ;; esac
  if on prepare; then
    pous_prepare
    [ -z "$TAGGER" ] || build_tagger
  fi
  on gpu || exit 0
  prepared "$S/pous-prepared"; [ -z "$TAGGER" ] || prepared "$S/libbwtag.so"
  HOST="benchmarks/resource_accounting/chain.py --out $OUT/chain.json --shapes $SHAPES --batches $BATCHES --lengths $LENGTHS"
  HOST+=" --arms $ARMS --reps $REPS --budget-min $((LEASE_MIN - 3)) --rounds $JOINT ${ROUND_MIN:+--round-min $ROUND_MIN}"
  HOST+=" $TAGGER ${PROBE:+--probe}"
  HOST+=" ${SIDECAR:+--sidecar} --host-bench"
  stamp "the chain hosting the PoUS harness under gpu-lease $LEASE (at most $LEASE_MIN min), VCPU $VCPU RCPU $RCPU"
  SMI_MS=${SMI_MS:-}
  export OUT LEASE_MIN HOST POUS_WORK POUS_WS POUS_AUDITS POUS_SEGMENTS SMI_MS VCPU RCPU
  set +e
  lease bash -c '
    set -euo pipefail
    eval "$PICK"
    SMI=
    if [ -n "$SMI_MS" ]; then
      nvidia-smi -i "$UUID" --format=csv -lms "$SMI_MS" --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,temperature.memory,clocks_event_reasons.active \
        > "$OUT/smi.csv" 2>&1 &
      SMI=$!
    fi
    rc=0
    timeout $((LEASE_MIN * 60 - 30)) bash benchmarks/pous/pod.sh RESEARCH_RUN_DIR="$OUT/pous" IMPL=p2-v3-mc-gpu \
      WORK="$POUS_WORK" WS="$POUS_WS" AUDITS="$POUS_AUDITS" MAX_SEGMENTS="$POUS_SEGMENTS" LOAD_TOKENS=0 TOKENS=1 WORKERS=8 \
      VCPU="$VCPU" RCPU="$RCPU" BANDWIDTH_SALT="$UUID" BENCH_HOST="$HOST" || rc=$?
    [ -z "$SMI" ] || kill "$SMI"
    exit "$rc"
  ' 2>&1 | tee "$OUT/chain.log"
  rc=${PIPESTATUS[0]}
  set -e
  stamp "lease ended, pod.sh exited $rc"
  exit "$rc"
fi

PEARL=; case ",$ARMS," in *,pearl-c*) PEARL=1 ;; esac
KERNEL=(); CHECK=
if [ -n "$PEARL" ]; then
  if on prepare; then
    if [ -f "$S/ship.built" ]; then stamp "the ship in this scratch already"
    else
      stamp "ship build"
      CUDA=$CUDA nice bash benchmarks/pouw/pearl_c_sm120/build.sh "$S/ship" > "$OUT/build.log" 2>&1 \
        || { stamp "RUN_FAIL_BUILD rc=$?"; tail -20 "$OUT/build.log"; exit 11; }
      touch "$S/ship.built"
    fi
    mkdir -p "$OUT/ship" && cp "$S/ship/MANIFEST.sha256" "$S/ship/sass.json" "$S/ship/toolchain.txt" "$S/ship/source.txt" "$OUT/ship/"
    (cd "$S/ship" && sha256sum pearl_c_sm120.cubin) | tee "$OUT/ship/cubin.sha256"
    cmp -s "$S/ship/run.py" benchmarks/pouw/pearl_c_sm120/run.py || { stamp "RUN_FAIL_SHIP: the ship's run.py is not the tree's"; exit 12; }
  fi
  on gpu && prepared "$S/ship.built"
  mkdir -p "${PASSES%/*}"
  KERNEL=(--kernel benchmarks/pouw/pearl_c_sm120 --cubin "$S/ship/pearl_c_sm120.cubin" --retain "$PASSES" --verify-at "$VERIFY_AT"
          --coins-dir "$OUT/coins")
  CHECK=$S/ship/check/check.json
fi
if [ -n "${POUS:-}" ]; then
  case $SHAPES in *,*) stamp "RUN_FAIL_POUS: POUS=1 takes one shape, not $SHAPES"; exit 2 ;; esac
  [ "$(basename "$PWD")" != "${RESEARCH_SOURCE_SHA:-}" ] || { stamp "RUN_FAIL_POUS: POUS=1 builds Lean in the tree: --cwd clone"; exit 2; }
  POUS_WORK=${POUS_WORK:-/workspace/pouw/fill-out/pous/work}; POUS_WS=${POUS_WS:-llama3.1-8b/layer0+head}
  POUS_AUDITS=${POUS_AUDITS:-20}; POUS_SEGMENTS=${POUS_SEGMENTS:-8}
  mkdir -p "$OUT/pous"
  if on prepare; then pous_prepare; fi
  on gpu && prepared "$S/pous-prepared"
  [ -z "${POUS_IDLE:-}" ] || KERNEL+=(--beside-idle)
  KERNEL+=(--beside "$OUT/pous/bench" --beside-log "$OUT/pous.log" --beside-cpus "$VCPU,$RCPU" --beside-cmd
           "bash benchmarks/pous/pod.sh RESEARCH_RUN_DIR=$OUT/pous IMPL=p2-v3-mc-gpu WORK=$POUS_WORK WS=$POUS_WS AUDITS=$POUS_AUDITS MAX_SEGMENTS=$POUS_SEGMENTS LOAD_TOKENS=0 TOKENS=1 WORKERS=8 VCPU=$VCPU RCPU=$RCPU")
fi

rc=0
if on gpu; then
  stamp "chain under gpu-lease $LEASE (at most $LEASE_MIN min)"
  PROBE=${PROBE:-}; SIDECAR=${SIDECAR:-}
  export OUT PY CHECK SHAPES BATCHES LENGTHS ARMS REPS LEASE_MIN PROBE SIDECAR
  set +e
  lease bash -c '
    set -euo pipefail
    eval "$PICK"
    if [ -n "$CHECK" ]; then
      PEARLC_EXPECT_UUID=$UUID SHAPES="" VARIANTS=sm120 "$PY" "$(dirname "$(dirname "$CHECK")")/run.py" "$CHECK" > "$OUT/gpu1-check.jsonl" \
        || { echo "RUN_FAIL_GPU1_CHECK: run.py exited $?" >&2; exit 4; }
    fi
    exec timeout $((LEASE_MIN * 60 - 30)) "$PY" benchmarks/resource_accounting/chain.py --out "$OUT/chain.json" --shapes "$SHAPES" \
      --batches "$BATCHES" --lengths "$LENGTHS" --arms "$ARMS" --reps "$REPS" --budget-min $((LEASE_MIN - 3)) ${PROBE:+--probe} \
      ${SIDECAR:+--sidecar} "${@}"
  ' _ "${KERNEL[@]}" 2>&1 | tee "$OUT/chain.log"
  rc=${PIPESTATUS[0]}
  set -e
  stamp "lease ended, chain exited $rc"
fi

if on verify && [ -n "$PEARL" ] && [ -f "$OUT/chain.json" ]; then
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
