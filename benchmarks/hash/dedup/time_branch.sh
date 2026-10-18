#!/usr/bin/env bash
# Times a branch's Pearl-C -h3 arm against today's on the PoUW workload, as STEP-SPLIT's iteration 0 (r20261018-175257-7357)
# timed -h3: served_profile.sh's GPU part (identity, ship and GPU 1 checks, then e2e.py under served_profile.py: Llama-3.1-8B,
# both engines resident, fp8 / pearlc / fp8 / pearlc-nohash / fp8 / bf16 interleaved rep by rep, gates before timing), one
# window a one-GPU preemptible lease.  Run from the branch's tree (`--source` a clean worktree at its head, `--cwd source`),
# with this script sent as an input:
#   research run --on vy-nebius-2 --project verity --campaign hash-loop --source WORKTREE --cwd source \
#     --send benchmarks/hash/dedup/time_branch.sh --env GPU_LEASE_WHO=kernels-dedup --env "ARM_ENV=VAR=VALUE ..." -- \
#     bash -c 'bash "$RESEARCH_RUN_DIR"/inputs/time_branch.sh'
# ARMS names the arms (default "base arm"): base runs with BASE_ENV (default none, today's -h3 in the same tree), arm with
# ARM_ENV.  Each window runs every arm, in ABBA order across windows, so both arms see the same node within minutes.
# WINDOWS picks the windows (STEP-SPLIT's names).  An nsys window's analysis (served-profile.json) is kept; its report and
# SQLite are deleted.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?}; OUT=$D/out; mkdir -p "$OUT"
WORK=${WORK:-/workspace/pouw/mvp-e2e}; PY=$WORK/venv312/bin/python; CUDA=/usr/local/cuda-13.0
WINDOWS=${WINDOWS:-whole-8192 eager-8192 whole-8192-nsys eager-2560}
ARMS=${ARMS:-base arm}; BASE_ENV=${BASE_ENV:-}; ARM_ENV=${ARM_ENV:-}
MEM_GB=${MEM_GB:-128}; LEASE_MIN=${LEASE_MIN:-25}
: "${GPU_LEASE_WHO:?pass --env GPU_LEASE_WHO=<id>}"
case " $WINDOWS " in *" eager-2560"*)
  grep -q -- '--prefill' benchmarks/pouw/pearl_c_vllm/served_profile.sh || {
    echo "time-branch: this tree's served_profile.sh has no PREFILL (it came with #2549), so eager-2560 would time prefill 8192;"
    echo "  drop eager-2560* from WINDOWS, or time a tree that has the harness"; exit 12; } ;;
esac
HEAD=$(git rev-parse HEAD 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}"); echo "tree $HEAD" | tee "$OUT/tree.txt"
stamp() { echo "time-branch: $* at $(date -u +%H:%M:%SZ)"; }
{ nproc; uptime; df -h /workspace | tail -1; gpu-lease status; } > "$OUT/node.txt" 2>&1
S=$(mktemp -d /workspace/pouw/tmp-time-branch.XXXXXX) || exit 13; trap 'rm -rf "$S"' EXIT
stamp "ship build"
CUDA=$CUDA bash benchmarks/pouw/pearl_c_sm120/build.sh "$S/ship-${HEAD:0:8}" > "$OUT/build.log" 2>&1 \
  || { stamp "BUILD_FAIL"; tail -20 "$OUT/build.log"; exit 11; }
tar -cf "$S/ship.tar" -C "$S" "ship-${HEAD:0:8}"
(cd "$S/ship-${HEAD:0:8}" && sha256sum pearl_c_sm120.cubin) | tee "$OUT/cubin.sha256"
export SCHEME=${SCHEME:-pearl-c-sm120-v1-h3} SHIP_TAR=$S/ship.tar WORK

window() {  # NAME ARM NSYS VAR=VALUE ...
  local name=$1 arm=$2 nsys=$3; shift 3
  local W=$OUT/$name-$arm P=$S/$name-$arm env_arm
  mkdir -p "$W" "$P"
  if [ "$arm" = base ]; then env_arm=$BASE_ENV; else env_arm=$ARM_ENV; fi
  stamp "window $name-$arm: NSYS=$nsys $* $env_arm"
  uptime > "$W/load.txt"
  # shellcheck disable=SC2086
  env "$@" $env_arm NSYS="$nsys" OUT="$W" gpu-lease 1 --wait --preemptible --max-min "$LEASE_MIN" --mem-gb "$MEM_GB" -- \
    timeout -k 30 $((LEASE_MIN * 60 - 90)) bash benchmarks/pouw/pearl_c_vllm/served_profile.sh gpu > "$W/window.log" 2>&1
  local rc=$?; echo "$rc" > "$W/rc"; uptime >> "$W/load.txt"
  stamp "window $name-$arm exited $rc"; tail -3 "$W/window.log" | cut -c1-300
  if [ "$nsys" = 1 ] && [ -f "$W/served.nsys-rep" ]; then
    mv "$W/served.nsys-rep" "$P/"
    nsys export --type sqlite --force-overwrite true -o "$P/served.sqlite" "$P/served.nsys-rep" > "$W/export.log" 2>&1
    "$PY" benchmarks/pouw/pearl_c_vllm/served_profile.py --analyze "$P/served.sqlite" --out "$W/served-profile.json" \
      > "$W/analyze.log" 2>&1
    stamp "window $name-$arm analyzed $?"
  fi
  rm -rf "$P"
}

i=0
for w in $WINDOWS; do
  if [ $((i % 2)) = 0 ]; then order=$ARMS; else order=$(echo "$ARMS" | tr ' ' '\n' | tac | tr '\n' ' '); fi
  i=$((i + 1))
  for arm in $order; do
    case $w in
      eager-8192) window "$w" "$arm" 0 REPS=5 ;;
      eager-8192-nsys) window "$w" "$arm" 1 REPS=2 ;;
      whole-8192-nsys) window "$w" "$arm" 1 REPS=2 GRAPHS=1 FP8_GRAPHS=1 WHOLE_STEP=1 NSYS_ARGS=--cuda-graph-trace=node ;;
      eager-2560) window "$w" "$arm" 0 REPS=5 PREFILL=2560 ;;
      whole-8192) window "$w" "$arm" 0 REPS=5 GRAPHS=1 FP8_GRAPHS=1 WHOLE_STEP=1 WHOLE_DEFER=1 ;;
      eager-2560-nsys) window "$w" "$arm" 1 REPS=2 PREFILL=2560 ;;
      *) stamp "unknown window $w" ;;
    esac
  done
done

"$PY" - "$OUT" <<'EOF'
import json, statistics, sys
from pathlib import Path
for w in sorted(Path(sys.argv[1]).iterdir()):
    f = w / "e2e.json"
    if not f.is_file():
        continue
    o = json.loads(f.read_text())
    print(f"== {w.name}: {o.get('error') or 'ok'}")
    for m, r in o.get("results", {}).items():
        for key, reps in r.get("reps", {}).items():
            xs = [x.get("step_s", x.get("s")) * 1e3 for x in reps]
            if xs:
                print(f"  {m:14s} {key:22s} median {statistics.median(xs):9.2f} ms  [{min(xs):.2f}, {max(xs):.2f}]  n={len(xs)}")
    p = w / "served-profile.json"
    if p.is_file():
        d = json.loads(p.read_text()).get("phases", {}).get("pearlc", {})
        for ph in ("prefill", "step"):
            if ph in d:
                print(f"  profile pearlc {ph}: {json.dumps({k: round(v, 3) for k, v in d[ph].get('hash_split_ms', {}).items()})}")
EOF
stamp "done"
