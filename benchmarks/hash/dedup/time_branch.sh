#!/usr/bin/env bash
# Times a branch's Pearl-C -h3 arm against today's on the PoUW workload, as STEP-SPLIT's iteration 0 (r20261018-175257-7357)
# timed -h3: served_profile.sh's GPU part (identity, ship and GPU 1 checks, then e2e.py under served_profile.py: Llama-3.1-8B,
# both engines resident, fp8 / pearlc / fp8 / pearlc-nohash / fp8 / bf16 interleaved rep by rep, gates before timing).  Run
# from the branch's tree (`--source` a clean worktree at its head, `--cwd source`), with this script sent as an input:
#   research run --on vy-nebius-2 --project verity --campaign hash-loop --source WORKTREE --cwd source \
#     --send benchmarks/hash/dedup/time_branch.sh --env GPU_LEASE_WHO=kernels-dedup --env "ARM_ENV=VAR=VALUE ..." -- \
#     bash -c 'bash "$RESEARCH_RUN_DIR"/inputs/time_branch.sh'
# ARMS names the arms (default "base arm"): base runs with BASE_ENV (default none, today's -h3 in the same tree), arm with
# ARM_ENV.  A window runs every arm back to back in one one-GPU preemptible lease (at most LEASE_MIN minutes), in ABBA order
# across windows, so both arms see the same GPU within minutes; PAIR=0 gives each arm a lease of its own, for a node so loaded
# that one arm's window takes over half a lease (r20261019-000632-a0b8: model load alone took 7 min).  WINDOWS picks the windows: W4's (r20261018-193331-0f54: eager,
# whole, eager-nsys, whole-nsys; the default whole eager whole-nsys) or STEP-SPLIT's (eager-8192, whole-8192, eager-2560 and
# their -nsys).  An nsys window's analysis (served-profile.json) is kept; its report and SQLite are deleted.
# served-profile-pd.json is the same analysis with #2629's decode launches (pd_commit_*, pd_tree_*, pd_tiles_*) classed as
# hashing: served_profile.py's categories lack pd_*, so in a tree with #2629 its split leaves them in "other".
set -uo pipefail
stamp() { echo "time-branch: $* at $(date -u +%H:%M:%SZ)"; }

if [ "${1:-}" = pair ]; then  # inside a window's lease: pair NAME NSYS ARM... -- VAR=VALUE...
  name=$2 nsys=$3; shift 3; arms=()
  while [ "$1" != -- ]; do arms+=("$1"); shift; done; shift
  per=$(( (LEASE_MIN * 60 - 90) / ${#arms[@]} ))
  for arm in "${arms[@]}"; do
    W=$OUT/$name-$arm; mkdir -p "$W"
    if [ "$arm" = base ]; then env_arm=$BASE_ENV; else env_arm=$ARM_ENV; fi
    stamp "window $name-$arm: NSYS=$nsys $* $env_arm"
    uptime > "$W/load.txt"
    # shellcheck disable=SC2086
    env "$@" $env_arm NSYS="$nsys" OUT="$W" timeout -k 30 "$per" \
      bash benchmarks/pouw/pearl_c_vllm/served_profile.sh gpu > "$W/window.log" 2>&1
    rc=$?; echo "$rc" > "$W/rc"; uptime >> "$W/load.txt"
    stamp "window $name-$arm exited $rc"; tail -3 "$W/window.log" | cut -c1-300
  done
  exit 0
fi

D=${RESEARCH_RUN_DIR:?}; OUT=$D/out; mkdir -p "$OUT"
WORK=${WORK:-/workspace/pouw/mvp-e2e}; PY=$WORK/venv312/bin/python; CUDA=/usr/local/cuda-13.0
WINDOWS=${WINDOWS:-whole eager whole-nsys}
ARMS=${ARMS:-base arm}; BASE_ENV=${BASE_ENV:-}; ARM_ENV=${ARM_ENV:-}
MEM_GB=${MEM_GB:-128}; LEASE_MIN=${LEASE_MIN:-28}; PAIR=${PAIR:-1}
: "${GPU_LEASE_WHO:?pass --env GPU_LEASE_WHO=<id>}"
case " $WINDOWS " in *" eager-2560"*)
  grep -q -- '--prefill' benchmarks/pouw/pearl_c_vllm/served_profile.sh || {
    echo "time-branch: this tree's served_profile.sh has no PREFILL (it came with #2549), so eager-2560 would time prefill 8192;"
    echo "  drop eager-2560* from WINDOWS, or time a tree that has the harness"; exit 12; } ;;
esac
HEAD=$(git rev-parse HEAD 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}"); echo "tree $HEAD" | tee "$OUT/tree.txt"
{ nproc; uptime; df -h /workspace | tail -1; gpu-lease status; } > "$OUT/node.txt" 2>&1
S=$(mktemp -d "${SCRATCH:-/workspace/pouw}/tmp-time-branch.XXXXXX") || exit 13; trap 'rm -rf "$S"' EXIT
stamp "ship build"
CUDA=$CUDA bash benchmarks/pouw/pearl_c_sm120/build.sh "$S/ship-${HEAD:0:8}" > "$OUT/build.log" 2>&1 \
  || { stamp "BUILD_FAIL"; tail -20 "$OUT/build.log"; exit 11; }
tar -cf "$S/ship.tar" -C "$S" "ship-${HEAD:0:8}"
(cd "$S/ship-${HEAD:0:8}" && sha256sum pearl_c_sm120.cubin) | tee "$OUT/cubin.sha256"
export SCHEME=${SCHEME:-pearl-c-sm120-v1-h3} SHIP_TAR=$S/ship.tar WORK OUT BASE_ENV ARM_ENV LEASE_MIN

window() {  # NAME NSYS ARM... -- VAR=VALUE ...
  local name=$1 nsys=$2 arm g W P arms=() groups=()
  shift 2
  while [ "$1" != -- ]; do arms+=("$1"); shift; done; shift
  if [ "$PAIR" = 1 ]; then groups=("${arms[*]}"); else groups=("${arms[@]}"); fi
  for g in "${groups[@]}"; do
    stamp "lease for $name ($g)"
    # shellcheck disable=SC2086
    gpu-lease 1 --wait --preemptible --max-min "$LEASE_MIN" --mem-gb "$MEM_GB" -- bash "$0" pair "$name" "$nsys" $g -- "$@"
    stamp "lease for $name ($g) exited $?"
  done
  [ "$nsys" = 1 ] || return 0
  for arm in "${arms[@]}"; do
    W=$OUT/$name-$arm P=$S/$name-$arm
    [ -f "$W/served.nsys-rep" ] || continue
    mkdir -p "$P"; mv "$W/served.nsys-rep" "$P/"
    nsys export --type sqlite --force-overwrite true -o "$P/served.sqlite" "$P/served.nsys-rep" > "$W/export.log" 2>&1
    "$PY" benchmarks/pouw/pearl_c_vllm/served_profile.py --analyze "$P/served.sqlite" --out "$W/served-profile.json" \
      > "$W/analyze.log" 2>&1
    stamp "window $name-$arm analyzed $?"
    "$PY" - "$P/served.sqlite" "$W/served-profile-pd.json" > "$W/analyze-pd.log" 2>&1 <<'EOF'
import importlib.util, json, re, sys
from pathlib import Path
db, out = map(Path, sys.argv[1:3])
sys.argv = [sys.argv[0], "--analyze", str(db), "--out", str(out)]  # served_profile.py instruments e2e on import without it
spec = importlib.util.spec_from_file_location("served_profile", "benchmarks/pouw/pearl_c_vllm/served_profile.py")
sp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sp)
sp.CATEGORIES = (("pearlc:pd_commit", r"^pd_commit_"), ("pearlc:pd_tree", r"^pd_tree_"),
                 ("pearlc:tile_hash", r"^pd_tiles_")) + tuple(sp.CATEGORIES)
sp._CAT = [(c, re.compile(p)) for c, p in sp.CATEGORIES]
sp.HASH_GROUPS["a_rows"] = tuple(sp.HASH_GROUPS["a_rows"]) + ("pearlc:pd_commit", "pearlc:pd_tree")
out.write_text(json.dumps(sp.analyze(db), indent=1))
EOF
    stamp "window $name-$arm analyzed with pd $?"
    rm -rf "$P"
  done
}

i=0
for w in $WINDOWS; do
  # shellcheck disable=SC2086
  if [ $((i % 2)) = 0 ]; then set -- $ARMS; else set -- $(echo "$ARMS" | tr ' ' '\n' | tac); fi
  i=$((i + 1))
  case $w in
    eager) window "$w" 0 "$@" -- REPS=5 FP8_GRAPHS=1 ;;
    whole) window "$w" 0 "$@" -- REPS=5 FP8_GRAPHS=1 GRAPHS=1 WHOLE_STEP=1 WHOLE_DEFER=1 ;;
    eager-nsys) window "$w" 1 "$@" -- REPS=2 FP8_GRAPHS=1 NSYS_ARGS=--cuda-graph-trace=node ;;
    whole-nsys) window "$w" 1 "$@" -- REPS=2 FP8_GRAPHS=1 GRAPHS=1 WHOLE_STEP=1 WHOLE_DEFER=1 NSYS_ARGS=--cuda-graph-trace=node ;;
    eager-8192) window "$w" 0 "$@" -- REPS=5 ;;
    eager-8192-nsys) window "$w" 1 "$@" -- REPS=2 ;;
    whole-8192-nsys) window "$w" 1 "$@" -- REPS=2 GRAPHS=1 FP8_GRAPHS=1 WHOLE_STEP=1 NSYS_ARGS=--cuda-graph-trace=node ;;
    eager-2560) window "$w" 0 "$@" -- REPS=5 PREFILL=2560 ;;
    whole-8192) window "$w" 0 "$@" -- REPS=5 GRAPHS=1 FP8_GRAPHS=1 WHOLE_STEP=1 WHOLE_DEFER=1 ;;
    eager-2560-nsys) window "$w" 1 "$@" -- REPS=2 PREFILL=2560 ;;
    *) stamp "unknown window $w" ;;
  esac
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
    for name in ("served-profile.json", "served-profile-pd.json"):
        p = w / name
        if not p.is_file():
            continue
        doc = json.loads(p.read_text())
        d = doc.get("phases", {}).get("pearlc", {})
        for ph in ("prefill", "step"):
            if ph in d:
                print(f"  {name} pearlc {ph}: {json.dumps({k: round(v, 3) for k, v in d[ph].get('hash_split_ms', {}).items()})}")
        if "prefill" in d:
            stats = [k["ms_per_rep"] for k in doc.get("top_kernels", {}).get("pearlc", {}).get("prefill", []) if k["kernel"] == "stats_s5"]
            print(f"  {name} pearlc prefill: memcpy {d['prefill'].get('memcpy_ms')} ms, stats_s5 {stats} ms")
EOF
stamp "done"
