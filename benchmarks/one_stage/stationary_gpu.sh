#!/usr/bin/env bash
# Weights-stationary proof blocks against the per-expert cut's on one GPU: each ST_UNITS type of
# benchmarks/served_units/stage.py staged at half and all of a 2^35-bit statement (stage.py's half,full on one built circuit),
# every type staging at once on the node's CPUs while the CUDA prover builds, and each type proved by `flock-circuit prove --zk
# --gpu` (zk_cost.py prove, arm dev: the device's witness, as the overhead card's) once it is staged, in ST_UNITS' order: one
# warm session then ST_RUNS timed, every block drawn. A type the prover refuses (past Flock's range) is logged and the rest go on.
# Run from the verity tree on a GPU node:
#   bash benchmarks/one_stage/stationary_gpu.sh ST_UNITS="st-w13:M=1 st-w13:M=8" ST_GPU=0
# env: ST_UNITS (stage.py's type specs, each with its own counts as SPEC@C1,C2)  ST_COUNTS=half,full  ST_RUNS=2  ST_ARMS=dev
#      ST_ARMS1 (the arms at one prover; unset: ST_ARMS', e.g. dev,plaindev)  ST_VERIFY_AHEAD=N (FC_VERIFY_AHEAD=N at K > 1)
#      ST_GPU=0 (the device)  ST_WORK=/dev/shm (the staged statements' scratch)  ST_EXEC_WORK=/workspace/tmp (the builds'
#      and the venv's, on a mount that executes)
#      ST_PROVE_CPUS=0-31 (the prover's and the build's cores; the stagers take the rest)
#      ST_HOLD (a file whose text names this run's slot: written once the file is gone, waiting up to ST_HOLD_WAIT=600 s, and
#      removed at exit)  ST_DEADLINE (UTC epoch seconds after which no point starts)
#      ST_OVERLAP=1,2,4,8 (K prover processes at once on the one GPU against one verifier, zk_cost.py prove --provers K, on
#      the points of full/K and full/(2K) blocks; give the type counts down to full/(2 max K), e.g. pe-w13@2048,1024,512,256,128;
#      unset: one)  ST_CACHE=1 (then FC_STMT_CACHE=1 on each type's largest point at one prover and its best point and K;
#      only: every point and K with it on, none with it off; both: every point and K off, then on)
#      ST_MPS=1 (the provers through NVIDIA's MPS, one control daemon for the job: their kernels run at once, not time-sliced)
#      ST_KEEP_BIN (a path the built prover is copied to)  ST_BIN (a prover built so: no build)
#      ST_CPU=1 (no CUDA: the CPU prover, `prove --zk` without --gpu, for checking the runner on a CPU node)
#      ST_STAGE_JOBS=N (at most N types staging at once; unset: every type)
#      ST_SERVER=1,2,4 (a long-running proof server instead of the points: once every type is staged, proof_server.py run
#      with K provers for each K, on ST_MIX's stream (each type's blocks a sampled replay block, type=blocks,...),
#      ST_SESSIONS=120 timed sessions a prover after ST_SERVER_WARM warm ones, FC_VERIFY_AHEAD=ST_VERIFY_AHEAD; stage each
#      type at quarter,half,full)
#      ST_ZK (a directory the staged statements stay in after the run, so a later run with it reuses each type staged there
#      with the same spec and device check; unset: the run's scratch)
#      ST_MEM_GIB (the host's memory where no cgroup shows it, a RunPod pod's memoryInGb: the server stops at 8 GiB free of it)
#      ST_SERVER_TYPES=a,b (the server proves only these of the staged types the device prover takes)
#      ST_ROUTE=1 (each type to one of the K provers, which hold every circuit once between them: proof_server.py --route)
#      ST_SPLIT=N (statements of full/N blocks whatever K is: proof_server.py --split; unset: full/K)
# Kept in $OUT: per type and count its stage.json, the sessions' prove.out, prove.err and serve.log, and the records;
# never a circuit, a prover's file or a proof.
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
REPO=$(pwd); O=${OUT:-${RESEARCH_RUN_DIR:?}/out}; mkdir -p "$O"
W=$(mktemp -d "${ST_WORK:-/dev/shm}/stationary.XXXXXX")
mkdir -p "${ST_EXEC_WORK:-/workspace/tmp}" && X=$(mktemp -d "${ST_EXEC_WORK:-/workspace/tmp}/stationary.XXXXXX")
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$O/job.log"; }
HOLD=${ST_HOLD:-}; DEADLINE=${ST_DEADLINE:-0}
MINE="proofs stationary ${RESEARCH_RUN_ID:-local}: GPU ${ST_GPU:-0}$( [ "$DEADLINE" = 0 ] || date -u -d "@$DEADLINE" +' until %H:%MZ')"
STAGERS=(); MPS=
killtree() { local c; for c in $(pgrep -P "$1"); do killtree "$c"; done; kill "$1" 2>/dev/null; }
cleanup() {
  [ -n "$MPS" ] && echo quit | nvidia-cuda-mps-control && log "MPS: control daemon stopped"
  for p in "${STAGERS[@]}"; do killtree "$p"; done
  [ -n "$HOLD" ] && [ "$(cat "$HOLD" 2>/dev/null)" = "$MINE" ] && rm -f "$HOLD" && log "released $HOLD"
  rm -rf "$W" "$X"
}
trap cleanup EXIT
if [ -n "$HOLD" ]; then
  for _ in $(seq 1 $(( ${ST_HOLD_WAIT:-600} / 10 ))); do [ -e "$HOLD" ] || break; sleep 10; done
  if [ -e "$HOLD" ]; then log "$HOLD is still held: $(cat "$HOLD")"; HOLD=; exit 75; fi
  mkdir -p "$(dirname "$HOLD")" && echo "$MINE" > "$HOLD" && log "took $HOLD: $MINE"
fi
NP=$(nproc); PC=${ST_PROVE_CPUS:-0-31}; SC=${ST_STAGE_CPUS:-$(( ${PC#*-} + 1 ))-$((NP - 1))}
{ hostname; lscpu | grep -E 'Model name|^CPU\(s\)'; nproc; free -g; nvidia-smi; git rev-parse HEAD; } > "$O/host.txt" 2>&1
log "host: $NP CPUs ($(lscpu | sed -n 's/^Model name: *//p')), prover on $PC, stagers on $SC, GPU ${ST_GPU:-0}"
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH PYTHONDONTWRITEBYTECODE=1 UV_PROJECT_ENVIRONMENT=$X/venv UV_CACHE_DIR=$X/uv-cache
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
uv sync -q --all-packages --extra torch-cpu > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 5; }
PY=$X/venv/bin/python
export PYTHONPATH=$REPO:$REPO/verity/ml/flock/python:$REPO/integrations/vllm:$REPO/tools/verity/circuit_check/src:$REPO/benchmarks/numerical/python
Z=${ST_ZK:-$W/zk}; mkdir -p "$Z/records"; echo '[]' > "$Z/plan.json"
SPECS=(); JOBS=${ST_STAGE_JOBS:-0}
stage_type() {
  taskset -c "$SC" "$PY" benchmarks/served_units/stage.py "$1" "$2" "$3/c{n}/stage" > "$3/stage.log" 2>&1 \
    && touch "$3/staged" || touch "$3/failed"
}
for spec in ${ST_UNITS:?}; do
  u=${spec%@*}; COUNTS=${ST_COUNTS:-half,full}; [ "$u" = "$spec" ] || COUNTS=${spec#*@}
  t=$(echo "$u" | tr ':,=' '---' | tr -d '-'); sd=$Z/$t; mkdir -p "$sd"
  if [ -e "$sd/staged" ] && [ "$(cat "$sd/spec" 2>/dev/null)" = "$spec" ]; then log "$u: staged before, reused"; continue; fi
  rm -rf "$sd"/{staged,failed,device-ok,c*}; echo "$spec" > "$sd/spec"
  if [ "$JOBS" -gt 0 ]; then
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
  fi
  stage_type "$u" "$COUNTS" "$sd" &
  STAGERS+=($!); SPECS+=("$u|$t")
  log "$u: staging $COUNTS"
done &
STAGERS+=($!)
for spec in ${ST_UNITS:?}; do u=${spec%@*}; SPECS+=("$u|$(echo "$u" | tr ':,=' '---' | tr -d '-')"); done

GPU=1; [ "${ST_CPU:-0}" = 1 ] && GPU=0
# CUDA 13.3 unpacked without root, as zk_cost.sh does, then the prover (flock-circuit --features gpu)
CR=$X/cuda-root; todo="cuda-nvcc-13-3 cuda-cudart-dev-13-3"; seen=
[ $GPU = 1 ] || todo=
[ -n "$todo" ] && ! apt-cache show cuda-nvcc-13-3 > /dev/null 2>&1 && [ "$(id -u)" = 0 ] && apt-get update -qq > "$O/apt-update.log" 2>&1
while [ -n "$todo" ]; do
  set -- $todo; q=$1; shift; todo="$*"
  case " $seen " in *" $q "*) continue ;; esac
  seen="$seen $q"
  todo="$todo $(apt-cache show "$q" 2>/dev/null | awk '/^Depends:/ {sub(/^Depends: /, ""); print; exit}' | tr ',' '\n' \
    | sed 's/(.*//; s/|.*//; s/ //g' | grep -E -- '-13-3$' | tr '\n' ' ')"
done
if [ $GPU = 1 ]; then
  mkdir -p "$CR/debs" && (cd "$CR/debs" && apt-get download $(for q in $seen; do
    echo "$q=$(apt-cache show "$q" | awk '/^Version:/ {print $2; exit}')"; done)) > "$O/cuda-unpack.log" 2>&1 \
    && for d in "$CR"/debs/*.deb; do dpkg-deb -x "$d" "$CR"; done
  [ -x "$CR/usr/local/cuda-13.3/bin/nvcc" ] || { log "no CUDA 13.3"; tail -5 "$O/cuda-unpack.log"; exit 1; }
  export CUDA_TK=$CR/usr/local/cuda-13.3 LD_LIBRARY_PATH=$CR/usr/local/cuda-13.3/compat:$CR/usr/local/cuda-13.3/lib64:${LD_LIBRARY_PATH:-}
fi
if [ -x "${ST_BIN:-}" ]; then
  BIN=$ST_BIN; log "build: reusing $BIN"
else
  mkdir -p "$O/m0-build"
  log "build: building flock-circuit (GPU=$GPU)"
  RESEARCH_RUN_DIR=$O/m0-build FLOCK_WORK=$X MODE=build GPU=$GPU taskset -c "$PC" bash verity/ml/flock/pod/60-circuit.sh \
    > "$O/m0-build/build.log" 2>&1 || { tail -40 "$O/m0-build/build.log"; exit 6; }
  BIN=$X/flock-gpu-link/flock/target/release/flock-circuit
  [ -n "${ST_KEEP_BIN:-}" ] && mkdir -p "$(dirname "$ST_KEEP_BIN")" && cp "$BIN" "$ST_KEEP_BIN" && log "build: kept as $ST_KEEP_BIN"
fi
sha256sum "$BIN" | tee "$O/binaries.sha256" | tee -a "$O/job.log"
export CUDA_VISIBLE_DEVICES=${ST_GPU:-0} FC_PHASE_TIMES=1
if [ "${ST_MPS:-0}" = 1 ] && [ $GPU = 1 ]; then
  export CUDA_MPS_PIPE_DIRECTORY=$X/mps-pipe CUDA_MPS_LOG_DIRECTORY=$X/mps-log
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  nvidia-cuda-mps-control -d && MPS=1 && log "MPS: control daemon up (the provers share one device context)"
  [ -n "$MPS" ] || { log "MPS: the control daemon didn't start"; exit 1; }
fi
CPUFLAG=(); [ $GPU = 1 ] || CPUFLAG=(--cpu)

# zk P K [ARGS]: plan point P's warm process (zk_cost.py prove) by K provers, its ZKCOST lines into job.log: ST_ARMS1's arms
# at one prover, ST_ARMS' and FC_VERIFY_AHEAD=ST_VERIFY_AHEAD at more
zk() {
  local p=$1 k=$2 arms=${ST_ARMS1:-${ST_ARMS:-dev}} ahead=; shift 2
  [ "$k" = 1 ] || { arms=${ST_ARMS:-dev}; ahead=${ST_VERIFY_AHEAD:-}; }
  FC_VERIFY_AHEAD=$ahead taskset -c "$PC" "$PY" benchmarks/one_stage/zk_cost.py prove --bin "$BIN" --arms "$arms" --cold 0 \
    --warm 1 --runs "${ST_RUNS:-2}" --points "$p" --out "$Z" --provers "$k" "${CPUFLAG[@]}" "$@" 2>&1 | tee -a "$O/zk.log" \
    | grep ZKCOST | tee -a "$O/job.log"
  return "${PIPESTATUS[0]}"
}

# prove P: plan point P by one prover; with ST_OVERLAP by K provers at once against one verifier, for each K of it at which
# the point's c is full/K or full/(2K) (full: its type's largest staged count). A prover's device arena is about
# 16.5 x 2^(m-3) bytes, held for its process (66 GiB at m = 35), so K provers of full/K blocks fill one device as one of full
# does, and every K gets two sizes
prove() {
  local p=$1 k c full rcs=0 on ons=("")
  [ "${ST_CACHE:-0}" = only ] && ons=(--stmt-cache)
  [ "${ST_CACHE:-0}" = both ] && ons=("" --stmt-cache)
  if [ -z "${ST_OVERLAP:-}" ]; then
    for on in "${ons[@]}"; do zk "$p" 1 $on || rcs=1; done
    return $rcs
  fi
  read -r c full < <("$PY" -c 'import json, sys; pl = json.load(open(sys.argv[1])); q = pl[int(sys.argv[2])]
print(q["c"], max(x["c"] for x in pl if x["tag"] == q["tag"]))' "$Z/plan.json" "$p")
  for k in ${ST_OVERLAP//,/ }; do
    [ $((c * k)) = "$full" ] || [ $((2 * c * k)) = "$full" ] || continue
    for on in "${ons[@]}"; do
      if [ "$DEADLINE" != 0 ] && [ "$(date +%s)" -ge "$DEADLINE" ]; then log "point $p, $k at once $on: past the deadline"; continue; fi
      zk "$p" "$k" $on || rcs=1
    done
  done
  return $rcs
}

# cache P...: with ST_CACHE=1, the statement cache on for one type's points P: at one prover on its largest statement, and at
# the point and K whose provers proved the most blocks a second with it off
cache() {
  [ "${ST_CACHE:-0}" = 1 ] || return 0
  local rcs=0 p k
  while read -r p k; do
    if [ "$DEADLINE" != 0 ] && [ "$(date +%s)" -ge "$DEADLINE" ]; then log "point $p, cache, $k at once: past the deadline"; continue; fi
    zk "$p" "$k" --stmt-cache || rcs=1
  done < <("$PY" - "$Z/records/prove.jsonl" "$Z/plan.json" "$@" <<'EOF2'
import json, sys
plan, pts = json.load(open(sys.argv[2])), [int(x) for x in sys.argv[3:]]
idx = {(plan[p]["tag"], plan[p]["c"]): p for p in pts}
recs = [json.loads(x) for x in open(sys.argv[1]) if x.strip()]
rate = {(idx[(r["tag"], r["c"])], r["provers"]): r["provers"] * r["c"] / (sum(r["cycle_s"]) / len(r["cycle_s"]))
        for r in recs if r.get("cycle_s") and r["accepted"] and not r.get("stmt_cache") and (r["tag"], r["c"]) in idx}
first = max(pts, key=lambda p: plan[p]["c"])
for x in dict.fromkeys([(first, 1)] + ([max(rate, key=rate.get)] if rate else [])):
    print(*x)
EOF2
)
  return $rcs
}

keep() {
  (cd "$Z" && find . -type f -size -64M ! -name circuit.txt ! -name '*.bin' ! -path '*/sessions/*' -print0 \
    | xargs -0 -r -I{} cp --parents {} "$O/")
}
if [ -n "${ST_SERVER:-}" ]; then
  log "server: waiting for every type's staging"
  while :; do
    left=0
    for e in "${SPECS[@]}"; do t=${e#*|}; [ -e "$Z/$t/staged" ] || [ -e "$Z/$t/failed" ] || left=1; done
    [ $left = 0 ] && break
    if [ "$DEADLINE" != 0 ] && [ "$(date +%s)" -ge "$DEADLINE" ]; then log "server: past the deadline while staging"; break; fi
    sleep 15
  done
  # a stream's prover loads every statement before its first session, so a type the device prover refuses is left out
  TYPES=()
  for e in "${SPECS[@]}"; do
    t=${e#*|}; [ -e "$Z/$t/failed" ] && log "${e%|*}: stage refused or failed: $(grep -vE '^\s' "$Z/$t/stage.log" | tail -c 600)"
    [ -e "$Z/$t/staged" ] || continue
    log "${e%|*}: staged $("$PY" -c 'import glob, json, sys
print(", ".join("%s at k_log %s (%s s)" % (r["n"], r["k_log"], r["seconds"]["total"])
                for r in (json.load(open(f)) for f in sorted(glob.glob(sys.argv[1] + "/c*/stage/stage.json")))))' "$Z/$t")"
    d=$(ls -dv "$Z/$t"/c*/stage | head -1)
    # the check loads the verifier's statement (circuit and public file) once: its seconds and peak RSS are a type's
    # load cost, kept in load.json, which a later run with ST_ZK reuses
    grep -q '"rc": 0' "$Z/$t/load.json" 2>/dev/null || taskset -c "$PC" "$PY" - "$Z/$t" "$BIN" statement --zk --gpu \
      --instances "$d/pub.bin" --circuit "$d/circuit.txt" <<'EOF2'
import json, os, resource, subprocess, sys, time
z, cmd = sys.argv[1], sys.argv[2:]
t0 = time.time()
with open(z + "/device-check.log", "w") as f:
    rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
json.dump({"rc": rc, "seconds": round(time.time() - t0, 1), "circuit_gib": round(os.path.getsize(cmd[-1]) / 2**30, 2),
           "maxrss_gib": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20, 2)}, open(z + "/load.json", "w"))
EOF2
    if grep -q '"rc": 0' "$Z/$t/load.json" 2>/dev/null; then
      TYPES+=("${e%|*}"); log "${e%|*}: the device prover takes it; load $(cat "$Z/$t/load.json")"
    else
      log "${e%|*}: the device prover refuses it: $(tail -c 400 "$Z/$t/device-check.log")"
    fi
  done
  # ST_SERVER_TYPES: only these of them, e.g. the ones whose statements fit the host's memory together
  [ -n "${ST_SERVER_TYPES:-}" ] && TYPES=($(for x in "${TYPES[@]}"; do [[ ",$ST_SERVER_TYPES," == *",$x,"* ]] && echo "$x"; done))
  OK=$(IFS=,; echo "${TYPES[*]}")
  [ -n "$OK" ] || { log "server: no type to prove"; exit 1; }
  rc=0; KS=(${ST_SERVER//,/ }); n=${#KS[@]}
  for k in "${KS[@]}"; do
    # each K its share of what is left before the deadline
    left=$(( (DEADLINE == 0 ? 3 * 3600 * n : DEADLINE - $(date +%s)) / n )); n=$((n - 1))
    [ "$left" -gt 300 ] || { log "server: K=$k not run: past the deadline"; rc=1; continue; }
    FC_VERIFY_AHEAD=${ST_VERIFY_AHEAD:-1} taskset -c "$PC" "$PY" benchmarks/one_stage/proof_server.py run --bin "$BIN" \
      --mix "${ST_MIX:?}" --staged "$Z" --types "$OK" --provers "$k" --sessions "${ST_SESSIONS:-120}" --warm "${ST_SERVER_WARM:-0}" \
      --out "$O/server-k$k" --scratch "$W/server-k$k" --deadline "$left" --mem-limit "${ST_MEM_GIB:-0}" \
      $( [ "${ST_ROUTE:-0}" = 1 ] && echo --route) --split "${ST_SPLIT:-0}" "${CPUFLAG[@]}" 2>&1 | tee -a "$O/server.log" \
      | grep SERVER | tee -a "$O/job.log"
    [ "${PIPESTATUS[0]}" = 0 ] || rc=1
    rm -rf "$W/server-k$k"
  done
  keep
  log "server: done, rc $rc"
  exit "$rc"
fi
rc=0; declare -A DONE
while :; do
  ran=0; pending=0
  for e in "${SPECS[@]}"; do
    u=${e%|*}; t=${e#*|}; sd=$Z/$t
    [ -n "${DONE[$t]:-}" ] && continue
    if [ -e "$sd/failed" ]; then
      DONE[$t]=1; log "$u: stage refused or failed: $(grep -vE '^\s' "$sd/stage.log" | tail -c 600)"; continue
    fi
    [ -e "$sd/staged" ] || { pending=1; continue; }
    DONE[$t]=1
    if [ "$DEADLINE" != 0 ] && [ "$(date +%s)" -ge "$DEADLINE" ]; then log "$u: staged after the deadline, not proved"; continue; fi
    pts=$("$PY" - "$Z" "$u" "$t" "$sd" "${ST_ARMS:-dev},${ST_ARMS1:-}" 2>> "$O/job.log" <<'EOF'
import json, sys
from pathlib import Path
z, u, t, sd, arms = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4]), list(dict.fromkeys(filter(None, sys.argv[5].split(","))))
plan, pts = json.loads((z / "plan.json").read_text()), []
for st_file in sorted(sd.glob("c*/stage/stage.json"), key=lambda f: -json.loads(f.read_text())["n"]):
    st, d = json.loads(st_file.read_text()), st_file.parent.parent
    c = st["n"]
    (d / f"draw-k{c}.json").write_text(json.dumps({"law": "subset", "population": c, "k": c, "units": list(range(c))}))
    plan.append({"unit": u, "tag": t, "c": c, "k": c, "arms": arms, "dir": str(d), "stage_dir": str(d / "stage"),
                 "pin": (d / "stage" / "pin.txt").read_text().strip(), "inst": str(d / "stage" / "inst.bin"),
                 "pub": str(d / "stage" / "pub.bin"), "partition": None})
    pts.append(str(len(plan) - 1))
    print({k: st.get(k) for k in ("n", "k_log", "ands", "unit_nnz", "macs", "maxrss_gb", "circuit_bytes")}, st["seconds"],
          file=sys.stderr)
(z / "plan.json").write_text(json.dumps(plan, indent=1))
print(",".join(pts))
EOF
)
    [ -n "$pts" ] || { log "$u: no points"; rc=1; continue; }
    log "$u: staged; prove points $pts"
    for p in ${pts//,/ }; do
      if [ "$DEADLINE" != 0 ] && [ "$(date +%s)" -ge "$DEADLINE" ]; then log "$u: point $p not proved: past the deadline"; continue; fi
      prove "$p" || rc=1
    done
    cache ${pts//,/ } || rc=1
    keep
    rm -f "$sd"/c*/stage/circuit.txt "$sd"/c*/stage/*.bin
    ran=1; break
  done
  [ $ran = 1 ] && continue
  [ $pending = 1 ] || break
  if [ "$DEADLINE" != 0 ] && [ "$(date +%s)" -ge "$DEADLINE" ]; then log "deadline: types still staging are not proved"; break; fi
  sleep 15
done
keep
log "stationary: done, rc $rc"
exit "$rc"
