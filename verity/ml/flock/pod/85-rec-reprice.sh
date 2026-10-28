#!/usr/bin/env bash
# rec-reprice: recursive witness ZK priced at m = 35 on node 1's GPU prover, against today's --zk on the same statement (the
# dense rollup's K = 4096 GEMM coordinate, shape 430e5aad). One step per job (research run --queue
# --cwd source); the big files stay under $FLOCK_WORK/data/$TAG for the next step, out/ gets the records:
#   STEP=build   (CPU) the GPU prover (60-circuit.sh's build, keyed by its sources as 70-class-sweep.sh keys it), Lean's
#                flock-verify (rec-env.sh's, keyed by its sources), and the inner statement staged (class_statement
#                --program-module rec_reprice.py:program --partition q-template-instance --stage-only, BATCH=fixed N,
#                untiled, every instance's own rows: FLOCK_OWN_ROWS=1) twice: with plain SHA-512 leaves (FLOCK_PLAIN_LEAVES=1, the
#                recursive session's) into data/$TAG/stage, and with hm96 leaves (today's --zk) into data/$TAG/stage-zk
#                (STAGE_FROM=DIR: another job's data/$TAG, both stagings; STAGE=0: neither, the caller filled
#                data/$TAG/stage: circuit.txt, inst-N.bin, pub-N.bin, partition.json, program.json and draw.json, and with
#                an own-partition file there the later steps keep its partition, program and draw)
#   STEP=inner   (GPU leases) the plain inner statement proved with ZK off (M0, FC_COINS=os) against the rec proxy
#                (verity_flock.rec_live, the inner phase's firewall: salted tree tops, the schedule of the statement's public
#                shape, rec_live --schedule-for; Register answered with the verifier's full draw, rec_live.full_draw over the
#                inner statement's program and partition, data/$TAG/stage/draw.json; WARM + RUNS sessions), upstream's
#                replay of the last one and V*'s hashing
#                reference (rec_vstar) over it, and over two forged copies of its private.json (a round's top: one node one
#                bit off, one salt one bit off), which it must refuse (CHECKS=0: STEP=icheck's); then (INNER_LOOPBACK=1) the
#                hm96 statement on loopback, M0 and --zk, WARM + RUNS sessions each (today's numbers)
#   STEP=icheck  (CPU) the replay and rec_vstar on the inner's last session, the schedule the sessions were held to against
#                the statement's (derived again), each session's own schedule, how it ended and its message count, the coin
#                server's record against the clear tops, and the tree-top controls rec_vstar must refuse by name
#   STEP=vstage  (CPU) V*'s statements over the inner's last proxied session (verity_flock.rec_vstage) into data/$TAG/v: the
#                verifier's fold points (rec_vstage points), its extra claims and comb (InnerFold.lean as an executable,
#                cached by its sources, on the plain statement: --plain-leaves), then each level's RecOpen, the algebra's
#                InnerRepCheck and the forgeries (FORGE="message sum comb top top-salt coef dirs", each under forged-<kind>/),
#                JOBS at once; the job fails if one does not stage
#   STEP=vcap    (CPU) RecCap alone at every level over the inner's last proxied session (rec_vstage.stage_cap: each
#                level's tops under the firewall's salts, no running sums) into data/$TAG/v/cap (level 0) and v/cap/L<l>,
#                beside the statements vstage staged, whose sessions stay valid (then oprove LEVELS=cap,cap-L1,..)
#   STEP=oprove  (GPU leases) each staged statement proved with --zk on loopback, WARM + RUNS sessions per MODES (dev host
#                firewall; a forgery's: one, in dev mode whatever MODES says, by a prover that sends its proofs though the
#                witness fails, FC_ZK_SELF_CHECK=skip), and upstream's replay --zk of the last (REPLAY=0: STEP=oreplay's);
#                CONTROLS=DIR: the firewall's negative controls on that statement; OPROVE_JOBS=4 statements at once while their estimated
#                peaks sum to at most OPROVE_MEM_GB=48. MODES=firewall alone, WARM=0 RUNS=1: each statement fixed before
#                any session starts gets exactly one independent session, never drawn again, every exit recorded
#                (out/oprove/statements.sha512, exits.tsv)
#   STEP=oreplay (CPU) upstream's replay --zk of each statement's session oprove kept
#   STEP=lean    (CPU) Lean's flock-verify verify --zk on each statement's last session, with V*'s own inputs (the inner
#                shape, the statement's index and bindings, the shared rows, the inner transcript) (INNER_LEAN=1: also the inner's,
#                --coins os and the draw held to its law, --stratified N --program --partition, on
#                verity/flock-circuit+plain-leaves, which Lean's verify checks without --zk
#                (ProvedScope.leafOk); ZK_LEAN=1: also today's --zk session; LEVELS=none: neither of V*'s), and
#                flock-verify statement on the inner statement with the partition object and program vstage wrote. The job
#                fails unless each statement (staged or proved) and ZK_LEAN's session exited 0 with every VERDICT accepted,
#                each forgery refused by Lean at one of its parts (a firewall's refusal, or a session with no proof, is
#                "not exercised" and fails),
#                and the inner statement's units were derived from the program (lean_verdicts.py; INNER_LEAN=1: the
#                inner's session too)
# The prover runs on the job's first PCPUS cores, the verifier (proxy, serve, replay) on the rest. Every step writes
# out/summary-$STEP.json. Only a prove invocation holds a GPU, leased for it alone (on_gpu), so every STEP is launched without
# --gpus (a CPU phase): staging, the proxy, serve, replay and Lean never hold one.
# env: FLOCK_WORK=/workspace/jobs/rec-reprice  TAG=k4096  K=4096 (GemmCoordinate_v2's contraction)  N=2048 (instances: the
#      population program's every unit, rec_reprice.py)
#      CUDA_TK= (default: CUDA_REDIST=13.3.1's archives under FLOCK_WORK)  SM=120  WARM=1  RUNS=3  PCPUS=16
#      LEVELS= (comma-separated statement names, default every one: L3,alg; oprove and lean also forged-sum-L6, alg-p1,
#              ..; a name without its part, alg, names every part)
#      FORGE="message sum comb top top-salt coef dirs" (vstage)  JOBS=4 (vstage)  INNER_LOOPBACK=1 (inner)
#      LEAN_JOBS= LEAN_MEM_GB=48 (lean: verifications at once, default LEAN_MEM_GB / 5 up to the job's cores)
#      REC_CIRCUIT_CACHE=$FLOCK_WORK/rec-circuits (V*'s circuit texts kept per shape, rec_circuits.py; empty: none kept)
#      GPU_MAX_MIN=10 GPU_MEM_GB=32 (each prove's lease)  GPU_PREEMPTIBLE=1 (empty: not preemptible)  GPU_TRIES=3
#      GPU_LEASE= (the lease command; default the queue's for this node)
#      GPU_ON= (the GPUs a lease may take, gpu-lease --on: a NUMA node's, beside the job's taskset)
#      SHIP=1 (build: the binaries also into out/ship, each named by its cache key, for a node that builds nothing)
#      PREBUILT=DIR (another node's out/ship, e.g. research run --send's inputs/: each binary copied into its cache, and a
#              step that would build one refuses instead)
#      WARM_FROM=/workspace/jobs/proofs-vllm-moe/flock-gpu-link (a checkout whose cargo target starts the build warm)
#      MODES=dev  REPLAY=1  CONTROLS= (oprove)  LOOPBACK_DRAW=all (a --zk loopback's draw: every unit; none: no draw)
#      GPU=1 (0: the CPU prover end to end: built GPU=0, no CUDA, no lease, prove without --gpu; for a small K and N)
#      NEG_INNER="challenge query" (lean: each honest V* session verified again over a copy of the inner session with one
#              coin of rep 0 flipped in its session.json and its coin server's record alike, rec_inner.py forge --what W,
#              as part forged-inner-W-<name>; the parts that read the flipped coin refuse it, and lean_verdicts.py requires
#              one to; the step refuses a W none of the parts it checks reads: challenge needs an alg part, query L0;
#              empty: none)
#      REC_COINS=0 (lean; 1: the inner statement set up once per inner session, flock-verify rec-coins, and each verify given
#              that record, --rec-inner-coins, rather than the inner circuit it would parse again)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
STEP=${STEP:?STEP=build|inner|icheck|vstage|vcap|oprove|oreplay|lean}
[ -z "${GPU_LEASE_UUID:-}" ] || { echo "REFUSED: the job holds a GPU lease (launched with --gpus): launch it without, each prove leases its own"; exit 2; }
REPO=$(pwd); O=${RESEARCH_RUN_DIR:-/tmp/rec-reprice-job}/out; mkdir -p $O
FW=${FLOCK_WORK:-/workspace/jobs/rec-reprice}; W=$FW/flock-circuit; L=$FW/cache; TAG=${TAG:-k4096}; DD=$FW/data/$TAG
mkdir -p $W $L $DD || exit 1
[ -w "$HOME" ] || { export HOME=$FW/home; mkdir -p $HOME; }
unset CARGO_TARGET_DIR
# the prover's kernels use clmad, which CUDA 13.0's ptxas (node 1's host toolkit) doesn't know: without CUDA_TK, NVIDIA's
# redistributable CUDA_REDIST archives (sha256-checked) are unpacked once under $FW, and their runtime is the prover's
CUDA_REDIST=${CUDA_REDIST:-13.3.1}; export CUDA_TK=${CUDA_TK:-$FW/cuda-$CUDA_REDIST}
GPU=${GPU:-1}; GPUF=; [ $GPU = 1 ] && GPUF=--gpu
if [ $GPU = 1 ] && [ ! -x $CUDA_TK/bin/nvcc ]; then
  exec 7>$FW/cuda.lock; flock 7
  [ -x $CUDA_TK/bin/nvcc ] || python3 - $CUDA_REDIST $CUDA_TK > $O/cuda-redist.log 2>&1 <<'EOF' || { cat $O/cuda-redist.log; exit 1; }
import hashlib, json, shutil, sys, tarfile, urllib.request
from pathlib import Path
ver, dest = sys.argv[1], Path(sys.argv[2])
base = "https://developer.download.nvidia.com/compute/cuda/redist/"
d = json.load(urllib.request.urlopen(f"{base}redistrib_{ver}.json"))
tmp = dest.with_name(dest.name + ".tmp")
shutil.rmtree(tmp, ignore_errors=True)
tmp.mkdir(parents=True)
for k in ("cuda_nvcc", "cuda_cudart", "cuda_crt", "libnvvm", "cccl", "libnvptxcompiler", "cuda_culibos", "cuda_cuobjdump", "cuda_nvdisasm"):
    x = d[k]["linux-x86_64"]
    data = urllib.request.urlopen(base + x["relative_path"]).read()
    if hashlib.sha256(data).hexdigest() != x["sha256"]:
        raise SystemExit(f"{k}: sha256 mismatch")
    arc = tmp / Path(x["relative_path"]).name
    arc.write_bytes(data)
    with tarfile.open(arc) as t:
        for m in t.getmembers():
            parts = Path(m.name).parts
            if len(parts) > 1:
                m.name = str(Path(*parts[1:]))
                t.extract(m, tmp)
    arc.unlink()
    print(k, d[k]["version"], x["sha256"], flush=True)
(tmp / "lib64").symlink_to("lib")
tmp.rename(dest)
EOF
  exec 7>&-
fi
export PATH=$CUDA_TK/bin:$PATH LD_LIBRARY_PATH=$CUDA_TK/lib64:${LD_LIBRARY_PATH:-}
SM=${SM:-120}; N=${N:-2048}; K=${K:-4096}; WARM=${WARM:-1}; RUNS=${RUNS:-3}; NS=$((WARM + RUNS))
CPUS=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')
NC=$(echo $CPUS | tr ',' '\n' | wc -l); PC=${PCPUS:-16}; [ $PC -ge $NC ] && PC=$(( NC / 2 ))
PCPU=$(echo $CPUS | cut -d, -f1-$PC); VCPU=$(echo $CPUS | cut -d, -f$((PC + 1))-)
{ echo "step=$STEP tag=$TAG cpus=$CPUS prover=$PCPU verifier=$VCPU cuda_visible=${CUDA_VISIBLE_DEVICES:-}"; lscpu | grep -E 'Model name|^CPU\(s\)'
  free -g | head -2; cat /proc/loadavg; nvidia-smi -L 2>/dev/null; } > $O/host-$STEP.txt 2>&1
[ -x /usr/bin/time ] || { echo "no /usr/bin/time" >&2; exit 1; }

KEY=$(cat verity/ml/flock/pod/20-gpu-link.sh verity/ml/flock/pod/60-circuit.sh verity/ml/flock/*.patch verity/ml/flock/cuda_*_patch.py \
  $(find verity/ml/flock/cuda verity/ml/flock/live -type f ! -name '*.md' | LC_ALL=C sort) | sha256sum | cut -c1-16)
BIN=$W/bin-$KEY-g$GPU-sm$SM; B=$BIN/flock-circuit; F=$FW/flock-gpu-link/flock
LSHA=$(cd verity/core/service/flock && find . -name .lake -prune -o -type f -print | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)
LEAN_BIN=$L/lean-$LSHA/flock-verify
# the Lean frame a firewalled prover (MODES=firewall) releases through, built beside flock-verify
export FC_FIREWALL_LEAN=$L/lean-$LSHA/flock-firewall
FSHA=$( (echo $LSHA; cat verity/ml/flock/tests/lean/InnerFold.lean) | sha256sum | cut -c1-16)
FOLD_BIN=$L/fold-$FSHA/inner-fold
# the ship's name for each binary: its cache key in the name, so a node takes only the one its own sources key
SHIPS="$B:flock-circuit-$KEY-g$GPU-sm$SM $LEAN_BIN:flock-verify-$LSHA $FC_FIREWALL_LEAN:flock-firewall-$LSHA $FOLD_BIN:inner-fold-$FSHA"
if [ -n "${PREBUILT:-}" ]; then
  for s in $SHIPS; do
    f=$PREBUILT/${s#*:}; [ -x ${s%%:*} ] || [ ! -f $f ] || { mkdir -p $(dirname ${s%%:*}) && cp $f ${s%%:*}.$$ && chmod +x ${s%%:*}.$$ && mv ${s%%:*}.$$ ${s%%:*}; }
  done
fi
# no_build WHAT: refuse a build under PREBUILT (the node builds nothing)
no_build() { [ -z "${PREBUILT:-}" ] || { echo "REFUSED: PREBUILT=$PREBUILT has no $1 for these sources (out/ship of a SHIP=1 build at this commit)"; exit 2; }; }
UV=$(command -v uv || echo $HOME/.local/bin/uv)
[ -x $UV ] || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; UV=$HOME/.local/bin/uv; }
[ -x $W/py/bin/python ] || { $UV venv -q --python 3.12 $W/py && $UV pip install -q --python $W/py/bin/python numpy blake3; } || exit 1
export PATH=$W/py/bin:$PATH PYTHONPATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/ml/flock/python:$REPO/integrations/vllm:$REPO/tools/verity/circuit_check/src
export FC_COINS=os FC_PIPELINE_DEPTH=${FC_PIPELINE_DEPTH:-4} FC_HOST_PREPIN=${FC_HOST_PREPIN:-1}
export REC_CIRCUIT_CACHE=${REC_CIRCUIT_CACHE-$FW/rec-circuits}
# the lease command: GPU_LEASE, else the queue's own for this node (infra/cluster's submit.wrappers: node 1's pool where its tool
# and pool file are, else gpu-lease's default)
if [ $GPU = 0 ]; then LEASE=(); elif [ -n "${GPU_LEASE:-}" ]; then read -ra LEASE <<< "$GPU_LEASE"; else mapfile -t LEASE < <(python3 - <<'EOF'
import os, sys
sys.path.insert(0, "infra/cluster/src")
from cluster.submit import GPU_LEASE, GPU_LEASE_N1, N1_LEASE_ENV
pool = dict(e.split("=", 1) for e in N1_LEASE_ENV)["GPU_LEASE_ALLOWED_FILE"]
print(*(["env", *N1_LEASE_ENV, GPU_LEASE_N1] if os.access(GPU_LEASE_N1, os.X_OK) and os.path.exists(pool) else [GPU_LEASE]), sep="\n")
EOF
); fi

free_port() { python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
# want NAME: NAME is in LEVELS (comma-separated names; empty: every one)
want() { [ -z "${LEVELS:-}" ] || [[ ",$LEVELS," == *",$1,"* || ",$LEVELS," == *",${1%-p[0-9]*},"* ]]; }
# stamp: each line of stdin with the time it arrived (a session's LIVE line is printed when its verdict is in)
stamp() { while IFS= read -r l; do printf '%s\t%s\n' "$(date +%s.%N)" "$l"; done; }
wait_serving() { for i in $(seq 3600); do grep -q SERVING $1 && return 0; kill -0 $2 2>/dev/null || return 1; sleep 1; done; return 1; }

# on_gpu DIR CMD...: CMD with one GPU leased for it alone, the only place this script holds a GPU (--wait; at most GPU_MAX_MIN
# minutes and GPU_MEM_GB of host memory; preemptible unless GPU_PREEMPTIBLE is empty: a waiter may stop it, CMD then exits 143
# and `again` runs its step afresh). CMD's stdout passes through; its stderr and the lease's own lines (`gpu-lease: ...`) go to
# DIR/prove.err, and the leased GPU's memory in use (MiB) to DIR/gpu-mib.txt every half second while CMD runs.
on_gpu() {
  local d=$1 pre=${GPU_PREEMPTIBLE-1}; shift
  : > $d/prove.err
  if [ $GPU = 0 ]; then "$@" 2> $d/prove.err; return $?; fi
  gpu_mib $d & local s=$!
  "${LEASE[@]}" 1 --wait ${GPU_ON:+--on $GPU_ON} --max-min ${GPU_MAX_MIN:-10} --mem-gb ${GPU_MEM_GB:-32} ${pre:+--preemptible} -- "$@" 2> $d/prove.err
  local rc=$?
  kill $s; wait $s 2>/dev/null
  return $rc
}
gpu_mib() {
  set +x; local i=
  while [ -z "$i" ]; do sleep 0.5; i=$(sed -n 's/^gpu-lease: GPUs \([0-9]*\).*/\1/p' $1/prove.err); done
  while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i; sleep 0.5; done > $1/gpu-mib.txt 2>/dev/null
}

# again FN ARGS: FN (loopback, proxied: each starts its verifier afresh), again while a waiter preempted its lease (143), at most
# GPU_TRIES times
again() {
  local t rc
  for t in $(seq ${GPU_TRIES:-3}); do
    "$@"; rc=$?; [ $rc = 143 ] || return $rc
    echo "$1 ${2:-}: lease preempted, try $t" | tee -a $O/preempted.txt
  done
  return $rc
}

# prove_live NAME VERIFIER_ADDR INST CIRCUIT [--zk]: WARM + RUNS sessions of the GPU prover on the prover's cores, under a lease
# of its own, into $O/$STEP/NAME (prove.tsv: stamped stdout, prove.err, time-prove.txt, gpu-mib.txt); the prover's status
prove_live() {
  local d=$O/$STEP/$1 addr=$2 inst=$3 circ=$4 zk=${5:-}
  mkdir -p $d; echo $WARM > $d/warm.txt
  # other work on the prover's cores just before it starts (queue jobs naming no pool share 96-127): busy share over 2 s
  python3 - $PCPU > $d/bgload.txt <<'EOF'
import sys, time
cpus = {f"cpu{c}" for c in sys.argv[1].split(",")}
def snap():
    return {f[0]: list(map(int, f[1:])) for f in (l.split() for l in open("/proc/stat")) if f[0] in cpus}
a = snap(); time.sleep(2); b = snap()
busy = [1 - (b[c][3] + b[c][4] - a[c][3] - a[c][4]) / max(1, sum(b[c]) - sum(a[c])) for c in sorted(a)]
print(round(sum(busy) / len(busy), 3))
EOF
  echo "${PENV:-}" > $d/prove-env.txt
  on_gpu $d env ${PENV:-} FC_VERIFY_AHEAD=$NS FC_ZK_SELF_CHECK=${SELF_CHECK:-stop} taskset -c $PCPU /usr/bin/time -v -o $d/time-prove.txt \
    $B prove $zk $GPUF --verifier $addr --instances $inst --circuit $circ --warm $WARM --runs $RUNS | stamp > $d/prove.tsv
  local rc=${PIPESTATUS[0]}
  grep -h panicked $d/prove.err | head -3
  return $rc
}

# loopback NAME PUB INST CIRCUIT [--zk]: a verifier process (serve, on the verifier's cores, every session at once) and
# prove_live against it; the sessions under $DD/$STEP-NAME/sessions; the prover's status
loopback() {
  local name=$1 pub=$2 inst=$3 circ=$4 zk=${5:-} s=$DD/$STEP-$1 PORT=$(free_port) PIN=$(sha512sum $4 | cut -d' ' -f1)
  rm -rf $s; mkdir -p $s $O/$STEP/$name
  # the verifier's draw on a --zk session (Lean refuses a session without one, D12), as rec-private-outer.sh's:
  # LOOPBACK_DRAW=all (default) every unit; none: no draw
  local DRAW=()
  [ -n "$zk" ] && [ "${LOOPBACK_DRAW:-all}" = all ] \
    && DRAW=(--draw subset:$(python3 -c 'import json, sys; print(json.loads(open(sys.argv[1], "rb").readline())["instances"])' $pub))
  taskset -c $VCPU $B serve $zk --listen 127.0.0.1:$PORT --out $s/sessions --instances $pub --circuit $circ --pin $PIN \
    "${DRAW[@]}" --operator loopback --concurrency $((NS + 1)) > $O/$STEP/$name/serve.log 2>&1 &
  local SP=$!
  wait_serving $O/$STEP/$name/serve.log $SP || { tail -20 $O/$STEP/$name/serve.log; return 1; }
  prove_live $name 127.0.0.1:$PORT $inst $circ $zk; local rc=$?
  kill $SP; wait $SP 2>/dev/null
  LAST=$(find $s/sessions -name session.json -printf '%T@ %h\n' | sort -n | tail -1 | cut -d' ' -f2)
  return $rc
}

# hello_at DIR: a firewalled prover's Lean Hello sets each session up at the statement STEP=lean verifies (live PROTOCOL.md
# §10.4), verity/flock-circuit with DIR's partition and program; read only under FC_FIREWALL=1
hello_at() {
  export FC_FIREWALL_STATEMENT=verity/flock-circuit
  FC_FIREWALL_HELLO_ARGS=; local k
  for k in partition program; do [ -f $1/$k.json ] && FC_FIREWALL_HELLO_ARGS+="--$k $1/$k.json "; done
  export FC_FIREWALL_HELLO_ARGS
}

# proxied: the rec proxy (WARM + RUNS sessions, each to $DD/inner/s<k>, held to $O/inner/schedule.json, Register answered
# with $DD/stage/draw.json) and prove_live against it; the prover's status
proxied() {
  rm -rf $DD/inner; mkdir -p $DD/inner
  taskset -c $VCPU python3 -m verity_flock.rec_live --listen 127.0.0.1:0 --out $DD/inner --sessions $NS --schedule $O/inner/schedule.json \
    --draw $DD/stage/draw.json > $O/inner/proxy/proxy.log 2>&1 &
  local PP=$!
  wait_serving $O/inner/proxy/proxy.log $PP || { cat $O/inner/proxy/proxy.log; return 1; }
  prove_live proxy $(awk '/^SERVING/ {print $2}' $O/inner/proxy/proxy.log) $D/inst-$N.bin $D/circuit.txt; local rc=$?
  [ $rc = 0 ] || kill $PP 2>/dev/null
  wait $PP; echo "proxy exit $?"
  # one session is written into --out itself, not s0 (rec_live.serve)
  if [ $NS = 1 ] && [ -e $DD/inner/public.json ]; then
    mkdir -p $DD/inner/s0 && for f in $DD/inner/*; do case ${f##*/} in s0|schedule.json) ;; *) mv $f $DD/inner/s0/ ;; esac; done
  fi
  return $rc
}

summary() {
  python3 - $O $STEP "$WARM" "$RUNS" <<'EOF'
import json, os, re, statistics, sys
from pathlib import Path
o, step, warm, runs = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

def text(p):
    return p.read_text() if p.exists() else ""

def tv(p, key):
    m = re.search(key + r".*: (\S+)", text(p))
    return m.group(1) if m else None

def gb(p):
    rss = tv(p, r"Maximum resident set size \(kbytes\)")
    return round(int(rss) / 2**20, 2) if rss else None

def live(d):
    """The WARM + RUNS sessions of one prove_live: per-statement medians over the timed ones, the session time as the
    rollup's (time between the last warm session's record and the last timed one's, over RUNS)."""
    rows = [(float(t), json.loads(l[5:])) for t, l in (ln.split("\t", 1) for ln in text(d / "prove.tsv").splitlines()) if l.startswith("LIVE")]
    served = sorted((json.loads(l.split("\t", 1)[1]) for l in text(d / "serve.log").splitlines() if l.startswith("SESSION\t")),
                    key=lambda v: v.get("session", ""))
    if not rows:
        return {"error": text(d / "prove.err")[-1500:], "served": served}
    warm = int(text(d / "warm.txt").strip() or sys.argv[3])
    timed = [(t, v) for t, v in rows if not v.get("warm")]
    warmed = [(t, v) for t, v in rows if v.get("warm")]
    def med(f, vs=None):
        xs = [x for x in (f(v) for v in (vs if vs is not None else [v for _, v in timed])) if x is not None]
        return statistics.median(xs) if xs else None
    t0 = warmed[-1][0] if warmed else None
    mib = [int(x) for x in text(d / "gpu-mib.txt").split() if x.strip().isdigit()]
    # the verifier's own seconds per session (serve's SESSION records, by session name), and --zk's mask-rank check
    # (ZKRANK, one per rep, in order), each summed per session; both over the timed sessions only
    ranks = [json.loads(l.split("\t", 1)[1])["s"] for l in text(d / "prove.err").splitlines() if l.startswith("ZKRANK\t")]
    per = len(ranks) // len(rows) if ranks and len(ranks) % len(rows) == 0 else 0
    rank_s = [sum(ranks[i * per:(i + 1) * per]) for i in range(len(rows))][warm:] if per else []
    # the prover process's CPU seconds over all its sessions, the firewall's per-session timings (both reps summed), its refusals
    cpu = [float(tv(d / "time-prove.txt", k) or "nan") for k in (r"User time \(seconds\)", r"System time \(seconds\)")]
    firewall = {k: med(lambda v: sum(b.get(k) or 0 for b in v.get("buckets") or [])) for k in ("t.firewall_recheck", "cpu.firewall_recheck")}
    stops = [l for l in text(d / "prove.err").splitlines() if l.startswith(("FIREWALL-REFUSED", "PROVER-STOPPED"))]
    return {"sessions": len(rows), "timed": len(timed), "accepted": [v.get("accepted") for _, v in rows],
            "env": text(d / "prove-env.txt").strip(), "prover_cpu_s": sum(cpu), "firewall": firewall, "refusals": stops[:8],
            "why": [v.get("why") for _, v in rows if v.get("accepted") is not True],
            "serve_why": [v.get("why") for v in served if v.get("accepted") is not True],
            "stopped": [v.get("prover_stopped") for _, v in rows],
            "prover_cores_busy_before": float(text(d / "bgload.txt").strip() or "nan"),
            "serve_verify_s": med(lambda v: v.get("verify_s"), served[warm:]),
            "serve_verify_all_s": [v.get("verify_s") for v in served], "serve_accepted": [v.get("accepted") for v in served],
            "zkrank_s": statistics.median(rank_s) if rank_s else None, "zkrank_all_s": ranks,
            "prove_total_s": med(lambda v: v.get("prove_total_s")), "prove_s": timed[-1][1].get("prove_s") if timed else None,
            "e2e_s": med(lambda v: v.get("e2e_s")), "witness_prebuilt_s": med(lambda v: v.get("witness_prebuilt_s") or 0),
            "session_s": (timed[-1][0] - t0) / len(timed) if t0 and timed else None,
            "proof_bytes": timed[-1][1].get("proof_bytes") if timed else None, "k_log": rows[0][1].get("k_log"),
            "dense_m": rows[0][1].get("dense_m"), "instances": rows[0][1].get("instances"),
            "verify_s": med(lambda v: v.get("verify_s") or 0) if any(v.get("verify_s") for _, v in timed) else None,
            "statement_digest": rows[0][1].get("statement_digest"), "prover_peak_gb": gb(d / "time-prove.txt"),
            "gpu_peak_mib": max(mib) if mib else None, "gpu_idle_mib": min(mib) if mib else None,
            "lease": [l for l in text(d / "prove.err").splitlines() if l.startswith("gpu-lease:")],
            "buckets": [{k: x for k, x in b.items() if k.startswith(("t.", "cpu.", "firewall_")) or k in ("device_witness", "native_sha")} for b in (timed[-1][1].get("buckets") or [])]
            if timed else None}

def replay(p):
    up = next((json.loads(l.split("\t", 1)[1]) for l in text(p).splitlines() if l.startswith("UPSTREAM")), None)
    return up if up else text(p)[-600:]

def lean(p):
    at = text(p).find("VERDICT ")
    if at < 0:
        # `flock-verify statement`'s line, read as lean_verdicts.py reads it
        st = next((json.loads(l) for l in text(p).splitlines() if l.startswith("{") and '"partition_check"' in l), None)
        if st is None:
            return text(p)[-600:]
        pc = st.get("partition_check") or {}
        return {"accepted": pc.get("units") == "derived", "why": pc.get("note", ""), "partition_check": pc,
                **{k: st.get(k) for k in ("statement_digest", "sigma", "circuit_sha512", "class_sha512")}}
    v = json.JSONDecoder(strict=False).raw_decode(text(p)[at + 8:])[0]
    # Main's stderr: inputs (from start), setup (from the session's start), then the session's whole time (setup included)
    ms = [int(x) for x in re.findall(r"^(?:inputs|setup|/\S+): (\d+) ms$", text(p.with_suffix(".err")), re.M)]
    split = {"inputs_s": ms[0] / 1e3, "setup_s": ms[1] / 1e3, "verify_s": (ms[2] - ms[1]) / 1e3} if len(ms) == 3 else {}
    return {**{k: v.get(k) for k in ("accepted", "why", "registered")}, **split}

s = {"step": step, "warm": warm, "runs": runs}
d = o / step
if step == "build":
    s["binary"] = text(o / "binary.sha256").strip()
    s["lean"] = text(o / "lean.sha").strip()
    for name in ("stage", "stage-zk"):
        s[name] = {"staged": [json.loads(l) for l in text(o / name / "staged.jsonl").splitlines() if l.strip()],
                   "wall": tv(o / name / "time-stage.txt", r"Elapsed \(wall clock\) time"), "peak_gb": gb(o / name / "time-stage.txt"),
                   "sizes": text(o / name / "sizes.txt")}
    s["statement"] = text(o / "statement.txt")[-2000:]
elif step == "inner":
    for name in ("proxy", "m0", "zk"):
        if (d / name).exists():
            s[name] = live(d / name)
    s["proxy_sessions"] = [json.loads(l.split("\t", 1)[1]) for l in text(d / "proxy" / "proxy.log").splitlines() if l.startswith("REC_INNER")]
    for name in ("proxy", "m0", "zk"):
        if (d / f"replay-{name}.txt").exists():
            s[f"replay_{name}"] = {"verdict": replay(d / f"replay-{name}.txt"), "wall": tv(d / f"time-replay-{name}.txt", r"Elapsed \(wall clock\) time"),
                                   "peak_gb": gb(d / f"time-replay-{name}.txt")}
    s["vstar"] = text(d / "vstar.json")[-3000:]
    s["vstar_wall"] = tv(d / "time-vstar.txt", r"Elapsed \(wall clock\) time")
    for f in ("node", "salt"):
        s[f"vstar_forged_{f}"] = (text(d / f"vstar-forged-{f}.json") + text(d / f"vstar-forged-{f}.err"))[-1500:]
elif step == "icheck":
    s["replay_proxy"] = {"verdict": replay(d / "replay-proxy.txt"), "wall": tv(d / "time-replay-proxy.txt", r"Elapsed \(wall clock\) time")}
    s["vstar"] = text(d / "vstar.json")[-3000:]
    s["vstar_wall"] = tv(d / "time-vstar.txt", r"Elapsed \(wall clock\) time")
    s["schedule"] = json.loads(text(d / "schedule.json") or "null")
    s["tops"] = text(d / "tops.txt")
    s["controls"] = {f.stem[4:]: json.loads(text(f) or "null") for f in sorted(d.glob("neg-*.json"))}
elif step == "oreplay":
    s["levels"] = {x.name: {"mode": text(x / "mode.txt").strip(), "replay": replay(x / "replay.txt"), "wall": tv(x / "time-replay.txt", r"Elapsed \(wall clock\) time"),
                            "peak_gb": gb(x / "time-replay.txt"), "sizes": text(x / "sizes.txt")} for x in sorted(p for p in d.iterdir() if p.is_dir())}
elif step == "vstage":
    pts = json.loads(text(d / "points.json")) if text(d / "points.json").strip() else {}
    fold = json.loads(text(d / "fold.json")) if text(d / "fold.json").strip() else {}
    s["points"] = {"m": pts.get("m"), "k_log": pts.get("k_log"), "points_s": pts.get("points_s"), "err": text(d / "points.err")[-600:],
                   "wall": tv(d / "time-points.txt", r"Elapsed \(wall clock\) time"), "peak_gb": gb(d / "time-points.txt")}
    s["fold"] = {"m": fold.get("m"), "k_log": fold.get("k_log"), "extras": len(fold.get("extras", [])), "err": text(d / "fold.err")[-600:],
                 "wall": tv(d / "time-fold.txt", r"Elapsed \(wall clock\) time"), "peak_gb": gb(d / "time-fold.txt")}
    s["statements"] = {}
    for f in sorted(d.glob("*.json")):
        if f.stem in ("points", "fold", "partition", "program"):
            continue
        s["statements"][f.stem] = {"stage": json.loads(text(f).splitlines()[0]) if text(f).strip() else text(d / f"{f.stem}.err")[-1500:],
                                   "wall": tv(d / f"time-{f.stem}.txt", r"Elapsed \(wall clock\) time"), "peak_gb": gb(d / f"time-{f.stem}.txt")}
    s["sizes"] = text(d / "sizes.txt")
elif step == "oprove":
    s["modes"] = os.environ.get("MODES", "dev")
    s["levels"] = {}
    for x in sorted(p for p in d.iterdir() if p.is_dir()):
        s["levels"][x.name] = {"live": live(x), "replay": replay(x / "replay.txt"), "replay_wall": tv(x / "time-replay.txt", r"Elapsed \(wall clock\) time"),
                               "replay_peak_gb": gb(x / "time-replay.txt"), "sizes": text(x / "sizes.txt")}
elif step == "lean":
    s["levels"] = {}
    for f in sorted(d.glob("*.log")):
        s["levels"][f.stem] = {"verdict": lean(f), "wall": tv(d / f"time-{f.stem}.txt", r"Elapsed \(wall clock\) time"),
                               "peak_gb": gb(d / f"time-{f.stem}.txt")}
(o / f"summary-{step}.json").write_text(json.dumps(s, indent=1))
print(json.dumps(s))
EOF
}

lean_build() {
  [ -x $LEAN_BIN ] && [ -x $FC_FIREWALL_LEAN ] && return 0
  no_build "flock-verify-$LSHA or flock-firewall-$LSHA"
  (
    export PATH=$HOME/.elan/bin:$PATH
    command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
    exec 9>$L/lean.lock; flock 9
    if [ ! -x $LEAN_BIN ] || [ ! -x $FC_FIREWALL_LEAN ]; then
      rm -rf $L/lean-src && mkdir -p $L/lean-src && cp -r verity/core/service/flock/. $L/lean-src/ \
        && (cd $L/lean-src && lake build flock-verify flock-firewall) && mkdir -p $L/lean-$LSHA \
        && cp $L/lean-src/.lake/build/bin/flock-verify $L/lean-src/.lake/build/bin/flock-firewall $L/lean-$LSHA/
    fi
  )
}

# fold_build: InnerFold.lean as an executable (vstage's fold points and comb), keyed by its sources and the Lean package's
fold_build() {
  [ -x $FOLD_BIN ] && return 0
  no_build inner-fold-$FSHA
  (
    export PATH=$HOME/.elan/bin:$PATH
    command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
    exec 9>$L/fold.lock; flock 9
    if [ ! -x $FOLD_BIN ]; then
      C=$L/fold-src; rm -rf $C && mkdir -p $C && cp -r verity/core/service/flock/. $C/ && cp verity/ml/flock/tests/lean/InnerFold.lean $C/ \
        && printf '\n[[lean_exe]]\nname = "inner-fold"\nroot = "InnerFold"\n' >> $C/lakefile.toml \
        && (cd $C && lake build inner-fold) && mkdir -p $L/fold-$FSHA && cp $C/.lake/build/bin/inner-fold $FOLD_BIN
    fi
  )
}

# stage_inner NAME PLAIN: the inner statement (N instances of GemmCoordinate_v2{K}, the population program of that template
# under Q_template_instance, rec_reprice.py: the partition's every unit, so the full draw over it proves every unit of the
# statement, which check 5 requires) staged into $DD/NAME from its cache $DD/NAME-cache, its trees' leaves plain SHA-512 when
# PLAIN is 1; its records in $O/NAME. Never tiled (FLOCK_GEMM_TILE): a tile's circuit names its unit class as its template,
# which no template query lists, and at K = 4096 a 4x4 tile does not fit a block anyway. Every instance's own rows
# (FLOCK_OWN_ROWS): Lean covers a shared-row public file only in a registered --zk session (ProvedScope.sharedRowsOk), and
# the population program's instances share no row, so the rows committed are the same
stage_inner() {
  local o=$O/$1 c=$DD/$1-cache
  mkdir -p $o
  K=$K N=$N FLOCK_STAGE_CACHE=$c FLOCK_PLAIN_LEAVES=$2 FLOCK_GEMM_TILE= FLOCK_OWN_ROWS=1 /usr/bin/time -v -o $o/time-stage.txt \
    python3 -m verity_flock.class_statement --program-module $REPO/verity/ml/flock/pod/rec_reprice.py:program \
    --partition q-template-instance --out $o --stage-only --batch fixed --n $N \
    --max-statement-bits $((1 << 35)) --batch-ands $((1 << 35)) --selftest 0 > $o/stage.log 2>&1
  echo "stage $1 exit $?"; tail -5 $o/stage.log
  local C=$(ls -td $c/*/ 2>/dev/null | head -1)
  [ -n "$C" ] && [ -f $C/inst-$N.bin ] || { echo "no staged statement under $c"; return 1; }
  mkdir -p $DD/$1 && for f in circuit.txt inst-$N.bin pub-$N.bin rec.json; do ln -f $C/$f $DD/$1/$f || cp $C/$f $DD/$1/$f; done
  du -sb $DD/$1/* | tee $o/sizes.txt
}

# inner_partition: the verifier's copies of what the inner statement binds, into $DD/stage (rec_reprice.py partition): the
# partition object (Q_template_instance over the population program, partition.json) and that program (program.json), which
# STEP=lean's statement check reads, and the full draw over them (rec_live.full_draw: the stratified law at K = the
# population, every unit; draw.json), which the rec proxy answers Register with
inner_partition() {
  [ -f $DD/stage/own-partition ] && return 0   # STAGE=0's caller wrote them
  K=$K N=$N python3 $REPO/verity/ml/flock/pod/rec_reprice.py partition $DD/stage
}

case $STEP in
build)
  lean_build > $O/build-lean.log 2>&1 &
  LP=$!
  # STAGE_FROM=DIR: another job's staged statements (its data/$TAG: stage and stage-zk) instead of staging them again
  if [ "${STAGE:-1}" = 0 ]; then
    { for f in circuit.txt inst-$N.bin pub-$N.bin partition.json program.json draw.json; do
        [ -s $DD/stage/$f ] || { echo "STAGE=0: no $DD/stage/$f"; exit 1; }; done; } > $O/stage-all.log 2>&1 &
  elif [ -z "${STAGE_FROM:-}" ]; then
    { stage_inner stage 1 && stage_inner stage-zk 0; } > $O/stage-all.log 2>&1 &
  else
    { for s in stage stage-zk; do
        [ -f $STAGE_FROM/$s/inst-$N.bin ] || { echo "no staged statement under $STAGE_FROM/$s"; exit 1; }
        mkdir -p $DD/$s && for f in circuit.txt inst-$N.bin pub-$N.bin rec.json; do ln -f $STAGE_FROM/$s/$f $DD/$s/$f || cp $STAGE_FROM/$s/$f $DD/$s/$f; done
      done; } > $O/stage-all.log 2>&1 &
  fi
  SPID=$!
  if [ ! -x $B ]; then
    no_build flock-circuit-$KEY-g$GPU-sm$SM
    exec 8>$W/build.lock; flock 8
    if [ ! -x $B ]; then
      [ -d $F ] || { [ -d ${WARM_FROM:-/workspace/jobs/proofs-vllm-moe/flock-gpu-link}/flock ] && cp -a ${WARM_FROM:-/workspace/jobs/proofs-vllm-moe/flock-gpu-link} $FW/flock-gpu-link; }
      RESEARCH_RUN_DIR=$O/build FLOCK_WORK=$FW MODE=build GPU=$GPU SM=$SM bash verity/ml/flock/pod/60-circuit.sh > $O/build.log 2>&1 \
        || { tail -40 $O/build.log; tail -40 $O/build/out/build-circuit.txt 2>/dev/null; exit 1; }
      mkdir -p $BIN && cp $F/target/release/flock-circuit $B
    fi
    exec 8>&-
  fi
  sha256sum $B | tee $O/binary.sha256
  wait $SPID; rc=$?; cat $O/stage-all.log; [ $rc = 0 ] || exit 1
  sha256sum $DD/stage/* $(ls $DD/stage-zk/* 2>/dev/null) > $O/stage-sha256.txt
  $B statement --instances $DD/stage/inst-$N.bin --circuit $DD/stage/circuit.txt > $O/statement.txt 2>&1; echo "statement exit $?"
  wait $LP; echo "lean build exit $?"
  [ -x $LEAN_BIN ] && sha256sum $LEAN_BIN > $O/lean.sha
  tail -5 $O/build-lean.log
  if [ "${SHIP:-0}" = 1 ]; then
    fold_build > $O/build-fold.log 2>&1 || { tail -30 $O/build-fold.log; exit 1; }
    mkdir -p $O/ship && for s in $SHIPS; do cp ${s%%:*} $O/ship/${s#*:} || exit 1; done
    (cd $O/ship && sha256sum *) | tee $O/ship.sha256
  fi
  ;;
inner)
  D=$DD/stage; [ -f $D/inst-$N.bin ] && [ -x $B ] || { echo "run STEP=build first"; exit 1; }
  mkdir -p $O/inner/proxy
  # INNER_PROXY=0: only today's numbers again (the proxied sessions V*'s statements were staged from stay as they are)
  if [ "${INNER_PROXY:-1}" = 1 ]; then
  # M0 against the rec proxy: WARM + RUNS sessions, each written to $DD/inner/s<k>, held to the schedule of the statement's
  # public shape (flock-circuit statement on the public file, then rec_live --schedule-for), derived before any session
  taskset -c $VCPU $B statement --instances $D/pub-$N.bin --circuit $D/circuit.txt > $O/inner/statement.txt 2>&1 \
    || { tail -5 $O/inner/statement.txt; exit 1; }
  python3 -m verity_flock.rec_live --schedule-for $O/inner/statement.txt > $O/inner/schedule.json || exit 1
  # the verifier's full draw, which verify requires of a session over the inner session (RecSetup.innerDrawn)
  inner_partition > $O/inner/partition.txt || exit 1
  cp $D/draw.json $O/inner/
  again proxied
  SESS=$DD/inner/s$((NS - 1))
  [ -f $SESS/session/session.json ] || { tail -20 $O/inner/proxy/proxy.log $O/inner/proxy/prove.err; exit 1; }
  # CHECKS=0: the replay and V*'s hashing reference wait for STEP=icheck (CPU), outside the GPU lease
  if [ "${CHECKS:-1}" = 1 ]; then
  taskset -c $VCPU /usr/bin/time -v -o $O/inner/time-replay-proxy.txt $B replay --instances $D/pub-$N.bin --circuit $D/circuit.txt \
    --sessions $SESS/session > $O/inner/replay-proxy.txt 2>&1
  taskset -c $VCPU /usr/bin/time -v -o $O/inner/time-vstar.txt python3 -m verity_flock.rec_vstar $SESS > $O/inner/vstar.json 2> $O/inner/vstar.err
  rc=$?; echo "vstar exit $rc"
  # the prover reports a stop (prover_stopped) and exits 0; the honest session's V* is what refuses one without its proofs
  [ $rc = 0 ] || { cat $O/inner/vstar.json; grep -h -m3 -e panicked -e prover_stopped $O/inner/proxy/prove.err $O/inner/proxy/prove.tsv
    mkdir -p $O/inner/session && cp -r $SESS/session $SESS/public.json $O/inner/session/; exit 1; }
  # the forged sessions V*'s reference must refuse: the session's private.json with the first salted top's first node (node)
  # or its salt (salt) one bit off, the public record and the proofs as they are
  for f in node salt; do
    F=$DD/inner-forged-$f; rm -rf $F; mkdir -p $F; cp $SESS/public.json $F/; ln -s $SESS/session $F/session
    python3 - $SESS/private.json $F/private.json $f <<'EOF'
import json, sys
p = json.load(open(sys.argv[1]))
op = next(r for r in p["rounds"] if r.get("tops"))["tops"][0]
key = {"node": "nodes", "salt": "salts"}[sys.argv[3]]
op[key][0] = "%02x" % (int(op[key][0][:2], 16) ^ 1) + op[key][0][2:]
json.dump(p, open(sys.argv[2], "w"))
EOF
    taskset -c $VCPU python3 -m verity_flock.rec_vstar $F > $O/inner/vstar-forged-$f.json 2> $O/inner/vstar-forged-$f.err
    echo "vstar forged-$f exit $? (1: refused)"
  done
  fi
  mkdir -p $O/inner/session && cp -r $SESS/session $SESS/public.json $DD/inner/schedule.json $O/inner/session/ && du -sb $SESS/* > $O/inner/session/sizes.txt
  fi
  # today's numbers on this node and build: the hm96 statement's M0 and --zk on loopback, the verifier on its own cores, every
  # verdict ahead
  if [ "${INNER_LOOPBACK:-1}" = 1 ]; then
  DZ=$DD/stage-zk; [ -f $DZ/inst-$N.bin ] || { echo "no $DZ (run STEP=build first)"; exit 1; }
  again loopback m0 $DZ/pub-$N.bin $DZ/inst-$N.bin $DZ/circuit.txt
  again loopback zk $DZ/pub-$N.bin $DZ/inst-$N.bin $DZ/circuit.txt --zk
  [ -n "${LAST:-}" ] && taskset -c $VCPU /usr/bin/time -v -o $O/inner/time-replay-zk.txt $B replay --zk --instances $DZ/pub-$N.bin \
    --circuit $DZ/circuit.txt --sessions $LAST > $O/inner/replay-zk.txt 2>&1
  fi
  ;;
icheck)
  # (CPU) the inner's last proxied session: upstream's replay, V*'s hashing reference over the salted tops (rec_vstar), that
  # the coin server's record holds none of the clear tops, and the tree-top controls (rec_inner.py forge: a top that is not
  # the committed one, a record of unsalted tops), each of which V*'s inputs must refuse by name
  D=$DD/stage; SESS=${INNER:-$DD/inner/s$((NS - 1))}
  [ -f $SESS/session/session.json ] || { echo "run STEP=inner first"; exit 1; }
  mkdir -p $O/icheck
  /usr/bin/time -v -o $O/icheck/time-replay-proxy.txt $B replay --instances $D/pub-$N.bin --circuit $D/circuit.txt \
    --sessions $SESS/session > $O/icheck/replay-proxy.txt 2>&1; echo "replay exit $?"
  /usr/bin/time -v -o $O/icheck/time-vstar.txt python3 -m verity_flock.rec_vstar $SESS > $O/icheck/vstar.json 2> $O/icheck/vstar.err
  echo "vstar exit $?"
  # the schedule the sessions were held to against the statement's (derived again from the public file), each session's own
  # schedule against it, how each ended, and each finished one's messages (the stop bound's: Hello, Commit, Link, every
  # Open, Round and Proof, Finish)
  $B statement --instances $D/pub-$N.bin --circuit $D/circuit.txt > $O/icheck/statement.txt 2>&1; echo "statement exit $?"
  python3 - $DD/inner $O/icheck/statement.txt > $O/icheck/schedule.json <<'EOF'
import json, sys
from pathlib import Path
from verity_flock import rec_live as R
d = Path(sys.argv[1])
pin = json.loads((d / "schedule.json").read_text())
ss = sorted(d.glob("s*/public.json"), key=lambda p: int(p.parent.name[1:]))

def messages(p):
    rec = json.loads((p.parent / "session" / "session.json").read_text())
    return sum(len(st["rounds"]) for st in rec["streams"]) + len(rec["streams"]) + len(rec["proof_sha512"]) + 4

print(json.dumps({"held_to_the_statements": pin == R.schedule_for(R.statement_of(Path(sys.argv[2]).read_text())), "sessions": len(ss),
                  "held": [p.parent.name for p in ss if (p.parent / "session").exists() and R.schedule_of(json.loads(p.read_text())) == pin],
                  "end": {p.parent.name: json.loads(p.read_text()).get("end") for p in ss},
                  "messages": {p.parent.name: messages(p) for p in ss if (p.parent / "session").exists()}}))
EOF
  cat $O/icheck/schedule.json
  python3 - $SESS > $O/icheck/tops.txt <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
pub, priv = json.loads((d / "public.json").read_text()), json.loads((d / "private.json").read_text())
text = (d / "public.json").read_text()
opened = [o for r in priv["rounds"] for o in r.get("tops", [])] + list(priv["roots"].values())
nodes = {n for o in opened for n in o["nodes"]}
print(json.dumps({"tops": pub.get("tops"), "clear_tops": len(opened), "clear_nodes": len(nodes), "clear_nodes_in_public": sum(n in text for n in nodes),
                  "public_bytes": len(text), "salted_top_bytes": sum(len(b) // 2 for r in pub["rounds"] for t in r["tops"] for b in t)
                  + sum(len(b) // 2 for v in pub["link"]["roots"].values() for b in v),
                  "message_rows": pub["message_rows"], "rounds": len(pub["rounds"])}))
EOF
  cat $O/icheck/tops.txt
  for w in top unsalted; do
    rm -rf $DD/neg-$w && mkdir -p $DD/neg-$w && cp -r $SESS/public.json $SESS/private.json $DD/neg-$w/ && ln -s $SESS/session $DD/neg-$w/session
    python3 verity/ml/flock/pod/rec_inner.py forge --what $w --inner $DD/neg-$w > $O/icheck/forge-$w.json
    python3 -m verity_flock.rec_vstar $DD/neg-$w > $O/icheck/neg-$w.json 2> $O/icheck/neg-$w.err; echo "neg $w vstar exit $?"; cat $O/icheck/neg-$w.json
  done
  ;;
vstage)
  # V*'s statements over the inner's last proxied session: the verifier's fold points (from the coins), its extra claims and
  # comb (InnerFold.lean as an executable, cached by its sources as STEP=build caches flock-verify; no proof read), then the
  # statements, each in its own process: first the verifier's values alone (its native seconds, and the running sums' salts
  # drawn once into $V/acc-salts.bin, the prover's, so that every statement reads one chain), then the algebra, the forgeries
  # and every level, JOBS at once
  INNER=${INNER:-$DD/inner/s$((NS - 1))}
  [ -f $INNER/public.json ] && [ -f $INNER/private.json ] || { echo "INNER=$INNER: no public.json and private.json (run STEP=inner first)"; exit 1; }
  fold_build > $O/build-fold.log 2>&1 || { tail -30 $O/build-fold.log; exit 1; }
  mkdir -p $O/vstage
  inner_partition || exit 1
  cp $DD/stage/partition.json $DD/stage/program.json $O/vstage/
  V=$DD/v; rm -rf $V; mkdir -p $V
  /usr/bin/time -v -o $O/vstage/time-points.txt python3 -m verity_flock.rec_vstage points $INNER $DD/stage/circuit.txt \
    $DD/stage/pub-$N.bin > $V/points.json 2> $O/vstage/points.err || { tail -20 $O/vstage/points.err; exit 1; }
  # an inner statement whose description is registered (stage/registered.json, verity_flock.inner_registered) is folded
  # against that registration (--registered)
  REG=(); [ -s $DD/stage/registered.json ] && REG=(--registered $DD/stage/registered.json)
  /usr/bin/time -v -o $O/vstage/time-fold.txt $FOLD_BIN --plain-leaves --partition $DD/stage/partition.json "${REG[@]}" verity/flock-circuit \
    $DD/stage/circuit.txt $DD/stage/pub-$N.bin $V/points.json os > $V/fold.json 2> $O/vstage/fold.err
  rc=$?; echo "fold exit $rc"; [ $rc = 0 ] || { tail -20 $O/vstage/fold.err; exit 1; }
  cp $V/points.json $V/fold.json $O/vstage/
  ALL=$(python3 -c "
import sys; from pathlib import Path; from verity_flock import rec_outer as O, rec_vstar as V
pub, priv, proofs = V.load(Path(sys.argv[1])); print(' '.join(str(l['level']) for l in O.levels(V.queries(pub, priv, proofs))))" $INNER)
  echo "levels: $ALL"
  vjob() {
    local name=$1; shift
    /usr/bin/time -v -o $O/vstage/time-$name.txt python3 -m verity_flock.rec_vstage stage $INNER $V/fold.json $V \
      --points $V/points.json --salts $V/acc-salts.bin "$@" > $O/vstage/$name.json 2> $O/vstage/$name.err
    local rc=$?; echo "$name exit $rc"; return $rc
  }
  export -f vjob; export O V INNER
  vjob verifier --levels none --no-algebra
  [ -f $V/acc-salts.bin ] || { tail -20 $O/vstage/verifier.err; exit 1; }
  { want alg && echo alg --levels none
    for f in ${FORGE-message sum comb top top-salt coef dirs}; do echo forged-$f --forge $f; done
    for l in $ALL; do want L$l && echo L$l --levels $l --no-algebra; done; } | xargs -P ${JOBS:-4} -L 1 bash -c 'vjob "$@"' _
  [ ${PIPESTATUS[1]} = 0 ] || { echo "vstage: a statement did not stage (out/vstage/<name>.err)"; exit 1; }
  (cd $V && du -sb $(find . -name '*.bin' -o -name circuit.txt | LC_ALL=C sort)) > $O/vstage/sizes.txt
  ;;
vcap)
  INNER=${INNER:-$DD/inner/s$((NS - 1))}
  [ -f $INNER/public.json ] && [ -f $INNER/private.json ] || { echo "INNER=$INNER: no public.json and private.json (run STEP=inner first)"; exit 1; }
  [ -f $DD/v/L0/registered.json ] || { echo "run STEP=vstage first: RecCap is staged beside V*'s other statements"; exit 1; }
  mkdir -p $O/vstage; rm -rf $DD/v/cap
  /usr/bin/time -v -o $O/vstage/time-cap.txt python3 -c '
import json, sys; from pathlib import Path; from verity_flock import rec_vstage as S
print(json.dumps(S.stage_cap(Path(sys.argv[1]), Path(sys.argv[2]))))' $INNER $DD/v/cap > $O/vstage/cap.json 2> $O/vstage/cap.err \
    || { tail -20 $O/vstage/cap.err; exit 1; }
  python3 -c 'import json, sys; r = json.load(open(sys.argv[1])); sys.exit(0 if r and all(c["unit_agrees"] and all(c["unit_agrees"]) for c in r.values()) else "RecCap: a level'"'"'s unit disagrees with its reference")' \
    $O/vstage/cap.json || exit 1
  cp $DD/v/cap/registered.json $O/vstage/cap-registered.json
  for d in $DD/v/cap/L*/; do [ -d "$d" ] && cp $d/registered.json $O/vstage/cap-$(basename $d)-registered.json; done
  ;;
oprove)
  V=$DD/v; [ -n "$(find $V -name inst.bin -print -quit 2>/dev/null)" ] || { echo "run STEP=vstage (or vcap) first"; exit 1; }
  # every staged statement (each level's, each of the algebra's parts, each forgery's: L3, alg-p0, forged-sum-L6, ..), proved with --zk on
  # loopback, a forgery's once; upstream's replay --zk of the last session, which (accepted or not) is kept under $DD/v-sess
  # beside the statement directory Lean reads it with. MODES names the prover's sides, WARM + RUNS sessions each per
  # statement: dev (the device's witness, rec-reprice's), host (FC_HOST_WITNESS=1, no firewall) and firewall (FC_FIREWALL=1
  # on the host witness: the send check and the level-0 recheck, PROTOCOL.md §10); the firewall's cost is firewall's minus
  # host's. The last mode's last session is the one oreplay and lean read.
  # Statements are proved OPROVE_JOBS at once, the largest first, while their estimated peaks (oprove_gb) sum to at most
  # OPROVE_MEM_GB; each in a subshell of its own (its WARM, RUNS, PENV, LAST and hello_at's exports), its log
  # $O/oprove/<name>.log.
  MODES=${MODES:-dev}; MODES=${MODES//,/ }
  for mode in $MODES; do case $mode in dev|host|firewall) ;; *) echo "MODES: dev host firewall"; exit 1 ;; esac; done
  [[ " $MODES " != *" firewall "* ]] || [ -x $FC_FIREWALL_LEAN ] || lean_build > $O/build-lean.log 2>&1 \
    || { tail -20 $O/build-lean.log; exit 1; }
  mkdir -p $DD/v-sess $O/oprove
  # Outer proofs through the firewall are independent sessions (AGENTS.md): every
  # staged statement is fixed before any session draws coins (statements.sha512, checked again once all have ended), gets
  # exactly one session, with its own loopback verifier, live coins and firewall pass, never a second one (no lease retry,
  # no preemption: a failed session is its statement's failure), and every statement's exit is recorded (exits.tsv), an
  # honest one's failure failing the step.
  if [[ " $MODES " == *" firewall "* ]]; then
    [ "$MODES" = firewall ] || { echo "REFUSED: MODES=$MODES: a firewalled oprove gives each staged statement exactly one session; run MODES=dev or host in a job of its own"; exit 2; }
    [ $((WARM + RUNS)) = 1 ] || { echo "REFUSED: WARM=$WARM RUNS=$RUNS: a firewalled oprove gives each staged statement exactly one session (WARM=0 RUNS=1)"; exit 2; }
    export GPU_TRIES=1 GPU_PREEMPTIBLE=
  fi
  (cd $V && find . \( -name circuit.txt -o -name pub.bin -o -name inst.bin -o -name partition.json -o -name program.json \
    -o -name registered.json \) | LC_ALL=C sort | xargs sha512sum) > $O/oprove/statements.sha512
  : > $O/oprove/exits.tsv
  W0=$WARM R0=$RUNS
  oprove_one() {
    local C=$1 l=${1//\//-} n mode rc=0 LM=$MODES
    # a forgery's one session is the dev prover's, a dishonest prover running no firewall: a firewall would hold it back,
    # leaving Lean nothing to check (lean_verdicts.py: "not exercised")
    case $l in forged-*) WARM=0 RUNS=1 SELF_CHECK=skip LM=dev ;; *) WARM=$W0 RUNS=$R0 SELF_CHECK=stop ;; esac; NS=$((WARM + RUNS))
    hello_at $V/$C
    for mode in $LM; do
      case $mode in dev) PENV= ;; host) PENV=FC_HOST_WITNESS=1 ;; firewall) PENV="FC_FIREWALL=1 FC_HOST_WITNESS=1" ;; esac
      n=$l; [ "$MODES" = dev ] || n=$l-$mode
      LAST=; again loopback $n $V/$C/pub.bin $V/$C/inst.bin $V/$C/circuit.txt --zk || rc=$?
    done
    PENV=
    # the last mode's LIVE verdicts, which STEP=lean reads for a FIREWALL-REFUSED stop when no session reached it
    grep -h $'\tLIVE\t' $O/oprove/$n/prove.tsv 2>/dev/null | cut -f3- > $DD/v-sess/$l.live
    du -sb $V/$C/inst.bin $V/$C/pub.bin $V/$C/circuit.txt ${LAST:+$LAST/*.bin} > $O/oprove/$n/sizes.txt
    [ -n "${LAST:-}" ] || { echo "$n: no session"; return 1; }
    # REPLAY=0: upstream's replay waits for STEP=oreplay (CPU), outside the GPU lease
    if [ "${REPLAY:-1}" = 1 ]; then
      taskset -c $VCPU /usr/bin/time -v -o $O/oprove/$n/time-replay.txt $B replay --zk --instances $V/$C/pub.bin \
        --circuit $V/$C/circuit.txt --sessions $LAST > $O/oprove/$n/replay.txt 2>&1
      echo "$n replay exit $?"
    fi
    rm -rf $DD/v-sess/$l && cp -r $LAST $DD/v-sess/$l && echo $V/$C > $DD/v-sess/$l.stmt && echo $mode > $DD/v-sess/$l.mode
    return $rc
  }
  # oprove_gb: each statement's estimated peak GB, largest first. The fit to the CPU prover's peaks at K=64 N=64
  # (RecOpen 6.4 to 32.5, InnerRepCheck 22.3, RecCap 2.3) is linear in the circuit text's GB, 2^(k_log
  # - 20) and the instances; with a quarter's margin, at least 2
  mapfile -t TODO < <(python3 - $V <<'EOF'
import json, math, sys
from pathlib import Path
out = []
for d in Path(sys.argv[1]).rglob("inst.bin"):
    d = d.parent
    with open(d / "circuit.txt") as f:
        f.readline()
        meta = json.loads(f.readline()[5:])
    with open(d / "inst.bin", "rb") as f:
        n = json.loads(f.readline())["instances"]
    gb = 25.4 * (d / "circuit.txt").stat().st_size / 1e9 + 0.3 * 2 ** (meta["k_log"] - 20) + 0.06 * n - 7.5
    out.append((max(2, math.ceil(1.25 * gb)), str(d.relative_to(sys.argv[1]))))
for gb, c in sorted(out, key=lambda x: (-x[0], x[1])):
    print(gb, c)
EOF
)
  echo "oprove: ${#TODO[@]} statements, OPROVE_JOBS=${OPROVE_JOBS:-4} OPROVE_MEM_GB=${OPROVE_MEM_GB:-48}: ${TODO[*]}"
  declare -A RUNNING=(); USED=0
  for t in "${TODO[@]}"; do
    gb=${t%% *} C=${t#* }; l=${C//\//-}; want $l || continue
    while [ ${#RUNNING[@]} -gt 0 ] && { [ ${#RUNNING[@]} -ge ${OPROVE_JOBS:-4} ] || [ $((USED + gb)) -gt ${OPROVE_MEM_GB:-48} ]; }; do
      wait -n -p FIN; [ -n "${RUNNING[$FIN]:-}" ] && USED=$((USED - ${RUNNING[$FIN]%% *})) && echo "oprove ${RUNNING[$FIN]#* } done" && unset "RUNNING[$FIN]"
    done
    { oprove_one $C > $O/oprove/$l.log 2>&1; printf '%s\t%s\n' $l $? >> $O/oprove/exits.tsv; } &
    RUNNING[$!]="$gb $l"; USED=$((USED + gb)); echo "oprove $l started (${gb} GB est, ${USED} GB running)"
  done
  wait
  WARM=$W0 RUNS=$R0 NS=$((W0 + R0))
  (cd $V && sha512sum --quiet -c $O/oprove/statements.sha512) || { echo "a staged statement changed while sessions ran"; exit 1; }
  python3 - $O/oprove/exits.tsv "${TODO[@]#* }" <<'EOF' || exit 1
import sys
got = dict(l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if l.strip())
bad = [f"{l}: exit {got[l]}" for l in (c.replace("/", "-") for c in sys.argv[2:]) if got.get(l, "0") != "0" and not l.startswith("forged-")]
print(f"oprove: {len(got)} statements, one session each: exits {dict(sorted(got.items()))}")
sys.exit(f"oprove: an honest statement's session failed, and none is drawn again: {', '.join(bad)}" if bad else 0)
EOF
  # CONTROLS=DIR (a statement's directory under data/$TAG/v, e.g. L6): on that statement, one firewalled session per
  # FC_FIREWALL_TAMPER control (the device's side dishonest in one named way), each of which the firewall must refuse by name
  # without the tampered message leaving, the session stopping at its Finish (live PROTOCOL.md §10.1)
  if [ -n "${CONTROLS:-}" ]; then
    C=$V/$CONTROLS; hello_at $C
    for w in root witness label opening algebra algebra-proof shape; do
      PENV="FC_FIREWALL=1 FC_HOST_WITNESS=1 FC_FIREWALL_TAMPER=$w" WARM=0 RUNS=1 NS=1 again loopback neg-${CONTROLS//\//-}-$w $C/pub.bin $C/inst.bin $C/circuit.txt --zk
    done
  fi
  ;;
oreplay)
  # (CPU) upstream's replay --zk of each statement's session that oprove kept (v-sess), with the statement's sizes
  [ -d $DD/v-sess ] || { echo "run STEP=oprove first"; exit 1; }
  mkdir -p $O/oreplay
  for S in $(ls -d $DD/v-sess/*/ 2>/dev/null | sed 's:/$::' | sort -V); do l=$(basename $S); C=$(cat $S.stmt)
    want $l || continue
    mkdir -p $O/oreplay/$l && cp $S.mode $O/oreplay/$l/mode.txt 2>/dev/null
    /usr/bin/time -v -o $O/oreplay/$l/time-replay.txt $B replay --zk --instances $C/pub.bin --circuit $C/circuit.txt \
      --sessions $S > $O/oreplay/$l/replay.txt 2>&1
    echo "$l replay exit $?"
    du -sb $C/inst.bin $C/pub.bin $C/circuit.txt $S/*.bin > $O/oreplay/$l/sizes.txt
  done
  ;;
lean)
  [ -x $LEAN_BIN ] || lean_build > $O/build-lean.log 2>&1 || { tail -20 $O/build-lean.log; exit 1; }
  mkdir -p $O/lean
  # LEAN_JOBS verifications at once, the largest circuit first: one core each, at most 4.3 GB at K=64 N=64,
  # so by default one per 5 GB of LEAN_MEM_GB, at most the job's cores
  JOBS=${LEAN_JOBS:-$(( ${LEAN_MEM_GB:-48} / 5 < NC ? ${LEAN_MEM_GB:-48} / 5 : NC ))}
  # V*'s values against the verifier's own coins (verify --rec-inner-*, RecCoins.check): the inner statement and the inner
  # session STEP=vstage staged every statement from, a forgery's too, with the coin server's transcript of it, whose
  # root_B, later tops and link points every statement's tops and check 5 are held to (RecVStar.checkTopRows,
  # RecCoins.checkLink) and whose c_k its rec-c<k> rows are (RecVStar.checkPubRows)
  INNER=${INNER:-$DD/inner/s$((NS - 1))}
  rec_inner_files() { echo "--rec-inner-circuit $DD/stage/circuit.txt --rec-inner-public $DD/stage/pub-$N.bin --rec-inner-partition $DD/stage/partition.json --rec-inner-statement verity/flock-circuit+plain-leaves --rec-inner-statement-coins os --rec-inner-draw $DD/stage/draw.json --rec-inner-session $1/session"; }
  # REC_COINS=1: the inner statement set up once per inner session (flock-verify rec-coins, into its rec-coins.json), each
  # verify reading that record (--rec-inner-coins) in place of the inner statement's files, which at a served lane's size
  # every verify would otherwise parse again
  rec_coins() { [ "${REC_COINS:-0}" = 1 ] || return 0
    /usr/bin/time -v -o $O/lean/time-rec-coins-$2.txt $LEAN_BIN rec-coins $(rec_inner_files $1) > $1/rec-coins.json 2> $O/lean/rec-coins-$2.err \
      || { tail -5 $O/lean/rec-coins-$2.err; rm -f $1/rec-coins.json; return 1; }; }
  rec_inner_at() { if [ -s $1/rec-coins.json ]; then echo "--rec-inner-coins $1/rec-coins.json --rec-inner-session $1/session --rec-inner-transcript $1/public.json"
    else echo "$(rec_inner_files $1) --rec-inner-transcript $1/public.json"; fi; }
  rm -f $INNER/rec-coins.json; rec_coins $INNER inner || exit 1
  REC_INNER=$(rec_inner_at $INNER)
  NEGS=
  # a forged inner coin only the parts that read it can refuse (lean_verdicts.py's COVERS): without one among the parts
  # this step checks, the step refuses to run it rather than judge it
  VPARTS=$(ls -d $DD/v-sess/*/ 2>/dev/null | sed 's:/$::' | xargs -rn1 basename | grep -v '^forged-' | while read -r l; do want $l && echo $l; done)
  for w in ${NEG_INNER-challenge query}; do
    [ -z "$VPARTS" ] || python3 verity/ml/flock/pod/lean_verdicts.py --cover inner-$w $VPARTS \
      || { echo "lean: NEG_INNER=$w: include a part that reads it in LEVELS, or leave $w out of NEG_INNER"; exit 1; }
  done
  for w in ${NEG_INNER-challenge query}; do
    FI=$DD/inner-neg-$w; rm -rf $FI; mkdir -p $FI && cp -r $INNER/session $INNER/public.json $INNER/private.json $FI/ \
      && python3 verity/ml/flock/pod/rec_inner.py forge --what $w --inner $FI > $O/lean/forge-inner-$w.json || exit 1
    rec_coins $FI neg-$w || echo "rec-coins refused the forged inner $w: its verifies read the inner statement's files"
    NEGS="$NEGS $w"
  done
  # V*'s own inputs (RecVStar.checkStmt; without them every V* statement is refused at setup): the inner shape (rec-inner
  # of the inner statement), each statement's index in rec-statements' list and its bindings (rec_bindings of its
  # partition object and program), and the one shared rows file every part is verified against (rows.json,
  # rec_vstage.staged_shared_rows: the honest algebra's row ports but v and out, R's levels' tops, and each level's tops
  # rec-top-L<l> as L<l> and its cap read them; without the algebra, the first level's acc as acc0), whose digest
  # lean_verdicts.py's rec-compose holds every part's verdict to
  if [ -d $DD/v-sess ]; then
    $LEAN_BIN rec-inner --circuit $DD/stage/circuit.txt --public $DD/stage/pub-$N.bin --partition $DD/stage/partition.json \
      --program $DD/stage/program.json > $O/lean/rec-inner.json 2> $O/lean/rec-inner.err || { tail -5 $O/lean/rec-inner.err; exit 1; }
    $LEAN_BIN rec-statements --rec-inner $O/lean/rec-inner.json > $O/lean/rec-statements.txt 2>&1 || { tail -5 $O/lean/rec-statements.txt; exit 1; }
    python3 - $O/lean $DD/v $DD/v-sess <<'EOF' || exit 1
import json, sys
from pathlib import Path
from verity_flock import rec_bindings as RB
from verity_flock import rec_vstage as VS
o, v, vs = map(Path, sys.argv[1:])
(line,) = [json.loads(ln[9:]) for ln in (o / "rec-statements.txt").read_text().splitlines() if ln.startswith("RECVSTAR ")]
names = [s[0] for s in line["statements"]]
try:
    (o / "rows.json").write_text(json.dumps(VS.staged_shared_rows(v)))
except ValueError as e:
    sys.exit(str(e))
for s in sorted(p for p in vs.iterdir() if p.is_dir()):
    c = Path((vs / f"{s.name}.stmt").read_text().strip())
    with (c / "circuit.txt").open() as f:
        unit = f.readline().split()[1]
    flags = ["--rec-inner", str(o / "rec-inner.json"), "--rec-rows", str(o / "rows.json")]
    if unit in names:
        flags += ["--rec-part", str(names.index(unit))]
    try:
        b = RB.bindings(json.loads((c / "partition.json").read_text()), json.loads((c / "program.json").read_text()))
        (o / f"bindings-{s.name}.json").write_text(json.dumps(b))
        flags += ["--rec-bindings", str(o / f"bindings-{s.name}.json")]
    except ValueError as e:
        print(f"{s.name}: no bindings ({e}); Lean refuses it at setup")
    (o / f"own-{s.name}.txt").write_text(" ".join(flags))
    print(s.name, unit, " ".join(flags))
EOF
  fi
  for S in $(for s in $(ls -d $DD/v-sess/*/ 2>/dev/null | sed 's:/$::'); do echo "$(stat -c %s $(cat $s.stmt)/circuit.txt) $s"; done | sort -k1,1nr -k2V | cut -d' ' -f2); do l=$(basename $S); C=$(cat $S.stmt)
    want $l || continue
    OWN=$(cat $O/lean/own-$l.txt)
    echo "/usr/bin/time -v -o $O/lean/time-$l.txt $LEAN_BIN verify --zk --statement verity/flock-circuit --circuit $C/circuit.txt --public $C/pub.bin --partition $C/partition.json --program $C/program.json --registered $C/registered.json $OWN $REC_INNER --session $S > $O/lean/$l.log 2> $O/lean/$l.err; echo $l lean exit \$?"
    case $l in forged-*) continue ;; esac
    for w in $NEGS; do n=forged-inner-$w-$l
      echo "/usr/bin/time -v -o $O/lean/time-$n.txt $LEAN_BIN verify --zk --statement verity/flock-circuit --circuit $C/circuit.txt --public $C/pub.bin --partition $C/partition.json --program $C/program.json --registered $C/registered.json $OWN $(rec_inner_at $DD/inner-neg-$w) --session $S > $O/lean/$n.log 2> $O/lean/$n.err; echo $n lean exit \$?"
    done
  done | xargs -d '\n' -P ${JOBS:-4} -I{} sh -c "{}"
  if [ "${INNER_LEAN:-0}" = 1 ]; then
    /usr/bin/time -v -o $O/lean/time-inner.txt $LEAN_BIN verify --statement verity/flock-circuit+plain-leaves --circuit $DD/stage/circuit.txt \
      --public $DD/stage/pub-$N.bin --partition $DD/stage/partition.json --program $DD/stage/program.json --stratified $N --coins os \
      --session $DD/inner/s$((NS - 1))/session > $O/lean/inner.log 2> $O/lean/inner.err
    echo "inner lean exit $?"
  fi
  # ZK_LEAN=1: today's --zk on the same statement (the inner step's last loopback --zk session), for Lean's like-for-like
  if [ "${ZK_LEAN:-0}" = 1 ]; then
    ZS=$(ls -d $DD/inner-zk/sessions/l* | sort | tail -1)
    /usr/bin/time -v -o $O/lean/time-zk.txt $LEAN_BIN verify --zk --statement verity/flock-circuit --circuit $DD/stage-zk/circuit.txt \
      --public $DD/stage-zk/pub-$N.bin --partition $DD/stage/partition.json --program $DD/stage/program.json --session $ZS > $O/lean/zk.log 2> $O/lean/zk.err
    echo "zk lean exit $?"
  fi
  # the inner statement with the verifier's partition object and program (the setup AcceptsPublic states): Lean derives its
  # units from the program, each an activation of the circuit's template
  if [ -f $DD/stage/circuit.txt ]; then
    /usr/bin/time -v -o $O/lean/time-inner-statement.txt $LEAN_BIN statement --statement verity/flock-circuit+plain-leaves \
      --circuit $DD/stage/circuit.txt --public $DD/stage/pub-$N.bin --partition $DD/stage/partition.json \
      --program $DD/stage/program.json --coins os > $O/lean/inner-statement.log 2> $O/lean/inner-statement.err
    echo "inner-statement lean exit $?"
  fi
  # fail closed: every statement vstage staged or oprove kept a session of, ZK_LEAN's, and the inner statement's
  PARTS=$( { [ -d $DD/v ] && (cd $DD/v && find . -name inst.bin -printf '%h\n' | sed 's:^\./::; s:/:-:g')
    for S in $(ls -d $DD/v-sess/*/ 2>/dev/null | sed 's:/$::'); do basename $S; done; } | sort -uV | while read -r l; do want $l && echo $l; done)
  for l in $(ls -d $DD/v-sess/*/ 2>/dev/null | sed 's:/$::' | xargs -rn1 basename | sort -V); do
    case $l in forged-*) continue ;; esac
    want $l && for w in $NEGS; do PARTS="$PARTS forged-inner-$w-$l"; done
  done
  [ "${ZK_LEAN:-0}" = 1 ] && PARTS="$PARTS zk"
  [ "${INNER_LEAN:-0}" = 1 ] && PARTS="$PARTS inner"
  [ -f $DD/stage/circuit.txt ] && PARTS="$PARTS inner-statement"
  if [ "${LEVELS:-}" = none ] && [ -z "$PARTS" ]; then echo "lean: LEVELS=none and no ZK_LEAN: no part to check"
  else cp $DD/v-sess/*.live $O/lean/ 2>/dev/null
    python3 verity/ml/flock/pod/lean_verdicts.py $([ -s $O/lean/rows.json ] && echo "--compose $LEAN_BIN") $O/lean $PARTS; LEAN_RC=$?; fi
  ;;
esac
summary
exit ${LEAN_RC:-0}
