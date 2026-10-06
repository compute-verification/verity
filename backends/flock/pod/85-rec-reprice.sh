#!/usr/bin/env bash
# rec-reprice: recursive witness ZK priced at m = 35 on node 1's GPU prover, against today's --zk on the same statement (the
# dense rollup's K = 4096 GEMM coordinate, r20261002-062836-96bb's shape 430e5aad). One step per job (research run --queue
# --cwd source); the big files stay under $FLOCK_WORK/data/$TAG for the next step, out/ gets the records:
#   STEP=build   (CPU) the GPU prover (60-circuit.sh's build, keyed by its sources as 70-class-sweep.sh keys it), Lean's
#                flock-verify (rec-env.sh's, keyed by its sources), and the inner statement staged (class_statement
#                --definitions DEFS --stage-only, BATCH=fixed N) twice: with plain SHA-512 leaves (FLOCK_PLAIN_LEAVES=1, the
#                recursive session's) into data/$TAG/stage, and with hm96 leaves (today's --zk) into data/$TAG/stage-zk
#   STEP=inner   (GPU leases) the plain inner statement proved with ZK off (M0, FC_COINS=os) against the rec proxy
#                (verity_flock.rec_live, WARM + RUNS sessions), upstream's replay of the last one and V*'s hashing reference
#                (rec_vstar) over it, and over two forged copies of its private.json (a round's top: one node one bit off, one
#                salt one bit off), which it must refuse; then (INNER_LOOPBACK=1) the hm96 statement on loopback, M0 and --zk,
#                WARM + RUNS sessions each (today's numbers)
#   STEP=vstage  (CPU) V*'s statements over the inner's last proxied session (verity_flock.rec_vstage) into data/$TAG/v: the
#                verifier's fold points (rec_vstage points), its extra claims and comb (InnerFold.lean as an executable,
#                cached by its sources, on the plain statement: --plain-leaves), then each level's RecOpen, the algebra's
#                InnerRepCheck and the forgeries (FORGE="message sum comb top top-salt coef dirs", each under forged-<kind>/),
#                JOBS at once
#   STEP=oprove  (GPU leases) each staged statement proved with --zk on loopback, WARM + RUNS sessions (a forgery's: one, by a
#                prover that sends its proofs though the witness fails, FC_ZK_SELF_CHECK=skip), and upstream's replay --zk
#                of the last
#   STEP=lean    (CPU) Lean's flock-verify verify --zk on each statement's last session (INNER_LEAN=1: also the inner's,
#                --coins os, which Lean's tags refuse while they pin hm96 leaves; ZK_LEAN=1: also today's --zk session;
#                LEVELS=none: neither of V*'s)
# The prover runs on the job's first PCPUS cores, the verifier (proxy, serve, replay) on the rest. Every step writes
# out/summary-$STEP.json. Only a prove invocation holds a GPU, leased for it alone (on_gpu), so every STEP is launched without
# --gpus (a CPU phase): staging, serve, replay and Lean never hold one.
# env: FLOCK_WORK=/workspace/jobs/rec-reprice  TAG=k4096  K=4096 (the Gemm_v2 contraction: DEFS is Gemm_v2{K,N=4096})  N=2048
#      CUDA_TK= (default: CUDA_REDIST=13.3.1's archives under FLOCK_WORK)  SM=120  WARM=1  RUNS=3  PCPUS=16
#      LEVELS= (comma-separated statement names, default every one: L3,alg; oprove and lean also forged-sum-L6, alg-p1,
#              ..; a name without its part, alg, names every part)
#      FORGE="message sum comb top top-salt coef dirs" (vstage)  JOBS=4 (vstage, lean)  INNER_LOOPBACK=1 (inner)
#      GPU_MAX_MIN=10 GPU_MEM_GB=32 (each prove's lease)  GPU_PREEMPTIBLE=1 (empty: not preemptible)  GPU_TRIES=3
#      GPU_LEASE= (the lease command; default the queue's for this node)
#      WARM_FROM=/workspace/jobs/proofs-vllm-moe/flock-gpu-link (a checkout whose cargo target starts the build warm)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
STEP=${STEP:?STEP=build|inner|vstage|oprove|lean}
[ -z "${GPU_LEASE_UUID:-}" ] || { echo "REFUSED: the job holds a GPU lease (launched with --gpus): launch it without, each prove leases its own"; exit 2; }
REPO=$(pwd); O=${RESEARCH_RUN_DIR:-/tmp/rec-reprice-job}/out; mkdir -p $O
FW=${FLOCK_WORK:-/workspace/jobs/rec-reprice}; W=$FW/flock-circuit; L=$FW/cache; TAG=${TAG:-k4096}; DD=$FW/data/$TAG
mkdir -p $W $L $DD || exit 1
[ -w "$HOME" ] || { export HOME=$FW/home; mkdir -p $HOME; }
unset CARGO_TARGET_DIR
# the prover's kernels use clmad, which CUDA 13.0's ptxas (node 1's host toolkit) doesn't know: without CUDA_TK, NVIDIA's
# redistributable CUDA_REDIST archives (sha256-checked) are unpacked once under $FW, and their runtime is the prover's
CUDA_REDIST=${CUDA_REDIST:-13.3.1}; export CUDA_TK=${CUDA_TK:-$FW/cuda-$CUDA_REDIST}
if [ ! -x $CUDA_TK/bin/nvcc ]; then
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

KEY=$(cat backends/flock/pod/20-gpu-link.sh backends/flock/pod/60-circuit.sh backends/flock/*.patch backends/flock/cuda_*_patch.py \
  $(find backends/flock/cuda backends/flock/live -type f | LC_ALL=C sort) | sha256sum | cut -c1-16)
BIN=$W/bin-$KEY-g1-sm$SM; B=$BIN/flock-circuit; F=$FW/flock-gpu-link/flock
LSHA=$(cd backends/flock/verifier/lean && find . -name .lake -prune -o -type f -print | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)
LEAN_BIN=$L/lean-$LSHA/flock-verify
UV=$(command -v uv || echo $HOME/.local/bin/uv)
[ -x $UV ] || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; UV=$HOME/.local/bin/uv; }
[ -x $W/py/bin/python ] || { $UV venv -q --python 3.12 $W/py && $UV pip install -q --python $W/py/bin/python numpy blake3; } || exit 1
export PATH=$W/py/bin:$PATH PYTHONPATH=$REPO:$REPO/catalog:$REPO/kernels:$REPO/experimental:$REPO/benchmarks/numerical/python:$REPO/backends/flock/python:$REPO/integrations/vllm:$REPO/tools/circuit_check/src
export FC_COINS=os FC_PIPELINE_DEPTH=${FC_PIPELINE_DEPTH:-4} FC_HOST_PREPIN=${FC_HOST_PREPIN:-1}
# the lease command: GPU_LEASE, else the queue's own for this node (tools/cluster's submit.wrappers: node 1's pool where its tool
# and pool file are, else gpu-lease's default)
if [ -n "${GPU_LEASE:-}" ]; then read -ra LEASE <<< "$GPU_LEASE"; else mapfile -t LEASE < <(python3 - <<'EOF'
import os, sys
sys.path.insert(0, "tools/cluster/src")
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
  gpu_mib $d & local s=$!
  "${LEASE[@]}" 1 --wait --max-min ${GPU_MAX_MIN:-10} --mem-gb ${GPU_MEM_GB:-32} ${pre:+--preemptible} -- "$@" 2> $d/prove.err
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
    echo "$1 $2: lease preempted, try $t" | tee -a $O/preempted.txt
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
  on_gpu $d env FC_VERIFY_AHEAD=$NS FC_ZK_SELF_CHECK=${SELF_CHECK:-stop} taskset -c $PCPU /usr/bin/time -v -o $d/time-prove.txt \
    $B prove $zk --gpu --verifier $addr --instances $inst --circuit $circ --warm $WARM --runs $RUNS | stamp > $d/prove.tsv
  local rc=${PIPESTATUS[0]}
  grep -h panicked $d/prove.err | head -3
  return $rc
}

# loopback NAME PUB INST CIRCUIT [--zk]: a verifier process (serve, on the verifier's cores, every session at once) and
# prove_live against it; the sessions under $DD/$STEP-NAME/sessions; the prover's status
loopback() {
  local name=$1 pub=$2 inst=$3 circ=$4 zk=${5:-} s=$DD/$STEP-$1 PORT=$(free_port) PIN=$(sha512sum $4 | cut -d' ' -f1)
  rm -rf $s; mkdir -p $s $O/$STEP/$name
  taskset -c $VCPU $B serve $zk --listen 127.0.0.1:$PORT --out $s/sessions --instances $pub --circuit $circ --pin $PIN \
    --operator loopback --concurrency $((NS + 1)) > $O/$STEP/$name/serve.log 2>&1 &
  local SP=$!
  wait_serving $O/$STEP/$name/serve.log $SP || { tail -20 $O/$STEP/$name/serve.log; return 1; }
  prove_live $name 127.0.0.1:$PORT $inst $circ $zk; local rc=$?
  kill $SP; wait $SP 2>/dev/null
  LAST=$(find $s/sessions -name session.json -printf '%T@ %h\n' | sort -n | tail -1 | cut -d' ' -f2)
  return $rc
}

# proxied: the rec proxy (WARM + RUNS sessions, each to $DD/inner/s<k>) and prove_live against it; the prover's status
proxied() {
  rm -rf $DD/inner
  taskset -c $VCPU python3 -m verity_flock.rec_live --listen 127.0.0.1:0 --out $DD/inner --sessions $NS > $O/inner/proxy/proxy.log 2>&1 &
  local PP=$!
  wait_serving $O/inner/proxy/proxy.log $PP || { cat $O/inner/proxy/proxy.log; return 1; }
  prove_live proxy $(awk '/^SERVING/ {print $2}' $O/inner/proxy/proxy.log) $D/inst-$N.bin $D/circuit.txt; local rc=$?
  [ $rc = 0 ] || kill $PP 2>/dev/null
  wait $PP; echo "proxy exit $?"
  return $rc
}

summary() {
  python3 - $O $STEP "$WARM" "$RUNS" <<'EOF'
import json, re, statistics, sys
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
    return {"sessions": len(rows), "timed": len(timed), "accepted": [v.get("accepted") for _, v in rows],
            "why": [v.get("why") for _, v in rows if v.get("accepted") is not True],
            "serve_why": [v.get("why") for v in served if v.get("accepted") is not True],
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
            "buckets": [{k: x for k, x in b.items() if k.startswith("t.") or k in ("device_witness", "native_sha")} for b in (timed[-1][1].get("buckets") or [])]
            if timed else None}

def replay(p):
    up = next((json.loads(l.split("\t", 1)[1]) for l in text(p).splitlines() if l.startswith("UPSTREAM")), None)
    return up if up else text(p)[-600:]

def lean(p):
    at = text(p).find("VERDICT ")
    if at < 0:
        return text(p)[-600:]
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
elif step == "vstage":
    pts = json.loads(text(d / "points.json")) if text(d / "points.json").strip() else {}
    fold = json.loads(text(d / "fold.json")) if text(d / "fold.json").strip() else {}
    s["points"] = {"m": pts.get("m"), "k_log": pts.get("k_log"), "points_s": pts.get("points_s"), "err": text(d / "points.err")[-600:],
                   "wall": tv(d / "time-points.txt", r"Elapsed \(wall clock\) time"), "peak_gb": gb(d / "time-points.txt")}
    s["fold"] = {"m": fold.get("m"), "k_log": fold.get("k_log"), "extras": len(fold.get("extras", [])), "err": text(d / "fold.err")[-600:],
                 "wall": tv(d / "time-fold.txt", r"Elapsed \(wall clock\) time"), "peak_gb": gb(d / "time-fold.txt")}
    s["statements"] = {}
    for f in sorted(d.glob("*.json")):
        if f.stem in ("points", "fold", "partition"):
            continue
        s["statements"][f.stem] = {"stage": json.loads(text(f).splitlines()[0]) if text(f).strip() else text(d / f"{f.stem}.err")[-1500:],
                                   "wall": tv(d / f"time-{f.stem}.txt", r"Elapsed \(wall clock\) time"), "peak_gb": gb(d / f"time-{f.stem}.txt")}
    s["sizes"] = text(d / "sizes.txt")
elif step == "oprove":
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
  (
    export PATH=$HOME/.elan/bin:$PATH
    command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
    exec 9>$L/lean.lock; flock 9
    if [ ! -x $LEAN_BIN ]; then
      rm -rf $L/lean-src && mkdir -p $L/lean-src && cp -r backends/flock/verifier/lean/. $L/lean-src/ \
        && (cd $L/lean-src && lake build flock-verify) && mkdir -p $L/lean-$LSHA && cp $L/lean-src/.lake/build/bin/flock-verify $LEAN_BIN
    fi
  )
}

# stage_inner NAME PLAIN: the inner statement (one Call of Gemm_v2{K, N=4096}, its GemmCoordinate_v2 shape at N instances)
# staged into $DD/NAME from its cache $DD/NAME-cache, its trees' leaves plain SHA-512 when PLAIN is 1; its records in $O/NAME
stage_inner() {
  local o=$O/$1 c=$DD/$1-cache
  mkdir -p $o
  python3 -c "import json, sys; json.dump({'Gemm_v2{K=%s,N=4096,DOT={\"fn\":\"HopperBF16WgmmaDot16_v1\"}}' % sys.argv[1]: 1151}, open(sys.argv[2], 'w'))" $K $o/defs.json
  FLOCK_STAGE_CACHE=$c FLOCK_PLAIN_LEAVES=$2 FLOCK_GEMM_TILE=${FLOCK_GEMM_TILE:-4x4} /usr/bin/time -v -o $o/time-stage.txt \
    python3 -m verity_flock.class_statement --definitions $o/defs.json --partition q-word --out $o --stage-only --batch fixed --n $N \
    --max-statement-bits $((1 << 35)) --batch-ands $((1 << 35)) --selftest 0 > $o/stage.log 2>&1
  echo "stage $1 exit $?"; tail -5 $o/stage.log
  local C=$(ls -td $c/*/ 2>/dev/null | head -1)
  [ -n "$C" ] && [ -f $C/inst-$N.bin ] || { echo "no staged statement under $c"; return 1; }
  mkdir -p $DD/$1 && for f in circuit.txt inst-$N.bin pub-$N.bin rec.json; do ln -f $C/$f $DD/$1/$f || cp $C/$f $DD/$1/$f; done
  du -sb $DD/$1/* | tee $o/sizes.txt
}

case $STEP in
build)
  lean_build > $O/build-lean.log 2>&1 &
  LP=$!
  { stage_inner stage 1 && stage_inner stage-zk 0; } > $O/stage-all.log 2>&1 &
  SPID=$!
  if [ ! -x $B ]; then
    exec 8>$W/build.lock; flock 8
    if [ ! -x $B ]; then
      [ -d $F ] || { [ -d ${WARM_FROM:-/workspace/jobs/proofs-vllm-moe/flock-gpu-link}/flock ] && cp -a ${WARM_FROM:-/workspace/jobs/proofs-vllm-moe/flock-gpu-link} $FW/flock-gpu-link; }
      RESEARCH_RUN_DIR=$O/build FLOCK_WORK=$FW MODE=build GPU=1 SM=$SM bash backends/flock/pod/60-circuit.sh > $O/build.log 2>&1 \
        || { tail -40 $O/build.log; tail -40 $O/build/out/build-circuit.txt 2>/dev/null; exit 1; }
      mkdir -p $BIN && cp $F/target/release/flock-circuit $B
    fi
    exec 8>&-
  fi
  sha256sum $B | tee $O/binary.sha256
  wait $SPID; rc=$?; cat $O/stage-all.log; [ $rc = 0 ] || exit 1
  $B statement --instances $DD/stage/inst-$N.bin --circuit $DD/stage/circuit.txt > $O/statement.txt 2>&1; echo "statement exit $?"
  wait $LP; echo "lean build exit $?"
  [ -x $LEAN_BIN ] && sha256sum $LEAN_BIN > $O/lean.sha
  tail -5 $O/build-lean.log
  ;;
inner)
  D=$DD/stage; [ -f $D/inst-$N.bin ] && [ -x $B ] || { echo "run STEP=build first"; exit 1; }
  mkdir -p $O/inner/proxy
  # INNER_PROXY=0: only today's numbers again (the proxied sessions V*'s statements were staged from stay as they are)
  if [ "${INNER_PROXY:-1}" = 1 ]; then
  # M0 against the rec proxy: WARM + RUNS sessions, each written to $DD/inner/s<k>
  again proxied
  SESS=$DD/inner/s$((NS - 1))
  [ -f $SESS/session/session.json ] || { tail -20 $O/inner/proxy/proxy.log $O/inner/proxy/prove.err; exit 1; }
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
  mkdir -p $O/inner/session && cp -r $SESS/session $SESS/public.json $O/inner/session/ && du -sb $SESS/* > $O/inner/session/sizes.txt
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
vstage)
  # V*'s statements over the inner's last proxied session: the verifier's fold points (from the coins), its extra claims and
  # comb (InnerFold.lean as an executable, cached by its sources as STEP=build caches flock-verify; no proof read), then the
  # statements, each in its own process: first the verifier's values alone (its native seconds, and the running sums' salts
  # drawn once into $V/acc-salts.bin, the prover's, so that every statement reads one chain), then the algebra, the forgeries
  # and every level, JOBS at once
  INNER=${INNER:-$DD/inner/s$((NS - 1))}
  [ -f $INNER/public.json ] && [ -f $INNER/private.json ] || { echo "INNER=$INNER: no public.json and private.json (run STEP=inner first)"; exit 1; }
  FSHA=$( (echo $LSHA; cat backends/flock/tests/lean/InnerFold.lean) | sha256sum | cut -c1-16)
  FOLD_BIN=$L/fold-$FSHA/inner-fold
  (
    export PATH=$HOME/.elan/bin:$PATH
    command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
    exec 9>$L/fold.lock; flock 9
    if [ ! -x $FOLD_BIN ]; then
      C=$L/fold-src; rm -rf $C && mkdir -p $C && cp -r backends/flock/verifier/lean/. $C/ && cp backends/flock/tests/lean/InnerFold.lean $C/ \
        && printf '\n[[lean_exe]]\nname = "inner-fold"\nroot = "InnerFold"\n' >> $C/lakefile.toml \
        && (cd $C && lake build inner-fold) && mkdir -p $L/fold-$FSHA && cp $C/.lake/build/bin/inner-fold $FOLD_BIN
    fi
  ) > $O/build-fold.log 2>&1 || { tail -30 $O/build-fold.log; exit 1; }
  mkdir -p $O/vstage
  # the verifier's copy of the partition object the inner circuit binds: q-word over the build step's one-Call program
  python3 - $K $DD/stage/partition.json $DD/stage/circuit.txt <<'EOF' || exit 1
import json, sys
from verity.primitives.circuits import partition_object as PO
from verity_flock import class_statement as CS, partition_units as PU
defs = {'Gemm_v2{K=%s,N=4096,DOT={"fn":"HopperBF16WgmmaDot16_v1"}}' % sys.argv[1]: 1151}
program, _calls, _prov = CS.distinct_calls(None, None, defs, None)
units = PU.partition_units(program, "q-word")
obj = PO.build(units.program, units.query)
with open(sys.argv[3]) as f:
    f.readline()
    meta = json.loads(f.readline()[5:])
assert (meta["partition"]["digest"], meta["program_sha512"]) == (PO.digest(obj), obj["program"]), "not the circuit's partition"
open(sys.argv[2], "w").write(json.dumps(obj, sort_keys=True))
print("partition", PO.digest(obj), "program", obj["program"])
EOF
  cp $DD/stage/partition.json $O/vstage/
  V=$DD/v; rm -rf $V; mkdir -p $V
  /usr/bin/time -v -o $O/vstage/time-points.txt python3 -m verity_flock.rec_vstage points $INNER $DD/stage/circuit.txt \
    $DD/stage/pub-$N.bin > $V/points.json 2> $O/vstage/points.err || { tail -20 $O/vstage/points.err; exit 1; }
  /usr/bin/time -v -o $O/vstage/time-fold.txt $FOLD_BIN --plain-leaves --partition $DD/stage/partition.json verity/flock-circuit \
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
    echo "$name exit $?"
  }
  export -f vjob; export O V INNER
  vjob verifier --levels none --no-algebra
  [ -f $V/acc-salts.bin ] || { tail -20 $O/vstage/verifier.err; exit 1; }
  { want alg && echo alg --levels none
    for f in ${FORGE-message sum comb top top-salt coef dirs}; do echo forged-$f --forge $f; done
    for l in $ALL; do want L$l && echo L$l --levels $l --no-algebra; done; } | xargs -P ${JOBS:-4} -L 1 bash -c 'vjob "$@"' _
  (cd $V && du -sb $(find . -name '*.bin' -o -name circuit.txt | LC_ALL=C sort)) > $O/vstage/sizes.txt
  ;;
oprove)
  V=$DD/v; [ -f $V/acc-salts.bin ] || { echo "run STEP=vstage first"; exit 1; }
  # every staged statement (each level's, each of the algebra's parts, each forgery's: L3, alg-p0, forged-sum-L6, ..), proved with --zk on
  # loopback, a forgery's once; upstream's replay --zk of the last session, which (accepted or not) is kept under $DD/v-sess
  # beside the statement directory Lean reads it with
  mkdir -p $DD/v-sess
  W0=$WARM R0=$RUNS
  for C in $(cd $V && find . -name inst.bin -printf '%h\n' | sed 's:^\./::' | sort -V); do
    l=${C//\//-}; want $l || continue
    case $l in forged-*) WARM=0 RUNS=1 SELF_CHECK=skip ;; *) WARM=$W0 RUNS=$R0 SELF_CHECK=stop ;; esac; NS=$((WARM + RUNS)); LAST=
    again loopback $l $V/$C/pub.bin $V/$C/inst.bin $V/$C/circuit.txt --zk
    du -sb $V/$C/inst.bin $V/$C/pub.bin $V/$C/circuit.txt ${LAST:+$LAST/*.bin} > $O/oprove/$l/sizes.txt
    [ -n "${LAST:-}" ] || { echo "$l: no session"; continue; }
    taskset -c $VCPU /usr/bin/time -v -o $O/oprove/$l/time-replay.txt $B replay --zk --instances $V/$C/pub.bin \
      --circuit $V/$C/circuit.txt --sessions $LAST > $O/oprove/$l/replay.txt 2>&1
    echo "$l replay exit $?"
    rm -rf $DD/v-sess/$l && cp -r $LAST $DD/v-sess/$l && echo $V/$C > $DD/v-sess/$l.stmt
  done
  WARM=$W0 RUNS=$R0 NS=$((W0 + R0))
  ;;
lean)
  [ -x $LEAN_BIN ] || lean_build > $O/build-lean.log 2>&1 || { tail -20 $O/build-lean.log; exit 1; }
  mkdir -p $O/lean
  for S in $(ls -d $DD/v-sess/*/ 2>/dev/null | sed 's:/$::' | sort -V); do l=$(basename $S); C=$(cat $S.stmt)
    want $l || continue
    echo "/usr/bin/time -v -o $O/lean/time-$l.txt $LEAN_BIN verify --zk --statement verity/flock-circuit --circuit $C/circuit.txt --public $C/pub.bin --partition $C/partition.json --program $C/program.json --registered $C/registered.json --session $S > $O/lean/$l.log 2> $O/lean/$l.err; echo $l lean exit \$?"
  done | xargs -d '\n' -P ${JOBS:-4} -I{} sh -c "{}"
  if [ "${INNER_LEAN:-0}" = 1 ]; then
    /usr/bin/time -v -o $O/lean/time-inner.txt $LEAN_BIN verify --statement verity/flock-circuit --circuit $DD/stage/circuit.txt \
      --public $DD/stage/pub-$N.bin --partition $DD/stage/partition.json --coins os --session $DD/inner/s$((NS - 1))/session > $O/lean/inner.log 2> $O/lean/inner.err
    echo "inner lean exit $?"
  fi
  # ZK_LEAN=1: today's --zk on the same statement (the inner step's last loopback --zk session), for Lean's like-for-like
  if [ "${ZK_LEAN:-0}" = 1 ]; then
    ZS=$(ls -d $DD/inner-zk/sessions/l* | sort | tail -1)
    /usr/bin/time -v -o $O/lean/time-zk.txt $LEAN_BIN verify --zk --statement verity/flock-circuit --circuit $DD/stage-zk/circuit.txt \
      --public $DD/stage-zk/pub-$N.bin --partition $DD/stage/partition.json --session $ZS > $O/lean/zk.log 2> $O/lean/zk.err
    echo "zk lean exit $?"
  fi
  ;;
esac
summary
true
