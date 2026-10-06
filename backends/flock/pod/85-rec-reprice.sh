#!/usr/bin/env bash
# rec-reprice: recursive witness ZK priced at m = 35 on node 1's GPU prover, against today's --zk on the same statement (the
# dense rollup's K = 4096 GEMM coordinate, r20261002-062836-96bb's shape 430e5aad). One step per job (research run --queue
# --cwd source); the big files stay under $FLOCK_WORK/data/$TAG for the next step, out/ gets the records:
#   STEP=build   (CPU) the GPU prover (60-circuit.sh's build, keyed by its sources as 70-class-sweep.sh keys it), Lean's
#                flock-verify (rec-env.sh's, keyed by its sources), and the inner statement staged (class_statement
#                --definitions DEFS --stage-only, BATCH=fixed N) into data/$TAG/stage
#   STEP=inner   (GPU) the inner statement proved with ZK off (M0, FC_COINS=os) against the rec proxy (verity_flock.rec_live,
#                WARM + RUNS sessions), upstream's replay of the last one and V*'s hashing reference (rec_vstar) over it; then
#                the same statement on loopback, M0 and --zk, WARM + RUNS sessions each (today's numbers, this node and build)
#   STEP=ostage  (CPU) V*'s RecOpen statements, one per Ligerito level of the inner's last session (rec_outer --link-rows)
#   STEP=alg     (CPU) V*'s algebra: InnerClaims.lean's claims on the inner's last session, and rec_residuals' InnerRepCheck
#                on both reps staged into data/$TAG/alg/honest
#   STEP=oprove  (GPU) each level (and the algebra, once staged) proved with --zk on loopback, WARM + RUNS sessions;
#                upstream's replay --zk of the last
#   STEP=lean    (CPU) Lean's flock-verify verify --zk on each statement's last session (INNER_LEAN=1: also the inner's,
#                --coins os; ZK_LEAN=1: also today's --zk session; LEVELS=none: neither of V*'s)
# The prover runs on the job's first PCPUS cores, the verifier (proxy, serve, replay) on the rest. Every step writes
# out/summary-$STEP.json.
# env: FLOCK_WORK=/workspace/jobs/rec-reprice  TAG=k4096  K=4096 (the Gemm_v2 contraction: DEFS is Gemm_v2{K,N=4096})  N=2048
#      CUDA_TK= (default: CUDA_REDIST=13.3.1's archives under FLOCK_WORK)  SM=120  WARM=1  RUNS=3  PCPUS=16
#      LEVELS= (default every level)  JOBS=4 (ostage, lean)
#      WARM_FROM=/workspace/jobs/proofs-vllm-moe/flock-gpu-link (a checkout whose cargo target starts the build warm)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
STEP=${STEP:?STEP=build|inner|ostage|oprove|lean}
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
GI=${CUDA_VISIBLE_DEVICES:-0}; GI=${GI%%,*}

free_port() { python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
# stamp: each line of stdin with the time it arrived (a session's LIVE line is printed when its verdict is in)
stamp() { while IFS= read -r l; do printf '%s\t%s\n' "$(date +%s.%N)" "$l"; done; }
# gpu_sample FILE: the job's GPU's memory in use (MiB), every half second, until FILE.stop exists
gpu_sample() { rm -f $1.stop; while [ ! -e $1.stop ]; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GI 2>/dev/null; sleep 0.5; done > $1; }
wait_serving() { for i in $(seq 3600); do grep -q SERVING $1 && return 0; kill -0 $2 2>/dev/null || return 1; sleep 1; done; return 1; }

# prove_live NAME VERIFIER_ADDR INST CIRCUIT [--zk]: WARM + RUNS sessions of the GPU prover on the prover's cores, into
# $O/$STEP/NAME (prove.tsv: stamped stdout, time-prove.txt, gpu-mib.txt)
prove_live() {
  local d=$O/$STEP/$1 addr=$2 inst=$3 circ=$4 zk=${5:-}
  mkdir -p $d
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
  gpu_sample $d/gpu-mib.txt & local GS=$!
  FC_VERIFY_AHEAD=$NS taskset -c $PCPU /usr/bin/time -v -o $d/time-prove.txt $B prove $zk --gpu --verifier $addr --instances $inst \
    --circuit $circ --warm $WARM --runs $RUNS 2> $d/prove.err | stamp > $d/prove.tsv
  touch $d/gpu-mib.txt.stop; wait $GS
  grep -h panicked $d/prove.err | head -3
}

# loopback NAME PUB INST CIRCUIT [--zk]: a verifier process (serve, on the verifier's cores, every session at once) and
# prove_live against it; the sessions under $DD/$STEP-NAME/sessions
loopback() {
  local name=$1 pub=$2 inst=$3 circ=$4 zk=${5:-} s=$DD/$STEP-$1 PORT=$(free_port) PIN=$(sha512sum $4 | cut -d' ' -f1)
  rm -rf $s; mkdir -p $s $O/$STEP/$name
  taskset -c $VCPU $B serve $zk --listen 127.0.0.1:$PORT --out $s/sessions --instances $pub --circuit $circ --pin $PIN \
    --operator loopback --concurrency $((NS + 1)) > $O/$STEP/$name/serve.log 2>&1 &
  local SP=$!
  wait_serving $O/$STEP/$name/serve.log $SP || { tail -20 $O/$STEP/$name/serve.log; return 1; }
  prove_live $name 127.0.0.1:$PORT $inst $circ $zk
  kill $SP; wait $SP 2>/dev/null
  LAST=$(find $s/sessions -name session.json -printf '%T@ %h\n' | sort -n | tail -1 | cut -d' ' -f2)
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
    if not rows:
        return {"error": text(d / "prove.err")[-1500:]}
    timed = [(t, v) for t, v in rows if not v.get("warm")]
    warmed = [(t, v) for t, v in rows if v.get("warm")]
    med = lambda f: statistics.median([f(v) for _, v in timed]) if timed else None
    t0 = warmed[-1][0] if warmed else None
    mib = [int(x) for x in text(d / "gpu-mib.txt").split() if x.strip().isdigit()]
    # the verifier's own seconds per session (serve's SESSION records, by session name), and --zk's mask-rank check
    # (ZKRANK, one per rep, in order), each summed per session; both over the timed sessions only
    served = sorted((json.loads(l.split("\t", 1)[1]) for l in text(d / "serve.log").splitlines() if l.startswith("SESSION\t")),
                    key=lambda v: v.get("session", ""))
    ranks = [json.loads(l.split("\t", 1)[1])["s"] for l in text(d / "prove.err").splitlines() if l.startswith("ZKRANK\t")]
    per = len(ranks) // len(rows) if ranks and len(ranks) % len(rows) == 0 else 0
    rank_s = [sum(ranks[i * per:(i + 1) * per]) for i in range(len(rows))][warm:] if per else []
    return {"sessions": len(rows), "timed": len(timed), "accepted": [v.get("accepted") for _, v in rows],
            "prover_cores_busy_before": float(text(d / "bgload.txt").strip() or "nan"),
            "serve_verify_s": statistics.median([v["verify_s"] for v in served[warm:]]) if len(served) > warm else None,
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
    s["staged"] = [json.loads(l) for l in text(o / "stage" / "staged.jsonl").splitlines() if l.strip()]
    s["stage_wall"], s["stage_peak_gb"] = tv(o / "time-stage.txt", r"Elapsed \(wall clock\) time"), gb(o / "time-stage.txt")
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
elif step == "ostage":
    s["levels"] = {}
    for f in sorted(d.glob("L*.json")):
        lv = f.stem
        s["levels"][lv] = {"stage": json.loads(text(f).splitlines()[0]) if text(f).strip() else text(d / f"{lv}.err")[-1500:],
                           "wall": tv(d / f"time-{lv}.txt", r"Elapsed \(wall clock\) time"), "peak_gb": gb(d / f"time-{lv}.txt")}
elif step == "alg":
    s["claims"] = (lambda c: {"m": c.get("m"), "k_log": c.get("k_log"), "extras": [len(r["extras"]) for r in c.get("reps", [])]})(
        json.loads(text(d / "claims.json"))) if text(d / "claims.json").strip() else text(d / "claims.err")[-1500:]
    s["claims_wall"], s["claims_peak_gb"] = tv(d / "time-claims.txt", r"Elapsed \(wall clock\) time"), gb(d / "time-claims.txt")
    s["stage"] = json.loads(text(d / "stage.json").splitlines()[0]) if text(d / "stage.json").strip() else text(d / "stage.err")[-1500:]
    s["stage_wall"], s["stage_peak_gb"] = tv(d / "time-stage.txt", r"Elapsed \(wall clock\) time"), gb(d / "time-stage.txt")
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

case $STEP in
build)
  lean_build > $O/build-lean.log 2>&1 &
  LP=$!
  # the inner statement: one Call of Gemm_v2{K, N=4096}, its GemmCoordinate_v2 shape staged at N instances
  mkdir -p $O/stage
  python3 -c "import json, sys; json.dump({'Gemm_v2{K=%s,N=4096,DOT={\"fn\":\"HopperBF16WgmmaDot16_v1\"}}' % sys.argv[1]: 1151}, open(sys.argv[2], 'w'))" $K $O/stage/defs.json
  export FLOCK_STAGE_CACHE=$DD/stage-cache FLOCK_GEMM_TILE=${FLOCK_GEMM_TILE:-4x4}
  /usr/bin/time -v -o $O/time-stage.txt python3 -m verity_flock.class_statement --definitions $O/stage/defs.json --partition q-word \
    --out $O/stage --stage-only --batch fixed --n $N --max-statement-bits $((1 << 35)) --batch-ands $((1 << 35)) --selftest 0 \
    > $O/stage/stage.log 2>&1 &
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
  wait $SPID; echo "stage exit $?"
  tail -5 $O/stage/stage.log
  C=$(ls -td $FLOCK_STAGE_CACHE/*/ 2>/dev/null | head -1)
  [ -n "$C" ] && [ -f $C/inst-$N.bin ] || { echo "no staged statement under $FLOCK_STAGE_CACHE"; exit 1; }
  mkdir -p $DD/stage && for f in circuit.txt inst-$N.bin pub-$N.bin rec.json; do ln -f $C/$f $DD/stage/$f || cp $C/$f $DD/stage/$f; done
  du -sb $DD/stage/* | tee $O/stage-sizes.txt
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
  rm -rf $DD/inner
  taskset -c $VCPU python3 -m verity_flock.rec_live --listen 127.0.0.1:0 --out $DD/inner --sessions $NS > $O/inner/proxy/proxy.log 2>&1 &
  PP=$!
  wait_serving $O/inner/proxy/proxy.log $PP || { cat $O/inner/proxy/proxy.log; exit 1; }
  ADDR=$(awk '/^SERVING/ {print $2}' $O/inner/proxy/proxy.log)
  prove_live proxy $ADDR $D/inst-$N.bin $D/circuit.txt
  wait $PP; echo "proxy exit $?"
  SESS=$DD/inner/s$((NS - 1))
  [ -f $SESS/session/session.json ] || { tail -20 $O/inner/proxy/proxy.log $O/inner/proxy/prove.err; exit 1; }
  taskset -c $VCPU /usr/bin/time -v -o $O/inner/time-replay-proxy.txt $B replay --instances $D/pub-$N.bin --circuit $D/circuit.txt \
    --sessions $SESS/session > $O/inner/replay-proxy.txt 2>&1
  taskset -c $VCPU /usr/bin/time -v -o $O/inner/time-vstar.txt python3 -m verity_flock.rec_vstar $SESS > $O/inner/vstar.json 2> $O/inner/vstar.err
  echo "vstar exit $?"
  mkdir -p $O/inner/session && cp -r $SESS/session $SESS/public.json $O/inner/session/ && du -sb $SESS/* > $O/inner/session/sizes.txt
  fi
  # today's numbers on this node and build: M0 and --zk on loopback, the verifier on its own cores, every verdict ahead
  loopback m0 $D/pub-$N.bin $D/inst-$N.bin $D/circuit.txt
  loopback zk $D/pub-$N.bin $D/inst-$N.bin $D/circuit.txt --zk
  [ -n "${LAST:-}" ] && taskset -c $VCPU /usr/bin/time -v -o $O/inner/time-replay-zk.txt $B replay --zk --instances $D/pub-$N.bin \
    --circuit $D/circuit.txt --sessions $LAST > $O/inner/replay-zk.txt 2>&1
  ;;
ostage)
  INNER=${INNER:-$DD/inner/s$((NS - 1))}
  [ -f $INNER/public.json ] || { echo "INNER=$INNER: no public.json (run STEP=inner first)"; exit 1; }
  ALL=$(python3 -c "
import sys; from pathlib import Path; from verity_flock import rec_outer as O, rec_vstar as V
pub, priv, proofs = V.load(Path(sys.argv[1])); print(' '.join(str(l['level']) for l in O.levels(V.queries(pub, priv, proofs))))" $INNER)
  echo "levels: $ALL"
  mkdir -p $O/ostage; rm -rf $DD/outer
  for l in ${LEVELS:-$ALL}; do echo $l; done | xargs -P ${JOBS:-4} -I{} sh -c \
    "/usr/bin/time -v -o $O/ostage/time-L{}.txt python3 -m verity_flock.rec_outer $INNER $DD/outer --levels {} --link-rows > $O/ostage/L{}.json 2> $O/ostage/L{}.err; echo L{} exit \$?"
  for l in ${LEVELS:-$ALL}; do du -sb $DD/outer/L$l/* > $O/ostage/sizes-L$l.txt; done
  ;;
alg)
  # V*'s algebra: InnerClaims.lean (as an executable, cached by its sources as 80-rec-inner.sh caches it) reads the inner's last
  # proxied session for each rep's extra claims and circuit fold, and rec_residuals stages InnerRepCheck on both reps
  INNER=${INNER:-$DD/inner/s$((NS - 1))}
  [ -f $INNER/session/session.json ] || { echo "INNER=$INNER: no session (run STEP=inner first)"; exit 1; }
  CSHA=$( (echo $LSHA; cat backends/flock/tests/lean/InnerClaims.lean) | sha256sum | cut -c1-16)
  CLAIMS_BIN=$L/claims-$CSHA/inner-claims
  (
    export PATH=$HOME/.elan/bin:$PATH
    command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
    exec 9>$L/claims.lock; flock 9
    if [ ! -x $CLAIMS_BIN ]; then
      C=$L/claims-src; rm -rf $C && mkdir -p $C && cp -r backends/flock/verifier/lean/. $C/ && cp backends/flock/tests/lean/InnerClaims.lean $C/ \
        && printf '\n[[lean_exe]]\nname = "inner-claims"\nroot = "InnerClaims"\n' >> $C/lakefile.toml \
        && (cd $C && lake build inner-claims) && mkdir -p $L/claims-$CSHA && cp $C/.lake/build/bin/inner-claims $CLAIMS_BIN
    fi
  ) > $O/build-claims.log 2>&1 || { tail -30 $O/build-claims.log; exit 1; }
  mkdir -p $O/alg
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
  cp $DD/stage/partition.json $O/alg/
  /usr/bin/time -v -o $O/alg/time-claims.txt $CLAIMS_BIN --partition $DD/stage/partition.json verity/flock-circuit \
    $DD/stage/circuit.txt $DD/stage/pub-$N.bin $INNER/session os > $INNER/claims.json 2> $O/alg/claims.err
  rc=$?; echo "claims exit $rc"; [ $rc = 0 ] || { tail -5 $O/alg/claims.err; exit 1; }
  cp $INNER/claims.json $O/alg/
  rm -rf $DD/alg
  /usr/bin/time -v -o $O/alg/time-stage.txt python3 -m verity_flock.rec_residuals $INNER $DD/alg > $O/alg/stage.json 2> $O/alg/stage.err
  echo "alg stage exit $?"
  du -sb $DD/alg/honest/* > $O/alg/sizes.txt
  ;;
oprove)
  [ -d $DD/outer ] || [ "${LEVELS:-}" = alg ] || { echo "run STEP=ostage first"; exit 1; }
  # every level's RecOpen statement, then the algebra's (STEP=alg) when it is staged; LEVELS=0,1,alg picks some
  for C in $(ls -d $DD/outer/L* 2>/dev/null | sort -V) $( [ -f $DD/alg/honest/inst.bin ] && echo $DD/alg/honest); do
    l=$(basename $C); [ $l = honest ] && l=alg
    [ -n "${LEVELS:-}" ] && [[ ",$LEVELS," != *",${l#L},"* ]] && continue
    loopback $l $C/pub.bin $C/inst.bin $C/circuit.txt --zk
    [ -n "${LAST:-}" ] || continue
    taskset -c $VCPU /usr/bin/time -v -o $O/oprove/$l/time-replay.txt $B replay --zk --instances $C/pub.bin --circuit $C/circuit.txt \
      --sessions $LAST > $O/oprove/$l/replay.txt 2>&1
    echo "$l replay exit $?"
    rm -rf $DD/outer-sess/$l; mkdir -p $DD/outer-sess && cp -r $LAST $DD/outer-sess/$l
    du -sb $C/inst.bin $C/pub.bin $C/circuit.txt $LAST/*.bin > $O/oprove/$l/sizes.txt
  done
  ;;
lean)
  [ -x $LEAN_BIN ] || lean_build > $O/build-lean.log 2>&1 || { tail -20 $O/build-lean.log; exit 1; }
  mkdir -p $O/lean
  for S in $(ls -d $DD/outer-sess/* | sort -V); do l=$(basename $S); C=$DD/outer/$l; [ $l = alg ] && C=$DD/alg/honest
    [ -n "${LEVELS:-}" ] && [[ ",$LEVELS," != *",${l#L},"* ]] && continue
    PI=; [ -f $C/public-inputs.bin ] && PI="--public-inputs $C/public-inputs.bin"
    echo "/usr/bin/time -v -o $O/lean/time-$l.txt $LEAN_BIN verify --zk --statement verity/flock-circuit --circuit $C/circuit.txt --public $C/pub.bin --partition $C/partition.json --program $C/program.json --registered $C/registered.json $PI --session $S > $O/lean/$l.log 2> $O/lean/$l.err; echo $l lean exit \$?"
  done | xargs -d '\n' -P ${JOBS:-4} -I{} sh -c "{}"
  if [ "${INNER_LEAN:-0}" = 1 ]; then
    /usr/bin/time -v -o $O/lean/time-inner.txt $LEAN_BIN verify --statement verity/flock-circuit --circuit $DD/stage/circuit.txt \
      --public $DD/stage/pub-$N.bin --partition $DD/stage/partition.json --coins os --session $DD/inner/s$((NS - 1))/session > $O/lean/inner.log 2> $O/lean/inner.err
    echo "inner lean exit $?"
  fi
  # ZK_LEAN=1: today's --zk on the same statement (the inner step's last loopback --zk session), for Lean's like-for-like
  if [ "${ZK_LEAN:-0}" = 1 ]; then
    ZS=$(ls -d $DD/inner-zk/sessions/l* | sort | tail -1)
    /usr/bin/time -v -o $O/lean/time-zk.txt $LEAN_BIN verify --zk --statement verity/flock-circuit --circuit $DD/stage/circuit.txt \
      --public $DD/stage/pub-$N.bin --partition $DD/stage/partition.json --session $ZS > $O/lean/zk.log 2> $O/lean/zk.err
    echo "zk lean exit $?"
  fi
  ;;
esac
summary
true
