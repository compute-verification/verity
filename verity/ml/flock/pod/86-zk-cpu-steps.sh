#!/usr/bin/env bash
# zk-cpu-steps: --zk's two single-threaded CPU steps (the mask-rank check zk_veil::mask_rank_ok, ZKRANK; level 0's hiding draw
# level0_zk) priced on node 1's GPU prover, one build against another, on 85-rec-reprice.sh's statements (one Call of
# Gemm_v2{K, N=4096}, its GemmCoordinate_v2 shape at N instances). One step per job (research run --queue --cwd source);
# binaries and statements stay under $FLOCK_WORK for the next step, out/ gets the records:
#   STEP=build  (CPU) this tree's GPU prover (60-circuit.sh's build, keyed by its sources as 85-rec-reprice.sh keys it) and the
#               same build with the selftests' seed-injection beside it, Lean's flock-verify (keyed by its sources), and each
#               of STATEMENTS staged into data/k<K>/stage (class_statement --stage-only, BATCH=fixed N) with its partition object
#   STEP=prove  (GPU) for each build of BUILDS (the keys build steps printed; default this tree's), in order, on statement K:
#               loopback M0 (M0=1) and --zk, WARM + RUNS sessions each, upstream's replay --zk of the last --zk session, and
#               the seed-injection build's prover_is_deterministic --zk --gpu, every later build's transcript digest checked
#               against the first's (equal: byte-identical proofs and transcripts)
#   STEP=lean   (CPU) Lean's flock-verify verify --zk on each build's last --zk session of statement K; the job fails
#               unless every build's exited 0 with its VERDICT accepted (lean_verdicts.py), a build without one too
# The prover runs on the job's first PCPUS cores, the verifier (serve, replay) on the rest. Every step writes out/summary-$STEP.json.
# env: FLOCK_WORK=/workspace/jobs/zk-cpu-steps  STATEMENTS="4096:2048 14336:512" (build: K:N each)  K=4096 N=2048 (prove, lean)
#      BUILDS= (prove, lean)  M0=1  WARM=1  RUNS=3  PCPUS=16  SM=120
#      CUDA_TK=/workspace/jobs/rec-reprice/cuda-13.3.1 (node 1's host CUDA 13.0 ptxas doesn't know the kernels' clmad)
#      WARM_FROM=/workspace/jobs/rec-reprice/flock-gpu-link (a checkout whose cargo target starts the first build warm)
#      LEAN_FROM=/workspace/jobs/rec-reprice/cache (read only: its flock-verify of the same sources is used, not rebuilt)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
STEP=${STEP:?STEP=build|prove|lean}
REPO=$(pwd); O=${RESEARCH_RUN_DIR:-/tmp/zk-cpu-steps-job}/out; mkdir -p $O
FW=${FLOCK_WORK:-/workspace/jobs/zk-cpu-steps}; W=$FW/flock-circuit; L=$FW/cache
mkdir -p $W $L || exit 1
[ -w "$HOME" ] || { export HOME=$FW/home; mkdir -p $HOME; }
unset CARGO_TARGET_DIR
export CUDA_TK=${CUDA_TK:-/workspace/jobs/rec-reprice/cuda-13.3.1}
[ -x $CUDA_TK/bin/nvcc ] || { echo "CUDA_TK=$CUDA_TK: no nvcc" >&2; exit 1; }
export PATH=$CUDA_TK/bin:$PATH LD_LIBRARY_PATH=$CUDA_TK/lib64:${LD_LIBRARY_PATH:-}
SM=${SM:-120}; K=${K:-4096}; N=${N:-2048}; WARM=${WARM:-1}; RUNS=${RUNS:-3}; NS=$((WARM + RUNS))
DD=$FW/data/k$K; D=$DD/stage
CPUS=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')
NC=$(echo $CPUS | tr ',' '\n' | wc -l); PC=${PCPUS:-16}; [ $PC -ge $NC ] && PC=$(( NC / 2 ))
PCPU=$(echo $CPUS | cut -d, -f1-$PC); VCPU=$(echo $CPUS | cut -d, -f$((PC + 1))-)
{ echo "step=$STEP k=$K n=$N cpus=$CPUS prover=$PCPU verifier=$VCPU cuda_visible=${CUDA_VISIBLE_DEVICES:-}"; lscpu | grep -E 'Model name|^CPU\(s\)'
  free -g | head -2; cat /proc/loadavg; nvidia-smi -L 2>/dev/null; } > $O/host-$STEP.txt 2>&1
[ -x /usr/bin/time ] || { echo "no /usr/bin/time" >&2; exit 1; }

KEY=$(cat verity/ml/flock/pod/20-gpu-link.sh verity/ml/flock/pod/60-circuit.sh verity/ml/flock/*.patch verity/ml/flock/cuda_*_patch.py \
  $(find verity/ml/flock/cuda verity/ml/flock/live -type f | LC_ALL=C sort) | sha256sum | cut -c1-16)
BUILDS=${BUILDS:-$KEY}
bin() { echo $W/bin-$1-g1-sm$SM/flock-circuit; }
F=$FW/flock-gpu-link/flock
LSHA=$(cd verity/core/service/flock && find . -name .lake -prune -o -type f -print | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)
LEAN_FROM=${LEAN_FROM:-/workspace/jobs/rec-reprice/cache}
LEAN_BIN=$L/lean-$LSHA/flock-verify; [ -x $LEAN_FROM/lean-$LSHA/flock-verify ] && LEAN_BIN=$LEAN_FROM/lean-$LSHA/flock-verify
UV=$(command -v uv || echo $HOME/.local/bin/uv)
[ -x $UV ] || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; UV=$HOME/.local/bin/uv; }
[ -x $W/py/bin/python ] || { $UV venv -q --python 3.12 $W/py && $UV pip install -q --python $W/py/bin/python numpy blake3; } || exit 1
export PATH=$W/py/bin:$PATH PYTHONPATH=$REPO:$REPO/catalog:$REPO/kernels:$REPO/experimental:$REPO/benchmarks/numerical/python:$REPO/verity/ml/flock/python:$REPO/integrations/vllm:$REPO/tools/verity/circuit_check/src
export FC_COINS=os FC_PIPELINE_DEPTH=${FC_PIPELINE_DEPTH:-4} FC_HOST_PREPIN=${FC_HOST_PREPIN:-1}
GI=${CUDA_VISIBLE_DEVICES:-0}; GI=${GI%%,*}

free_port() { python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
# stamp: each line of stdin with the time it arrived (a session's LIVE line is printed when its verdict is in)
stamp() { while IFS= read -r l; do printf '%s\t%s\n' "$(date +%s.%N)" "$l"; done; }
# gpu_sample FILE: the job's GPU's memory in use (MiB), every half second, until FILE.stop exists
gpu_sample() { rm -f $1.stop; while [ ! -e $1.stop ]; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GI 2>/dev/null; sleep 0.5; done > $1; }
wait_serving() { for i in $(seq 3600); do grep -q SERVING $1 && return 0; kill -0 $2 2>/dev/null || return 1; sleep 1; done; return 1; }

# prove_live NAME BIN VERIFIER_ADDR [--zk]: WARM + RUNS sessions of BIN's GPU prover on the prover's cores, into $O/prove/NAME
# (prove.tsv: stamped stdout, time-prove.txt, gpu-mib.txt)
prove_live() {
  local d=$O/prove/$1 b=$2 addr=$3 zk=${4:-}
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
  FC_VERIFY_AHEAD=$NS taskset -c $PCPU /usr/bin/time -v -o $d/time-prove.txt $b prove $zk --gpu --verifier $addr --instances $D/inst-$N.bin \
    --circuit $D/circuit.txt --warm $WARM --runs $RUNS 2> $d/prove.err | stamp > $d/prove.tsv
  touch $d/gpu-mib.txt.stop; wait $GS
  grep -h panicked $d/prove.err | head -3
}

# loopback NAME BIN [--zk]: a verifier process (BIN serve, on the verifier's cores, every session at once) and prove_live
# against it; the sessions under $DD/prove-NAME/sessions, the last one's directory in $LAST
loopback() {
  local name=$1 b=$2 zk=${3:-} s=$DD/prove-$1 PORT=$(free_port) PIN=$(sha512sum $D/circuit.txt | cut -d' ' -f1)
  rm -rf $s; mkdir -p $s $O/prove/$name
  taskset -c $VCPU $b serve $zk --listen 127.0.0.1:$PORT --out $s/sessions --instances $D/pub-$N.bin --circuit $D/circuit.txt --pin $PIN \
    --operator loopback --concurrency $((NS + 1)) > $O/prove/$name/serve.log 2>&1 &
  local SP=$!
  wait_serving $O/prove/$name/serve.log $SP || { tail -20 $O/prove/$name/serve.log; return 1; }
  prove_live $name $b 127.0.0.1:$PORT $zk
  kill $SP; wait $SP 2>/dev/null
  LAST=$(find $s/sessions -name session.json -printf '%T@ %h\n' | sort -n | tail -1 | cut -d' ' -f2)
}

summary() {
  python3 - $O $STEP "$WARM" "$RUNS" "$BUILDS" <<'EOF'
import json, re, statistics, sys
from pathlib import Path
o, step, warm, runs, builds = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5].split()

def text(p):
    return p.read_text() if p.exists() else ""

def tv(p, key):
    m = re.search(key + r".*: (\S+)", text(p))
    return m.group(1) if m else None

def gb(p):
    rss = tv(p, r"Maximum resident set size \(kbytes\)")
    return round(int(rss) / 2**20, 2) if rss else None

def tagged(p, tag):
    return [json.loads(l.split("\t", 1)[1]) for l in text(p).splitlines() if l.startswith(tag + "\t")]

def per_session(xs, rows):
    """Records printed a fixed number of times per session (ZKRANK once per rep, ZKL0 once): each session's sum, timed only."""
    per = len(xs) // rows if xs and rows and len(xs) % rows == 0 else 0
    return [sum(xs[i * per:(i + 1) * per]) for i in range(rows)][warm:] if per else []

def live(d):
    """The WARM + RUNS sessions of one prove_live: medians over the timed ones, the session time as the rollup's (time between
    the last warm session's record and the last timed one's, over RUNS). Level 0's wait: rep 0's prove_s past its t.prove
    bucket (the --zk preparation's time past the witness build, which the prover adds to rep 0)."""
    rows = [(float(t), json.loads(l[5:])) for t, l in (ln.split("\t", 1) for ln in text(d / "prove.tsv").splitlines()) if l.startswith("LIVE")]
    if not rows:
        return {"error": text(d / "prove.err")[-1500:]}
    timed = [(t, v) for t, v in rows if not v.get("warm")]
    warmed = [(t, v) for t, v in rows if v.get("warm")]
    med = lambda f: statistics.median([f(v) for _, v in timed]) if timed else None
    t0 = warmed[-1][0] if warmed else None
    mib = [int(x) for x in text(d / "gpu-mib.txt").split() if x.strip().isdigit()]
    served = sorted(tagged(d / "serve.log", "SESSION"), key=lambda v: v.get("session", ""))
    ranks = [v["s"] for v in tagged(d / "prove.err", "ZKRANK")]
    rank_s = per_session(ranks, len(rows))
    l0 = [v["s"] for v in tagged(d / "prove.err", "ZKL0")]
    l0_s = per_session(l0, len(rows))
    pre_s = per_session([v["s"] for v in tagged(d / "prove.err", "ZKPRE")], len(rows))
    wait0 = lambda v: v["prove_s"][0] - v["buckets"][0]["t.prove"] if v.get("prove_s") and "t.prove" in (v.get("buckets") or [{}])[0] else None
    return {"sessions": len(rows), "timed": len(timed), "accepted": [v.get("accepted") for _, v in rows],
            "prover_cores_busy_before": float(text(d / "bgload.txt").strip() or "nan"),
            "serve_verify_s": statistics.median([v["verify_s"] for v in served[warm:]]) if len(served) > warm else None,
            "serve_verify_all_s": [v.get("verify_s") for v in served], "serve_accepted": [v.get("accepted") for v in served],
            "zkrank_s": statistics.median(rank_s) if rank_s else None, "zkrank_all_s": ranks,
            "zkl0_s": statistics.median(l0_s) if l0_s else None, "zkl0_all_s": l0, "zkl0": (tagged(d / "prove.err", "ZKL0") or [None])[0],
            "zkpre_s": statistics.median(pre_s) if pre_s else None,
            "level0_wait_s": med(wait0) if timed and wait0(timed[0][1]) is not None else None,
            "level0_wait_all_s": [wait0(v) for _, v in rows],
            "prove_total_s": med(lambda v: v.get("prove_total_s")), "prove_total_all_s": [v.get("prove_total_s") for _, v in rows],
            "prove_s": timed[-1][1].get("prove_s") if timed else None,
            "e2e_s": med(lambda v: v.get("e2e_s")), "witness_prebuilt_s": med(lambda v: v.get("witness_prebuilt_s") or 0),
            "session_s": (timed[-1][0] - t0) / len(timed) if t0 and timed else None,
            "proof_bytes": timed[-1][1].get("proof_bytes") if timed else None, "k_log": rows[0][1].get("k_log"),
            "dense_m": rows[0][1].get("dense_m"), "instances": rows[0][1].get("instances"),
            "statement_digest": rows[0][1].get("statement_digest"), "prover_peak_gb": gb(d / "time-prove.txt"),
            "gpu_peak_mib": max(mib) if mib else None,
            "buckets": [{k: x for k, x in b.items() if k.startswith("t.")} for b in (timed[-1][1].get("buckets") or [])] if timed else None}

def replay(p):
    up = next(iter(tagged(p, "UPSTREAM")), None)
    return {"verdict": up if up else text(p)[-600:], "wall": tv(p.with_name("time-" + p.name), r"Elapsed \(wall clock\) time"),
            "peak_gb": gb(p.with_name("time-" + p.name))}

def lean(p):
    at = text(p).find("VERDICT ")
    if at < 0:
        return text(p)[-600:]
    v = json.JSONDecoder(strict=False).raw_decode(text(p)[at + 8:])[0]
    # Main's stderr: inputs (from start), setup (from the session's start), then the session's whole time (setup included)
    ms = [int(x) for x in re.findall(r"^(?:inputs|setup|/\S+): (\d+) ms$", text(p.with_suffix(".err")), re.M)]
    split = {"inputs_s": ms[0] / 1e3, "setup_s": ms[1] / 1e3, "verify_s": (ms[2] - ms[1]) / 1e3} if len(ms) == 3 else {}
    return {**{k: v.get(k) for k in ("accepted", "why")}, **split}

s = {"step": step, "warm": warm, "runs": runs, "builds": builds}
d = o / step
if step == "build":
    s["key"] = text(o / "key.txt").strip()
    s["binary"] = text(o / "binary.sha256").strip()
    s["lean"] = text(o / "lean.sha").strip()
    s["staged"] = {f.parent.name: text(f) for f in sorted(o.glob("stage-*/statement.txt"))}
elif step == "prove":
    for b in builds:
        s[b] = {name: live(d / f"{b}-{name}") for name in ("m0", "zk") if (d / f"{b}-{name}").exists()}
        if (d / f"{b}-zk" / "replay.txt").exists():
            s[b]["replay_zk"] = replay(d / f"{b}-zk" / "replay.txt")
        s[b]["deterministic"] = next(iter(tagged(d / f"{b}-det.txt", "NEG")), text(d / f"{b}-det.err")[-800:] or None)
elif step == "lean":
    s["sessions"] = {f.stem: {"verdict": lean(f), "wall": tv(d / f"time-{f.stem}.txt", r"Elapsed \(wall clock\) time"),
                              "peak_gb": gb(d / f"time-{f.stem}.txt")} for f in sorted(d.glob("*.log"))}
(o / f"summary-{step}.json").write_text(json.dumps(s, indent=1))
print(json.dumps(s))
EOF
}

case $STEP in
build)
  echo $KEY > $O/key.txt
  (
    export PATH=$HOME/.elan/bin:$PATH
    command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
    exec 9>$L/lean.lock; flock 9
    if [ ! -x $LEAN_BIN ]; then
      rm -rf $L/lean-src && mkdir -p $L/lean-src && cp -r verity/core/service/flock/. $L/lean-src/ \
        && (cd $L/lean-src && lake build flock-verify) && mkdir -p $L/lean-$LSHA && cp $L/lean-src/.lake/build/bin/flock-verify $LEAN_BIN
    fi
  ) > $O/build-lean.log 2>&1 &
  LP=$!
  # each statement: one Call of Gemm_v2{K, N=4096}, its GemmCoordinate_v2 shape staged at N instances, and the verifier's copy
  # of the partition object its circuit binds (q-word over the one-Call program)
  for kn in ${STATEMENTS:-4096:2048 14336:512}; do
    k=${kn%:*} n=${kn#*:}; dd=$FW/data/k$k; so=$O/stage-k$k; mkdir -p $so
    [ -f $dd/stage/inst-$n.bin ] && [ -f $dd/stage/partition.json ] && { du -sb $dd/stage/* > $so/sizes.txt; continue; }
    python3 -c "import json, sys; json.dump({'Gemm_v2{K=%s,N=4096,DOT={\"fn\":\"HopperBF16WgmmaDot16_v1\"}}' % sys.argv[1]: 1151}, open(sys.argv[2], 'w'))" $k $so/defs.json
    FLOCK_STAGE_CACHE=$dd/stage-cache FLOCK_GEMM_TILE=${FLOCK_GEMM_TILE:-4x4} /usr/bin/time -v -o $so/time-stage.txt \
      python3 -m verity_flock.class_statement --definitions $so/defs.json --partition q-word --out $so/out --stage-only --batch fixed \
      --n $n --max-statement-bits $((1 << 35)) --batch-ands $((1 << 35)) --selftest 0 > $so/stage.log 2>&1
    echo "stage k$k exit $?"; tail -5 $so/stage.log
    C=$(ls -td $dd/stage-cache/*/ 2>/dev/null | head -1)
    [ -n "$C" ] && [ -f $C/inst-$n.bin ] || { echo "no staged statement under $dd/stage-cache"; exit 1; }
    mkdir -p $dd/stage && for f in circuit.txt inst-$n.bin pub-$n.bin; do ln -f $C/$f $dd/stage/$f || cp $C/$f $dd/stage/$f; done
    python3 - $k $dd/stage/partition.json $dd/stage/circuit.txt <<'EOF' || exit 1
import json, sys
from verity.ml.circuits import partition_object as PO
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
    du -sb $dd/stage/* > $so/sizes.txt
  done
  B=$(bin $KEY); BS=$(dirname $B)/flock-circuit-selftest
  if [ ! -x $B ] || [ ! -x $BS ]; then
    exec 8>$W/build.lock; flock 8
    if [ ! -x $B ] || [ ! -x $BS ]; then
      [ -d $F ] || { du -sh ${WARM_FROM:-/workspace/jobs/rec-reprice/flock-gpu-link}; cp -a ${WARM_FROM:-/workspace/jobs/rec-reprice/flock-gpu-link} $FW/flock-gpu-link; }
      RESEARCH_RUN_DIR=$O/build FLOCK_WORK=$FW MODE=build GPU=1 SM=$SM bash verity/ml/flock/pod/60-circuit.sh > $O/build.log 2>&1 \
        || { tail -40 $O/build.log; tail -40 $O/build/out/build-circuit.txt 2>/dev/null; exit 1; }
      mkdir -p $(dirname $B) && cp $F/target/release/flock-circuit $B
      # the selftests' build of the same checkout (60-circuit.sh MODE=selftest's features), for prover_is_deterministic
      (source $HOME/.cargo/env && cd $F && NVCC=$CUDA_TK/bin/nvcc cargo build --release -p flock-live --features sha512 --features glue \
        --features gpu --features seed-injection --bin flock-circuit) > $O/build-selftest.txt 2>&1 || { tail -40 $O/build-selftest.txt; exit 1; }
      cp $F/target/release/flock-circuit $BS
    fi
    exec 8>&-
  fi
  sha256sum $B $BS | tee $O/binary.sha256
  for kn in ${STATEMENTS:-4096:2048 14336:512}; do
    k=${kn%:*} n=${kn#*:}; $B statement --instances $FW/data/k$k/stage/inst-$n.bin --circuit $FW/data/k$k/stage/circuit.txt > $O/stage-k$k/statement.txt 2>&1
  done
  wait $LP; echo "lean build exit $?"
  [ -x $LEAN_BIN ] && sha256sum $LEAN_BIN > $O/lean.sha
  tail -5 $O/build-lean.log
  ;;
prove)
  [ -f $D/inst-$N.bin ] || { echo "k$K: run STEP=build first"; exit 1; }
  for b in $BUILDS; do [ -x $(bin $b) ] && [ -x $(dirname $(bin $b))/flock-circuit-selftest ] || { echo "build $b: run STEP=build at its tree"; exit 1; }; done
  mkdir -p $O/prove
  PIN_T=
  for b in $BUILDS; do
    B=$(bin $b)
    [ "${M0:-1}" = 1 ] && loopback $b-m0 $B
    LAST=; loopback $b-zk $B --zk
    if [ -n "$LAST" ]; then
      taskset -c $VCPU /usr/bin/time -v -o $O/prove/$b-zk/time-replay.txt $B replay --zk --instances $D/pub-$N.bin --circuit $D/circuit.txt \
        --sessions $LAST > $O/prove/$b-zk/replay.txt 2>&1
      echo "$b replay exit $?"
      rm -rf $DD/sess/$b-zk; mkdir -p $DD/sess && cp -r $LAST $DD/sess/$b-zk && du -sb $LAST/* > $O/prove/$b-zk/session-sizes.txt
    fi
    taskset -c $PCPU $(dirname $B)/flock-circuit-selftest selftest --zk --gpu --instances $D/inst-$N.bin --circuit $D/circuit.txt \
      --only prover_is_deterministic ${PIN_T:+--expect-transcript $PIN_T} > $O/prove/$b-det.txt 2> $O/prove/$b-det.err
    echo "$b determinism exit $?"; grep NEG $O/prove/$b-det.txt | cut -c1-600
    [ -n "$PIN_T" ] || PIN_T=$(grep -o '"transcript_sha256":"[0-9a-f]*"' $O/prove/$b-det.txt | cut -d'"' -f4)
  done
  ;;
lean)
  [ -x $LEAN_BIN ] || { echo "no Lean verifier $LEAN_BIN: run STEP=build first"; exit 1; }
  mkdir -p $O/lean
  for b in $BUILDS; do
    [ -f $DD/sess/$b-zk/session.json ] || { echo "build $b: no --zk session of k$K (run STEP=prove first)"; continue; }
    echo "/usr/bin/time -v -o $O/lean/time-$b.txt $LEAN_BIN verify --zk --statement verity/flock-circuit --circuit $D/circuit.txt --public $D/pub-$N.bin --partition $D/partition.json --session $DD/sess/$b-zk > $O/lean/$b.log 2> $O/lean/$b.err; echo $b lean exit \$?"
  done | xargs -d '\n' -P ${JOBS:-2} -I{} sh -c "{}"
  python3 verity/ml/flock/pod/lean_verdicts.py $O/lean $BUILDS; LEAN_RC=$?
  ;;
esac
summary
exit ${LEAN_RC:-0}
