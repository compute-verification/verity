#!/usr/bin/env bash
# flock_s8_block: Flock --zk's price for proof blocks of ROWS adjacent cells of one int8 GEMM output row (program.py), each
# instance's rows hashed in the circuit, on one GPU node. Run from the tree (research run --on NODE --cwd source):
#   1. build    flock-circuit with the CUDA prover (60-circuit.sh MODE=build), outside any lease
#   2. stage    each K:N[:R] of the statement lists (class_statement --stage-only, N instances of S8Matmul_v1{N=R, K}; R
#               defaults to ROWS), outside any lease, into data/kK-nN-rR; and with WINDOW, every proof block of a served
#               window's member MEMBER (rec_inner.py stage-window: the window's committed values, the circuit composed for
#               SUBCIRCUIT) as the statements `served` (hm96 leaves, today's --zk statement) and `served-plain` (plain SHA-512
#               leaves), held first, with SERVE_RUN, to the run root SERVE_RUN/windows.jsonl published for window SERVE_WINDOW
#   3. sanity   a loopback --zk session set on the CPU of a K = 64, N = 2 statement (and on the GPU first thing in step 4;
#               SANITY=0 skips that one)
#   4. gpu      under one `gpu-lease 1` of at most LEASE_MIN minutes: per statement, a loopback --zk session set (a verifier
#               process on VCPUS cores, WARM + RUNS prover sessions with --gpu on the rest), and per statement of
#               PLAIN_STATEMENTS the same set without --zk (gpu-plain), sampling the GPU's utilization, memory and, where
#               dcgmi runs, its tensor-pipe and SM activity
#               and with NSYS=1, a last session set (WARM=0, RUNS=2) of NSYS_STATEMENT under Nsight Systems' CUDA trace, with
#               --zk and, with PLAIN_STATEMENTS, without (gpu-plain-nsys): its kernel and memcpy time summaries in out/nsys
#               (the GPU's busy time, not its utilization counter's)
#   5. cpu      per statement of CPU_STATEMENTS (and without --zk, CPU_PLAIN_STATEMENTS), the same session set without --gpu
#               (the CPU prover), no lease
#   6. lean     Lean's flock-verify (verity/protocols/flock/verifier, built into FLOCK_WORK keyed by its sources) on every session
#               steps 4 and 5 kept of `served` and `served-plain`, on the verifier's cores: `verify --zk` of served's --zk sets,
#               and `verify` of served-plain's plain sets as the plain-leaves statement on its full sample; per session its verdict,
#               wall and peak memory in out/lean (hm96 leaves without --zk are outside Lean's plain scope, so not verified)
# A session set passes once the verifier recorded each of its sessions and accepted it (out/MODE/NAME/verdicts.jsonl).
# out/summary.json: per statement and mode, the median timed session's prove time, its phases, GPU busy share and peaks.
# env: STATEMENTS="4096:1 4096:8 14336:1"  CPU_STATEMENTS="4096:1"  PLAIN_STATEMENTS=  CPU_PLAIN_STATEMENTS=  ROWS=8  WARM=1
#      RUNS=3  VCPUS=4  LEASE_MIN=20  SM=90  DRAW=all (a --zk verifier samples every proof block, subset:N, and a plain one
#      of a served statement its staged full sample, draw.json; unset, no sample)
#      WINDOW=DIR (prove_path.py commit's window dir, the prover's files included) MEMBER=0  SUBCIRCUIT=ID  SERVE_RUN=DIR
#      SERVE_WINDOW=0  CUDA_TK (default: verity/protocols/flock/pod/cuda-13.3.sh's)  NSYS=1  NSYS_STATEMENT (default the first)
#      FLOCK_WORK (default: a directory under /workspace/tmp, kept between steps; it holds the staged values)
#      STEPS="build stage sanity gpu cpu" (lean: add it)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
REPO=$(pwd); O=${RESEARCH_RUN_DIR:-/tmp/flock-s8-block}/out; mkdir -p $O
FW=${FLOCK_WORK:-/workspace/tmp/flock-s8-block}; mkdir -p $FW || exit 1
STATEMENTS=${STATEMENTS:-"4096:1 4096:8 14336:1"}; CPU_STATEMENTS=${CPU_STATEMENTS:-"4096:1"}; ROWS=${ROWS:-8}
PLAIN_STATEMENTS=${PLAIN_STATEMENTS:-}; CPU_PLAIN_STATEMENTS=${CPU_PLAIN_STATEMENTS:-}; DRAW=${DRAW:-}
SANITY=${SANITY:-1}; WARM=${WARM:-1}; RUNS=${RUNS:-3}; NS=$((WARM + RUNS)); SM=${SM:-90}; STEPS=${STEPS:-"build stage sanity gpu cpu"}
export FLOCK_WORK=$FW
. verity/protocols/flock/pod/cuda-13.3.sh
. verity/protocols/spot/verifier/one_stage/verifier_env.sh
export PATH=$CUDA_TK/bin:$HOME/.cargo/bin:$PATH LD_LIBRARY_PATH=$CUDA_TK/lib64:${LD_LIBRARY_PATH:-}
export FC_PIPELINE_DEPTH=${FC_PIPELINE_DEPTH:-4} FC_HOST_PREPIN=${FC_HOST_PREPIN:-1}
CPUS=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))'); NC=$(echo $CPUS | tr , '\n' | wc -l)
VC=${VCPUS:-4}; PCPU=$(echo $CPUS | cut -d, -f1-$((NC - VC))); VCPU=$(echo $CPUS | cut -d, -f$((NC - VC + 1))-)
B=$FW/flock-gpu-link/flock/target/release/flock-circuit; PY=$FW/flock-circuit/py/bin/python
has() { case " $STEPS " in *" $1 "*) return 0 ;; esac; return 1; }
{ echo "cpus=$CPUS prover=$PCPU verifier=$VCPU"; lscpu | grep -E 'Model name|^CPU\(s\)'; free -g | head -2
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv; command -v dcgmi nsys ncu; } > $O/host.txt 2>&1

if has build; then
  echo "== build $(date -u +%T)"
  MODE=build GPU=1 SM=$SM FLOCK_WORK=$FW RESEARCH_RUN_DIR=$O/build bash verity/protocols/flock/pod/60-circuit.sh > $O/build.log 2>&1 \
    || { tail -40 $O/build.log $O/build/out/build.log 2>/dev/null; exit 1; }
  sha256sum $B | tee $O/binary.sha256
fi
UV=$(command -v uv || echo $HOME/.local/bin/uv)
[ -x $UV ] || { curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1; UV=$HOME/.local/bin/uv; }
[ -x $PY ] || { $UV venv -q --python 3.12 $(dirname $(dirname $PY)) && $UV pip install -q --python $PY numpy blake3; } || exit 1
export PYTHONPATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/protocols/flock/prover/python:$REPO/tools/verity/circuit_check/src:$REPO/adapters/vllm

# spec K:N[:R] or served[-plain]: "K N R NAME" (a served statement's N its member's n, K and R -)
spec() {
  case $1 in served*) echo "- $(cat $FW/data/$1/n 2>/dev/null) - $1"; return ;; esac
  local k n r; IFS=: read -r k n r <<< "$1"; r=${r:-$ROWS}; echo "$k $n $r k$k-n$n-r$r"
}
# stage SPEC: the statement of N instances of R rows at K into $FW/data/NAME (circuit.txt, inst-N.bin, pub-N.bin, rec.json)
stage() {
  local k n r name; read -r k n r name <<< "$(spec $1)"
  local d=$FW/data/$name o=$O/stage/$name
  mkdir -p $d $o
  [ -f $d/rec.json ] && [ -f $d/inst-$n.bin ] && { cp $d/rec.json $o/rec.json; return 0; }   # staged by an earlier run
  K=$k ROWS=$r N=$n FLOCK_STAGE_CACHE=$d/cache FLOCK_OWN_ROWS=1 FLOCK_PACK_WORDS=1 /usr/bin/time -v -o $o/time-stage.txt \
    $PY -m verity_flock.class_statement --program-module $REPO/benchmarks/flock_s8_block/program.py:program \
    --partition q-template-instance --out $o/out --stage-only --batch fixed --n $n --max-ands $((1 << 34)) \
    --max-statement-bits $((1 << 35)) --batch-ands $((1 << 35)) --selftest 0 > $o/stage.log 2>&1 || { tail -5 $o/stage.log; return 1; }
  local C=$(ls -td $d/cache/*/ | head -1)
  for f in circuit.txt inst-$n.bin pub-$n.bin rec.json; do ln -f $C/$f $d/$f; done
  cp $C/rec.json $o/rec.json; du -sb $d/circuit.txt $d/inst-$n.bin $d/pub-$n.bin > $o/sizes.txt
}
# stage_window: WINDOW's member MEMBER into $FW/data/served (hm96) and served-plain (plain leaves), each with its n in n
stage_window() {
  local o=$O/stage/served n s
  mkdir -p $o $O/stage/served-plain
  if [ -n "${SERVE_RUN:-}" ]; then   # the window's rows the served run's: its run root the one windows.jsonl published
    $PY -c 'import json, sys
w = json.load(open(sys.argv[1] + "/window.json"))
pub = [x for x in map(json.loads, open(sys.argv[2] + "/windows.jsonl")) if int(x["window"]) == int(sys.argv[3])]
ok = len(pub) == 1 and pub[0]["run_root"] == w["run"]
print(json.dumps({"served_root_held": ok, "window_run_root": w["run"], "published": [x["run_root"] for x in pub],
                  "registration_digest": w["registration_digest"]}))
sys.exit(0 if ok else 1)' "$WINDOW" "$SERVE_RUN" "${SERVE_WINDOW:-0}" > $o/served-root.json || { cat $o/served-root.json; return 1; }
  fi
  /usr/bin/time -v -o $o/time-stage.txt $PY $REPO/verity/protocols/flock/prover/pod/rec_inner.py stage-window --window "$WINDOW" \
    --member ${MEMBER:-0} --subcircuit "${SUBCIRCUIT:?WINDOW needs SUBCIRCUIT}" --sets $FW/data/served-sets \
    --out $FW/data/served-plain --zk-out $FW/data/served > $o/stage.json 2> $o/stage.err || { tail -20 $o/stage.err; return 1; }
  n=$($PY -c 'import json, sys; print(json.load(open(sys.argv[1]))["n"])' $o/stage.json) || return 1
  cp $o/stage.json $O/stage/served-plain/
  for s in served served-plain; do
    echo $n > $FW/data/$s/n; du -sb $FW/data/$s/circuit.txt $FW/data/$s/inst-$n.bin $FW/data/$s/pub-$n.bin > $O/stage/$s/sizes.txt
  done
}
if has stage; then
  echo "== stage $(date -u +%T)"
  if [ -n "${WINDOW:-}" ]; then stage_window || exit 1; fi
  for kn in $({ has sanity || [ $SANITY = 1 ]; } && echo 64:2) \
      $(printf '%s\n' $STATEMENTS $PLAIN_STATEMENTS $CPU_STATEMENTS $CPU_PLAIN_STATEMENTS | grep -v '^served' | sort -u); do
    stage $kn || exit 1
  done
fi
[ -x $B ] || ! { has sanity || has gpu || has cpu; } || { echo "no flock-circuit at $B (STEPS without build needs a FLOCK_WORK that has one)"; exit 1; }

free_port() { python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
wait_serving() { for i in $(seq 1800); do grep -q SERVING $1 && return 0; kill -0 $2 2>/dev/null || return 1; sleep 1; done; return 1; }
stamp() { while IFS= read -r l; do printf '%s\t%s\n' "$(date +%s.%N)" "$l"; done; }
# gpu_sample DIR: until DIR/stop exists, the GPU's utilization, memory, SM clock and power (nvidia-smi) each ~0.2 s, stamped,
# then the prover's cores' busy jiffies (/proc/stat over PCPU); and where dcgmi runs its SM-active, tensor-pipe-active and
# DRAM-active shares (fields 1002, 1004, 1005), stamped
gpu_sample() {
  rm -f $1/stop
  local P2=; command -v dcgmi > /dev/null && { dcgmi dmon -e 1002,1004,1005 -d 200 > >(stamp > $1/dcgm.tsv) 2>&1 & P2=$!; }
  while [ ! -e $1/stop ]; do
    printf '%s\t%s, %s\n' "$(date +%s.%N)" "$(nvidia-smi --query-gpu=utilization.gpu,memory.used,clocks.sm,power.draw \
      --format=csv,noheader,nounits -i ${GPU_LEASE_UUID:-0} 2>&1)" \
      "$(awk -v c=",$PCPU," '$1 ~ /^cpu[0-9]/ && index(c, "," substr($1, 4) ",") {b += $2 + $3 + $4 + $7 + $8 + $9} END {print b}' /proc/stat)"
    sleep 0.2
  done > $1/smi.tsv
  # dcgmi itself (P2): it outlives a closed pipe, and may outlive a SIGTERM
  [ -n "$P2" ] && { kill $P2 2>/dev/null; timeout 5 tail --pid=$P2 -f /dev/null; kill -9 $P2 2>/dev/null; wait $P2 2>/dev/null; }
}
# session SPEC MODE: a verifier process and WARM + RUNS prover sessions against it, into $O/MODE/NAME; MODE gpu or cpu, and
# with "plain" in it (gpu-plain, cpu-plain) without --zk. 0 once the verifier recorded NS sessions and accepted each: the
# prover exits 0 after a session the verifier refused too
session() {
  local k n r name mode=$2 G= Z=--zk DR=; read -r k n r name <<< "$(spec $1)"
  local d=$FW/data/$name o=$O/$2/$name
  case $mode in gpu*) G=--gpu ;; esac
  case $mode in *plain*) Z= ;; esac
  [ "$DRAW" = all ] && if [ -n "$Z" ]; then DR="--draw subset:$n"; elif [ -f $d/draw.json ]; then DR="--draw-file $d/draw.json"; fi
  mkdir -p $o; rm -rf $d/sessions-$mode
  local PORT=$(free_port) PIN=$(sha512sum $d/circuit.txt | cut -d' ' -f1)
  taskset -c $VCPU $B serve $Z --listen 127.0.0.1:$PORT --out $d/sessions-$mode --instances $d/pub-$n.bin --circuit $d/circuit.txt \
    --pin $PIN $DR --operator loopback --concurrency $((NS + 1)) > $o/serve.log 2>&1 &
  local SP=$!
  wait_serving $o/serve.log $SP || { tail -20 $o/serve.log; return 1; }
  gpu_sample $o & local GS=$!
  RAYON_NUM_THREADS=$((NC - VC)) FC_VERIFY_AHEAD=$NS taskset -c $PCPU /usr/bin/time -v -o $o/time-prove.txt ${WRAP:-} $B prove $Z $G \
    --verifier 127.0.0.1:$PORT --instances $d/inst-$n.bin --circuit $d/circuit.txt --warm $WARM --runs $RUNS \
    2> $o/prove.err | stamp > $o/prove.tsv
  local rc=${PIPESTATUS[0]} i
  echo "prove $name $mode exit $rc"
  touch $o/stop; wait $GS
  # serve records a session just after its verdict leaves
  [ $rc != 0 ] || for i in $(seq 300); do [ $(cat $d/sessions-$mode/index.jsonl 2>/dev/null | wc -l) -ge $NS ] && break; sleep 1; done
  kill $SP; wait $SP 2>/dev/null
  cp $d/sessions-$mode/index.jsonl $o/verdicts.jsonl 2>/dev/null
  grep -h panicked $o/prove.err | head -3
  python3 -c 'import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
bad = ["%s: %s" % (r.get("session"), r.get("why") or r.get("aborted")) for r in rows if r.get("accepted") is not True]
if bad or len(rows) != int(sys.argv[2]):
    sys.exit(f"{sys.argv[1]}: the verifier accepted {len(rows) - len(bad)} of {sys.argv[2]} sessions: " + "; ".join(bad))' \
    $o/verdicts.jsonl $NS
}

if has sanity; then
  echo "== sanity $(date -u +%T)"
  session 64:2 cpu || exit 1
fi
NR=$FW/nsys-root; NSB=$NR/opt/nvidia/nsight-systems/2026.1.3/bin/nsys
if has gpu && [ "${NSYS:-0}" = 1 ] && [ ! -x $NSB ]; then
  mkdir -p $NR/debs && (cd $NR/debs && { apt-get download nsight-systems-2026.1.3 2>/dev/null || curl -fsSLO \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/nsight-systems-2026.1.3_2026.1.3.425-1_amd64.deb; }) \
    && for f in $NR/debs/*.deb; do dpkg-deb -x $f $NR; done
  [ -x $NSB ] || { echo "nsys unpack failed: NSYS=0"; NSYS=0; }
fi
if has gpu; then
  echo "== gpu $(date -u +%T)"
  export -f session spec gpu_sample free_port wait_serving stamp
  export B O FW PCPU VCPU NC VC NS WARM RUNS STATEMENTS PLAIN_STATEMENTS DRAW SANITY ROWS NSB NSYS NSYS_STATEMENT
  GPU_LEASE_WHO=${GPU_LEASE_WHO:-flock-s8-block} gpu-lease 1 --wait --max-min ${LEASE_MIN:-20} -- bash -c '
    nvidia-smi -L > $O/lease-gpu.txt
    [ "${SANITY:-1}" = 1 ] && session 64:2 gpu
    for kn in $STATEMENTS; do session $kn gpu; done
    for kn in $PLAIN_STATEMENTS; do session $kn gpu-plain; done
    if [ "${NSYS:-0}" = 1 ]; then
      rm -rf $FW/nsys-reps
      for m in gpu-nsys ${PLAIN_STATEMENTS:+gpu-plain-nsys}; do
        mkdir -p $FW/nsys-reps/$m
        WARM=0 RUNS=2 NS=2 WRAP="$NSB profile -t cuda --sample=none --cpuctxsw=none -o $FW/nsys-reps/$m/prove-%p" \
          session ${NSYS_STATEMENT:-${STATEMENTS%% *}} $m
      done
    fi'
  if [ "${NSYS:-0}" = 1 ]; then
    mkdir -p $O/nsys; ls -laR $FW/nsys-reps > $O/nsys/reps.txt 2>&1
    for f in $FW/nsys-reps/*/*.nsys-rep; do
      $NSB stats -q -r cuda_gpu_kern_sum,cuda_gpu_mem_time_sum -f csv -o $O/nsys/$(basename $(dirname $f))-$(basename ${f%.nsys-rep}) $f \
        >> $O/nsys/stats.log 2>&1
    done
  fi
fi
if has cpu; then
  echo "== cpu $(date -u +%T)"
  for kn in $CPU_STATEMENTS; do session $kn cpu; done
  for kn in $CPU_PLAIN_STATEMENTS; do session $kn cpu-plain; done
fi
LEAN_BAD=0
if has lean; then
  echo "== lean $(date -u +%T)"
  LSHA=$(python3 verity/protocols/flock/pod/lake_closure.py key verity/protocols/flock/verifier)
  LV=$FW/lean-$LSHA/flock-verify
  if [ ! -x $LV ]; then
    export PATH=$HOME/.elan/bin:$PATH
    command -v lake > /dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none > /dev/null
    rm -rf $FW/lean-src && LEAN_SRC=$(python3 verity/protocols/flock/pod/lake_closure.py copy verity/protocols/flock/verifier $FW/lean-src) \
      && (cd $LEAN_SRC && /usr/bin/time -v -o $O/time-lean-build.txt lake build flock-verify) > $O/lean-build.log 2>&1 \
      && mkdir -p $(dirname $LV) && cp $LEAN_SRC/.lake/build/bin/flock-verify $LV || { tail -30 $O/lean-build.log; exit 1; }
  fi
  mkdir -p $O/lean
  for name in served served-plain; do
    d=$FW/data/$name; [ -f $d/n ] || continue; n=$(cat $d/n)
    for set in $d/sessions-*; do
      mode=${set##*/sessions-}
      case $mode in *nsys*) continue ;; *plain*) [ $name = served-plain ] || continue
        V=(--statement verity/flock-circuit+plain-leaves --stratified $n --coins os) ;;
        *) [ $name = served ] || continue; V=(--zk --statement verity/flock-circuit) ;; esac
      o=$O/lean/$mode/$name; mkdir -p $o
      for S in $set/*/; do
        [ -f $S/session.json ] || continue; l=$(basename $S)
        taskset -c $VCPU /usr/bin/time -v -o $o/time-$l.txt $(verifier_env) $LV verify "${V[@]}" --circuit $d/circuit.txt --public $d/pub-$n.bin \
          --partition $d/partition.json --program $d/program.json --session $S > $o/$l.log 2>&1
        echo $? > $o/$l.rc; [ "$(cat $o/$l.rc)" = 0 ] || { LEAN_BAD=1; echo "lean $mode $name $l refused: $(tail -1 $o/$l.log)"; }
      done
    done
  done
fi
echo "== summary $(date -u +%T)"
python3 $REPO/benchmarks/flock_s8_block/summary.py $O | tee $O/summary.json
exit $LEAN_BAD
