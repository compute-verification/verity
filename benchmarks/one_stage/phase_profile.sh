#!/usr/bin/env bash
# A prover phase's profile and A/B on one GPU of a pod, from the source tree under research run (`--cwd source`). AS=base records
# the source's build as the base; AS=new builds the source and compares it with the base. State persists under KW (default
# /workspace/jobs/phase) between runs on the pod. STEPS (default setup,build,stage,ident,tl):
#   setup  CUDA 13.3 (cuda-13.3.sh), Nsight Systems 2026.1.3 and a Python venv, unpacked under KW
#   build  the source through 70-class-sweep.sh BUILD_ONLY=1, its binaries in KW/AS.env
#   stage  UNITS (default W13, the census's decode MoE gate-up block, whose transcripts main pins) at full with every unit drawn
#          (zk_cost.py stage), once per KW and UNITS
#   test   TESTS (default the hm96 trees', the arena layout's and level 0's fused folds') under pytest, slow ones included, on
#          the pod's nvcc and GPU, against b684b12's cuda-ghash from the build's Flock checkout; a skip fails
#   ident  prover_is_deterministic pinned to main's digests under Flock ZK and Flock's circuit prover; gpu_paths_agree under both
#   tl     one warm process (warm 1 + RUNS Flock ZK dev sessions) under nsys, CUDA and OS calls traced; phase_profile.py splits each
#          rep's PHASE (default t.ligerito) window, anchored at ANCHOR's launch (default lf_fold_base_pair)
#   an     phase_profile.py again on AS's newest trace (KW/traces; PROVE_OUT: its session records, where they aren't beside it)
#   session  session_profile.py on AS's newest trace (the tl step's): every phase of a session's prove time, ranked in ms a
#          session; with GAPS (ms), also the host's calls and the GPU's ops in that window around each rep's device windows
#   peak   for each build (base and new) and each of PEAK_UNITS (default W13's and gq's statements, `;` between them) at full,
#          each staged once per KW (gq's stage takes about 50 min): a prover_is_deterministic selftest under Flock ZK and
#          under Flock's circuit prover, and one Flock ZK dev session, each under nsys with its allocations traced
#          (device_peak.py: the peak of live device allocations and what was live then) and the GPU's processes sampled every
#          100 ms by nvidia-smi (which also counts the CUDA context)
#   ab     ROUNDS rounds, order rotating, of one warm process (warm 1 + 3 Flock ZK dev sessions) per build, AB_BASE and AB_NEW
#          (default base and new: the builds of AS=base and AS=new); PHASE's median a rep and a session
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run under research run}; O=$D/out; mkdir -p $O
REPO=$(pwd); W=${KW:-/workspace/jobs/phase}; mkdir -p $W; AS=${AS:-new}; STEPS=${STEPS:-setup,build,stage,ident,tl}
W13='census:MoeExpertFp8BlockGemm_v3{K=6144,N=1,G=128,KB=128,SWAP=false,DOT={"fn":"HopperE4m3MmaV2Dot32_v1"}}'
END=${END:-$(( $(date +%s) + 6 * 3600 ))}; ROUNDS=${ROUNDS:-3}; RUNS=${RUNS:-2}; UNITS=${UNITS:-$W13}
PHASE=${PHASE:-t.ligerito}; ANCHOR=${ANCHOR:-lf_fold_base_pair}
PEAK_UNITS=${PEAK_UNITS:-$W13;gq}
MAIN_ZK=493e51d79e3d5c9dc0457e7670245e84218a3cb360b133cd3a35206a7ce7106fce3db0bf2db19d7f158f7994bd16f1b03f3bb056b257c0a2ccd6082aa36eaae6
MAIN_M0=0f0dfa3c0bb8a743e1b40ded2b8813ca27758061f1fb194e68be62435dd521dec815920b33ad22ec3a4b6a7f68b4c5d3a4fc2cdbc80e3501b4088eaba53f2337
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a $O/job.log; }
has() { [[ ",$STEPS," == *",$1,"* ]]; }
fits() { [ $(( $(date +%s) + $1 * 60 )) -lt $END ] || { log "$2: under $1 min left before END, skipped"; return 1; }; }
S=$(mktemp -d $W/scratch.XXXXXX); trap 'rm -rf "$S"' EXIT
log "phase_profile: source $(git rev-parse HEAD 2>/dev/null) as $AS, steps $STEPS, units $UNITS, end $(date -u -d @$END +%H:%MZ)"
nvidia-smi -L | tee -a $O/job.log; lscpu | grep -E 'Model name|^CPU\(s\)' | tee -a $O/job.log; free -g | head -2 | tee -a $O/job.log
if has setup; then
  if ! apt-cache show cuda-nvcc-13-3 > /dev/null 2>&1; then
    . /etc/os-release; K=$S/cuda-keyring.deb
    curl -fsSL -o $K https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID/./}/x86_64/cuda-keyring_1.1-1_all.deb \
      && dpkg -i $K > /dev/null && apt-get update -qq > /dev/null 2>&1
  fi
  set +u; FLOCK_WORK=$W . verity/protocols/flock/pod/cuda-13.3.sh; set -u
  echo "$CUDA_TK" > $W/cuda.env
  NR=$W/nsys-root
  if [ ! -x $NR/opt/nvidia/nsight-systems/2026.1.3/bin/nsys ]; then
    mkdir -p $NR/debs && (cd $NR/debs && { apt-get download nsight-systems-2026.1.3 2>/dev/null || curl -fsSLO \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/nsight-systems-2026.1.3_2026.1.3.425-1_amd64.deb; }) \
      && for d in $NR/debs/*.deb; do dpkg-deb -x $d $NR; done
  fi
  [ -x $HOME/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
  [ -x $W/py/bin/python3 ] || { $HOME/.local/bin/uv venv -q --python 3.12 $W/py && $HOME/.local/bin/uv pip install -q --python $W/py/bin/python numpy blake3; }
  log "setup: CUDA $(cat $W/cuda.env), nsys $(ls $NR/opt/nvidia/nsight-systems/ 2>/dev/null), python $($W/py/bin/python3 --version 2>&1)"
fi
CUDA_TK=$(cat $W/cuda.env 2>/dev/null); export CUDA_TK LD_LIBRARY_PATH=$CUDA_TK/lib64:${LD_LIBRARY_PATH:-} PATH=$CUDA_TK/bin:$HOME/.cargo/bin:$HOME/.local/bin:$PATH
NSYS=$W/nsys-root/opt/nvidia/nsight-systems/2026.1.3/bin/nsys
PY=$W/py/bin/python3
export PYTHONPATH=$REPO:$REPO/verity/protocols/flock/prover/python:$REPO/adapters/vllm:$REPO/tools/verity/circuit_check/src:$REPO/benchmarks/numerical/python
for i in 1 2 3 4 5 6; do
  miss=$($PY -c 'import circuit_check.targets, verity.circuits.defs, verity.protocols.spot.verifier.one_stage.draw' 2>&1 \
    | sed -n "s/.*No module named '\([^'.]*\).*/\1/p" | tail -1)
  [ -z "$miss" ] && break
  log "python: no module $miss; the venv holds only numpy and blake3 (setup)"; exit 1
done
fail=0
Z=$W/zk-$(echo "$UNITS" | tr -c "a-zA-Z0-9\n" -)
if has stage && [ ! -s $Z/plan.json ]; then
  (nice -n 5 $PY benchmarks/one_stage/zk_cost.py stage --out $Z --units "$UNITS" --sizes full --draws c:dev \
    > $O/stage.txt 2>&1; echo $? > $S/stage.rc) & STAGE_PID=$!
fi
if has build; then
  FLOCK_WORK=$W/fw OUT=$O/build-$AS bash verity/protocols/flock/pod/70-class-sweep.sh BUILD_ONLY=1 GPU=1 SM=90 > $O/build-$AS.txt 2>&1
  log "build $AS rc=$?"
  B=$(sed -n 's/^BINARY=//p' $O/build-$AS.txt | tail -1); SB=$(sed -n 's/^SELFTEST_BINARY=//p' $O/build-$AS.txt | tail -1)
  [ -x "$B" ] && [ -x "$SB" ] || { log "build $AS: no binaries"; tail -30 $O/build-$AS.txt | tee -a $O/job.log; exit 1; }
  sha256sum $B $SB | tee $O/binaries-$AS.sha256 >> $O/job.log; echo "$B $SB" > $W/$AS.env
fi
if has test; then
  $PY -c 'import pytest' 2> /dev/null || $HOME/.local/bin/uv pip install -q --python $PY pytest
  T=${TESTS:-verity/protocols/flock/prover/tests/test_hm96_leaves.py verity/protocols/flock/prover/tests/test_arena_layout.py verity/protocols/flock/prover/tests/test_ligerito_folds.py}
  G=$W/ghash-b684b12
  [ -d $G/cuda-ghash ] || for F in $W/fw/*/flock; do mkdir -p $G && git -C $F archive b684b12 cuda-ghash | tar -x -C $G && break; done
  (cd verity/protocols/flock/prover/tests && PYTHONPATH=$PWD:$PYTHONPATH VERITY_ML_FLOCK_CUDA_GHASH=$G/cuda-ghash \
    $PY -m pytest -q -rs -p no:cacheprovider $(for t in $T; do echo $REPO/$t; done)) > $O/test-$AS.txt 2>&1
  rc=$?; tail -3 $O/test-$AS.txt | sed "s/^/test-$AS /" | tee -a $O/job.log; log "test-$AS rc=$rc"
  [ $rc = 0 ] && ! grep -q skipped $O/test-$AS.txt || fail=1
fi
if [ -n "${STAGE_PID:-}" ]; then
  wait $STAGE_PID; log "stage rc=$(cat $S/stage.rc): $(grep ZKCOST $O/stage.txt | tail -1)"
fi
read -r B SB < $W/$AS.env || { log "no $AS build in $W"; exit 1; }
[ -s $Z/plan.json ] || { log "no plan at $Z"; exit 1; }
read -r DEVP INST CIRCUIT < <($PY -c 'import json, sys
p = json.load(open(sys.argv[1])); d = next(i for i, x in enumerate(p) if x["k"] == x["c"])
print(d, p[d]["inst"], p[d]["stage_dir"] + "/circuit.txt")' $Z/plan.json)
log "plan: dev point $DEVP"
case_run() {  # case_run NAME ZFLAG DIGEST
  local name=$1 z=$2 want=$3
  $SB selftest $z --gpu --only prover_is_deterministic --instances $INST --circuit $CIRCUIT --expect-transcript $want \
    > $O/ident-$AS-$name.txt 2> $O/ident-$AS-$name.err
  local rc=$?; grep -h '"case"' $O/ident-$AS-$name.txt | cut -c1-600 | sed "s/^/$name /" | tee -a $O/job.log
  log "ident-$AS-$name rc=$rc"
  [ $rc = 0 ] && grep -q '"transcript_pinned": *true' $O/ident-$AS-$name.txt || { log "ident-$AS-$name: not pinned to main"; fail=1; }
}
only_run() {  # only_run CASE NAME ZFLAG
  local c=$1 name=$2 z=$3
  $SB selftest $z --gpu --only $c --instances $INST --circuit $CIRCUIT > $O/$c-$AS-$name.txt 2> $O/$c-$AS-$name.err
  local rc=$?; grep -h '"case"' $O/$c-$AS-$name.txt | cut -c1-700 | sed "s/^/$c-$name /" | tee -a $O/job.log
  log "$c-$AS-$name rc=$rc"
  [ $rc = 0 ] && grep -q '"pass": *true' $O/$c-$AS-$name.txt && ! grep -q '"skipped"' $O/$c-$AS-$name.txt || { log "$c-$AS-$name: not passed"; fail=1; }
}
if has ident; then
  case_run m1 --zk $MAIN_ZK
  case_run m0 "" $MAIN_M0
  only_run gpu_paths_agree m1 --zk; only_run gpu_paths_agree m0 ""
fi
run_dev() {  # run_dev DIR BIN RUNS: one warm process of RUNS timed Flock ZK dev sessions
  local dir=$1 bin=$2 runs=$3 z=$S/zk-$1
  fits 6 $dir || return 75
  rm -rf $z; mkdir -p $z/records $O/$dir; cp $Z/plan.json $z/; cp $Z/records/stage.jsonl $z/records/ 2>/dev/null
  timeout -k 20 1200 $PY benchmarks/one_stage/zk_cost.py prove --bin $bin --out $z --points $DEVP --arms dev --cold 0 --warm 1 \
    --runs $runs 2>&1 | grep ZKCOST | cut -c1-400 | sed "s/^/$dir /" | tee -a $O/job.log
  (cd $z && find . -type f \( -name prove.err -o -name prove.out -o -name session.json -o -name '*.jsonl' \) -size -8M -print0 \
    | xargs -0 -r -I{} cp --parents {} $O/$dir/)
}
if has tl; then
  [ -x $NSYS ] || { log "no nsys at $NSYS"; exit 1; }
  printf '#!/bin/bash\n[ "$1" = prove ] && exec %s profile -t cuda,osrt --sample=none --cpuctxsw=none -o %s --force-overwrite true %s "$@"\nexec %s "$@"\n' \
    "$NSYS" "$S/tl-$AS" "$B" "$B" > $S/nsys-wrap; chmod +x $S/nsys-wrap
  run_dev tl-$AS $S/nsys-wrap $RUNS
  if [ -s $S/tl-$AS.nsys-rep ]; then
    $NSYS export --type sqlite --force-overwrite true -o $S/tl-$AS.sqlite $S/tl-$AS.nsys-rep > /dev/null 2>&1
    $NSYS stats -r cuda_gpu_kern_sum,cuda_api_sum,osrt_sum -f csv --force-overwrite true -o $O/tl-$AS $S/tl-$AS.nsys-rep > /dev/null 2>&1
    T=$W/traces/tl-$AS-$(date -u +%H%M); mkdir -p $W/traces
    cp $S/tl-$AS.sqlite $T.sqlite; cp $(find $O/tl-$AS -name prove.out | head -1) $T.prove.out
    xz -T0 -c $S/tl-$AS.nsys-rep > $O/tl-$AS.nsys-rep.xz
  else log "tl-$AS: no report"; fail=1; fi
fi
if has tl || has an; then
  T=$(ls -t $W/traces/tl-$AS-*.sqlite 2>/dev/null | head -1); T=${T%.sqlite}
  if [ -n "$T" ]; then
    $PY benchmarks/one_stage/phase_profile.py --sqlite $T.sqlite --prove-out ${PROVE_OUT:-$T.prove.out} --phase $PHASE --anchor $ANCHOR --skip 2 \
      --segments 2 --json $O/tl-$AS-phase.json > $O/tl-$AS-phase.txt 2>&1
    head -40 $O/tl-$AS-phase.txt | sed "s/^/tl-$AS /" | tee -a $O/job.log
  else log "no trace of $AS under $W/traces"; fi
fi
if has session; then
  T=$(ls -t $W/traces/tl-$AS-*.sqlite 2>/dev/null | head -1); T=${T%.sqlite}
  if [ -n "$T" ]; then
    $PY benchmarks/one_stage/session_profile.py --sqlite $T.sqlite --prove-out ${PROVE_OUT:-$T.prove.out} --skip 1 \
      ${GAPS:+--gaps $GAPS} --json $O/session-$AS.json > $O/session-$AS.txt 2>&1
    sed "s/^/session-$AS /" $O/session-$AS.txt | tee -a $O/job.log
  else log "no trace of $AS under $W/traces"; fail=1; fi
fi
if has peak; then
  [ -x $NSYS ] || { log "no nsys at $NSYS"; exit 1; }
  peak_run() {  # peak_run NAME CMD...: CMD (whose nsys report is $S/pk-NAME) with the GPU's processes sampled every 100 ms
    local name=$1; shift
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -lms 100 > $O/peak-$name.apps 2> /dev/null &
    local sp=$!
    "$@"; local rc=$?
    kill $sp 2> /dev/null; wait $sp 2> /dev/null
    local mib; mib=$(awk -F', *' '$2 ~ /^[0-9]+$/ && $2 + 0 > m {m = $2 + 0} END {print m + 0}' $O/peak-$name.apps)
    local traced=null
    if [ -s $S/pk-$name.nsys-rep ]; then
      $NSYS export --type sqlite --force-overwrite true -o $S/pk-$name.sqlite $S/pk-$name.nsys-rep > /dev/null 2>&1
      $PY benchmarks/one_stage/device_peak.py $S/pk-$name.sqlite --json $O/peak-$name.json > $O/peak-$name.txt 2>&1
      sed "s/^/peak $name /" $O/peak-$name.txt | tee -a $O/job.log
      traced=$($PY -c 'import json, sys; p = json.load(open(sys.argv[1]))["processes"]; print(p[0]["peak_mib"] if p else "null")' \
        $O/peak-$name.json 2> /dev/null || echo null)
      rm -f $S/pk-$name.nsys-rep $S/pk-$name.sqlite
    fi
    log "peak $name rc=$rc traced $traced MiB allocated, nvidia-smi $mib MiB"
    echo "{\"case\": \"$name\", \"rc\": $rc, \"traced_mib\": $traced, \"smi_mib\": $mib}" >> $O/peak.jsonl
  }
  PK="$NSYS profile -t cuda --cuda-memory-usage=true --sample=none --cpuctxsw=none --force-overwrite true"
  IFS=';' read -ra PU <<< "$PEAK_UNITS"
  for U in "${PU[@]}"; do
    ZU=$W/zk-$(echo "$U" | tr -c "a-zA-Z0-9\n" -); tag=$(echo "${U#census:}" | cut -d: -f1 | cut -d'{' -f1)
    if [ ! -s $ZU/plan.json ]; then
      fits 90 "stage $U" || continue
      nice -n 5 $PY benchmarks/one_stage/zk_cost.py stage --out $ZU --units "$U" --sizes full --draws c:dev > $O/stage-$tag.txt 2>&1
      log "stage $U rc=$?: $(grep ZKCOST $O/stage-$tag.txt | tail -1)"
    fi
    [ -s $ZU/plan.json ] || { log "peak $U: no plan"; fail=1; continue; }
    read -r UP UI UC < <($PY -c 'import json, sys
p = json.load(open(sys.argv[1])); d = next(i for i, x in enumerate(p) if x["k"] == x["c"])
print(d, p[d]["inst"], p[d]["stage_dir"] + "/circuit.txt")' $ZU/plan.json)
    for v in base new; do
      read -r VB VSB < $W/$v.env || { log "peak: no $v build"; fail=1; continue; }
      fits 20 "peak $v $tag" || continue
      for m in m1 m0; do
        n=$v-$tag-$m-selftest
        peak_run $n $PK -o $S/pk-$n $VSB selftest $([ $m = m1 ] && echo --zk) --gpu --only prover_is_deterministic --instances $UI \
          --circuit $UC > $O/peak-$v-$tag-$m.txt 2>&1
      done
      n=$v-$tag-m1-dev
      printf '#!/bin/bash\n[ "$1" = prove ] && exec %s -o %s %s "$@"\nexec %s "$@"\n' "$PK" "$S/pk-$n" "$VB" "$VB" > $S/pk-wrap; chmod +x $S/pk-wrap
      z=$S/peak-$v-$tag; rm -rf $z; mkdir -p $z/records; cp $ZU/plan.json $z/; cp $ZU/records/stage.jsonl $z/records/ 2>/dev/null
      peak_run $n timeout -k 20 1800 $PY benchmarks/one_stage/zk_cost.py prove --bin $S/pk-wrap --out $z --points $UP --arms dev \
        --cold 0 --warm 1 --runs 1 > $O/peak-$v-$tag-dev.txt 2>&1
      grep ZKCOST $O/peak-$v-$tag-dev.txt | cut -c1-300 | sed "s/^/peak-$v-$tag /" | tee -a $O/job.log
    done
  done
fi
if has ab; then
  read -r BB _ < $W/${AB_BASE:-base}.env || { log "no ${AB_BASE:-base} build"; exit 1; }
  read -r NB _ < $W/${AB_NEW:-new}.env || { log "no ${AB_NEW:-new} build"; exit 1; }
  log "ab: base is ${AB_BASE:-base}'s build ($BB), new is ${AB_NEW:-new}'s ($NB)"
  for r in $(seq $ROUNDS); do
    if [ $((r % 2)) = 1 ]; then order="base new"; else order="new base"; fi
    for v in $order; do
      case $v in base) run_dev ab-base-r$r $BB 3;; new) run_dev ab-new-r$r $NB 3;; esac
    done
  done
  $PY - $O $PHASE <<'PYEOF' 2>&1 | tee -a $O/job.log
import json, pathlib, statistics, sys
o, phase = pathlib.Path(sys.argv[1]), sys.argv[2]
for v in ("base", "new"):
    tot, ph, ses = [], [], []
    for p in sorted(o.glob(f"ab-{v}-r*/**/prove.out")):
        recs = [json.loads(l.split("\t", 1)[1]) for l in p.read_text().splitlines() if l.startswith("LIVE\t")][1:]
        tot += [r["prove_total_s"] for r in recs]
        ph += [b[phase] for r in recs for b in r["buckets"] if phase in b]
        ses += [sum(b.get(phase, 0.0) for b in r["buckets"]) for r in recs]
    if tot:
        print(f"ab {v}: prove_total_s median {statistics.median(tot):.4f} over {len(tot)} sessions {[round(x, 3) for x in tot]}; "
              f"{phase} median {1e3 * statistics.median(ph):.2f} ms a rep over {len(ph)}, {1e3 * statistics.median(ses):.2f} ms a "
              f"session {[round(1e3 * x, 1) for x in ses]}")
PYEOF
fi
log "phase_profile: done fail=$fail"
exit $fail
