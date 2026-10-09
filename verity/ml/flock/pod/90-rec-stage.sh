#!/usr/bin/env bash
# Staging UniversalUnit_v1{G, N_IN, N_OUT} as the recursion's inner statement at a real private circuit's size (rec_stage.py).
#   MODE=measure  today's path (rec_inner.universal's staging) on each class of LADDER, side by side, one process each:
#                 $OUT/measure-G-N_IN-N_OUT.jsonl (each phase's seconds and peak RSS) and .log; PROFILE_G: that class
#                 under cProfile as well ($OUT/profile-*.txt)
#   MODE=fast     rec_inner.py universal --fast (verity_flock.universal_stage) on CLS, the private circuit NETLIST (a
#                 universal.Netlist JSON, relative to the run dir: research run --send FILE puts it in inputs/) padded to
#                 the class, else a random one: DIR/stage and DIR/other as 87-rec-private.sh stages them, kept under
#                 $FLOCK_WORK/data/rec-stage/$TAG; $OUT/fast-$TAG.jsonl (phases, RESULT), time-fast-$TAG.txt, files-$TAG.txt
#   MODE=inner    that statement's inner session on the CPU (M0) against the rec proxy, and its replay: $OUT/inner/
#   MODE=fold     V*'s points from that session and InnerFold.lean's fold of it over the statement: $OUT/fold/
# Run from the verity tree (research run --cwd source). Big files under $FLOCK_WORK (default /workspace/jobs/rec-stage).
# env: LADDER="128,64,32 256,64,32 512,64,32 1024,64,32 2048,64,32"  PROFILE_G=128  N=8  CPUS=16  TIMEOUT=10800 (seconds, per class)
#      CLS NETLIST TAG (default CLS-nN)  (MODE=fast)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
if [ -z "${RS_PINNED:-}" ]; then
  . verity/ml/flock/pod/cpu-slices.sh
  hold_slices "" "${CPUS:-16}" || exit 1
  export RS_PINNED=$SLICE_CPUS
  exec taskset -c $SLICE_CPUS bash "$0" "$@"
fi
REPO=$(pwd); O=${OUT:-${RESEARCH_RUN_DIR:-/tmp/rec-stage-job}/out}; mkdir -p $O
FW=${FLOCK_WORK:-/workspace/jobs/rec-stage}; mkdir -p $FW || exit 1
[ -w "$HOME" ] || { export HOME=$FW/home; mkdir -p $HOME; }
UV=$(command -v uv || echo $HOME/.local/bin/uv)
PYV=$FW/py
[ -x $PYV/bin/python ] || { $UV venv -q --python 3.12 $PYV && $UV pip install -q --python $PYV/bin/python numpy blake3; } || exit 1
export PATH=$PYV/bin:$PATH PYTHONPATH=$REPO:$REPO/catalog:$REPO/kernels:$REPO/verity/ml/flock/python
{ lscpu | grep -E 'Model name|^CPU\(s\)'; echo "pinned=$RS_PINNED"; free -g | head -2; cat /proc/loadavg; } > $O/host.txt 2>&1
S=$FW/scratch/${RESEARCH_RUN_ID:-local-$$}; rm -rf $S; mkdir -p $S || exit 1
trap 'rm -rf $S' EXIT
PY=verity/ml/flock/pod/rec_stage.py
MODE=${MODE:-measure}; N=${N:-8}; TIMEOUT=${TIMEOUT:-10800}

if [ $MODE = measure ]; then
  pids=()
  for c in ${LADDER:-128,64,32 256,64,32 512,64,32 1024,64,32 2048,64,32}; do
    t=${c//,/-}
    prof=(); [ "${PROFILE_G:-128}" = "${c%%,*}" ] && prof=(--profile $S/profile-$t.prof)
    ( timeout $TIMEOUT /usr/bin/time -f '%e s %M KiB' -o $O/measure-$t.time python3 $PY measure --cls $c --n $N --out $S/$t "${prof[@]}" \
        > $O/measure-$t.jsonl 2> $O/measure-$t.log
      echo "exit=$?" >> $O/measure-$t.time
      ls -l $S/$t >> $O/measure-$t.log 2>&1
      [ -n "${prof[*]}" ] && python3 -c "import pstats, sys; pstats.Stats(sys.argv[1], stream=open(sys.argv[2], 'w')).sort_stats(sys.argv[3]).print_stats(40)" \
        $S/profile-$t.prof $O/profile-$t-tottime.txt tottime
      [ -n "${prof[*]}" ] && python3 -c "import pstats, sys; pstats.Stats(sys.argv[1], stream=open(sys.argv[2], 'w')).sort_stats(sys.argv[3]).print_stats(60)" \
        $S/profile-$t.prof $O/profile-$t-cumtime.txt cumtime
      rm -rf $S/$t ) &
    pids+=($!)
  done
  wait "${pids[@]}"
  grep -h RESULT $O/measure-*.jsonl | sed 's/^RESULT //' > $O/summary.jsonl
  exit 0
fi

if [ $MODE = fast ]; then
  CLS=${CLS:?MODE=fast needs CLS}; TAG=${TAG:-${CLS//,/-}-n$N}; NL=()
  if [ -n "${NETLIST:-}" ]; then
    case $NETLIST in /*) ;; *) NETLIST=${RESEARCH_RUN_DIR:?}/$NETLIST ;; esac
    [ -s $NETLIST ] || { echo "no netlist at $NETLIST" >&2; exit 1; }
    sha256sum $NETLIST > $O/netlist.sha256; NL=(--netlist $NETLIST)
  fi
  DD=$FW/data/rec-stage/$TAG; rm -rf $DD; mkdir -p $DD || exit 1
  command -v research >/dev/null && research keep $DD --owner @proofs --expires 2026-10-23 --priority low --why "rec-stage $TAG" > /dev/null 2>&1
  timeout $TIMEOUT /usr/bin/time -v -o $O/time-fast-$TAG.txt python3 $PY fast --cls $CLS --n $N --out $DD "${NL[@]}" \
    > $O/fast-$TAG.jsonl 2> $O/fast-$TAG.log
  rc=$?; echo "exit=$rc" >> $O/time-fast-$TAG.txt
  { echo "dir=$DD"; ls -l $DD/stage $DD/other; sha256sum $DD/stage/* $DD/other/*; } > $O/files-$TAG.txt 2>&1
  grep -h RESULT $O/fast-$TAG.jsonl | sed 's/^RESULT //' > $O/summary.json
  exit $rc
fi

if [ $MODE = inner ]; then
  # the inner session of MODE=fast's statement $FW/data/rec-stage/$TAG/stage, proved on the CPU (M0, FC_COINS=os) against the
  # rec proxy as 87-rec-private.sh's inner() proves it (no firewall), and the live build's replay; M0 is rec-env.sh's build of
  # these sources, copied from a job that has it (this job does not build it)
  TAG=${TAG:?MODE=inner needs TAG}; DD=$FW/data/rec-stage/$TAG; ST=$DD/stage; I=$DD/inner; D=$O/inner; mkdir -p $D; rm -rf $I
  [ -s $ST/circuit.txt ] || { echo "no staged statement at $ST" >&2; exit 1; }
  KEY=$(cat verity/ml/flock/pod/20-gpu-link.sh verity/ml/flock/pod/60-circuit.sh verity/ml/flock/*.patch verity/ml/flock/cuda_*_patch.py \
    $(find verity/ml/flock/cuda verity/ml/flock/live -type f | LC_ALL=C sort) | sha256sum | cut -c1-16)
  SM=${SM:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d .)}; SM=${SM:-89}
  BIN=$FW/flock-circuit/bin-$KEY-g0-sm$SM
  if [ ! -x $BIN/flock-circuit ]; then
    for c in /workspace/jobs/*/flock-circuit/bin-$KEY-g0-sm$SM; do
      [ -x $c/flock-circuit ] && mkdir -p $BIN && cp $c/flock-circuit $BIN/ && break
    done
  fi
  [ -x $BIN/flock-circuit ] || { echo "no CPU build of M0 $KEY-sm$SM on this node (rec-env.sh builds it)" >&2; exit 1; }
  B=$BIN/flock-circuit; sha256sum $B > $O/binary.sha256
  export FC_COINS=os RAYON_NUM_THREADS=$(echo $RS_PINNED | tr ',' '\n' | wc -l)
  export PYTHONPATH=$PYTHONPATH:$REPO/experimental:$REPO/benchmarks/numerical/python:$REPO/integrations/vllm
  PUB=$(ls $ST/pub-*.bin | head -1); INST=$(ls $ST/inst-*.bin | head -1)
  /usr/bin/time -v -o $D/time-statement.txt $B statement --instances $PUB --circuit $ST/circuit.txt > $D/statement.txt 2>&1 \
    || { tail -5 $D/statement.txt; exit 1; }
  python3 -m verity_flock.rec_live --schedule-for $D/statement.txt > $D/schedule.json || exit 1
  python3 -m verity_flock.rec_live --listen 127.0.0.1:0 --out $I --schedule $D/schedule.json > $D/proxy.log 2>&1 &
  PP=$!; ADDR=
  for i in $(seq 600); do ADDR=$(awk '/^SERVING/ {print $2}' $D/proxy.log); [ -n "$ADDR" ] && break; kill -0 $PP 2>/dev/null || break; sleep 0.5; done
  [ -n "$ADDR" ] || { cat $D/proxy.log; exit 1; }
  timeout $TIMEOUT /usr/bin/time -v -o $D/time-prove.txt $B prove --verifier $ADDR --instances $INST --circuit $ST/circuit.txt --runs 1 --warm 0 \
    > $D/prove.txt 2> $D/prove.err
  echo "prove exit $?" | tee $D/prove.exit
  wait $PP; echo "proxy exit $?" >> $D/prove.exit
  [ -f $I/session/session.json ] || { tail -20 $D/proxy.log $D/prove.err; exit 1; }
  /usr/bin/time -v -o $D/time-replay.txt $B replay --instances $PUB --circuit $ST/circuit.txt --sessions $I/session > $D/replay.txt 2>&1
  echo "replay exit $?" >> $D/prove.exit
  cp $I/public.json $D/ && du -sb $I/session/* $I/*.json > $D/sizes.txt
  cat $D/prove.exit
  exit 0
fi

if [ $MODE = fold ]; then
  # V*'s points from MODE=inner's session and its native side (InnerFold.lean's inner-fold) on the statement, as
  # 87-rec-private.sh's fold() runs it: $OUT/fold/; inner-fold is 87's build of these sources, copied from a job that has it
  TAG=${TAG:?MODE=fold needs TAG}; DD=$FW/data/rec-stage/$TAG; ST=$DD/stage; I=$DD/inner; D=$O/fold; mkdir -p $D
  [ -f $I/public.json ] || { echo "no inner session at $I (MODE=inner)" >&2; exit 1; }
  TSHA=$( (find verity/core/service/flock verity/core/service/firewall verity/core/protocols/interface verity/core/difftest/lean \
    -name .lake -prune -o -type f -print | LC_ALL=C sort | xargs sha256sum; cat verity/ml/flock/tests/lean/InnerFold.lean) | sha256sum | cut -c1-16)
  T=$FW/tools-$TSHA
  if [ ! -x $T/inner-fold ]; then
    for c in /workspace/jobs/*/cache/*/tools-$TSHA; do
      [ -x $c/inner-fold ] && mkdir -p $T && cp $c/inner-fold $T/ && break
    done
  fi
  [ -x $T/inner-fold ] || { echo "no inner-fold of tools-$TSHA on this node (87-rec-private.sh builds it)" >&2; exit 1; }
  sha256sum $T/inner-fold > $O/inner-fold.sha256
  export PYTHONPATH=$PYTHONPATH:$REPO/experimental:$REPO/benchmarks/numerical/python:$REPO/integrations/vllm
  PUB=$(ls $ST/pub-*.bin | head -1)
  /usr/bin/time -v -o $D/time-points.txt python3 -m verity_flock.rec_vstage points $I $ST/circuit.txt $PUB > $D/points.json 2> $D/points.err \
    || { tail -5 $D/points.err; exit 1; }
  timeout $TIMEOUT /usr/bin/time -v -o $D/time-fold.txt $T/inner-fold --plain-leaves --partition $ST/partition.json verity/flock-circuit \
    $ST/circuit.txt $PUB $D/points.json os > $D/fold.json 2> $D/fold.err
  rc=$?; echo "fold exit $rc" | tee $D/fold.exit
  exit $rc
fi
echo "unknown MODE=$MODE" >&2; exit 2
