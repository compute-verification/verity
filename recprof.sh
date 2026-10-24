#!/usr/bin/env bash
# RECFAST: pod 85's recursive path on the CPU prover (GPU=0) at a small shape, every stage under /usr/bin/time:
# build (staging the inner statement), inner (the proxied M0 session, its replay and V*'s hashing reference, then the same
# statement's M0 and --zk on loopback), vstage (fold points, InnerFold, every V* statement), vcap, oprove (each V* statement
# --zk on loopback), lean (flock-verify verify --zk with --rec-inner-session on each). Run from the source root (--cwd source).
# Prebuilt binaries: hidden_zk's cache on the node (same patches and features as 60-circuit.sh's GPU=0 build), keyed as it keys
# them; InnerFold is built on a copy of the cached Lean tree, .lake included.
# env: K=64 N=64 CPUS=32 PCPUS=16 STEPS="build inner vstage vcap oprove lean" PERF= (space-separated steps run under perf)
set -uo pipefail
SRC=$(pwd)
OUT=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$OUT"
export FLOCK_WORK=${FLOCK_WORK:-/workspace/research/cache/recfast}
K=${K:-64} N=${N:-64} NC=${CPUS:-32}
export TAG=${TAG:-k$K-n$N}
mkdir -p $FLOCK_WORK
research keep $FLOCK_WORK --owner @proofs --expires 2026-10-31 --priority normal --why "RECFAST recursion profile builds and data" >/dev/null 2>&1 || true

if [ -z "${RECPROF_PINNED:-}" ]; then
  # the NC least busy cores of this job's affinity over 2 s
  C=$(python3 - "$NC" <<'EOF'
import os, sys, time
mine = sorted(os.sched_getaffinity(0))
def snap():
    return {int(f[0][3:]): list(map(int, f[1:])) for f in (l.split() for l in open("/proc/stat")) if f[0].startswith("cpu") and f[0] != "cpu"}
a = snap(); time.sleep(2); b = snap()
busy = {c: 1 - (b[c][3] + b[c][4] - a[c][3] - a[c][4]) / max(1, sum(b[c]) - sum(a[c])) for c in mine}
print(",".join(map(str, sorted(sorted(mine, key=lambda c: busy[c])[:int(sys.argv[1])]))))
EOF
)
  echo "pinned $C" | tee "$OUT/pinned.txt"
  RECPROF_PINNED=$C exec taskset -c $C bash "$0" "$@"
fi

HZ=/workspace/research/cache/pouw-hidden-zk
tk() { find "$@" -type f -not -path '*/.lake/*' -not -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-16; }
LK=$(tk verity/core/service/flock) FK=$(tk verity/ml/flock/live verity/ml/flock/*.patch)
KEY=$(cat verity/ml/flock/pod/20-gpu-link.sh verity/ml/flock/pod/60-circuit.sh verity/ml/flock/*.patch verity/ml/flock/cuda_*_patch.py \
  $(find verity/ml/flock/cuda verity/ml/flock/live -type f ! -name '*.md' | LC_ALL=C sort) | sha256sum | cut -c1-16)
LSHA=$(cd verity/core/service/flock && find . -name .lake -prune -o -type f -print | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)
BIN=$FLOCK_WORK/flock-circuit/bin-$KEY-g0-sm120; L=$FLOCK_WORK/cache
echo "lean key $LK flock key $FK pod key $KEY lsha $LSHA" | tee "$OUT/keys.txt"
export PATH=$HOME/.elan/bin:$PATH
if [ ! -x $BIN/flock-circuit ] && [ -e $HZ/flock-$FK/.done ]; then mkdir -p $BIN && cp $HZ/flock-$FK/flock-circuit-prove $BIN/flock-circuit; fi
if [ ! -x $L/lean-$LSHA/flock-verify ] && [ -e $HZ/lean-$LK/.done ]; then
  mkdir -p $L/lean-$LSHA && cp $HZ/lean-$LK/.lake/build/bin/flock-verify $L/lean-$LSHA/flock-verify
fi
FSHA=$( (echo $LSHA; cat verity/ml/flock/tests/lean/InnerFold.lean) | sha256sum | cut -c1-16)
if [ ! -x $L/fold-$FSHA/inner-fold ] && [ -e $HZ/lean-$LK/.done ]; then
  T0=$(date +%s)
  C=$L/fold-src-$FSHA; rm -rf $C && mkdir -p $C && cp -a $HZ/lean-$LK/. $C/ && rm -f $C/.done && cp verity/ml/flock/tests/lean/InnerFold.lean $C/ \
    && printf '\n[[lean_exe]]\nname = "inner-fold"\nroot = "InnerFold"\n' >> $C/lakefile.toml \
    && (cd $C && lake build inner-fold) > "$OUT/build-fold.log" 2>&1 && mkdir -p $L/fold-$FSHA && cp $C/.lake/build/bin/inner-fold $L/fold-$FSHA/inner-fold
  echo "inner-fold build $(( $(date +%s) - T0 )) s, exit $?" | tee -a "$OUT/keys.txt"; tail -3 "$OUT/build-fold.log"
fi
ls -la $BIN $L/lean-$LSHA $L/fold-$FSHA 2>&1 | tee -a "$OUT/keys.txt"

PDIR=$(mktemp -d /workspace/research/cache/recfast/perf.XXXXXX)
trap 'rm -rf "$PDIR"' EXIT
rc=0
# the proxy writes s<k> only for two sessions or more, so the inner step runs one warm and one timed session and every later
# step reads the timed one (INNER); the outer statements one session each
INNER=$FLOCK_WORK/data/$TAG/inner/s1
for s in ${STEPS:-build inner vstage vcap oprove lean}; do
  t0=$(date +%s.%N)
  echo "STEP $s start $t0 load $(cut -d' ' -f1-3 /proc/loadavg)" | tee -a "$OUT/steps.txt"
  ARGS=(STEP=$s GPU=0 K=$K N=$N FORGE= NEG_INNER= PCPUS=${PCPUS:-16} JOBS=${JOBS:-4})
  if [ $s = inner ]; then ARGS+=(WARM=1 RUNS=1); else ARGS+=(WARM=0 RUNS=1 INNER=$INNER); fi
  PRE=(); [[ " ${PERF:-} " == *" $s "* ]] && PRE=(perf record -F ${PERF_FREQ:-5} -g -o $PDIR/perf-$s.data --)
  "${PRE[@]}" /usr/bin/time -v -o "$OUT/time-step-$s.txt" bash verity/ml/flock/pod/85-rec-reprice.sh "${ARGS[@]}" > "$OUT/step-$s.log" 2>&1
  r=$?
  echo "STEP $s exit $r wall $(python3 -c "import sys,time; print(round(time.time()-float(sys.argv[1]),1))" $t0)" | tee -a "$OUT/steps.txt"
  tail -15 "$OUT/step-$s.log"
  if [ -s $PDIR/perf-$s.data ]; then
    perf report -i $PDIR/perf-$s.data --no-children --sort comm,dso,symbol --stdio -g none 2>/dev/null | head -150 > "$OUT/perf-$s-flat.txt"
    perf report -i $PDIR/perf-$s.data --children --sort comm,symbol --stdio -g none 2>/dev/null | head -250 > "$OUT/perf-$s-children.txt"
    perf report -i $PDIR/perf-$s.data --no-children --sort comm --stdio -g none 2>/dev/null | head -60 > "$OUT/perf-$s-comm.txt"
  fi
  [ $r = 0 ] || { rc=$r; break; }
done
exit $rc
