# Sourced by the rec-v0 jobs (80-rec-inner.sh, 87-rec-c3-open.sh, 88-rec-c3-inner.sh) after their CPU pinning: the run's
# directories ($O out/, $S scratch, $L cache), host.txt, /usr/bin/time, the Lean verifier ($LEAN_BIN, and $FIREWALL_BIN beside it, built in the background: wait $LP), the
# prover build ($B, 60-circuit.sh's, keyed by its sources) and the job's Python ($REPO's packages on PYTHONPATH).
REPO=$(pwd); O=${RESEARCH_RUN_DIR:-/tmp/rec-v0-job}/out; mkdir -p $O
[ -z "${FLOCK_WORK:-}" ] && [ ! -w /workspace ] && export FLOCK_WORK=/workspace/jobs/rec-v0
FW=${FLOCK_WORK:-/workspace}; W=$FW/flock-circuit; mkdir -p $W || exit 1
[ -w "$HOME" ] || { export HOME=$FW/home; mkdir -p $HOME; }
S=$FW/scratch/${RESEARCH_RUN_ID:-local-$$}; rm -rf $S; mkdir -p $S || exit 1
L=$FW/cache/rec-v0; mkdir -p $L || exit 1
unset CARGO_TARGET_DIR
trap 'rm -rf $S' EXIT
{ lscpu | grep -E 'Model name|^CPU\(s\)'; echo "pinned=$RI_PINNED rayon=$RAYON_NUM_THREADS"; free -g | head -2; cat /proc/loadavg; } > $O/host.txt 2>&1
[ -x /usr/bin/time ] || { apt-get install -y -qq time >/dev/null 2>&1 || { apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq time >/dev/null 2>&1; }; }
[ -x /usr/bin/time ] || { echo "no /usr/bin/time" >&2; exit 1; }

LSHA=$(python3 verity/protocols/flock/pod/lake_closure.py key verity/protocols/flock/verifier)
LEAN_BIN=$L/lean-$LSHA/flock-verify
# the inner session's send check (rec_live --lean), built beside it
FIREWALL_BIN=$L/lean-$LSHA/flock-firewall
(
  export PATH=$HOME/.elan/bin:$PATH
  command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
  exec 9>$L/lean.lock; flock 9
  if [ ! -x $LEAN_BIN ] || [ ! -x $FIREWALL_BIN ]; then
    rm -rf $L/lean-src && LEAN_SRC=$(python3 verity/protocols/flock/pod/lake_closure.py copy verity/protocols/flock/verifier $L/lean-src) \
      && (cd $LEAN_SRC && lake build flock-verify flock-firewall) && mkdir -p $L/lean-$LSHA \
      && cp $LEAN_SRC/.lake/build/bin/flock-verify $LEAN_SRC/.lake/build/bin/flock-firewall $L/lean-$LSHA/
  fi
) > $O/build-lean.log 2>&1 &
LP=$!

SM=${SM:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d .)}; SM=${SM:-89}
KEY=$(cat verity/protocols/flock/pod/20-gpu-link.sh verity/protocols/flock/pod/60-circuit.sh verity/protocols/flock/prover/*.patch verity/protocols/flock/prover/cuda_*_patch.py \
  $(find verity/protocols/flock/prover/cuda verity/protocols/flock/prover/live -type f | LC_ALL=C sort) | sha256sum | cut -c1-16)
BIN=$W/bin-$KEY-g0-sm$SM; F=$FW/flock-gpu-link/flock
exec 8>$W/build.lock; flock 8
if [ -x $BIN/flock-circuit ] && [[ $(ldd $BIN/flock-circuit 2>&1) == *"not found"* ]]; then
  rm -rf $BIN || { echo "$BIN: built against another glibc, and this job can't remove it" >&2; exit 1; }
fi
if [ ! -x $BIN/flock-circuit ]; then
  RESEARCH_RUN_DIR=$O/build FLOCK_WORK=$FW MODE=build GPU=0 SM=$SM bash verity/protocols/flock/pod/60-circuit.sh > $O/build.log 2>&1 || { tail -30 $O/build.log; exit 1; }
  mkdir -p $BIN && cp $F/target/release/flock-circuit $BIN/flock-circuit
fi
exec 8>&-
B=$BIN/flock-circuit; sha256sum $B | tee $O/binary.sha256
UV=$(command -v uv || echo $HOME/.local/bin/uv)
[ -x $UV ] || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; UV=$HOME/.local/bin/uv; }
[ -x $W/py/bin/python ] || { $UV venv -q --python 3.12 $W/py && $UV pip install -q --python $W/py/bin/python numpy blake3; } || exit 1
# Every entry must exist: importlib.metadata opens a missing one as a zip, and the suite guard counts that open as a read.
export PATH=$W/py/bin:$PATH PYTHONPATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/protocols/flock/prover/python:$REPO/adapters/vllm
