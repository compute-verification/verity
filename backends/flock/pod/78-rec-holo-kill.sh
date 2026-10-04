#!/usr/bin/env bash
# rec-holo's kill experiment for route P (rec_holo_kill.rs): one circuit in which no unit repeats, composed from Verity's
# Boolean builders (rec_holo_netlist.py), proved two ways on upstream Flock b684b12 with rec_holo_gkr_stream.rs appended to
# flock-core's product_gkr.rs: route P (the public gate table, the wiring product GKR dense and streaming, σ's commitment and
# opening) and C-Flock's rows of the same circuit (one block, tiled COPIES times, fitted to one execution). Two forms: `full`
# (the whole circuit, base k_log BASE_KLOG) and `sampled` (UNITS proof units of 2^UNIT_LOG cells each, as drawn). Each mode is
# its own process (peak memory per process); JSON lines in out/kill-<form>.jsonl, the load before each in out/load.txt.
# Run from the verity tree (research run --cwd source --declared-output 'out/*'); builds and circuits under W, outside the run.
# env: THREADS=32 RUNS=3 FORMS="full sampled" BASE_KLOG=23 FILL=0.9 UNITS=8 UNIT_LOG=22 COPIES="8 16 32"
#      GKRS="dense:0:0 stream:1:0 stream:1:1 stream:2:0 stream:2:1 fast:1:1 fast:2:1" (how:unstored:double)
#      W=$FLOCK_WORK/rec-holo-kill-b684b12
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
REPO=$(pwd); R=${RESEARCH_RUN_DIR:-/tmp/rec-holo-kill}; O=$R/out; mkdir -p $O
W=${W:-${FLOCK_WORK:-$HOME/.cache}/rec-holo-kill-b684b12}; mkdir -p $W
S=$(mktemp -d $W/circuits.XXXXXX) || exit 1
trap 'rm -rf "$S"' EXIT
[ -x $HOME/.cargo/bin/cargo ] || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable >/dev/null
source $HOME/.cargo/env
[ -x $HOME/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
[ -x $W/py/bin/python ] || { $HOME/.local/bin/uv venv -q --python 3.12 $W/py && $HOME/.local/bin/uv pip install -q --python $W/py/bin/python numpy blake3; } || exit 1
[ -d $W/flock/.git ] || git clone -q https://github.com/succinctlabs/flock.git $W/flock || exit 1
cd $W/flock && git checkout -q -f b684b12 && git clean -qfd crates || exit 1
cp $REPO/backends/flock/pod/rec_holo_kill.rs crates/flock-prover/examples/rec_holo_kill.rs
cat $REPO/backends/flock/pod/rec_holo_gkr_stream.rs >> crates/flock-core/src/product_gkr.rs
export CARGO_TARGET_DIR=$W/target
unset RAYON_NUM_THREADS
cargo build --release -p flock-prover --example rec_holo_kill -j $(nproc) > $O/build.txt 2>&1 || { tail -30 $O/build.txt; exit 1; }
B=$CARGO_TARGET_DIR/release/examples/rec_holo_kill; T=${THREADS:-32}; export RUNS=${RUNS:-3}
PYTHONPATH=$REPO/packages/verity/src:$REPO/backends/flock/python $W/py/bin/python $REPO/backends/flock/pod/rec_holo_netlist.py \
  --out $S/net > $O/netlist.txt 2>&1 || { tail -30 $O/netlist.txt; exit 1; }
{ lscpu | grep -E 'Model name|^CPU\(s\)'; echo "nproc $(nproc) threads $T"; uptime; } | tee $O/host.txt
fail=0
run() {  # run FORM ARGS...: one process, its JSON lines appended to out/kill-FORM.jsonl
  local f=$1; shift
  echo "$f $* load $(cut -d' ' -f1-3 /proc/loadavg)" >> $O/load.txt
  RAYON_NUM_THREADS=$T $B "$@" | tee -a $O/kill-$f.jsonl
  [ ${PIPESTATUS[0]} = 0 ] || { echo "FAILED $f $*" | tee -a $O/failed.txt; fail=1; }
}
run selftest selftest
for form in ${FORMS:-full sampled}; do
  D=$S/$form
  if [ $form = full ]; then
    BASE_KLOG=${BASE_KLOG:-23} FILL=${FILL:-0.9} run $form build $D $S/net full || continue
  else
    UNITS=${UNITS:-8} UNIT_LOG=${UNIT_LOG:-22} run $form build $D $S/net sampled || continue
  fi
  [ -f $D/meta.json ] || { echo "FAILED $form build" | tee -a $O/failed.txt; fail=1; continue; }
  cp $D/meta.json $O/meta-$form.json
  run $form gate $D
  for g in ${GKRS:-dense:0:0 stream:1:0 stream:1:1 stream:2:0 stream:2:1 fast:1:1 fast:2:1}; do
    IFS=: read how unstored double <<< "$g"; run $form gkr $D $how $unstored $double
  done
  run $form sigma $D
  [ $form = full ] || run $form sigma $D ${UNIT_LOG:-22}
  for c in ${COPIES:-8 16 32}; do run $form base $D $c; done
  rm -rf $D
done
uptime | tee -a $O/host.txt
exit $fail
