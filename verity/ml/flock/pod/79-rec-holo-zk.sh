#!/usr/bin/env bash
# rec-holo: the kill experiment's circuits (78-rec-holo-kill.sh's `build`) proved as C-Flock proves them, M0 and --zk
# (rec_holo_zk.rs), on C-Flock's checkout: Flock b684b12 with the patches 20-gpu-link.sh and 60-circuit.sh apply (flock-link,
# flock-gpu-link, the live crate, sha512, glue, zk, vtime), CPU only. Per form: the gate table, σ's commitment and opening
# (and, under --zk, a re-registration), the acyclicity table, and the circuit's C-Flock rows tiled COPIES times. Each
# measurement is its own process; JSON lines in out/zk-<form>.jsonl, the load before each in out/load.txt.
# Form `cmp`: the sampled circuit built twice, without and with each block's CMP_D-bit read comparator (READ_D), their gate
# tables alternated GATE_REPS times per mode, then the per-unit acyclicity table of the comparator build at each block of
# ACYC_UNIT_KLOGS="16".
# Run from the verity tree (research run --cwd source --declared-output 'out/*'); builds and circuits under W, outside the run.
# env: THREADS=32 RUNS=2 FORMS="full sampled" BASE_KLOG=23 FILL=0.9 UNITS=8 UNIT_LOG=22 COPIES="8 16 32" MODES="m0 zk"
#      STEPS="gate sigma acyc base" (full and sampled) ACYC_FULL=1 (the full form's acyclicity table as one table, at each
#      block of ACYC_FULL_KLOGS="23"; the per-unit table's block is ACYC_KLOG, default 16) CMP_D=33 GATE_REPS=3
#      W=$FLOCK_WORK/rec-holo-zk-b684b12
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
REPO=$(pwd); R=${RESEARCH_RUN_DIR:-/tmp/rec-holo-zk}; O=$R/out; mkdir -p $O
W=${W:-${FLOCK_WORK:-$HOME/.cache}/rec-holo-zk-b684b12}; mkdir -p $W
S=$(mktemp -d $W/circuits.XXXXXX) || exit 1
trap 'rm -rf "$S"' EXIT
[ -x $HOME/.cargo/bin/cargo ] || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable >/dev/null
source $HOME/.cargo/env
[ -x $HOME/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
[ -x $W/py/bin/python ] || { $HOME/.local/bin/uv venv -q --python 3.12 $W/py && $HOME/.local/bin/uv pip install -q --python $W/py/bin/python numpy blake3; } || exit 1
unset RAYON_NUM_THREADS
# the circuit builder: upstream b684b12 with rec_holo_kill.rs (as 78-rec-holo-kill.sh builds it)
K=$W/kill; [ -d $K/.git ] || git clone -q https://github.com/succinctlabs/flock.git $K || exit 1
(cd $K && git checkout -q -f b684b12 && git clean -qfd crates \
  && cp $REPO/verity/ml/flock/pod/rec_holo_kill.rs crates/flock-prover/examples/rec_holo_kill.rs \
  && cat $REPO/verity/ml/flock/pod/rec_holo_gkr_stream.rs >> crates/flock-core/src/product_gkr.rs \
  && CARGO_TARGET_DIR=$W/target-kill cargo build --release -p flock-prover --example rec_holo_kill -j $(nproc)) > $O/build-kill.txt 2>&1 \
  || { tail -30 $O/build-kill.txt; exit 1; }
BK=$W/target-kill/release/examples/rec_holo_kill
# C-Flock's checkout (20-gpu-link.sh's and 60-circuit.sh's patch chain, GPU=0)
F=$W/cflock; [ -d $F/.git ] || git clone -q https://github.com/succinctlabs/flock.git $F || exit 1
(cd $F && git checkout -q -f b684b12 && git checkout -q -- . && git clean -qfd && rm -rf crates/flock-live cuda-ghash/prove_chunk.cuh \
  && git apply $REPO/verity/ml/flock/flock-link-b684b12.patch && git apply $REPO/verity/ml/flock/flock-gpu-link-b684b12.patch \
  && cp -r $REPO/verity/ml/flock/live crates/flock-live \
  && { grep -q '"crates/flock-live"' Cargo.toml || sed -i 's|    "crates/flock-transcript",|    "crates/flock-transcript",\n    "crates/flock-live",|' Cargo.toml; } \
  && rm -f crates/flock-merkle/src/hm96.rs && git apply $REPO/verity/ml/flock/flock-sha512-b684b12.patch \
  && git apply $REPO/verity/ml/flock/flock-glue-b684b12.patch && git apply $REPO/verity/ml/flock/flock-zk-b684b12.patch \
  && git apply $REPO/verity/ml/flock/flock-vtime-b684b12.patch \
  && mkdir -p crates/flock-live/examples && cp $REPO/verity/ml/flock/pod/rec_holo_zk.rs crates/flock-live/examples/ \
  && CARGO_TARGET_DIR=$W/target-cflock cargo build --release -p flock-live --features sha512 --example rec_holo_zk -j $(nproc)) \
  > $O/build-cflock.txt 2>&1 || { tail -30 $O/build-cflock.txt; exit 1; }
BZ=$W/target-cflock/release/examples/rec_holo_zk; T=${THREADS:-32}; export RUNS=${RUNS:-2}
sha256sum $BK $BZ | tee $O/binary.sha256
PYTHONPATH=$REPO:$REPO/verity/ml/flock/python $W/py/bin/python $REPO/verity/ml/flock/pod/rec_holo_types.py \
  --out $S/types > $O/types.txt 2>&1 || { tail -30 $O/types.txt; exit 1; }
{ lscpu | grep -E 'Model name|^CPU\(s\)'; echo "nproc $(nproc) threads $T"; free -g | head -2; uptime; } | tee $O/host.txt
fail=0
run() {  # run FORM BIN ARGS...: one process, its JSON lines appended to out/zk-FORM.jsonl
  local f=$1 b=$2; shift 2
  echo "$f $* load $(cut -d' ' -f1-3 /proc/loadavg)" >> $O/load.txt
  RAYON_NUM_THREADS=$T $b "$@" | tee -a $O/zk-$f.jsonl
  [ ${PIPESTATUS[0]} = 0 ] || { echo "FAILED $f $*" | tee -a $O/failed.txt; fail=1; }
}
has() { case " ${STEPS:-gate sigma acyc base} " in *" $1 "*) return 0 ;; esac; return 1; }
for form in ${FORMS:-full sampled}; do
  D=$S/$form
  if [ $form = cmp ]; then
    U=${UNIT_LOG:-22}
    READ_D=0 UNITS=${UNITS:-8} UNIT_LOG=$U run cmp $BK build $D-0 $S/types sampled || continue
    READ_D=${CMP_D:-33} UNITS=${UNITS:-8} UNIT_LOG=$U run cmp $BK build $D-1 $S/types sampled || continue
    cp $D-1/meta.json $O/meta-cmp.json
    for md in ${MODES:-m0 zk}; do
      for i in $(seq ${GATE_REPS:-3}); do run cmp $BZ gate $D-0 $md; run cmp $BZ gate $D-1 $md; done
    done
    for kl in ${ACYC_UNIT_KLOGS:-16}; do for md in ${MODES:-m0 zk}; do ACYC_KLOG=$kl run cmp $BZ acyc $D-1 $md $U; done; done
    rm -rf $D-0 $D-1
    continue
  fi
  if [ $form = full ]; then
    BASE_KLOG=${BASE_KLOG:-23} FILL=${FILL:-0.9} run $form $BK build $D $S/types full || continue
    U=
  else
    UNITS=${UNITS:-8} UNIT_LOG=${UNIT_LOG:-22} run $form $BK build $D $S/types sampled || continue
    U=${UNIT_LOG:-22}
  fi
  [ -f $D/meta.json ] || { echo "FAILED $form build" | tee -a $O/failed.txt; fail=1; continue; }
  cp $D/meta.json $O/meta-$form.json
  for md in ${MODES:-m0 zk}; do
    has gate && run $form $BZ gate $D $md
    has sigma && run $form $BZ sigma $D $md
    [ -n "$U" ] && { has sigma && run $form $BZ sigma $D $md $U; has acyc && run $form $BZ acyc $D $md $U; }
  done
  if has base; then
    for c in ${COPIES:-8 16 32}; do for md in ${MODES:-m0 zk}; do run $form $BZ base $D $c $md; done; done
  fi
  if [ $form = full ] && [ "${ACYC_FULL:-1}" = 1 ] && has acyc; then
    for kl in ${ACYC_FULL_KLOGS:-23}; do for md in ${MODES:-m0 zk}; do ACYC_KLOG=$kl run $form $BZ acyc $D $md; done; done
  fi
  rm -rf $D
done
uptime | tee -a $O/host.txt
exit $fail
