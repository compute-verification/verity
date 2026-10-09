#!/usr/bin/env bash
# rec-holo: pricing kernels for a hidden-circuit argument (rec_holo_bench.rs: F128 multiply rates, the wiring product GKR per
# cell, a degree-3 sumcheck per term) beside upstream's own per-row baseline on the same machine: the SHA-256 R1CS prover
# (sha2_proof, m = 15 + SHA2_LOG2S, per-phase times under PCS_TRACE=1) and the PCS commit (pcs_commit). All on upstream
# Flock b684b12 unpatched. Run from the verity tree (research run --cwd source); results in out/, builds under W.
# env: THREADS=32  GKR_MUS="20 22 24"  SC_MUS="20 22 24"  MUS_1T="20 22"  SHA2_LOG2S=12  COMMIT_M="27 30"  RUNS=3
#      LC_CASES="20:11 20:17 24:11 24:21" (m:k_log, the lincheck kernel; k_log ≤ m − 3)  LC_D=15  CB_KLOGS="21 24" CB_D=15 (the comb gather)
#      UPSTREAM=1 (0: skip sha2_proof and pcs_commit)  W=$FLOCK_WORK/rec-holo-b684b12 (default under $HOME/.cache)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
REPO=$(pwd); R=${RESEARCH_RUN_DIR:-/tmp/rec-holo}; O=$R/out; mkdir -p $O
W=${W:-${FLOCK_WORK:-$HOME/.cache}/rec-holo-b684b12}; mkdir -p $W
[ -x $HOME/.cargo/bin/cargo ] || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable >/dev/null
source $HOME/.cargo/env
[ -d $W/flock/.git ] || git clone -q https://github.com/succinctlabs/flock.git $W/flock || exit 1
cd $W/flock && git checkout -q -f b684b12 && git clean -qfd crates || exit 1
cp $REPO/verity/ml/flock/pod/rec_holo_bench.rs crates/flock-prover/examples/rec_holo_bench.rs
export CARGO_TARGET_DIR=$W/target
unset RAYON_NUM_THREADS
cargo build --release -p flock-prover --example rec_holo_bench -j $(nproc) > $O/build.txt 2>&1 || { tail -30 $O/build.txt; exit 1; }
T=${THREADS:-32}; B=$CARGO_TARGET_DIR/release/examples/rec_holo_bench
{ lscpu | grep -E 'Model name|^CPU\(s\)'; echo "nproc $(nproc) threads $T"; uptime; } | tee $O/host.txt
RAYON_NUM_THREADS=$T RUNS=${RUNS:-3} LC_CASES="${LC_CASES:-20:11 20:17 24:11 24:21}" $B | tee $O/kernels-${T}t.jsonl || exit 1
RAYON_NUM_THREADS=1 RUNS=1 GKR_MUS="${MUS_1T:-20 22}" SC_MUS="${MUS_1T:-20 22}" LC_CASES= CB_KLOGS=21 $B | tee $O/kernels-1t.jsonl || exit 1
[ "${UPSTREAM:-1}" = 1 ] || { uptime | tee -a $O/host.txt; exit 0; }
RAYON_NUM_THREADS=$T PCS_TRACE=1 SHA2_LOG2S="${SHA2_LOG2S:-12}" cargo bench -p flock-prover --bench sha2_proof > $O/sha2_proof.txt 2>&1 \
  || { tail -30 $O/sha2_proof.txt; exit 1; }
RAYON_NUM_THREADS=$T FLOCK_COMMIT_M="${COMMIT_M:-27 30}" cargo bench -p flock-prover --bench pcs_commit > $O/pcs_commit.txt 2>&1 \
  || { tail -30 $O/pcs_commit.txt; exit 1; }
uptime | tee -a $O/host.txt
