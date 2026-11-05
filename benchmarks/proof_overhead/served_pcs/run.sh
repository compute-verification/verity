#!/usr/bin/env bash
# served_pcs (README.md): a served window committed with Flock's PCS and opened at evaluation points, beside hashing the
# same bytes as hm96 rows. Builds the crate inside a Flock b684b12 checkout with this tree's patches, as
# verity/ml/flock/check_build.sh does, in a scratch directory it deletes. Run from the tree:
#   research run --on auto --cwd source -- bash benchmarks/proof_overhead/served_pcs/run.sh M="26 28 30"
# env: M, RATES, CLAIMS, ROW_BYTES (crate/src/main.rs), HASHES (the PCS leaves to run, each its own process; default
# "hm96-sha512 sha512"), RUST_TOOLCHAIN (default check_build.sh's pin)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
OUT=${RESEARCH_RUN_DIR:-$PWD}/out; mkdir -p "$OUT"
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
RUST_TOOLCHAIN=${RUST_TOOLCHAIN:-1.98.1}
export PATH="$HOME/.cargo/bin:$PATH"
command -v rustup >/dev/null \
  || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none >/dev/null
rustup toolchain list | grep -q "^$RUST_TOOLCHAIN-" || rustup toolchain install "$RUST_TOOLCHAIN" --profile minimal >/dev/null
export RUSTUP_TOOLCHAIN=$RUST_TOOLCHAIN CARGO_TARGET_DIR=$S/target
git clone -q https://github.com/succinctlabs/flock "$S/flock"
cd "$S/flock"
git checkout -q b684b12
for p in link gpu-link sha512 glue zk vtime; do
  git apply "$ROOT/verity/ml/flock/flock-$p-b684b12.patch"
done
ln -s "$ROOT/verity/ml/flock/live" crates/flock-live
cp -r "$ROOT/benchmarks/proof_overhead/served_pcs/crate" crates/served-pcs
grep -q '"crates/flock-live"' Cargo.toml || sed -i 's|    "crates/flock-transcript",|    "crates/flock-transcript",\n    "crates/flock-live",|' Cargo.toml
sed -i 's|    "crates/flock-transcript",|    "crates/flock-transcript",\n    "crates/served-pcs",|' Cargo.toml
cargo build --release -q -p served-pcs
{ lscpu | grep -E 'Model name|^CPU\(s\)'; nproc; } > "$OUT/host.txt"
: > "$OUT/served_pcs.jsonl"
first=1
for h in ${HASHES:-hm96-sha512 sha512}; do
  HASH=$h ROWS_BASELINE=$first "$CARGO_TARGET_DIR/release/served-pcs" 2> "$OUT/stderr-$h.txt" | tee -a "$OUT/served_pcs.jsonl" \
    || echo "served-pcs under $h failed: $(tail -3 "$OUT/stderr-$h.txt")"
  first=0
done
