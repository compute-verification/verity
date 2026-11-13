#!/usr/bin/env bash
# check's flock-circuit build step (tools/verity/check/check.py): a Flock b684b12 checkout with the Rust patches the pods apply
# (../pod/20-gpu-link.sh, ../pod/60-circuit.sh; no CUDA), this tree's live crate linked in, then `cargo check` of flock-circuit in each
# CPU feature set a pod builds: the class sweep's proving build (sha512), 60-circuit's (sha512, glue), and the selftests'
# (sha512, glue, seed-injection). A proving build never has seed-injection, so code only that feature defines must stay behind it.
# `check_build.sh test` (check's flock-rust-tests step) runs the Rust tests on the same checkout instead, in release as upstream's
# CI does: flock-live's unit tests in the proving build's and the selftests' feature sets, and every test of the six crates the
# patches change (the live crate's features don't reach them, so one run).
# `check_build.sh bin DEST` builds the proving binary on the same checkout instead, as ../pod/60-circuit.sh builds it on a CPU
# (release, sha512 and glue), and copies it to DEST while it holds the checkout: a test that proves runs its own copy, which
# no later build of another tree replaces. A test's build fetches nothing: its cargo runs offline, and it exits 3 where this
# machine has no checkout, not the toolchain or not every crate of this tree's build, which the steps above leave.
# env: FLOCK_CHECKOUT (default $HOME/.cache/verity/flock-b684b12, cloned when missing; its target/ is kept between runs),
# RUST_TOOLCHAIN (default the pin below, installed through rustup when missing: a machine's default toolchain doesn't decide it,
# and an agent VM's, 1.83, can't parse upstream's edition-2024 manifests)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
F=${FLOCK_CHECKOUT:-$HOME/.cache/verity/flock-b684b12}
RUST_TOOLCHAIN=${RUST_TOOLCHAIN:-1.98.1}
export PATH="$HOME/.cargo/bin:$PATH"
mkdir -p "$(dirname "$F")"
exec 9>"$F.lock"
flock 9
if [ "${1:-}" = bin ]; then
  toolchains=$(command -v rustup >/dev/null && rustup toolchain list || true)
  if ! grep -q "^$RUST_TOOLCHAIN-" <<<"$toolchains" || [ ! -d "$F/.git" ]; then
    echo "check_build.sh bin: this machine has no Rust $RUST_TOOLCHAIN under rustup or no Flock checkout at $F, and a test's" \
      "build fetches neither: check's flock-circuit-build step leaves both" >&2
    exit 3
  fi
fi
command -v rustup >/dev/null \
  || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none >/dev/null
rustup toolchain list | grep -q "^$RUST_TOOLCHAIN-" || rustup toolchain install "$RUST_TOOLCHAIN" --profile minimal >/dev/null
export RUSTUP_TOOLCHAIN=$RUST_TOOLCHAIN
[ -d "$F/.git" ] || git clone -q https://github.com/succinctlabs/flock "$F"
cd "$F"
git checkout -q b684b12 && git checkout -q -- . && git clean -qfd -e target
for p in link gpu-link sha512 glue zk vtime; do
  git apply "$ROOT/verity/protocols/flock/prover/flock-$p-b684b12.patch"
done
ln -s "$ROOT/verity/protocols/flock/prover/live" crates/flock-live
grep -q '"crates/flock-live"' Cargo.toml || sed -i 's|    "crates/flock-transcript",|    "crates/flock-transcript",\n    "crates/flock-live",|' Cargo.toml
if [ "${1:-}" = test ]; then
  # --lib: the live crate's bins hold no tests, and flock-pure-gpu and flock-gpu-link predate the SHA-512 patch
  # sha512 alone is the proving build: its refusals of a fixed seed only mean something without seed-injection
  for features in sha512 sha512,glue,seed-injection; do
    echo "check_build: cargo test --release flock-live --lib --features $features"
    cargo test --release -q -p flock-live --lib --features "$features"
  done
  echo "check_build: cargo test --release flock-core flock-parallel flock-prover flock-merkle flock-transcript flock-hash"
  # take_reuses_given_buffer flakes: the scratch pool is process-global, so another module's test running in parallel can
  # take the buffer it gave back
  cargo test --release -q -p flock-core -p flock-parallel -p flock-prover -p flock-merkle -p flock-transcript -p flock-hash \
    -- --skip scratch::tests::take_reuses_given_buffer
  exit
fi
if [ "${1:-}" = bin ]; then
  echo "check_build: cargo build --release flock-circuit --features sha512,glue"
  if ! cargo tree --offline -p flock-live --features sha512,glue >/dev/null 2>&1; then
    echo "check_build.sh bin: this machine holds not every crate of this tree's build, and a test's build fetches none:" \
      "check's flock-circuit-build step fetches them" >&2
    exit 3
  fi
  cargo build --release --offline -q -p flock-live --bin flock-circuit --features sha512,glue
  cp target/release/flock-circuit "${2:?check_build.sh bin DEST}"
  exit
fi
for features in sha512 sha512,glue sha512,glue,seed-injection; do
  echo "check_build: flock-circuit --features $features"
  cargo check -q -p flock-live --bin flock-circuit --features "$features"
done
