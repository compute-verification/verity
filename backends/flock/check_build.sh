#!/usr/bin/env bash
# check's flock-circuit build step (tools/check/check.py): a Flock b684b12 checkout with the Rust patches the pods apply
# (pod/20-gpu-link.sh, pod/60-circuit.sh; no CUDA), this tree's live crate linked in, then `cargo check` of flock-circuit in each
# CPU feature set a pod builds: the class sweep's proving build (sha512), 60-circuit's (sha512, glue), and the selftests'
# (sha512, glue, seed-injection). A proving build never has seed-injection, so code only that feature defines must stay behind it.
# env: FLOCK_CHECKOUT (default $HOME/.cache/verity/flock-b684b12, cloned when missing; its target/ is kept between runs)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
F=${FLOCK_CHECKOUT:-$HOME/.cache/verity/flock-b684b12}
command -v cargo >/dev/null || [ -x "$HOME/.cargo/bin/cargo" ] \
  || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable >/dev/null
export PATH="$HOME/.cargo/bin:$PATH"
mkdir -p "$(dirname "$F")"
exec 9>"$F.lock"
flock 9
[ -d "$F/.git" ] || git clone -q https://github.com/succinctlabs/flock "$F"
cd "$F"
git checkout -q b684b12 && git checkout -q -- . && git clean -qfd -e target
for p in link gpu-link sha512 glue zk; do
  git apply "$ROOT/backends/flock/flock-$p-b684b12.patch"
done
ln -s "$ROOT/backends/flock/live" crates/flock-live
grep -q '"crates/flock-live"' Cargo.toml || sed -i 's|    "crates/flock-transcript",|    "crates/flock-transcript",\n    "crates/flock-live",|' Cargo.toml
for features in sha512 sha512,glue sha512,glue,seed-injection; do
  echo "check_build: flock-circuit --features $features"
  cargo check -q -p flock-live --bin flock-circuit --features "$features"
done
