#!/usr/bin/env bash
# The recorded agreement job (PROTOCOL.md §17.1) on a CPU pod: build both verifiers from source, then run ci.py on every
# replayable set of vectors.json.
#
#   bash verity/ml/flock/verifier/ci-pod.sh SEED=20260926 FUZZ=20 JOBS=4
#   (a job run from the tree, with ci-bundle.tar.gz sent)
#
# ci-bundle.tar.gz (made by `ci-bundle.sh`): flock/ (upstream b684b12 with PR #83's link patches, no live crate), live/<commit>/
# (PR #83's live crate at each version the sets use, with that version's vectors patch applied; `flock.patch` and `features`
# when that version's Flock needs its own patch), store/<art id prefix>/ (every stored input the sets name). The pod needs no credentials: its inputs are shipped, its outputs go through the run's custody.
set -euo pipefail
for kv in "$@"; do export "CI_$kv"; done
SRC=$(pwd)
RUN=${RESEARCH_RUN_DIR:?run under research run}
W=${CI_WORK:-$HOME/flock-ci}
mkdir -p "$W"
BUNDLE=$(ls "$RUN"/inputs/ci-bundle*.tar.gz | head -1)
export CI_BUNDLE_SHA256=$(sha256sum "$BUNDLE" | cut -d' ' -f1)
tar -xzf "$BUNDLE" -C "$W"

command -v cargo >/dev/null || { curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable >/dev/null; }
source "$HOME/.cargo/env"
command -v elan >/dev/null || { curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null; }
export PATH=$HOME/.elan/bin:$PATH

UPS=()
for c in $(ls "$W/live"); do
  rm -rf "$W/flock/crates/flock-live"
  cp -r "$W/live/$c" "$W/flock/crates/flock-live"
  bin=flock-circuit
  [ -f "$W/flock/crates/flock-live/src/bin/flock-netlist.rs" ] && bin=flock-netlist
  feat=()
  [ -f "$W/live/$c/features" ] && feat=(--features "$(cat "$W/live/$c/features")")
  [ -f "$W/live/$c/flock.patch" ] && (cd "$W/flock" && patch -s -p1 < "$W/live/$c/flock.patch")
  (cd "$W/flock" && cargo build --release -q -p flock-live --bin "$bin" "${feat[@]}" -j "$(nproc)")
  [ -f "$W/live/$c/flock.patch" ] && (cd "$W/flock" && patch -s -R -p1 < "$W/live/$c/flock.patch")
  cp "$W/flock/target/release/$bin" "$W/upstream-$c"
  UPS+=(--upstream "$c=$W/upstream-$c")
  echo "built upstream $c ($bin)"
done

(cd "$SRC/verity/core/service/flock" && lake build)
LEAN=$SRC/verity/core/service/flock/.lake/build/bin/flock-verify

SETS=()
[ -n "${CI_SETS:-}" ] && SETS=(--sets "$CI_SETS")
python3 "$SRC/verity/ml/flock/verifier/ci.py" --lean "$LEAN" "${UPS[@]}" --work "$W/work" --inputs "$W/store" \
  --jobs "${CI_JOBS:-4}" --seed "${CI_SEED:-20260926}" --fuzz "${CI_FUZZ:-20}" "${SETS[@]}" --out "$RUN"
