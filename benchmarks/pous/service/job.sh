#!/usr/bin/env bash
# PoUS through the proof service (service/driver.py) with C-Flock's prover and the Lean verifier of record, on a node's CPUs:
#   research run --on NODE --project verity --cwd source --campaign pous --declared-output 'out/*' \
#     -- bash benchmarks/pous/service/job.sh [K=4] [BPS=2] [SEGMENTS=1] [SCHEME=p2-16448/v3] [SIGNER=lms LMS_KEY=DIR] [FROM_AUDIT=F]
# Builds flock-circuit (Flock b684b12 and backends/flock's patches: link, gpu-link, sha512, glue, zk, vtime; features sha512,
# glue), the Lean flock-verify and a venv with numpy, once per input tree under $SVC_CACHE (kept), then runs the driver:
# the CPU setup and a logical-clock session (default), or FROM_AUDIT, a GPU harness's service-input.json. Results go to
# $RESEARCH_RUN_DIR/out (verdict.json, summary.json, witness/), scratch to a mktemp directory the job removes.
# SIGNER=lms signs with S1's key in LMS_KEY (made there with $(nproc) keygen workers when absent).
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
SRC=$(pwd)
OUT=${RESEARCH_RUN_DIR:?}/out
mkdir -p "$OUT"
mkdir -p "${TMPDIR:-/tmp}" && W=$(mktemp -d "${TMPDIR:-/tmp}/pous-service.XXXXXX")
trap 'rm -rf "$W"' EXIT
CACHE=${SVC_CACHE:-/workspace/pouw/fill-out/pous/work/service-cache}
TH=$(nproc)
export RAYON_NUM_THREADS=$TH
log() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$OUT/job.log"; }

tree_key() { (cd "$SRC" && find "$@" -type f -not -path '*/.lake/*' -not -path '*/__pycache__/*' -print0 | sort -z \
  | xargs -0 sha256sum | sha256sum | cut -c1-16); }
build_circuit() {   # DEST/flock-circuit
  command -v cargo >/dev/null || [ -x "$HOME/.cargo/bin/cargo" ] || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null
  export PATH="$HOME/.cargo/bin:$PATH"
  local D=$1 F=$W/flock
  rm -rf "$F" "$D.tmp"
  git clone -q https://github.com/succinctlabs/flock "$F" && cd "$F" && git checkout -q b684b12 || return 1
  for p in link gpu-link; do git apply "$SRC/backends/flock/flock-$p-b684b12.patch" || return 1; done
  cp -r "$SRC/backends/flock/live" crates/flock-live
  grep -q '"crates/flock-live"' Cargo.toml || sed -i 's|    "crates/flock-transcript",|    "crates/flock-transcript",\n    "crates/flock-live",|' Cargo.toml
  rm -f crates/flock-merkle/src/hm96.rs
  for p in sha512 glue zk vtime; do git apply "$SRC/backends/flock/flock-$p-b684b12.patch" || return 1; done
  mkdir -p "$D.tmp"
  CARGO_TARGET_DIR=$W/target cargo build -q --release -p flock-live --features sha512,glue --bin flock-circuit -j "$TH" \
    && cp "$W/target/release/flock-circuit" "$D.tmp/flock-circuit" || return 1
  rm -rf "$D" && mv "$D.tmp" "$D"
}
build_lean() {      # DEST/.lake/build/bin/flock-verify
  command -v lake >/dev/null || [ -x "$HOME/.elan/bin/lake" ] || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
    | sh -s -- -y --default-toolchain none >/dev/null
  export PATH="$HOME/.elan/bin:$PATH"
  local D=$1
  rm -rf "$D.tmp" && mkdir -p "$D.tmp" && (cd "$SRC/backends/flock/verifier/lean" && tar --exclude=.lake -cf - .) | tar -xf - -C "$D.tmp" \
    && (cd "$D.tmp" && lake build flock-verify) && rm -rf "$D" && mv "$D.tmp" "$D"
}
build_python() {    # DEST: a venv
  command -v uv >/dev/null || [ -x "$HOME/.local/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
  export PATH="$HOME/.local/bin:$PATH"
  rm -rf "$1" && uv venv -q --python 3.12 "$1" && uv pip install -q --python "$1/bin/python" numpy==2.5.3 blake3==1.0.9
}
cached() {          # cached NAME KEY BUILDER: the build's directory, built once per key under the cache's lock
  local name=$1 key=$2 builder=$3
  mkdir -p "$CACHE" && local D=$CACHE/$name-$key
  ( flock 9; [ -e "$D/.done" ] || { $builder "$D" > "$OUT/build-$name.log" 2>&1 && touch "$D/.done"; } ) 9> "$CACHE/.lock-$name-$key"
  echo "$D"
}

log "pous service: source ${RESEARCH_SOURCE_SHA:-$(git -C "$SRC" rev-parse HEAD 2>/dev/null)}, $TH CPUs, cache $CACHE"
{ lscpu | grep -E 'Model name|^CPU\(s\)'; taskset -pc $$; grep -E 'MemTotal|MemAvailable' /proc/meminfo; } > "$OUT/host.txt" 2>&1
cached flock "$(tree_key backends/flock/live backends/flock/*.patch)" build_circuit > "$W/flock.dir" &
cached lean "$(tree_key backends/flock/verifier/lean)" build_lean > "$W/lean.dir" &
cached venv numpy2.5.3-blake3-1.0.9 build_python > "$W/venv.dir" &
wait
BIN=$(cat "$W/flock.dir")/flock-circuit LEAN=$(cat "$W/lean.dir")/.lake/build/bin/flock-verify PY=$(cat "$W/venv.dir")/bin/python
for p in "$BIN" "$LEAN" "$PY"; do
  [ -x "$p" ] || { log "build FAILED: $p"; for f in "$OUT"/build-*.log; do echo "== $f"; tail -30 "$f"; done | tee -a "$OUT/job.log"; exit 1; }
done
sha256sum "$BIN" "$LEAN" | tee "$OUT/binaries.sha256" | tee -a "$OUT/job.log"
log "built"

export PYTHONPATH=$SRC:$SRC/backends/flock/python:$SRC/catalog
ARGS=(--out "$OUT" --flock "$BIN" --lean "$LEAN")
if [ -n "${FROM_AUDIT:-}" ]; then ARGS+=(--from-audit "$FROM_AUDIT")
else ARGS+=(--scheme "${SCHEME:-p2-16448/v3}" --bps "${BPS:-2}" --segments "${SEGMENTS:-1}" --k "${K:-4}"); fi
[ "${SIGNER:-test}" = lms ] && ARGS+=(--signer lms --lms-key "${LMS_KEY:?SIGNER=lms takes LMS_KEY=DIR}" --keygen-workers "$TH")
[ "${ZK:-1}" = 0 ] && ARGS+=(--no-zk)
log "driver: ${ARGS[*]}"
"$PY" "$SRC/benchmarks/pous/service/driver.py" "${ARGS[@]}" 2>&1 | tee "$OUT/driver.log"
rc=${PIPESTATUS[0]}
log "driver exit $rc: $(cat "$OUT/verdict.json" 2>/dev/null)"
exit "$rc"
