#!/usr/bin/env bash
# GUEST lane pod procedure (reports/GUEST-2026-09-09.md §2): install the pinned SP1 toolchain, build the guest
# TWICE from a clean target with DEFAULT features, pin each build (e2e/guest_build.py), build the host once, and
# measure reproducibility.  Every stage writes under $OUT so the laptop can rsync results back between stages.
#
#   bash e2e/pod_guest_build.sh /workspace/veritor-guest /workspace/guest-out
set -euo pipefail
REPO=${1:?repo}
OUT=${2:?out dir}
SP1_TAG=${SP1_TAG:-v6.4.0}          # sp1-sdk / sp1-zkvm 6.4.0 in zk/sp1/Cargo.lock; host const SP1_VERSION "6.4.0"
SRC_COMMIT=${SRC_COMMIT:-}              # the real lane-guest commit (the pod checkout is a tree import)
SRC_TREE=${SRC_TREE:-}                  # git rev-parse $SRC_COMMIT:zk/sp1 on the laptop
PIN_EXTRA=""; [ -n "$SRC_COMMIT" ] && PIN_EXTRA="--source-commit $SRC_COMMIT --expect-tree-sha $SRC_TREE"
mkdir -p "$OUT"
log() { echo "[pod_guest_build $(date -u +%H:%M:%S)] $*" | tee -a "$OUT/stages.log"; }
export PATH="$HOME/.cargo/bin:$HOME/.sp1/bin:$PATH"
export CARGO_TERM_COLOR=never

stage_toolchain() {
  log "toolchain: rustup + sp1up $SP1_TAG"
  if ! command -v cargo >/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >>"$OUT/toolchain.log" 2>&1
  fi
  export PATH="$HOME/.cargo/bin:$PATH"
  if ! command -v cargo-prove >/dev/null || ! cargo prove --version 2>/dev/null | grep -q "${SP1_TAG#v}"; then
    curl -L https://sp1up.succinct.xyz | bash >>"$OUT/toolchain.log" 2>&1
    export PATH="$HOME/.sp1/bin:$PATH"
    sp1up --version "$SP1_TAG" >>"$OUT/toolchain.log" 2>&1
  fi
  {
    echo "cargo_prove: $(cargo prove --version 2>&1 | head -1)"
    echo "rustc_succinct: $(rustup run succinct rustc --version 2>&1 | tail -1 || true)"
    echo "rustc_stable: $(rustc --version)"
    echo "cargo: $(cargo --version)"
    echo "sp1up_requested: $SP1_TAG"
    echo "uname: $(uname -a)"; echo "nproc: $(nproc)"; echo "mem: $(free -g | awk '/Mem/{print $2" GB"}')"
  } | tee "$OUT/toolchain.txt"
  # succinct toolchain versions for the pin
  rustup toolchain list 2>/dev/null | tee "$OUT/rustup-toolchains.txt" || true
}

build_guest() {  # $1 = build number
  local n=$1
  local d="$OUT/build-$n"
  mkdir -p "$d"
  log "build $n: clean target, cargo prove build (DEFAULT features)"
  rm -rf "$REPO/zk/sp1/target/elf-compilation" "$REPO/zk/sp1/guest/target" "$REPO/zk/sp1/target/riscv32im-succinct-zkvm-elf" 2>/dev/null || true
  ( cd "$REPO/zk/sp1/guest" && /usr/bin/time -v cargo prove build 2>"$d/build.stderr" >"$d/build.stdout" ) || { log "build $n FAILED"; tail -40 "$d/build.stderr"; return 1; }
  grep -E "Elapsed|Maximum resident" "$d/build.stderr" | tee "$d/time.txt"
  local elf
  elf=$(find "$REPO/zk/sp1" -path "*riscv32im-succinct-zkvm-elf/release/veritor-zk-guest" -type f | head -1)
  [ -n "$elf" ] || elf=$(find "$REPO/zk/sp1" -path "*elf-compilation*" -name veritor-zk-guest -type f | head -1)
  log "build $n: ELF $elf sha256 $(sha256sum "$elf" | cut -c1-64)"
  cp "$elf" "$d/veritor-zk-guest.elf"
  sha256sum "$d/veritor-zk-guest.elf" | tee "$d/elf.sha256"
  cargo prove vkey --elf "$elf" 2>&1 | tee "$d/vkey.txt" || true
  ( cd "$REPO" && python3 e2e/guest_build.py pin --repo "$REPO" --elf "$elf" --features "" \
      --vk-hash "$(grep -oE '0x[0-9a-f]{64}' "$d/vkey.txt" | head -1 || true)" \
      --build-argv "cd zk/sp1/guest && cargo prove build" $PIN_EXTRA --out "$d/guest_build.json" ) || true
}

build_host() {
  local d="$OUT/host"
  mkdir -p "$d"
  log "host: cargo build --release -p veritor-zk-host (build.rs rebuilds + embeds the guest with default features)"
  ( cd "$REPO/zk/sp1" && /usr/bin/time -v cargo build --release -p veritor-zk-host 2>"$d/build.stderr" >"$d/build.stdout" ) || { log "host build FAILED"; tail -40 "$d/build.stderr"; return 1; }
  grep -E "Elapsed|Maximum resident" "$d/build.stderr" | tee "$d/time.txt"
  cp "$REPO/zk/sp1/target/release/veritor-zk-host" "$d/veritor-zk-host"
  sha256sum "$d/veritor-zk-host" | tee "$d/host.sha256"
  SP1_PROVER=cpu "$d/veritor-zk-host" info | tee "$d/info.json"
  local elf
  elf=$(find "$REPO/zk/sp1" -path "*riscv32im-succinct-zkvm-elf/release/veritor-zk-guest" -type f | head -1)
  ( cd "$REPO" && python3 e2e/guest_build.py pin --repo "$REPO" --elf "$elf" --host "$d/veritor-zk-host" --features "" \
      --build-argv "cd zk/sp1 && cargo build --release -p veritor-zk-host" $PIN_EXTRA --out "$d/veritor-zk-host.guest_build.json" ) || true
}

case "${3:-all}" in
  toolchain) stage_toolchain ;;
  build1) build_guest 1 ;;
  build2) build_guest 2 ;;
  host) build_host ;;
  all) stage_toolchain; build_guest 1; build_guest 2; build_host
       log "reproducibility: $(cat "$OUT/build-1/elf.sha256" | cut -c1-64) vs $(cat "$OUT/build-2/elf.sha256" | cut -c1-64)"
       ;;
esac
log "done ${3:-all}"
