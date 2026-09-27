#!/usr/bin/env bash
# Upstream's verifier builds, the other side of the agreement job, built once and stored (`upstream.json` pins them):
#
#   research run --on M --project verity --source . --cwd source --send ci-bundle.tar.gz -- bash backends/flock/verifier/upstream-build.sh
#
# From the bundle (`ci-bundle.sh`) it builds `ci-pod.sh`'s upstream loop, one `upstream-<PR #83 commit>` per live version, and
# writes $RESEARCH_RUN_DIR/upstream.tar.gz (the binaries and BUILD.json: the flock commit, the Rust toolchain, the build host's
# glibc, the newest glibc symbol version the binaries need, the bundle's and each binary's sha256), packed reproducibly, and
# $RESEARCH_RUN_DIR/inputs.tar.gz (the bundle's store/, the sets' stored inputs).
#
# Build on the oldest glibc a check runs on: the binaries need the build host's symbol versions (a 24.04 build needs
# GLIBC_2.39 and cannot start on 22.04's 2.35). On RunPod: `--image runpod/base:1.3.3-ubuntu2204`.
set -euo pipefail
RUN=${RESEARCH_RUN_DIR:?run under research run}
W=${WORK:-$HOME/flock-upstream-build}
rm -rf "$W" && mkdir -p "$W/out"
BUNDLE=$(ls "$RUN"/inputs/ci-bundle*.tar.gz | head -1)
tar -xzf "$BUNDLE" -C "$W"

command -v cargo >/dev/null || { curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable >/dev/null; }
source "$HOME/.cargo/env"
QUOTA=$(awk '$1 != "max" {print int($1 / $2)}' /sys/fs/cgroup/cpu.max 2>/dev/null || true)
J=${QUOTA:-$(nproc)}

for c in $(ls "$W/live"); do
  rm -rf "$W/flock/crates/flock-live"
  cp -r "$W/live/$c" "$W/flock/crates/flock-live"
  bin=flock-circuit
  [ -f "$W/flock/crates/flock-live/src/bin/flock-netlist.rs" ] && bin=flock-netlist
  feat=()
  [ -f "$W/live/$c/features" ] && feat=(--features "$(cat "$W/live/$c/features")")
  [ -f "$W/live/$c/flock.patch" ] && (cd "$W/flock" && patch -s -p1 < "$W/live/$c/flock.patch")
  (cd "$W/flock" && cargo build --release -q -p flock-live --bin "$bin" "${feat[@]}" -j "$J")
  [ -f "$W/live/$c/flock.patch" ] && (cd "$W/flock" && patch -s -R -p1 < "$W/live/$c/flock.patch")
  cp "$W/flock/target/release/$bin" "$W/out/upstream-$c"
  echo "built upstream $c ($bin)"
done

GLIBC_NEEDS=$(for b in "$W"/out/upstream-*; do objdump -T "$b"; done | grep -o 'GLIBC_[0-9.]*' | sed 's/GLIBC_//' | sort -V | tail -1)
python3 - "$W/out" "$BUNDLE" "$(git -C "$W/flock" rev-parse HEAD 2>/dev/null || echo unknown)" "$(rustc -V)" "$(cargo -V)" \
  "$(rustc -vV | awk '/^host:/ {print $2}')" "$(ldd --version | head -1)" "$GLIBC_NEEDS" <<'EOF'
import hashlib, json, sys
from pathlib import Path
out, bundle, flock, rustc, cargo, host, build_glibc, glibc = Path(sys.argv[1]), Path(sys.argv[2]), *sys.argv[3:]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
bins = {p.name.removeprefix("upstream-"): sha(p) for p in sorted(out.glob("upstream-*"))}
doc = {"flock": flock, "rustc": rustc, "cargo": cargo, "host": host, "build_glibc": build_glibc, "glibc": glibc,
       "bundle_sha256": sha(bundle), "binaries": bins}
(out / "BUILD.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
print(json.dumps(doc))
EOF
pack() { tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -cf - -C "$1" . | gzip -n > "$2"; }
pack "$W/out" "$RUN/upstream.tar.gz"
pack "$W/store" "$RUN/inputs.tar.gz"
sha256sum "$RUN/upstream.tar.gz" "$RUN/inputs.tar.gz"
