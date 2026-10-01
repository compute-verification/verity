#!/usr/bin/env bash
# Prepare a check pod, once, so that every pod of the merge queue's pool computes the same verdict keys (`verdicts.py`).
#
#   bash tools/check/pod_setup.sh            from a source tree, on the pod: by hand, or `research run --on POD --cwd source -- bash
#                                            tools/check/pod_setup.sh`
#
# A suite's key records whether `lake`, `cargo` and `nvidia-smi` are on the PATH (`suites.py`), so a pod that gains `lake` at its
# first Lean step would key its later runs differently from its first. This puts the same tools on every pod before its first
# check: uv (UV_VERSION), elan with the Lean toolchain every Lake package pins, and a minimal Rust toolchain. It changes nothing
# already in place, and it prints what it found last, one line per tool, so two pods can be compared.
#
# It reports its setup done by writing POD_SETUP_MARKER (default ~/.research/pod-setup.json) last, and removes it first, so a
# pod reports done only once all of this has succeeded. `research run` returns once a run launches, so a chain that starts a
# check right after launching this one would race it (train TVC lost an hour that way); the merge queue's dispatcher claims
# nothing for a pod without the marker.
set -euo pipefail
UV_VERSION=${UV_VERSION:-0.12.20}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TOOLCHAIN=$(cat "$ROOT/backends/flock/verifier/lean/lean-toolchain")
export PATH="$HOME/.elan/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
MARKER=${POD_SETUP_MARKER:-$HOME/.research/pod-setup.json}
mkdir -p "$(dirname "$MARKER")"
rm -f "$MARKER"

if [ "$(uv --version 2>/dev/null | cut -d' ' -f2)" != "$UV_VERSION" ]; then
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh >/dev/null
fi
if ! command -v elan >/dev/null; then
  curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
fi
elan toolchain install "$TOOLCHAIN" >/dev/null 2>&1 || elan toolchain list | grep -q "${TOOLCHAIN#*:}"
elan default "$TOOLCHAIN" >/dev/null
if ! command -v cargo >/dev/null; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable >/dev/null
fi
for bin in /usr/local/bin "$HOME/.elan/bin" "$HOME/.cargo/bin"; do             # non-login shells (research run's) see them too
  for t in "$bin"/*; do [ -x "$t" ] && [ ! -e "/usr/local/bin/$(basename "$t")" ] && ln -s "$t" /usr/local/bin/ || true; done
done
echo "uv $(uv --version | cut -d' ' -f2)"
echo "lake $(lake --version | head -1)"
echo "cargo $(cargo --version)"
echo "nvidia-smi $(command -v nvidia-smi >/dev/null && echo present || echo absent)"
python3 -c 'import datetime, json, os, sys
doc = {"done_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "uv": sys.argv[2], "lake": sys.argv[3],
       "cargo": sys.argv[4], "toolchain": sys.argv[5]}
with open(sys.argv[1] + ".tmp", "w") as f:
    json.dump(doc, f)
os.replace(sys.argv[1] + ".tmp", sys.argv[1])' "$MARKER" "$(uv --version | cut -d' ' -f2)" "$(lake --version | head -1)" "$(cargo --version)" "$TOOLCHAIN"
echo "setup done: $MARKER"
