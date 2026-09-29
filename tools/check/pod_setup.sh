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
set -euo pipefail
UV_VERSION=${UV_VERSION:-0.12.20}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TOOLCHAIN=$(cat "$ROOT/backends/flock/verifier/lean/lean-toolchain")
export PATH="$HOME/.elan/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"

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
