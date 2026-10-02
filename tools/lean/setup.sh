#!/usr/bin/env bash
# Lean for a Lake package on a Linux machine: elan at the package's `lean-toolchain`, its dependencies from `lake-manifest.json`,
# and Mathlib's prebuilt oleans (`lake exe cache get`) when Mathlib is among them. Idempotent.
#   bash tools/lean/setup.sh PACKAGE_DIR
set -euo pipefail
pkg=${1:?usage: setup.sh PACKAGE_DIR}
if ! command -v elan >/dev/null 2>&1 && [ ! -x "$HOME/.elan/bin/elan" ]; then
  curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
fi
export PATH="$HOME/.elan/bin:$PATH"
cd "$pkg"
elan toolchain install "$(tr -d '[:space:]' < lean-toolchain)"
lake --version
# `cache get` opens its downloads in parallel; under a soft limit of 1024 (some pods) part of them fail with EMFILE, and
# `lake build` then compiles those Mathlib modules, which the audit's `dependencies` check refuses.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
lake exe cache get || echo "setup.sh: no Mathlib cache fetched (Mathlib not a dependency, or the cache is unreachable); lake build compiles it"
