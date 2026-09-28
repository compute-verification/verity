#!/usr/bin/env bash
# Lean for a Lake package on a Linux machine: elan at the package's `lean-toolchain` (elan's default too, on a machine
# that has none), its dependencies (the pinned bundle from the store when the repository pins one, else
# `lake-manifest.json` through lake), Mathlib's prebuilt oleans (`lake exe cache get`) when Mathlib is among them, and
# main's cached build of the package itself (`cache.py get`, every file checked against its sha256). Idempotent.
# `LEAN_CACHE=off` skips the build cache, as the audit does.
#   bash tools/verity/lean/setup.sh PACKAGE_DIR
set -euo pipefail
pkg=${1:?usage: setup.sh PACKAGE_DIR}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if ! command -v elan >/dev/null 2>&1 && [ ! -x "$HOME/.elan/bin/elan" ]; then
  curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
fi
export PATH="$HOME/.elan/bin:$PATH"
cd "$pkg"
toolchain=$(tr -d '[:space:]' < lean-toolchain)
elan toolchain install "$toolchain"
# Without a default (`--default-toolchain none` above), `lean` fails outside a package; a default already chosen stays.
(cd / && lean --version >/dev/null 2>&1) || elan default "$toolchain"
lake --version
root=$(git rev-parse --show-toplevel 2>/dev/null || true)
if [ ! -e .lake/packages ] && [ -n "$root" ] && [ -f "$root/tools/verity/lean/lean_audit.py" ] && command -v uv >/dev/null 2>&1; then
  (cd "$root" && uv run --quiet python tools/verity/lean/lean_audit.py --restore "$(realpath --relative-to="$root" .)") \
    || echo "setup.sh: the pinned dependency bundle didn't restore; lake fetches the dependencies"
fi
lake env true
# `cache get` opens its downloads in parallel; under a soft limit of 1024 (some pods) part of them fail with EMFILE, and
# `lake build` then compiles those Mathlib modules, which the audit's `dependencies` check refuses.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
lake exe cache get || echo "setup.sh: no Mathlib cache fetched (Mathlib not a dependency, or the cache is unreachable); lake build compiles it"
python3 "$here/cache.py" get . || echo "setup.sh: a file from the build cache failed its check; lake build compiles those modules"
