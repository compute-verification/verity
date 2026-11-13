#!/usr/bin/env bash
# Lean for a Lake package on a Linux machine: elan at the package's `lean-toolchain` (elan's default too, on a machine
# that has none), its dependencies (the pinned bundle from the store when the repository pins one, else
# `lake-manifest.json` through lake), Mathlib's prebuilt oleans (`lake exe cache get`) when Mathlib is among them, and
# main's cached build of the package itself (`cache.py get`, every file checked against its sha256). Idempotent.
# `LEAN_CACHE=off` skips the build cache, as the audit does. A PACKAGE_DIR that holds no package here is taken through
# the recorded layout moves (`moved.py`: a path the package had in an earlier layout is the package's path now), with a
# line saying so: the Cloud Agent environment's install command names a path the repository can't change, and a move must not
# fail its builds.
#   bash tools/verity/lean/setup.sh PACKAGE_DIR
set -euo pipefail
pkg=${1:?usage: setup.sh PACKAGE_DIR}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ ! -f "$pkg/lean-toolchain" ]; then        # no package: no directory, or one a move left with only its build
  now=$(python3 "$here/moved.py" "$pkg")
  echo "setup.sh: $pkg is $now since the layout moves"
  pkg=$(cd "$here/../../.." && pwd)/$now
fi
if ! command -v elan >/dev/null 2>&1 && [ ! -x "$HOME/.elan/bin/elan" ]; then
  curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
fi
export PATH="$HOME/.elan/bin:$PATH"
# A layout move leaves a package's untracked build directory at its former path. In this checkout, bring each along, so
# the dependencies and the build a machine already has are the moved package's.
case $(realpath --relative-to="$here/../../.." "$pkg") in
  ..*) ;;
  *) python3 "$here/moved.py" --carry || echo "setup.sh: no build directory was looked for at a package's former path" ;;
esac
cd "$pkg"
toolchain=$(tr -d '[:space:]' < lean-toolchain)
elan toolchain install "$toolchain"
# Without a default (`--default-toolchain none` above), `lean` fails outside a package; a default already chosen stays.
(cd / && lean --version >/dev/null 2>&1) || elan default "$toolchain"
lake --version
root=$(cd "$here/../../.." && pwd)
if [ ! -e .lake/packages ] && [ -f "$root/tools/verity/lean/lean_audit.py" ] && command -v uv >/dev/null 2>&1; then
  rel=$(realpath --relative-to="$root" .)
  (cd "$root" && uv run --quiet python tools/verity/lean/lean_audit.py --restore "$rel") \
    || echo "setup.sh: the pinned dependency bundle didn't restore; lake fetches the dependencies"
fi
lake env true
# `cache get` opens its downloads in parallel; under a soft limit of 1024 (some pods) part of them fail with EMFILE, and
# `lake build` then compiles those Mathlib modules, which the audit's `dependencies` check refuses.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
lake exe cache get || echo "setup.sh: no Mathlib cache fetched (Mathlib not a dependency, or the cache is unreachable); lake build compiles it"
python3 "$here/cache.py" get . || echo "setup.sh: a file from the build cache failed its check; lake build compiles those modules"
