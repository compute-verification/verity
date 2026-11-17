#!/usr/bin/env bash
# Lean for a Lake package on a Linux machine: elan at the package's `lean-toolchain` (elan's default too, on a machine
# that has none), its dependencies (the pinned bundle from the store when the repository pins one, else
# `lake-manifest.json` through lake), Mathlib's prebuilt oleans (`lake exe mathlib/cache get`) when Mathlib is among them, and
# main's cached build of the package itself (`cache.py get`, every file checked against its sha256). Idempotent.
# `LEAN_CACHE=off` skips the build cache, as the audit does. A PACKAGE_DIR that holds no package here is taken through
# the recorded layout moves (`moved.py`: a path the package had in an earlier layout is the package's path now), with a
# line saying so: the Cloud Agent environment's install command names a path the repository can't change, and a move must not
# fail its builds.
# The toolchain is a release of Lean (a tag of leanprover/lean4, `leanprover/lean4:vX.Y.Z`): elan downloads any
# other from a repository the tree names, and its `lean` and `lake` would run here. Mathlib's cache is fetched by
# Mathlib's own executable (`mathlib/cache`), never by a name the workspace looks up, which the tree could declare.
# `--outside` stops before the first command that loads the package's workspace (which runs its dependencies' lakefiles)
# and `--fetch` runs only those commands: `check.py` runs the first as itself and the second in a sandbox of its own,
# with the network, that writes only the packages' `.lake` and the dependency caches. Without either, both run here.
#   bash tools/verity/lean/setup.sh [--outside|--fetch] PACKAGE_DIR
set -euo pipefail
mode=all
case "${1:-}" in --outside|--fetch) mode=${1#--}; shift;; esac
pkg=${1:?usage: setup.sh [--outside|--fetch] PACKAGE_DIR}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ ! -f "$pkg/lean-toolchain" ]; then        # no package: no directory, or one a move left with only its build
  now=$(python3 "$here/moved.py" "$pkg")
  echo "setup.sh: $pkg is $now since the layout moves"
  pkg=$(cd "$here/../../.." && pwd)/$now
fi
toolchain=$(tr -d '[:space:]' < "$pkg/lean-toolchain")
if ! [[ $toolchain =~ ^leanprover/lean4:v[0-9][0-9A-Za-z.+-]*$ ]]; then
  echo "setup.sh: $pkg/lean-toolchain names $(printf '%q' "$toolchain"), not a release of leanprover/lean4" \
    "(leanprover/lean4:vX.Y.Z): elan would download it from wherever it names and run its lean and lake here" >&2
  exit 2
fi
export PATH="$HOME/.elan/bin:$PATH"
if [ "$mode" = fetch ]; then
  cd "$pkg"
  lake env true
  # `cache get` opens its downloads in parallel; under a soft limit of 1024 (some pods) part of them fail with EMFILE, and
  # `lake build` then compiles those Mathlib modules, which the audit's `dependencies` check refuses.
  ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
  if python3 -c 'import json, sys; sys.exit(not any(p.get("name") == "mathlib" for p in json.load(open(sys.argv[1])).get("packages", [])))' \
      lake-manifest.json 2>/dev/null; then
    lake exe mathlib/cache get || echo "setup.sh: no Mathlib cache fetched (the cache is unreachable); lake build compiles it"
  fi
  python3 "$here/cache.py" get . || echo "setup.sh: a file from the build cache failed its check; lake build compiles those modules"
  exit 0
fi
if ! command -v elan >/dev/null 2>&1 && [ ! -x "$HOME/.elan/bin/elan" ]; then
  curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
fi
# A layout move leaves a package's untracked build directory at its former path. In this checkout, bring each along, so
# the dependencies and the build a machine already has are the moved package's.
case $(realpath --relative-to="$here/../../.." "$pkg") in
  ..*) ;;
  *) python3 "$here/moved.py" --carry || echo "setup.sh: no build directory was looked for at a package's former path" ;;
esac
cd "$pkg"
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
[ "$mode" = outside ] || exec bash "$here/setup.sh" --fetch "$(pwd)"
