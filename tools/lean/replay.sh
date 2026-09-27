#!/usr/bin/env bash
# Kernel replay of a built Lake package: every module the package built (its own .olean files) is re-checked declaration by
# declaration by the kernel, with the toolchain's `leanchecker` (lean4checker, built at the toolchain, when an older toolchain does
# not ship it).  `#print axioms` does not see a declaration added with `debug.skipKernelTC` (it still lists only the standard
# axioms); the replay does.
#   bash tools/lean/replay.sh PACKAGE_DIR          (after `lake build` in PACKAGE_DIR; REPLAY_JOBS parallel checks, default 4)
set -euo pipefail
pkg=$(cd "${1:?usage: replay.sh PACKAGE_DIR}" && pwd)
export PATH="$HOME/.elan/bin:$PATH"
cd "$pkg"
tc=$(tr -d '[:space:]' < lean-toolchain)
ver=${tc##*:}
bin=$(lake env sh -c 'command -v leanchecker' 2>/dev/null || true)
if [ -z "$bin" ]; then
  chk="$HOME/lean4checker-$ver"
  if [ ! -x "$chk/.lake/build/bin/lean4checker" ]; then
    rm -rf "$chk"
    git clone -q --depth 1 --branch "$ver" https://github.com/leanprover/lean4checker "$chk"
    echo "$tc" > "$chk/lean-toolchain"
    (cd "$chk" && lake build lean4checker > /dev/null)
  fi
  bin="$chk/.lake/build/bin/lean4checker"
fi
lib="$pkg/.lake/build/lib/lean"
[ -d "$lib" ] || lib="$pkg/.lake/build/lib"
mods=$(cd "$lib" && find . -name '*.olean' | sed 's#^\./##; s#\.olean$##; s#/#.#g' | sort)
[ -n "$mods" ] || { echo "replay.sh: no built modules under $lib"; exit 1; }
n=$(wc -w <<<"$mods")
errs=$(mktemp -d)
# shellcheck disable=SC2016
if ! xargs -I{} -P "${REPLAY_JOBS:-4}" sh -c 'lake env "$0" "$1" > /dev/null 2> "$2/$1.err" || { echo "$1" >> "$2/FAILED"; }' \
     "$bin" "{}" "$errs" <<<"$mods" || [ -s "$errs/FAILED" ]; then
  first=$(head -n 1 "$errs/FAILED" 2>/dev/null)
  echo "replay.sh: kernel replay FAILED on $(wc -l < "$errs/FAILED" 2>/dev/null || echo '?') module(s), first $first: $(head -c 400 "$errs/$first.err" 2>/dev/null)"
  exit 1
fi
echo "replay.sh: $n module(s) replayed by the kernel ($ver, $(basename "$bin"))"
