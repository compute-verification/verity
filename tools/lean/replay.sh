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
# A check killed by a signal (137: the cgroup's OOM killer; a module importing Mathlib takes several GB) is no kernel verdict: it is
# retried alone, and reported as a kill if it dies again.  Only a checker that exits with an error is a replay failure.
# shellcheck disable=SC2016
check='lake env "$0" "$1" > /dev/null 2> "$2/$1.err"; rc=$?; [ $rc = 0 ] || echo "$1 $rc" >> "$2/rc"'
xargs -I{} -P "${REPLAY_JOBS:-2}" sh -c "$check" "$bin" "{}" "$errs" <<<"$mods" || true
if [ -s "$errs/rc" ]; then
  mv "$errs/rc" "$errs/first"
  while read -r m rc; do
    if [ "$rc" -ge 128 ]; then sh -c "$check" "$bin" "$m" "$errs"; else echo "$m $rc" >> "$errs/rc"; fi
  done < "$errs/first"
fi
if [ -s "$errs/rc" ]; then
  read -r m rc < "$errs/rc"
  if [ "$rc" -ge 128 ]; then
    echo "replay.sh: replay KILLED (signal $((rc - 128)), e.g. out of memory: not a kernel verdict) on $(wc -l < "$errs/rc") module(s), first $m"
  else
    echo "replay.sh: kernel replay FAILED on $(wc -l < "$errs/rc") module(s), first $m (exit $rc): $(head -c 400 "$errs/$m.err")"
  fi
  exit 1
fi
echo "replay.sh: $n module(s) replayed by the kernel ($ver, $(basename "$bin"))"
