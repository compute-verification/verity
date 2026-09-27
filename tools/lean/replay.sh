#!/usr/bin/env bash
# Kernel replay of a built Lake package: every module the package built (its own .olean files) is re-checked declaration by
# declaration by the kernel, with lean4checker at the package's toolchain.  `#print axioms` does not see a declaration added with
# `debug.skipKernelTC` (it still lists only the standard axioms); the replay does.
#   bash tools/lean/replay.sh PACKAGE_DIR          (after `lake build` in PACKAGE_DIR)
set -euo pipefail
pkg=$(cd "${1:?usage: replay.sh PACKAGE_DIR}" && pwd)
export PATH="$HOME/.elan/bin:$PATH"
tc=$(tr -d '[:space:]' < "$pkg/lean-toolchain")
ver=${tc##*:}
chk="$HOME/lean4checker-$ver"
if [ ! -x "$chk/.lake/build/bin/lean4checker" ]; then
  rm -rf "$chk"
  git clone -q --depth 1 --branch "$ver" https://github.com/leanprover/lean4checker "$chk"
  echo "$tc" > "$chk/lean-toolchain"
  (cd "$chk" && lake build lean4checker > /dev/null)
fi
bin="$chk/.lake/build/bin/lean4checker"
lib="$pkg/.lake/build/lib/lean"
[ -d "$lib" ] || lib="$pkg/.lake/build/lib"
mods=$(cd "$lib" && find . -name '*.olean' | sed 's#^\./##; s#\.olean$##; s#/#.#g' | sort)
[ -n "$mods" ] || { echo "replay.sh: no built modules under $lib"; exit 1; }
n=0
cd "$pkg"
for m in $mods; do
  lake env "$bin" "$m" > /dev/null 2> "/tmp/replay-$m.err" || { echo "replay.sh: kernel replay FAILED on $m: $(head -c 400 "/tmp/replay-$m.err")"; exit 1; }
  n=$((n + 1))
done
echo "replay.sh: $n module(s) replayed by the kernel ($ver)"
