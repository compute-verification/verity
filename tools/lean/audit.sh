#!/usr/bin/env bash
# Independent build and axiom audit of Lean packages.
#   bash tools/lean/audit.sh PKG_DIR[:CHECK_FILE] ...
# For each package: `setup.sh`, `lake build`, then `lake env lean CHECK_FILE` (default: the package's CheckAxioms.lean), then
# `replay.sh` (every built module re-checked by the kernel). Passes only when every build succeeds, no `.lean` file declares an
# `axiom`, every `#print axioms` line in the check files reports nothing beyond propext, Classical.choice and Quot.sound (no
# sorryAx), and every module replays. Hypotheses such as BCHKS25 never show in `#print axioms`. `#print axioms` alone does not see
# a declaration added with `debug.skipKernelTC`, so the replay is required, and negative-control/ (a false theorem added with the
# kernel skipped) must fail it first, at each audited toolchain.
# `lake build` runs the audited tree's code (compile-time IO: #eval, run_cmd, initialize, a Lean lakefile), which could rewrite the
# check files, these scripts or the toolchain before they run. So the scripts run from a copy outside the tree, and the check files,
# that copy and the toolchain's bin/ are hashed before the first build and verified after each package: a change fails the audit.
# A build running as the same user could still tamper with this process; isolating the check (another user or pod reading only the
# .olean files) is stronger.
set -uo pipefail
root=$(pwd)
out=${RESEARCH_RUN_DIR:-/tmp}/lean-audit
mkdir -p "$out"
export PATH="$HOME/.elan/bin:$PATH"
fail=0
controlled=" "
tools=$(mktemp -d)
cp -r "$root/tools/lean/." "$tools/"
declare -A pinned
pin() { local f; for f in "$@"; do pinned["$f"]=$(sha256sum "$f" | cut -d' ' -f1); done; }
toolchain_bin() { echo "$HOME/.elan/toolchains/$(tr -d '[:space:]' < "$1" | sed 's#/#--#; s#:#---#')/bin"; }
integrity() {   # LABEL: every pinned file still hashes as it did before the first build
  local f bad=""
  for f in "${!pinned[@]}"; do
    [ "$(sha256sum "$f" 2>/dev/null | cut -d' ' -f1)" = "${pinned[$f]}" ] || bad="$bad $f"
  done
  [ -z "$bad" ] || { echo "AUDIT integrity: FAIL after $1, changed:$bad"; return 1; }
}
while IFS= read -r f; do pin "$f"; done < <(find "$tools" -type f)
for spec in "$@"; do
  p=${spec%%:*}; c=${spec#*:}; [ "$c" = "$spec" ] && c=CheckAxioms.lean
  pin "$root/$p/$c"
done
negative_control() {   # TOOLCHAIN_FILE: the replay must reject negative-control/ at this toolchain
  local tc; tc=$(tr -d '[:space:]' < "$1")
  case "$controlled" in *" $tc "*) return 0;; esac
  local d; d=$(mktemp -d)
  cp -r "$root/tools/lean/negative-control/." "$d/" && echo "$tc" > "$d/lean-toolchain"
  bash "$root/tools/lean/setup.sh" "$d" > "$out/negative-control.setup.log" 2>&1 \
    && (cd "$d" && lake build > "$out/negative-control.build.log" 2>&1) \
    || { echo "AUDIT negative control: FAIL, it did not build at $tc (see negative-control.build.log)"; return 1; }
  if bash "$root/tools/lean/replay.sh" "$d" > "$out/negative-control.replay.log" 2>&1; then
    echo "AUDIT negative control: FAIL, the kernel replay accepted a skipKernelTC declaration at $tc"; return 1
  fi
  echo "AUDIT negative control: the kernel replay rejects a skipKernelTC declaration at $tc"
  controlled="$controlled$tc "
}
for spec in "$@"; do
  pkg=${spec%%:*}; chk=${spec#*:}; [ "$chk" = "$spec" ] && chk=CheckAxioms.lean
  tag=$(echo "$pkg" | tr '/' '_')
  echo "== $pkg ($chk)"
  bash "$root/tools/lean/setup.sh" "$root/$pkg" > "$out/$tag.setup.log" 2>&1 || { echo "AUDIT $pkg: setup failed"; fail=1; continue; }
  cd "$root/$pkg"
  s=$(date +%s)
  lake build > "$out/$tag.build.log" 2>&1; rc=$?
  echo "AUDIT $pkg: lake build rc=$rc in $(( $(date +%s) - s )) s"
  [ "$rc" = 0 ] || { tail -30 "$out/$tag.build.log"; fail=1; cd "$root"; continue; }
  if grep -rln --include='*.lean' --exclude-dir=.lake -E '^\s*axiom\s' . ; then echo "AUDIT $pkg: FAIL, an axiom is declared"; fail=1; fi
  want=$(grep -c '^#print axioms' "$chk")
  lake env lean "$chk" > "$out/$tag.axioms.txt" 2>&1; arc=$?
  [ "$arc" = 0 ] || { echo "AUDIT $pkg: FAIL, $chk did not elaborate"; tail -20 "$out/$tag.axioms.txt"; fail=1; cd "$root"; continue; }
  got=$(grep -c -E "depends on axioms|does not depend on any axioms" "$out/$tag.axioms.txt")
  [ "$got" = "$want" ] || { echo "AUDIT $pkg: FAIL, $want #print axioms lines, $got answers"; fail=1; }
  if grep -q sorryAx "$out/$tag.axioms.txt"; then echo "AUDIT $pkg: FAIL, sorryAx"; fail=1; fi
  extra=$(grep -o "depends on axioms: \[[^]]*\]" "$out/$tag.axioms.txt" | sed 's/.*\[//; s/\]//' | tr ',' '\n' | tr -d ' ' \
          | grep -v -x -E 'propext|Classical.choice|Quot.sound' | sort -u)
  [ -z "$extra" ] || { echo "AUDIT $pkg: FAIL, extra axioms: $extra"; fail=1; }
  echo "AUDIT $pkg: $want theorems audited"
  negative_control lean-toolchain || fail=1
  if bash "$root/tools/lean/replay.sh" . > "$out/$tag.replay.log" 2>&1; then
    echo "AUDIT $pkg: $(tail -n 1 "$out/$tag.replay.log" | sed 's/^replay.sh: //')"
  else
    echo "AUDIT $pkg: FAIL, $(tail -n 1 "$out/$tag.replay.log")"; fail=1
  fi
  cd "$root"
done
[ "$fail" = 0 ] && echo "AUDIT: PASS" || echo "AUDIT: FAIL"
exit $fail
