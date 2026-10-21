#!/bin/bash
# vy-challenger-pin COMMIT < TAR: pin the tree the challenger runs from, as root. TAR is infra's
# `git archive COMMIT verity benchmarks/live_challenger` of a pushed commit; it is unpacked root-owned at /opt/vy-challenger/COMMIT,
# never changed after. The Lean toolchain its sampled_proofs package names is installed under /opt/elan (root's) and linked as
# COMMIT/.lean-bin. Each Lake package's .lake is auditor's, and `lake build sampled-proofs-verifier` runs there as auditor, so a
# session's first coin doesn't wait on a build. Last, /opt/vy-challenger/current -> COMMIT. A session runs from the tree it was
# admitted at: auditor.py refuses one whose pin moved. Rerunning it for an installed COMMIT skips the unpack and reads no TAR.
#
#   git archive COMMIT verity benchmarks/live_challenger | ssh vy-admin@HOST sudo vy-challenger-pin COMMIT
set -euo pipefail
die() { echo "vy-challenger-pin: $*" >&2; exit "${2:-1}"; }
c=${1:-}
[[ $c =~ ^[0-9a-f]{40}$ ]] || die "usage: vy-challenger-pin COMMIT < TAR (COMMIT: 40 hex digits)" 2
[ "$(id -u)" = 0 ] || die "run as root" 3
id auditor > /dev/null 2>&1 || die "no auditor user: run vy-challenger-setup first"
O=/opt/vy-challenger T=/opt/vy-challenger/$c
export ELAN_HOME=/opt/elan
install -d -m 0755 -o root -g root "$O"
if [ ! -d "$T" ]; then
  tmp=$(mktemp -d "$O/.pin.XXXXXX")
  tar -x -C "$tmp" --no-same-owner --no-same-permissions
  [ -f "$tmp/benchmarks/live_challenger/auditor.py" ] || { rm -rf "$tmp"; die "the archive has no benchmarks/live_challenger/auditor.py"; }
  chown -R root:root "$tmp"
  chmod -R u+rwX,go+rX,go-w "$tmp"
  mv "$tmp" "$T"
fi
pkg=$T/verity/core/service/sampled_proofs
tc=$(tr -d '[:space:]' < "$pkg/lean-toolchain")
[[ $tc =~ ^leanprover/lean4:v[0-9A-Za-z.-]+$ ]] || die "$pkg/lean-toolchain: $tc"
if [ ! -x "$ELAN_HOME/bin/elan" ]; then
  curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
    | sh -s -- -y --no-modify-path --default-toolchain none > /dev/null
fi
"$ELAN_HOME/bin/elan" toolchain install "$tc" > /dev/null
bin=$ELAN_HOME/toolchains/$(printf '%s' "$tc" | sed 's|/|--|g; s|:|---|g')/bin
[ -x "$bin/lake" ] || die "no lake in $bin"
chmod -R go+rX,go-w "$ELAN_HOME"
ln -sfn "$bin" "$T/.lean-bin"
find "$T" \( -name lakefile.toml -o -name lakefile.lean \) -printf '%h\n' | while read -r d; do
  install -d -m 0755 -o auditor -g auditor "$d/.lake"
done
runuser -u auditor -- env -i HOME=/var/lib/vy-challenger PATH="$T/.lean-bin:/usr/bin:/bin" sh -c "cd '$pkg' && lake build sampled-proofs-verifier" \
  | tail -1
ln -sfn "$c" "$O/current.tmp"
mv -Tf "$O/current.tmp" "$O/current"
echo "pinned: $O/current -> $c"
