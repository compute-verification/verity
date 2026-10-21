#!/bin/bash
# vy-verifier-pin COMMIT < TAR: pin the tree the verifier runs from, as root. TAR is infra's `git archive` of a pushed commit (below);
# it is unpacked root-owned at /opt/vy-verifier/COMMIT, never changed after. The Lean toolchain the remote audit's Lake package names
# is installed under /opt/elan (root's) and linked as COMMIT/.lean-bin. Each Lake package's .lake is auditor's, and `lake build
# remote-audit-verifier remote-audit-difftest` and `sampled-proofs-verifier` (what a gate run's kit builds first) run there as
# auditor, so neither a run nor a replay waits on a build. Last,
# /opt/vy-verifier/current -> COMMIT. A run runs from the tree it was requested at: host.py refuses one whose pin moved. Rerunning
# it for an installed COMMIT skips the unpack and reads no TAR.
#
#   git archive COMMIT verity benchmarks/remote_session benchmarks/live_verifier tools/research/src/research \
#     tools/research/store.pod.toml | ssh vy-admin@HOST sudo vy-verifier-pin COMMIT
set -euo pipefail
die() { echo "vy-verifier-pin: $*" >&2; exit "${2:-1}"; }
c=${1:-}
[[ $c =~ ^[0-9a-f]{40}$ ]] || die "usage: vy-verifier-pin COMMIT < TAR (COMMIT: 40 hex digits)" 2
[ "$(id -u)" = 0 ] || die "run as root" 3
id auditor > /dev/null 2>&1 || die "no auditor user: run vy-verifier-setup first"
O=/opt/vy-verifier T=/opt/vy-verifier/$c
export ELAN_HOME=/opt/elan
install -d -m 0755 -o root -g root "$O"
if [ ! -d "$T" ]; then
  tmp=$(mktemp -d "$O/.pin.XXXXXX")
  tar -x -C "$tmp" --no-same-owner --no-same-permissions
  for f in benchmarks/live_verifier/host.py benchmarks/remote_session/verifier.py tools/research/src/research/store/cli.py; do
    [ -f "$tmp/$f" ] || { rm -rf "$tmp"; die "the archive has no $f"; }
  done
  chown -R root:root "$tmp"
  chmod -R u+rwX,go+rX,go-w "$tmp"
  mv "$tmp" "$T"
fi
pkg=$T/verity/core/protocols/remote_bandwidth/lean
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
runuser -u auditor -- env -i HOME=/var/lib/vy-verifier PATH="$T/.lean-bin:/usr/bin:/bin" \
  sh -c "cd '$pkg' && lake build remote-audit-verifier remote-audit-difftest &&
         cd '$T/verity/core/service/sampled_proofs' && lake build sampled-proofs-verifier" | tail -1
ln -sfn "$c" "$O/current.tmp"
mv -Tf "$O/current.tmp" "$O/current"
echo "pinned: $O/current -> $c"
