#!/bin/bash
# vy-verifier-pin COMMIT < TAR: pin the tree the verifier runs from, as root. TAR is infra's `git archive` of a pushed commit (below);
# it is unpacked root-owned at /opt/vy-verifier/COMMIT, never changed after. The Lean toolchain PoUB's Lake package names
# is installed under /opt/elan (root's) and linked as COMMIT/.lean-bin; elan itself is release ELAN, its archive checked against
# ELAN_SHA256 before it runs. Each Lake package's .lake is auditor's, and `lake build
# remote-audit-verifier remote-audit-difftest` and `sampled-proofs-verifier` (what a gate run's kit builds first) run there as
# auditor, so neither a run nor a replay waits on a build. The pin (commit, UTC time) is appended to /opt/vy-verifier/pins.jsonl,
# root's and append-only (`chattr +a`, host.py `record_pin`), and every run's request names its line (RT4-INT-40). Last,
# /opt/vy-verifier/current -> COMMIT. A run runs from the tree it was requested at: host.py refuses one whose pin moved. Rerunning
# it for an installed COMMIT skips the unpack and reads no TAR, and logs the pin again.
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
ELAN=v4.2.4 ELAN_SHA256=42b94d4244e8353142c456ec0e4ca6528fd898a6c604d4059f494e706e431f63
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
  [ "$(uname -m)" = x86_64 ] || die "elan $ELAN is pinned for x86_64, not $(uname -m)"
  e=$(mktemp -d)
  curl -sSfL -o "$e/elan.tar.gz" "https://github.com/leanprover/elan/releases/download/$ELAN/elan-x86_64-unknown-linux-gnu.tar.gz"
  echo "$ELAN_SHA256  $e/elan.tar.gz" | sha256sum -c --quiet - \
    || { rm -rf "$e"; die "elan $ELAN's archive isn't the pinned one (sha256 $ELAN_SHA256)"; }
  tar -xzf "$e/elan.tar.gz" -C "$e" elan-init
  "$e/elan-init" -y --no-modify-path --default-toolchain none > /dev/null
  rm -rf "$e"
fi
"$ELAN_HOME/bin/elan" toolchain install "$tc" > /dev/null
bin=$ELAN_HOME/toolchains/$(printf '%s' "$tc" | sed 's|/|--|g; s|:|---|g')/bin
[ -x "$bin/lake" ] || die "no lake in $bin"
chmod -R go+rX,go-w "$ELAN_HOME"
ln -sfn "$bin" "$T/.lean-bin"
# The built executables stay auditor's, in auditor's .lake: only auditor ever runs them (vy-verifier@ and vy-verifier-replay@,
# and the gate's `close` through runuser), each start's `lake build` (a no-op once built) writes there, and the gate, root's,
# runs none of them and reads what they leave only to report it (host.py's gate imports nothing outside the standard library).
find "$T" \( -name lakefile.toml -o -name lakefile.lean \) -printf '%h\n' | while read -r d; do
  install -d -m 0755 -o auditor -g auditor "$d/.lake"
done
runuser -u auditor -- env -i HOME=/var/lib/vy-verifier PATH="$T/.lean-bin:/usr/bin:/bin" \
  sh -c "cd '$pkg' && lake build remote-audit-verifier remote-audit-difftest &&
         cd '$T/verity/core/service/sampled_proofs' && lake build sampled-proofs-verifier" | tail -1
/usr/bin/python3 -I "$T/benchmarks/live_verifier/host.py" pinned "$c"
ln -sfn "$c" "$O/current.tmp"
mv -Tf "$O/current.tmp" "$O/current"
echo "pinned: $O/current -> $c"
