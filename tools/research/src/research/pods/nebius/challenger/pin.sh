#!/bin/bash
# vy-challenger-pin COMMIT < TAR: pin the tree the challenger runs from, as root. COMMIT must be on origin/main's first-parent
# history, which this script fetches from GitHub itself (a commits-only partial clone in /var/cache/vy-challenger/main.git), so the
# host pins only what drive landed. TAR is infra's `git archive COMMIT verity benchmarks/live_challenger benchmarks/network_traces
# integrations/vllm/manifests/checkpoints-int8.json` of that commit (the manifest pins the checkpoints `model-register` takes, and
# network_traces is the certifier rehearsal's, `certifier-rehearse`); it is
# unpacked root-owned at /opt/vy-challenger/COMMIT, never changed after. The Lean toolchain its sampled_proofs package names is
# installed under /opt/elan (root's) and linked as COMMIT/.lean-bin; elan itself is release ELAN, its archive checked against
# ELAN_SHA256 before it runs. Each Lake package's .lake is auditor's, and `lake build sampled-proofs-verifier` (and
# `network-records`, the certifier rehearsal's check) runs there as auditor, so a session's first coin doesn't wait on a build. The pin (COMMIT, the main it was checked against, UTC time) is
# appended to /opt/vy-challenger/pins.jsonl, root's and append-only (auditor.py `record_pin`, `chattr +a`), which each ledger
# open seals (RT4-INT-40). Last, /opt/vy-challenger/current -> COMMIT. A session runs from the tree it was admitted at:
# auditor.py refuses one whose pin moved. Rerunning it for an installed COMMIT skips the unpack and reads no TAR, and logs the
# pin again.
#
#   git archive COMMIT verity benchmarks/live_challenger benchmarks/network_traces integrations/vllm/manifests/checkpoints-int8.json \
#     | ssh vy-admin@HOST sudo vy-challenger-pin COMMIT
set -euo pipefail
die() { echo "vy-challenger-pin: $*" >&2; exit "${2:-1}"; }
c=${1:-}
[[ $c =~ ^[0-9a-f]{40}$ ]] || die "usage: vy-challenger-pin COMMIT < TAR (COMMIT: 40 hex digits)" 2
[ "$(id -u)" = 0 ] || die "run as root" 3
id auditor > /dev/null 2>&1 || die "no auditor user: run vy-challenger-setup first"
O=/opt/vy-challenger T=/opt/vy-challenger/$c
REPO=https://github.com/compute-verification/verity G=/var/cache/vy-challenger/main.git
ELAN=v4.2.4 ELAN_SHA256=42b94d4244e8353142c456ec0e4ca6528fd898a6c604d4059f494e706e431f63
export ELAN_HOME=/opt/elan
install -d -m 0700 -o root -g root "$(dirname "$G")"
if [ ! -d "$G" ]; then
  git clone -q --bare --filter=tree:0 --single-branch --branch main "$REPO" "$G" || die "can't clone $REPO's main"
fi
git -C "$G" fetch -q origin +refs/heads/main:refs/heads/main || die "can't fetch $REPO's main"
m=$(git -C "$G" rev-parse refs/heads/main)
git -C "$G" rev-list --first-parent refs/heads/main > "$G/first-parent"
grep -qx "$c" "$G/first-parent" || die "$c is not on origin/main's first-parent history (main is $m): pin a commit drive landed"
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
find "$T" \( -name lakefile.toml -o -name lakefile.lean \) -printf '%h\n' | while read -r d; do
  install -d -m 0755 -o auditor -g auditor "$d/.lake"
done
runuser -u auditor -- env -i HOME=/var/lib/vy-challenger PATH="$T/.lean-bin:/usr/bin:/bin" sh -c "cd '$pkg' && lake build sampled-proofs-verifier" \
  | tail -1
net=$T/verity/core/protocols/network/lean
if [ -d "$net" ] && [ -f "$T/benchmarks/network_traces/certifier_serve.py" ]; then
  runuser -u auditor -- env -i HOME=/var/lib/vy-challenger PATH="$T/.lean-bin:/usr/bin:/bin" sh -c "cd '$net' && lake build network-records" \
    | tail -1
fi
/usr/bin/python3 -I "$T/benchmarks/live_challenger/auditor.py" pinned "$c" "$m"
ln -sfn "$c" "$O/current.tmp"
mv -Tf "$O/current.tmp" "$O/current"
echo "pinned: $O/current -> $c (main $m)"
