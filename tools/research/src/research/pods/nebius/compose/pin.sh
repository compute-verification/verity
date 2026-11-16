#!/bin/bash
# vy-compose-pin COMMIT BRANCH < TAR: pin the tree vllm's compose verb runs from on vy-auditor-2, as root.
# TAR is infra's `git archive COMMIT` of a pushed commit, unpacked root-owned at /opt/vy-compose/COMMIT and never changed after.
# As auditor (HOME=/var/lib/vy-compose, elan through $HOME/.elan -> /opt/elan): the tree's .venv (uv sync --all-packages
# --extra torch-cpu --frozen), and the compose's Lean executables prebuilt (served_cases.py's vstar-recursive, spot-one-stage
# and inference-only, served_composite.sh's products remote, remote-v1 and onsite); every .lake stays auditor's, since
# served_composite.sh and served_cases.py still run `lake build` there. Then host.py records
# the pin with its branch in /opt/vy-compose/pins.jsonl (append-only) and /opt/vy-compose/current -> COMMIT. The verifier's
# root (/opt/vy-verifier) and its pins are untouched. Rerunning for an installed COMMIT skips the unpack and reads no TAR.
#
# `--prepare` as a third argument unpacks and builds only: no pin line, no move of current (a staging commit, built ahead).
#
#   git archive COMMIT | ssh vy-admin@vy-auditor-2 sudo vy-compose-pin COMMIT BRANCH [--prepare]
set -euo pipefail
die() { echo "vy-compose-pin: $*" >&2; exit "${2:-1}"; }
c=${1:-} br=${2:-}
[[ $c =~ ^[0-9a-f]{40}$ ]] || die "usage: vy-compose-pin COMMIT BRANCH < TAR (COMMIT: 40 hex digits)" 2
[[ $br =~ ^[A-Za-z0-9._/-]+$ ]] || die "BRANCH: $br" 2
[ "$(id -u)" = 0 ] || die "run as root" 3
id auditor > /dev/null 2>&1 || die "no auditor user"
O=/opt/vy-compose T=/opt/vy-compose/$c H=/var/lib/vy-compose
export ELAN_HOME=/opt/elan
install -d -m 0755 -o root -g root "$O"
install -d -m 0700 -o auditor -g auditor "$H"
if [ ! -d "$T" ]; then
  tmp=$(mktemp -d "$O/.pin.XXXXXX")
  tar -x -C "$tmp" --no-same-owner --no-same-permissions
  for f in verity/protocols/resource_accounting/memory/poub/attester/live_verifier/host.py benchmarks/one_stage/served_composite.sh \
           verity/protocols/proofs/spot_flock_recursion/verifier/lean/lakefile.toml uv.lock; do
    [ -f "$tmp/$f" ] || { rm -rf "$tmp"; die "the archive has no $f"; }
  done
  chown -R root:root "$tmp"
  chmod -R u+rwX,go+rX,go-w "$tmp"
  mv "$tmp" "$T"
fi
builds=("vstar-recursive verity/protocols/proofs/spot_flock_recursion/verifier/lean"
        "spot-one-stage verity/protocols/proofs/spot/verifier/lean"
        "inference-only verity/protocols/workload_classification/training_detection/inference_only/verifier/lean"
        "remote verity/protocols/compose/verifier/remote" "remote-v1 verity/protocols/compose/verifier/remote"
        "onsite verity/protocols/compose/verifier/onsite")
for b in "${builds[@]}"; do
  tc=$(tr -d '[:space:]' < "$T/${b#* }/lean-toolchain")
  [[ $tc =~ ^leanprover/lean4:v[0-9A-Za-z.-]+$ ]] || die "$T/${b#* }/lean-toolchain: $tc"
  [ -z "${tc0:-}" ] || [ "$tc" = "$tc0" ] || die "the composite's packages name two toolchains ($tc0, $tc)"
  tc0=$tc
  "$ELAN_HOME/bin/elan" toolchain install "$tc" > /dev/null
done
chmod -R go+rX,go-w "$ELAN_HOME"
# lake and lean straight from the pinned toolchain (elan's proxy would write settings.toml under root's /opt/elan): auditor's
# $HOME/.elan/bin links to its bin, so served_composite.sh's `lake build` resolves the same toolchain at compose time
bin=$ELAN_HOME/toolchains/$(printf '%s' "$tc" | sed 's|/|--|g; s|:|---|g')/bin
[ -x "$bin/lake" ] || die "no lake in $bin"
[ -L "$H/.elan" ] && rm -f "$H/.elan"
install -d -m 0755 -o auditor -g auditor "$H/.elan" "$H/.elan/bin"
for x in "$bin"/*; do ln -sfn "$x" "$H/.elan/bin/$(basename "$x")"; done
find "$T" \( -name lakefile.toml -o -name lakefile.lean \) -printf '%h\n' | while read -r d; do
  install -d -m 0755 -o auditor -g auditor "$d/.lake"
done
install -d -m 0755 -o auditor -g auditor "$T/.venv"
as_auditor() { runuser -u auditor -- env -i HOME="$H" PATH="$H/.elan/bin:/usr/local/bin:/usr/bin:/bin" UV_CACHE_DIR="$H/.cache/uv" \
  UV_PROJECT_ENVIRONMENT="$T/.venv" sh -c "$1"; }
as_auditor "cd '$T' && uv sync --all-packages --extra torch-cpu --frozen" | tail -2
for b in "${builds[@]}"; do
  as_auditor "cd '$T/${b#* }' && lake build '${b%% *}'" | tail -1
  [ -x "$T/${b#* }/.lake/build/bin/${b%% *}" ] || die "${b%% *} did not build in ${b#* }"
done
if [ "${3:-}" = --prepare ]; then
  echo "prepared (not pinned): $T built; pins.jsonl and $O/current unchanged"
  exit 0
fi
/usr/bin/python3 -I "$T/verity/protocols/resource_accounting/memory/poub/attester/live_verifier/host.py" pinned "$c" "$br"
ln -sfn "$c" "$O/current.tmp"
mv -Tf "$O/current.tmp" "$O/current"
echo "pinned: $O/current -> $c ($br)"
