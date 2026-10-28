#!/usr/bin/env bash
# Prepare a check pod, once, so that every pod of the merge queue's pool computes the same verdict keys (`verdicts.py`).
#
#   bash tools/verity/check/pod_setup.sh            from a source tree, on the pod: by hand, or `research run --on POD --cwd source -- bash
#                                            tools/verity/check/pod_setup.sh`
#
# A suite's key records whether `lake`, `cargo` and `nvidia-smi` are on the PATH (`suites.py`), so a pod that gains `lake` at its
# first Lean step would key its later runs differently from its first. This puts the same tools on every pod before its first
# check: git that has `merge-tree --write-tree`, the locale LC_ALL names, uv (UV_VERSION), elan with the Lean toolchain every Lake
# package pins, a minimal Rust toolchain, and one check slot over the pod's CPUs. It changes nothing already in place, and it
# prints what it found last, one line per tool, so two pods can be compared.
#
# It reports its setup done by writing POD_SETUP_MARKER (default ~/.research/pod-setup.json) last, and removes it first, so a
# pod reports done only once all of this has succeeded. `research run` returns once a run launches, so a chain that starts a
# check right after launching this one would race it (train TVC lost an hour that way); the merge queue's dispatcher claims
# nothing for a pod without the marker.
set -euo pipefail
UV_VERSION=${UV_VERSION:-0.12.20}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
TOOLCHAIN=$(cat "$ROOT/verity/core/service/flock/lean-toolchain")
export PATH="$HOME/.elan/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
MARKER=${POD_SETUP_MARKER:-$HOME/.research/pod-setup.json}
mkdir -p "$(dirname "$MARKER")"
rm -f "$MARKER"

# `check.landable` and `research merge` compare trees with `git merge-tree --write-tree` (git 2.38); Ubuntu 22.04 ships 2.34, and
# its `add-apt-repository` breaks on the pod's python, so the git-core PPA goes in by hand. The image sets LC_ALL=en_US.UTF-8
# without generating it, and bash's warning about that lands in output the tests compare.
SUDO=$([ "$(id -u)" = 0 ] || echo sudo)
if ! printf '2.38\n%s\n' "$(git --version 2>/dev/null | cut -d' ' -f3)" | sort -V -C; then
  . /etc/os-release
  $SUDO mkdir -p /etc/apt/keyrings
  curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xE1DD270288B4E6030699E45FA1715D88E1DF1F24" \
    | $SUDO gpg --dearmor --yes -o /etc/apt/keyrings/git-core.gpg
  echo "deb [signed-by=/etc/apt/keyrings/git-core.gpg] https://ppa.launchpadcontent.net/git-core/ppa/ubuntu $VERSION_CODENAME main" \
    | $SUDO tee /etc/apt/sources.list.d/git-core.list >/dev/null
  $SUDO apt-get update -qq >/dev/null && $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git >/dev/null
fi
if [ -n "${LC_ALL:-}" ] && ! locale -a 2>/dev/null | grep -qix "$(echo "$LC_ALL" | sed 's/UTF-8/utf8/I')"; then
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq locales >/dev/null && $SUDO locale-gen "$LC_ALL" >/dev/null
fi
# The image's Ubuntu 22.04 trust store still holds the roots that cross-signed the compromised eMudhra intermediates
if ls /etc/ssl/certs | grep -q -E 'emSign_ECC_Root_CA_-_G3|Comodo_AAA_Services_root'; then
  $SUDO apt-get update -qq >/dev/null
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --only-upgrade ca-certificates 'libnss3*' >/dev/null
fi
if [ "$(uv --version 2>/dev/null | cut -d' ' -f2)" != "$UV_VERSION" ]; then
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | $SUDO env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh >/dev/null
fi
if ! command -v elan >/dev/null; then
  curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
fi
elan toolchain install "$TOOLCHAIN" >/dev/null 2>&1 || elan toolchain list | grep -q "${TOOLCHAIN#*:}"
elan default "$TOOLCHAIN" >/dev/null
if ! command -v cargo >/dev/null; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable >/dev/null
fi
for bin in "$HOME/.elan/bin" "$HOME/.cargo/bin"; do                            # non-login shells (research run's) see them too
  for t in "$bin"/*; do
    if [ -x "$t" ] && [ ! -e "/usr/local/bin/$(basename "$t")" ]; then $SUDO ln -s "$t" /usr/local/bin/; fi
  done
done
# A run starts on the system's PATH, not this script's: vy-nebius-3 reported "setup done" with no links (research can't write
# /usr/local/bin without sudo) and its check preflight then refused every train for an hour (24 Oct).
for t in uv elan lake cargo; do
  if ! env -i HOME="$HOME" PATH=/usr/local/bin:/usr/bin:/bin "$t" --version >/dev/null 2>&1; then
    echo "pod_setup: $t does not run on a run's PATH (/usr/local/bin:/usr/bin:/bin): as root, ln -s $(command -v "$t") /usr/local/bin/" >&2
    exit 1
  fi
done
# `slot.py --twin` claims a pair on its home machine over /etc/vy/peers with the vy-cluster key: a node without them refuses every
# twinned check (vy-nebius-3, 24 Oct). A peer this node can't reach is only reported: cpu-4 and cpu-5 reach no uk-south2 node.
PEERS=${CHECK_PEERS:-/etc/vy/peers}
for f in "$PEERS" "$HOME/.ssh/vy-cluster"; do
  [ -s "$f" ] || { echo "pod_setup: no $f: copy it from vy-nebius-2, add this node to every node's $PEERS and to the from= of vy-cluster's line in each peer's authorized_keys" >&2; exit 1; }
done
while read -r name rest; do
  case "$name" in ''|'#'*) continue;; esac
  ssh -n -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 $rest true 2>/dev/null || echo "pod_setup: peer $name unreachable from here" >&2
done < "$PEERS"
# One check slot over the pod's CPUs, so `slot.py` runs one check at a time: with no slots file it runs each check as it comes,
# and two circuit-checks side by side OOM-killed a 256 GB pod (the layout move's shard 5, 5 Oct). A slots file in place is kept.
SLOTS=${CHECK_SLOTS:-/workspace/research/locks/slots}
if [ ! -e "$SLOTS" ]; then
  mkdir -p "$(dirname "$SLOTS")"
  python3 -c 'import os, sys
cpus, runs = sorted(os.sched_getaffinity(0)), []
for c in cpus:
    if runs and c == runs[-1][1] + 1:
        runs[-1][1] = c
    else:
        runs.append([c, c])
with open(sys.argv[1], "w") as f:
    f.write("# NAME CPUS: one check at a time on this pod (tools/verity/check/pod_setup.sh)\na " + ",".join(f"{a}-{b}" if a != b else f"{a}" for a, b in runs) + "\n")' "$SLOTS"
fi
echo "git $(git --version | cut -d' ' -f3)"
echo "uv $(uv --version | cut -d' ' -f2)"
echo "lake $(lake --version | head -1)"
echo "cargo $(cargo --version)"
echo "nvidia-smi $(command -v nvidia-smi >/dev/null && echo present || echo absent)"
echo "slots $(grep -v '^#' "$SLOTS" | tr '\n' ' ')"
python3 -c 'import datetime, json, os, sys
doc = {"done_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "uv": sys.argv[2], "lake": sys.argv[3],
       "cargo": sys.argv[4], "toolchain": sys.argv[5]}
with open(sys.argv[1] + ".tmp", "w") as f:
    json.dump(doc, f)
os.replace(sys.argv[1] + ".tmp", sys.argv[1])' "$MARKER" "$(uv --version | cut -d' ' -f2)" "$(lake --version | head -1)" "$(cargo --version)" "$TOOLCHAIN"
echo "setup done: $MARKER"
