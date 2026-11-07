# Sourced by every script in this kit: where the kit, the run's tree, the research command and the logs are on this VM,
# and the nodes, paths and pins Monday's run uses. Export any of the first group to override it.
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# A checkout of the repository whose origin is the team's, with its synced .venv (`uv sync --all-packages`): `research`
# and the python that imports research.pods.runpod (the ssh key, from the VM's injected secrets).
MAIN_REPO=${MAIN_REPO:-/workspace}
TREE=${TREE:-$HOME/wt-odserve-monday}
REF=${REF:-cursor/onsite-push1b-061a}
# The head of REF the kit was staged for: tree.sh refuses another unless WANT is set to it (or WANT= for any head).
WANT=${WANT-9068336929e0bdd40dc6f60cab2b8975c3809e70}
LOGS=${LOGS:-$HOME/odserve-monday-logs}
FOLLOW_NODE=${FOLLOW_NODE:-vy-nebius-cpu-12}
RESEARCH=${RESEARCH:-$(command -v research || echo "$MAIN_REPO/.venv/bin/research")}
PY=${PY:-$(dirname "$RESEARCH")/python}

SERVED=vy-dc-h200-2 SERVED_IP=204.12.188.151
SETTINGS_NAME=settings-monday.env
ONSITE=/workspace/research/lanes/network-accounting/onsite
SERVE_ROOT=/workspace/research/vllm-served-verdicts
NETACCT=/workspace/incoming/netacct
ROW=/workspace/research/vllm-composed-h200int8/r20261102-233856-d4d6/llama32-1b__int8__h200__tp1__b4__i256__o32__mixed__greedy__bi-eager
# H1's registered weights root under rr/circuit.txt's window Program (the gate's models), and the challenger gate's pin,
# the first line of rr/main.txt (main's first-parent line from it, which serve.sh writes).
H1_ROOT=c3c8a1ba
SERVE_GATE_PIN=f0c4b1355fb7d8b37b1f8fcceccd14b4789270a1
# The auditor gate's pinned commit, which the settings job and the follow-on's certifier key read hold it to.
CERTIFIER_GATE_PIN=701c8c0073a63fb9d3fdc6d9a993e3d5a5039656
VCLASS=8-32-4-4-16

follow_ip() {
  case $1 in
    vy-nebius-cpu-11) echo 81.85.4.136 ;;
    vy-nebius-cpu-12) echo 81.85.1.89 ;;
    *) echo "REFUSED: $1 is neither follow-on node (vy-nebius-cpu-11, vy-nebius-cpu-12)" >&2; return 3 ;;
  esac
}
tree_ok() {  # 0 iff TREE is a clean checkout of REF's head on origin (and WANT's, where WANT is set)
  local want head
  want=$(git -C "$TREE" ls-remote origin "refs/heads/$REF" | cut -f1) head=$(git -C "$TREE" rev-parse HEAD 2> /dev/null)
  [ -n "$want" ] && [ "$head" = "$want" ] && git -C "$TREE" diff --quiet HEAD && { [ -z "$WANT" ] || [ "$head" = "$WANT" ]; } \
    || { echo "REFUSED: $TREE is at ${head:-nothing} (or dirty), not the head of $REF ${want:-?}${WANT:+, the kit head $WANT}: run tree.sh"; return 3; }
}
[ -x "$RESEARCH" ] || { echo "REFUSED: no research command ($RESEARCH): uv sync --all-packages in $MAIN_REPO, or set RESEARCH" >&2; return 3 2> /dev/null || exit 3; }
PATH=$(dirname "$RESEARCH"):$PATH
mkdir -p "$LOGS"
