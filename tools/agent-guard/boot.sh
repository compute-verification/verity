#!/usr/bin/env bash
# `tools/agent-guard/boot.sh [WORKSPACE]`: put the agent guard, which lives in compute-verification/infra, on this VM whatever
# branch WORKSPACE (default /workspace) has checked out. The Cloud Agent environment's `start` runs it on every boot, from
# origin/main, so a branch older than it still gets it:
#
#   git -C /workspace fetch -q origin main && git -C /workspace show origin/main:tools/agent-guard/boot.sh | bash -s -- /workspace
#
# When the branch doesn't track `.cursor/hooks.json` it first writes one that runs DEST's hook (default /opt/agent-guard),
# fail-closed, and excludes it in `.git/info/exclude`: Cursor reads hooks once, early in the session, so the file comes before
# any fetch, and until DEST's copy is there every command is refused. Then it brings INFRA (default /opt/infra) to infra's
# main, cloning it the first time, with WORKSPACE's GitHub helper, and runs INFRA's `tools/agent-guard/boot.sh`, which copies
# the guard to DEST and does the rest of the VM's setup (its header says what). When infra can't be fetched it runs the
# checkout INFRA already has. Without one, or when that boot.sh fails, Verity's last guard runs this boot: DEST's copy is
# kept if its hook refuses `env`, else the guard origin/main had before it moved to infra goes there, and that tree's
# boot.sh runs. If DEST's hook still doesn't refuse `env`, it is removed, so that the fail-closed hook refuses every command.
set -euo pipefail
# Verity's last boot.sh runs with AGENT_GUARD_FALLBACK set: a tree whose boot.sh is this file would run it forever
[ -z "${AGENT_GUARD_FALLBACK:-}" ] || { echo "agent-guard: Verity's last guard is this boot.sh; stopping" >&2; exit 1; }
ws=${1:-/workspace}
dest=${AGENT_GUARD_DEST:-/opt/agent-guard}
infra=${AGENT_GUARD_INFRA:-/opt/infra}
url=${AGENT_GUARD_INFRA_URL:-https://verity-agent@github.com/compute-verification/infra.git}
sudo=sudo
mkdir -p "$dest" 2>/dev/null && [ -w "$dest" ] && sudo=
if git -C "$ws" ls-files --error-unmatch .cursor/hooks.json >/dev/null 2>&1; then
  echo "agent-guard: $ws tracks .cursor/hooks.json; its hook runs"
else
  mkdir -p "$ws/.cursor"
  cat > "$ws/.cursor/hooks.json" <<EOF
{
  "version": 1,
  "hooks": {
    "beforeShellExecution": [
      {"command": "python3 $dest/shell_hook.py", "timeout": 10, "failClosed": true}
    ],
    "preToolUse": [
      {"command": "python3 $dest/task_hook.py", "matcher": "Task", "timeout": 10},
      {"command": "python3 $dest/task_hook.py", "matcher": "CallDynamicTool", "timeout": 10},
      {"command": "python3 $dest/task_hook.py", "matcher": "^MCP:", "timeout": 10}
    ]
  }
}
EOF
  exclude=$(git -C "$ws" rev-parse --git-path info/exclude)
  case $exclude in /*) ;; *) exclude=$ws/$exclude ;; esac
  mkdir -p "$(dirname "$exclude")"
  grep -qxF /.cursor/hooks.json "$exclude" 2>/dev/null || echo /.cursor/hooks.json >> "$exclude"
  echo "agent-guard: wrote $ws/.cursor/hooks.json (excluded), running $dest/shell_hook.py"
fi
refuses_env() {
  printf '{"hook_event_name":"beforeShellExecution","command":"env | cut -d= -f1","cwd":"/"}' \
    | python3 "$dest/shell_hook.py" 2>/dev/null | grep -q '"deny"'
}
# git as this user, never root: the helper caches its token in WORKSPACE's git dir
pull() {
  local auth=(-c credential.helper= -c "credential.helper=!python3 '$ws/.git/verity-auth/verity-github.py' credential"
              -c credential.useHttpPath=true)
  if [ ! -w "$infra" ] && [ ! -w "$(dirname "$infra")" ]; then
    sudo -n mkdir -p "$infra" && sudo -n chown "$(id -u):$(id -g)" "$infra" || return 1
  fi
  if [ -e "$infra/.git" ]; then
    GIT_TERMINAL_PROMPT=0 git "${auth[@]}" -C "$infra" fetch -q "$url" main && git -C "$infra" checkout -qf -B main FETCH_HEAD
  else
    GIT_TERMINAL_PROMPT=0 git "${auth[@]}" clone -q "$url" "$infra"
  fi
}
if pull; then
  echo "agent-guard: $infra is infra's main ($(git -C "$infra" log -1 --format='%h %cs'))"
elif [ -f "$infra/tools/agent-guard/boot.sh" ]; then
  echo "agent-guard: infra's main can't be fetched; running $infra's checkout ($(git -C "$infra" log -1 --format='%h %cs'))" >&2
fi
if [ -f "$infra/tools/agent-guard/boot.sh" ] && bash "$infra/tools/agent-guard/boot.sh" "$ws" < /dev/null && refuses_env; then
  exit 0
fi
echo "agent-guard: no guard from infra (above); Verity's last guard runs this boot" >&2
last=$(git -C "$ws" rev-list -1 origin/main -- tools/agent-guard/shell_hook.py 2>/dev/null || true)
[ -z "$last" ] || git -C "$ws" cat-file -e "$last:tools/agent-guard/shell_hook.py" 2>/dev/null || last=$last^
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
if [ -n "$last" ] && git -C "$ws" archive "$last" tools/agent-guard 2>/dev/null | tar -x -C "$tmp" 2>/dev/null; then
  if refuses_env; then
    echo "agent-guard: keeping the guard in $dest" >&2
  else
    $sudo mkdir -p "$dest"
    $sudo cp "$tmp"/tools/agent-guard/* "$dest"/
    $sudo chmod 755 "$dest"/*
    echo "agent-guard: put Verity's last guard ($(git -C "$ws" rev-parse --short "$last")) in $dest" >&2
  fi
  AGENT_GUARD_FALLBACK=1 bash "$tmp/tools/agent-guard/boot.sh" "$ws" < /dev/null \
    || echo "agent-guard: Verity's last boot.sh failed (above)" >&2
fi
refuses_env && exit 0
[ ! -e "$dest/shell_hook.py" ] || $sudo rm -f "$dest/shell_hook.py"
echo "agent-guard: no hook in $dest refuses \`env\`: no guard on this VM, so the fail-closed hook refuses every command" >&2
exit 1
