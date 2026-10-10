#!/usr/bin/env bash
# `tools/agent-guard/boot.sh [WORKSPACE]`: put the agent guard on this VM whatever branch WORKSPACE (default /workspace) has
# checked out. The Cloud Agent environment's `start` runs it on every boot, from origin/main, so a branch older than the
# guard still gets it:
#
#   git -C /workspace fetch -q origin main && git -C /workspace show origin/main:tools/agent-guard/boot.sh | bash -s -- /workspace
#
# Cloud agents run project hooks only (`WORKSPACE/.cursor/hooks.json`; `~/.cursor/hooks.json` is not read), so a branch
# without that file runs no shell hook. When the branch doesn't track `.cursor/hooks.json` it first writes one that runs DEST's hook, fail-closed, and excludes it in
# `.git/info/exclude` (git overwrites an excluded file when a branch that tracks it is checked out): Cursor reads hooks once,
# early in the session, so the file comes before the fetch, and until DEST's copy is there every command is refused. Then it
# copies origin/main's `tools/agent-guard` to DEST (default /opt/agent-guard), or keeps an earlier copy when origin/main can't
# be read, checks that the hook refuses `env`, and runs DEST's `install.sh` (the redacting `env` and the git guard ahead of
# /usr/bin); the hook stays in place if that refuses.
set -euo pipefail
ws=${1:-/workspace}
dest=${AGENT_GUARD_DEST:-/opt/agent-guard}
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
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
if git -C "$ws" fetch -q origin main && git -C "$ws" archive origin/main tools/agent-guard | tar -x -C "$tmp"; then
  $sudo mkdir -p "$dest"
  $sudo cp "$tmp"/tools/agent-guard/* "$dest"/
  $sudo chmod 755 "$dest"/*
elif [ -f "$dest/shell_hook.py" ]; then
  echo "agent-guard: origin/main unreadable; using the copy already in $dest" >&2
else
  echo "agent-guard: origin/main unreadable and no copy in $dest: no guard on this VM" >&2
  exit 1
fi
verdict=$(printf '{"hook_event_name":"beforeShellExecution","command":"env | cut -d= -f1","cwd":"/"}' | python3 "$dest/shell_hook.py")
case $verdict in
  *'"deny"'*) ;;
  *) echo "agent-guard: $dest/shell_hook.py let \`env | cut -d= -f1\` through: $verdict" >&2; exit 1 ;;
esac
if [ -z "${AGENT_GUARD_SKIP_INSTALL:-}" ]; then
  bash "$dest/install.sh" || echo "agent-guard: install.sh failed or refused (above); the shell hook is in place" >&2
fi
