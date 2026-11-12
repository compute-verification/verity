#!/usr/bin/env bash
# `tools/agent-guard/boot.sh [WORKSPACE]`: put the agent guard's files and tools on this VM. The Cloud Agent environment's
# `start` runs it on every boot, from origin/main:
#
#   git -C /workspace fetch -q origin main && git -C /workspace show origin/main:tools/agent-guard/boot.sh | bash -s -- /workspace
#
# It decides nothing about which hooks an agent's session runs. Of this repository's files a cloud agent runs only the
# tracked `.cursor/hooks.json` of the branch its VM boots on, read once, at boot: `~/.cursor/hooks.json` is not read, nor a
# hooks.json at a workspace root above the checkout. `start` ends before Cursor checks that branch out, so this script sees
# the snapshot's checkout and can give no branch a hook: a branch whose hooks.json Cursor can't load runs none until it
# merges one that loads and a new VM boots from it, and `research` says so on such a VM (`inbox.py guard`).
#
# When WORKSPACE's checkout tracks no `.cursor/hooks.json` it first writes one that runs DEST's hook, fail-closed, and
# excludes it in `.git/info/exclude` (git overwrites an excluded file when a branch that tracks it is checked out): a session
# that starts from that checkout reads it, and until DEST's copy is there every command is refused. Then it
# copies the files of origin/main's `tools/agent-guard` (not its tests) to DEST (default /opt/agent-guard), or keeps an
# earlier copy when origin/main can't be read, checks that the hook refuses `env`, and runs DEST's `install.sh` (the redacting `env` and the git guard ahead of
# /usr/bin); the hook stays in place if that refuses. Then it starts the VM's OOM recorder (oom_watch.py) and the tmux session
# the VM keeps (kept_session.py), whose server holds this boot's secrets for a shell that later loses them. It also puts research
# and the GitHub helper on PATH in ~/.bashrc ($AGENT_GUARD_BASHRC), once, and sets WORKSPACE's merge drivers (`.gitattributes`
# names them; git keeps them only in a clone's config) from origin/main's `research.clone.RECORD_DRIVER`, as every run clone
# has them: without them `git merge-tree` here merges a lean-audit.json line by line and calls clean what the queue refuses.
set -euo pipefail
ws=${1:-/workspace}
dest=${AGENT_GUARD_DEST:-/opt/agent-guard}
sudo=sudo
mkdir -p "$dest" 2>/dev/null && [ -w "$dest" ] && sudo=
if git -C "$ws" ls-files --error-unmatch .cursor/hooks.json >/dev/null 2>&1; then
  echo "agent-guard: $ws tracks .cursor/hooks.json, left as it is: a session runs the hooks of the branch it boots on"
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
    ],
    "stop": [
      {"command": "python3 $dest/stop_hook.py", "timeout": 60}
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
  $sudo find "$tmp/tools/agent-guard" -maxdepth 1 -type f -exec cp -t "$dest" {} +
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
# research (the workspace's venv) and the GitHub helper on every later shell's PATH, since a fresh VM has neither
line='export PATH=/workspace/.git/verity-auth/bin:/workspace/.venv/bin:$PATH'
rc=${AGENT_GUARD_BASHRC:-$HOME/.bashrc}
grep -qxF "$line" "$rc" 2>/dev/null || echo "$line" >> "$rc"
if git -C "$ws" show origin/main:tools/research/src/research/clone.py > "$tmp/clone.py" 2>/dev/null; then
  python3 - "$ws" "$tmp/clone.py" <<'PY' || echo "agent-guard: merge drivers not set in $ws (above)" >&2
import importlib.util, subprocess, sys
spec = importlib.util.spec_from_file_location("clone", sys.argv[2])
clone = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clone)
for k, v in clone.RECORD_DRIVER.items():
    subprocess.run(["git", "-C", sys.argv[1], "config", k, v], check=True)
print(f"agent-guard: {len(clone.RECORD_DRIVER)} merge driver settings in {sys.argv[1]}'s git config")
PY
fi
if [ -z "${AGENT_GUARD_SKIP_INSTALL:-}" ]; then
  bash "$dest/install.sh" || echo "agent-guard: install.sh failed or refused (above); the shell hook is in place" >&2
  # a snapshot older than USN-8944-1 still trusts the roots that cross-signed the compromised eMudhra intermediates
  if ls /etc/ssl/certs | grep -q -E 'emSign_ECC_Root_CA_-_G3|Comodo_AAA_Services_root'; then
    { sudo -n apt-get update -qq && sudo -n env DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=l \
        apt-get install -y -qq --only-upgrade ca-certificates 'libnss3*'; } > /tmp/agent-trust-upgrade.log 2>&1 \
      && echo "agent-guard: ca-certificates $(dpkg-query -W -f='${Version}' ca-certificates)" \
      || echo "agent-guard: the ca-certificates upgrade failed (/tmp/agent-trust-upgrade.log)" >&2
  fi
  line='export UV_SYSTEM_CERTS=1'          # uv reads the system trust store, not the roots it ships
  grep -qxF "$line" "$rc" 2>/dev/null || echo "$line" >> "$rc"
  # the VM's OOM recorder: one per VM (it holds its lock; a second exits), so a reboot or a second boot.sh is harmless
  setsid nohup python3 "$dest/oom_watch.py" loop >> /tmp/agent-oom-watch.log 2>&1 < /dev/null &
  echo "agent-guard: OOM recorder started ($dest/oom_watch.py, records in ~/.research/agent-oom.jsonl)"
  # the tmux session the VM keeps (kept_session.py), started with Cursor's tmux where the VM has it, as an agent's shell would
  PATH=$([ -x /exec-daemon/tmux ] && echo /exec-daemon:)$PATH python3 "$dest/kept_session.py" start || true
fi
