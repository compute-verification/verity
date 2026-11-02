#!/bin/sh
# `hook.sh NAME`: run the installed agent-guard hook NAME ($AGENT_GUARD_DEST, default /opt/agent-guard) on Cursor's hook event
# (stdin). `.cursor/hooks.json` runs every hook through it. A hook installed outside any checkout can't find the Slack registry
# by itself (Verity's last guard looks beside its own file), so this names it in AGENT_GUARD_REGISTRY: origin/main's,
# extracted once per version into AGENT_GUARD_CACHE, so that a branch cut before a holder was registered still knows the
# holder, else this checkout's. With no hook installed python3 fails, and the fail-closed shell hook refuses the command.
ws=$(cd "$(dirname "$0")/../.." && pwd)
cache=${AGENT_GUARD_CACHE:-${HOME:-/tmp}/.cache/agent-guard}
reg=$(git -C "$ws" rev-parse -q --verify origin/main:.agents/skills/using-slack/registry.json 2>/dev/null)
if [ -z "${AGENT_GUARD_REGISTRY:-}" ]; then
  AGENT_GUARD_REGISTRY=$ws/.agents/skills/using-slack/registry.json
  if [ -n "$reg" ] && mkdir -p "$cache" 2>/dev/null; then
    [ -f "$cache/registry-$reg.json" ] || { git -C "$ws" cat-file blob "$reg" > "$cache/.r.$$" \
      && mv -f "$cache/.r.$$" "$cache/registry-$reg.json"; } 2>/dev/null
    [ -f "$cache/registry-$reg.json" ] && AGENT_GUARD_REGISTRY=$cache/registry-$reg.json
  fi
  export AGENT_GUARD_REGISTRY
fi
exec python3 "${AGENT_GUARD_DEST:-/opt/agent-guard}/$1"
