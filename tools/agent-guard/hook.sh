#!/bin/sh
# `hook.sh NAME`: run the installed agent-guard hook NAME ($AGENT_GUARD_DEST, default /opt/agent-guard) on Cursor's hook event
# (stdin). `.cursor/hooks.json` runs every hook through it. A hook installed outside any checkout can't find the Slack registry
# by itself (Verity's last guard looks beside its own file), so this names it in AGENT_GUARD_REGISTRY: origin/main's,
# extracted once per version into AGENT_GUARD_CACHE, so that a branch cut before a holder was registered still knows the
# holder, else this checkout's. When DEST has no NAME, Verity's last guard runs instead, found in origin/main's history as
# boot.sh's fallback finds it and extracted once into the cache under its commit, which the cache's `last` then names, so a
# later call is one stat; its Task hook takes the registry from AGENT_GUARD_REGISTRY. When that fails too, python3 gets a
# path that doesn't exist, and the fail-closed shell hook refuses the command.
ws=$(cd "$(dirname "$0")/../.." && pwd)
dest=${AGENT_GUARD_DEST:-/opt/agent-guard}
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
[ -f "$dest/$1" ] && exec python3 "$dest/$1"
[ -f "$cache/last/$1" ] && exec python3 "$cache/last/$1"
last=$(git -C "$ws" rev-list -1 --first-parent origin/main -- tools/agent-guard/shell_hook.py 2>/dev/null)
[ -z "$last" ] || git -C "$ws" cat-file -e "$last:tools/agent-guard/shell_hook.py" 2>/dev/null \
  || last=$(git -C "$ws" rev-parse -q --verify "$last^" 2>/dev/null)
if [ -n "$last" ] && [ ! -f "$cache/$last/$1" ]; then
  tmp=$(mkdir -p "$cache" && mktemp -d "$cache/.x.XXXXXX") \
    && git -C "$ws" archive "$last:tools/agent-guard" | tar -x -C "$tmp" && mv -T "$tmp" "$cache/$last" 2>/dev/null
  [ -n "${tmp:-}" ] && rm -rf "$tmp"
fi
if [ -n "$last" ] && [ -f "$cache/$last/$1" ]; then
  ln -s "$last" "$cache/.l.$$" 2>/dev/null && mv -fT "$cache/.l.$$" "$cache/last" 2>/dev/null
  rm -f "$cache/.l.$$"
  exec python3 "$cache/$last/$1"
fi
exec python3 "$dest/$1"
