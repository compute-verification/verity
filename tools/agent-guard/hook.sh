#!/bin/sh
# `hook.sh NAME`: run origin/main's agent-guard hook NAME (shell_hook.py, task_hook.py) on Cursor's hook event (stdin), with
# origin/main's Slack registry. `.cursor/hooks.json` runs every hook through it, so a branch that tracks an older hook still
# runs the current one. origin/main's `tools/agent-guard` is extracted once per tree into AGENT_GUARD_CACHE; when origin/main
# can't be read, the branch's own copy and registry run.
name=$1
ws=$(cd "$(dirname "$0")/../.." && pwd)
own=$ws/tools/agent-guard/$name
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
tree=$(git -C "$ws" rev-parse -q --verify origin/main:tools/agent-guard 2>/dev/null) || exec python3 "$own"
dir=$cache/$tree
if [ ! -f "$dir/$name" ]; then
  tmp=$(mkdir -p "$cache" && mktemp -d "$cache/.x.XXXXXX") \
    && git -C "$ws" archive "$tree" | tar -x -C "$tmp" && mv -T "$tmp" "$dir" 2>/dev/null
  [ -n "${tmp:-}" ] && rm -rf "$tmp"
  [ -f "$dir/$name" ] || exec python3 "$own"
fi
exec python3 "$dir/$name"
