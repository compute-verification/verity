#!/bin/sh
# `hook.sh NAME`: run the agent-guard hook NAME (shell_hook.py, task_hook.py, stop_hook.py) on Cursor's hook event (stdin),
# with origin/main's Slack registry. `.cursor/hooks.json` runs /opt/agent-guard/hook.sh, the copy boot.sh puts there from
# origin/main at boot, and that copy runs only the hook beside it: no file of the checkout runs, whatever branch is checked
# out. On a machine with neither /opt/agent-guard nor Cursor's cloud runtime (a desktop agent's laptop), hooks.json runs
# origin/main's text of this file out of git instead (any $0 that is not an absolute path), which runs origin/main's
# `tools/agent-guard`, extracted once per tree into AGENT_GUARD_CACHE: never the working tree's copy. Either way, when its
# hook can't be found it refuses (`deny`) and says why. Every program it and its hooks start is found on the fixed PATH
# below, whose directories only root can write, and nothing the environment names (PYTHON*, GIT_*, BASH_ENV, LD_*, the git
# guard's own switches) can make them load a checkout's file: hooks.json scrubs the same before running this file.
PATH=/usr/local/bin:/usr/bin:/bin; export PATH
name=$1
refuse() {
  m="Refused by the agent guard (hook.sh): $1. The guard never runs a checkout's own copy of itself."
  printf '{"permission":"deny","agent_message":"%s","user_message":"%s"}\n' "$m" "$m"
  exit 0
}
v=$(awk 'BEGIN{for(k in ENVIRON)if(k~/^(PYTHON|GIT_)[A-Za-z0-9_]*$/)print k}') || refuse "awk could not list the environment"
unset ENV BASH_ENV CDPATH LD_PRELOAD LD_LIBRARY_PATH LD_AUDIT VERITY_REAL_GIT VERITY_GIT_GUARD AGENT_GUARD_GIT_HOOKS \
  AGENT_GUARD_CHAIN_HOOKS $v
ws=$(git rev-parse --show-toplevel 2>/dev/null) || ws=$PWD
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
case $0 in
  /*) here=$(dirname "$0")
      [ -r "$here/$name" ] || refuse "$here/$name is missing or unreadable: rerun origin/main's boot.sh"
      exec /usr/bin/python3 -E -S "$here/$name" ;;
esac
tree=$(git -C "$ws" rev-parse -q --verify origin/main:tools/agent-guard 2>/dev/null) \
  || refuse "no /opt/agent-guard here, and $ws has no origin/main to run the guard from: fetch origin main"
dir=$cache/$tree
if [ ! -f "$dir/$name" ]; then
  tmp=$(mkdir -p "$cache" && mktemp -d "$cache/.x.XXXXXX") \
    && git -C "$ws" archive "$tree" | tar -x -C "$tmp" && { [ -d "$dir" ] || mv "$tmp" "$dir"; }
  [ -n "${tmp:-}" ] && rm -rf "$tmp"
  [ -f "$dir/$name" ] || refuse "origin/main's tools/agent-guard/$name could not be extracted into $cache"
fi
exec /usr/bin/python3 -E -S "$dir/$name"
