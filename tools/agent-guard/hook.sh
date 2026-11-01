#!/bin/sh
# `hook.sh NAME`: run the installed agent-guard hook NAME ($AGENT_GUARD_DEST, default /opt/agent-guard) on Cursor's hook event
# (stdin), with this checkout's Slack registry. A session that read `.cursor/hooks.json` when its hooks ran through this file
# runs it until the session ends, since Cursor reads hooks once, so this file must outlive every such session.
export AGENT_GUARD_REGISTRY="${AGENT_GUARD_REGISTRY:-$(dirname "$0")/../../.agents/skills/using-slack/registry.json}"
exec python3 "${AGENT_GUARD_DEST:-/opt/agent-guard}/$1"
