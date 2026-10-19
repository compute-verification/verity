#!/bin/sh
# `research msg ARGS` on the control pod, as root (INF-123): for a handle holder who needs a credential that lives only here, such as
# `slack-app export|diff|apply` (the app configuration token pair) and `chat open|close|list`. Over ssh:
#   research pods ssh vy-control-verity -- env RESEARCH_AGENT_URL=https://cursor.com/agents/<id> /data/infra/msg.sh chat list
# Code and registry: /data/infra/msg-repo, a sparse checkout (tools/research, .agents/skills/using-slack) of origin's $REF (default
# main; a PR's head is pull/N/head), fetched with the read-only deploy key on every call. Credentials go only into the environment:
# SLACK_BOT_TOKEN from /data/state/secrets/msg.env (root, 0600); the app's token pair is /data/slack-app/token.json
# ($RESEARCH_SLACK_APP_STATE), the one copy (top and comms, 19 Oct 15:33Z), which `slack-app` rotates in place.
# The notes (inbox routes; people.d, which `chat open` pushes) are its own clone,
# /data/infra/msg-notes, made once from the steward's with its origin and credential helper and pulled on every call, so it never
# writes the clone the budgets guard reads. A command that writes the registry leaves it in the checkout until the next call;
# this prints its diff, to commit in a registry PR. Deployed as /data/infra/msg.sh.
S=/data/state/secrets
R=/data/infra/msg-repo
N=/data/infra/msg-notes
ST=/data/steward/research-notes
REG=.agents/skills/using-slack/registry.json
REF=${REF:-main}
export GIT_SSH_COMMAND="ssh -i $S/verity-deploy -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$S/known_hosts"
export GIT_TERMINAL_PROMPT=0 LC_ALL=C.UTF-8
exec 9> /root/msg.lock
flock 9
if [ ! -d "$R/.git" ]; then
  git clone -q --filter=blob:none --no-checkout git@github.com:compute-verification/verity.git "$R" &&
    git -C "$R" sparse-checkout set tools/research .agents/skills/using-slack || exit 1
fi
git -C "$R" fetch -q origin "$REF" && git -C "$R" checkout -q -f --detach FETCH_HEAD || { echo "msg.sh: could not fetch $REF" >&2; exit 1; }
echo "msg.sh: research msg at $(git -C "$R" rev-parse --short HEAD) ($REF)" >&2
if [ ! -d "$N/.git" ]; then
  git clone -q "$ST" "$N" || exit 1
  for k in remote.origin.url credential.helper user.name user.email; do git -C "$N" config "$k" "$(git -C "$ST" config --get "$k")"; done
fi
git -C "$N" pull -q --ff-only || echo "msg.sh: could not pull $N; it may be stale" >&2
(set -a; . $S/msg.env; set +a
 export RESEARCH_SLACK_APP_STATE=/data/slack-app/token.json RESEARCH_NOTES=$N PYTHONPATH=tools/research/src
 cd "$R" && exec python3.12 -m research msg "$@")
rc=$?
git -C "$R" diff --quiet -- "$REG" || { echo "msg.sh: the command changed $REG; commit this in a registry PR:" >&2; git -C "$R" --no-pager diff -- "$REG"; }
exit $rc
