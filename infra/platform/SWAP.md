# Moving Verity's leads onto the platform, and back

Exact commands, in order. Run them from `infra/platform/` in a Verity checkout on a machine where `research msg` works
(it reads Slack and publishes inbox routes to the notes repository) and that holds the keys in DESIGN.md Section 13:
Daniel's laptop, or an @infra agent's machine for every step but archiving and relaunching Cursor agents, which need
Daniel's key. Nothing here prints a secret: values go in through `vercel env add`, which reads them from stdin.

```bash
cd infra/platform
npm ci
export VERCEL="npx --yes vercel@65.12.1"
```

## 1. Set up the platform, once (about 35 minutes)

**The project.** On the Pro team, link a new project named `platform-agents`. Never link an existing project.

```bash
$VERCEL link --yes --scope <the team's slug> --project platform-agents
```

In the project's settings, set Git deployments to one protected branch only.

**The secrets.** They live in `.env.platform` on this machine (git-ignored, mode 600), since Vercel never returns a
sensitive value and the commands below need some of them. Generate the platform's own three, then append the rest,
one `NAME=value` line each, with an editor (never `echo`, which leaves them in shell history):

```bash
umask 077
for k in PLATFORM_SECRET PLATFORM_API_TOKEN CRON_SECRET; do printf '%s=%s\n' "$k" "$(openssl rand -hex 32)"; done > .env.platform
"${EDITOR:-vi}" .env.platform
```

The lines to add: `DATABASE_URL`, `SLACK_SIGNING_SECRET`, `SLACK_BOT_TOKEN`, `LINEAR_WEBHOOK_SECRET`, `VERCEL_TOKEN`,
`VERCEL_TEAM_ID`, `PLATFORM_GITHUB_BROKER_URL`, `PLATFORM_JOBS_URL`, `PLATFORM_JOBS_TOKEN`, `PLATFORM_JOBS_LINE`, and
optionally `LINEAR_API_KEY` and `NOTION_API_KEY` (only if a lead is granted Linear or Notion).

**The environment.** Each secret goes in from stdin, as a sensitive value; then the configuration, which isn't secret:

```bash
while IFS='=' read -r k v; do
  printf %s "$v" | $VERCEL env add "$k" --environment production --sensitive --yes
done < .env.platform
config() { printf %s "$2" | $VERCEL env add "$1" --environment production --no-sensitive --yes; }
config PLATFORM_SANDBOX vercel
config PLATFORM_REPO_URL https://github.com/compute-verification/verity.git
config PLATFORM_SANDBOX_TIMEOUT_MINUTES 240
config PLATFORM_DEFAULT_MODEL anthropic/claude-opus-5.5
config PLATFORM_ALERT_CHANNEL "$(python3 -c 'import json; print(json.load(open("../../.agents/skills/using-slack/registry.json"))["channels"]["agent-alerts"]["id"])')"
```

`PLATFORM_REPO_URL` is required on Vercel: replies are posted by `research` from the lead's checkout.

**Deploy.** The database schema is created on the first request.

```bash
$VERCEL deploy --prod
export PLATFORM_URL=https://<the production domain it printed>
while IFS='=' read -r k v; do         # what the commands below read, without sourcing the file
  case $k in PLATFORM_API_TOKEN|SLACK_SIGNING_SECRET|LINEAR_WEBHOOK_SECRET) export "$k=$v" ;; esac
done < .env.platform
```

**Slack and Linear.**

- In the platform's Slack app (it only receives events), set the Events request URL to `$PLATFORM_URL/api/slack/events`,
  subscribe to `message.channels`, and invite its bot to `#agent-coordination` and to every channel a lead subscribes
  to (`#agent-alerts`, `#ask-daniel`, `#ask-the-team`, `#resource-accounting`, and the chat channels
  `research msg subscriptions` names).
- In Linear's settings, add a webhook on Issue events to `$PLATFORM_URL/api/linear/webhook`. Its signing secret is the
  `LINEAR_WEBHOOK_SECRET` set above.

**Smoke.** First the scripted harness, then each real one. Each registers a throwaway lead (`smoke` unless `--lane`
says otherwise) and pauses it at the end; exit 0 is a pass.

```bash
npm run smoke -- --signed
npm run smoke -- --lane smoke-claude --harness claude-code --wait-minutes 10
npm run smoke -- --lane smoke-codex --harness codex --wait-minutes 10
```

`--signed` signs a doorbell and a Linear event with the deployment's secrets; its channel is made up, so its reply is
refused by Slack (a `post-error` addressed to the doorbell's thread) rather than posted.

## 2. Move one lead (about 5 minutes)

For lane `LANE` (a handle in `.agents/skills/using-slack/registry.json`), with its Cursor agent `BC_ID`:

1. Ask the Cursor lead for its handoff, and save it as `cutover/LANE/handoff.md`. It ends its turn after writing it.

2. Move the handle's inbox route to the platform lead. This keeps the inbox thread and its unread doorbells, and
   prints the inbox's channel and ts:

   ```bash
   research msg names open --as LANE --succeeds --holder "$(npm run -s swap id LANE)"
   ```

3. Register the lead, from research's own record of its inbox and subscriptions, and start it:

   ```bash
   python3 scripts/leads.py LANE --budget 1000 > cutover/LANE.json
   npm run -s swap all cutover/LANE.json --kick "You are on the platform now. Read your doc. Call watch_thread on each thread you still await a person in (your open #ask-daniel cards: research msg ask-daniel pages --as LANE), then carry on."
   ```

   The output shows the lead, its `agentId`, its gateway key id and its run id. `leads.py` writes
   `cutover/LANE/charter.md` from the registry's remit when it is absent; edit it before `swap all` if the lead needs
   more than its remit.

4. Check it hears and answers. Ask it something as the handle you hold (an @infra agent: `--as infra`); its reply
   appears in that post's thread. A person can instead ask in `#ask-the-team` naming `@LANE`, which the Slack events
   service rings it for.

   ```bash
   research msg send --as YOUR_HANDLE --to @LANE --text "Platform check: reply with one line saying where you are running."
   npm run -s swap status LANE
   ```

5. Archive the Cursor agent from Daniel's laptop: `python3 ../cursor/launch_coordinator.py archive BC_ID`. Until
   then both leads hear the inbox, so do this step as soon as step 4 passes.

## 3. Move all leads (about 10 minutes more)

The leads to move are the handles whose open inbox a Cursor cloud agent holds. With every one's handoff saved under
`cutover/<handle>/handoff.md`:

```bash
HANDLES=$(research msg names list --json | python3 -c 'import json, sys; print(" ".join(r["name"] for r in json.load(sys.stdin) if not r.get("closed") and str(r.get("holder") or "").startswith("bc-")))')
for h in $HANDLES; do research msg names open --as "$h" --succeeds --holder "$(npm run -s swap id "$h")"; done
python3 scripts/leads.py $HANDLES --budget 1000 > LEADS.json
npm run -s swap all LEADS.json --kick "You are on the platform now. Read your doc. Call watch_thread on each thread you still await a person in, then carry on."
npm run -s swap status
```

Then step 2.4 for each lead, and archive each Cursor agent.

## 4. Roll back one lead (about 5 minutes)

```bash
npm run -s swap pause LANE                      # the platform stops taking LANE's events at once
npm run -s swap doc LANE > cutover/LANE/back.md # its doc, as the Cursor successor's handoff
```

From Daniel's laptop, relaunch it on Cursor with the doc as its handoff, at the next generation (the archived agent
holds the current one):

```bash
python3 ../cursor/launch_coordinator.py launch LANE --predecessor BC_ID --text-file cutover/LANE/back.md \
  --pairing-thread TS --generation N
```

Its brief opens the inbox with `research msg names open --as LANE --succeeds`, which takes the route back with the
inbox thread and its unread doorbells. `npm run -s swap resume LANE` undoes a pause.

To roll back every lead, run the same for each, or pause them all at once:

```bash
for h in $HANDLES; do npm run -s swap pause "$h"; done
```

## 5. After a deploy

Each lead moves to new code after 25 turns or a day idle. To move them at once:

```bash
npm run -s swap roll
```
