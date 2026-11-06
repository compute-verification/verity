# platform: checkpoints, Fri 6 Nov 2026

The notes clone refused the push (no lane binding), so the hourly lines live here on the branch.

- 06:25Z: read inputs (both agent-system pages, the desk answer on cloud agents, the broker design, `infra/cursor`,
  agent-guard, the control README, the infra repo's broker, relay and approvals). Branch `cursor/platform-vercel-510a`.
  No Task tool in this session, so no workers: the three design passes and the build are done by this agent alone.
  Vercel MCP reaches only Daniel's personal Hobby account (no team); the infra repo is readable via `gh` but not
  pushable, so the prototype goes here under `infra/platform/`.
- 06:50Z: design merged (Workflow DevKit executor, Postgres record, harness turns in persistent named sandboxes).
  Package skeleton typechecks; the first end-to-end test (Slack message, durable lead workflow, fake harness turn in a
  local sandbox, doc update, thread reply) passes in 4.5 s.
- 07:20Z: built and pushed: inbox, turns, workers, jobs, asks between leads, MCP server, transcripts, spend, recovery
  (failed turn, stuck events, stale turn, idle roll), lead admin endpoint, swap and smoke clients, weekly rot check.
  19 unit and 15 end-to-end tests pass in about 16 s. Live on Vercel Sandbox: the credential-brokering firewall swaps a
  placeholder for the real value at the egress and the sandbox never sees it. A full live deployment isn't possible on
  the reachable account (Hobby, no shared database provisioned, no API token); next is a live harness turn on a throwaway `platform-smoke`
  deployment, then the report.
- 08:00Z: the live harness turn is blocked: the Vercel MCP lost access to the only scope it reached (403, needs a
  person to re-authorize), and this VM holds no Vercel token or model key. The deployable one-function smoke is ready
  (`smoke/live-turn/`). Found and fixed while preparing it: the harness refused to add its model rule beside GitHub rules
  installed outside its policy manager (now one manager, with a test that reproduces the refusal); a wake right after a
  roll started a second run; a re-register overwrote the doc (admin writes are now revisions); gateway keys now follow
  Vercel's documented API (quota at creation, budget changes reach the key). Found reviewing the cutover: sandboxes had
  no checkout (now cloned once per sandbox, fetched each turn, a worker's on its branch); bot posts were all dropped, so
  a lead on the platform couldn't hear agents still on Cursor or alerts (now only the platform's own); a bare
  user-group mention routed nowhere; leads couldn't reach Slack, Linear or Notion beyond their reply (now brokered like
  GitHub, per lead). Rot check passes against latest; 23 unit and 18 end-to-end tests pass. Next: the report.
- 08:50Z: leads now wake as on Cursor. A lead's one wake is a doorbell in its inbox thread in the coordination
  channel, rung by `research msg` (which decides who a post is for) or by the people-post service; its reply is posted
  with `research msg reply --as LANE` from its sandbox, under its stable agent id, in the doorbell's thread. The
  platform's own Slack routing (mentions, thread ownership, a channel fallback) would have misrouted in the shared
  channel and dropped posts between leads, and is gone. Linear issue creations and state changes wake the lead of the
  issue's team through a signed webhook. A local smoke against the built server passes (doorbell, inbox-only, Linear,
  pause). Found by it: a relative sandbox directory broke `git config --global` (fixed, with a test), and taking
  another lead's inbox or team was an empty 500 (now a 409 naming the holder). The local server had inherited the VM's
  Slack token and tried two replies to a made-up channel, which Slack refused; local runs now unset it. 26 unit and 20
  end-to-end tests pass. top's inbox for this lane is still absent. Next: the report's remaining sections, SWAP.md.
- 09:10Z: a lead now hears what it heard on Cursor: top-level posts in the channels `research msg subscriptions` lists
  for it, and replies in threads it watches (`watch_thread`, which replaces `subscribe_slack_thread`). A failed turn
  that told no thread is alerted once. `SWAP.md` has the exact commands to set up, move one lead, move all and roll
  back, each checked against the CLIs' help; `scripts/leads.py` builds a lead's entry from research's routes and
  subscriptions (checked on two real handles). `smoke --signed` passes locally in 6 s. The live harness turn is still
  blocked: the Vercel connection reads Daniel's scope but writes are refused (403) until a person re-authorizes it.
  26 unit and 22 end-to-end tests pass. top's inbox for this lane is absent. Next: the Notion report, the final message.
