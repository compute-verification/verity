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
