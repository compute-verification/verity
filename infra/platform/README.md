# The research platform

Verity's leads and their workers on Vercel: each lead a durable workflow with one inbox, each wake one turn of Claude
Code or Codex (through the AI SDK harness) in a persistent sandbox, with the workstream doc as its only memory and
Postgres as the record of everything. `DESIGN.md` is the design, `SWAP.md` the cutover and rollback runbook.

This directory is staged in Verity for want of a pushable infra repository and never lands on Verity's `main` (Verity
keeps no infra, and allows no `SWAP.md` or `CHECKPOINTS.md`). Two files read Verity's tree by relative path and change
when it moves: `test/unit.test.ts` parses research's routing vectors
(`tools/research/tests/slack_routing_vectors.json`; vendor them), and `scripts/leads.py` reads the handle registry
(`RESEARCH_SLACK_REGISTRY` overrides it).

## Layout

- `app/api/`: the Next.js routes, each mounting a function of `src/http.ts`: Slack events, the Linear webhook, the
  cron tick and rot check, job callbacks, the admin endpoint (`leads`), transcripts and MCP.
- `workflows/`: `lead.ts` (one run per lead, holding the hook `lead:<lane>`), `worker.ts`, `turn.ts` (the steps of a
  turn: begin, slice, end, park) and the weekly rot check.
- `src/`: the record (`schema.ts`, `store.ts`), waking (`wake.ts`), the cron (`cron.ts`), Slack (`slack.ts`, doorbells
  and replies through `research msg`), Linear (`linear.ts`), host tools (`tools.ts`), MCP (`mcp.ts`), instructions
  (`prompt.ts`), credentials and the brokered APIs (`credentials.ts`, `services.ts`), prices (`prices.ts`), jobs
  (`jobs.ts`, `job-done.ts`), sandboxes (`sandbox/vercel.ts`, `sandbox/local.ts`), harnesses (`harness/`, with the
  scripted `fake.ts` the tests drive), and the runtime that wires them (`runtime.ts`).
- `scripts/`: `swap.ts` (the cutover commands), `leads.py` (LEADS.json from research's routes and subscriptions),
  `smoke.ts` (a running platform, end to end over HTTP) and `rot-check.sh`.
- `smoke/live-turn/`: a one-function deployment that runs one real harness turn in a Vercel Sandbox.

## Commands

```bash
npm ci
npm run check                    # typecheck, unit suite, end-to-end suite (about 20 seconds)
npm run build                    # next build
npm run swap -- id LANE          # a lead's stable agent id; the other swap commands need PLATFORM_URL and PLATFORM_API_TOKEN
npm run smoke -- --signed        # a running platform: also needs SLACK_SIGNING_SECRET and LINEAR_WEBHOOK_SECRET
scripts/rot-check.sh             # every dependency at @latest: typecheck and both suites
```

Run locally on the local world, a directory sandbox, PGlite and the scripted harness: set `PLATFORM_API_TOKEN`,
`SLACK_SIGNING_SECRET` and `LINEAR_WEBHOOK_SECRET` to local values, then

```bash
npm run build && env -u SLACK_BOT_TOKEN -u LINEAR_API_KEY -u NOTION_API_KEY npx next start -p 3100
PLATFORM_URL=http://localhost:3100 npm run smoke -- --signed
```

Unset those three: with them set, the local server posts to Slack and calls Linear and Notion with real credentials.

## Tests

`test/unit.test.ts` and `test/sandbox-policy.test.ts` are the unit suite; `test/e2e.integration.test.ts` and
`test/recovery.integration.test.ts` run the whole platform in one process on Workflow DevKit's local world, the real
`HarnessAgent` driving the scripted harness in directory sandboxes, PGlite, and an in-memory Slack and job queue.
Neither suite needs a network or a credential.
