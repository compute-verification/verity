# The research platform: Verity's agents on Vercel

The platform runs Verity's leads (the coordinators) and their workers on Vercel instead of Cursor. A lead is a durable
workflow with one inbox. Slack doorbells, Linear issue events, cron ticks, finished jobs, finished workers, other
leads' questions and MCP calls all land in that inbox. Each wake is one disposable turn of a native coding harness, Claude Code or Codex,
inside a persistent sandbox. The lead's workstream doc is its only memory. Postgres keeps the record: leads, events,
turns, full transcripts, usage and cost, doc revisions, workers, jobs and credential grants. Credentials stay on the
host. A sandbox holds placeholders, and Vercel's sandbox firewall swaps in the real value on the way out.

This directory holds the design (this file), the code, the tests, the cutover runbook (`SWAP.md`) and the hourly notes
of the night it was built (`CHECKPOINTS.md`). It is staged in Verity for want of a pushable infra repository. Its home is
the infra repository: Verity keeps no infra (`tests/test_infra_paths.py`) and allows no `SWAP.md` or `CHECKPOINTS.md`
(`tests/test_repository.py`), so this directory never lands on Verity's `main`.

## 1. Requirements

From the brief:

- **R1, swap at a moment's notice.** Any lead moves to the platform, or back, with one command, in minutes. The
  same holds for all leads at once. A lead's model and harness change with one command, including to a model that
  doesn't run on Cursor.
- **R2, centralized billing and usage.** Every turn's tokens and cost are recorded per lead. Each lead has a budget
  that a system outside the agent enforces, and month-to-date spend is one query.
- **R3, centralized credentials.** Credentials are issued in one place, recorded by reference, and never written
  into an agent's environment.
- **R4, transcripts.** Every turn's full transcript (instructions, prompt, reasoning, tool calls and results, reply)
  is stored and searchable by lead, time and text.
- **R5, native harnesses.** Claude Code and Codex run through the AI SDK harness, and the platform has no agent loop
  of its own.
- **R6, durable leads.** A lead wakes on Slack, cron, a finished job or worker, or another lead. It never loses an
  event, and it never runs two turns at once.
- **R7, workers.** A lead starts workers as separate runs, each in its own sandbox and on its own branch, and is woken
  when each finishes.
- **R8, heavy work off the sandbox.** GPU work, Lean and circuit builds and test suites go to the team's job queue, and
  the job's end wakes the lead.
- **R9, a surface.** Slack, plus an MCP server for anyone outside: `ask_lead`, `assign`, `status`.
- **R10, self-healing.** Every failure is detected and acted on without a human.
- **R11, small and pinned.** Few moving parts, pinned versions, and a cheap scheduled check that they still work.

From what failed on Cursor (the incident record is in the report page, not here):

- **P1.** A held run left a coordinator deaf for hours, and wakes that arrived mid-turn were cancelled or refused
  ("agent busy").
- **P2.** A billing stop or a usage limit stopped every agent. Relaunching needed Daniel's laptop.
- **P3.** Workers stopped mid-step and nobody noticed. A stopped worker can't be resumed, or moved to another model.
- **P4.** Subscriptions lapse after 30 days and timers after 7. Nothing can be sent to a running agent.
- **P5.** Every VM holds about twenty credentials, and every worker can read every key.
- **P6.** Heavy work on 16 GB VMs was killed for lack of memory.
- **P7.** The team is capped at 20 cloud agents, and Cursor's own spend is unmeasured.

## 2. What Cursor gives us, and what disappears

| Cursor provides | On the platform | Status |
| --- | --- | --- |
| An agent loop and its model access | The harness's own Claude Code or Codex loop, with models through the AI Gateway | Replaced |
| A VM per agent, booted from our environment | A persistent named Vercel Sandbox per lead and per worker, with the repository cloned once and fetched each turn | Replaced |
| Wakes: Slack and Linear apps, thread subscriptions, timers | One inbox table. Replies in a lead's inbox thread and its Linear team's issue events arrive at our endpoints, ticks come from cron, and jobs, workers, leads and MCP write events directly | Replaced, with no lapses |
| Sending to a running agent (missing; the guard's inbox works around it) | An event that arrives mid-turn waits in the inbox for the next turn | Fixed |
| Agent identity (Cursor OIDC, which the GitHub broker trusts) | The project's Vercel OIDC, with the lead named in `X-Verity-Agent` | Needs the broker to trust it |
| Team secrets on every VM | The platform project's environment, sealed per-lead gateway keys, and header rules at the sandbox firewall | Narrower |
| Task subagents (workers) | `dispatch_worker`: a workflow run per worker, whose end wakes the lead | Replaced |
| The dashboard, artifacts and Daniel's view of every transcript | `status` and `search_transcripts` over MCP, and the `turns` and `transcript_parts` tables | Replaced, without a UI |
| Relaunch from Daniel's laptop with his key | `swap register` and `swap roll` over the admin API | Replaced |
| Environment snapshots, start script and agent guard | The harness bootstrap (the checkout), the sandbox network allowlist, and no credentials inside | Replaced |
| Cursor's own coding agent and its quality on our work | Claude Code or Codex; neither is measured on our work | **Lost until measured** |
| Long shells on a 16 GB VM | Sandboxes are for editing; builds and tests go to jobs | By design |

## 3. Design

### 3.1 Three priors, judged

- **A, Vercel-native.** Use Eve for leads, Workflow, Sandbox, the AI Gateway, OIDC, crons and Vercel's own
  observability for everything. Least code, but Eve is a fast-moving preview. Its session model (a long-lived agent
  that holds context) is the shape that failed on Cursor, and it ties the record to Vercel's retention.
- **B, minimal and portable.** A durable executor plus Postgres, with the harness for leads and workers alike, behind a
  sandbox interface with a local implementation. Few parts, testable offline, but it reimplements what Vercel gives for
  free (credential injection, budgets, cron).
- **C, least change.** Keep the Cursor-era machinery (the workstream doc and handoffs, Slack threads as the surface, the
  jobs API, the GitHub broker, `research msg`) and replace only the agent host.

**The merge.** B's core: Workflow DevKit as the executor, Postgres as the record, one harness path for leads and
workers, and a sandbox interface whose local implementation runs the whole platform in tests. A's Vercel pieces wherever
they remove code: the sandbox firewall for credentials, AI Gateway keys with budgets, OIDC, crons. C's reuse: Slack
threads as the surface, the workstream doc as memory, the jobs API for heavy work, the GitHub broker. Eve is set aside.
A lead turn is a disposable harness session over a doc, which needs only Workflow, and Eve's churn would be a second
preview to pin beside the harness.

### 3.2 Moving parts

One Vercel project (`platform-agents`) holds a Next.js app with:

- **Workflow DevKit** runs `leadWorkflow` (one per lead, holding the hook `lead:<lane>`), `workerWorkflow` (one per
  worker) and `rotCheckWorkflow`.
- **Vercel Sandbox** holds one persistent named sandbox per lead (`lead-<lane>`) and per worker (`worker-<id>`). It is
  stopped (snapshotted) when idle and resumed on the next turn.
- **The AI SDK harness** (`@ai-sdk/harness` with the Claude Code and Codex adapters, and `@ai-sdk/workflow-harness`)
  runs one turn as time-boxed slices.
- **Postgres** (Neon) holds the record. Tests use PGlite.
- **The AI Gateway** carries every model call. Each lead may have its own key with a monthly quota; otherwise calls
  use the project's OIDC token.
- **Vercel Cron** calls `/api/cron/tick` every minute and `/api/cron/rot-check` weekly.

Outside the project, it reuses Slack (an app of its own for events, Verity's bot for posting), `research msg` and the
inbox routes in the notes repository, a Linear webhook, the GitHub broker (for App tokens) and the jobs API (for heavy
work).

The HTTP surface (`src/http.ts`, mounted under `app/api/`):

| Route | Who calls it | Auth |
| --- | --- | --- |
| `POST /api/slack/events` | Slack | Slack's signature (`SLACK_SIGNING_SECRET`), five-minute window |
| `POST /api/linear/webhook` | Linear | Linear's signature (`LINEAR_WEBHOOK_SECRET`), one-minute window |
| `GET /api/cron/tick`, `GET /api/cron/rot-check` | Vercel Cron | `CRON_SECRET` |
| `POST /api/jobs/callback/<token>` | A job runner, optionally | The URL's per-job token (a capability for one job) |
| `GET`/`POST /api/leads` | `swap.ts`, `smoke.ts` | `PLATFORM_API_TOKEN` |
| `GET /api/transcripts` | People, scripts | `PLATFORM_API_TOKEN` |
| `POST /api/mcp` | MCP clients | `PLATFORM_API_TOKEN` |

### 3.3 A lead

`leadWorkflow` (`workflows/lead.ts`) claims the hook `lead:<lane>` with `experimental_force`, so the newest run always
owns it. It records its run id, then drains:

1. **`beginLeadTurn`** (a step) takes the lead's lease and claims every pending event. It records the instructions and
   the prompt as the turn's first transcript parts, opens the sandbox, and writes `WORKSTREAM.md` from the database. The
   database is the doc's source of truth.
2. **`sliceTurn`** (a step, repeated while the harness says `ready_for_next_step`) reopens the sandbox with freshly
   issued credentials and builds a `HarnessAgent` for the lead's harness and model, with three host tools. It runs one
   time slice (`runHarnessAgentTimeSlice`, `PLATFORM_SLICE_SECONDS`, 600 by default), stores the slice's transcript parts
   and usage, and renews the lease.
3. **`endTurn`** (a step) reads `WORKSTREAM.md` back as a new doc revision and prices the usage. It replies once in every
   Slack thread the events came from, with `research msg reply` run in the sandbox (Section 3.5); a tick's turn posts
   nothing. It answers each lead that asked, and releases the lease.
4. **`parkLead`** stops the sandbox once the inbox is empty. The run then waits on its hook.

After 25 turns a run starts its successor on the latest deployment and ends, which bounds each run's event log and moves
leads onto new code. A newer run claiming the hook ends the older one (`HookForceClaimedError`).

**Waking** (`src/wake.ts`) inserts the event, deduplicated by key, then calls `resumeHook`. If no run holds the hook,
it starts one, unless the recorded run is still pending or running (it registers its hook before its first step, and
will drain the event).

**The harness session is disposable.** Nothing carries over between turns but the doc, the sandbox's files (the
checkout, caches) and the record. A turn that fails says so in the thread ("This turn failed (…). Its transcript is
…"), and the next event starts clean.

### 3.4 Workers, jobs and asks

The lead's host tools (`src/tools.ts`) run in the step, outside the sandbox. Each tool's ids derive from the turn and
the tool call, so a retried step finds what it already made.

- **`dispatch_worker(task, branch?, harness?, model?)`** records a worker and starts `workerWorkflow`. The worker is one
  harness turn in its own sandbox, whose checkout sits on its own branch (`cursor/<lane>-<hex>` by default), with GitHub
  credentials only. Its report wakes the lead in the thread that asked (`worker-done`).
- **`submit_job(command, requirements?, leaseSeconds?)`** submits a `research-run` job to the jobs API with a callback
  URL. The job's end wakes the lead (`job-done`), from the callback or from the cron reading the job's state from the
  queue, whichever comes first, once.
- **`ask_lead(lane, question)`** puts an `ask` event in the other lead's inbox. Its answer comes back as an `answer`
  event in the asker's thread.

### 3.5 Slack and MCP

**Slack works as it does on Cursor, and the platform has no routing rule of its own.** Who a post is for is
`research msg`'s rule (`route_targets`, pinned by research's routing vectors), applied by whoever sends the post: it
rings a doorbell in each addressed handle's inbox thread in `#agent-coordination`. People's posts in `#ask-the-team`
and the chat channels are rung the same way by the Slack events service. A lead on Cursor subscribes to its inbox
thread; a lead on the platform is registered with it (`swap register --channel C --inbox TS`), and any reply there
wakes it, except a post under the lead's own name or the platform's.

A doorbell (`from AUTHOR, [#CHANNEL ]thread T[, reply R]: TEXT`) names the thread the post is in, and the turn's final
message is posted there with `research msg reply --as LANE`, run in the lead's sandbox. So the reply rings whoever
research's rule says, a lead on Cursor included, and both kinds of lead can talk while some have moved. The lead uses
`research msg` for everything else (asks, announcements, closing an ask, Linear), as its skills say. A reply research
refuses (over 120 words in the front door, say) is recorded and handed back to the lead once.

A lead holds its handle's inbox route under a stable agent id (`swap id LANE`, a UUID derived from the lane), which
its sandbox exports as `RESEARCH_AGENT_URL`, so `--as LANE` passes research's holder check. Moving a lead is
`research msg names open --as LANE --succeeds --holder ID`, which keeps the inbox thread and its unread doorbells.

**Linear.** A lead on Cursor subscribes to its Linear team's issue creations and state changes, never comments.
Linear's webhook delivers the same to `/api/linear/webhook`, which wakes the lead registered with the issue's team key
(`--linear KEY`).

The MCP server (`src/mcp.ts`, stateless streamable HTTP) has five tools:

- `ask_lead` queues a question and can wait up to 50 s for the answer;
- `assign` hands a lead a task, answered in a given Slack thread or read back with `get_answer`;
- `get_answer` reads what became of an event;
- `status` reports each lead's pending events, running turn, last reply, running workers, and month-to-date spend
  against its budget, with the gateway's own figure for the lead's key;
- `search_transcripts` searches by lead, time window and text.

The APIs research and the lead call (Slack's, Linear's and Notion's, as granted per lead) go through the sandbox
firewall like GitHub's (Section 6).

## 4. Requirements and pain points, mapped

| Need | Where the design meets it |
| --- | --- |
| R1, P2 | `swap register`, `all`, `pause`, `resume`, `roll` and `doc` (`scripts/swap.ts`) over `/api/leads`; a lead's `model` and `harness` are fields, so moving to a new model is `register LANE --harness codex --model openai/<id>` |
| R2, P7 | Per-turn tokens and cost in `turns`, priced at list price (`src/prices.ts`, from the gateway's model list); per-lead gateway keys with a monthly quota that the gateway enforces; `status` shows the platform's figure beside the gateway's |
| R3, P5 | `src/credentials.ts` and `src/services.ts`: tokens are issued per sandbox open, recorded in `grants` by digest or key id, and injected at the firewall |
| R4 | `transcript_parts` with a generated `tsvector` and a GIN index, read through `/api/transcripts` and MCP `search_transcripts` |
| R5 | `@ai-sdk/harness` with the Claude Code and Codex adapters; the platform's code schedules turns and slices and runs no model loop |
| R6, P1, P4 | Each event is a row before anything else happens; a lease means one turn at a time per lead; a wake never waits on a running turn; events wait in the inbox and nothing lapses |
| R7, P3 | `workerWorkflow`; a worker's failure is reported to its lead like its success; workers are runs, not agents, so no seat cap applies |
| R8, P6 | `submit_job`, with the instructions forbidding heavy work in the sandbox; job ends arrive by callback or by polling |
| R9 | Doorbells in each lead's inbox thread (`/api/slack/events`) and replies through `research msg`, as on Cursor; Linear issue events (`/api/linear/webhook`); and `/api/mcp` with `ask_lead`, `assign`, `get_answer`, `status`, `search_transcripts` |
| R10 | The cron's recovery rules (Section 9); visible failed turns; force-claimed hooks |
| R11 | One project and one database; exact pins; `scripts/rot-check.sh`, run weekly in a sandbox by `rotCheckWorkflow` |

## 5. Data model

Every table lives in `src/schema.ts`, which runs idempotently on each cold start. Workflow run data is kept for days,
and this record is what lasts.

| Entity | Table | What a row is |
| --- | --- | --- |
| Lead | `leads` | One lead: lane, charter, doc and `doc_rev`, harness, model, inbox (channel and thread ts, unique), Linear team key (unique), granted `apis`, cron interval and next tick, monthly budget, gateway key id and sealed key, status (`active`, `paused`, `retired`), current run id, lease |
| Inbox event | `events` | One thing for a lead: source (`slack`, `linear`, `cron`, `job`, `worker`, `mcp`, `lead`), kind, payload, `reply_to` (a Slack thread, an asking lead, or an MCP caller), `dedupe_key` (unique), `turn_id` once claimed; pending events are indexed |
| Turn | `turns` | One harness turn of a lead or worker: run id, status, harness, model, start and end, token counts by kind, `cost_usd`, final text, error, doc revision before and after |
| Transcript | `transcript_parts` | One part of a turn, in order: instructions, prompt, reasoning, text, tool call, tool result, file change, error, finish, post, post error; JSON content plus searchable text |
| Doc history | `doc_revisions` | Every version of a lead's doc, with the turn that wrote it (or `admin` for an import or replacement) |
| Usage | columns of `turns` | Input, output, cache-read, cache-write and reasoning tokens, and their list-price cost; the gateway's per-key spend is read live |
| Credential grant | `grants` | One issuance: subject (lane or worker id), kind (`github-app-token`, `api:slack`, `ai-gateway-key`, …), scope, a reference (key id or a 16-hex SHA-256 prefix, never the value), issue and expiry times |
| Worker | `workers` | Task, branch, harness, model, status, run id, where its report goes, its result |
| Job | `jobs` | Spec, status, callback token, the queue's job id, where its outcome goes, result |
| Alert | `alerts` | A raised alert's key, so each alert is posted once |

## 6. Security

**No long-lived credential in a sandbox.** A sandbox's environment and files hold only placeholders, such as
`platform-github-placeholder` in git's credential helper and as `RESEARCH_NOTES_TOKEN`, and
`SLACK_BOT_TOKEN=platform-slack-placeholder` and `LINEAR_API_KEY=platform-linear-placeholder` in `~/.verity-env`. The network policy is allow-listed by host. For each
brokered host it carries a header rule: a request whose `authorization` equals the placeholder form has it replaced
with the real value outside the sandbox. Each rule applies only to its host. A request with any other value passes
unchanged, and the real value never enters the sandbox. The rules are replaced on every sandbox open, so the previous
slice's tokens drop out. This was checked live on Vercel Sandbox: Bearer and git Basic swaps, other values unchanged,
unlisted hosts unresolvable, the swapped value absent inside, and the API reporting rules by header name only. The
platform installs its rules through the harness's own policy manager. Otherwise the harness refuses to add its model
rule, a defect reproduced in `test/sandbox-policy.test.ts`.

**Model access.** By default a turn calls the gateway with the deployment's OIDC token, fetched fresh for each slice.
A lead with a budget has its own gateway key, created with a monthly quota and kept AES-256-GCM-sealed in `leads` under
a key derived from `PLATFORM_SECRET`. The harness puts the token into the gateway rule, so the sandbox holds a
placeholder here too.

**GitHub.** For every sandbox open the platform asks the broker for a one-hour `verity-agents` App token. It sends the
project's OIDC token and names the lead or worker in `X-Verity-Agent`. A Vercel OIDC subject names the project, not the
lead, so the broker authenticates the project and only attributes the agent. Our code sets the header from the
lead's lane; the model never chooses it.

**APIs per lead** (`src/services.ts`). Slack's Web API, Linear and Notion are brokered the same way, and so are the
notes repository's raw files, under the GitHub token. A lead gets the APIs in its `apis` field (Slack and Linear by
default, Notion only when granted), and a worker gets GitHub only. The Slack token is Verity's bot's, since
`research msg` tells its own posts and doorbells by that bot's id. That gives a platform lead the bot's full scopes,
where a lead on Cursor reaches Slack through the Slack broker's narrower set of methods. Header rules match a host,
not a path, so narrowing this needs the broker to accept the platform's identity (Section 12).

**What an agent can read and do.** It can read its sandbox: the doc, the checkout and its own files. It can reach the
allow-listed hosts: the gateway, GitHub, npm, PyPI, and its granted APIs. Through its host tools it can start workers,
submit jobs and ask leads. It cannot read the database, another lead's doc or sandbox, the platform's environment or
any real credential. Two limits are inherent to an agent that codes:

- it can push to any branch the App may write, so only branch protection holds `main`;
- it can send anything it reads to an allow-listed host, such as a public branch or a Slack post.

**Inputs are untrusted.** Anyone in the workspace can wake a lead, and the budget bounds what a wake can spend. Slack
and Linear requests are signature-checked, job callbacks carry a one-job capability, and the cron route takes
`CRON_SECRET`. A doorbell's text is the sender's: the lead reads it as `research msg` lets any agent read it, and the
using-slack skill's `verify-author` applies before acting on a request to change what runs.

**The admin token.** `PLATFORM_API_TOKEN` guards `/api/leads`, `/api/transcripts` and `/api/mcp`, and transcripts can
hold whatever a tool printed. Per-caller tokens are an open item (Section 12).

**Where the secrets live.** They are in the platform project's environment: `DATABASE_URL`, `PLATFORM_SECRET`,
`PLATFORM_API_TOKEN`, `CRON_SECRET`, the Slack app's signing secret and Verity's bot token, the Linear webhook's secret,
`VERCEL_TOKEN`, the jobs token, and `LINEAR_API_KEY` and `NOTION_API_KEY` if used. Anyone who can deploy the project can read them, so the project deploys from one
protected branch.

## 7. Cost per lead per month

Prices are list prices, read from public pages (sources below). The usage figures are assumptions for a busy lead, not
measurements. Cursor's cost for the same work is unmeasured, since that needs a Cursor admin key.

| Item | Assumption | Per month |
| --- | --- | --- |
| Lead turns, Opus 5.5 | 40 turns a day (1,200 a month); each turn makes 10 model calls over a 30k-token context, 85% of it read from the prompt cache and the rest written to it, plus 8k output tokens; about \$0.44 a turn | \$523 |
| Lead sandbox | 2 vCPU and 4 GB, running 3 minutes a turn at 10% CPU; about \$0.0055 a turn | \$7 |
| Lead snapshots | Three kept, about 3 GB each | \$1 |
| Worker model, Opus 5.5 | 90 worker-hours a month; each hour makes 60 calls over a 60k-token context, 90% cached, with 1.5k output tokens a call; about \$4.25 an hour | \$382 |
| Worker sandboxes | 4 vCPU and 8 GB at 25% CPU; about \$0.30 an hour | \$27 |
| Workflow | About 31k events and 0.1 GB written | \$1 |
| Functions | Steps hold about 1 GB while a slice runs, plus the minute cron (shared) | \$2 |
| Database | A share of a small Neon database (about \$15 a month for all leads) | \$1 |
| **Total** | | **≈ \$944**, of which the platform is ≈ \$39 (4%) |

For 15 leads this is about \$14,200 a month, or \$170,000 a year, plus the Pro plan's \$20 a month (which includes
\$20 of usage credit). Model tokens are 96% of it, so the levers are the model and turns per day. The same lead turn
on Sonnet 5.5 or `gpt-6.1-sol` costs about \$0.22, half as much. The gateway adds no markup.

Sources:

- AI Gateway: no markup on provider list prices, <https://vercel.com/docs/ai-gateway/pricing>. Per-model rates
  (dollars per million tokens: input, output, cache read, cache write) come from the gateway's model list,
  <https://ai-gateway.vercel.sh/v1/models>:
  - Opus 5.5: 4, 20, 0.2, 5;
  - Sonnet 5.5: 2, 10, 0.1, 2.5;
  - `gpt-6.1-sol`: 2, 10, 0.1, 2.5;
  - `gpt-5.3-codex`: 1.75, 14, 0.175.
- Sandbox: \$0.128 per active-CPU hour, \$0.0212 per GB-hour of provisioned memory, \$0.60 per million creations, and
  snapshots at \$0.08 per GB-month, <https://vercel.com/docs/sandbox/pricing>.
- Workflow: \$0.02 per thousand events, and \$0.50 per GB written and per GB-month retained,
  <https://vercel.com/docs/workflow/pricing>.
- Functions: \$0.128 per active-CPU hour, \$0.0106 per GB-hour, and \$0.60 per million invocations,
  <https://vercel.com/docs/functions/usage-and-pricing>.
- Pro plan: \$20 a month with \$20 of credit, <https://vercel.com/docs/plans/pro-plan>.
- Neon: <https://neon.com/pricing>.

## 8. Plan limits that shape the design

These are Vercel's documented limits as read for this design, and they are why the team must be on Pro:

- **Cron.** Hobby runs crons once a day; the tick needs every minute.
- **Functions.** Hobby allows 300 s per function and Pro 800 s; a slice is 600 s.
- **Sandboxes.** A session is capped at 45 minutes on Hobby and 24 hours on Pro (`PLATFORM_SANDBOX_TIMEOUT_MINUTES`).
  Pro allows up to 8 vCPU and 16 GB.
- **Workflow.** A run allows 25,000 events and 10,000 steps; 25 turns a run stays far below either. Run data is kept
  7 days on Pro, which is why the record is ours.

Lean builds don't fit a sandbox: they peak near 15 GB against 16 GB at most, so Lean always runs as a job.

## 9. Failure modes, and how each is detected without a human

| Failure | Detected by | What happens |
| --- | --- | --- |
| A lead's run is gone (failed, cancelled, expired) | The next wake's `resumeHook` finds no hook | The wake starts a fresh run, which drains the inbox |
| Events wait and nothing runs | The tick: pending events older than 15 minutes, with no lease held | A fresh run is started and an alert posted once |
| An old or stuck run still holds the hook | A newer run claims it | The old run ends as `superseded` |
| Two runs at once | The lease | Only the lease holder takes events; the other waits |
| A turn runs too long | The tick: a turn running over 120 minutes | An alert is posted once; each slice is bounded by `PLATFORM_SLICE_SECONDS` |
| The harness crashes, or the model errors | The slice's status is `failed` | The turn is recorded as failed and the thread is told, with the error and the transcript id |
| A reply isn't posted (research refuses it, or Slack does) | The post's exit status | A `post-error` part records why, and the lead is handed a `reply-failed` event once; a turn on `reply-failed` alone hands back nothing |
| The sandbox's bootstrap fails (the clone, or installing uv or `research`) | The step throws | The turn fails; the bootstrap's marker is written only on success, so the next session runs it again |
| A lead turn fails and no thread hears of it (a tick's turn, or a notice that couldn't be posted) | The tick: a failed lead turn with no `post` part | An alert is posted once, with the error |
| A lead's inbox route moves (its handle opened again elsewhere) | **Not detected** | Doorbells ring the new inbox and the platform lead hears nothing (Section 12) |
| A step's function dies mid-slice | The workflow runtime | The step is retried from the last recorded state; tool ids, posts and events are idempotent |
| A worker fails | `workerWorkflow` ends | Its lead is woken with `worker-done: failed` |
| A job ends without calling back | The tick reads each open job's state from the queue | The lead is woken once |
| A budget is exhausted | The gateway refuses the key | Failed turns say so; `status` shows spend against budget |
| A sandbox or its snapshot is lost | `getOrCreate` makes a new one | The bootstrap re-clones; the doc is in the database, so nothing but unpushed files is lost |
| A token expires mid-turn | Nothing needs to | Tokens are issued per slice, and a slice is shorter than a token's hour |
| A run grows too long | `TURNS_PER_RUN` = 25 | The run hands over to a successor |
| Code drifts from the current deployment | Turns-per-run, and the tick rolling a run idle for 24 hours | Leads move to the latest deployment |
| An SDK release breaks the platform | The weekly rot check: a fresh sandbox installs every dependency at `@latest`, typechecks and runs both suites | The result is posted to the alert channel |
| A Slack event is lost while the platform is down | Slack retries three times | **Not detected after that** (Section 12) |
| A Linear event is lost while the platform is down | Linear retries | **Not detected after that** (Section 12) |
| Vercel itself is down | Nothing on Vercel | **A probe from outside Vercel is needed** (Section 12) |

## 10. Cutover and rollback

`SWAP.md` has the exact commands. The estimate, with every key and account below in hand:

- **Setup, once, about 35 minutes.** Link the project, create the database, set the environment, deploy, point the
  platform's Slack app at `/api/slack/events` and a Linear webhook at `/api/linear/webhook`, and run the fake-harness
  smoke and then a real-harness smoke.
- **One lead, about 5 minutes.** The Cursor lead writes its handoff. `swap id LANE` prints the lead's stable agent id,
  and `research msg names open --as LANE --succeeds --holder ID` moves the handle's inbox route to it, keeping the
  inbox thread, whose channel and ts it prints. `swap register` imports the handoff as the doc, sets the inbox, Linear
  team, model, budget and APIs, starts a run and kicks it. A `research msg send --to @LANE` then gets its reply in its
  thread, and the Cursor agent is archived.
- **All leads, about 10 minutes more** with `swap all LEADS.json`, built from the handle registry and the inbox routes.

Rollback of one lead takes about 5 minutes: `swap pause LANE` stops the platform taking its events, `swap doc LANE`
exports the doc, and `launch_coordinator.py launch` with `--text-file` relaunches it on Cursor from Daniel's laptop;
its successor's `names open --succeeds` takes the inbox route back. Undoing a pause is `swap resume LANE`.

## 11. What retires

These retire once all leads have moved:

- `infra/cursor/` (the coordinator launcher, its fixed ids and generations, and the handover ritual).
- `tools/agent-guard/`: the shell, Task, env, git and worktree hooks, the OOM recorder and the inbox. The sandbox's
  allowlist and the absence of credentials replace them, and so does the inbox table.
- Cursor's Slack and Linear subscriptions, timers and their renewal loops.
- The agent watch's Cursor rules: held runs, runs stuck in CREATING, refused wakes and relaunch.
- The chats archive (replaced by `transcript_parts`), and the Cursor team secrets that reach every VM.

These stay: the jobs API and `research run` on pods, the GitHub broker, drive, the notes repository, Linear, Notion
and Slack.

## 12. Open questions

1. **Slack.** Recommended: a Slack app of the platform's own that only receives events (`channels:history` in
   `#agent-coordination`), while every post goes out under Verity's bot through `research msg`, so research's own-post
   and doorbell tests hold unchanged. The other option is pointing the team app's events at the platform, which takes
   them from the slack-events service.
2. **The broker.** It must trust the platform project's Vercel OIDC subject. A project-level identity lets any code in
   the project mint App tokens for any lead. Deploying only from a protected branch bounds that, and one project per
   lead removes it at the cost of a project each. The Slack broker accepting the same identity would let a platform
   lead's Slack calls go through its narrower set of methods, instead of the bot token's full scopes (Section 6).
3. **The `research` CLI.** It is installed in each lead's sandbox, and `research msg` works there through the
   brokered Slack token and the notes repository's raw files (read under the GitHub App token, which must be able to
   read the notes repository). Subcommands that go through the site's brokers (`research run`, `research goals`)
   need a Cursor or Vercel OIDC token, which a sandbox doesn't hold by design; a lead submits heavy work with
   `submit_job` instead. Which of those subcommands leads need is open. Unverified live: research in a Vercel
   sandbox.
4. **Linear identity.** A brokered personal key files every issue as that person. The Linear app's client
   credentials (`createAsUser` per handle) need the host to mint the app token, which is a small change in
   `credentials.service`.
5. **Inbox routes.** The platform reads a lead's inbox from its registration, not from the notes repository's routes.
   If the handle is opened again elsewhere, its doorbells ring the new inbox and the platform lead hears nothing. The
   tick should compare each lead's registration with its route and alert on a difference.
6. **Uptime.** The control pod's watch should probe the platform from outside Vercel, and a missed Slack or Linear
   event should be reconciled (Slack's `conversations.replies` on each inbox thread, Linear's issues by update time).
7. **Harness endpoints.** Should the adapters use the gateway's dedicated endpoints (`/claude-code`, `/codex/v1`) or
   their defaults? The live smoke shows which works.
8. **Codex and new OpenAI models.** Does the Codex adapter accept a model the day it ships on the gateway? The swap is
   one command, but this is unverified.
9. **Per-caller admin tokens** and a read-only scope for transcripts.
10. **Per-lead attribution of Sandbox and Functions cost.** Vercel bills per project, so the platform's per-lead figure
    covers model spend only.

## 13. Keys and accounts Daniel provides

Names only. Values go into the project's environment with `vercel env add`, never into a message.

- **A Vercel team on Pro** with AI Gateway billing, and in it a project `platform-agents` deployed from one protected
  branch. Per-minute cron, 800 s functions and sandbox sessions over 45 minutes need Pro.
- **`VERCEL_TOKEN`**, a token belonging to a user who can manage the team's AI Gateway keys and budgets, and
  **`VERCEL_TEAM_ID`**.
- **`DATABASE_URL`**, a Postgres database (a Neon project or branch).
- **A Slack app for the platform that only receives events**: its **`SLACK_SIGNING_SECRET`**. It needs the
  `channels:history` scope and the `message.channels` event, sent to `/api/slack/events`, and its bot is invited to
  `#agent-coordination`. It never posts.
- **`SLACK_BOT_TOKEN`**: Verity's own bot token, held by @infra, which `research msg` posts under and tells its own
  posts and doorbells by. The platform brokers it into each lead's sandbox and posts its alerts with it.
- **`LINEAR_WEBHOOK_SECRET`**: the signing secret of a Linear webhook on Issue events, pointed at `/api/linear/webhook`.
- **The GitHub broker trusting the project's OIDC subject**, and its URL as **`PLATFORM_GITHUB_BROKER_URL`**. The App
  token it mints must be able to read the notes repository, where `research msg` reads the inbox routes.
- **The jobs API**: a token with `jobs:enqueue` (**`PLATFORM_JOBS_TOKEN`**), **`PLATFORM_JOBS_URL`**, and a budget line
  **`PLATFORM_JOBS_LINE`**.
- **`LINEAR_API_KEY`** (or the Linear app's client credentials, Section 12) and **`NOTION_API_KEY`**, for the leads
  granted them.
- **A budget line** for Vercel and the gateway, with an alert.
- **Optional:** **`CURSOR_ADMIN_API_KEY`**, to measure what Cursor costs for the same work.
- **For the live smoke tonight:** re-authorize the Vercel connection for the scope that holds `platform-smoke`, or
  provide a `VERCEL_TOKEN`.

Whoever deploys generates `PLATFORM_SECRET`, `PLATFORM_API_TOKEN` and `CRON_SECRET`. Configuration that isn't secret:
`PLATFORM_SANDBOX=vercel`, `PLATFORM_SANDBOX_TIMEOUT_MINUTES`, `PLATFORM_REPO_URL` (required on Vercel, since replies
are posted by `research` from the checkout), `PLATFORM_ALERT_CHANNEL`, `PLATFORM_DEFAULT_MODEL`, and
`PLATFORM_ROTCHECK_REPO`, `_REF` and `_DIR`.

## 14. Versions and the rot check

Every dependency is pinned exactly in `package.json`, and `package-lock.json` is committed. The harness packages move
together, since they release about daily and warn of breaking changes. `scripts/rot-check.sh` prints the pins, installs
every dependency at `@latest`, prints what resolved, then typechecks and runs both suites. `rotCheckWorkflow` runs it
weekly in a fresh sandbox and posts the result. An upgrade is a change to the pins that passes `npm run check`.

## 15. Tests

`npm run check` runs the typecheck, the unit suite (26 tests) and the end-to-end suite (21 tests, about 10 seconds).
The unit suite parses every doorbell in research's own routing vectors
(`tools/research/tests/slack_routing_vectors.json`), so a change to the doorbell format fails here. The end-to-end
suite runs the whole platform in one process, with:

- real Workflow DevKit runs (the local world);
- the real `HarnessAgent` driving a scripted harness (`src/harness/fake.ts`) in directory sandboxes;
- PGlite, and an in-memory Slack and job queue.

It covers:

- doorbells waking a lead, and its reply in the doorbell's thread, a channel's included;
- only the inbox waking a lead, and none of its own or the platform's posts;
- Linear issue events, signed and routed by team;
- dedupe and signatures;
- a refused reply handed back once;
- cron ticks posting nothing, and jobs ending by callback and by polling;
- workers and asks between leads;
- MCP, transcripts and usage;
- credentials kept out of sandboxes and transcripts, and the sandbox's `research` environment;
- run supersession;
- the admin endpoint, including an inbox another lead holds;
- checkouts;
- recovery: a failed turn, stuck events, a stale turn, an idle roll, and a failure no thread heard of.

`npm run smoke` drives a running platform over its public API: an MCP question, the turn's transcript, usage and doc.
With `--signed` it also signs what Slack and Linear would send: posts that must wake nothing, a doorbell whose reply
is addressed to its own thread, and an issue of the lead's team. Its channel is made up, so on a deployment the reply
is refused by Slack rather than posted. Against a local `next start`, unset `SLACK_BOT_TOKEN`, `LINEAR_API_KEY` and
`NOTION_API_KEY`, or the local server uses real credentials. `smoke/live-turn/` is a one-function deployment that runs
one real harness turn in a Vercel Sandbox and checks by digest that the model credential is in no process environment
or file inside it.
