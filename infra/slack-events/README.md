# Slack events service

A Vercel function that takes Slack's Events and Interactivity requests, puts each onto a Vercel queue, and routes it
with `research msg`'s own code (`tools/research/src/research/slack.py`, copied in at build time). A person's post in a
chat channel (`#chat-NAME`), `#ask-the-team` or `#resource-accounting` rings the lanes it names. A post that names no
one goes to the admin, with a routing line in its thread (`consumer.NO_MODEL`) saying no lane was named and routing by
model is off. In a person's chat, their own post that names no one goes instead to the lane their people.d record
names (`lane`, set by `research msg chat open|expect --lane HANDLE`), with a routing line saying it is their usual
lane; a lane the registry doesn't have falls back to the admin. Nothing is routed in `#agent-coordination`, where the sender rings its own doorbells. A newcomer whom
`research msg chat expect` published gets their chat when they join the workspace. The service keeps no database:
Slack is the record (below).

## Routes

Every route is `api/index.py`, the FastAPI app `slack_events/app.py` builds; `service/vercel.json` rewrites
`/api/slack/*` and `/api/cron/*` to it and schedules the crons.

| Route | Caller | What it does |
|---|---|---|
| `POST /api/slack/events` | Slack's Events API | Checks Slack's signature (HMAC SHA-256 of `v0:{ts}:{raw body}` with `SLACK_SIGNING_SECRET`, at most 5 minutes old), answers `url_verification`, drops our own bot's posts, publishes the event to the `slack-events` topic keyed on its `event_id`, and answers 200, or 503 when the publish fails so Slack retries. Nothing runs after the answer. |
| `POST /api/slack/interactivity` | Slack's Interactivity | The same check, then the payload onto the same topic, keyed on its action (not handled yet: Needs). |
| `GET /api/slack/health` | whoever holds `HEALTH_TOKEN` | The numbers under Health. |
| `GET /api/cron/drain` | Vercel cron, every minute, `CRON_SECRET` | Consumes the topic (`subscriber.drain`, below). |
| `GET /api/cron/watchdog` | Vercel cron, every 5 minutes, `CRON_SECRET` | Rings again a chat post with no 👀 and no lane reply after 15 minutes (the names of its routing line, else the rule's), and the admin as well, once, when that line sent it to the person's usual lane; rings the admin and says so in the thread after 60; every doorbell it rings is URGENT (a person is waiting); once a day (16:00Z) posts the open chat threads to the admin's inbox; rings `crons.STALL_TO` when the queue stalls, and publishes the hourly canary (Health). |
| `GET /api/cron/reconcile` | Vercel cron, every 10 minutes, `CRON_SECRET` | Re-enqueues, keyed on channel and ts, a person's post of the last hour that should ring someone and has neither a routing line nor a doorbell. |

## Modules

All under `service/`, the Vercel project's Root Directory: Python 3.13 on Fluid Compute, `framework: null`, installed
from `service/pyproject.toml` and its `service/uv.lock`, which hashes every package (the uv workspace excludes
them, so Vercel installs them alone; `uv lock` in `service/` after a pin moves).

| Module | What it is |
|---|---|
| `api/index.py` | Vercel's function: the app, for every route. |
| `slack_events/app.py` | The HTTP side: signature and bearer checks, publishing to the topic, health, the cron routes. |
| `slack_events/subscriber.py` | The queue's consumer in poll mode: `drain` claims messages as consumer group `slack-events-drain` until the topic is empty or 45 s have passed. |
| `slack_events/consumer.py` | One message: route a person's post (`deliver`, routing lines), dead-letter after 5 deliveries, the canary, a newcomer's `team_join`. |
| `slack_events/crons.py` | The watchdog (reminders, escalation, the daily list, `stall`, `canary`) and reconciliation. |
| `slack_events/router.py` | `route_unaddressed`, the seam for routing by model: None (no-model mode) unless `ROUTER_MODEL_ENABLED=1`; its docstring is the plan for the AI Gateway call, which isn't built. |
| `slack_events/deps.py` | What the service runs against (routing code, registry, queue, Slack, cache, clock): `production()` from the environment, fakes in the tests. |
| `slack_events/vendor.py` | Loads the routing code and registry: `_vendor/` in a deployment, the repository's own files from a checkout. |
| `build.py` | The build step: copies `slack.py` and `.agents/skills/using-slack/registry.json` into `_vendor/` (never committed), so a deployment runs `main`'s routing code byte for byte. |
| `ignore.py` | Vercel's `ignoreCommand`: build only `main`, and only when the service's inputs changed. |

`tests/` is this directory's suite (`pyproject.toml`, a workspace member with the same pins as `service/pyproject.toml`).
It runs against fakes: Slack through `Slack`'s transport, the cache, the queue, the notes repository's contents API and
the clock. No test calls Slack, Vercel, GitHub or a model.

## Where to look

| Concern | Where |
|---|---|
| Who a post rings | `research msg deliver` in `slack.py`; the service adds only the routing line (`consumer.route`, `consumer.pick`) |
| A person's usual lane | `research msg chat open\|expect --lane`; `consumer.pick` (`usual_lane`), `crons.watchdog` |
| Retries, redelivery, dead letters | `consumer.consume`, `consumer.dead_letter`; Slack is the record, below |
| Reminders, escalation, the daily list | `crons.watchdog`, `crons.daily` |
| A stalled queue | `crons.stall`, `crons.canary`; Health, below |
| Posts that were never routed | `crons.reconcile` |
| Routing by model | `router.py` |
| Newcomers | `consumer.newcomer`; Newcomers, below |
| Signatures, bearers | `app.signed`, `app.bearer` |
| Building and deploying | `build.py`, `ignore.py`, `vercel.json`; Deploy, below |

Logs are JSON lines of ids, decisions and errors. They never carry a post's text.

## Slack is the record

The service keeps no database. What it has routed is what it posted in Slack, which it and the crons read back:

- **A crash.** The queue keeps a message invisible while its handler runs (the SDK renews the lease) and redelivers it
  when the handler fails or its function dies. `event_id` is the publish's idempotency key, so Slack's retries of one
  event are one message.
- **A post that names no one.** Before ringing anyone, the consumer reads the post's thread (with its messages'
  metadata) for its own routing line: posted by `router` on Verity's bot, its metadata `verity_routing_line` with the
  post's ts and the names it rang. Finding one, it rings those names, with no second pick and no second model call;
  otherwise it picks, posts the line, then rings. `deliver`'s once-per-post check skips a lane already rung, so a
  redelivery after the line rings only the lanes still missing, and one before the line starts from scratch.
- **A post routed by its leading names** needs no line: its doorbells in the lanes' inboxes are its record.
- **A route that can't be read** (the published `routes.d` copy answering an error, or no JSON) leaves its doorbell
  NOT RUNG, so the delivery fails and is retried, like a failed post; only a lane read as having no route is logged as
  having no inbox and left.
- **Reconciliation** reads the same two records: a post that should ring someone and has neither is enqueued again.
- Where the bot can't read a channel's history, a redelivery can't find the line: it picks again and may post a second
  line, though `deliver` still rings each lane at most once. Invite the bot to `#ask-the-team` and `#resource-accounting`.

## Health

`GET /api/slack/health` keeps its keys; null means unknown. Each value comes from Vercel's Runtime Cache (read-through,
best effort): the cache is regional and may drop a value at any time, so a cold cache reads as unknown, never as down.

- `last_success_at`: when a delivery or a cron last finished.
- `last_done_at`: when the consumer last finished a message (no cron writes it), and `last_enqueued_at`: when the app
  last published an event. An enqueued time after the done time means events wait. The watchdog rings the service's
  owner (`crons.STALL_TO`) when that has lasted 10 minutes, at most hourly (`crons.stall`), since `last_success_at`,
  which the crons keep fresh, looks healthy while nothing is consumed.
- `canary_sent_at` and `canary_done_at`: each hour the watchdog publishes a synthetic `canary` event, stamped as an
  enqueue, which only the consumer can mark done. So the stall check proves the drain on a day no person posts.
- `oldest_unprocessed_age_s`: the age now of the oldest unrouted post reconciliation found on its last run (0 when it
  found none); unknown once that finding is 30 minutes old.
- `dead_letter_count` and `dead_letter_since`: dead letters counted since `since`. The client can't count a topic's
  messages, so the consumer counts each dead letter itself (not atomic: two at once may count as one); an evicted count
  starts again at 0 on reconciliation's next run, with a new `since`.
- `queue`, `dead_letter`: the topics. `note` says the above.

## The queue: polled, not pushed

Vercel pushes a Python subscriber only from its Python framework build, which compiles a `[[tool.vercel.subscribers]]`
entry in `pyproject.toml` into a queue-triggered function ([Python SDK](https://vercel.com/docs/queues/python-sdk),
"Consuming messages in push mode"); a `vercel.json` `experimentalTriggers` entry is documented for JavaScript functions.
This project's build (`framework: null`, `buildCommand`, `api/index.py`) compiles neither, so a push subscriber would
never run: production would publish every event and consume none while health looks clean. So the service polls:

- `subscriber.drain` calls the SDK's `poll(topic, "slack-events-drain", limit=10, lease_duration=120)`, which
  authenticates as `send` does. Entering a delivery renews its lease while the handler runs and acknowledges it on a
  clean exit; `RetryAfter(delay)` makes it visible again after `delay`; any other exception leaves it to come back.
- A new consumer group reads the topic from its oldest message still retained (24 h), so its first run routes what
  the previous day left unrouted, late.
- `service/pyproject.toml` declares no subscriber: a push consumer beside the drain would be a second consumer group,
  and every post would route twice.
- A post waits for the next minute's drain: up to a minute before its doorbells.

## Newcomers

`research msg chat expect NAME --email E [--note TEXT] [--lane HANDLE]` publishes `people.d/NAME.json` with status `expected`, the email
and no channel; `chat list` shows it, and no command routes by it. When someone joins the workspace, Slack sends
`team_join`. The consumer looks up their email with `users.info` (scope `users:read.email`) and, for an expected
person's email, runs `chat open NAME --user U...` with `--start-here` the onboarding page (`consumer.START_HERE`):
the private channel, the person and Cursor's bot invited, the welcome pinned, the page bookmarked, and the record
published as open. A join of anyone else posts nothing. With no notes clone on Vercel, `slack.py` reads people.d from
the published files and writes it through GitHub's contents API (`RESEARCH_NOTES_TOKEN`). The published files can serve
a write minutes stale: a delivery that fails after the record is written is redelivered, may still read the person as
expected and runs `chat open` again, which reuses the channel and may post the welcome a second time.

## Deploy

Production is `verity-slack-events` (https://verity-slack-events.vercel.app), behind the docs site's rewrite of
`/api/slack/*`. A landing deploys it: Git deployments build only `main`, and only when `infra/slack-events/`, `slack.py` or
the registry changed since the deployment before (`service/ignore.py`, vercel.json's `ignoreCommand`). Console's
`vercel deploy --prod` from main always builds, as the fallback. The project's setup:

1. A Vercel project on this repository with Root Directory `infra/slack-events/service`, and "Include files
   outside the root directory in the Build Step" on (`build.py` reads the repository around it, and fails without it).
2. Vercel Queues on the team (beta, every plan). The topics need nothing: `slack-events` and `slack-events-dead` exist
   once published to, and the drain's consumer group once it polls. Runtime Cache needs nothing either: Vercel gives every
   function one.
3. Environment: `SLACK_SIGNING_SECRET` (the Slack app's Basic Information), `SLACK_BOT_TOKEN` (Verity's bot, as on the
   control pod), `RESEARCH_NOTES_TOKEN` (reads the published `routes.d` and `people.d`, and writes `people.d` when a
   newcomer's chat opens, so it needs write access to the notes repository), `HEALTH_TOKEN` (any long random string;
   whoever polls health holds it) and `CRON_SECRET` (Vercel sends it to the crons). `RESEARCH_ROUTES_URL` overrides
   where the routes are read from.
4. In the Slack app: Event Subscriptions on, Request URL `https://<deployment>/api/slack/events`, bot events
   `message.channels`, `message.groups` and `team_join`. Slack delivers message events only for channels the bot is in:
   invite it to `#ask-the-team` and `#resource-accounting` (it reads their threads for its routing lines, too). Leave
   the Interactivity Request URL where it is (Needs, below).
5. For each person invited: `research msg chat expect NAME --email E` before they join.

## Needs

- **Interactivity is not built.** `#ask-daniel`'s buttons are decided by the website's `/api/agent-approvals` route,
  which only a card's asker can reach (Cursor OIDC), and `ring_decision` needs that route's record. The route here
  accepts and enqueues an action, and the consumer acknowledges it as `interactivity-unhandled` (a TODO in
  `consumer.process`). Point Slack's Interactivity Request URL here only once the decision is readable from this service.
- **Health takes a shared bearer, not Cursor OIDC.** Nothing in this repository verifies a Cursor OIDC token server side
  (the broker's check lives in the website); `HEALTH_TOKEN` stands in until it can be shared.
- **Routing by model is off** (`router.py`).
- **Unverified until a deploy:** the drain's first `done` lines (polling from a Vercel function, a per-minute cron),
  and Runtime Cache's behavior across regions.
