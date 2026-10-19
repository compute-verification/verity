# Slack events service

Slack's Events and Interactivity requests, onto a Vercel queue, routed by `research msg`'s own code (COMM-32). A person's
post in a chat channel (`#chat-NAME`), `#ask-the-team` or `#resource-accounting` rings the lanes it names, as
`research msg deliver --channel KEY --event -` does today on @top's watcher. A post that names no one goes to @top, with
a routing line in its thread: `sent to @top: no lane named; routing by model is off`. Nothing is routed in
`#agent-coordination`: there the sender rings its own doorbells.

## Outline

- `service/` is the Vercel project (its Root Directory): Python 3.13 on Fluid Compute, `framework: null`.
  - `api/index.py`, the HTTP routes (`slack_events/app.py`):
    - `POST /api/slack/events`: checks Slack's signature (HMAC SHA-256 of `v0:{ts}:{raw body}` with
      `SLACK_SIGNING_SECRET`, at most 5 minutes old), answers `url_verification`, drops our own bot's posts, publishes
      the event to the `slack-events` topic synchronously keyed on its `event_id`, and answers 200, or 503 when the
      publish fails, so Slack retries. Nothing runs after the answer.
    - `POST /api/slack/interactivity`: the same check, then the payload onto the same topic, keyed on its action.
    - `GET /api/slack/health` (bearer `HEALTH_TOKEN`): `oldest_unprocessed_age_s`, `dead_letter_count`,
      `last_success_at`, `queue` and `dead_letter`.
    - `GET /api/cron/watchdog` and `/api/cron/reconcile` (Vercel's cron, bearer `CRON_SECRET`).
  - `api/consume.py`, the queue's consumer (`experimentalTriggers` in `vercel.json`, topic `slack-events`), runs
    `slack_events/consumer.py`. Each `event_id` is leased in Redis (`processing:<owner>`, NX, 120 s; a later delivery
    takes over an expired lease) and marked `done` for 7 days only after its doorbells and its routing line are posted.
    A failure retries with backoff; after 5 deliveries the message goes to `slack-events-dead`, a failure line goes in
    its thread and @top is rung.
  - `slack_events/crons.py`: the watchdog (every 5 minutes) rings again a chat post with no 👀 and no lane reply after
    15 minutes, rings @top and says so in the thread after 60, and once a day (16:00Z) posts the open chat threads to
    @top's inbox. Reconciliation (every 10 minutes) re-enqueues, keyed on channel and ts, a person's post of the last
    hour that has no routing record and no routing line. Both keep no state but what they post, and stamp
    `last_success_at`.
  - `slack_events/router.py`: `route_unaddressed`, the seam for routing by model. It returns None (no-model mode)
    unless `ROUTER_MODEL_ENABLED=1`; its docstring is the plan for the AI Gateway call, which isn't built.
  - `build.py`, the build step: copies `tools/research/src/research/slack.py` and the registry
    (`.agents/skills/using-slack/registry.json`) into `_vendor/`, so the deployment runs `main`'s routing code byte for
    byte. `_vendor/` is never committed; from a checkout the service reads the tree's own files.
- `tests/`, this directory's suite (`pyproject.toml`, a workspace member with the same pins as `service/requirements.txt`).
  It runs against fakes: Slack through `Slack`'s transport, Redis, the queue and the clock. No test calls Slack,
  Vercel, Redis or a model.

The service's root has no `pyproject.toml`: Vercel's Python build would read it (and the workspace's around it), so the
suite's lives one level up.

Logs are JSON lines of ids, decisions and errors. They never carry a post's text.

## Deploy

Nothing is deployed yet. To deploy (Daniel, or Codex with his Vercel access):

1. A Vercel project on this repository with Root Directory `integrations/slack-events/service`, and "Include files
   outside the root directory in the Build Step" on (`build.py` reads the repository around it, and fails without it).
2. Vercel's Upstash Redis integration on the project (it sets `KV_REST_API_URL` and `KV_REST_API_TOKEN`; or set
   `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN`).
3. Vercel Queues on the team. The `slack-events` topic, its consumer and `slack-events-dead` need no setup beyond
   `vercel.json`.
4. Environment: `SLACK_SIGNING_SECRET` (the Slack app's Basic Information), `SLACK_BOT_TOKEN` (Verity's bot, as on the
   control pod), `RESEARCH_NOTES_TOKEN` (reads the published `routes.d` and `people.d`), `HEALTH_TOKEN` (any long random
   string; whoever polls health holds it) and `CRON_SECRET` (Vercel sends it to the crons). `RESEARCH_ROUTES_URL`
   overrides where the routes are read from.
5. In the Slack app: Event Subscriptions on, Request URL `https://<deployment>/api/slack/events`, bot events
   `message.channels` and `message.groups`. Slack delivers events only for channels the bot is in: invite it to
   `#ask-the-team` (it reads that channel's threads, too). Leave the Interactivity Request URL where it is (see below).
6. Then retire @top's watcher hand-off of chat and `#ask-the-team` posts (the using-slack skill).

## Needs

- **Interactivity is not built.** `#ask-daniel`'s buttons are decided by the website's `/api/agent-approvals` route,
  which only a card's asker can reach (Cursor OIDC), and `ring_decision` needs that route's record. The route here
  accepts and enqueues an action, and the consumer acknowledges it as `interactivity-unhandled` (a TODO in
  `consumer.process`). Point Slack's Interactivity Request URL here only once the decision is readable from this service.
- **Health takes a shared bearer, not Cursor OIDC.** Nothing in this repository verifies a Cursor OIDC token server side
  (the broker's check lives in the website); `HEALTH_TOKEN` stands in until it can be shared.
- **`#resource-accounting` routes once the registry lists it** (with its id) under that key; it isn't there yet.
- **Routing by model is off** (`router.py`).
- **Unverified until a deploy:** the queue trigger in `vercel.json` (Vercel's documented Python form is a
  `[[tool.vercel.subscribers]]` entry in a `pyproject.toml`, which this root can't have), that `_vendor/` and
  `slack_events/` reach the functions' bundles (`includeFiles`), and the Upstash client against a real database.
