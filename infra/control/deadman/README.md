# The control pod's dead-man's switch

The control pod watches everything else, so something outside it has to notice when the pod itself stops. This is that
something: one function on Vercel, called by a cron every 10 minutes. The watchers supervisor on the pod writes a
heartbeat to Neon every minute. When the heartbeat is 15 minutes old (the pod or its supervisor stopped), or the cloud
watch's last pass recorded in it is 30 minutes old, the function posts in #agent-alerts naming infra, and again every 6
hours while that lasts.

## Try it

`GET /api/deadman` on the deployed project, without the cron's secret, answers 200 or 503 with the heartbeat's age and
posts nothing. Its tests need nothing beyond `uv sync`:

~~~sh
uv run pytest tests/infra/ops/test_deadman.py
~~~

## Deploying it

Console deploys it in its Vercel team (a Pro plan: Hobby crons run once a day), as its own project with root directory
`infra/control/deadman`, and sets:

| Variable | Value |
| --- | --- |
| `STORE_PRIVATE_READER_URL` | Neon's `store_private_reader` connection string (read-only) |
| `SLACK_BOT_TOKEN` | Verity's Slack bot token (`chat:write` in #agent-alerts) |
| `CRON_SECRET` | a fresh random string; Vercel's cron sends it as `Authorization: Bearer ...` |
| `DEADMAN_HOST` | optional: the row's host, default `vy-control-verity` |

It is the standard library alone, so it has no `requirements.txt`.
