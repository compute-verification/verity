# The control pod's dead-man's switch

A Vercel project of one function, `api/deadman.py`, and a cron (`vercel.json`) that calls it every 10 minutes. It reads the
heartbeat that the watchers supervisor on the control pod (`infra/control/watchers.py`) upserts into Neon's
`store_private.control_heartbeat` every minute. When the heartbeat is 15 minutes old (the pod or its supervisor stopped), or the
cloud watch's last pass recorded in it is 30 minutes old, it posts in #agent-alerts naming @infra, and again every 6 hours while
that lasts. `GET /api/deadman` without the cron's secret answers 200 or 503 with the heartbeat's age and posts nothing.

Console deploys it in its Vercel team (a Pro plan: Hobby crons run once a day), as its own project with root directory
`infra/control/deadman`, and sets:

| Variable | Value |
| --- | --- |
| `STORE_PRIVATE_READER_URL` | Neon's `store_private_reader` connection string (read-only) |
| `SLACK_BOT_TOKEN` | Verity's Slack bot token (`chat:write` in #agent-alerts) |
| `CRON_SECRET` | a fresh random string; Vercel's cron sends it as `Authorization: Bearer ...` |
| `DEADMAN_HOST` | optional: the row's host, default `vy-control-verity` |

Standard library only, so no `requirements.txt`. Tests: `infra/ops/tests/test_deadman.py`.
