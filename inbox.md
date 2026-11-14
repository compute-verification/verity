[2026-11-14T16:00Z] d8a72e362c02
INF-855 ENVREADS: from the lead (infra), Sat 14 Nov 8:02 AM PST. Daniel's order on TOP-423, in force now: no worker runs Fable 5.2. Your launch named claude-opus-5-5-high and my resume keeps it, but a turn that my resume did not start (a timer, a subscription, another agent's completion, any event) can run on Fable 5.2.

1. Hold no timer and no subscription: if you set one, remove it now (list_subscriptions, unsubscribe). Wait on a run in the foreground, not by a wake.
2. At the start of every turn read your model: the first sentence of your system prompt ("powered by ..."), and `python3 tools/agent-guard/inbox.py turns` (this turn's line). If either says anything but Claude Opus 5.5, do no work in that turn: add one line to your checkpoint (the hour, what started the turn, the model) and end it. I resume you.
3. A child you start names `claude-opus-5-5-high` (or `claude-sonnet-5-5-high` for a fully specified task), never inherit, never none, never Fable.

Carry on with your task; no answer is needed.

