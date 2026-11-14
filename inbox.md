[2026-11-14T08:32Z] 2d7b01a1bbf9
## Coordinator, Sat 14 Nov 08:32Z: model rule: no timer of your own; end the turn at a run's wait

MODEL RULE, from Daniel to me directly tonight (#9393). You were launched as Opus 5.5 High, which is what the rule wants for your work. infra found tonight (INF-779) that a turn started by a worker's OWN timer or event, and not by its launcher, runs on the launcher's model, Fable 5.2, whatever the launch named. So, from now:

1. Set no timer and no subscription of your own. If you hold one, unsubscribe it now (list_subscriptions, then unsubscribe).
2. When you would wait on a run for more than a few minutes: push what you have, write a one-line checkpoint note, and END YOUR TURN. The first line of your final message: `WAITING: <run id> at <head12>, ETA HH:MMZ; on pass: <the next step>; on fail: <what to read>`. I resume you when the run ends; a resume by me keeps your model.
3. Never leave work only on your VM across a turn's end: push every commit to your own branch (a merge you have not yet verified goes to a scratch branch of yours, named in the note).
4. This replaces any "wake myself" step in your brief. "Do not end a turn to wait for the lead" still holds for everything that is not a run's wait: work that can go on, goes on.

