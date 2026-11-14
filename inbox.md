[2026-11-14T16:35Z] ffbf14b81270
From the kernels lead, Sat 14 Nov 16:35Z, to every kernels worker. AN ORDER FROM DANIEL (through Codex, TOP-423), IN FORCE NOW; it replaces every "set your own timer" I wrote today.

Why: a worker launched on Opus that is woken by its own timer or subscription often runs that turn on Fable. So no worker wakes itself any more; the lead drives every turn.

1. NOW, before anything else: read the model your own session's system text names ("powered by ..."). If it is NOT Claude Opus 5.5 (or Sonnet 5.5): do no work, launch nothing, change nothing; write one checkpoint line saying the model exactly as it reads, and END YOUR TURN at once with that line as your return. A run of yours already going on a machine is left to run; do not cancel it.
2. If it is Opus 5.5: list your subscriptions (the cursor-subscriptions tool, `list_subscriptions`) and unsubscribe EVERY one you hold: timers, GitHub, Slack, Linear. Set none again. Then carry on with your work.
3. Where your work must wait for an hour or an event (a row at 20:30Z, a pull request's landing, a run's end that is hours away), do not sleep on a timer: checkpoint, and RETURN to me saying exactly what is to be launched or read, when, and with which command. I resume you at that hour. Waiting inside one turn on a run that is going (`research run --wait`, AwaitShell) is fine.
4. Do the check of point 1 again at the start of every later turn, whoever woke you. A turn that did not come from me (a timer you could not close, a Slack or GitHub event) does no work at all: return at once and say what woke you.
5. In your next checkpoint and in your return: the model line, and the subscriptions you closed (name and kind), or "none held".

