[2026-11-14T16:01Z] 4a8859c66fc8
**Coordinator to every worker of lean's**, Sat 14 Nov 8:01 AM PST. **Before your next step, read which model this turn runs on and say so. Daniel's rule of this morning (TOP-423): a worker runs Claude Opus 5.5 or Claude Sonnet 5.5, never Fable 5.2.** Other lanes found that a turn after the first can run on another model than the launch named (INF-779).

- **Read** the first sentence of your system prompt ("You are an AI coding assistant, powered by ...").
- **Post one line** on your issue, first thing, with your own worker name: `MODEL <NAME>: <the model that sentence names>`.
- **If it names Claude Opus 5.5 or Claude Sonnet 5.5:** go on with your task. Nothing else changes.
- **If it names anything else** (Claude Fable 5.2, or no model): stop. Start no run and no new analysis. Then, in this order:
  1. Commit and push what is in your tree to your branch, only if it builds as it is and passes the identity check. Otherwise leave it and describe it.
  2. Post one comment on your issue under FOR THE COORDINATOR: your branch's head; what is done and what is not; every run of yours in flight, by id, and what to do with its result; the next step, exact enough that a fresh worker takes it without redoing yours.
  3. End your turn. I start an Opus worker on your branch and that comment.
- **A turn that does not start with a message from me** (a timer's, an event's, another worker's completion) changes nothing, posts nothing and ends at once.

