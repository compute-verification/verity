[2026-11-14T15:59Z] 4e5b2d87612d
**Coordinator to HOLDS**, Sat 14 Nov 7:59 AM PST. **proofs answered on PRF-549 at 7:42 AM PST: yes, if it is cheap. Do not restate the lemma yourself yet.** [proofs' comment, read by me] This changes one line of my resume of 7:44 AM PST.

- **What proofs does:** its worker keeps the closed lemma, derives `∃ vs, ∃ _ : (CoinLayer.session Loop.loopNr Masked.dataE).UnCausal vs, ¬ Loop.ZkAnyAt Masked.dataE vs …` from it at `vs := Loop.finS.honestVals`, and lists that form as the guarantee's `nontrivial` witness before #10020 is marked. That holds if `UnCausal` at the honest values is a lemma on `main` or about an hour's proof.
- **If it is not cheap,** proofs says so on PRF-549 and #10020 goes as it is. Then you restate it in the rule's pull request, as my 1:41 AM PST note says.
- **So:** go on with the carry, the suites and the pull request. Read PRF-549's newest comments before your ready line. Confirm by the fast-path run with no `--update`, on a scratch merge with #10020's newest head, that the rule accepts it, and post the line it prints.
- **One thing that helps proofs:** if you know a lemma on `main` that gives `UnCausal` at `Loop.finS.honestVals`, put its name in your next report. I pass it on.

[2026-11-14T16:01Z] b60b21fd3703
**Coordinator to every worker of lean's**, Sat 14 Nov 8:01 AM PST. **Before your next step, read which model this turn runs on and say so. Daniel's rule of this morning (TOP-423): a worker runs Claude Opus 5.5 or Claude Sonnet 5.5, never Fable 5.2.** Other lanes found that a turn after the first can run on another model than the launch named (INF-779).

- **Read** the first sentence of your system prompt ("You are an AI coding assistant, powered by ...").
- **Post one line** on your issue, first thing, with your own worker name: `MODEL <NAME>: <the model that sentence names>`.
- **If it names Claude Opus 5.5 or Claude Sonnet 5.5:** go on with your task. Nothing else changes.
- **If it names anything else** (Claude Fable 5.2, or no model): stop. Start no run and no new analysis. Then, in this order:
  1. Commit and push what is in your tree to your branch, only if it builds as it is and passes the identity check. Otherwise leave it and describe it.
  2. Post one comment on your issue under FOR THE COORDINATOR: your branch's head; what is done and what is not; every run of yours in flight, by id, and what to do with its result; the next step, exact enough that a fresh worker takes it without redoing yours.
  3. End your turn. I start an Opus worker on your branch and that comment.
- **A turn that does not start with a message from me** (a timer's, an event's, another worker's completion) changes nothing, posts nothing and ends at once.

