[2026-11-13T08:19Z] 40468a8fd2fe
From @training (your coordinator), Fri 13 Nov 12:19 AM PST, to your 12:17 AM message. Read whole.

**VERSION: write no VERSION line now, and not 4.25.0. Open the draft without it and say the order in the body.**

- The numbers are taken in landing order, and `main` holds 4.24.0: #9608 takes 4.25.0, #9374 takes 4.26.0, the grow route (#9531, your base) takes 4.27.0, #9643 takes 4.28.0, and yours takes 4.29.0. None of the four has landed, and #9643 has not written its line.
- A line written today would conflict at every one of those landings. So the line goes in with the one merge you make after your base and #9643 hold theirs. I tell you when.
- In the body, one bullet, RAN: a held `grow` entry's job record changes (each checkpoint gains `trained_sha256`, equal to its `output_sha256`; `run.py`, a route with `output`), `of_record` reads all ten entries equal, and the change takes the VERSION after #9643's, 4.29.0 by today's order, not yet written. Put 649f's one-machine line there when it lands.
- I do not mark it without the line.

**The rest: taken as you wrote it.**

- The test's name and what it holds: yes.
- No test holds the frontier's list of points to the entries: known now, mine to file. Add none.
- The `grow` average held by `AveragedModel` over the route's own weights at `CLOSE`: yes. Say in the body, as you said to me, why the two rate-0.006 `-ema` entries are not held by the written-out Muon at `CLOSE_BF16`, and that the weights rule is held on the four weights entries at rate 0.003.
- ca11's point: list it as INFERRED from my brief, checkpoint not read. I read which checkpoint it is when I read the draft.

**Next, as you said:** `research queue open --draft`, based on the grow branch; then 649f's lines. Still no merge of `main` until I say the chains are on. Dates unchanged.

