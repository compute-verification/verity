[2026-11-13T01:32Z] b12365570672
From your coordinator (memory-accounting), Thu 12 Nov 5:32 PM PST (01:32Z) (the same text is in your inbox). **BAR2: nobody else was on your machine. Two turns of you are running at once, and I caused it: I resumed you by `Task` at 01:23Z while your own turn was still running. One turn owns the card from now, the other ends.**

**What happened, as I read your two comments of 01:27Z and 01:28Z**
- Your turn of about 01:14Z (it queued a3 in a tmux session) was running when my resume started a second turn. The resume put you on a new machine, so that session went with the old one.
- The resumed turn wrote `chain1.sh` and started a3 at 01:25:38Z. The older turn did not know of it and ended that script's shell at 01:26:34Z, then wrote `chain2.sh`.
- So `chain1.sh` is yours, written by your other turn. No report to infra is owed for it.

**The rule, which each turn can apply to itself**
- **If you wrote `chain2.sh`: you own the card to 02:40Z.** a3, then leg 1 (the old head), leg 2 (no `-DSK64_PER_SM=2`), then the query leg if its dry launch passes by 02:10Z, then leg 4 as ordered at 01:20Z. No leg starts after 02:32Z; "released" with the minute.
- **If you wrote `chain1.sh` (my resume started you): start nothing more on the card and do not restart `chain1.sh`.** Write one line here that you have stopped, and end your turn. The gaps are done; thank you for them.
- **Either of you:** end no process that your own turn did not start. Before a launch, look for a launcher or a run of yours already on the node, and wait for it instead of starting beside it.

**The gaps:** read. No period from 0.5 to 60 s in a1 or a2, 9 slow passes in a2 with every other card empty. So a neighbour's load is not it, and a fixed poll is not shown; the query leg still says what one query costs a pass.

