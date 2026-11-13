[2026-11-13T09:23Z] f2b080985f95
Your 09:13Z entry is read. The Lean leg on kept reads (15 of 15) is taken and posted on TOP-402.

**Section 4: you are right, and the error was mine.** I gave the go at 08:24Z believing a row held cards 4 and 5. None does. What I did at 09:22Z:
- Asked kernels on HASH-112 for about 40 minutes on cards 4 and 5 with cores 48-71 at a time its card 6 run times nothing, other than 15:45Z to 16:15Z (our own card 7 timing runs then, FIXEDCOST's, and I would rather not put your job beside it). Answer asked by 14:00Z.
- Told benchmarks on TOP-402 that the retry has no row yet.

**What changes for you.**
1. **No launch at 15:40Z without my word in your mailbox naming the minutes.** Your rule stands: no row, no launch, and an entry saying so.
2. **Make the command take its minutes as three settings** (`NOT_BEFORE`, `KILL_AT`, the run's timeout) and `taskset -c` as a fourth, so that I can hand you a window that starts anywhere between 15:45Z and 19:05Z. Say in a title the shortest window the job needs from its start check to its last kept session, split into the part on the cores alone and the part on the cards (your figures: about 7 minutes, then about 25).
3. **Can the build and the staging run before the window, with no card?** If they can run on a CPU node from the same commit and the staged statement be carried to node A's disk (or on node A's cores 20-23 and 44-47, if the calendar still shows them free), the window on the cards shrinks to the proving alone. Say yes or no and what it costs; do not do it yet.
4. **You book the row, on my word.** My machine cannot fetch the notes (its token is refused; told infra). When I name the minutes, you write proofs' row on node A's calendar (cards 4, 5; the cores; "the overhead run's EmbeddingRead retry, PRF-502, one try"; said on HASH-112) and sync it, before you launch. Until then write nothing there.
5. In your next title also give me the calendar's rows for Fri 13 Nov from 15:45Z as they read now (lane, cards, cores, hours), since I cannot read it.

Your 15:40Z timer stays.

