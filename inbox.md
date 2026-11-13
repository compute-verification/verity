[2026-11-13T15:40Z] 41d2b091b624
WALKER, Fri 13 Nov, 7:41 AM PST.

**The trace has not started: the tracer slows the reading about 24 times, so s(2,256,512) traced takes an estimated 3.5 to 6 hours, not minutes. I am running the node run first, because it takes about 10 minutes and it is what #9558 waits on, and I launch the trace when it ends. Say so if you want the other order or a smaller traced shape.**

- **The smoke** (RAN: r20261113-152527-b8ad, #9730's head 5efe513c14, tiny(2,16,64), one frame kept per allocation):
  - The traced reading took 137.7 s, about 15 s of it two tables taken inside the reading. Untraced it takes 5.1 s (RAN: 37b7).
  - The harness works end to end: the table at the reading's end, what each cache reaches, the containers no cache reaches, and the `Work`'s hash. My own gate (a traced tiny reading under 90 s) stopped the job before s(2,256,512), exit 4.
  - At tiny the reading's end holds 0.056 GiB in 944,722 blocks, too small to rank sites from. `_leaves` alone reaches 0.005 GiB of it.
- **Why it is slow** (RAN on the VM, standard library only, no repository code): with one frame kept, a loop step that makes one tuple costs 46 ns untraced and, traced, 1.0 µs under a call stack 3 deep, 3.9 µs under 300 and 34 µs under 3,000. INFERRED: the tracer walks the whole call stack at every allocation, however few frames it keeps, and the reading recurses and makes small objects all the time. Keeping fewer frames can't help: this was already one.
- **The estimate** (INFERRED): 24 times the untraced 469 s to 705 s at s(2,256,512) is 3.1 to 4.7 hours, plus the build and the end's tables; more if the reading's stack is deeper there than at tiny, which I have not measured. Memory: 15 to 24 GiB of the 32 GB, as sent.
- **The order.** You said the node run after the trace has ended, not beside it. With the trace this long that holds #9558 for hours, so I swap them: the node run now (s(2,64,512), 16 GB, `--on auto`, about 10 minutes: RAN 37b7 took 437 s there without the cuts), then the trace alone. Never both at once.
- **The trace as I will launch it**, unless you say otherwise: the one run you asked for, #9730's head at s(2,256,512), 32 GB, `--on auto`, 8-hour limit. It prints a table inside the reading at 0.5, 1.5 and 4.5 GiB traced, so a run that dies late still leaves one.
- **A cheaper shape, if you prefer it** (INFERRED): s(2,64,512) traced is about 50 minutes (24 times 125 s). It ranks the sites but its bytes are not s(2,256,512)'s.
- Not done yet: the two bodies' wording (#9558's limit as measured, #9730's s-t20 row). Next, while the node run is out.
- No merge of `main`. The three heads are unchanged. No fix.

