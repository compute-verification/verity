[2026-11-13T09:33Z] 0e79e4b6624f
training, Fri 13 Nov, 1:33 AM PST. TRN-145, your 1:20 AM report.
**Ruled: #9676 stays as it is and lands as it is. Your 3 comes next, as its own PR cut from #9676's head, with its design sent to me before any code. 1 and 2 wait for 3's trace and become one small PR of whatever 3 leaves standing. One exception: if 54ec stops on memory, 1 and 2 go first, because then they are what reads s-t20.**

Why this order:
- 3 is the fix, and 1 and 2 are not. Time in ctx² × heads × layers is what stops a submitted circuit of real size from being read at all. A flat memory profile at the same hours does not change that.
- 3 removes the valuing that 1's memo and 2's `got` exist for, so doing them first may be work 3 deletes.
- 3 is the reader doing what the design rule asks of anything that reads a circuit: decide once per Definition and multiply through the batch.
- #9676 has its runs on its head. Nothing is added to it.

3's design message, before code. Keep it short:
1. The rule in one sentence: what the reader reads of a softmax's score operand to say "the mask makes these rows differ", and where it is decided.
2. It reads structure only: the selects, and what their condition compares. Never a size or a shape that happens to match.
3. Outside the form the reader falls back to today's word-by-word valuing: the same answer, slower. It never guesses. The fallback stays, as the test's reference.
4. The test: every `Work` equal with the rule on and off, on each shape both can read (tiny and s(2,64,512) at least). And one circuit whose mask is off the form by one word, which must read as word-by-word does.
5. Before and after: time and peak for s(2,256,512) and tiny(2,1024,64), and s-t20 if it then reads.

Runs:
- d4d5 and 72e0: two agreeing traces are enough. Stop both, by run id, unless you need one's end for a figure you will cite. If you keep one, tell me which and for what.
- 54ec: let it run. Send me its end, or its last heartbeat if it stops.

Branches: no merge of `main` until the chains land (INF-783). I read #9558 whole this morning, then #9676.

