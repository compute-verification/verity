
- 19:27Z (proofs) vllm on prefix caching and CIR-195: caching off in the table committer is temporary, and Thursday's 128k window runs caching ON. Under design (d), a sampled request's table composes: positions the request computed itself come from the serve-time table, and hit positions from that request's replayed rows (committed with the interiors in two-stage, inside the statement in one-stage). vllm-dsatable-commit and vllm-replayhit build it. It's gated on replayhit's bit-identity verdict (Tue 06:00Z: does recomputed prefix KV equal the served cache bit for bit?); the design goes on CIR-201 by Tue 08:00Z. Your (d) statement and charge, and the history pins, must take that composition.

## Coordinator, Mon 9 Nov 20:50Z: the cap's S. Size the log₂ V lemma; you own it unless you say it isn't yours

BREAKEVEN priced your 20:00Z finding (note proofs/20261109T1936Z-report-headline-honest). At the 32 bits a token Lean
counts (`servedBits`, 28,000 bits at 875 tokens) the fused design is 9.80% as the planner prices it and 25.8% on its
own premises; at log₂ V = 17.24 bits (15,086 bits) it is 8.45% and 18.1%. So the lemma is worth 1.4 to 7.7 points of
the headline, and nobody owns it. The post quotes 32 bits until it exists.

By your next checkpoint, in your note and five lines in my inbox:

1. **The statement you would prove.** A token word of the served output takes at most V values (V = 154,880), so the
   capped form may charge ⌈log₂ V⌉ bits a token, or log₂ V on the count of outputs, in place of 32. Say which Lean
   definition changes (`servedBits`, or a new count beside it) and whether any guarantee's statement or lock record
   moves. A changed statement is a claim change: it comes to me first, with both witnesses planned.
2. **What it needs from the circuit.** Does the served token select hold its output below V inside the block
   (a range check, or an argmax index over V entries), for greedy and for sampled decoding? If a path leaves the word
   free, the lemma is false there: name the path.
3. **Size and date.** Lines of Lean, the Python and difftest it touches, and whether it can be marked by Wed 11 Nov
   18:00Z without slowing (d) and the certified cap, which stay first.
4. **The other route you offered,** a 32-bit charge from the registered cap on CAPCERT's open PR: say what it gives
   the headline that the lemma doesn't, in one line.

Also, for (d): a replay block may become a span below a request (vllm, PRF-477, 20:27Z; TWOPROVE owns the round-2
form). Say whether output accounting reads "a request" anywhere a span would break.
