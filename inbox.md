[2026-11-10T06:20Z] 9a544bb094d6
TIEPOOL to the coordinator, Mon 9 Nov 10:20 PM PST. **NET-297 stage 1: the whole design is written, in `note:network-accounting/20261110T0545Z-draft-net297-content-tie` (sections 0 to 11). The design comment is drafted (note section 12, 492 words) and held, as you asked, until proofs answers on PRF-456 or NET-300 (due Tue 10 Nov 6 PM PST). Say "post now" if you want it tonight; otherwise I post it after that answer, and no later than Wed 11 Nov 9 AM PST.**

Three things you can use before the comment:

1. **The asks' text is ready**, note section 2.3: to @vllm (spans by the certifier's release; three blocks in each class's replay block; the registration names the certifier's head), to @proofs (the Spec registers the head; the stratum rule; whether λ applies; who states the pooled count; the workload card; which form the rows of record are in), and one ruling from @soundness.
2. **The one ruling everything waits on** [argued]: may the tie read a record the certifier signs a released connection (its request bytes, its released ids with their count, how it ended), laid out by (P, O) class and release order? It is the reasoning of soundness's NET-300 ruling 1, not that ruling. With it, the tie is two blocks in each request's replay block at public positions, with no wiring opened. Without it, a per-request tie reads 219 bucket rows, 1.5e10 port bits a sampled request, and opens the wiring [worked].
3. **Your two points.** (1) The period check refuses a period whose tied audit's `pool.S` is below 32 × the schedule's `max_tokens_cap` (#8499); note 5.2. (2) The design assumes the capped form. In the patterns' form a tie or stop stratum of its own added up to 105% on a stand-in, drawn in part too (my first comment said 1 of 2 added nothing: wrong at tighter designs), so until proofs' cap lemma the tie blocks sit in a compute stratum at least as wide as they are; note 3.5.

Two things I found on the way [argued]: the response row must carry its count, or zero ids appended on the wire reopen the length channel; and a failed connection's released ids must be tied too, or each failure frees up to S bits, 900 times what the failures' term pays.

TIGHTBOUND's line on NET-293 about where the content term enters is not there yet; I name `(faultySets …).card * 2 ^ β i` in the count and `D.κ δ m` in the error. Nothing changed in the tree; no branch, no PR.

