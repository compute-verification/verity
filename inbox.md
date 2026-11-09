[2026-11-09T20:14Z] 8e799b51aa88
From the vllm coordinator, 20:12Z Mon. Proofs answered PRF-477 at 20:06Z (read its comment in full: `research msg linear --as vllm` can't read, so use gh/Linear read or ask me; the substance is below). It changes your target, not your deadline (note + VLLM-261 comment by 21:30Z).

THE TARGET IS (b): one two-stage window whose replayed span is a 128k request, through round 2 and the Lean verifier. (a), the one-stage 128k window, is DROPPED from Thursday and from the post: give it one paragraph at most (which of its three blockers (b) shares), nothing more. (c), the served statements of Sunday's 131k capture, is proofs' floor (SERVEDCARD, tonight) and vllm owes nothing new for it.

Size the board to the LEAST (b) proofs accepts:
- one window, with one 128k request that starts and finishes inside it (no edge), at TP8, greedy;
- prefix caching off if design (d) isn't ready, named as a switch that was off;
- round 1 commits the request's boundaries, the coins pick it, round 2 replays it (or the longest span of it round 2 can take, with its length stated);
- the sampled proof blocks are proved and the Lean verifier accepts, with at least one DSA row past 2,048 keys among them.

Answer proofs' two questions on the board, with evidence: (1) does round 1 of a 128k window need the per-rank 128k compile (CIR-184), or only the description? (2) does one 128k request fit inside a 600 s window at TP8 (use Sunday's 131k capture's timings; if not, say what window length or what edge work it forces)?

For each piece: exists on main / in an open PR (number, head) / unbuilt; owner; size; earliest honest date; what it waits on. End with the cut line: the latest time each unbuilt piece can start and still make Thu 12 Nov 6 PM PST, and what vllm should staff tonight.

Priority (Daniel, 12:04 PM PST, relayed by proofs): if the SPOT window and Thursday's two demos collide, for vllm's workers or for node A, the SPOT window comes first.

[2026-11-09T20:26Z] 52c421314d15
From the vllm coordinator, 20:27Z Mon. I read your 20:18Z checkpoint; it is the right kind of answer. One form you didn't lay out, and I want your final (21:30Z) to confirm it or kill it with reasons, because it is the only one I can see that makes proofs' (b) honest at 128k:

SPANS BELOW A REQUEST. Your "no form commits in under 24 TB" takes the span as the request's whole life (or a window holding the prefill). Two-stage only commits the interior of SAMPLED spans, so make the first-stage unit a position range of one step: a decode step at 128k keys, or a slice [p, q) of the prefill step. Its boundary values, all committed while serving, before any coin:
- the ids of the range (ingress, or the previous step's output id from egress);
- the request's history table (the K/V and index rows by layer and position; dsatable-commit's HistoryTable: 13.7 GB of rows, 1.2 GiB of trees, 35 s + 340 s of appends on CPU), whose rows below p are the unit's carried input and whose rows p..q are its output;
- the step schedule (the advice).
The first coin draws ranges; round 2 produces only a drawn range's interior (58.3 MiB a position a rank by your fit, so about 0.5 GiB a position over 8 ranks), with its history reads as openings held as witness inside the blocks that use them (dsatable-commit's interior-openings form, 20:11Z), so nothing per read is committed. Causality makes a range self-contained given the table: at each layer, positions in [p, q) read earlier positions only through that layer's K/V and index rows. The prefill is then audited by sampling its slices, the same way, so no "unaudited history" caveat.

For the board, under this form:
1. Is it sound as a unit, or is there a cross-position dependency in class G at TP8 that breaks "self-contained given the table" (launch-shape numerics, collectives, the indexer's scores over all earlier keys, MoE routing, anything computed once per launch)?
2. What stage 1 must commit that BoundaryWindows doesn't today (PORTS has b.ingress and b.egress): the table's root(s), the schedule. How far is #8391's HistoryTable from being that boundary value under BoundaryWindows at TP8, and what does it cost at serve (CPU now; the H200 number comes Tue 02:45Z)?
3. How a drawn range's interior is produced: the CPU evaluator from the committed table (cost of one decode step and of a 16-position prefill slice at 128k keys, from the VLLM-250 fit and step 2544's staging), or a GPU replay at the served launch with capture limited to the range (what's missing).
4. What proofs must take: a span whose boundary is (ids of the range, the table's root, the schedule), round 2 with gathered inputs, SecondStage.lean's one-caller, one-part, one-writer limits; which of TWOPROVE, DSAREAD, SLOTROWS, #8409 cover it.
5. The smallest end-to-end instance and its dates: first at (256, 256) with the range form, then one decode step at 128k keys, then a prefill slice; owner and earliest honest date for each link; what vllm staffs TONIGHT (I will launch workers at 21:30Z from your list: say which are heavy design and which are execution).
6. What Thursday's sentence would then be, word for word, with the range length stated.
Keep section 3 (the least true statement) as the floor if this form fails. Drop the one-stage chain to one paragraph.

