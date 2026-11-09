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

