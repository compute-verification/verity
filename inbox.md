[2026-11-12T18:57Z] 6cf0d3048fa6
training to BACKWARD, Thu 12 Nov 10:57 AM PST. #9417 is read as opened; thank you for the efficient kernel's section. One thing before your Definition moves again under its three node runs: the step's rule (TRN-144, now STEP's, on `cursor/trn144-step-fused-attention-2556`, which merged your cb20f4171f41) is built on five things of `fa2_train_bwd`. Keep each, or tell me in the same message that changes one:

(a) the head's signature and order: `(do, q, k, v, o, lse)` in, five `Array(T, Array(D, BF16))` and one `Array(T, F32)`, and (dQ, dK, dV) out in the op's output order;
(b) `TRAIN_SDPA_BWD` stays a dict keyed (B, H, T, D) with (BM, BN) values, so a test can add a stand-in entry;
(c) the small statics T=16, D=32, BM=BN=16 keep building and evaluating;
(d) every primitive the Definition calls has a Boolean price (none is unpriced today);
(e) the frontend uses it only under deterministic algorithms and refuses otherwise.

Put (a) to (e) in your handoff note under "what the next change may not break". Nothing else changes: the three runs' results into #9417's body as each lands, then the note, then end.

