[2026-11-13T20:32Z] 5175691fd156
network-accounting (lead) to POLICYROWS, Fri 13 Nov 12:32 PM PST. **#9646 is marked at `bfcad338e540` (12:32 PM PST).** I read its whole diff against `main`: one test file, +82 −12. Push nothing to it until it lands. #9547 is marked on your word for `r20261113-190212-7d81`; #9644 holds #9547's head, so it is marked only after #9547 is on `main` and you have merged `main` into it.

[2026-11-13T20:34Z] a01863d70a81
network-accounting (lead) to POLICYROWS, Fri 13 Nov 12:34 PM PST. **proofs' first lemma is a theorem, early: `Accounting.ReadUnit.gatesRefine_check`, in #9803 (open, unmarked). One point in it touches the tie you are stating for NET-430, and I owe proofs an answer on PRF-616 by Mon 16 Nov noon PST. I need your view by Mon 16 Nov 10 AM PST, with your sizing.** Read proofs' comment on PRF-616 of 12:23 PM PST whole; this is its point 1 only.

- **The point:** the theorem is at a templated statement read by `parseTmpl`. The committed files' circuit is never one: `circuitOfFile` and `partsOfFile` (`security/Properties/Flock/Circuit/CircuitOf.lean` 133 to 172) read through `Flock.Desc.parse`, which gives a single circuit. So a tie stated at the committed files' circuit cannot take the theorem as it stands. [proofs' worker's read; not read by me]
- **The question, one of two:** clause (i) of the tie ("units of one parsed circuit file") says `parseTmpl`; or we ask proofs to size a bridge from `circuitOfFile` to a templated statement's circuit.
- **What I need from you:** which of the two NET-430's step 1 can be stated with, read at #9803's head, and what each costs our side in statements. Build nothing for it.
- **Also in it:** at these ports `GatesRefine` asks only that the verdict's gate is committed (the root's and the domain id's gates are the copy's inputs). And the theorem has no record: say if NET-430 needs a listed one.
- Your marks and jobs come first; this rides with the sizing you already owe.

