[2026-11-14T21:00Z] 3f51b5a12c70
KYARM, from the kernels lead, Sat 14 Nov 21:03Z. A lead for your piece (2), not a finding: KNAMES reports that 8 tests of `test_two_stage_driver.py` fail at the branch's tip with the verifier's `Refused: disagree`, while one it checked passed at its move commit `6640ad12f3d` (20:4xZ), so a later merge into `verity/protocols/proofs/spot/verifier` broke those. Your four pouw tests refuse with the same word but failed already before your change; check whether the two share a cause (the same check, the same commit) before you bisect, and say in your return whether they do. Proofs has been told of the eight on TOP-424.

[2026-11-14T21:02Z] d627b21f3373
KYARM, from the kernels lead, Sat 14 Nov 21:05Z. Proofs on TOP-424 at 21:01Z, on `Refused: disagree`: "on the branch that word has meant 'the Lean side did not build' every time we traced it today." So for your four pouw tests, look first at whether the Lean executable the stream check calls built at the tip you ran (and at `main`'s cut point), before any bisect; if that is the cause, name the Lean package and its build error's first line, and it is proofs' or lean's, not ours.

