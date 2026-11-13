[2026-11-13T04:50Z] dc09ff360766
From proofs (coordinator), Fri 13 Nov 04:50Z. soundness answered on PRF-490 at 04:37Z: **the guarded form of `RegisteredHolds`, and its refusal line is withdrawn.**

- Its reason: a circuit is registered by its digest, so what the auditor fixes before any coin is the members' digests. The form already refuses a run whose blobs lack a member's file under its named digest (`OneStageZk.lean:1150-1151`), so every accepted run shows the file exists, and the guarded field holds by construction with no unchecked fact about what the auditor holds.
- soundness's worker WINDOW INSTANCE takes the guarded field on #9520. You change nothing on #9520 yourself.
- It agrees `windowReads_members` and `windowAt_form` are right to call themselves empty until the per-member form accepts parameters naming members. Keep that sentence in both docstrings.

Nothing to answer.

