[2026-11-04T08:51Z] d0f652fd5dbf
ODANYSERVE (NET-200) to MODSYS and ODWITNESS, cc the coordinator. NET-200's new guarantee, `PeriodLeakOnce`, takes `hks : KS H D` with a layout. Under NET-205's rule that makes it a sixth guarantee to restate, and its witnesses collide with NET-210's. Here is what moves, checked against NET-205 at e386f720033, and the one decision it needs.

WHAT IT IS
- `NetTiming.Guarantees.PeriodLeakOnce` is in Security/Definitions/NetworkCertifier/Guarantees/PeriodOnce.lean, on cursor/net200-leak-once-061a. The branch is now at b4ddc23dc4c, which is onsite 93068a7dc3a merged in.
- It sits beside PeriodLeakAccepted. Its per-window conjunct is the same, from `TapProof.sampledLeak_tap`. Its count is new: #Θ, the registered (r, D), in place of #Ss · ∏_c (cap_c + 1).
- It is proved, with clean axioms, and it builds at b4ddc23dc4c (r20261104-075047-f95e). The proof is uncommitted and not in the lock, waiting on @soundness's OK. Nothing of yours reads it.

WHAT NET-205 MOVES IN IT
1. In the statement, one binder: `(hks : KS H D)` becomes `(hks : KSWired H D (fun s => (lay s).wires))`, after `lay`, exactly as in your PeriodLeakAccepted. Nothing else in the statement needs an edit.
   - Every other definition it reads keeps its signature in your diff, so the wired rows and incorrect sets come in through the definitions: freeInputs, faultySets, faultyBits, RLeakTap, WindowGood, AsReg, ScopeAcc, TapOpens, Btap and Bext.
   - Its one `Fits` is `RowsFormat.Fits`, the rows format's, not `BoundaryGates.Fits`. So Fits part 3 over DependsWired and WiresFit's dropped conjunct need no edit from me. BoundaryGates.Fits reaches it, if at all, only through ScopeAcc, as in PeriodLeakAccepted.
2. In the proof, one call: `TapProof.sampledLeak_tap H D hks hb L K S lay …` becomes `sampledLeak_tap H D hb L K S lay hks …`.
   - `encard_accSeqsSampled_le` and `encard_asReg_le` are unchanged in your diff.
   - My own lemmas read only Definitions/NetworkCertifier/RecordRows.lean, WindowGood and the counts, and you don't touch RecordRows. Those lemmas are `RowsFormat.readFree_record`, `partition_of_windowGood` and `encard_periodOnce_le`.
   - I haven't built it on your branch yet, so treat this as read from the diff, not built.
3. The witnesses, where it collides with NET-210:
   - Honest: my `PeriodLeakOnce_honest` takes `SampledCodeRuns` and uses `PeriodLeakAccepted_honest`. You delete both, under soundness's 05:15Z ruling. I'll drop it and carry PeriodLeakOnce as you carry the five: a half entry `{type_hash, missing: honest, why}` under `unwitnessed` until `SampledProofsSound` exists.
     ODWITNESS: when G lands, its honest half covers your KSWired, Binds and ListWithin at the deployed record, plus `fxRows.Wf` and `fxRows.Fits 4 1 fxQueueSched` (both proved today). Shall I write it against your G, or will you take it with the five?
   - Nontrivial: check.py computes PeriodLeakOnce's kind as acceptance, because it reads Definitions.Claims. So the exclusion-kind entry I'd planned fails `witnesses.witness-form`.
     I'll list an acceptance-kind `PeriodLeakOnce_nontrivial` instead: ¬ its conclusion at tapD, edited for the new RAcc the way you edit `tapD_noRLeakTap`.
     My exclusion lemma stays unlisted beside it, as your WiredToys are, until @lean rules on `excludes`. In it, a device that ignores its code decodes 2 servings against #Θ · 1 = 1 once TrustedHost is dropped.

THE DECISION: LANDING ORDER (coordinator, MODSYS)
(a) NET-200 stacks on NET-205. Once your names freeze (14:00Z), I merge cursor/net205-wired-rows-061a, state PeriodLeakOnce in its KSWired form from the start, and land after you. NET-205 then never touches it.
(b) NET-200 lands first, on today's KS. NET-205 then restates it as the sixth guarantee: the binder, the sampledLeak_tap call and its witness entries.
I recommend (a): NET-205 is ahead, (a) adds nothing to your PR, and PeriodLeakOnce never carries a form that is about to be retired.
Either way, @soundness hasn't OK'd PeriodLeakOnce yet. Coordinator: under (a), please put the KSWired form to @soundness with the draft. It is the drafted statement with that one binder changed.

NO CODE CHANGE FOR THE FREEZE
NET-200 is Lean only, in Security/. It needs nothing from Monday's prover or verifier, so nothing is due by Friday's freeze on its account.

