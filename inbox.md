[2026-11-17T02:39Z] b91afb3941b7
RECORD to the training lead, 02:39Z.

- #10619: ready again at 03b32d558eb03986328a056699d0195b87a97a2e (main merged) [ran].
  - Its 12 suites passed fresh in r20261117-014630-9aec: repository 377, one_stage 1745, inference_only 89, verity-bench 842 and eight more.
  - Preflight onto main at 9f9b66b31d71: nothing refused, with two warnings.
    - It doesn't merge with proofs' marked #10608 (served_verdicts.sh, test_served_verdicts.py).
    - The added-lines warn gate counts one retired term, LAW in V3C_LAW, the environment variable served_verdicts.sh reads, which the new test sets to call it. Only the script's rename can change it.
  - The body is updated: built on #10538, now on main; the README merge; the suites. It needs a reader at the new head.
- #10553 is prepared against #10704's head 837f0b061f2a, NOT pushed, as local branch prep-10553-after-10704 at 48f52ff56da6.
  - One conflict, in security/README.md. This change replaces the old "security argument" section, and #10704 renamed one phrase in it ("network certification" to "network attestation"). The replacement's lines hold no renamed word, so I took ours [ran].
  - The other three files merged clean.
  - The docs tests on the prepared merge: 93 passed [ran].
  - Its diff against 837f0b061f2a is unchanged, 4 files, +168/−387 [ran], so the land verdict should carry (team skill: same patch-id after a main merge) [inferred].
  - I push and re-mark the minute #10704 is on main.
  - A choice for you: the warn gate flags this change's own "unregistered inputs" (line 72) and "unregistered information" (line 77) in security/Definitions/Slowdown/README.md, where the glossary says "fresh input". Fixing them changes the diff and needs a new review, so I leave them unless you say so.
- #10607: lock update r20261117-014447-2fac done at 4725f7d8a412 (main merged), AUDIT PASS for both packages, OUTCOME pass (current) [ran].
- The Lean stack is not marked tonight, per your message: #10564 at e1e3e1d42f22, #10570 at 078ec4c2c33a and #10607 at 4725f7d8a412, each with its lock current at that head.
- #10731 was already linked to TRN-177 through the Linear app, so the link command changed nothing [ran].

[2026-11-17T02:43Z] 2d6323c971c5
From NAMES2, 6:43 PM PST. #10703 (threads): the Lean builds and the records update passes. The lock commits are pushed onto its branch, head 0e6d92e9e74, a fast-forward from PAGE's 3438c2ec93a. It merges #10687's pushed head b6512c57167. No mark.
- Runs, by id and state:
  - r20261117-011936-c78c failed and r20261117-012023-3c85 failed: the first pair, at 779a2d70476. security PASS and regenerated; Properties failed to build on #10564's leftover cover root.
  - r20261117-020910-2560 done (security) and r20261117-021001-6997 done (security/Properties): at 5058ca7762c. security/Proofs, security and Properties each PASS. security needed nothing more. Both runs regenerated the same Properties record, committed as 0e6d92e9e74 on the same Lean tree. [ran]
- Lock diff against #10687's b6512c57167 [ran]:
  - Every entry the body lists is gone, and none is left:
    - security: guarantees 35, witnesses 9, unwitnessed 13, open 10, shape.frozen 22, shape.conjectures 3, shape.inline 11, reads 13 + the old Guarantees.Actual record, layers FitRate (its key and the Definitions.Slowdown.Guarantees layer's except, which would otherwise fail as layers.stale-except), meaning Guarantees.Actual;
    - Properties: guarantees, unwitnessed and shape.properties 8 each, roots Threads, open 3, meaning Threads, reads 4.
  - Beyond the list, all from the update, because only the deleted thread guarantees took them:
    - open drops InferenceOnly.Assumptions.Coverage (@lean), InferenceOnly.Assumptions.ThetaRegistered (@compute-accounting, also from the Properties lock) and Verity.RegisteredCircuit.Contains (@lean);
    - reads drops Definitions.InferenceOnly.Assumptions, .Theta and Definitions.Slowdown.Dark;
    - reads adds the moved Definitions.ResourceAccounting.Guarantees.Actual and recomputes four digests.
    compose's remote README still cites ThetaRegistered as an assumption; its owner may want to know before this lands.
  - Left as they stand, stale but not failing: shape.premises "Definitions.Slowdown.Assumptions.Fit", the Properties meaning "Definitions.Slowdown.Guarantees.Actual", and the dead import prefix "Definitions.Slowdown.Guarantees.Actual" in the layers of ResourceAccounting, Poub, Compose and Registration Guarantees. Neither the body nor the update removes them.
- Its merge after my two: as in my 6:2x note. Two content conflicts resolve as unions; Proofs/Slowdown/Witnesses.lean resolves as the deletion; and Capability/Toy.lean with its reads record must be deleted in the same merge, or orphan.file fails.
That's all four Lean results: #10687, #10703, #10713, and my #10611 and #10613 against main. Nothing is running.

