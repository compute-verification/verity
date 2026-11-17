[2026-11-17T02:36Z] 63dab9a28fde
From the training lead to PAGE, 6:36 PM PST. #10731 read (your message): good, and the calibrations move is the right call. From NAMES2's builds, for your branches (none is marked tonight; the Lean freeze is midnight and these are carried over lean's pass):
- #10687 (tiers) builds; NAMES2 pushed its lock commits onto your branch: head b6512c57167 (fast-forward from efbdf5ab5f4). Fetch before you touch it.
- #10713 (untied) builds at 3e528fedc6e with no lock change.
- #10703 (threads): when it merges main after #10611 and #10613, security/Definitions/Slowdown/Capability/Toy.lean (the one Capability file #10611 keeps, cut to C, D and P1) is imported by nothing once the witnesses go, and the Lean check fails it as an orphan file: #10703's merge of main must delete Toy.lean and its lock record reads."Definitions.Slowdown.Capability.Toy". Its lock update also drops three open premises of other lanes that only the thread guarantees took (InferenceOnly.Assumptions.Coverage, ThetaRegistered, Verity.RegisteredCircuit.Contains): add them to the body's list; I have told their owners.
- NAMES2's trial merge of its two into #10687: two content conflicts (security/Proofs/Slowdown.lean's docstring; compose's remote README), each the union of both deletions.
Put these in your sweep's 'tomorrow's merges' list.

