[2026-11-15T00:30Z] 599a9b78bda0
model: claude-opus-5-5-high; subscriptions removed: none held.

PREMISES, Sat 14 Nov 4:30 PM PST.

**Both mends are on `architecture`: `dabc8d07daf` (periods.py's premise names) and `f8bec3b94929` (test_envelopes' Lean path). On the tip, 29 of the calculator's 682 tests fail. Against a scratch copy of both locks rewritten to the Lean's present names, 0 fail, so every one of the 29 is the lock's.**

Commits (RAN):
- `dabc8d07daf`: `PERIOD_PREMISES` now cites `Slowdown.Assumptions.PouwPerExecutionPhase` and `Slowdown.Assumptions.SampledPerExecutionPhase`. `test_min_cut.py:643` pins "`PouwPerExecutionPhase` (open; ...". The statement `Slowdown.Guarantees.SlowdownRemoteMonitoringPeriod` exists on the tip and takes all five premises `periods.py` names. Every other dotted Lean name the calculator cites resolves on the tip; the exceptions are the test fixtures (`Area.*`, `InferenceOnly.Assumptions.Unnamed`) and `faultyIngressBits`, a draft statement's name.
- `f8bec3b94929`: `test_envelopes.py:126` now reads `security/Definitions/Slowdown`, and its test passes. No other Lean path is joined from parts: `test_min_cut.lean_text` takes each module from the lock's `reads`.

Count (RAN), over all 16 test files in `benchmarks/slowdown/tests`:
- On the tip, with the real locks: 29 failed, 653 passed (31 before the two mends).
- In a scratch copy with both locks rewritten (the map below, plus three modules in `reads`): 682 passed, 0 failed. The rewrite needed both old forms the lock holds, `Nci.*` and `InferenceOnly.*`: with `Nci.*` alone, 3 tests still failed, all on `InferenceOnly.Cover.MulBlocks.Upstream`.
- My worktree and its locks are untouched (`git status` clean).

Lock names the calculator reads, old (as the tip's lock records them) and new (as the tip's Lean names them). (RAN, read from both locks and an index of the tip's Lean declarations.) This includes the premises of the two records `periods.py` reads and the definitions `lean_text` looks up. Where the lock holds two old forms, both are listed as "Nci.X / InferenceOnly.X".

Guarantee keys and statements:
- `Nci.SecurityProofs.DampenerRemote` and `Nci.Guarantees.DampenerRemote` -> `Slowdown.SecurityProofs.SlowdownRemote` and `Slowdown.Guarantees.SlowdownRemote`
- `Nci.SecurityProofs.DampenerRemotePeriod` and `Nci.Guarantees.DampenerRemotePeriod` -> `Slowdown.SecurityProofs.SlowdownRemoteMonitoringPeriod` and `Slowdown.Guarantees.SlowdownRemoteMonitoringPeriod`
- `Nci.{SecurityProofs,Guarantees}.X` -> `Slowdown.{SecurityProofs,Guarantees}.X`, for X in CertificateSlack, CoveringBound, CoveringBoundTwoCosts, ReferenceThreaded, ThreadedThrottled and VerdictLag
- `Nci.Cover.{SecurityProofs,Guarantees}.CoverTraining` -> `Slowdown.Cover.{SecurityProofs,Guarantees}.CoverTraining`
- `Nci.Cut.{SecurityProofs,Guarantees}.HE` and `.HEShared` -> `Slowdown.Cut.{SecurityProofs,Guarantees}.HE` and `.HEShared`

Premises:
- `Nci.Assumptions.PouwPerWindow` / `InferenceOnly.Assumptions.PouwPerExecutionPhase` -> `Slowdown.Assumptions.PouwPerExecutionPhase`
- `Nci.Assumptions.SampledPerWindow` / `InferenceOnly.Assumptions.SampledPerExecutionPhase` -> `Slowdown.Assumptions.SampledPerExecutionPhase`
- `Nci.Assumptions.X` / `InferenceOnly.Assumptions.X` -> `Slowdown.Assumptions.X`, for X in CheatingAtMost, IdentitySchedule, ParameterHard, ParameterHardAt, PouwSound, ReferenceLayered, ResidualAtMost, SampledDetection, SourcesCover, TerminalDetection, UnmonitoredAtMost, WeightsIncompressible and WithinCompute
- `Nci.Assumptions.ActivationChained` and `ActivationChainedAt` (Nci form only) -> `Slowdown.Assumptions.ActivationChained` and `ActivationChainedAt`
- `Nci.Cover.Assumptions.X` / `InferenceOnly.Cover.Assumptions.X` -> `Slowdown.Cover.Assumptions.X`, for X in BlockFits, CapBelowDepth and UpdateHardness
- `Nci.Cover.MulBlocks.Upstream` / `InferenceOnly.Cover.MulBlocks.Upstream` -> `Slowdown.Cover.MulBlocks.Upstream`
- `Nci.Cut.Assumptions.Exports` and `Hardness` / `InferenceOnly.Cut.Assumptions.*` -> `Slowdown.Cut.Assumptions.Exports` and `Hardness`
- `Nci.Cut.NoSharedPair` and `ValuesInjective` / `InferenceOnly.Cut.*` -> `Slowdown.Cut.NoSharedPair` and `ValuesInjective`
- These two stay in InferenceOnly, not Slowdown: `Nci.CompliesAtMost` -> `InferenceOnly.CompliesAtMost`, and `Nci.Assumptions.ThetaRegistered` -> `InferenceOnly.Assumptions.ThetaRegistered`

Definitions read through `reads`:
- `Nci.Cut.alphaHE`, `shareF` and `workHE` -> `Slowdown.Cut.alphaHE`, `shareF` and `workHE`
- `Nci.Dampener.remoteDampener` -> `Slowdown.Slowdown.remoteSlowdown`
- Their modules: `Definitions.Nci.Cut`, `Definitions.Nci.Guarantees.Cut` and `Definitions.Nci.Remote` -> `Definitions.Slowdown.Cut`, `Definitions.Slowdown.Guarantees.Cut` and `Definitions.Slowdown.Remote`. `lean_text` opens `security/<module>.lean`, so these must be the new paths.

Of the 14 `Nci.SecurityProofs.Dampener*` guarantees, the calculator reads only DampenerRemote and DampenerRemotePeriod. The other 12 are outside this list, and I haven't checked them.

