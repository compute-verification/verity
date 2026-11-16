[2026-11-16T20:14Z] e72425d9af92
TRN-177 map, supplement for top's 2:00 PM post (PAGE). Preliminary where marked. The full map went to your inbox at 12:10 PM PST; this adds what your 12:13 message asks for and two corrections.

## Corrections to the 12:10 map

- **`Partition.IsCK` has blocks of at most c gates, as the page does** [read]. The difference is the Glossary's "exactly c gates", a term for top, not a Lean gap.
- **The lock has 10 `ResourceAccounting.SecurityProofs.Actual.*` guarantees, not 5** [ran]. Two (`ServerBlocks`, `ServerBlocksAt`) move unchanged. Eight read Threads and go with that family.

## Counts per class (Lean in training's area; listed guarantees across `security/lean-audit.json` and `security/Properties/lean-audit.json`) [ran]

| class | files | lines | listed guarantees |
|---|---|---|---|
| keep: the page's theorem | 6 | 2,126 | 12 reworded (Cut) |
| restate (CutCharged, aggregator, Covering's `FeedsWithin`/`InferenceBlock`) | 3 | 335 | 0 (#9519 adds `HECharged`) |
| unsure (Untied; the Remote joins) | 1 | 863 | 8 |
| move (Actual model, Dark) | 2 | 142 | 2 |
| delete | 97 | 12,396 | 109 |

## Delete list, largest first [ran; Python sizes are line counts of tracked files]

| family | Lean files / lines | listed guarantees | also |
|---|---|---|---|
| Threads, Layered, Fit, Piece, Actual/Slowdown/Threaded | 23 / 2,848 | 41 + 8 properties (lean 37+4, training 4) | `benchmarks/nci` 2,920 Python lines, README 28,470 B |
| tiers (`Slowdown`, `Remote` definitions, `Escape`) | 11 / 2,130 | 4 (+8 unsure joins) | the tier half of `benchmarks/slowdown` |
| formalism v2 | 13 / 1,804 | 20, 3 named assumptions | `formalism.py` 1,082 lines; 3 `claim_ids.json` rows |
| Capability (with Coverage's `CapabilityFloor`) | 15 / 1,363 | 21 | README 13,875 B |
| cover theorem, BitView | 12 / 1,130 | 3 + 1 property | |
| ServingBlocks | 6 / 1,037 | 9 | `theorem.py` and what only it reads |
| Covering's rest, CoveringTwoCosts | 8 / 911 | 2 (compute-accounting) | |
| Factor, Witness, Witnesses | 4 / 815 | (counted above) | |
| Coverage, CoverageToy, Tasks | 5 / 358 | (in Capability's 21) | |

## Differences page / Glossary, settled for the page (list for top's Glossary PR) [read]

- **ρ := nc/|C| is the slowdown.** The Glossary row retiring ρ goes, and slowdown(𝒢, 𝒜, η) gives way to nc/|C|.
- **θ is "registered information"**, as top settled.
- **"at most c gates"**, not "exactly c".
- **"inputs outside θ"**, the page's "unregistered input", against the Glossary's retired "unregistered input" and its "fresh input".
- **"the slowdown theorem"** for "the slowdown lower bound".
- **New terms:** word, width, cut κ(A → B) and κ_θ(B), matmul structure, MAC, operand word, T1 (operands are hard), T2 (exports are wide), training circuit, joined, pairs within reach, export width.
- **Goes up, because it would change other lanes' text: "wire" and "gate".** The page calls every node a wire and a non-input a gate. Core's one `Verity.Circuit`, which every lane reads, calls every node a gate. The page's Boolean circuit has fan-in at most 2, and Core's `Circuit.Boolean` has positive fan-in and any basis. My recommendation: training's statement uses the page's words and gives Core's names. Core is not renamed.

## Renames all the way down (default yes); the noes, one reason each

- **(H) and (E) to T1 and T2:** yes in prose and Lean (Tue 9 AM window, with moved records). **No for the claim ids** `training-cut-hardness` and `training-cut-exports`: they are what a lock and a credit record name. A new id would be a second record of one assumption.
- **`HE` and `HEShared`: no.** The page's own table names them.
- **`Slowdown.Cut` to the page's `Nci.Cut`: no.** "NCI" is retired. The page's citation is the thing to fix.
- **`min_cut.py`: no for now.** `tests/test_one_slowdown.py` and four callers load it by that path. It is renamed with them in check 5's PR, not separately.
- **The stored key `rho` (circuit properties): yes**, with its schema. Stored runs rerun.

## Checks 2 to 5 on main by Tue 17 Nov 1:00 PM PST

- **Five workers, disjoint by files:**
  - A: formalism v2, Lean and Python, plus `claim_ids.json`.
  - B: Capability, Coverage and Tasks, Covering's rest, and the compose remote README's mentions.
  - C: Threads, Layered and Fit, Actual/Slowdown/Threaded, the cover theorem, the inference-only properties and Proofs, and `benchmarks/nci`.
  - D: the tiers, ServingBlocks, and `benchmarks/slowdown` apart from `min_cut.py`.
  - PAGE: check 1, check 2's renames, check 5's READMEs, and the term list to top.
- **Shared files are serialized, not split.** `security/lean-audit.json` (each PR runs `check.py --update` on its rebase), the two aggregators, and `Witness.lean` and `Witnesses.lean`: each worker removes only its own lines, and the last PR deletes the files.
- **Landing window:** Tue 7:00 AM to 12:00 PM PST of its own. Order A, B, D, C by imports, PAGE's renames in the 9:00 AM Lean window, the READMEs last.
- **Reader:** one held for the window who reads the lock diffs. Lean is told before C, since 41 of its guarantees go.
- **Risks to Tue 1:00 PM:** the 8 unsure Remote joins and #10383 need a ruling by Mon evening. Every family needs a Lean build per PR, and C alone is about a quarter of the area.

## Every listed guarantee we would delete or reword, one list (for Daniel at 6:00 PM) [ran]

### D1 formalism v2 (training): 20
`Slowdown.Formalism.SecurityProofs.Counting`, `Slowdown.Formalism.SecurityProofs.Counting_honest`, `Slowdown.Formalism.SecurityProofs.Counting_nontrivial`, `Slowdown.Formalism.SecurityProofs.FitHonestServing`, `Slowdown.Formalism.SecurityProofs.FitServing`, `Slowdown.Formalism.SecurityProofs.FitServing_honest`, `Slowdown.Formalism.SecurityProofs.FitServing_nontrivial`, `Slowdown.Formalism.SecurityProofs.FitTraining`, `Slowdown.Formalism.SecurityProofs.FitTraining_honest`, `Slowdown.Formalism.SecurityProofs.FitTraining_nontrivial`, `Slowdown.Formalism.SecurityProofs.Security`, `Slowdown.Formalism.SecurityProofs.Security_honest`, `Slowdown.Formalism.SecurityProofs.Security_nontrivial`, `Slowdown.Formalism.SecurityProofs.Slowdown`, `Slowdown.Formalism.SecurityProofs.Slowdown_honest`, `Slowdown.Formalism.SecurityProofs.Slowdown_nontrivial`, `Slowdown.Formalism.SecurityProofs.TokenCap`, `Slowdown.Formalism.SecurityProofs.TokenCapClean`, `Slowdown.Formalism.SecurityProofs.TokenCap_honest`, `Slowdown.Formalism.SecurityProofs.TokenCap_nontrivial`

### D2 Capability (training): 21
`Slowdown.SecurityProofs.CapabilityFloor`, `Slowdown.SecurityProofs.CapabilityFloorHonest`, `Slowdown.SecurityProofs.CapabilityFloorNontrivial`, `Slowdown.SecurityProofs.HeldBudget`, `Slowdown.SecurityProofs.HeldBudgetHonest`, `Slowdown.SecurityProofs.HeldBudgetNontrivial`, `Slowdown.SecurityProofs.ImportOrRecompute`, `Slowdown.SecurityProofs.ImportOrRecomputeHonest`, `Slowdown.SecurityProofs.ImportOrRecomputeNontrivial`, `Slowdown.SecurityProofs.RecomputeByGates`, `Slowdown.SecurityProofs.RecomputeByGatesHonest`, `Slowdown.SecurityProofs.RecomputeByGatesNontrivial`, `Slowdown.SecurityProofs.RecomputeByState`, `Slowdown.SecurityProofs.RecomputeByStateHonest`, `Slowdown.SecurityProofs.RecomputeByStateNontrivial`, `Slowdown.SecurityProofs.RecoveredBudget`, `Slowdown.SecurityProofs.RecoveredBudgetHonest`, `Slowdown.SecurityProofs.RecoveredBudgetNontrivial`, `Slowdown.SecurityProofs.StatHeldBudget`, `Slowdown.SecurityProofs.StatHeldBudgetHonest`, `Slowdown.SecurityProofs.StatHeldBudgetNontrivial`

### D3 Threads, Layered, Fit (lean): 41
`ResourceAccounting.SecurityProofs.Actual.SlowdownServerBlocksCap`, `…CapSlots`, `…Actual.ThreadedRate`, `…ThreadedRateSlots`, `…ThreadedRuntime`, `…ThreadedRuntimeDark`, `…ThreadedRuntimeDarkSlots`, `…ThreadedRuntimeSlots`, `Slowdown.SecurityProofs.FitBound`, `Slowdown.SecurityProofs.FitBoundSlots`, `Slowdown.SecurityProofs.FitRate`, `Slowdown.SecurityProofs.LayeredTraining`, `Slowdown.SecurityProofs.LayeredTraining_honest`, `Slowdown.SecurityProofs.LayeredTraining_nontrivial`, `Slowdown.SecurityProofs.ReferenceThreaded`, `Slowdown.SecurityProofs.SlowdownThreadedSlots_honest`, `Slowdown.SecurityProofs.SlowdownThreadedSlots_nontrivial`, `Slowdown.SecurityProofs.SlowdownThreaded_honest`, `Slowdown.SecurityProofs.SlowdownThreaded_nontrivial`, `Slowdown.SecurityProofs.ThreadedOfLayered`, `Slowdown.SecurityProofs.ThreadedReads`, `Slowdown.SecurityProofs.ThreadedReadsSlots`, `Slowdown.SecurityProofs.ThreadedThrottled`, `Slowdown.SecurityProofs.ThreadedThrottledDark`, `Slowdown.SecurityProofs.ThreadedThrottledDarkSlots`, `Slowdown.SecurityProofs.ThreadedThrottledDarkSlots_nontrivial`, `Slowdown.SecurityProofs.ThreadedThrottledDark_nontrivial`, `Slowdown.SecurityProofs.ThreadedThrottledOn`, `Slowdown.SecurityProofs.ThreadedThrottledOnSlots`, `Slowdown.SecurityProofs.ThreadedThrottledOnSlots_nontrivial`, `Slowdown.SecurityProofs.ThreadedThrottledOn_nontrivial`, `Slowdown.SecurityProofs.ThreadedThrottledSlots`, `Slowdown.SecurityProofs.ThreadedThrottledSlots_honest`, `Slowdown.SecurityProofs.ThreadedThrottledSlots_nontrivial`, `Slowdown.SecurityProofs.ThreadedThrottled_honest`, `Slowdown.SecurityProofs.ThreadedThrottled_nontrivial`, `Slowdown.SecurityProofs.ThreadedTraining`, `Slowdown.SecurityProofs.ThreadedTrainingOn`, `Slowdown.SecurityProofs.ThreadedTrainingOn_nontrivial`, `Slowdown.SecurityProofs.ThreadedTraining_honest`, `Slowdown.SecurityProofs.ThreadedTraining_nontrivial`

### D4 cover theorem (training): 3
`Slowdown.Cover.SecurityProofs.CoverTraining`, `Slowdown.Cover.SecurityProofs.CoverTraining_honest`, `Slowdown.Cover.SecurityProofs.CoverTraining_nontrivial`

### D5 Covering, CoveringTwoCosts (compute-accounting): 2
`Slowdown.SecurityProofs.CoveringBound`, `Slowdown.SecurityProofs.CoveringBoundTwoCosts`

### D6 ServingBlocks (training): 9
`ResourceAccounting.SecurityProofs.ServingBlocks.ServingBlockBound`, `ResourceAccounting.SecurityProofs.ServingBlocks.ServingBlockBoundRemote`, `ResourceAccounting.SecurityProofs.ServingBlocks.ServingBlockBoundRemote_honest`, `ResourceAccounting.SecurityProofs.ServingBlocks.ServingBlockBoundRemote_nontrivial`, `ResourceAccounting.SecurityProofs.ServingBlocks.ServingBlockBound_honest`, `ResourceAccounting.SecurityProofs.ServingBlocks.ServingBlockBound_nontrivial`, `ResourceAccounting.SecurityProofs.ServingBlocks.UncertifiedElementary`, `ResourceAccounting.SecurityProofs.ServingBlocks.UncertifiedElementary_honest`, `ResourceAccounting.SecurityProofs.ServingBlocks.UncertifiedElementary_nontrivial`

### D7 tiers (lean): 4
`Slowdown.SecurityProofs.Slowdown`, `Slowdown.SecurityProofs.SlowdownExecutionPhases`, `Slowdown.SecurityProofs.SlowdownThreaded`, `Slowdown.SecurityProofs.SlowdownThreadedSlots`

### D8 inference-only check properties over older bounds (training 5, lean 4): 9
`InferenceOnlyCheck.Properties.SecurityProofs.CheckedCoverTraining`, `InferenceOnlyCheck.Properties.SecurityProofs.DescThrottles`, `InferenceOnlyCheck.Properties.SecurityProofs.DescThrottlesDark`, `InferenceOnlyCheck.Properties.SecurityProofs.DescThrottlesDarkSlots`, `InferenceOnlyCheck.Properties.SecurityProofs.DescThrottlesSlots`, `InferenceOnlyCheck.Properties.SecurityProofs.Throttles`, `InferenceOnlyCheck.Properties.SecurityProofs.ThrottlesDark`, `InferenceOnlyCheck.Properties.SecurityProofs.ThrottlesDarkSlots`, `InferenceOnlyCheck.Properties.SecurityProofs.ThrottlesSlots`

### R Cut, reworded (training): 12
`Slowdown.Cut.SecurityProofs.Bound`, `Slowdown.Cut.SecurityProofs.Bound_nontrivial`, `Slowdown.Cut.SecurityProofs.CountHE`, `Slowdown.Cut.SecurityProofs.CountHE_honest`, `Slowdown.Cut.SecurityProofs.CountHE_nontrivial`, `Slowdown.Cut.SecurityProofs.Cut_honest`, `Slowdown.Cut.SecurityProofs.HE`, `Slowdown.Cut.SecurityProofs.HEShared`, `Slowdown.Cut.SecurityProofs.HEShared_nontrivial`, `Slowdown.Cut.SecurityProofs.HE_nontrivial`, `Slowdown.Cut.SecurityProofs.Slowdown`, `Slowdown.Cut.SecurityProofs.Slowdown_nontrivial`

### U Remote joins, unsure: 8
`Slowdown.SecurityProofs.CertificateSlack`, `Slowdown.SecurityProofs.SlowdownRemote`, `Slowdown.SecurityProofs.SlowdownRemoteMonitoringPeriod`, `Slowdown.SecurityProofs.SlowdownRemoteMonitoringPeriod_honest`, `Slowdown.SecurityProofs.SlowdownRemoteMonitoringPeriod_nontrivial`, `Slowdown.SecurityProofs.SlowdownRemote_honest`, `Slowdown.SecurityProofs.SlowdownRemote_nontrivial`, `Slowdown.SecurityProofs.VerdictLag`


### M moved, statement unchanged (lean): 2
`ResourceAccounting.SecurityProofs.Actual.ServerBlocks`, `ResourceAccounting.SecurityProofs.Actual.ServerBlocksAt`

