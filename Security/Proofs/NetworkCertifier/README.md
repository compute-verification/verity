# Network certifier Lean package (`network-timing`)

Definitions, guarantees and proofs for the bucketed certifier's timing channel: how much a covert sender inside a
spatial unit X can carry across its boundary ∂X through **when** frames leave, once the declared Program fixes their
content and order. Conventions follow POUS's package (`verity/core/protocols/pous/lean/`): the same toolchain and Mathlib pin,
`autoImplicit` off, no `axiom` (an assumption is a named `Prop` hypothesis), and a check file.

~~~text
../../Definitions/NetworkCertifier/Grid.lean   trusted: rows, grids, occupancy, left-packing, fill, and the FIFO certifier (queue, released, fifoGrid)
../../Definitions/NetworkCertifier/Run.lean    trusted: statuses, observations, the certifier, the interaction (trace, obsRun, wireRun), the
                               Program, acceptance (Accepted, AcceptedUpTo) and accepted observation sequences (AccSeqs)
../../Definitions/NetworkCertifier/Ingress.lean  trusted: the interaction read inward, a link's frames with the timing erased (rowFrames,
                               linkFrames), and acceptance as it bears on ingress (StatusComplete)
../../Definitions/NetworkCertifier/Schedule.lean trusted: the Program's logical schedule as `schedule.py` defines it (ClockSync, SyncSet,
                               LogicalFrame, eligible, handingBuckets, enqueued, expectedRow) and the constant-rate
                               Program it gives (constantRate)
../../Models/NetworkCertifier/Assumptions.lean     trusted: the named hypotheses W2–W6 (W4's allocation among them), J, and Theorem 3's two
                               dataflow conditions
../../Models/NetworkCertifier/Guarantees.lean      trusted: each guarantee as a `def G : Prop` (Proposition 2, Theorem 3, Theorem 3 for
                               the constant-rate Program, the ingress bound)
Lemmas.lean     the statements that are only steps of the guarantees' proofs, Theorem 1(b) among them,
                               each a `def G : Prop` under the same namespace (`Guarantees.G`)
../NetworkCertifier.lean  proved: `SecurityProofs.G : Guarantees.G` for each guarantee and lemma, from the modules below
Occupancy.lean  Proposition 2 (the occupancy counts, fill, and the FIFO certifier's tightness)
Counting.lean   Theorem 1(b) (the tree lemma on accepted runs, zero error, W6 on accepted runs)
Advice.lean     Theorem 3 (the 2^A cap, one clock-sync value per window, the fixed clock, the
                                         block caps)
Schedule.lean   Theorem 3 for the constant-rate Program (the advice cap, causality, seven bits
                                         per window opened, the composed statement) and its witnesses
Ingress.lean    the ingress bound (the occupancy of the ingress grid, read from inside X)
PerWindow.lean  the last two over the run's first `m` whole windows: the factor per window and
                                         per link-window
Sanity.lean     a satisfiability witness for every named hypothesis, the joint witnesses, and the
                                         toy of the September draft
Code.lean       the certifier program (`certifier_program`, `NetworkCertifier`), which the difftest runs, equal to the
                               definitions above: four theorems, not guarantees (below)
Accepted.lean   an accepted run of the network certificate check is a run of the constant-rate Program
                               (`security_properties`' `AcceptedRunsAreProgramRuns`, through Code.lean), and
                               Theorem 3's count of what accepted runs can carry (`EncardAcceptedObservationsLe`),
                               each unless the run yields a SHA-512 collision or an LMS forgery
../../lean-audit.json          the guarantee records (`security`'s lock) for tools/verity/lean/check.py
CheckAxioms.lean               one #print axioms line per statement and witness, then three more lemmas
~~~

~~~bash
lake exe cache get                                          # Mathlib's prebuilt oleans; never build Mathlib from source
lake build                                                  # NetTiming (the default target)
python3 ../../../tools/verity/lean/check.py .                      # Verity's Lean check
lake env lean CheckAxioms.lean                              # the axioms of the 35 statements, 16 witnesses and 3 lemmas
# the difftest is the certifier program's: see its package, ../../../verity/core/protocols/certifier/lean/
~~~

**Trusted without proof:** the statements in `../../Definitions/NetworkCertifier/`, `../../Models/NetworkCertifier/Assumptions.lean` and
`../../Models/NetworkCertifier/Guarantees.lean`; Lean's kernel; the Mathlib pin. Everything else is checked by machine. Docstring paths such
as `docs/…` and `internal/…` name network accounting's project store, where the covert-capacity note
(`docs/network-transparency/timing-channel.md`) and the reviews live.

**The audit.** `tools/verity/lean/check.py` audits every declaration under `NetTiming`: allowed axioms (only `propext`,
`Classical.choice` and `Quot.sound`), named assumptions, escapes, orphan files, the import layers (`NetTiming.Protocol`,
`NetTiming.Assumptions` and `NetTiming.Guarantees` import only one another and Mathlib, never a proof), the dependency
record, a kernel replay, and the guarantees' records, whose meaning it follows through those three (`meaning`) and the
certifier program's modules. The difftest (`runs` in the certifier program's own `lean-audit.json`) checks that on 10,000
random instances the program gives what the Python audit gives; `Code.lean` proves the program equal to these
definitions.
When a changed record lands on `main`, Daniel gets a DM showing it (Verity's `AGENTS.md`).

## The model

**The run.** A step is one bucket of the whole boundary ∂X. At each step the operator's move is the ingress, the sender's
move uses the secret, and the certifiers record the observation (the egress rows, one jitter class, and every boundary link's
status) and emit the wire. `Interaction.trace` builds the run by recursion; the operator reacts to everything it has
observed.

**Acceptance is separate from emission.** `Accepted P op k o a` holds when, at every step before `k`, the recorded rows are
the Program's rows for the ingress so far and the declared advice, and every status is complete. It reads only rows and
statuses (W5's record). A violation is recorded and the run rejected, with no halt, so every theorem is about accepted
runs, and no violation policy is modelled.

**Ingress is the mirror image** (`NetTiming.Protocol.Ingress`). The same `Interaction` is read inward: the secret is the
outside's message `m`, held by `snd m` (the operator and colluding endpoints, choosing when ingress frames arrive); `op`
is X's devices; the record's rows are what the ingress links release into X; the wire is everything that enters X. The
outside sees every record and can recompute X's moves from them, so it reacts to X's egress. The ingress frames'
content and order are the Program's input, charged as reads; `Guarantees.EncardIngressObsDecodableLe` bounds what
their timing carries into X for a decoder reading the records, and `Guarantees.EncardIngressDecodableLe` for one
reading the wire. Both are proved: on the runs counted, a record sequence is determined by each step's frame count on
each link and its jitter class, since under W4 a row is its frames in slot order and the content fixes which frames
come next. `Guarantees.EncardIngressObsDecodableLeContents` and `Guarantees.EncardIngressDecodableLeContents` drop the
one prescribed content: messages whose frames follow some content assignment in a set `Cs` number at most `#Cs` times
the timing bound: for the set `Ms` of messages counted, `log2 #Ms ≤ log2 #Cs` plus the timing bits. The K accounting
charges content bits, as reads, plus timing bits. Taking `log2 #Cs` to be those content bits is an accounting step, not
part of the theorem. It is exact under fixed-length frames (W3): N frames of B bits have `2^(N·B)` contents, so for
`Cs` all of them, `log2 #Cs` is the `N·B` bits read.

**The shipped schedule** (`NetTiming.Protocol.Schedule`). `constantRate T W ss frames` is `schedule.py`'s Program:
windows of `T` buckets, `W` of them with advice, one clock-sync value of `ss` (Σ_sync) per window, and the Program's
frames, anchors and logical indices `frames ins c ℓ`, read from the ingress and the content advice. A frame is eligible
at `anchor + δ + ⌊λ/ρ⌋`, handed at `h(k) = max(eligible(k), h(k−1))`, and released by `fifoGrid`. Its advice type is
`Ac × (Fin W → ss.Elem)`, so the sync is its only timing advice by definition, and Theorem 3's clock-sync form holds for
it with no dataflow hypothesis (`Guarantees.EncardAccSeqsLeConstantRate`). The block cap's two conditions fail for it in
their natural instances (`constantRate_not_framesRead_counts`, `constantRate_not_windowBlocks`), so the advice cap is its
bound: the horizon is one block (`timing-channel.md` §2.4).

## The trusted layer (`NetTiming.Assumptions`)

| Note | Lean | Says |
|---|---|---|
| W1 fixed parameters | structural | the links, window ends, frame size, jitter bound and view are arguments fixed before the run; the certifier has no secret input |
| W2 regeneration | `Regenerates w view` | the wire is a function of the step and the observation, faults included |
| W3 padding | `Padded w B` | every released frame has length B |
| W4 packing | `Packed w` | every recorded row is left-packed |
| W4 allocation | `Allocated w r` | link `ℓ`'s every recorded row holds at most `r ℓ` real frames |
| W5 untimed status | `UntimedStatus w winEnd` | a status other than complete appears only at a window's last step |
| W6 fail-closed | `FailClosed w winEnd` | after a fault the window's later rows are empty, and a status fails at the window's end |
| J jitter | `JitterBounded w D` | every step's jitter class is below D |
| Theorem 3, condition 1 | `FramesReadNoTimingAdvice P step occ` | a step's rows read the timing advice only through the predicted occupancy |
| Theorem 3, condition 2 | `OccupancyReadsOwnBlock blk occ` | a cell's predicted occupancy reads only its own block's timing advice |

For the ingress bound the same `Prop`s are read inward: `Regenerates` is W2 for what enters X, `Packed` and
`Allocated` are W4 for the rows the ingress links release into X, and `JitterBounded` is J for their release times.

`NetTiming.SecurityProofs.Sanity` shows every hypothesis is satisfiable: one certifier meets W2–W6 and J together, one
Program meets both conditions while its timing advice still changes its rows, `joint_witness` meets every hypothesis
of `Guarantees.EncardDecodableLeBlocks` at once, with a reacting operator, and `ingress_witness` meets every hypothesis
of `Guarantees.EncardIngressDecodableLe` at once, with a bound it meets exactly (`ingress_witness_tight`).
`allocated_witness` meets them again with an allocation below the slot count (`r = 1 < S = 2`): its 8 messages meet
that bound exactly, below the 27 of `r = S` (`allocated_witness_tight`), so `Allocated` has content.

## The records

The lock (`lean-audit.json`'s `guarantees`) holds the 22 statements of `Guarantees.lean`, each listed as
`NetTiming.SecurityProofs.G` and proved by the lemma of the same statement whose name it spells in UpperCamelCase. All 22
are under `shape.frozen` as `frozen (no-function)`: each is stated over the model (an arbitrary `Interaction`, an abstract
`Program`'s `AccSeqs`, `expectedRow`, or the grids themselves), not over the network check's code, and an accepted run
reaches them through `AcceptedRunsAreProgramRuns`. Four are the theorems whose bounds the calibration reports
(`verity.core.protocols.certifier.calibration`, `PROTOCOL.md`'s "Capacity"):

- egress, for the shipped schedule (`NetTiming.SecurityProofs.Schedule`, `NetTiming.SecurityProofs.PerWindow`):
  `EncardDecodableLeConstantRate` and `EncardDecodableLeConstantRatePerWindow`;
- ingress, the K charge (`NetTiming.SecurityProofs.Ingress`, `NetTiming.SecurityProofs.PerWindow`):
  `EncardIngressObsDecodableLe` and `EncardIngressObsDecodableLePerWindow`.

The other 18 are the rest of the Capacity table's bounds and the schedule's causality rule:

- **Proposition 2** (`NetTiming.SecurityProofs.Occupancy`): `CardFree`, `CardLeftPacked`, `CardCountsEqLe`,
  `EncardFifoGridsLeOfAvailable`, `EncardFifoGrids`.
- **Theorem 3** (`NetTiming.SecurityProofs.Advice`): `EncardAccSeqsLeSync`, `EncardAccSeqsLeFixedClock`,
  `EncardAccSeqsLeBlocks`, `EncardAccSeqsLeBlocksCounts`.
- **Theorem 3 for the constant-rate Program** (`NetTiming.SecurityProofs.Schedule`, `NetTiming.SecurityProofs.PerWindow`):
  `EncardAccSeqsLeConstantRate`, `EncardAccSeqsLeConstantRateFixedClock`, `ExpectedRowCausal`,
  `EncardAccSeqsLeConstantRateWindows`, `EncardAccSeqsLeConstantRatePerWindow`.
- **The ingress bound** (`NetTiming.SecurityProofs.Ingress`, `NetTiming.SecurityProofs.PerWindow`):
  `EncardIngressDecodableLe`, `EncardIngressObsDecodableLeContents`, `EncardIngressDecodableLeContents`,
  `EncardIngressObsDecodableLeContentsPerWindow`.

**The program's agreements, proved and audited, not guarantees.** Four theorems tie the difftest to these definitions
(`NetTiming.SecurityProofs.Code`): the difftest holds the Python audit
to the certifier program (`verity/core/protocols/certifier/lean/`, Lean core only, `NetworkCertifier`), and
these say the program computes what the definitions above define. Each is stated in `Code.lean` over both sides'
definitions, relating them by their fields, since `security` requires nothing but Mathlib
(`tests/test_lean_packages.py`) and so `Models/` cannot name the program:

- `CodeExpectedRow`: `NetworkCertifier.expectedRow` is `expectedRow` for clock syncs with the same δ and ρ and frames with the same
  payloads, anchors and logical indices (the difftest's `rows`, `diff` and `mismatch`);
- `CodeAcceptedConstantRate`: `NetworkCertifier.Accepted` of `NetworkCertifier.constantRate` is `Accepted` of `constantRate` for observations
  with the same rows and complete statuses, operators giving the same ingress on them, related frames and advice naming
  the same values (`accepted`);
- `CodeStatusComplete`: `NetworkCertifier.StatusComplete` is `StatusComplete` for observations with the same complete statuses
  (`complete`);
- `CodeSyncSet`: every sync set of the program has a reference sync set with the same values and conversely (`advice`).

They aren't guarantees because `four_names` holds every `NetTiming` guarantee to `SecurityProofs.G : Guarantees.G`,
and a `Guarantees` module can't state them: it would have to import the program. They are built and checked like every
declaration, and no citation may call them guarantees.

**Proved and audited, not guarantees.** The 13 statements that are only steps of these proofs are lemmas, in
`Lemmas.lean` under their old names (`Guarantees.G`): nothing relies on them as proved (README "Glossary"), so they have
no record. Each is still `SecurityProofs.G : Guarantees.G`, built and checked like every declaration:

- **Proposition 2** (`NetTiming.SecurityProofs.Occupancy`): `FillInjective`, `ExistsFillOfLeftPacked`, `CardCountsLeLe`,
  `ReleasedSelf`, `SumReleasedLe`.
- **Theorem 1(b)** (`NetTiming.SecurityProofs.Counting`): `EncardLeProdOfBranching`, `EncardAcceptedRunsLe`,
  `EncardObsDecodableLe`, `EncardDecodableLe`, `NoFaultOfAccepted`.
- **Theorem 3** (`NetTiming.SecurityProofs.Advice`): `RecordedRowEqFill`, `EncardAccSeqsLeAdvice`,
  `EncardDecodableLeBlocks`.

So are the 16 witnesses, checked tests rather than guarantees, under their own names (`NetTiming.SecurityProofs.Sanity`,
and the last three in `NetTiming.SecurityProofs.Schedule`):
`witnessCertifier_assumptions`, `witnessCertifier_jitter_varies`, `witnessProgram_conditions`,
`witnessProgram_timing_matters`, `joint_witness`, `joint_witness_bound`, `zP_accepts_one`, `zP_constOp_accepts_one`,
`ingress_witness`, `ingress_witness_tight`, `allocated_witness`, `allocated_witness_tight`, `toy_leftPacked`,
`constantRate_not_framesRead_counts`, `constantRate_not_windowBlocks`, `constantRate_joint_witness`; and
`occ_fill`, `leftPacked_occ_fill` and `leftPacked_fifoGrid` (`NetTiming.SecurityProofs.Occupancy`). They use only the
three axioms, but they have no record in `lean-audit.json`, so no citation may call them guarantees. Listing one is an
`--update` like any other.

The statements were frozen and reviewed by a red team before their proofs, and a second round's verdict was GO. The
ingress bound's statements were frozen before their proofs as well; the review of the first two is on the pull request
that added them (danielreuter/verity#894), and the review of the content versions is on the one stacked on it. The
constant-rate Program's five statements and the definitions they read were frozen on `sorry` stubs and recorded before
their proofs. Their statement review, on danielreuter/verity#949, gave GO with one change: the four counts take
`n ≤ W·T`. All 35 statements (22 guarantees, 13 lemmas) and 16 witnesses are proved. The reference
implementation, `verity/core/protocols/certifier` (`verity.core.protocols.certifier`), mirrors 19 of the statements and the three
lemmas by name in its tests; its `PROTOCOL.md` lists them per test file. The witnesses have no Python counterpart.

## Modelling choices a reader of the statements checks

1. **Steps are buckets across all of ∂X.** The note states its bounds per link-window. The guarantees count steps, and
   Theorem 1(b)'s tree lemma is still stated per step. The per-window ones read the horizon of the run's first `m` whole
   windows of `T` buckets, `n = m·T`, each with its parent's hypotheses at that horizon and no other: the shipped
   schedule's bound is `#Ac · (#Σ_sync · D^T)^m` (`EncardAccSeqsLeConstantRatePerWindow`,
   `EncardDecodableLeConstantRatePerWindow`), and the ingress bound `(∏_ℓ (r ℓ + 1)^T · D^T)^m` per content assignment
   (`EncardIngressObsDecodableLePerWindow`, and its `Contents` version). The egress factor per window is the bound's,
   not a count per window: content advice and an earlier window's sync can first show in a later window, so one window
   may continue a given prefix in more than `#Σ_sync · D^T` ways. So the egress bound is cumulative from the run's
   first window, and a temporal unit that starts later is not bounded by its own windows. Ingress is not affected:
   `Packed`, `Allocated` and `JitterBounded` bound every step's branching whatever came before.
2. **The Program reads ingress up to and including step t.** The note's W1 lag allows only ingress before t. This is
   more permissive, so the bounds cover more Programs.
3. **Theorem 3 counts all accepted sequences,** not only those a run produces, which makes it stronger and needs no
   interaction.
4. **Jitter** is a free per-step class below D. The operator reads it, since it reads the whole observation.
5. **The integrity term** (wrong proof units) is not modelled: this is the exact binding.
6. **Statuses** cover every boundary link, ingress included, since an ingress status is an egress channel.
7. **Counts** are `Set.encard` in `ℕ∞`, so a bound also says the set is finite.

The ingress bound adds these:

8. **Ingress reuses the interaction, read inward.** The outside's party is `snd`, X's devices are `op`, and both see
   every record. The outside gets every move of X's (recomputed from the records) rather than only the egress wire,
   which is at least as strong as the real outside. X's egress rows are determined by the records too: the ingress
   `Obs` holds only the ingress rows, but X's enqueues are `op`'s moves (functions of the records), the egress certifier
   releases them by W4, and an outside fault on an egress link shows as a status, which `Obs` holds. So the outside's
   feedback is complete. A decoder inside X may read what X's devices read: the record version covers it, and the wire
   version follows under W2. Read inward, W2 (`Regenerates`) also covers whatever outside-influenced thing X can see of
   its own egress (backpressure, release timing): it is all a function of the record.
9. **Conditioned on the content.** The messages counted are those whose frames on each link, in the order released, are
   a prefix of one prescribed list `c ℓ` (the Program's input, charged as reads; order is content, since the Program
   reads frames in order). How many frames have arrived by the horizon is timing, and is in the count. Messages that
   also vary the content are bounded by partitioning them by their content: the `Contents` versions count those
   following some assignment in `Cs`, at most `#Cs` × the bound.
10. **Acceptance is the statuses only** (`StatusComplete`). The verifier also checks the egress rows against the
    Program, which only removes runs, so the bound covers every accepted run and more. W5 and W6 are not needed: W2
    and W4 hold for every action, faults included, and complete statuses carry nothing.
11. **One allocation per link, one bucket width.** Link `ℓ` has allocation `r ℓ` (`Allocated`) within the shared slot
    count `S`, and every link shares the step, a bucket of ∂X. Ingress links with different bucket widths are not
    modelled. Over the first `m` whole windows each link-window contributes `(r ℓ + 1)^T`
    (`EncardIngressObsDecodableLePerWindow`).
12. **No Program, no clock.** The bound reads no Program and no advice: the outside sender chooses when frames arrive,
    so no clock pins ingress timing. Jitter on the inward release adds `D^n` (J, `D = 1` when tested ideal). W3's frame
    length and the FIFO rule are not used; the FIFO certifier attains the bound (`EncardFifoGrids`, read for the ingress
    grid).

The constant-rate Program adds these:

13. **Exactly `schedule.py`.** Each definition in `NetTiming.Protocol.Schedule` names the Python it mirrors, and computes
    the same rows: δ is a `ℕ`, ρ a positive `Rat`, `⌊λ/ρ⌋` is Python's `(λ · den) // num`, a frame anchored past the last
    window with advice is handed at `max(last, W·T, anchor)`, and frames handed at or after the horizon `W·T` are never
    enqueued. Past the horizon, where Python has no grid, the queue drains with nothing enqueued. The audit checks `W`
    windows, so the four counts take `n ≤ W·T` (`hn`): every step they count is a step the audit checks.
14. **The frames are the Program's computation, left abstract.** `frames ins c ℓ` gives link `ℓ`'s frames in canonical
    order, with their content, anchors and logical indices, from the ingress and the content advice `c`. Every guarantee
    holds for every such function, so it covers any way the Program fixes anchors (the recorded bucket of the request a
    frame answers) and indices. Content advice that moves anchors is counted in `#Ac`.
15. **One clock per spatial unit.** Every egress link reads the same window's sync (decision A's default), and the
    audit enforces it (danielreuter/verity#956). A clock per link is a different advice type, covered by
    `EncardAccSeqsLeAdvice` directly.
16. **The allocation is the slot count `S`** on every egress link, as in Python, where a row has `allocation` slots.
17. **In-set advice.** The advice ranges over `ss.Elem`, the values of Σ_sync: a run whose declared sync is outside the
    set is an `advice` violation and rejected, so its advice is never counted. `ExpectedRowCausal` takes any clock-sync
    values, in the set or not, which is what the audit's verdicts on earlier windows rely on.
