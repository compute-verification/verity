# Network certifier: the timing channel's guarantees and proofs

How much a covert sender inside a spatial unit X can carry across its boundary ∂X through **when** frames leave, once the
declared Program fixes their content and order; how much enters X the same way; and what an accepted sampled audit, and
a monitoring period of them, establishes about it. The statements are in `Security/Definitions/NetworkCertifier/`
(package `security`, namespace `NetTiming`); the proofs are here (package `security_proofs`), each guarantee as
`NetTiming.SecurityProofs.G : NetTiming.Guarantees.G` (`Security/Proofs/NetworkCertifier.lean`, from the modules
below). The certifier program
the network check runs is code Lean in `verity/core/protocols/certifier/lean/` (`NetworkCertifier`); `Code.lean` proves
it equal to the definitions here, and its difftests hold it to the Python audit.

~~~text
python tools/verity/lean/check.py --build Security    the Lean check of `security` with `security_proofs` beside it
~~~

The audit needs a machine of 160 GiB or more; `tools/verity/lean/README.md` and the lean-proofs skill
(`.agents/skills/lean-proofs/SKILL.md`) say where it runs and give the fast path. The guarantees' records are in
`Security/lean-audit.json`. Trusted without proof: the statements in `Security/Definitions/NetworkCertifier/`, Lean's
kernel and the Mathlib pin.

## Where each statement and proof lives

Statements, in `Security/Definitions/NetworkCertifier/`:

| Module | States |
|---|---|
| `Grid.lean` | rows, grids, occupancy, left-packing, fill, the FIFO certifier (`fifoGrid`) |
| `Run.lean` | statuses, observations, the interaction (`trace`, `obsRun`, `wireRun`), the Program, acceptance (`Accepted`, `AcceptedUpTo`), accepted observation sequences (`AccSeqs`) |
| `Ingress.lean` | the interaction read inward (`rowFrames`, `linkFrames`) and acceptance as it bears on ingress (`StatusComplete`) |
| `Schedule.lean` | the Program's logical schedule as `schedule.py` defines it, and the constant-rate Program (`constantRate`) |
| `Assumptions.lean` | the named hypotheses W2–W6 and J, and Theorem 3's two dataflow conditions |
| `Guarantees.lean` | Proposition 2, Theorem 3, Theorem 3 for the constant-rate Program, the ingress bound |
| `AdviceString.lean`, `Assumptions/AdviceString.lean`, `Guarantees/AdviceString.lean` | the advice string's per-window caps (`WithinCaps`), the compiler, acceptance within the caps, and the leak bound |
| `AdviceV1.lean`, `Guarantees/AdviceV1.lean`, `Guarantees/AdviceV1Kraft.lean` | advice v1's charge, `ChargeCount_v1`'s reference one for one, and the leak bound with it |
| `WindowCharge.lean`, `Guarantees/WindowCharge.lean`, `Guarantees/WindowChargeKraft.lean` | the whole window's charge, `ChargeCount_v2`'s reference one for one, and the leak bound with it |
| `Warden.lean`, `Tap.lean`, `RecordRows.lean`, `Guarantees/Warden.lean` | the network certifier on a server's uplink, which re-frames every byte into cells on a public schedule and records what it sends, as one device; the record format the Python certifier writes (`RowsFormat`); its channel bound |
| `Premises.lean` | what a property of the certifier may take besides acceptance: `DeviceIntact`, the `trusted-host` premise |
| `SampledAudit.lean`, `Guarantees/SampledAudit.lean` | the sampled audit and what its acceptance says about the egress at `δ` (`KSAccepted`) |
| `SampledCharge.lean`, `SampledReg.lean`, `TapRows.lean`, `Guarantees/SampledCharge.lean` | what the faulty proof blocks add under a sampled audit (`faultyBits`, `faultySets`, `AccSeqsSampled`), the registered inputs and their finder (`Breg`), the certifier's rows at the transfer gates (`TapOpens`, `RLeakTap`, `Btap`) |
| `Period.lean`, `HorizonLeak.lean`, `Guarantees/Period.lean`, `Guarantees/Horizon.lean`, `Guarantees/HorizonLeak.lean` | a monitoring period of sampled audits (`WindowGood`, `HorizonGood`) and a horizon of them |

Proofs, here:

| Module | Proves |
|---|---|
| `Occupancy.lean` | Proposition 2 (the occupancy counts, fill, the FIFO certifier's tightness) |
| `Counting.lean` | Theorem 1(b) (the tree lemma on accepted runs) |
| `Advice.lean` | Theorem 3 (the `2^A` cap, one clock-sync value per window, the fixed clock, the block caps) |
| `Schedule.lean`, `PerWindow.lean` | Theorem 3 for the constant-rate Program, and the bounds over a run's first `m` whole windows |
| `Ingress.lean` | the ingress bound |
| `AdviceString.lean`, `AdviceV1.lean`, `WindowCharge.lean` | the leak bound, and `BitsCount` for advice v1 and for the whole-window charge (Kraft) |
| `Warden.lean`, `Tap.lean`, `RecordRows.lean` | the network certifier's channel bound; its cells read back from its own record (`tap_cells_read`); the Python rows format is a record format (`RowsFormat.tapRecord`) |
| `SampledAuditAccepted.lean`, `SampledAuditInstance.lean` | `SampledAuditAccepted`, and its `KS` hypothesis discharged at the masked and plain records |
| `SampledCharge.lean`, `SampledLeak.lean`, `SampledLeakReg.lean`, `SampledLeakRegPlain.lean`, `TapRows.lean` | the leak bound under a sampled audit, at the registration's values, and at the certifier's rows |
| `Period.lean`, `PeriodSum.lean`, `Horizon.lean`, `HorizonLeak.lean` | the monitoring period's and the horizon's bounds |
| `Code.lean` | the certifier program equals these definitions (below) |
| `Accepted.lean` | an accepted run of the network check is a run of the constant-rate Program (`Properties.NetworkCertifier.AcceptedRunsAreProgramRuns`), unless it yields a SHA-512 collision or an LMS forgery |
| `Lemmas.lean`, `AdviceV1Counted.lean` | the statements that are only steps of these proofs, and the retired bounds with a separate count code |
| `Sanity.lean` | a satisfiability witness for every named hypothesis, and joint witnesses |
| `Witnesses.lean` | each acceptance guarantee's two lemmas: `G_honest`, its hypotheses at the deployed record, and `G_nontrivial`, its conclusion false at a verifier that accepts everything |

## The model

- **The run** (`Run.lean`). A step is one bucket of the whole boundary ∂X. At each step the operator's move is the
  ingress, the sender's move uses the secret, and the certifiers record the observation (the egress rows, one jitter
  class, every boundary link's status) and emit the wire. The operator reacts to everything it has observed.
- **Acceptance is separate from emission.** `Accepted P op k o a` holds when, at every step before `k`, the recorded
  rows are the Program's rows for the ingress so far and the declared advice, and every status is complete. A
  violation is recorded and the run rejected, with no halt, so every theorem is about accepted runs.
- **Ingress is the mirror image** (`Ingress.lean`). The same interaction read inward: the secret is the outside's
  message, the record's rows are what the ingress links release into X. The frames' content and order are the
  Program's input, charged as reads; the bounds count what their timing carries.
- **The shipped schedule** (`Schedule.lean`). `constantRate T W ss frames` is `schedule.py`'s Program: windows of `T`
  buckets, `W` of them with advice, one clock-sync value of Σ_sync per window. Its sync is its only timing advice by
  definition, so Theorem 3's clock-sync form holds for it with no dataflow hypothesis.

## The named assumptions

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
| the advice string's charge | `BitsCount`, `CompilerDeterministic`, `CompilerPrecommitted` | the charge counts, and the compiler is deterministic and fixed before the run |
| the auditor's device | `trusted-host` (`Verity.Assumptions.TrustedHost`) | the network certifier runs its code as built |
| sampling | `Claims.Sampling.KS`, `Claims.Sampling.Commitment.Binds`, `Claims.Flock.ListWithin` | sampling's knowledge soundness, the trace commitment's binding, a decoder listing every trace within Flock's radius |

For the ingress bound the same `Prop`s are read inward. `Sanity.lean` shows every hypothesis satisfiable, and meets every
hypothesis of `Guarantees.EncardDecodableLeBlocks` and of `Guarantees.EncardIngressDecodableLe` at once.

## The records

Every guarantee below is listed in `Security/lean-audit.json` as `NetTiming.SecurityProofs.G` and is under
`shape.frozen` as `frozen (no-function)`: each is stated over the model (an arbitrary `Interaction`, an abstract
`Program`'s `AccSeqs`, `expectedRow`, or the grids; advice v1's over its charge as `Definitions` states it; the
channel bound's over the certifier's code as `Definitions` states it), not over the network check's code. An
accepted run of the network check reaches them through `AcceptedRunsAreProgramRuns` (`Accepted.lean`). The calibration
(`verity.core.protocols.certifier.calibration`) reports the bounds of `EncardDecodableLeConstantRate`,
`EncardDecodableLeConstantRatePerWindow`, `EncardIngressObsDecodableLe` and `EncardIngressObsDecodableLePerWindow`.

- **Proposition 2** (`Occupancy.lean`): `CardFree`, `CardLeftPacked`, `CardCountsEqLe`, `EncardFifoGridsLeOfAvailable`,
  `EncardFifoGrids`.
- **Theorem 3** (`Advice.lean`): `EncardAccSeqsLeSync`, `EncardAccSeqsLeFixedClock`, `EncardAccSeqsLeBlocks`,
  `EncardAccSeqsLeBlocksCounts`, each under the hypotheses of `Assumptions.lean` its statement names.
- **Theorem 3 for the constant-rate Program** (`Schedule.lean`, `PerWindow.lean`): `EncardAccSeqsLeConstantRate`,
  `EncardAccSeqsLeConstantRateFixedClock`, `ExpectedRowCausal`, `EncardAccSeqsLeConstantRateWindows`,
  `EncardAccSeqsLeConstantRatePerWindow`, `EncardDecodableLeConstantRate`, `EncardDecodableLeConstantRatePerWindow`.
- **The ingress bound** (`Ingress.lean`, `PerWindow.lean`): `EncardIngressObsDecodableLe`, `EncardIngressDecodableLe`,
  `EncardIngressObsDecodableLeContents`, `EncardIngressDecodableLeContents`, `EncardIngressObsDecodableLePerWindow`,
  `EncardIngressObsDecodableLeContentsPerWindow`, under `Regenerates`, `Packed`, `Allocated` and `JitterBounded`.
- **The leak bound** (`AdviceString.lean`): `EncardAccSeqsWithinLeConstantRate`, `EncardDecodableWithinLeCompiled`,
  `EncardDecodableWithinLeCompiledPerWindow`, under `BitsCount`, `CompilerDeterministic` and `CompilerPrecommitted`.
- **Advice v1's charge** (`AdviceV1.lean`): `EncardChargedLe` (at most `2^c` decision streams charged `c` bits or
  fewer, under any registered book), and the leak bound with it and no assumption on the charge:
  `EncardAccSeqsWithinLeConstantRateV1`, `EncardDecodableWithinLeCompiledV1`,
  `EncardDecodableWithinLeCompiledPerWindowV1`. That `ChargeCount_v1`'s circuit computes the Lean charge is
  circuit-check's, not proved here. The same four with a separate count code (`…Counted`) are retired to `Lemmas.lean`:
  advice v1 codes its counts inside the arithmetic coder, which the whole-window charge counts.
- **Advice v1's whole-window charge** (`WindowCharge.lean`): `EncardChargedLeWhole`,
  `EncardAccSeqsWithinLeConstantRateV1Whole`, `EncardDecodableWithinLeCompiledV1Whole`,
  `EncardDecodableWithinLeCompiledPerWindowV1Whole`. That `ChargeCount_v2`'s circuit computes the Lean charge is
  circuit-check's, not proved here.
- **The network certifier's channel bound** (`Warden.lean`), what the client sees of the server side:
  `WardenViewEqClientView`, `EncardWardenViewsLe`, `EncardWardenDecodableLe`, under `trusted-host`.
- **The sampled audit implies `Accepted`** (`SampledAuditAccepted.lean`, `SampledAuditInstance.lean`):
  `SampledAuditAccepted`, under `Claims.Sampling.KS`.
- **The leak bound under a sampled audit** (`SampledCharge.lean`, `SampledLeak.lean`, `SampledLeakReg.lean`):
  `EncardAccSeqsSampledLe`, the count of the sequences accepted with a possible faulty set; `SampledLeakAccepted`, what
  an accepted sampled audit establishes, except with probability `δ + ε_ks + ε_bind`; and `SampledLeakRegistered`, the
  same at the registration's values, except with probability `δ + ε_ks + ε_bind + Pr[B_reg]`. Under
  `Claims.Sampling.KS` and `Claims.Flock.ListWithin`, and `SampledLeakRegistered` under
  `Claims.Sampling.Commitment.Binds` too.
- **The leak through the network certifier over a monitoring period** (`Period.lean`): `PeriodLeakAccepted`, the
  leak bound at the certifier's rows (`sampledLeak_tap`) composed with the read-back of its record (`tap_cells_read`),
  under `Claims.Sampling.KS`, `Claims.Sampling.Commitment.Binds`, `Claims.Flock.ListWithin` and `trusted-host`. The
  union over the period's live coins is not stated.
- **The faulty proof blocks over a horizon** (`Horizon.lean`): `HorizonWrongUnitsLe`, with no named assumption: over
  `m` windows on independent live coins, every window accepts while the horizon holds at least `F` faulty proof blocks
  with probability at most `(1 - q)^F + Σ_i ε_i`, whatever `m` is.
- **The leak over a monitoring period** (`HorizonLeak.lean`): `HorizonLeakAccepted`, conditional on two per-window
  inputs taken as hypotheses (the conditional escape and the leak relation at the window's faulty set), under
  `trusted-host`. The faulty proof blocks cost `(ln(1/δ)/q + 1) · (b + log2 (N + 1))` bits once over the period,
  `b = 16` by a proof block's output limit.

**The program's agreements, proved and audited, not guarantees** (`Code.lean`): `CodeExpectedRow`,
`CodeAcceptedConstantRate`, `CodeStatusComplete`, `CodeSyncSet`, `CodeChargesV1`, `CodeChargesWhole` say the certifier
program computes what these definitions define. They are stated over both sides' definitions, since `security` requires
nothing but Mathlib and so `Guarantees` cannot name the program.

**Lemmas and witnesses, not guarantees.** The statements that are only steps of these proofs are in `Lemmas.lean`
(Proposition 2's `FillInjective`, `ExistsFillOfLeftPacked`, `CardCountsLeLe`, `ReleasedSelf`, `SumReleasedLe`;
Theorem 1(b)'s `EncardLeProdOfBranching`, `EncardAcceptedRunsLe`, `EncardObsDecodableLe`, `EncardDecodableLe`,
`NoFaultOfAccepted`; Theorem 3's `RecordedRowEqFill`, `EncardAccSeqsLeAdvice`, `EncardDecodableLeBlocks`; the retired
`…Counted` bounds), the witnesses in `Sanity.lean` and `Schedule.lean`, and the honest and nontrivial lemmas in
`Witnesses.lean`. They have no record, so no citation may call them guarantees.

## Modelling choices a reader of the statements checks

1. **Steps are buckets across all of ∂X.** The per-window guarantees read the horizon of the run's first `m` whole
   windows of `T` buckets, `n = m·T`: the shipped schedule's bound is `#Ac · (#Σ_sync · D^T)^m`
   (`EncardAccSeqsLeConstantRatePerWindow`, `EncardDecodableLeConstantRatePerWindow`), and the ingress bound
   `(∏_ℓ (r ℓ + 1)^T · D^T)^m` per content assignment (`EncardIngressObsDecodableLePerWindow`). The egress factor per
   window is the bound's, not a count per window: content advice and an earlier window's sync can first show in a later
   window. So the egress bound is cumulative from the run's first window, and a temporal unit that starts later is not
   bounded by its own windows. Ingress is not affected.
2. **The Program reads ingress up to and including step t.** This is more permissive than ingress before t only, so
   the bounds cover more Programs.
3. **Theorem 3 counts all accepted sequences,** not only those a run produces, which makes it stronger and needs no
   interaction.
4. **Jitter** is a free per-step class below D. The operator reads it, since it reads the whole observation.
5. **The integrity term** (faulty proof blocks) is not modelled in the timing bounds: this is the exact binding. The
   sampled audit's bounds add it.
6. **Statuses** cover every boundary link, ingress included, since an ingress status is an egress channel.
7. **Counts** are `Set.encard` in `ℕ∞`, so a bound also says the set is finite.

The ingress bound adds these:

8. **Ingress reuses the interaction, read inward.** The outside's party is `snd`, X's devices are `op`, and both see
   every record, which is at least as strong as the real outside. A decoder inside X may read what X's devices read:
   the record version covers it, and the wire version follows under W2.
9. **Conditioned on the content.** The messages counted are those whose frames on each link, in the order released,
   are a prefix of one prescribed list (the Program's input, charged as reads). How many frames have arrived by the
   horizon is timing, and is in the count. The `Contents` versions count messages following some assignment in `Cs`,
   at most `#Cs` times the bound.
10. **Acceptance is the statuses only** (`StatusComplete`). The verifier also checks the egress rows against the
    Program, which only removes runs. W5 and W6 are not needed.
11. **One allocation per link, one bucket width.** Link `ℓ` has allocation `r ℓ` (`Allocated`) within the shared slot
    count `S`. Ingress links with different bucket widths are not modelled.
12. **No Program, no clock.** The bound reads no Program and no advice: the outside sender chooses when frames arrive.
    Jitter on the inward release adds `D^n`. The FIFO certifier attains the bound (`EncardFifoGrids`).

The constant-rate Program adds these:

13. **Exactly `schedule.py`.** Each definition in `Schedule.lean` names the Python it mirrors and computes the same
    rows. The audit checks `W` windows, so the four counts take `n ≤ W·T`: every step they count is a step the audit
    checks.
14. **The frames are the Program's computation, left abstract.** `frames ins c ℓ` gives link `ℓ`'s frames in canonical
    order from the ingress and the content advice `c`. Every guarantee holds for every such function. Content advice
    that moves anchors is counted in `#Ac`.
15. **One clock per spatial unit.** Every egress link reads the same window's sync, and the audit enforces it. A clock
    per link is a different advice type, covered by `EncardAccSeqsLeAdvice` directly.
16. **The allocation is the slot count `S`** on every egress link, as in Python, where a row has `allocation` slots.
17. **In-set advice.** The advice ranges over `ss.Elem`, the values of Σ_sync: a run whose declared sync is outside the
    set is an `advice` violation and rejected. `ExpectedRowCausal` takes any clock-sync values, in the set or not, which
    is what the audit's verdicts on earlier windows rely on.
