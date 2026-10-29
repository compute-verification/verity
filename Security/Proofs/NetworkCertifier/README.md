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
../../Definitions/NetworkCertifier/Assumptions.lean     trusted: the named hypotheses W2–W6 (W4's allocation among them), J, and Theorem 3's two
                               dataflow conditions
../../Definitions/NetworkCertifier/Guarantees.lean      trusted: each guarantee as a `def G : Prop` (Proposition 2, Theorem 3, Theorem 3 for
                               the constant-rate Program, the ingress bound)
../../Definitions/NetworkCertifier/AdviceString.lean  trusted: the per-window caps on the advice string's charge (WithinCaps),
                               the compiler and the compiled Program, and acceptance within the caps (AcceptedWithin,
                               AccSeqsWithin)
../../Definitions/NetworkCertifier/Assumptions/AdviceString.lean  trusted: the charge's named assumption (BitsCount) and the
                               compiler's two (CompilerDeterministic, CompilerPrecommitted)
../../Definitions/NetworkCertifier/Guarantees/AdviceString.lean   trusted: the leak bound, the advice string's charge in the accepted-runs bound
../../Definitions/NetworkCertifier/AdviceV1.lean  trusted: advice v1's charge, `ChargeCount_v1{N}`'s reference one for one: the
                               counting arithmetic coder (Tally, countStep), the book's KT estimate at min(run, 2),
                               a window's charge (window) and a run's (charges), the streams no window refuses (Streams)
../../Definitions/NetworkCertifier/Guarantees/AdviceV1Kraft.lean  trusted: advice v1's charge counts (Kraft), and the leak bound with it
../../Definitions/NetworkCertifier/Guarantees/AdviceV1.lean  trusted: the leak bound through the wire with advice v1's charge
../../Definitions/NetworkCertifier/WindowCharge.lean  trusted: advice v1's whole window charged, `ChargeCount_v2{N, H, O}`'s
                               reference one for one: the window's symbols in `code_v1`'s order (syms: n_arrived, the
                               decisions, n_hits, the hits, n_resumes, n_order, the order entries, n_lag), each integer
                               a bin and a uniform offset (binned, uniform), counted in one segment (window), the carry
                               the decisions' and the last order entry's position and rank, a run's charges (charges)
../../Definitions/NetworkCertifier/Guarantees/WindowChargeKraft.lean  trusted: the whole-window charge counts (Kraft), and the
                               leak bound with it
../../Definitions/NetworkCertifier/Guarantees/WindowCharge.lean  trusted: the leak bound through the wire with the whole-window
                               charge
../../Definitions/NetworkCertifier/Warden.lean  trusted: the warden (design v3): its cells (CellView), its public schedule
                               (WardenSchedule), what reaches it from the server (Arrivals), its code (wardenLane,
                               wardenCode), the fail cell (failCell) and the client's view (clientView)
../../Definitions/NetworkCertifier/Tap.lean  trusted: the tap, the warden and the certifier as one auditor device on a
                               server's uplink: its record format (TapRecord: on a schedule it holds, every cell its
                               code sends reads back from its record) and its one code run giving the cells and the
                               record (tapCode)
../../Definitions/NetworkCertifier/RecordRows.lean  trusted: the network certifier's record as the Python one writes it
                               (RowsFormat): a row a bucket of frames that name their connections, in ordinal order
                               (bucketFrames), the lanes read back from them (lanesOfRows), and the schedules it holds (Fits)
../../Definitions/NetworkCertifier/Guarantees/Warden.lean  trusted: the warden's channel bound (the client's view factors through
                               the fail cells, its count, and the whole run's bound with the advice string and content)
../../Definitions/NetworkCertifier/SampledCharge.lean  trusted: what the faulty proof blocks add under a sampled audit: their
                               committed bits (faultyBits), the possible faulty sets at δ (faultySets), the
                               sequences accepted with one (AcceptedFaulty, AccSeqsSampled), and what an accepted
                               audit gives that count (AdviceIn, freeInputs, AsReg, RLeak)
../../Definitions/NetworkCertifier/Guarantees/SampledCharge.lean  trusted: the leak bound under a sampled audit, and what
                               an accepted one establishes (SampledLeakAccepted)
../../Definitions/NetworkCertifier/TapRows.lean  trusted: the network certifier's rows: its registration at the
                               transfer gates (tapVal, TapOpens), the leak relation at them (RLeakTap) and the
                               collision finder against its openings (Btap)
../../Definitions/NetworkCertifier/Period.lean  trusted: a monitoring period of sampled audits: a statement registered
                               as a window's (SameRegistration) and a window's good event (WindowGood)
../../Definitions/NetworkCertifier/Guarantees/Period.lean  trusted: the leak through the network certifier over a
                               monitoring period of sampled audits (PeriodLeakAccepted)
../../Definitions/NetworkCertifier/Guarantees/Horizon.lean  trusted: the wrong units over a horizon of sampled audits on
                               independent live coins (HorizonWrongUnitsLe)
../../Definitions/NetworkCertifier/HorizonLeak.lean  trusted: a window's leak relation at one faulty set (LeakAt) and a
                               monitoring period's good event (HorizonGood)
../../Definitions/NetworkCertifier/Guarantees/HorizonLeak.lean  trusted: the leak through the network certifier over a
                               monitoring period of sampled audits, conditional on two per-window inputs
                               (HorizonLeakAccepted)
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
AdviceString.lean  the leak bound (the advice string's charge, through the wire, per window) and the two
                                         encodings whose length meets `BitsCount`
AdviceV1.lean   advice v1's charge counts: a coder state's weight (width over 2^bits) is kept by a
                                         scaling and split by a symbol, so the weights 2^(−charge) sum to at most 1;
                                         `BitsCount` for advice v1 (bitsCount), and the leak bounds with it; with
                                         counts, the decisions' weight times the counts' under any Kraft count code
                                         (bitsCount_counted, kraft_of_prefixFree), the retired `…Counted` bounds
AdviceV1Counted.lean  advice v1's charge with a separate count code (CountCode, lengths meeting Kraft): a
                                         window's charge (windowCounted) and a run's, read only by the retired bounds
WindowCharge.lean  the whole-window charge counts: symbol sequences that pairwise split (after a common
                                         prefix, one model's disjoint outcomes) weigh at most the first state
                                         (sum_owt_codeAll_le), and two windows' sequences from one carry split
                                         (syms_split); `BitsCount` for it (bitsCount), and the leak bounds with it
Warden.lean     the warden's channel bound: its lane is the client view of its first late cell
                                         (wardenLane_eq), the count, the whole run's bound
SampledCharge.lean  the leak bound under a sampled audit: by the layout's causality, holding each faulty
                                         committed gate at one bit leaves every step's egress as it was
SampledLeak.lean  the leak bound under an accepted sampled audit: its relation (RAcc) gives the leak
                                         relation (rLeak_of_rAcc), composed with SampledAuditAccepted as a
                                         hypothesis (sampledLeak; sampledLeak_reg at the registered inputs pinned)
Tap.lean        what the tap's certifier saw is what its warden sent: under trusted-host, on a schedule its
                                         record format holds, its cells read back from the record its same run writes
                                         (tap_cells_read), and each data cell its lane sends before the fail cell is
                                         in it (tap_data_read)
RecordRows.lean  the rows format reads back the code's cells on every schedule that fits it (read_record), so it
                                         is a record format (tapRecord); #guards hold it to the Python network
                                         certifier's egress records
TapRows.lean    the leak bound at the network certifier's rows: the sampling relation at a trace that agrees
                                         with its openings gives the leak relation at them (rLeakTap_of_r, through the
                                         trace filled in with its values), and a disagreement is a collision
                                         (tapColl_of_disagree); sampledLeak_tap
Period.lean     the leak through the network certifier over a monitoring period: each good window's
                                         registered observations are among AccSeqsSampled at its registration
                                         (mem_accSeqsSampled_of_windowGood), and the secrets the network certifier's
                                         cells decode to, read back from those observations and counted over the
                                         windows (encard_periodGood_le, periodLeak)
PeriodSum.lean  the period sum: per-window leak bounds `Pr[acc w ∧ b w < moved w] ≤ ε w` summed over a
                                         period's windows by a union bound (Pr_exists_mem_le_sum, periodLeak_le_sum)
Horizon.lean    the wrong units over a horizon: a uniform draw misses `f` units with probability at most
                                         `(1 - k/n)^f` (pr_disjoint_le), the escapes multiply over windows on
                                         independent coins (pr_adaptive_le), and with each window's soundness error
                                         added, the horizon's bound (horizonWrongUnits)
HorizonLeak.lean  the leak through the network certifier over a monitoring period: the escapes multiply with
                                         each window's error added (pr_adaptive_add_le, the monitoring-period step
                                         Periods.prCoin_forall_prod_le at bounds capped at 1), the families of faulty sets
                                         under `Fmax` units (exists_famOf_eq), and the secrets the client's cells
                                         decode to at each draw, read back from the certifier's record
                                         (encard_horizonGood_le, horizonLeak)
Sanity.lean     a satisfiability witness for every named hypothesis, the joint witnesses, and the
                                         toy of the September draft
Witnesses.lean  each acceptance guarantee's two lemmas: `G_honest`, its hypotheses at the deployed
                                         record, and `G_nontrivial`, its conclusion false at a verifier
                                         that accepts everything
Code.lean       the certifier program (`certifier_program`, `NetworkCertifier`), which the difftests run, equal to the
                               definitions above: six theorems, not guarantees (below)
Accepted.lean   an accepted run of the network certificate check is a run of the constant-rate Program
                               (`security_properties`' `AcceptedRunsAreProgramRuns`, through Code.lean), and
                               Theorem 3's count of what accepted runs can carry (`EncardAcceptedObservationsLe`),
                               each unless the run yields a SHA-512 collision or an LMS forgery
../../lean-audit.json          the guarantee records (`security`'s lock) for tools/verity/lean/check.py
~~~

~~~bash
lake exe cache get                                          # Mathlib's prebuilt oleans; never build Mathlib from source
lake build                                                  # NetTiming (the default target)
python3 ../../../tools/verity/lean/check.py .                      # Verity's Lean check
# the difftest is the certifier program's: see its package, ../../../verity/core/protocols/certifier/lean/
~~~

**Trusted without proof:** the statements in `../../Definitions/NetworkCertifier/`, `../../Definitions/NetworkCertifier/Assumptions.lean`,
`../../Definitions/NetworkCertifier/Guarantees.lean` and their `AdviceString`, `AdviceV1` and `Warden` modules under `../../Definitions/NetworkCertifier/Assumptions/` and
`../../Definitions/NetworkCertifier/Guarantees/`; Lean's kernel; the Mathlib pin. Everything else is checked by machine. The docstrings'
`docs/network-transparency/timing-channel.md` is the covert-capacity note, which is kept outside this repository.

**The audit.** `tools/verity/lean/check.py` audits every declaration under `NetTiming`: allowed axioms (only `propext`,
`Classical.choice` and `Quot.sound`), named assumptions, escapes, orphan files, the import layers (`NetTiming.Protocol`,
`NetTiming.Assumptions` and `NetTiming.Guarantees` import only one another and Mathlib, never a proof), the dependency
record, a kernel replay, and the guarantees' records, whose meaning it follows through those three (`meaning`) and the
certifier program's modules. The difftests (`runs` in the certifier program's own `lean-audit.json`) check that on 10,000
random instances the program gives what the Python audit gives, on 4,000 that its advice v1 charge gives what
`arith_encode.reference_charge` gives, and on 4,000 that its whole-window charge gives what
`window_charge.reference_window` gives; `Code.lean` proves the program equal to these definitions.
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

The lock (`lean-audit.json`'s `guarantees`) holds the 22 statements of `Guarantees.lean`, the leak bound's three
(`Guarantees/AdviceString.lean`), advice v1's four for decision-only windows
(`../../Definitions/NetworkCertifier/Guarantees/AdviceV1Kraft.lean` and `Guarantees/AdviceV1.lean`), the whole window's
four (`../../Definitions/NetworkCertifier/Guarantees/WindowChargeKraft.lean` and `Guarantees/WindowCharge.lean`), the warden's three (`Guarantees/Warden.lean`), the sampled audit's charge
(`../../Definitions/NetworkCertifier/Guarantees/SampledCharge.lean`), the monitoring period's leak through the network certifier
(`Guarantees/Period.lean`), the horizon's wrong units (`../../Definitions/NetworkCertifier/Guarantees/Horizon.lean`)
and the leak over a monitoring period (`../../Definitions/NetworkCertifier/Guarantees/HorizonLeak.lean`), each listed as
`NetTiming.SecurityProofs.G` and proved by the lemma of the same statement whose name it spells in UpperCamelCase (the
last two by `horizonWrongUnits` and `horizonLeak`). All 40 are under `shape.frozen` as
`frozen (no-function)`: each is stated over the model (an arbitrary `Interaction`, an abstract `Program`'s `AccSeqs`,
`expectedRow`, or the grids themselves; the leak bound also over an abstract charge and compiler; advice v1's over its
charge as `Definitions` states it, the mirror of `ChargeCount_v1{N}`'s or `ChargeCount_v2{N, H, O}`'s Python reference,
and an abstract compiler; the
warden's over the warden's code as `Definitions` states it, which no compiled program implements yet; the sampled charge
over the sequences the circuit's Program accepts with a possible faulty set, and the period's over both), not over the network check's code, and an
accepted run reaches the first 25 and advice v1's six leak bounds through `AcceptedRunsAreProgramRuns`. Four are the
theorems whose bounds the calibration reports (`verity.core.protocols.certifier.calibration`, `PROTOCOL.md`'s
"Capacity"):

- egress, for the shipped schedule (`NetTiming.SecurityProofs.Schedule`, `NetTiming.SecurityProofs.PerWindow`):
  `EncardDecodableLeConstantRate` and `EncardDecodableLeConstantRatePerWindow`;
- ingress, the K charge (`NetTiming.SecurityProofs.Ingress`, `NetTiming.SecurityProofs.PerWindow`):
  `EncardIngressObsDecodableLe` and `EncardIngressObsDecodableLePerWindow`.

The other 37 are the rest of the Capacity table's bounds and the schedule's causality rule:

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
- **The leak bound** (`NetTiming.SecurityProofs.AdviceString`): `EncardAccSeqsWithinLeConstantRate`,
  `EncardDecodableWithinLeCompiled`, `EncardDecodableWithinLeCompiledPerWindow`.
- **Advice v1's charge** (`NetTiming.SecurityProofs.AdviceV1`): `EncardChargedLe`, at most `2^c` decision streams
  charged `c` bits or fewer (Kraft, under any registered book), and the leak bound's three with advice v1's charge and
  no assumption on it, `EncardAccSeqsWithinLeConstantRateV1`, `EncardDecodableWithinLeCompiledV1`,
  `EncardDecodableWithinLeCompiledPerWindowV1`. That `ChargeCount_v1`'s circuit computes the Lean charge is
  circuit-check's (the circuit against `arith_encode.py`'s reference), not proved here. The same four for windows that
  send nonzero counts under a separate count code (`…Counted`, `AdviceV1Counted.lean`) are retired to `Lemmas.lean`:
  advice v1 codes its counts inside the arithmetic coder, which the next four count.
- **Advice v1's whole-window charge** (`NetTiming.SecurityProofs.WindowCharge`): `EncardChargedLeWhole`, at most `2^c`
  advice values of `W` windows (each window's decisions, hits and order entries) charged `c` bits or fewer, with every
  count coded inside the coder as `code_v1` codes it (Kraft, under any registered book), and the leak bound's three
  with it, `EncardAccSeqsWithinLeConstantRateV1Whole`, `EncardDecodableWithinLeCompiledV1Whole`,
  `EncardDecodableWithinLeCompiledPerWindowV1Whole`. That `ChargeCount_v2`'s circuit computes the Lean charge is
  circuit-check's (the circuit against `window_charge.py`'s `reference_window`), not proved here.
- **The warden's channel bound** (`NetTiming.SecurityProofs.Warden`): `WardenViewEqClientView`, `EncardWardenViewsLe`,
  `EncardWardenDecodableLe`, under `trusted-host` (`Verity.Assumptions.TrustedHost`).
- **The sampled audit implies `Accepted`** (`SampledAuditAccepted.lean`): `SampledAuditAccepted`, `KS`'s bound at
  `RAcc δ` on the statements whose layout fits the program: no faulty proof block is drawn `all`, and the advice holding the
  faulty proof blocks' committed gates at the trace's values makes `Accepted` every run the trace carries.
  `SampledAuditInstance.lean` discharges its `KS` hypothesis from `Claims.Sampling.sampling_assembled_of`'s `KS`
  conjunct: `sampledAuditAccepted_of` over any Flock record, `sampledAuditAccepted_zk` and `sampledAuditAccepted_plain`
  at the masked and plain records.
- **The leak bound under a sampled audit** (`SampledCharge.lean`):
  `EncardAccSeqsSampledLe`, the count of the sequences accepted with a possible faulty set: the inputs other than the
  ingress, the faulty proof blocks' committed bits and the choice of faulty set. `SampledLeakAccepted` is what an accepted
  sampled audit establishes: with `SampledAuditAccepted` as its hypothesis and `Claims.Sampling.KS` at the deployment,
  the carried sequences of length `k` are among those the count bounds except with probability `δ + ε_ks + ε_bind`
  (`RLeak`), and the count holds with `c` the number of free inputs not in the statement's registered inputs `reg`
  (`freeInputs`, at `AsReg`), at whatever values the trace holds at `reg`. It is `SampledLeak.lean`'s
  `sampledLeak_reg`. `RAcc` pins the inputs to the trace, so the values at `reg` are the trace's; that the
  registration's commitment binds them is a separate statement, not yet proved. Both `SampledAuditAccepted` and
  `Claims.Sampling.KS` are named assumptions in the lock.
- **The leak through the network certifier over a monitoring period** (`Period.lean`): `PeriodLeakAccepted` composes
  the leak bound at the network certifier's rows (`TapRows.lean`'s `sampledLeak_tap`) with the read-back of its
  record, so the bound's content term comes from the audits: over a monitoring period's windows, each registered as a
  statement, where the network certifier registers its observations at the transfer gates (`TapOpens`, its openings
  under the statement's roots), the secrets of a covert sender whose every window meets its good event (`WindowGood`)
  number at most `#Ss · ∏_c (cap_c + 1) · ∏_i 2^#(freeInputs_i \ reg_i) · #faultySets_i(δ) · 2^β_i · Dj^k_i`, and each
  window's audit fails the leak relation at the network certifier's rows (`RLeakTap`) with probability at most
  `δ + ε_ks + ε_bind + Pr[B_tap]`. Every unit outside the faulty set is correct, so the extracted trace opens every
  transfer gate there, at the network certifier's value unless the two openings collide (`B_tap`, under
  `CommitmentScheme.Binds`); filled in with the network certifier's values, it carries its records, and the faulty units'
  gates are the advice the count already charges, so the count is unchanged. The network certifier is one device on
  the server's uplink, and it functions as an active warden (`Definitions/NetworkCertifier/Tap.lean`; Daniel, 25 Oct):
  one run of its code (`tapCode`, under one `trusted-host`) gives the cells the client receives and the observations
  the audits read, its record of those cells, so on every schedule its record format holds (`TapRecord.fits`) the
  cells read back from the observations (`tap_cells_read`, a lemma, not a premise). The format the Python network
  certifier writes, a row a bucket of frames naming their connections (`RowsFormat`,
  `Definitions/NetworkCertifier/RecordRows.lean`), is one (`RowsFormat.tapRecord`), on the schedules whose cells fall
  in the monitoring period's buckets, at most `S` a bucket, and its `#guard`s hold it to the Python records. The record
  holds the fail cells and padding too, so the proof counts the cells through the
  observations and needs no fail-cell charge: `∏_c (cap_c + 1)` is room. Its named assumptions are
  `Claims.Sampling.KS`, `CommitmentScheme.Binds`, `Claims.Flock.ListWithin` and `trusted-host`. The union over the monitoring
  period's live coins and the registration's binding of the prover's values at `reg` are not yet stated.
  `Claims.Sampling.KS` stands in for proofs' live bound (`Flock.Guarantees.SampledKSLive`), which replaces it once
  locked, with live forms of `SampledAuditAccepted` and `SampledLeakAccepted`.
- **The wrong units over a horizon** (`Horizon.lean`): `HorizonWrongUnitsLe` is the bound over many periods, cumulative
  rather than per window, and a fact of probability with no named assumption. Over `m` windows on independent live
  coins (the horizon's coins are the product `Fin m → Ω`), window `i` drawing a uniform `k_i`-subset of its `n_i` units,
  with the units wrong at its registration fixed by the earlier windows' coins and its audit accepting while its draw
  meets one with probability at most `ε_i`: for `q ≤ k_i / n_i` in every window, every window accepts while the horizon
  holds at least `F` wrong units with probability at most `(1 - q)^F + Σ_i ε_i`, whatever `m` is, and
  `(1 - q)^F ≤ δ` once `F ≥ ln(1/δ)/q`. The draw's law and each window's error are hypotheses, which A3 and
  `Claims.Sampling.KS` discharge where the bound is applied to the sampled verifier.
- **The leak through the network certifier over a monitoring period** (`HorizonLeak.lean`): `HorizonLeakAccepted` is
  the count `K_horizon`, conditional on two per-window inputs taken as plain hypotheses, at the faulty set `W_i` of the
  witness computed from window `i`'s registration (read from the earlier windows' coins): the conditional escape
  `hesc`, `Pr[acc_i | earlier coins] ≤ (1 − q)^|W_i| + e_i` with `q` the design's per-window detection probability
  (no draw's form: a uniform `kd_i`-of-`n_i` draw meets it at `q = kd_i/n_i`), and the leak relation at that set
  `hleak`, `Pr[acc_i ∧ ¬LeakAt W_i] ≤ e'_i`, read on the network certifier's record. From them every window accepts
  outside the monitoring period's good event (`HorizonGood`: the faulty sets hold under `Fmax ≥ ln(1/δ)/q` units and
  every window meets its leak relation) with probability at most `δ + Σ_i e_i + Σ_i e'_i`, and at each draw the secrets
  meeting it number at most
  `#Ss · ∏_c (cap_c + 1) · (N + 1)^Fmax · 2^(b·Fmax) · ∏_i 2^#(freeInputs_i \ reg_i) · Dj^k_i`, `N = Σ_i n_i`: the
  faulty units cost `(ln(1/δ)/q + 1) · (b + log2 (N + 1))` bits once over the monitoring period, `b = 16` by a proof
  unit's output limit. The network certifier is one device, as in the period's bound: on every schedule its record
  format holds, the cells the client receives read back from its record (`tap_cells_read`), so its one named assumption is `trusted-host`. Over live audits the two
  inputs are `Flock.Guarantees.SampledKSLive` at each window's history and the live `SampledLeakAccepted` (P2),
  neither discharged, so it is not the deployed system's bound over a monitoring period until they are.

**The program's agreements, proved and audited, not guarantees.** Six theorems tie the difftests to these definitions
(`NetTiming.SecurityProofs.Code`): the difftests hold the Python audit and advice v1's charge references
to the certifier program (`verity/core/protocols/certifier/lean/`, Lean core only, `NetworkCertifier`), and
these say the program computes what the definitions above define. Each is stated in `Code.lean` over both sides'
definitions, relating them by their fields, since `security` requires nothing but Mathlib
(`tests/test_lean_packages.py`) and so `Definitions/` cannot name the program:

- `CodeExpectedRow`: `NetworkCertifier.expectedRow` is `expectedRow` for clock syncs with the same δ and ρ and frames with the same
  payloads, anchors and logical indices (the difftest's `rows`, `diff` and `mismatch`);
- `CodeAcceptedConstantRate`: `NetworkCertifier.Accepted` of `NetworkCertifier.constantRate` is `Accepted` of `constantRate` for observations
  with the same rows and complete statuses, operators giving the same ingress on them, related frames and advice naming
  the same values (`accepted`);
- `CodeStatusComplete`: `NetworkCertifier.StatusComplete` is `StatusComplete` for observations with the same complete statuses
  (`complete`);
- `CodeSyncSet`: every sync set of the program has a reference sync set with the same values and conversely (`advice`);
- `CodeChargesV1`: `NetworkCertifier.AdviceV1.charges` of the windows' decision streams is `AdviceV1.charges` for books
  with the same fields (the charge difftest's `charges`), the charge `BitsCount` for advice v1 is proved of;
- `CodeChargesWhole`: `NetworkCertifier.WindowCharge.charges` of the windows is `WindowCharge.charges` for books with
  the same tables and priors and windows with the same decisions, hits and order entries (the window difftest's
  `charges`), the charge `EncardChargedLeWhole` is proved of.

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
`n ≤ W·T`. All 38 statements (25 guarantees, 13 lemmas) and 16 witnesses are proved. The reference
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
5. **The integrity term** (faulty proof blocks) is not modelled: this is the exact binding.
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
