# Network certifier Lean package (`network-timing`)

Definitions, guarantees and proofs for the bucketed certifier's timing channel: how much a covert sender inside a
spatial unit X can carry across its boundary ∂X through **when** frames leave, once the declared Program fixes their
content and order. Conventions follow POUS's package (`verity/protocols/pous/lean/`): the same toolchain and Mathlib pin,
`autoImplicit` off, no `axiom` (an assumption is a named `Prop` hypothesis), and a check file.

~~~text
../../Definitions/NetworkCertifier/Grid.lean   trusted: rows, grids, occupancy, left-packing, fill, and the FIFO certifier (queue, released, fifoGrid)
../../Definitions/NetworkCertifier/Run.lean    trusted: statuses, observations, the certifier, the interaction (trace, obsRun, wireRun), the
    Program, acceptance (Accepted, AcceptedUpTo) and accepted observation sequences (AccSeqs)
../../Definitions/NetworkCertifier/Ingress.lean  trusted: the interaction read inward, a link's frames with the timing erased (rowFrames,
    linkFrames), and acceptance as it bears on ingress (StatusComplete)
../../Definitions/NetworkCertifier/Schedule.lean trusted: the Program's schedule over send ticks (LogicalFrame, handingFrom,
    enqueued, expectedRow, one FIFO per execution block) and the constant-rate Program it
    gives (constantRate)
../../Definitions/NetworkCertifier/Assumptions.lean     trusted: the named hypotheses W2–W6 (W4's allocation among them)
    and J
../../Definitions/NetworkCertifier/Guarantees.lean      trusted: each guarantee as a `def G : Prop` (Proposition 2, Theorem 3, Theorem 3 for
    the constant-rate Program, the ingress bound)
../../Definitions/NetworkCertifier/AdviceV1.lean  trusted: advice v1's charge, `ChargeCount_v1{N}`'s reference one for one: the
    counting arithmetic coder (Tally, countStep), the book's KT estimate at min(run, 2),
    a window's charge (window) and a run's (charges), the streams no window refuses (Streams)
../../Definitions/NetworkCertifier/WindowCharge.lean  trusted: advice v1's whole window charged, `ChargeCount_v2{N, H, O, L}`'s
    reference one for one: the window's symbols in `code_v1`'s order (syms: n_arrived, the
    decisions, n_hits, the hits, n_resumes, n_order, the order entries, n_lag), each integer
    a bin and a uniform offset (binned, uniform), counted in one segment, then the length
    terms' tail (tailBits) (window), the carry the decisions' and the last order entry's
    position and rank, a run's charges (charges)
../../Definitions/NetworkCertifier/Warden.lean  trusted: the warden (design v3): its cells (CellView), its public schedule
    (CertifierSchedule), what reaches it from the server (Arrivals), its code (certifierLane,
    certifierCode), the fail cell (failCell) and the client's view (clientView)
../../Definitions/NetworkCertifier/Tap.lean  trusted: the tap, the warden and the certifier as one auditor device on a
    server's uplink: its record format (RecordFormat: on a schedule it holds, every cell its
    code sends reads back from its record) and its one code run giving the cells and the
    record (recordingCode)
../../Definitions/NetworkCertifier/RecordRows.lean  trusted: the network certifier's record as the Python one writes it
    (RowsFormat): a row a bucket of frames that name their connections, in ordinal order
    (bucketFrames), the lanes read back from them (lanesOfRows), and the schedules it holds (Fits)
../../Definitions/NetworkCertifier/SampledCharge.lean  trusted: what the faulty proof blocks add under a sampled audit: their
    committed bits (faultyBits), the possible faulty sets at δ (faultySets), the
    sequences accepted with one (AcceptedFaulty, AccSeqsSampled), and what an accepted
    audit gives that count (AdviceIn, freeInputs, AsReg, RLeak); the runs before a
    run, which a period's operator reads (runsBefore)
../../Definitions/NetworkCertifier/Guarantees/SampledCharge.lean  trusted: the leak bound under a sampled audit, and what
    an accepted one establishes (SampledLeakAccepted)
../../Definitions/NetworkCertifier/TapRows.lean  trusted: the network certifier's rows: its registration at the
    transfer gates (certifierVal, CertifierRegisters, fillTrace), the leak relation at them
    (RLeakCertifier) and the collision finder against its openings (Btap)
../../Definitions/NetworkCertifier/Period.lean  trusted: a monitoring period of sampled audits: a statement registered
    as a window's (SameRegistration) and a window's good event (WindowGood)
../../Definitions/NetworkCertifier/Guarantees/Period.lean  trusted: the leak through the network certifier over a
    monitoring period of sampled audits (PeriodLeakAccepted)
../../Definitions/NetworkCertifier/Guarantees/Pooled.lean  trusted: the incorrect proof blocks over a monitoring period
    of sampled audits on independent live coins (PooledIncorrectBlocksLe)
../../Definitions/NetworkCertifier/PooledLeak.lean  trusted: a window's leak relation at one faulty set (LeakAt) and a
    monitoring period's good event (PooledGood)
../../Definitions/NetworkCertifier/Guarantees/PooledLeak.lean  trusted: the leak through the network certifier over a
    monitoring period of sampled audits, conditional on two per-window inputs
    (PooledLeakAccepted)
../../Definitions/NetworkCertifier/Checked.lean  trusted: the sequences accepted with a run a check passes
    (AcceptedChecked, AccSeqsChecked, LeakAtChecked, RLeakChecked), a check that pins its
    inputs (CheckedUnique) and misses at most εchk on given runs (CheckedWitness, traceRun),
    and the monitoring
    period's good event and named inputs (PooledGoodChecked, PeriodChannel, PeriodAdaptive,
    PeriodEscapes, PeriodLeaksChecked)
../../Definitions/NetworkCertifier/Guarantees/Checked.lean  trusted: the leak bound with checked inputs
    (SampledLeakChecked)
../../Definitions/NetworkCertifier/Carried.lean  trusted: runs chained through the pool's root: the record rule over a
    period's runs (poolOpen, PoolChained), a check that pins the closing row
    (CheckedClosing), the sequences accepted from an opened row (AcceptedCarried,
    LeakAtCarried), accepted runs opening the roots the record rule names (CarriedRoots),
    the restarts serving picks among registered patterns (ResetsIn),
    and the period's good event and named input (PooledGoodCarried, PeriodLeaksCarried,
    PoolColl); one run: the sequences accepted from an opened row (AccSeqsCarried) and
    the leak relation at the rows its trace holds (OpensAtInputs, RLeakCarried)
../../Definitions/NetworkCertifier/Guarantees/Carried.lean  trusted: the leak bounds over a carried pool
    (SampledLeakCarried, PooledLeakCarried)
../../Definitions/NetworkCertifier/PeriodConditions.lean  trusted: a period statement's conditions (WindowsPlaced,
    RecordFormat.ReadsBy, PeriodChannelReg) and the windows' count (windowsCount)
../../Definitions/NetworkCertifier/Assumptions/RegisteredInputs.lean  trusted: RegisteredInputsHeld, an open premise
../../Definitions/NetworkCertifier/PeriodChecked.lean  trusted: the run at the network certifier's rows
    (certifierRun) and the leak relation there (RLeakCertifierChecked)
../../Definitions/NetworkCertifier/Guarantees/PeriodChecked.lean  trusted: WindowGoodChecked and PeriodLeakChecked,
    stated and not proved
../../Definitions/NetworkCertifier/Witness.lean  trusted: the instances the honest and nontrivial lemmas are
    stated at, each built by hand (trivLay, notLay, certOps; the honest certifier at a
    Program, the reference schedule, the verifier and the sender that accept and leak
    everything, a sampled audit, the rows format of the fixture `fxQueue`)
Lemmas.lean     the statements that are only steps of the guarantees' proofs, Theorem 1(b) among them,
    each a `def G : Prop` under the same namespace (`Guarantees.G`)
../NetworkCertifier.lean  proved: `SecurityProofs.G : Guarantees.G` for each guarantee and lemma, from the modules below
Occupancy.lean  Proposition 2 (the occupancy counts, fill, and the FIFO certifier's tightness)
Counting.lean   Theorem 1(b) (the tree lemma on accepted runs, zero error, W6 on accepted runs)
Advice.lean     Theorem 3 (the 2^A cap, and at k bits per execution block)
Schedule.lean   Theorem 3 for the constant-rate Program (the advice cap of k bits per execution block,
    causality, the blocks opened, the composed statement) and its witness
Ingress.lean    the ingress bound (the occupancy of the ingress grid, read from inside X)
PerWindow.lean  the last two over the run's first `m` whole execution blocks, and the ingress bound
    over the first `m` periods of `T` buckets: the factor per block and per period
Warden.lean     the warden's code is the client view: its lane is the client view of its first late
    cell (certifierLane_eq), and so are its code's cells (certifierCode_eq), which
    Tap reads the network certifier's cells from
SampledCharge.lean  the leak bound under a sampled audit: by the layout's causality, holding each faulty
    committed gate at one bit leaves every step's egress as it was; a period
    whose window's set the windows before it fix (runsBefore_congr) numbers at
    most the product of the windows' bounds (encard_chain_le)
SampledLeak.lean  the leak bound under an accepted sampled audit: its relation (RAcc) gives the leak
    relation (rLeak_of_rAcc), composed with SampledAuditAccepted as a
    hypothesis (sampledLeak; sampledLeak_reg at the registered inputs pinned)
Tap.lean        what the tap's certifier saw is what its warden sent: under trusted-host, on a schedule its
    record format holds, its cells read back from the record its same run writes
    (certifier_cells_read), and each data cell its lane sends before the fail cell is
    in it (certifier_data_read)
RecordRows.lean  the rows format reads back the code's cells on every schedule that fits it (read_record), so it
    is a record format (recordFormat); #guards hold it to the Python network
    certifier's egress records
TapRows.lean    the leak bound at the network certifier's rows: the sampling relation at a trace that agrees
    with its openings gives the leak relation at them (rLeakCertifier_of_r, through the
    trace filled in with its values), and a disagreement is a collision
    (certifierColl_of_disagree); sampledLeak_certifier
Period.lean     the leak through the network certifier over a monitoring period: each good window's
    registered observations are among AccSeqsSampled at its registration
    (mem_accSeqsSampled_of_windowGood), and the secrets the network certifier's
    cells decode to, read back from those observations and counted over the
    windows over any sets of accepted sequences (encard_periodSeqs_le;
    encard_periodGood_le, periodLeak; periodLeak_fixed at operators fixed before
    the period)
PeriodSum.lean  the period sum: per-window leak bounds `Pr[acc w ∧ b w < moved w] ≤ ε w` summed over a
    period's windows by a union bound (Pr_exists_mem_le_sum, periodLeak_le_sum)
Pooled.lean     the incorrect proof blocks over a monitoring period: a uniform draw misses `f` blocks with
    probability at most `(1 - k/n)^f` (pr_disjoint_le), the escapes multiply over
    windows on independent coins (pr_adaptive_le), and with each window's
    soundness error added, the monitoring period's bound (pooledIncorrectBlocks)
PooledLeak.lean  the leak through the network certifier over a monitoring period: the escapes multiply with
    each window's error added (pr_adaptive_add_le, the monitoring-period step
    Periods.prCoin_forall_prod_le at bounds capped at 1), the families of faulty sets
    under `Fmax` units (exists_famOf_eq), and the secrets the client's cells
    decode to at each draw, read back from the certifier's record
    (encard_pooledGood_le, pooledLeak; pooledLeak_fixed at operators fixed
    before the period), each over any per-window event
    (encard_pooledFam_le, pr_pooled_le), the count also over any event on the
    whole period (encard_pooledJoint_le)
Checked.lean    the leak bound with checked inputs: a sequence accepted with a valid run is fixed by its
    inputs off the registered and checked ones, the held bits and its jitter
    (encard_acceptedChecked_le, through rows_full), and RAcc gives the leak
    relation with a valid run except where the check misses
    (ksLinOn_rLeakChecked, checkedWitness_traceRun); sampledLeakChecked
Carried.lean    the leak bounds over a carried pool: one run's relation at the rows its extracted trace
    holds (rLeakCarried_of, ksLinOn_rLeakCarried) and its count from a fixed
    opened row (encard_accSeqsCarried_le); by induction over the runs, a period's
    observations are fixed by each run's inputs off its registered, checked and
    opening gates, the held bits and its jitter, since each run opens the row the
    run before left and reads only the runs before it (runsBefore_congr,
    carried_step, encard_carried_le), and an opened row whose root matches but
    whose bits don't is a collision (rowSeg_bind); restarts serving picks
    from a registered set where every audit accepts multiply the count by its
    size (encard_pooledCarried_le); sampledLeakCarried, pooledLeakCarried; at
    the empty pool, the period's bound with checked inputs over windows that
    each hold a pool of their own (pooledLeakChecked_of_carried)
PoolRoots.lean  the opened rows' roots, a lemma: a run's check that holds its opened and left rows to the roots it
    registers (CheckedRoots) and its leak relation give them those roots
    (digests_of_leakAt), and the record rule registers at a run's opening the root
    of the row poolOpen names (digest_open_of_leakAt), so CarriedRoots holds
    (rootsOfChecked)
PeriodCheckedChecks.lean  PeriodLeakChecked held to the trace's run is false (freeL_traceRun_insufficient,
    periodLeakChecked_traceRun_fails, freeL_certifierRun_miss); its nontrivial lemma
Sanity.lean     a satisfiability witness for every named hypothesis, the joint witnesses, and the
    draft's toy
Witnesses.lean  each acceptance guarantee's two lemmas: `G_honest`, its hypotheses at the deployed
    record, and `G_nontrivial`, its conclusion false at a verifier
    that accepts everything
WitnessesChecked.lean  the nontrivial lemma of SampledLeakChecked; its honest lemma is
    SampledAuditAccepted_honest
WitnessesCarried.lean  SampledLeakCarried's honest lemma, at the toy code with a run that opens its row
    at its input gates, and the nontrivial lemmas of SampledLeakCarried and
    PooledLeakCarried; PooledLeakCarried's honest lemma waits on a deployed check
WitnessToys.lean  what holds at those instances, which the count and period witnesses conclude from
CountWitnesses.lean  the thirteen counts' pairs, seven egress and six ingress: the honest run is accepted at
    `fxRate` or at `sampProg`, or has every status complete, and each bound is
    false at the verifier that accepts everything (`acceptAll`)
PeriodWitnesses.lean  two more instances of PooledLeakAccepted: its schedule premises at `fxQueue`'s, and its bound
    false at one incorrect proof block
Code.lean       the certifier program (`certifier_program`, `NetworkCertifier`), which the difftests run, equal to the
    definitions above: five theorems, not guarantees (below)
Accepted.lean   an accepted run of the network certificate check is a run of the constant-rate Program
    (`security_properties`' `AcceptedRunsAreProgramRuns`, through Code.lean), and
    Theorem 3's count of what accepted runs can carry (`EncardAcceptedObservationsLe`),
    each unless the run yields a SHA-512 collision or an LMS forgery
Causal.lean     a later window's sync never changes an earlier window's grid (`ExpectedGridCausal`, from
    the program's hand ticks), and the count over the first `m` windows of the
    Program's runs (`EncardProgramRunsLeWindows`, with its honest and nontrivial
    lemmas) and of the check's accepted runs (encard_acceptedObservations_take_le)
../../lean-audit.json          the guarantee records (`security`'s lock) for tools/verity/lean/check.py
~~~

~~~bash
lake exe cache get                                          # Mathlib's prebuilt oleans; never build Mathlib from source
lake build                                                  # NetTiming (the default target)
python3 ../../../tools/verity/lean/check.py .                      # Verity's Lean check
# the difftest is the certifier program's: see its package, ../../../verity/protocols/certifier/device/program/lean/
~~~

**Trusted without proof:** the statements in `../../Definitions/NetworkCertifier/`,
`../../Definitions/NetworkCertifier/Assumptions.lean`, `../../Definitions/NetworkCertifier/Guarantees.lean` and its
modules under `../../Definitions/NetworkCertifier/Guarantees/`; Lean's kernel; the Mathlib pin. Everything else is
checked by machine. The docstrings' `docs/network-transparency/timing-channel.md` is the covert-capacity note, which is
kept outside this repository.

**The audit.** `tools/verity/lean/check.py` audits every declaration under `NetTiming`: allowed axioms (only `propext`,
`Classical.choice` and `Quot.sound`), named assumptions, escapes, orphan files, the import layers (`NetTiming.Protocol`,
`NetTiming.Assumptions` and `NetTiming.Guarantees` import only one another and Mathlib, never a proof), the dependency
record, a kernel replay, and the guarantees' records, whose meaning it follows through those three (`meaning`) and the
certifier program's modules. The difftests (`runs` in the certifier program's own `lean-audit.json`) check that on 10,000
random instances the program gives what the Python audit gives, on 4,000 that its advice v1 charge gives what
`arith_encode.reference_charge` gives, and on 4,000 that its whole-window charge gives what
`window_charge.reference_window` gives; `Code.lean` proves the program equal to these definitions.
When a changed record lands on `main`, a DM shows it (the team skill).

## The model

**The run.** A step is one bucket of the whole boundary ∂X. At each step the operator's move is the ingress, the sender's
move uses the secret, and the certifiers record the observation (the egress rows, one jitter class, and every boundary link's
status) and emit the wire. `Interaction.trace` builds the run by recursion; the operator reacts to everything it has
observed.

**Acceptance is separate from emission.** `Accepted P op k o a` holds when, at every step before `k`, the recorded rows are
the Program's rows for the ingress so far and the advice, and every status is complete. It reads only rows and
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

**The reference schedule** (`NetTiming.Protocol.Schedule`). `constantRate T g E frames` is the Program of one link's
monitoring period: `E` execution blocks, each of `g·T` buckets, and block `e`'s frames, with their content and send
ticks, `frames ins e (a e) ℓ`, read from the ingress and `e`'s own advice. A frame is handed at
`h(k) = max(send(k), h(k−1))` and released by `fifoGrid`, with the FIFO empty at each execution block's first bucket.
Its advice type is `Fin E → Adv`, the protocol's `Adv` the `|a|`-bit strings, so Theorem 3's `2^(E·k)` form holds for
it with no dataflow hypothesis (`Guarantees.EncardAccSeqsLeConstantRate`). The check is this Program as one execution
block of its `W` windows (`CodeAcceptedConstantRate`).

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

For the ingress bound the same `Prop`s are read inward: `Regenerates` is W2 for what enters X, `Packed` and
`Allocated` are W4 for the rows the ingress links release into X, and `JitterBounded` is J for their release times.

`NetTiming.SecurityProofs.Sanity` shows every hypothesis is satisfiable: one certifier meets W2–W6 and J together, a
Program with no timing advice accepts exactly one secret despite a reacting operator (`zP_accepts_one`), and
`ingress_witness` meets every hypothesis of `Guarantees.EncardIngressDecodableLe` at once, with a bound it meets exactly
(`ingress_witness_tight`).
`allocated_witness` meets them again with an allocation below the slot count (`r = 1 < S = 2`): its 8 messages meet
that bound exactly, below the 27 of `r = S` (`allocated_witness_tight`), so `Allocated` has content.

## The records

The lock (`lean-audit.json`'s `guarantees`) holds the 18 statements of `Guarantees.lean`, the sampled audit's charge
(`../../Definitions/NetworkCertifier/Guarantees/SampledCharge.lean`), the monitoring period's leak through the network certifier
(`Guarantees/Period.lean`), the monitoring period's incorrect proof blocks
(`../../Definitions/NetworkCertifier/Guarantees/Pooled.lean`) and the leak over a monitoring period
(`../../Definitions/NetworkCertifier/Guarantees/PooledLeak.lean`), each listed as
`NetTiming.SecurityProofs.G` and proved by the lemma of the same statement whose name it spells in UpperCamelCase (the
last two by `pooledIncorrectBlocks` and `pooledLeak`). All 28 are under `shape.frozen` as
`frozen (no-function)`: each is stated over the model (an arbitrary `Interaction`, an abstract `EgressPrediction`'s `AccSeqs`,
`expectedRow`, or the grids themselves; the sampled charge over the sequences the circuit's Program accepts with a
possible incorrect set; the period's also over the network certifier's code as `Definitions` states it, which no compiled
program implements yet), not over the network check's code, and an accepted run reaches the first 18 through
`AcceptedRunsAreProgramRuns`. Three are the
theorems whose bounds the calibration reports (`verity.protocols.certifier.verifier.calibration`, `PROTOCOL.md`'s
"Capacity"):

- egress over a monitoring period, for the reference schedule (`NetTiming.SecurityProofs.Schedule`):
  `EncardDecodableLeConstantRate`;
- ingress, the K charge (`NetTiming.SecurityProofs.Ingress`, `NetTiming.SecurityProofs.PerWindow`):
  `EncardIngressObsDecodableLe` and `EncardIngressObsDecodableLePerWindow`.

The calibration's egress bits per window (`clock_sync.bits_per_window`, held by `test_network_certifier_calibration.py`)
are the check's bound over its first `m` windows, `security_properties`' `EncardProgramRunsLeWindows` (`Causal.lean`),
not one of these: the check is the reference Program as one execution block of its `W` windows, where the per-block
forms give only the `W`-window total, `EncardAcceptedObservationsLe`'s, so the finer count comes from the program's
hand ticks, which a window's sync moves only at or after the window's start (`ExpectedGridCausal`). A window's bits
given the run before it have no Lean statement for the check: the queue and the hand ticks carry across windows.

The other 25 are the rest of the Capacity table's bounds and the schedule's causality rule:

- **Proposition 2** (`NetTiming.SecurityProofs.Occupancy`): `CardFree`, `CardLeftPacked`, `CardCountsEqLe`,
  `EncardFifoGridsLeOfAvailable`, `EncardFifoGrids`.
- **Theorem 3, `k` advice bits per execution block** (`NetTiming.SecurityProofs.Advice`,
  `NetTiming.SecurityProofs.Schedule`, `NetTiming.SecurityProofs.PerWindow`): `EncardAccSeqsLeAdviceBits` for any
  Program, and for the constant-rate Program `EncardAccSeqsLeConstantRate`, `ExpectedRowCausal`,
  `EncardAccSeqsLeConstantRateWindows`, `EncardAccSeqsLeConstantRatePerWindow` and, through the wire,
  `EncardDecodableLeConstantRatePerWindow`. Advice of any finite type is
  `EncardAccSeqsLeAdvice` (`Lemmas.lean`), which the check's count (`EncardAcceptedObservationsLe`) reads at its clock
  syncs.
- **The ingress bound** (`NetTiming.SecurityProofs.Ingress`, `NetTiming.SecurityProofs.PerWindow`):
  `EncardIngressDecodableLe`, `EncardIngressObsDecodableLeContents`, `EncardIngressDecodableLeContents`,
  `EncardIngressObsDecodableLeContentsPerWindow`.
- **The sampled audit implies `Accepted`** (`SampledAuditAccepted.lean`): `SampledAuditAccepted`, the bound of
  `Claims.SamplingGeneric.Compiled.KS` at `RAcc δ`, at any compiled sampled verifier, on the statements whose layout
  fits the program: no incorrect proof block is sampled `all`, and the advice holding the incorrect proof blocks'
  committed gates at the trace's values makes `Accepted` every run the trace carries. `SampledAuditInstance.lean`
  discharges its `Compiled.KS` hypothesis at an assembled sampling record from the compiler at Flock
  (`Claims.SamplingGeneric.flockKS`): `sampledAuditAccepted_of` over any Flock record, `sampledAuditAccepted_zk` and
  `sampledAuditAccepted_plain` at the masked and plain records.
- **The leak bound under a sampled audit** (`SampledCharge.lean`):
  `EncardAccSeqsSampledLe`, the count of the sequences accepted with a possible incorrect set: the inputs other than the
  ingress, the incorrect proof blocks' committed bits and the choice of incorrect set. `SampledLeakAccepted` is what an accepted
  sampled audit establishes: with `SampledAuditAccepted` as its hypothesis and `Compiled.KS` at the compiled verifier,
  the carried sequences of length `k` are among those the count bounds except with probability `δ + ε_ks + ε_bind`
  (`RLeak`), and the count holds with `c` the number of free inputs not in the statement's registered inputs `reg`
  (`freeInputs`, at `AsReg`), at whatever values the trace holds at `reg`. It is `SampledLeak.lean`'s
  `sampledLeak_reg`. `RAcc` pins the inputs to the trace, so the values at `reg` are the trace's; that the
  registration's commitment binds them is a separate statement, not yet proved. `Compiled.KS` is an open premise in
  the lock, and `SampledAuditAccepted` is proved.
- **The leak through the network certifier over a monitoring period** (`Period.lean`): `PeriodLeakAccepted` composes
  the leak bound at the network certifier's rows (`TapRows.lean`'s `sampledLeak_certifier`) with the read-back of its
  record, so the bound's content term comes from the audits: over a monitoring period's windows, each registered as a
  statement, where the network certifier registers its observations at the transfer gates (`CertifierRegisters`, its openings
  under the statement's roots), the secrets of a covert sender whose every window meets its good event (`WindowGood`)
  number at most `#Ss · ∏_c (cap_c + 1) · ∏_i 2^#(freeInputs_i \ reg_i) · #faultySets_i(δ) · 2^β_i · Dj^k_i`, and each
  window's audit fails the leak relation at the network certifier's rows (`RLeakCertifier`) with probability at most
  `δ + ε_ks + ε_bind + Pr[B_tap]`. The period's operator is fixed before the period and the secrets, a function of the
  observations: window `i`'s operator reads only the windows before `i`, closed by then (`runsBefore`), and its own
  window only through the observations before each step. What it reads the count already charged to the earlier
  windows, so the windows' bounds still multiply, by induction over the windows (`SampledCharge.lean`'s
  `encard_chain_le`); `PooledLeakAccepted` takes the same operator, and at operators that read no earlier window
  each is the statement with one operator a window (`periodLeak_fixed`, `pooledLeak_fixed`).
  Every proof block outside the incorrect set is correct, so the extracted trace opens every
  transfer gate there, at the network certifier's value unless the two openings collide (`B_tap`, under
  `CommitmentScheme.Binds`); filled in with the network certifier's values, it carries its records, and the incorrect proof blocks'
  gates are the advice the count already charges, so the count is unchanged. The network certifier is one device on
  the server's uplink, and it shapes what it forwards rather than only observing it (`Definitions/NetworkCertifier/Tap.lean`):
  one run of its code (`recordingCode`, under one `trusted-host`) gives the cells the client receives and the observations
  the audits read, its record of those cells, so on every schedule its record format holds (`RecordFormat.fits`) the
  cells read back from the observations (`certifier_cells_read`, a lemma, not a premise). The format the Python network
  certifier writes, a row a bucket of frames naming their connections (`RowsFormat`,
  `Definitions/NetworkCertifier/RecordRows.lean`), is one (`RowsFormat.recordFormat`), on the schedules whose cells fall
  in the monitoring period's buckets, at most `S` a bucket, and its `#guard`s hold it to the Python records. The record
  holds the fail cells and padding too, so the proof counts the cells through the
  observations and needs no fail-cell charge: `∏_c (cap_c + 1)` is room. Its premises are `Compiled.KS`,
  `CommitmentScheme.Binds` and `trusted-host`. The union over the monitoring period's live coins and the registration's
  binding of the prover's values at `reg` are not yet stated. `Compiled.KS` stands in for proofs' live bound
  (`Flock.Guarantees.SampledKSLive`), which replaces it once locked, with live forms of `SampledAuditAccepted` and
  `SampledLeakAccepted`.
- **The incorrect proof blocks over a monitoring period** (`Pooled.lean`): `PooledIncorrectBlocksLe` is the bound over
  many periods, cumulative rather than per window, and a fact of probability with no named assumption. Over `m` windows
  on independent live coins (their coins are the product `Fin m → Ω`), window `i` sampling a uniform
  `k_i`-subset of its `n_i` proof blocks, with the blocks incorrect at its registration fixed by the earlier windows'
  coins and its audit accepting while its sample meets one with probability at most `ε_i`: for `q ≤ k_i / n_i` in every
  window, every window accepts while the monitoring period holds at least `F` incorrect blocks with probability at most
  `(1 - q)^F + Σ_i ε_i`, whatever `m` is, and
  `(1 - q)^F ≤ δ` once `F ≥ ln(1/δ)/q`. The sampling design and each window's error are hypotheses, which A3 and
  `Compiled.KS` discharge where the bound is applied to the sampled verifier.
- **The leak through the network certifier over a monitoring period** (`PooledLeak.lean`): `PooledLeakAccepted` is
  the count `K_pooled`, conditional on two per-window inputs taken as plain hypotheses, at the incorrect set `W_i` of the
  witness computed from window `i`'s registration (read from the earlier windows' coins): the conditional miss bound
  `hesc`, `Pr[acc_i | earlier coins] ≤ (1 − q)^|W_i| + e_i` with `1 − q` the design's per-window miss probability
  (no sample's form: a uniform `kd_i`-of-`n_i` sample meets it at `q = kd_i/n_i`), and the leak relation at that set
  `hleak`, `Pr[acc_i ∧ ¬LeakAt W_i] ≤ e'_i`, read on the network certifier's record. From them every window accepts
  outside the monitoring period's good event (`PooledGood`: the incorrect sets hold under `Fmax ≥ ln(1/δ)/q` units and
  every window meets its leak relation) with probability at most `δ + Σ_i e_i + Σ_i e'_i`, and at each sample the secrets
  meeting it number at most
  `#Ss · ∏_c (cap_c + 1) · (N + 1)^Fmax · 2^(b·Fmax) · ∏_i 2^#(freeInputs_i \ reg_i) · Dj^k_i`, `N = Σ_i n_i`: the
  incorrect proof blocks cost `(ln(1/δ)/q + 1) · (b + log2 (N + 1))` bits once over the monitoring period, `b = 16` by a
  proof block's output limit. The network certifier is one device, as in the period's bound: on every schedule its
  record format holds, the cells the client receives read back from its record (`certifier_cells_read`), so its one named
  assumption is `trusted-host`. Over live audits the two inputs are `Flock.Guarantees.SampledKSLive` at each window's
  history and the live `SampledLeakAccepted` (P2), neither discharged, so it is not the deployed system's bound over a
  monitoring period until they are.
- **The leak bounds with checked inputs** (`Checked.lean`, `Carried.lean`): inputs a check pins cost no advice bits. A
  check `valid` over a run's values pins the inputs `chk` where a valid run's values there are a function of the
  ingress, the other inputs and the incorrect proof blocks' held values (`CheckedUnique`), and misses an invalid
  accepted run with probability at most `εchk` on the runs it is held to (`CheckedWitness`, a scope; `traceRun` below).
  - `SampledLeakChecked` is `SampledLeakAccepted` with the leak relation and the count taken over valid runs only, at
    `κ + εchk`: the count charges `freeInputs \ (reg ∪ chk)`. Its honest lemma is `SampledAuditAccepted_honest`, at
    its one stated premise `Compiled.KS`.
  - `SampledLeakCarried` is `SampledLeakChecked` for a run that opens a row at its opening gates (`OpensAtInputs`)
    and leaves one at its closing gates. Its count charges `freeInputs \ (reg ∪ chk ∪ prow)` at a fixed opened row.
  - `PooledLeakCarried` is `PooledLeakAccepted` with each run's leak relation so taken (`LeakAtCarried`), over runs
    that open the row the run before left (`PoolChained`). Its period inputs are named (`PeriodChannel`,
    `PeriodAdaptive`, `PeriodEscapes`, `PeriodLeaksCarried`), and the escape is at `(1 − q)^Fmax` plus a collision
    term per opening. At the empty pool it gives the bound over windows that each hold a pool of their own, at their
    leak relations with a valid run (`LeakAtChecked`, `PeriodLeaksChecked`; `pooledLeakChecked_of_carried`).
  - `CheckedUnique` and `CheckedWitness` are open premises of all three: a deployed check proved to pin its inputs and
    bound its miss discharges them. `SampledLeakCarried`'s honest lemma is its premises at the toy code, with a run
    that opens its row at its circuit's input gates (`SampledLeakCarried_honest`). `PooledLeakCarried`'s waits on the
    pool circuit, which pins its closing row (`CheckedClosing`), and on its period's inputs.
- **The period statement with checked inputs** (`Guarantees/PeriodChecked.lean`), stated and not proved:
  `PeriodLeakAccepted` with each window's checked inputs left out of the count and the client's view read from the
  records under a registered parameter (`RecordFormat.ReadsBy`). Its check is held to the run at the network
  certifier's rows (`certifierRun`): held to the trace's run it is false (`PeriodCheckedChecks.lean`).
  `RegisteredInputsHeld` is its open premise.

**The program's agreements, proved and audited, not guarantees.** Five theorems tie the difftests to these definitions
(`NetTiming.SecurityProofs.Code`): the difftests hold the Python audit and advice v1's charge references
to the certifier program (`verity/protocols/certifier/device/program/lean/`, Lean core only, `NetworkCertifier`), and
these say the program computes what the definitions above define. Each is stated in `Code.lean` over both sides'
definitions, relating them by their fields, since `security` requires nothing but Mathlib
(`tests/test_lean_packages.py`) and so `Definitions/` cannot name the program:

- `CodeExpectedRow`: `NetworkCertifier.expectedRow` over its `W` windows is `expectedRow` of one execution block of
  them, for frames with the same payloads sent at the program's hand ticks (`NetworkCertifier.handingBuckets`) under
  its syncs (the difftest's `rows`, `diff` and `mismatch`);
- `CodeAcceptedConstantRate`: `NetworkCertifier.Accepted` of `NetworkCertifier.constantRate` is `Accepted` of
  `constantRate` with one execution block of the `W` windows, for observations with the same rows and complete
  statuses, operators giving the same ingress on them, and frames with the same payloads sent at the program's hand
  ticks under the syncs the advice names (`accepted`);
- `CodeStatusComplete`: `NetworkCertifier.StatusComplete` is `StatusComplete` for observations with the same complete statuses
  (`complete`);
- `CodeChargesV1`: `NetworkCertifier.AdviceV1.charges` of the windows' decision streams is `AdviceV1.charges` for books
  with the same fields (the charge difftest's `charges`);
- `CodeChargesWhole`: `NetworkCertifier.WindowCharge.charges` of the windows is `WindowCharge.charges` for books with
  the same tables and priors and windows with the same decisions, hits and order entries (the window difftest's
  `charges`).

They aren't guarantees because `four_names` holds every `NetTiming` guarantee to `SecurityProofs.G : Guarantees.G`,
and a `Guarantees` module can't state them: it would have to import the program. They are built and checked like every
declaration, and no citation may call them guarantees.

**Proved and audited, not guarantees.** The 12 statements that are only steps of these proofs are lemmas, in
`Lemmas.lean` under their old names (`Guarantees.G`): nothing relies on them as proved (README "Glossary"), so they have
no record. Each is still `SecurityProofs.G : Guarantees.G`, built and checked like every declaration:

- **Proposition 2** (`NetTiming.SecurityProofs.Occupancy`): `FillInjective`, `ExistsFillOfLeftPacked`, `CardCountsLeLe`,
  `ReleasedSelf`, `SumReleasedLe`.
- **Theorem 1(b)** (`NetTiming.SecurityProofs.Counting`): `EncardLeProdOfBranching`, `EncardAcceptedRunsLe`,
  `EncardObsDecodableLe`, `EncardDecodableLe`, `NoFaultOfAccepted`.
- **Theorem 3** (`NetTiming.SecurityProofs.Advice`): `RecordedRowEqFill`, `EncardAccSeqsLeAdvice`.

So are the 10 witnesses, checked tests rather than guarantees, under their own names (`NetTiming.SecurityProofs.Sanity`,
and the last in `NetTiming.SecurityProofs.Schedule`):
`witnessCertifier_assumptions`, `witnessCertifier_jitter_varies`, `zP_accepts_one`, `zP_constOp_accepts_one`,
`ingress_witness`, `ingress_witness_tight`, `allocated_witness`, `allocated_witness_tight`, `toy_leftPacked`,
`constantRate_joint_witness`; and
`occ_fill`, `leftPacked_occ_fill` and `leftPacked_fifoGrid` (`NetTiming.SecurityProofs.Occupancy`). They use only the
three axioms, but they have no record in `lean-audit.json`, so no citation may call them guarantees. Listing one is an
`--update` like any other.

The statements were frozen and reviewed by a red team before their proofs, and a second round's verdict was GO. The
ingress bound's statements were frozen before their proofs as well; the review of the first two is on the pull request
that added them, and the review of the content versions is on the one stacked on it. The
constant-rate Program's five statements and the definitions they read were frozen on `sorry` stubs and recorded before
their proofs. Their statement review gave GO with one change: the four counts take
`n ≤ W·T`. All 38 statements (25 guarantees, 13 lemmas) and 16 witnesses are proved. The reference
implementation, `verity/protocols/certifier` (`verity.protocols.certifier.verifier`), mirrors 19 of the statements and the three
lemmas by name in its tests; its `PROTOCOL.md` lists them per test file. The witnesses have no Python counterpart.

## Modelling choices a reader of the statements checks

1. **Steps are buckets across all of ∂X.** The note states its bounds per execution phase of a link. The guarantees
   count steps, and Theorem 1(b)'s tree lemma is still stated per step. The per-window ones read a monitoring period
   of whole blocks, each with its parent's hypotheses at that length and no other. The reference schedule's bound is
   `(2^k · D^(g·T))^m` over any `m` execution blocks of `g·T` buckets from block `e₀`, for the runs that agree on
   everything observed before block `e₀` (`EncardAccSeqsLeConstantRatePerWindow`,
   `EncardDecodableLeConstantRatePerWindow`; at `e₀ = 0` the run's first `m` blocks). It holds whatever came before
   because each block's rows read only its own advice and its FIFO starts empty, so a temporal unit that starts later
   is bounded by its own blocks. The ingress bound is `(∏_ℓ (r ℓ + 1)^T · D^T)^m` per content assignment over the
   first `m` periods of `T` buckets, `n = m·T` (`EncardIngressObsDecodableLePerWindow`, and its `Contents` version);
   `Packed`, `Allocated` and `JitterBounded` bound every step's branching whatever came before.
2. **The Program reads ingress up to and including step t.** The note's W1 lag allows only ingress before t. This is
   more permissive, so the bounds cover more Programs.
3. **Theorem 3 counts all accepted sequences,** not only those a run produces, which makes it stronger and needs no
   interaction.
4. **Jitter** is a free per-step class below D. The operator reads it, since it reads the whole observation.
5. **The integrity term** (incorrect proof blocks) is not modelled: this is the exact binding.
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
    modelled. Over the first `m` whole windows each execution phase contributes `(r ℓ + 1)^T`
    (`EncardIngressObsDecodableLePerWindow`).
12. **No Program, no clock.** The bound reads no Program and no advice: the outside sender chooses when frames arrive,
    so no clock pins ingress timing. Jitter on the inward release adds `D^n` (J, `D = 1` when tested ideal). W3's frame
    length and the FIFO rule are not used; the FIFO certifier attains the bound (`EncardFifoGrids`, read for the ingress
    grid).

The constant-rate Program adds these:

13. **Send ticks, one FIFO per execution block.** `NetTiming.Protocol.Schedule` takes each frame's send tick as the
    Program gives it: a frame is handed at `max(send, last)`, so hand ticks only rise in canonical order, and the FIFO
    starts empty at each execution block's first bucket, a frame handed at or after its span's end never enqueued. The
    certifier program (`NetworkCertifier.Schedule`, `schedule.py`) still computes its hand ticks from a declared clock
    sync per window, and over its `W` windows it is one execution block (`CodeExpectedRow`). The audit checks `E`
    execution blocks, so the counts take `n ≤ E·g·T` (`hn`): every step they count is a step the audit checks.
14. **The frames are the Program's computation, left abstract.** `frames ins e v ℓ` gives link `ℓ`'s frames of
    execution block `e` in canonical order, with their content and send ticks, from the ingress and `e`'s advice `v`.
    Every guarantee holds for every such function, so it covers any way the Program fixes send ticks. Advice that moves
    a send tick is counted in its block's `k` bits.
15. **One clock per spatial unit.** Every egress link reads the same window's sync (decision A's default), and the
    audit enforces it (`Audit.syncs`). A clock per link is a different advice type, covered by
    `EncardAccSeqsLeAdvice` directly.
16. **The allocation is the slot count `S`** on every egress link, as in Python, where a row has `allocation` slots.
17. **In-set advice.** The check's advice ranges over `SyncValue`, the values of Σ_sync: a run whose declared sync is
    outside the set is an `advice` violation and rejected, so its advice is never counted. That an earlier window's
    grid doesn't depend on a later window's sync, in the set or not, is `security_properties`' `ExpectedGridCausal`,
    over the check's `expectedGrid` and any clock-sync values (`Causal.lean`); the rest of a window's verdict reads only
    that window's records and syncs.
