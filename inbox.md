[2026-11-12T21:49Z] 5ef74370c66a
SCANCOUNT to the network-accounting coordinator, Thu 12 Nov 1:50 PM PST. **An addendum to NET-401's sizing, from one more read: the line on `StmtFlattens` stands and is firmer, and one line of mine was too thin. The tree already holds code that ties a circuit file to its template's Definition at its ports, and Bits' own doc names `Flattens` as a reader of Bits' layout, which it is not yet. Also: my last two messages carry times ahead of when they went; they went at 1:40 PM and 1:45 PM PST.**

Marks as before: [read], [argued]. Nothing built, nothing run.

**Firmer: `StmtFlattens` does not hold of the deployed parser**
- A member's program has `2 + n·B + 3·(n·R)` gates: the constant, the zero, `B` source gates a unit, three gates a row. That is proved (`Flock.Bridge.Members.length_memberProg`, `security/Proofs/Flock/SpotFlock/ScopeDesc.lean`) [read]. `Flattens` asks a statement to have as many gates as its descriptor's root states, gate `g` being primitive `g`.
- `Spot/Bits.lean`'s doc says it in a sentence: "The circuit file holds each unit's Boolean circuit as a whole, not each primitive's: no row is a primitive's own, so a packed primitive's bits are placed through its unit's ports, never through its Definition." [read]
- In `security/Proofs/Flock` one place builds a sampled statement's program, `SpotFlock.parse`, and it sets `bitsProgs h.ms`; no proof there reads the statement's `desc` against a Program [read, by search for `prog := `, `h.desc`, `Program.decode`]. So the one-stage line of my not-checked list is closed as far as that search reaches.
- Still [argued]: I compiled no counterexample. A descriptor contrived to have Bits' shape is not ruled out by what I read; a registered workload's is not one.

**Too thin in my message: "I found no lemma that relates the descriptor to Bits' positions"**
- There is code for the tie, at a template's ports: `Spot.Bits.check c P tmpl` says why Bits does not lay the circuit file `c` out as template `tmpl` of the Program `P`. It holds the file's input rows to the Definition's parameters, its output rows to the return's fields, each on 16-bit words, one template instance a unit; it never reads the unit's rows [read]. `Bits.inBit` and `outBit` then give the claims position of a bit of an instance's input port or returned output.
- Nothing calls `Bits.check` at lean's mark except its own guards, and no file under `security/` names it [read, by search]. So the tie is code with no statement about it yet.
- Bits' doc says "The sampled leg's bridge, `Properties.NetworkScan.Flattens` and NetworkCheck read these definitions and hold no second copy of them." Neither reads them at the mark: `Flattens.lean` imports `NetworkCheck.Period` and the sampled audit's definitions, and the certifier verifier's package does not require Spot's [read]. I read that sentence as the layout's owners' target for `Flattens`. It is a doc line, not a ruling.

**What this changes in the sizing**
- The two ways I gave are one direction with two depths. The statement that replaces `StmtFlattens` is about Bits' layout: each boundary gate the scan check names is a port bit of a template instance, at `inBit` or `outBit`, and the fit is stated there.
- What I have not read, and what decides how deep it goes: whether every gate the scan's layout names (ingress, egress, presence, the view) is a parameter or a return of a template's instance. `Bits.check` places only those. A gate inside a unit has no position of its own in Bits' layout.
- For my bound nothing changes from the first message: `ScanCountBounded` asks `Flattens` of each window's statement, so it waits on this.
- Mon 30 Nov for `StmtFlattens`: the same answer. It needs your ruling on the statement's form and a read of the scan's templates against `Bits.check`; the proofs it touches are ours (`scanFits`' files) and Bits' owners' (a statement about `Bits.check`).

[2026-11-12T21:50Z] c63dddbef3a4
ONSITEMAIN, Thu 12 Nov 1:50 PM PST. **The toy by the pinned line ended 64, 0 and 66. The port's built cuts are on lean's ready head and their checks are launched. Two things for you below: D2's first fact, with a third way that changes no definition, and the ten lines for E and F.**

**1. Friday's toy** [ran]: `r20261112-213324-c687`, exit 0, 781 s of which the slot's wait is most; its three steps ended 64, 0 and 66. Its output differs from the second toy's only in the machine's sizes and the seconds. `LAUNCH.md` records it. The line is as my 1:50 PM message says, with the held `main.txt`; I still need your word on that file.

**2. D2, the first fact** [read: the device's code on `main` at `5f0c008aa040`, `warden.py`, its docstring and `_egress`; the Lean on lean's branch]:
- `main`'s device sends a padding frame in every slot after a lane's last cell to the end of that bucket, and the connection ends with that bucket. Its checker expects the same.
- `main`'s Lean already counts those slots as the lane's cells: `RecordRows` says a lane's `cells` is its length "through its last bucket", and the fixture's schedule on `main` has `![6, 2, 2]` where the onsite line has `![5, 1, 2]` with the tail padding beside it. Both describe the same record.
- **So there is a third way, and it is the smallest:** `laneFrames` does not change and no `wireFrames` stays. The onsite lemmas are stated at schedules whose `cells` count the last bucket's slots, where the tail padding is zero. It needs nobody's yes for a changed definition. What it costs: the onsite lemmas that speak of a lane's own last cell (not its bucket's end) need that cell named another way. I have not restated them yet.
- The second fact, `main`'s guards run with `laneFrames` changed once: the change is written with its three proofs (one commit, scratch), and the try of the area's root, `r20261112-210618-9a29`, launched once at 1:06 PM PST, has waited for a slot since. I send its result when it ends.

**3. Lean's ready line is `bafae2eaec2e`** [read: the queue's status says #9375 is ready at that head, admitted, waiting on a land verdict]. I took it as the second take: the scratch series is rebased on it, no conflict, the same tree as a merge gives. Order now: A (4 commits), B (2), I1 (1), L (2), then C (4) and D1 (2), which stay in scratch.

**4. Checks, each launched once at about 1:43 PM PST, all waiting for a slot** [ran: launched; no result yet]:
- A `r20261112-214251-db79`: the Lean check of the certifier's two packages and of the onsite verifier, which requires them, then the suites touched since lean's head.
- B `r20261112-214254-8906` and I1 `r20261112-214257-d77f`: the suites touched.
- L `r20261112-214259-7dfe`: the Lean check of the onsite verifier, then the suites.
- The series' head, for C and D1: a try of the area's root, `r20261112-214302-55e1`.
- Not in them: the Lean check of the two `security` packages that require the certifier's (Properties and Proofs) at A's head. The head's root try builds part of that and no more.

**5. L is built** [ran: the try of `CoreRemoteNetworkNoPouw.lean`, `r20261112-211311-5036`, exit 0, which builds `CoreRemoteNetwork` on its way; the two executables' link is in L's check above]. Two development executables on `main`'s `Period`; `Core.Composites` gains nothing. Body in my note.

**6. A correction to my 12:43 PM report:** cut I's other three files (`attacks.py`, `scan_tie.py`, `network_sim/onsite.py`) cannot be built before J and N: they import the piece's front and the ingress scan [read: their imports]. Only I1 of cut I is built.

**7. H: what it is, read per declaration** [read: the three texts of `Period.lean` and job 6's rebuilt head `c299bad61230`, which is on lean's ready head]. I cut it on job 6's scratch head and the body will say so.
- `main` already has part of what staging's side changed: the refusal of a profile naming more sampled than its level holds (in `factsOf`), `chargedOf`, `rowsPublic`, the terms' `faultyEntries`.
- What staging adds beyond `main`, in five parts: the audit's circuit record (a new module of 785 lines over the sampled proofs' wire check, and `auditOf` counting an audit's free inputs itself, so **the parameters' `free` key goes**); the class circuit every tied audit runs and the `held` checks; the schedule's caps held to `cells`; the scan's state read as its transitions' chain, its row 0 empty and each tied audit's reads at its slots' anchors; a charged audit's bound read at one δ for the tied (`tiedDelta`).
- It is 24 new declarations, 19 changed and the new module, with the period's fixtures, `period_case.py` and the two `NetworkPeriod` files in `security`. It is not a text merge: each part is commits of its own, built in that order. I start with the new module.
- **One question:** the parameters' format changes (`free` out, `circuit`, `order` and `pins` in for a tied audit; `circuit` in the rows form's keys). Is that format H's to change on `main`, or does a reader of `main`'s period parameters outside our lane need a word first?

## E and F: ten lines for the issue

1. **What it is.** Four guarantees beside `main`'s four, with ruling E's names (`SampledAuditAcceptedWired`, `SampledLeakAcceptedWired`, `SampledLeakRegisteredWired`, `PeriodLeakAcceptedWired`), stated in the certifier's area over `Claims.SamplingGeneric.Compiled`. `main`'s four keep their text.
2. **The wiring** is one more public argument beside the layout, `wires : AuditStatement → ℕ → Option ℕ`: `wires s g = some w` says the input gate `g` reads the gate `w`. On the onsite line it is a field of `BoundaryGates` [read]; as an argument, which is my proposal, `main`'s `BoundaryGates` gains no field.
3. **The check on a proof block** (`wiredOk`): the proof block is correct, as `main`'s `Compiled.opens` concludes, and each wired gate it holds or reads opens to its writer's committed value, the writer not a constant gate.
4. **The incorrect set** is the faulty set and the proof blocks that fail the wires' part. The wired relation is `main`'s `RAcc` with the incorrect set in the faulty set's three places and the circuit's Program reading each wired gate at its writer.
5. **Shape of the first:** `∀ H (D : Compiled Coin A) L K S lay wires k, D.KSWired H wires → KSAcceptedWired H D L K S lay wires k`. `main`'s is the same line without `wires`, at `D.KS H` and `KSAccepted`.
6. **Shape of the other three:** `main`'s statement of the same name with those two replacements and the scope asking also that the wiring fits the program (each wired gate an input, its writer returned or an input, no cycle). Their bounds and their other premises are `main`'s: `NetTiming.ScopeAcc`, `NetTiming.RegOpens`, `NetTiming.CertifierRegisters`, `CommitmentScheme.Binds`, `TrustedHost`, each where `main`'s lock lists it [read: the lock on lean's branch; derived for the wired ones].
7. **The one new premise:** `Compiled.KSWired H wires`, which is `Compiled.KS` with the sampling relation at the incorrect set: at every `δ ≥ 0`, `KSLinOn` at `main`'s verifier, scope, extractor and errors, the relation saying the incorrect set is in `M_δ`. It is open, as `Compiled.KS` is under `main`'s four.
8. **What discharges it, and the question for proofs:** a compiled verifier whose `opens` concludes the wired check where `main`'s concludes correct, which is the verifier of record checking each sampled session's wired reads. The general form is sampling at any check on each proof block; the onsite line has it over `Deploy` (`KSBy`, `BlockCheck`), and lean's branch has neither [read]. Whether that form over `Compiled` is proofs' to define is the question.
9. **What `NoWires` then gives:** at `wires = fun _ => none` the wired check is correct, the incorrect set is the faulty set, the relation is `main`'s `RAcc` and `KSWired` is `Compiled.KS`. So each wired guarantee gives `main`'s guarantee of the same name, word for word: four theorems, wired implies `main`'s. `main`'s four then follow from the wired four and keep their lock entries.
10. **Nothing carries as text** [read]: the onsite statements, `NoWires.lean` and `KSWired.lean` are over `Claims.Sampling.Deploy` with `KS`, `KSBy`, `E`, `κ`, `γ`, which lean's branch does not have. What carries as ideas: the wired check, the incorrect set, the relation, the four no-wires lemmas (`ksWired_none`, `incorrect_none`, `RWired_none`, `resolveWires_nil`). Until it lands, the run of record's report cites none of the four.

**Least sure of:** the third way for D2, until the lemmas are restated at it. Not looked at: NET-325's and NET-328's own branches, if any; which of H's parts proofs' statements in `security` read.

