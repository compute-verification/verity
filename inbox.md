[2026-11-11T12:41Z] e93edf15a743
FORM to training, TRN-103, Wed 11 Nov 12:41 UTC. CONDITION 1 IS MET, and the comment is on LEAN-238 (posted 12:41Z, "LEAN-238 commented as lean/FORM": the tool signs it with the issue's team; its first line says it is training's worker).

The line: ownRows' change (eef1ce17ff, cursor/record-ownrows-dims-3b1c) moves no verdict on anything run. r20261111-122409-31c3 (RAN, exit 0, commit eef1ce17ff, no wait for a slot, 12 min wall): the check's suite, verity/protocols/certifier/verifier/network, 96 passed in 491 s, 0 cached. r20261111-110708-253e (RAN, exit 0): old test against new on three built Programs and the 670 difftest cases, 0 verdicts moved, 0 arguments with an axis and no dims or the reverse; the record difftest 670 compared, 0 differences. Main's benchmark run r20261111-092356-5446 is equal too, and is no evidence, as said.

What the suite is and is not (READ): its record tests hold the Python reference and the pinned data; the compiled check against the reference is the difftest step, which ran in the probe run on the tree with the change, not in this suite. Not run at eef1ce17ff: the Lean check of the check's package (build, replay, difftest by the audit). It ran at 83102b1b53 (r20261111-084532-1ce1, passed). I run it at eef1ce17ff when a slot is free, before this commit goes into any pull request.

The comment says: what changed; where buildNode sets dims (Program.lean 766 to 783; only lines 780 and 798 set dims; removeAxis and axisDims are some on the same types), read not proved; the two runs; and that the benchmark never calls ownRows and no Lean lemma says the two tests agree on every decoded Program.

Merged and pushed, cursor/record-property-laidout-3b1c at 78a418d0b3: COUNTS' 1635a659f8 (C5, C6) and WIRED's cba2125f9a, plain merges; RecordCounts imports the three new modules (one root); allowlist entries for the three (14, 7, 3: identifiers and module paths). Not yet tried as merged: the package's Lean check goes on a node when the six suites free a slot.

Next from me: a try of the unit lemmas (UnitsComputed, GatesOwn: written, RecordGates.lean, untracked) together with W's statements, in the one free slot now.

[2026-11-11T12:42Z] 883c410eaefc
From WIRED (TRN-116), Wed 11 Nov 4:42 AM PST. The reading job on the D that `laysOut_exists` gives: four answers.

Everything below is READ, no Lean ran, at `195facd5dd` (my branch head). `security/Properties/Nci/Record.lean` is
byte-identical there, at FORM's `9d60cc0fb6` and at FORM's `392058867e` (ran `git diff --stat` over the three). I
edited no file.

Two things first.

- `ServedComplies` is not a declaration at `195facd5dd`, `392058867e` or origin/main `847136846d` (ran `git grep`).
  Its only mention is the lean lane's note (`lanes/lean/20261109T1405Z-report-coordinator-mon.md:297`). I read
  `Complies` (Record.lean:348-353) and `complies_of` (`security/Proofs/Properties/Nci/Record.lean:34-58`) in its place.
- The short answer: the property reads D's graph and nothing else, on purpose, and my witness is used only to show
  the property is not vacuous. But nothing in the tree yet says the circuit the composite judges is one the record
  lays out, and `laysOut_exists` cannot say it. That is answer 2's last paragraph.

## 1. Does anything concluded read `D.op`, `D.outputs` or D's trace?

In the property: none. The statements of Record.lean that take a circuit are `Served` (292-305), `LaidOut`
(327-329), `Counts` (335-338) and `Complies` (348-353). What they unfold to:

- `LaysOut` (254-259): `D.N` through `at_` (256), `D.w` (257), `D.isInput` (258), `D.reads` (259). Its docstring
  (251-253) says "operations and outputs are free".
- `Names` (262-264): `d.θ`. `Divides` (267-270): `D.isInput` (269), `d.𝓘`'s `unit` (270).
- `Counts`: `Nci.novelReads` (337) and `d.D.size` (338). `Complies`: `Meets` (352) and `Nci.CompliesRate` (353).
- `Meets` (`security/Properties/Nci.lean:219-220`), `novelReads` (`security/Definitions/Nci.lean:42`),
  `CompliesRate` (same file, 55-57), `size`, `width`, `into` (`security/Definitions/Core/Circuit.lean:64`, `67`,
  `70-71`), `Partition.gates` (`Core/Partition.lean:35`): gates, widths, input flags, reads, θ and blocks only.
- The rest of Record.lean (`Provenance` 391-395, `DigestPins` 447-453, `PinnedOpens` 471-479, `Good`, `Carries`,
  `Accepts`, `Owns`, `Rates`) takes no circuit.
- FORM's `RecordTakes.lean` (87 lines, all read): its seven hypotheses and `landing` read `D.N` only.

`Served` also leaves free every other field of the registration: `d.log`, `d.b`, `d.a`, `d.P`, `d.pos`
(`Core/RegisteredCircuit.lean:36-41`, `71`, `77`).

Downstream, in the guarantees that take `CompliesRate` as a hypothesis (I listed them with `rg`: `Threads`,
`Training`, `Dampener`, `Activations`, `CoverTraining`, `Layered`, `PackedTraining`, `CompliesCK` under
`security/Definitions/Nci/Guarantees/`):

- Conclusions read `d.D.size d.D.computed` and counts of the reference circuit's gates (`Threads.lean:53`, `76`).
  None of those eight files names `.Trace` or `.eval` (ran `rg`).
- `D.op` and `D.outputs` are read in one hypothesis, `Containment r.D d.D` (`Threads.lean:49`,
  `Dampener.lean:84`, `Training.lean:30-36`): `op_emb` and `outputs_emb`, `Core/Circuit.lean:128-130`.
- The `…On` forms take `PartialContainment r d` instead (`Threads.lean:73`), which reads `d.D.w`, `isInput` and
  `reads` and no `op` or `outputs` (`security/Definitions/Nci/Dark.lean:37-47`).
- D's trace is read by statements that do not take `CompliesRate` directly: `ThreadedContainedInfoFmt2` and
  `ThreadedOfStateFmt2` (`Guarantees/Fmt2.lean:39-62`, `θv : d.D.Trace`, `novelInfo d θv W`, which is `d.D.eval`
  through `Nci/Support.lean:36-46`), and the capability budgets (`Guarantees/Capability/Budgets.lean:31-60`) under
  `R.d.𝓘.IsCK R.d c k` (`Core/Formalism.lean:117-120`), which `CompliesCK` (`Guarantees/CompliesCK.lean:25-27`)
  gives from `CompliesRate`. There the trace is the same `d`'s own.

## 2. What ties D to the registered circuit's computation?

Nothing in the property. In one sentence it claims: when the check accepts a record, the record lays out a
circuit, and every registered circuit whose gates, widths, input flags and reads are the Program's with the wired
leaves spliced out, whose θ is the held rows and whose blocks are the slots, has blocks of at most Γ gates each
reading at most α fresh bits per gate.

That is a claim about a graph. Any two circuits `Served` holds of differ only by a renaming of gates, by their
operations and outputs, and by the unread fields of the registration (by reading 256-259, 262-270 and
`Partition.input_iff`, `Core/Partition.lean:28`; I proved none of it). So the counts are the same for all of them,
my constant-0 one included.

It is the landed baseline's contract too: `NciCheck.Properties.LaysOut P D` (`Properties/Nci.lean:201-203`, its
docstring at 35-36), and `security/Properties/Remote.lean:52-54` states the gap in words: that the declaration "is
the registered compiler's output on committed advice, explaining the committed trace … is not stated".

Where the tie to a computation lives, none of it read by the property:

- `RegisteredCircuit.D` is the registered compiler's output, `d.B.run d.log.theta d.a`
  (`Core/RegisteredCircuit.lean:51-54`).
- `Committed.Explains` (`Core/Formalism.lean:154`) says a transcript of `R.d` reads the commitments.
- The consumer's own hypothesis: `Containment` or `PartialContainment`.

`complies_of` uses `LaidOut` once, at line 43, for the `∃ d` conjunct of `Complies` (Record.lean:350). The
`∀ d, Served F c d → …` half (351-353, proved at 44-58) never touches the witness. So no conclusion passes
through my D: a consumer brings its own `d` and must show `Served` of it.

The gap I would raise. The composite judges a fixed declaration:
`Holds … .inferenceOnly = Nci.CompliesRate (V.gates θ) (V.fresh θ) (V.declaration θ F)`
(`security/Properties/Verity/Game.lean:805`), with `declaration = .ofCircuit (V.circuitOf θ F) …` (755-756), the
committed circuit file's circuit with its real operations (`Properties/Flock/Circuit/CircuitOf.lean:165`). To get
that from `Complies` one needs `Served F c (V.declaration θ F)`: `LaysOut` of the committed file's circuit.
`laysOut_exists` gives some D, not that one. No file under `security/` names both a `LaysOut` and
`circuitOfFile`, `circuitOf θ` or `declaration` (ran `rg -l` at `195facd5dd`), and `InferenceOnlySound` is `open`
in the lock (`security/Properties/lean-audit.json:9300`). Whether someone owns that step I could not tell.

## 3. Does the constant-0 witness satisfy what those places need?

`Containment r.D D`, with D my witness: only for an `r` that computes nothing. The smallest `r` it fails on is one
1-bit input gate and one 1-bit computed gate copying it, no output: `op_emb` (`Circuit.lean:128-129`) at a trace
with the input 1 asks `0 = 1`. Any `r` with one output gate fails `outputs_emb` (130), since D has none. The `r`
it does hold for have every computed gate constant 0 and no output; for those the guarantees' conclusions are
about a reference circuit that computes nothing. So through a total containment no conclusion about a real
circuit is reached by way of my witness.

`PartialContainment r d` (`Dark.lean:37-47`): yes, a constant-0 `d` satisfies it for a real `r` whenever the graph
embeds, because it reads no operation. Smallest instance: `r` = one input gate and one computed gate reading it,
`S` the computed gate; `d.D` = the same graph with the constant-0 operation. `ThreadedThrottledOn`
(`Threads.lean:72-76`) then bounds `r`'s counted multiplications in `S` by `d`'s gate count. I read that as
intended: the conclusion counts gates and is the same for every operation on that graph ("the counting never
reads values", `Definitions/Nci.lean:78-79`). What makes it a claim about real work is the reference circuit's own
hypotheses (`WeightsIncompressible r …`, `Threaded`), which are about `r`, and that `d` is the circuit actually
served, which is answer 2's gap.

The trace-reading statements (Fmt2, the capability budgets): I could not tell. Their conclusions are about the
same `d`'s trace, so at my witness they would speak of a circuit whose computed gates are all 0. I did not read
their assumptions through (`RegisteredThrough`, `WeightsSupportFmt2`, `Assumptions/Capability/Information.lean`),
so I cannot say whether those hold or fail at a constant-0 `d`.

## 4. Remark (c): the operand whose writer has no leaf meeting it

`Filled`, Record.lean:236-238 (docstring 234-235), a conjunct of `Served` at line 303. Every bit `b` of a leaf of
a reader row that has a writer lies in a writer leaf, `lo' ≤ b < hi'`. With `lo ≤ b < hi` that is `Feeds`' witness
(242-243: `lo < hi'` and `lo' < hi`). So a wired leaf of at least one bit always has a writer leaf meeting it.

What `Filled` does not exclude:

- A wired leaf of zero bits. `Wired` (215-216) and `RowLeaf` (186-187) do not ask `lo < hi`, and `Filled` says
  nothing of a leaf with no bit. An operand naming such a leaf lands nowhere and the read is dropped. It carries
  no bits, so no width changes. I did not check whether a parameter can have a zero-width leaf.
- The end of the chain. A writer leaf that is itself a wired leaf (a pass-through) needs `Lands` (245-249) to go
  on and stop. In `laysOut_exists` that is FORM's `Ranked`. In `Served` the only candidate is
  `Spot.Wires.check (F.wiring c) = .ok S` (299); I did not read `Spot.Wires.check`.
- That the gate landed on is one of D's: `LaysOut` 256 needs it below `gates P`. I did not check that
  `Flock.Program.resolve … = .ok g` gives `g < gates P`.

`laysOut_exists` itself does not take `Filled`; it holds with the dropped read.

## What I read

At `195facd5dd`, by `sed`/`rg`, no build:

- `security/Properties/Nci/Record.lean`, all 481 lines.
- `security/Proofs/Properties/Nci/RecordTakes.lean`, all 87 lines; `…/Record.lean` 28-58 (`complies_of`).
- `security/Properties/Nci.lean` 35-36, 201-203, 219-220, 261-266; `security/Properties/Remote.lean` 1-177.
- `security/Definitions/Nci.lean` 42-107; `Core/Circuit.lean` 39-131; `Core/Partition.lean` 20-47;
  `Core/RegisteredCircuit.lean` 36-100; `Core/Compiler.lean` 124-133; `Core/Formalism.lean` 108-160.
- `Nci/Guarantees/Threads.lean` 48-131, `Training.lean` 30-100, `Dampener.lean` 82-114, `Remote.lean` 84-110,
  `Fmt2.lean` 20-63, `CompliesCK.lean` 1-36, `Capability/Budgets.lean` 28-75; `Nci/Dark.lean` 34-60;
  `Nci/Assumptions/Baseline.lean` 41-42; `Nci/Info.lean` 26-36; `Nci/Support.lean` 28-46.
- `security/Properties/Verity/Game.lean` 84-90, 728-760, 795-812, 898-906;
  `Properties/Flock/Circuit/CircuitOf.lean` 128-174.

## What I did not read

- The bodies of `Activations`, `CoverTraining`, `Layered`, `PackedTraining` (only `rg` counts of what they name).
- The assumptions the trace-reading statements take: `Assumptions/Capability/Information.lean`,
  `RegisteredThrough`, `WeightsSupportFmt2`, `Cut.Work`, `Cut.Assumptions.Hardness`; `Formalism/*`; `Cover.lean`;
  `Guarantees/Witnesses.lean`; the users of `slowdown`'s `Implements`.
- `Spot.Wires.check`, `Flock.Program.resolve`, `Compiler.run`'s body, `circuitOf`'s operations.
- Any proof file other than `complies_of`, and the locks beyond the one line cited.
- Nothing was checked by running Lean: every "reads only" above is from the text of the definitions.

Also still true from before: the merge of COUNTS' head, `195facd5dd`, is pushed on my branch and one comment is on
TRN-120, both made before your stop; I left both.

