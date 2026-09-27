# Flock interactive soundness: the named hypotheses

For the expert reviewer and for Table 1: every hypothesis the Flock soundness theorems take, and what discharges it. Each
is a `Prop` or a structure taken as an argument. The development declares no `axiom`, so `#print axioms` lists only
Lean's three standard axioms (`FlockSoundness/Check.lean`). A mis-stated hypothesis can make a theorem vacuous, but it
can never make Lean prove something false.

Each hypothesis's justification is its own file under `assumptions/`. What is proved is in `README.md` and, for the audit
layer, `FlockSoundness/Audit/README.md`. The derivations are in `DESIGN.md`.

## Named assumptions (taken without proof)

| | Statement | Lean | Taken by | Justification |
|---|---|---|---|---|
| **A2** | Expected-time collision resistance of SHA-512: a finder whose expected number of SHA-512 evaluations is `T` finds a collision with probability at most `T/2^256` | `Assumptions.SHA512ExpectedTimeCR` (`hCR`), for each commit string's finder | the link theorem (`flock_batched_linkSoundE`) and the composed audit (`Audit/FlockLinked.lean`) | `assumptions/a2-sha512-expected-time-cr.md` |
| **L1** | Each template's rows compute its gates in the program's Boolean circuit | outside Lean: the theorems are about the rows | every claim about the program's circuit | `assumptions/l1-template-rows.md` |

## Hypotheses of the proof, still to discharge

| Hypothesis | Lean | Taken by | To be discharged by | Justification |
|---|---|---|---|---|
| The link points' layout | `Statement.LinkLayout` (`hlay`) | every table theorem | the statement: PROTOCOL.md §16's statements satisfy it | `assumptions/link-layout.md` |
| The placement | `Placement`, `UnitPlace` | `lowering_sound`, `loweringSoundB_of_place`, `loweringSoundC_of_place`, the `_placed` forms | the statement builder (level 3) | `assumptions/placement.md` |
| The lowering | `Partition.LoweringSound`, `InputsLoweringSound`, `LoweringSoundB`, `LoweringSoundC` | the Flock audit theorems | compiled layer: `UnitPlace` and L1; oracle layer: the lowering theorem (level 3) | `assumptions/lowering.md` |
| Value binding | `Audit.Partition.ValueBinding` (`vb`) | the link theorem and the composed audit | the `hm96-sha512` leaf layout (M0) | `assumptions/value-binding.md` |
| The anchors | `Partition.AnchorsSound` (`δ_in`) | the audit's cone theorems | oracle layer: `InputsSound`, proved for Flock; compiled layer: `registered_weights` | `assumptions/anchors.md` |

## Discharged for Flock

The generic layers name these, and the Flock theorems prove them (`README.md`, `FlockSoundness/Audit/README.md`):

- `Arith.Correct` (`hA`), the GHASH arithmetic's facts: `execArith_correct`, for the executable's instance;
- `Analysis.KnowledgeSound` (`ε_ks`): `flock_batched_knowledgeSound`, `flock_compiled_knowledgeSound` and
  `flock_batched_knowledgeSoundE`;
- `Analysis.LinkSound` (`δ_link`): `flock_batched_linkSoundE`, under A2 and `ValueBinding`;
- `Partition.SessionSound` and `Partition.InputsSound` (oracle layer): `flock_session_sound` and `flock_inputs_sound`, at
  `2^-205`, given the lowering.

## Proved, not assumed

Reed–Solomon mutual correlated agreement up to the Johnson bound (A1 until Sep 27; `mcaError_le_bchks25`, from DKT26),
Merkle binding, ring switching, the Johnson list bounds, the sumcheck rounds and the pinned zerocheck coordinates
(`README.md` §1.6).

## Outside the theorem

- **Cryptography and coins** (`assumptions/cryptography-and-coins.md`): plain collision resistance of SHA-512 for the
  compiled theorem, charged as explicit collision-finder terms rather than a hypothesis; the coins; statement binding;
  and what Table 1 cites.
- **Trusted components** (`assumptions/trusted-components.md`): Lean's kernel, compiler and runtime, the model against
  the executable verifier, `Arith.Correct`'s range, the statements themselves, and the pinned dependencies.

A new hypothesis gets a row here and a justification file under `assumptions/`, named for it.
