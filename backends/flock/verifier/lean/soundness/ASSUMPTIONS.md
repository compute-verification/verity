# Flock interactive soundness: what the theorems assume

For the expert reviewer and for Table 1. This file states every named assumption and every named hypothesis the Flock
soundness theorems take, each in full; the detail behind them is in `assumptions/`. What is proved is in `README.md`
and, for the audit layer, `FlockSoundness/Audit/README.md`. `DESIGN.md` holds the derivations.

## Named assumptions (taken without proof)

There is one, A2, a `Prop` (`Assumptions.SHA512ExpectedTimeCR`) that the link theorem (`flock_batched_linkSoundE`)
takes as a hypothesis (`hCR`).
`README.md` §1.5 names one more, L1, about the lowering, stated below. The development declares no `axiom`. So a
mis-stated assumption can make a theorem vacuous, but it can never make Lean prove something false.

**A2. Expected-time collision resistance of SHA-512** (in CDGSY24's expected-time sense, ePrint 2024/1434), for the
audit's value layer (Daniel, Sep 27; constant corrected the same day).

- **Statement.** A collision finder whose expected number of SHA-512 evaluations is `T` succeeds with probability at
  most `T/2^256`. `SHA512ExpectedTimeCR` states it for one finder: a game with a fixed strategy, whose outcomes
  record the pair output and the evaluations made, `t` for each run of the prover.
- **Per finder.** Some finder outputs a collision after no evaluations, since collisions exist. So the `Prop` is not
  stated for all finders: the link theorem takes it for the finder it builds from the prover.
- **Why it is plausible, and where it is used:** `assumptions/a2-sha512-expected-time-cr.md`.
- **Upstream:** watch entries `collision-resistance`, `keyed-cr` and `rom-cr` (see Upstream status).

- **L1. Each template's rows compute its gates** (named hypothesis). That is, the rows' outputs are the gates' outputs, in
  the program's Boolean circuit, on every input. The audit certifies drawn units against their rows. That says something
  about the program's Boolean circuit only through L1.
  - **The evidence for it, and the path to discharge it:** `assumptions/l1-template-rows.md`.

A1, Reed–Solomon mutual correlated agreement up to the Johnson bound, is proved: `mcaError_le_bchks25`, from DKT26 in
ArkLib (`README.md` §1.6; watch entries `A1-johnson-mca` and `udr-mca`).

## Named hypotheses of the proof

- **The link points' layout** is a hypothesis (`Statement.LinkLayout`). Each region's free bits must be distinct and
  read link coordinates below `PT_LOCAL`, and the block coordinates `PT_LOCAL …` must fit inside the point. The
  statements of PROTOCOL.md §16 satisfy it.
- **The placement** (`Placement`; `lowering_sound`, `README.md` §1.5):
  - Placed (`Placement`) means two things:
    - the block matrices' rows at the unit's computed columns read exactly its rows' columns;
    - the rows at its constant read the pin, as every slot's constant is wired to it (spec §16.1).

**The audit layer's named hypotheses** (`Prop`s taken as hypotheses; no `axiom`; the theorems are in
`FlockSoundness/Audit/README.md`):

| Hypothesis | Term | Meaning | To be discharged by |
|---|---|---|---|
| `Analysis.KnowledgeSound` | `ε_ks` | the session accepts while extraction of a drawn unit (picked before the session's first coin) fails | `table_knowledge_sound_joint` (`README.md` §1.3), after Jensen over the prover's states |
| `Analysis.LinkSound` | `δ_link` | extraction succeeds but differs from `X` on the unit's committed inputs and outputs | compiled layer: `flock_batched_linkSoundE` (A2, `ValueBinding`) |
| `Partition.AnchorsSound` | `δ_in` | the session accepts while the committed inputs differ from the anchors' values | `registered_weights` (`README.md` §1.3); at the oracle layer, `InputsSound` (`oracle_anchorsSound`) |
| `Partition.InputsSound` | `δ_in` | oracle layer: the session accepts inputs other than the anchors' with probability at most `δ_in` | **proved for Flock** (`flock_inputs_sound`, `2^-205`), given `InputsLoweringSound` |
| `Partition.InputsLoweringSound` | — | inputs other than the anchors' leave some session table (the always-proved input statement) unsatisfied | the input unit's lowering (level 3) |
| `Partition.SessionSound` | `ε` | oracle layer only: a session with a drawn wrong unit accepts with probability at most `ε` | **proved for Flock** (`flock_session_sound`, `2^-205`), from `table_value_sound`, `execArith_correct` and the lowering |
| `Partition.LoweringSound` | — | a drawn wrong unit lies in a session table whose statement no committed table satisfies | the lowering theorem (level 3, another lane) |

Both per-unit hypotheses are averages over the whole audit, for every prover and every rule that picks a drawn unit
before the session's first coin, not bounds state by state.

| Named at the compiled layer | Meaning | Discharged by |
|---|---|---|
| `ExtractionAnalysis.LinkSound` (`analysisB`, `analysisC`, `analysisBE`) | `δ_link`: extraction succeeds, but every close satisfying candidate decodes, on the unit's committed wires, to values other than the committed transcript's | for `analysisBE`: `flock_batched_linkSoundE`, under A2 for each commit string's finder and `ValueBinding` (named; the `hm96` layout discharges it); composed in `flock_batched_count_linked`, `_drawn_linked` (`_placed`: from `UnitPlace`) |
| `Partition.LoweringSoundB`, `Partition.LoweringSoundC` | a message packing a witness that satisfies the statement of a drawn unit's table decodes, for that unit, to values on which it is correct | `loweringSoundB_of_place`, `loweringSoundC_of_place` from each drawn unit's `UnitPlace` (`lowering_sound`, `README.md` §1.5), given the statement builder's `Placement` (level 3) and L1 |

## Cryptographic assumptions and coins

- **Collision resistance of SHA-512** (`cr/sha-512`) for the Merkle openings (spec §14, with SHA-512 leaves and nodes).
  It is the only hash assumption of the compiled theorem: plain collision resistance, for every classical prover.
  The audit's value layer adds A2 (above).
  DESIGN.md §1 explains why SHA-512 rather than SHA-256, and how the bound depends on the hash's width.
- **No assumption for the rounds.** The verifier keeps each round's bytes, then draws that round's coins, and the
  replay compares bytes. This replaces the SHA-256 round digests. Keeping the bytes costs about the proof size, 0.7 MB
  per rep.
- **Statement binding** (the statement digest, Σ and `link_sha256`). These are in round 0's kept bytes, fixed before any
  coin. Their own collision resistance matters to what the statement refers to (registered data, linked proofs), not to
  this theorem.
- **`os` coins** (the default). Each round's coins are uniform and unknown to the prover before it sent the round. They
  come from the verifier's `/dev/urandom` (spec §7.1). The randomness is trusted, not proved.
- **`seed` coins** (PR #83, `coin-commit/sha256/v1`) add two assumptions: the seed commitment hides the seed, and
  SHA-256 in counter mode is a PRF. They are needed only if the seed mode becomes the verifier of record's source.
- **The random-oracle model is not used.** Its version of the compiled bound is only a cross-check (DESIGN.md §2).
- **Quantum provers** (DESIGN.md §9). The oracle layer holds against them, but the compiled theorem is classical. The
  compiled protocol is post-quantum sound only asymptotically, and only if SHA-512 is collapsing, which is a conjecture
  (Table 1 would need `collapsing/sha-512`). No concrete post-quantum bound is known at these parameters, and none is
  claimed.
- **Table 1** cites Flock's soundness under the `live-verifier` model and the collision resistance of each row's hashes
  (`cr/sha-512` in the C-Flock SHA-512 row); A1's proof took `rs-proximity-johnson` off. An audit's integrity profile
  would add A2 as `ecr/sha-512`, a `verity.claims` id that no row cites yet.

## What is pinned

`lean-audit.json` records each pinned theorem's statement (`pins`) and the definitions it reads (`reads`). The audit
fails when either changes until the record is rewritten, and a rewritten record needs a named statement reviewer. The
pins include the table theorems (`README.md` §1.1 and §1.2, with their `_exec` forms), `execArith_correct` and
`Merkle.opening_binding`; none takes a named assumption. The theorem that takes A2, `flock_batched_linkSoundE`, is not
pinned, nor are `flock_inputs_sound`, `flock_session_sound`, `lowering_sound` and `mcaError_le_bchks25`. A theorem that
is not in `pins` is still audited (standard axioms, kernel replay), but a change to its statement would not fail the
audit.

## Upstream status

Whether a dependency proves what these theorems assume, or what they prove by hand, is computed, not written here. The
watch list is the `upstream` section of `lean-audit.json`, one entry per statement:

- `A1-johnson-mca` and `udr-mca`: A1 (proved here from DKT26) and its unique-decoding fallback;
- `collision-resistance`, `keyed-cr` and `rom-cr`: A2, in three forms;
- `list-size` and `list-size-code`: the Johnson list bound (`ListSize.closeMsgs_card_le`);
- `tensor-fold`: the fold union in `LigeritoLevels`;
- `novel-basis`: the novel basis (`Arith.Correct.xhat_poly`, `L ≤ 40`).

`python tools/lean/upstream.py backends/flock/verifier/lean/soundness` prints each entry's status at the pinned
revisions: the upstream theorems about it, which are proved, and which nobody has acknowledged. The audit runs that pass
over ArkLib, with ArkLib's own `sorry` baseline, and fails on a proved theorem no entry acknowledges. `--lean`, the exact
pass over everything imported (VCVio and CompPoly too), is run at a dependency bump. L1 has no entry: no dependency
states this repository's rows.

## Trusted components

Not assumptions of the theorem, though the verdict depends on them: `assumptions/trusted-components.md`.
