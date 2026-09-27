# Flock interactive soundness: assumptions and theorems

For the expert reviewer and for Table 1. This file is the concise list of what is proved, what is assumed and what is
trusted. `DESIGN.md`, beside it, holds the derivations: the hash-width analysis, the compiled proof, the knowledge
extractor, the spec deviations, the zero-knowledge plan, the a16z staging, and the post-quantum and root-reuse analyses.

Lane `flock-soundness`, level 4a of the formal-verification plan: the interactive mode only. The Lean development is
this directory (`FlockSoundness/`). The verifier it is about is specified in `../../PROTOCOL.md` (PR #85). The work is
staged along a16z's zkVM security stages (DESIGN.md §7).

## 1. Theorems

### 1.1 Soundness at the oracle layer (`table_sound`, `table_sound_fast100`, `table_sound_fast100_34_35`)

Consider one table of a session, run as the verifier of record runs it: the `fast100` profile, run twice on one committed
root, with live coins, for any statement of PROTOCOL.md §16. For every prover strategy, the probability of this event is
at most `tableError(fast100 m, shape)`:

> the verifier accepts both reps, while no codeword within the level-0 proximity radius of the committed table packs a
> witness that satisfies the statement.

- **`tableError`** is a Lean function of the parameters (`Accounting/Bound.lean`), with one term per verifier coin. It
  is proved to be at most **2^-195.5 for 22 ≤ m ≤ 33** and **2^-195.4 for m = 34, 35** (`Accounting/Numbers.lean`).
- **The statement shapes covered** have at most 2^26 witness bits per block, 1,024 public regions and 64 link-point
  coordinates. That is every statement of §16, with room to spare.
- **The link points' layout** is a hypothesis (`Statement.LinkLayout`). Each region's free bits must be distinct and
  read link coordinates below `PT_LOCAL`, and the block coordinates `PT_LOCAL …` must fit inside the point. The
  statements of §16 satisfy it.
- **At this layer committed tables are sent whole.** The theorem is statistical, so it holds against quantum provers
  too.
- **For the executable's arithmetic** (`Instance.lean`), `table_sound_exec` and its two `fast100` corollaries take
  `Arith.Correct` from `execArith_correct`, so BCHKS25 Theorem 4.6 is their only hypothesis.
- **For the optimal prover** (`table_value_sound`): the same bound on `value`, which `table_sound` specializes to each
  strategy. This form composes across the tables of a batched session (§9, `value_batch_le`).
- **With M1's padded level 0** (`table_sound_pad`, `SoundnessPad.lean`; the model is `Model/LigeritoPad.lean`). M1 (#123,
  `backends/flock/live/PROTOCOL.md` §7) pads every level-0 lane with `t` coefficients, so level 0's code is
  `RS[2^d, 2^cols + t]`, of rate `ρ' = (2^cols + t)/2^d`. Each rep also folds its own `e` extra lanes with fresh coins.
  - **The bound is a function of `t` and `e`** (`padTableError`). It is `tableError` with level 0's radius, list size,
    queries and MCA numerator taken at `ρ'`, plus one MCA line and one coin on the restarted claim per extra lane.
  - **At #123's values** (`8d621454`): `q₀ = 277, 242, 229` at `m = 25, 26, 27`, `t = 2q₀`, and a pair of extra lanes per
    rep. The bound is at most **2^-196.5 per table** (`table_sound_pad_m1`).
  - **The committed values** are the witness lanes' messages `a − c₀·pad` of lane families within the padded radius
    (`CommittedPad`).
  - **The model is M1 in the clear.** The verifier computes the restarted claim `s + Σ_i ρ_i·e_i` itself. M1 masks the
    lane rounds and `e`, and its inner proof enforces the restart. That is the inner proof's soundness, a separate
    statement.
  - **New in the proof:**
    - the novel basis's degree structure, `deg X̂_c ≤ c` (`NovelBasis.lean`, derived from `Arith.Correct.xhat_poly`);
    - the padded code's list size, MCA and decoding (`PadCode.lean`, `LigeritoPadDecode.lean`);
    - the phases before Ligerito over any list of messages (`zerocheck_phase_of_list` and its three siblings, which
      the `…_of_card` wrappers instantiate at the close messages).

### 1.2 Compiled soundness (`table_sound_compiled`)

This is the protocol as run, in the adopted variant: SHA-512 Merkle caps, openings at the end of the proof, and the coin
server keeping each round's bytes (`Model/Compiled.lean`: the openings are the table's last message, after both reps'
coins, and every path is checked against its level's cap). For every classical prover `P*`, whose committed table
`T*(P*)` is its plurality table (`C0star`: at each level-0 position, the row `P*` opens there most often over the coins
after its level-0 cap):

> Pr[accepted, and no codeword within the level-0 proximity radius of `T*(P*)` packs a satisfying witness]
> ≤ `tableError` + `√(N_0·2Q_0·Adv_0)` + `Σ_{rep, ℓ ≥ 1} √(N_ℓ·Q_ℓ·Adv_{rep,ℓ})` + `Adv_×`.

- **`Adv_0` and `Adv_{rep,ℓ}`** are the success probabilities of explicit collision finders (`adv₀`, `advR`; DESIGN.md
  §2). Each runs `P*` to a cap and continues it twice, and every pair it counts is a SHA-512 collision
  (`op₀_conflict_collision`, `opR_conflict_collision`). Level 0 is opened by both reps, hence `2Q_0`.
- **`Adv_×`** is the probability that one run opens a position two ways, both verifying under one cap. Every such run
  is a collision (`SelfClash.collision`).
- **Under the generic bound** this is at most `2^-195.5 + t·2^{17.3−256} + t²/2^511` for a prover making `t` hash
  evaluations. That is at most 2^-128 up to `t ≈ 2^110.7` (m = 35) to `2^117` (m = 22).
- **No random oracle and no knowledge assumption** is used.

### 1.3 Knowledge soundness (`table_knowledge_sound`, `table_knowledge_sound_joint`; `registered_weights` to write)

It is needed for any claim about which data a proof used ("these proofs used the registered weights"), and for the
recursion track, whose outer proof must be a proof of knowledge (DESIGN.md §3).

**It is required for the partial-integrity composition of sampled audits** (§7). Halevi–Micali leaves are statistically
hiding, so every commit string opens to essentially every value, and plain soundness says nothing about which values a
unit used. The audit bound defines a wrong unit through extracted values, so it pays the table's knowledge error and a
link term per audit.

- **The extractor** (`Knowledge.lean`) works in three steps:
  - It reruns the prover after its level-0 cap `K` times.
  - It fills each level-0 position with the first verifying opening it saw there, and zero elsewhere.
  - It decodes the full table at the level-0 radius. There are at most `L₀` candidates (`closeMsgs_card_le`), and it
    succeeds when one of them packs a satisfying witness.

  Its cost is `K` reruns, `K·t` hash evaluations.
- **`table_knowledge_sound`.** If `P*` is accepted with probability `ε > ε_c⁻`, extraction fails with probability at
  most `(K·Adv₀ + N₀/(eK))/(ε − ε_c⁻)`.
  - `ε_c⁻` (`epsCminus`) is the compiled error without its level-0 rewinding term: `tableError`, the reps' cap terms
    and the self-clash probability.
  - It rests on `table_sound_compiled_of_table`: for any level-0 table `T`,
    `Pr[accepted ∧ T uncommitted] ≤ ε_c⁻ + Pr[a run opens a level-0 position off T]`. The reps' cap terms do not
    depend on `T` (`advRT_eq`).
- **`table_knowledge_sound_joint_tight`,** the form the audit consumes:
  `Pr[a fresh run accepts ∧ extraction fails] ≤ ε_c⁻ + K·Adv₀ + N₀/(eK)`, for every `ε`
  (`table_knowledge_sound_joint` states it with `2ε_c⁻ + 2(K·Adv₀ + N₀/(eK))`). At
  `K = √(N₀/(e·Adv₀))` this is at most `ε_c⁻ + t·2^-244.7` (m = 33).
- **`session_knowledge_sound`** (`Session.lean`) is the same form for one table inside a batched session.
  - The session is the protocol's (PROTOCOL.md §5). `Commit` carries every level-0 cap before any coin, the link
    points are one draw, and then every table runs concurrently in any order. The rest of the session is arbitrary.
  - The extractor reruns everything after `Commit`.
  - `ε_c⁻` and `Adv₀` are the table's terms computed in the session (`epsS`, `adv₀S`).
  - It rests on `session_sound_of_table`, the compiled bound for any level-0 table in the session. The oracle
    prover plays the whole session there, and the table's value bounds it (`orcSess_value_le`).
- **Several tables in one session** (`SessionBatch.lean`). `sessionB` is the protocol's batch: `Commit` with every
  level-0 cap, one link-point draw, then every table concurrently.
  - For each table `j`, `joint_le_B'` bounds `Pr[the session accepts] · Pr[table j's extraction fails]` by table `j`'s
    `ε_c⁻ + K·Adv₀ + N₀/(eK)` in the session (`ksBoundB`).
  - `one_le_fail_add_B'` is the audit's `cover`, for any link event that commitment implies.
- **Not yet:** the canonical HM96 values behind the witness, `registered_weights`, and the link theorem (`δ_link`,
  DESIGN.md §3). The first two need the SHA-512 leaf layout in the model, as does the link theorem's binding
  hypothesis. Under A2 (§3) `δ_link` is `t·N_s(8N₀/e)/2^256` per audit, which dominates the audit's hash term.
- **`registered_weights`.** Given the registrant's opening `(W, R)` of a root, the extracted rows equal `W` wherever the
  statement links that root, or Halevi–Micali binding gives a SHA-512 collision. So
  `Pr[accepted ∧ rows ≠ W] ≤ κ + Adv_CR(B_link)`.

### 1.4 Status

| Piece | a16z stage | Lean | Status |
|---|---|---|---|
| Interaction model: live coins, prover strategies, acceptance probability, optimal prover | (framework) | `Game/Basic.lean` | proved |
| A live coin hits a fixed nonzero polynomial's zero with probability at most degree/|F| (Schwartz–Zippel) | (framework) | `Game/Poly.lean` | proved |
| Two reps on independent coins, any interleaving: the error squares | 1a | `Game/Interleave.lean`, `value_interleave_le` | proved |
| The bound as a function of the parameters, and its value for every `m` | 1a | `Accounting/*` | proved |
| The verifier, following PROTOCOL.md §9–§16 | 1a | `Model/*` | written (definitions) |
| Composition: one table from one rep and the link points | 1a | `table_sound` | proved from `rep_sound` and `link_sound` |
| One rep is sound from a doomed start | 1a | `rep_sound` (`RepSound.lean`) | proved from the four phase lemmas below |
| One sumcheck round, in each of the spec's three encodings | 1a | `Rounds.lean` | proved |
| Zerocheck phase: the eq-reduction (pinned coordinates independent over GF(2)), the skip point, the residual rounds, the pin | 1a | `zerocheck_phase` (`RepSound.lean`), `zerocheck_phase_of_card` (`ZerocheckPhase.lean`) | proved |
| Lincheck phase: `α`, `β`, the rounds, the final check and the skip coin (statements have `k_log ≥ 6`, as `k_skip = 6` requires) | 1a | `lincheck_phase` (`RepSound.lean`), `lincheck_phase_of_card` (`LincheckPhase.lean`) | proved |
| Opening phase: ring switching (seven coins per claim) and the batch | 1a | `opening_phase` (`RepSound.lean`), `opening_phase_of_card` (`OpeningPhase.lean`) | proved |
| The list size: at most `L₀` codewords near the committed table (ArkLib's Johnson bound) | 1a | `closeMsgs_card_le` (`ListSize.lean`) | proved |
| Phase lemma: Ligerito, level by level: the folds (sumcheck and MCA), the next level's OOD binding, the queries and batching coins, the final check | 1a | `ligerito_phase` (`RepSound.lean`), `ligerito_sound` (`LigeritoPhase.lean`) | proved |
| The link points reduce region checks to opened claims | 1a | `link_sound` (`Soundness.lean`), `link_sound_of_card` (`LinkSound.lean`) | proved |
| M1's padded level 0 (§1.1): Ligerito with the padded code and the extra lanes, and the table theorem for any padding `t` and extra lanes `e` | 1a | `ligeritoPad_sound`, `table_sound_pad` (`LigeritoPad*.lean`, `SoundnessPad.lean`) | proved; the inner proof's soundness separately |
| The padded bound at #123's values: 2^-196.5 per table at m = 25–27 | 1a | `table_sound_pad_m1` (`Accounting/PaddedNumbers.lean`) | proved |
| Merkle binding: two verified openings of one position with different columns give a SHA-512 collision | 1b | `Merkle.opening_binding` | proved |
| Rewinding: openings off the plurality table cost collisions, `#bad² ≤ N·Q·#conflict` | 1b | `Rewinding.bad_sq_le`, `prBad_le` | proved |
| Compilation step: the oracle bound plus the rewinding term, for a table committed before any coin or after a prefix | 1c | `Compiled.compile_step`, `compile_step_prefix` | proved |
| Session binding: the coin server keeps each round's bytes, so they are fixed before the round's coins | 1c | the `Game` model (`send` before `coin`) | by construction; no hash |
| The compiled table theorem (§1.2): the compiled game (caps, openings at the end of the proof), the lockstep coupling with the plurality prover, rewinding at every cap | 1c, interactive | `table_sound_compiled` (`Model/Compiled.lean`, `Game/Lock.lean`, `CompiledLock.lean`, `CompiledSound.lean`) | proved |
| Knowledge soundness (§1.3): extract the committed witness, then Halevi–Micali binding to registered data | 1c, 1e | `table_knowledge_sound`, `table_knowledge_sound_joint` (`Knowledge.lean`); `registered_weights` | the table theorems proved; `registered_weights` and the link theorem (A2) **to write** (DESIGN.md §3); required for sampled audits (§7) |
| The block R1CS means the named circuit (the lowering theorem) | 1d | level 3, another lane; its generic step `Lowering.lean` (§1.5) | the generic step proved; the placement and L1 remain |
| The executable Lean verifier refines the model | 2 | with `flock-verifier` | **to write** |
| Reference prover: completeness and zero knowledge | 3, 1e | after M1 (§6) | planned |
| Sampled audits: the integrity profile over any partition and draw law, two-stage and its relations, minimax, tightness and insecurity (§9) | (composition) | `Audit/*` | proved; per-unit terms named |
| The Flock audit at the oracle layer: the batched session's projection to one table, `2^-195.4` (§9) | 1a, composition | `Audit/Flock.lean`, `flock_session_sound` | proved, given the lowering |
| The compiled lowering: the audit's circuit is the rows the verifier parses, and a satisfied table makes every placed unit correct, for every template (§1.5) | 1a | `Lowering.lean`, `lowering_sound`, `Prog.isRowsUnit` | proved, given the placement |
| L1: each template's rows compute its gates (§1.5) | (meaning) | named | evidence: the IR comparison tests; discharge by a verified lowering |

The proved results depend only on Lean's three standard axioms (`propext`, `Classical.choice`, `Quot.sound`). Run
`lake env lean FlockSoundness/Check.lean` to check.

### 1.5 The compiled lowering (`lowering_sound`, `Prog.isRowsUnit`; `Lowering.lean`)

- **The audit's circuit is the rows the verifier parses** (spec §16.5, `Flock/Net.lean`).
  - A template's unit is its input columns, its computed rows and its constant (`Rows`).
  - A computed row is `z_i = (⊕_{c∈a_i} z_c)·(⊕_{c∈b_i} z_c)`, reading earlier columns and the constant. It is one gate
    of the circuit (`Audit.Op.row`).
  - A program's circuit (`Prog.circuit`) is its units' rows, with each unit's input columns and constant wired to
    earlier gates. Every unit is an instance of its template's rows (`Prog.isRowsUnit`).
  - **A pinned file read as `Rows`.**
    - Its inputs are its self rows (`A = B = [i]`).
    - Its padding columns (`A = B = 0` in the input region) are forced zero and read by no row, so they are not columns
      of `Rows`.
    - Its computed rows lie between the input region and the constant, its last column.
  - The parser checks the input rows and the constant. `test_pinned_rows_read_earlier_columns`
    (`backends/flock/tests/test_ir_lowering.py`) checks the rest on the pinned RoPE and SiLU files: every computed row
    reads only earlier columns or the constant, and never a padding column.
- **The theorem** (`lowering_sound`, for every template). Suppose a witness satisfies a statement in which a unit's rows
  are placed in a block. Then the transcript decoded from that block makes the unit correct.
  - Placed (`Placement`) means two things:
    - the block matrices' rows at the unit's computed columns read exactly its rows' columns;
    - the rows at its constant read the pin, as every slot's constant is wired to it (spec §16.1).

    The decoded transcript then carries `true` on the constant (`UnitPlace.decode_one`).
  - `UnitPlace.correct` gives the audit's lowering hypothesis with one decoder per drawn unit. Two drawn units that
    share a wire hold it in different columns. The link, not the lowering, makes them agree.
- **The placement is the statement builder's job.** `Placement` states it. Proving it for the executable's statement
  (the slots with Δ XORed in, spec §16.1) is part of the level-3 refinement (§5).
- **L1. Each template's rows compute its gates** (named hypothesis). That is, the rows' outputs are the gates' outputs, in
  the program's Boolean circuit, on every input. The audit certifies drawn units against their rows. That says something
  about the program's Boolean circuit only through L1.
  - **Evidence today:** the IR comparison tests in `backends/flock/tests/test_ir_lowering.py`, on sampled inputs.
    - They evaluate the pinned rows against the IR's primitives:
      - RoPE on 8000 inputs (`test_rope_unit_matches_its_primitives`);
      - SiLU on every gate word and 8000 more (`test_silu_unit_covers_the_whole_table`);
      - both on the captured slices (`test_captured_slices`).
    - They evaluate the pieces the rows are built from, as gate circuits, against the IR's primitives, on 6000 inputs
      each (`test_pieces_match_the_ir_primitives`, `test_bf16_adds_match_the_ir_primitives`).
    - A test in the same file mutates one row and checks that the comparison catches it.

    They sample. They don't prove it.
  - **The path to discharge it:** a verified lowering. That means gadget proofs, with Lean re-deriving the rows from the
    gate circuit, so that the pinned rows equal the rows Lean derives. This matches the plan of record for V[B], which
    Lean generates with an equivalence proof.

## 2. The model

- **Interactive, live coins.** Each round's coins are drawn uniformly and independently after the prover has sent that
  round. Nothing is derived from a hash. Fiat–Shamir (C-fs) is out of scope.
- **Two layers.**
  - The *oracle layer* is where the statistical bound lives. There, a committed table is sent as a whole and the
    verifier reads the queried columns (§1.1).
  - The *compiled layer* replaces each table by its SHA-512 Merkle cap plus authentication paths, opened at the end of
    the proof. The coin server keeps each round's bytes before drawing that round's coins, and the replay compares
    bytes, so no hash binds the rounds (§1.2).
  - **The compiled layer is the adopted variant, not yet the spec.** PROTOCOL.md (PR #85) specifies SHA-256 trees
    (§14) and round digests (§6), following upstream. Daniel adopted SHA-512 trees and kept round bytes on
    2026-09-26 (DESIGN.md §1). The spec, upstream's Rust verifier used for cross-validation, and the provers must
    follow.
- **Sessions: batched and serialized** (Daniel, 2026-09-27; DESIGN.md §11.1). There is exactly one verifier.
  - **Batching.** A session carries every table from a time window, proved in parallel by many GPUs or datacenters.
    Sessions are serialized, and no other interactive protocol with the verifier runs alongside one.
  - **Before the first coin:** every table's statement and level-0 root, and the verifier's single coin commitment. That
    commitment is a SHA-512 Merkle root over Halevi–Micali commitments to every coin, per (table, repetition, round), so
    each table's stream is a subtree and each repetition's coins are independent.
  - **Each round:** the verifier opens a table's coins, with their paths, only after that table's bytes for the round are
    recorded. Every prover machine checks each opening against the root before answering.
  - **The coin server** runs one instance per datacenter, all opening the one commitment.
  - **Soundness per session** is the union bound over its tables, `Σ tableError`, since each table's bound holds for
    every prover strategy.
- **The committed values.** Flock runs Ligerito in the list-decoding regime, and no out-of-domain sample follows the
  commitment (spec §9.1; paper Remark 11). A committed table therefore binds the prover to a short list of codewords,
  not to one witness. The list has at most 50 entries at rate 1/2 (ArkLib's Johnson bound, `1/(2ηρ)` with η = 0.02).
  - "The committed values satisfy the circuit" means that some codeword of that list packs a satisfying witness
    (`Model.CommittedSatisfies`), which is exactly right for soundness.
  - Claims that use the committed values elsewhere (registered data, links to other proofs, recursion) need them
    extracted (§1.3).
- **The circuit.** The theorem is about the block R1CS the statement lowers to (spec §16.1), with the pinned constant
  column and the public regions. That this R1CS means the named circuit is the lowering theorem (level 3). Its generic
  step, for every template, is proved here (§1.5). The placement for the executable's statement and L1 remain.

## 3. Named assumptions (taken without proof)

There are two. Each is a `Prop` (`Assumptions.BCHKS25Thm46`, `Assumptions.SHA512ExpectedTimeCR`), and the theorems
that use one take it as a hypothesis (`hMCA`; A2's user, the link theorem, is planned). §1.5 names one more, L1, about
the lowering. The development declares no `axiom`. So a mis-stated assumption can make a theorem vacuous, but it can
never make Lean prove something false.

**A1. Reed–Solomon mutual correlated agreement up to the Johnson bound** (BCHKS25, ePrint 2025/2055, Theorem 4.6).

- **Statement.** Take a Reed–Solomon code with `k` coefficients on `n` points, reduced rate `ρ = (k − 1)/n`, and radius
  `0 < δ < 1 − √ρ`. The prover fixes any two words `u₀, u₁`. Over a uniform field element `γ`, consider the event that
  `u₀ + γ·u₁` agrees with a codeword on at least `(1 − δ)·n` positions on which `u₀, u₁` are not both codewords. Its
  probability is at most `a / |F|`, where:
  - `m = max(⌈√ρ/(1 − √ρ − δ)⌉, 3)`;
  - `a = (2(m + ½)^5 + 3(m + ½)·δ·ρ)/(3ρ^{3/2})·n + (m + ½)/√ρ`.
- **It is literally ArkLib's pinned statement.** ArkLib states it as `CodingTheory.rs_mcaError_le_in_johnson_range` and
  admits it. `FlockSoundnessTest/AssumptionIsArkLibs.lean` proves our `Prop` from ArkLib's by `exact`.
- **Where Flock uses it.** Every Ligerito folding challenge (spec §13.3), with the field GF(2^256) and the level's code
  embedded from GF(2^128). The paper's Lemma 9 turns it into "folding preserves the list", paying a row union of
  `2^(k−i)` at round `i`. Lemma 9 is to be proved from A1, not assumed.
- **One difference from the paper.** Flock's paper (Theorem 8) and its accounting use the multiplicity `⌈√ρ/(2η)⌉`.
  BCHKS25's proof supports it, but the paper does not state it. We use the printed one. This costs nothing visible:
  the term is below 2^-170, against 2^-97.8 for the query terms.
- **Status upstream.** Unproved in ArkLib, and not plannable (the design doc, level 4b).

**A2. Expected-time collision resistance of SHA-512** (in CDGSY24's expected-time sense, ePrint 2024/1434), for the
audit's value layer (Daniel, Sep 27; constant corrected the same day).

- **Statement.** A collision finder whose expected number of SHA-512 evaluations is `T` succeeds with probability at
  most `T/2^256`. `SHA512ExpectedTimeCR` states it for one finder: a game with a fixed strategy, whose outcomes
  record the pair output and the evaluations made, `t` for each run of the prover.
- **Why it is plausible.** A random oracle meets it with 0.33 bits to spare: its sharp bound is `T/E[τ_N]`, and
  hashing until the first collision costs `E[τ_N] ≈ √(π/2)·2^256 = 2^256.33` (DESIGN.md §3). The earlier `T/2^256.5`
  came from the fixed-budget bound `min(1, q²/2^513) ≤ q/2^256.5`, which that attack beats. It is linear in `T`
  because a finder can gamble.
- **Per finder.** Some finder outputs a collision after no evaluations, since collisions exist. So the `Prop` is not
  stated for all finders: the link theorem takes it for the finder it builds from the prover.
- **Where it is used.** Only the link term, `t·N_s(8N₀/e)/2^256` per audit (audit B `t·2^-205.1`, `2^-125.1` at
  `t = 2^80`). The strict-time `2√(t·2^-221.5)` and the random oracle's `t²/2^511` are comparisons (DESIGN.md §3).
  The compiled theorem, `δ_tree` and the knowledge term's conflicts keep strict-time `cr/sha-512` (§4).

**Not assumptions** (proved, or to be proved, from the libraries):

- **Ring switching.** The spec's variant (spec §12, A9: seven coins per claim, then batching, with Ligerito's sumcheck
  in place of the standalone one) is not Diamond–Posen's. Its soundness follows from the paper's Appendix B argument
  (injective recombination over GF(2)) plus Schwartz–Zippel. ArkLib's ring switching (legacy layer, 18 admits, the
  other variant) is not used.
- **Merkle binding.** Two different openings of one position against one cap give a collision: `Merkle.opening_binding`,
  proved, as VCVio proves for its trees.
- **Johnson list bounds.** ArkLib's `irs_lambda_le_johnson_mds`, proved.
- **Sumcheck rounds and univariate steps.** Root counting in Mathlib (`Polynomial.card_roots'`,
  `MvPolynomial.schwartz_zippel_totalDegree`).
- **The pinned zerocheck coordinates.** The 128 equality weights are independent over GF(2) and none equals 1. These
  are facts about constants, proved for the GHASH instance (`Arith.Correct`).

## 4. Cryptographic assumptions and coins

- **Collision resistance of SHA-512** (`cr/sha-512`) for the Merkle openings (spec §14, with SHA-512 leaves and nodes).
  It is the only hash assumption of the compiled theorem: plain collision resistance, for every classical prover.
  The audit's value layer adds A2 (§3).
  DESIGN.md §1 explains why SHA-512 rather than SHA-256, and how the bound depends on the hash's width.
- **No assumption for the rounds.** The coin server keeps each round's bytes, then draws that round's coins, and the
  replay compares bytes. This replaces the SHA-256 round digests. Keeping the bytes costs about the proof size, 0.7 MB
  per rep.
- **Statement binding** (the statement digest, Σ and `link_sha256`). These are in round 0's kept bytes, fixed before any
  coin. Their own collision resistance matters to what the statement refers to (registered data, linked proofs), not to
  this theorem.
- **`os` coins** (the default). Each round's coins are uniform and unknown to the prover before it sent the round. They
  come from the coin server's `/dev/urandom` (spec §7.1). The randomness is trusted, not proved.
- **`seed` coins** (PR #83, `coin-commit/sha256/v1`) add two assumptions: the seed commitment hides the seed, and
  SHA-256 in counter mode is a PRF. They are needed only if the seed mode becomes the verifier of record's source.
- **The random-oracle model is not used.** Its version of the compiled bound is only a cross-check (DESIGN.md §2).
- **Quantum provers** (DESIGN.md §9). The oracle layer holds against them, but the compiled theorem is classical. The
  compiled protocol is post-quantum sound only asymptotically, and only if SHA-512 is collapsing, which is a conjecture
  (Table 1 would need `collapsing/sha-512`). No concrete post-quantum bound is known at these parameters, and none is
  claimed.
- **Table 1** cites `computational-soundness` under `cr/sha-512`, `live-verifier` and `rs-proximity-johnson` (A1).
  An audit's integrity profile would add A2, a new claim in `verity.claims` once a render cites it.

## 5. Trusted components

These are not assumptions of the theorem, but the verdict still depends on them.

- **Lean's kernel.**
  - Re-check the proofs with an independent checker (nanoda, `lean4checker`).
  - Numeric checks use `decide +kernel`, which is kernel evaluation. The development never uses `native_decide`.
- **Lean's compiler, runtime and the C compiler, for the running binary.** Only the Lean verifier runs in production;
  upstream's Rust verifier is a CI cross-validation, not a second verifier. Proofs are about Lean definitions, not
  machine code.
  - **Mitigation:** the agreement tests of spec §17.1. The Rust implementation must reach the same verdict on the corpus
    and on fuzzed mutations, so a miscompilation that changes a tested verdict fails CI. Untested inputs stay exposed.
  - **Also trusted:** any `@[extern]` or `@[implemented_by]` in the verification logic. There is none today.
- **The model against the executable verifier.** Until the refinement theorem (if `Flock.Verify` accepts, the model
  accepts on the decoded proof and record), the theorem is about `Model`. Alignment rests on shared arithmetic
  (`Model.Arith`) and step order; the refinement proofs belong to the level-3 plan.
- **`Arith.Correct`.** These are facts about the concrete GHASH arithmetic: Lagrange weights on the skip nodes and the
  output coset, and the `Pcomb` weights on both (one `nodes` fact, so that all three use the same 128 points), the
  packing basis, the Reed–Solomon novel basis, the pinned constants, and the balance of query coins. They are proved for
  the instance the executable verifier defines (`execArith_correct`, from lane `flock-verifier`'s `FlockLevel3`). Two
  hold only on a range. The
  domain points `ω_q` are distinct only for `2^d ≤ 2^64` (`ω_q` is the low word of `q`). The novel basis is a basis
  only for `L ≤ 40` columns (the range the executable's normalisers are checked on). `Level.Fits` carries those bounds,
  and every `fast100` level fits.
- **The theorem statements.** A proof of the wrong statement is worthless. The reviewer's job is §1 of this file,
  `Soundness.lean`, `Model/*` and `Accounting/Bound.lean`.
- **Pinned dependencies.** ArkLib `b2e456fc`; through its manifest, VCVio `d7089e46`, CompPoly `v4.34.0-patch2` and
  Mathlib `v4.34.0`. Lean 4.34.0. The definitions used from them (Reed–Solomon codes, `mcaError`, Johnson bounds) are
  part of what the statement means.
- **Everything around the verifier:** the session record's custody, the coin server, and the registered roots.

## 6. Zero knowledge (planned; not yet proved)

The masked protocol and the reference prover are in DESIGN.md §6, and the session model in DESIGN.md §11.

The verifier is malicious (two nations checking each other), so honest-verifier zero knowledge is not an acceptable
fallback anywhere.

**The theorem** (to prove): Goldreich–Kahan simulation of one batched session (§2). For every malicious verifier with
auxiliary input, a simulator produces a view indistinguishable from the verifier's view of one batched session, from the
tables' public statements alone, and runs in expected polynomial time. The simulator:

1. makes one extraction run, an honest prover on a dummy witness;
2. estimates the verifier's completion probability `p̃` from fresh runs only, never counting the extraction run;
3. rewinds only to just after the coin commitment, at most `t/p̃` times, where `t` grows with the security parameter;
4. uses a fresh dummy opening in every rewind.

- **It rests on** special honest-verifier ZK of masked Flock: a simulator for every fixed coin sequence, which is VEIL's
  notion and the first zero-knowledge lemma to prove. The lemma is statistical. Its only caveat is ChaCha20 standing in
  for uniform prover randomness. The simulations of a session's tables compose, because their masks are independent.
- **Serialized sessions** then compose sequentially (Goldreich–Oren 1994).
- **Reserve, if concurrency is ever needed:** a single registered verifier key, or a Prabhakaran–Rosen–Sahai preamble
  (DESIGN.md §11.2).

**Expected assumptions** (to be confirmed against M1's spec):

- **Uniform prover randomness.** The masks, padding and leaf salts are uniform and independent. In production they are
  ChaCha20 output under a seed from the operating system. Treating that output as uniform is the one caveat to the
  lemma's statistical zero knowledge, and the seed's quality is trusted.
- **Leaf hiding.** Halevi–Micali leaves are statistically hiding in the standard model, by the leftover hash lemma, with
  no random oracle (`N·2^-256` for N leaves with a fresh key, §7). They are binding under collision resistance of their
  hash, SHA-512 in the adopted variant.
- **Binding of the verifier's coin commitment** (§2: a SHA-512 Merkle root over Halevi–Micali coin commitments), under
  SHA-512 collision resistance. Against quantum verifiers it needs a collapse-binding commitment (DESIGN.md §9).
- **Uniformity of opened columns** of a padded Reed–Solomon codeword when the queries are at most the padding (the
  AHIV22 Lemma 4.15 condition). This is to be proved, not assumed.

## 7. One root across many proofs

The weights are committed once and every later proof opens that root. The analysis is DESIGN.md §10.

- **Soundness over N tables**, by union bound: `N·2^-195.5 + N·t·2^{17.3−256}`, plus `t²/2^513` once for the root. So
  2^-128 holds while `N·t ≤ 2^110.7`. This counts proofs as statements about committed values. **Decision, with
  Daniel:** state security per table, or as a lifetime budget.
- **Partial integrity counts audits, and needs knowledge soundness** (the lifetime-soundness desk study, §2–§5).
  - In a sampled audit only the first wrong sampled unit matters. So an audit costs one table's hash term plus one
    serving-tree term (DESIGN.md §1.2): about `t·2^-(222…238)` per audit.
  - Halevi–Micali leaves are statistically hiding, so plain soundness is vacuous there: a unit is wrong only relative
    to extracted values. Each audit therefore also pays the table's knowledge error and a link term (§1.3).
  - Under A2 that term is `t·N_s(8N₀/e)/2^256` (audit B: 2^-125.1 at `t = 2^80`; strict time 2^-69.8), and it
    dominates. So 2^-128 over the lifetime holds while `N·t ≤ 2^65.4` to `2^86.7`, with `N` counting audits.
- **Binding** must hold for the root's whole life. **Design change:** use the SHA-512 leaf and tree scheme, since
  `hm96-sha256/v1` binds only up to `t ≈ 2^64.5`. Rotate on evidence, with a proof that both roots open to the same
  weights.
- **Hiding** is fixed by the leaf count, not by the number of proofs. **Design change:** draw the root's key at
  commitment, and never reuse salts.
- **Zero knowledge** holds per batched, serialized session (§2, §6). Throughput comes from proving in parallel inside
  a session, not from concurrency.
- **Statement leakage** accumulates over sessions. **Policy needed:** which predicates are public, and how many.
- **Linkability** is intended, and it needs knowledge soundness (§1.3).

## 8. What remains (by a16z stage)

1. **Knowledge soundness, the rest** (§1.3). The table theorems are proved. Remaining:
   - the link theorem under A2 (DESIGN.md §3; `note:20260927T1510Z-draft-expected-time-link-plan`), which the
     sampled-audit composition needs;
   - the canonical `hm96` values, then `registered_weights`, which need the SHA-512 leaf layout in the model.
2. **Stage 2, refinement to `Flock.Verify`** (with lane `flock-verifier`): the executable verifier accepts only if the
   model accepts.
3. **Stages 1e and 3, zero knowledge and the reference prover** (§6), after M1's masking spec: first the special
   honest-verifier ZK lemma, then Goldreich–Kahan simulation of one batched session.
4. **Stage 1d, the lowering theorem**, which belongs to level 3 in another lane. Stage 1e then glues everything into
   one theorem from the verifier's acceptance to "the committed values satisfy the named circuit".

## 9. The audit layer (`Audit/`)

The sampled audit's guarantee, the integrity profile of `docs/audit-protocols.md` §2, over any Boolean circuit, any
partition, any draw law, and any per-unit proof system. The derivations and the deviations from that plan are in
DESIGN.md §12. Everything here is proved; the per-unit proof system enters only through named hypotheses.

**The model.** The prover registers `R` (its roots); the verifier draws units from its own randomness after registration;
one batched session proves the drawn units (`audit`). The **committed transcript** `X` is an analysis function of the
prover's state at registration (its registration and its session strategy for every draw), so the draw is independent of
its wrong units; nobody computes it. A unit is **wrong** in `X` when its committed outputs differ from its gates applied
to its committed inputs (`Partition.Correct`). Every bound holds for every prover strategy (`Game.prob … σ`), the form
`table_sound_exec` takes.

**Named hypotheses** (`Prop`s taken as hypotheses; no `axiom`):

| Hypothesis | Term | Meaning | To be discharged by |
|---|---|---|---|
| `Analysis.KnowledgeSound` | `ε_ks` | the session accepts while extraction of a drawn unit (picked before the session's first coin) fails | `table_knowledge_sound_joint` (§1.3), after Jensen over the prover's states |
| `Analysis.LinkSound` | `δ_link` | extraction succeeds but differs from `X` on the unit's committed inputs and outputs | the link theorem under A2, to write (DESIGN.md §3) |
| `Partition.AnchorsSound` | `δ_in` | the session accepts while the committed inputs differ from the anchors' values | `registered_weights` (§1.3); at the oracle layer, `InputsSound` (`oracle_anchorsSound`) |
| `Partition.InputsSound` | `δ_in` | oracle layer: the session accepts inputs other than the anchors' with probability at most `δ_in` | **proved for Flock** (`flock_inputs_sound`, `2^-195.4`), given `InputsLoweringSound` |
| `Partition.InputsLoweringSound` | — | inputs other than the anchors' leave some session table (the always-proved input statement) unsatisfied | the input unit's lowering (level 3) |
| `Partition.SessionSound` | `ε` | oracle layer only: a session with a drawn wrong unit accepts with probability at most `ε` | **proved for Flock** (`flock_session_sound`, `2^-195.4`), from `table_value_sound`, `execArith_correct` and the lowering |
| `Partition.LoweringSound` | — | a drawn wrong unit lies in a session table whose statement no committed table satisfies | the lowering theorem (level 3, another lane) |

Both per-unit hypotheses are averages over the whole audit, for every prover and every rule that picks a drawn unit
before the session's first coin, not bounds state by state.

**What the reviewer checks here.** The theorems hold for every choice of committed transcript `X` and extractor `ext`,
so their meaning rests on `X` being the committed values. At the oracle layer `X` is the registration itself
(`Partition.oracle`). At the compiled layer it is the plurality of what the extractor recovers behind each commit
string, which the instantiation defines and `LinkSound` ties to the extractor. A consumer who opens values under the
roots gets `X`'s values up to that opening's own binding term, which the audit does not count.

**Theorems, one-stage** (`Audit/OneStage.lean`), with $e_L(B)$ the probability that the draw misses $B$:

- `audit_le`: $\Pr[\text{accept} \wedge E(X, S)] \le \Pr_S[E(X, S) \wedge S \cap \mathrm{wrong}(X) = \emptyset] +
  \varepsilon_{ks} + \delta_{link}$ for every event $E$ on the committed transcript and the draw;
- `audit_profile`: $\Pr[\text{accept} \wedge \mathrm{wrong}(X) \in \mathcal B] \le \sup_{B \in \mathcal B} e_L(B) +
  \varepsilon_{ks} + \delta_{link}$;
- `audit_count`: $\delta(K) = \mathrm{miss}_L(K) + \varepsilon_{ks} + \delta_{link}$;
- `audit_drawn`: drawn units are correct up to $\varepsilon_{ks} + \delta_{link}$, even for a consumer who looks after
  the draw;
- `audits_seq`: audits in sequence fail with at most the sum of their bounds, each holding from every history. Nested,
  this is the lifetime union bound, with every term against a prover that took part in all the audits;
- `Partition.audit_cone`: for committed wires $T$ fixed before the draw, $\Pr[\text{accept} \wedge X_T \ne
  \mathrm{eval}_C(a)_T] \le \max_{u \in \mathrm{cone}(T)} (1 - \pi_u) + \varepsilon_{ks} + \delta_{link} + \delta_{in}$;
- `Partition.audit_cone_count`: at least $j$ wrong units in a cone;
- `Partition.audit_covered`: a cone inside the draw, for wires chosen even after the draw: $\varepsilon_{ks} +
  \delta_{link} + \delta_{in}$;
- `Partition.oracle_count`, `oracle_profile`, `oracle_drawn`, `oracle_cone`: the same at the oracle layer, from
  `SessionSound ε` and `InputsSound δ_in`, with $\delta_{link} = 0$.

**Theorems, laws** (`Audit/Law.lean`): `bernoulli_escape`, a Bernoulli($p$) draw's escape $(1-p)^{|B|}$; `subset_escape`, the uniform $k$-subset's escape
$\binom{n-|B|}{k}/\binom{n}{k}$; `subset_miss`, its count curve $\binom{n-K}{k}/\binom{n}{k}$; `subset_escape_le_prod`, the product bound $e(B) \le \prod_{u \in B}(1 - \pi_u) = (1 -
k/n)^{|B|}$, so inclusion probabilities alone give a consumer a valid bound; `subset_minimax`, **uniform is
optimal**: no law that proves $k$ units on average has a smaller count curve, at any $K$; `demo_count_590`, the demo's
point (1,024 heads, 16 drawn): 590 or more wrong heads escape with probability at most $2^{-20}$, by exact arithmetic.

**Theorems, the stratified law** (`Audit/Stratified.lean`, the default draw law; `docs/sampling-strategies.md` §4):
`Law.stratified σ k`, an independent uniform $k_s$-subset inside each stratum of $\sigma : \mathrm{Fin}\,n \to \mathrm{Fin}\,m$;
`stratified_escape`, the product of hypergeometrics $\prod_s \binom{n_s - m_s}{k_s}/\binom{n_s}{k_s}$ with $m_s = |B \cap U_s|$,
which is what `verity.proofs.profile.Stratified` computes; `stratified_escape_floor`, **a floor catches its whole
stratum**: with $k_s \ge 1$, a set containing $U_s$ never escapes; `audit_whole_stratum` and
`extraction_audit_whole_stratum`, the one-stage profile instantiated, so an audit accepts while a whole floored stratum is
wrong with probability at most the per-unit terms.

**Theorems, two-stage** (`Audit/TwoStage.lean`):

- `twoStage_profile`, variant (b), nested sampling: the profile over the coarse partition with the effective escape
  $e_{\text{eff}}(B) = \mathbb E_{S_1}\big[\prod_{u \in B \cap S_1} \max_{v \subseteq u}(1 - \pi_{L_2 u}(v))\big]$, with
  $\varepsilon_{ks}$ and $\delta_{link}$ at the fine level;
- `effEscape_full`, `twoStage_full_profile`: **(b1) is one-stage over the coarse partition** ($e_{\text{eff}} = e_{L_1}$);
- `effEscape_bernoulli_prod`: with a Bernoulli first stage, $e_{\text{eff}}(B) = \prod_{u \in B}\big(1 - p + p\,m_u\big)$ for
  any second stage, $m_u$ the largest second-stage miss probability inside $u$;
- `Partition.two_stage_a_coarse`: **(a) is one-stage over the fine partition**, and its curve bounds coarse counts;
- `Partition.Refines.exists_wrong_fine`: a wrong coarse unit has a wrong fine unit under every interior, the lemma
  behind (b).
- `Refines.oracle₂_profile`, `oracle₂_full_profile`: the same at the oracle layer, from a sound fine-level session
  (`SessionSound₂ ε`), with the link term zero. These are the plan's `two_stage_b_profile` and
  `two_stage_full_eq_one_stage` as stated there.

**Composition** (`Audit/Circuit.lean`): `compose` (every unit correct and the inputs anchored give the circuit's
evaluation on every committed wire) and `compose_cone` (the same for a consumer's wires, from its cone alone).

**Tightness and insecurity** (`Audit/Examples.lean`, on `K` chains of `nv` NOT gates, one coarse unit per chain and one
fine unit per gate, with a uniform `k₂`-subset of each drawn chain's gates in the second stage and an ideal session):

- `two_stage_b_tight`: for every first-stage law, a prover whose chains are all wrong is accepted with probability
  exactly `twoStage_count`'s bound, $\mathbb E_{S_1}[q^{|S_1|}]$ with $q = \binom{n_v - 1}{k_2}/\binom{n_v}{k_2} = 1 -
  k_2/n_v$;
- `naive_two_level_unsound`: if the second stage leaves a gate undrawn and the first draws anything, that exceeds
  $\mathrm{miss}_{L_1}(K)$, the bound that one-stage over the coarse partition (and the `m`-dependent reading of
  `TwoStageLaw.profile` at $m = n_v$) would give. Nested sampling dilutes;
- `late_interior_insecure`: with the interior committed after the fine draw (`twoStageLate`), a wrong chain is accepted
  with probability 1.

- `effEscape_bernoulli`: **(b2) is one-stage over the coarse partition, thinned.** With a Bernoulli($p$) first stage,
  the effective escape of any set of chains is its escape under the one-stage Bernoulli($p\,k_2/n_v$) draw,
  $(1 - p\,k_2/n_v)^{|B|}$ (`Law.bernoulli_escape`, `Law.bernoulli_avg_prod`).

**The Flock instantiation, oracle layer** (`Audit/Flock.lean`, the only audit file that imports Flock's theorems):

- `value_batch_le`: in a batched session, whose tables run concurrently on a schedule the prover chooses, every table
  accepting is at most as likely as any one table accepting (`Game.value_interleave_le`, table by table);
- `FlockTable.value_le`: one table, with the executable's arithmetic, at most `2^-195.4` for `22 ≤ m ≤ 35`;
- `flock_session_sound`: Flock's batched session is `SessionSound` at `2^-195.4`, given `LoweringSound`;
- `flock_inputs_sound`: the input unit, likewise, given `InputsLoweringSound`;
- `flock_oracle_count`, `flock_oracle_count_subset`: the audit's count curve, $\delta(K) = \mathrm{miss}_L(K) + 2^{-195.4}$,
  and $\binom{n-K}{k}/\binom{n}{k} + 2^{-195.4}$ for the uniform $k$-subset;
- `flock_oracle_drawn`: drawn units correct up to $2^{-195.4}$;
- `flock_oracle_cone`: a consumer's wires $T$, $\max_{u \in \mathrm{cone}(T)}(1 - \pi_u) + 2 \cdot 2^{-195.4}$.

- `flock_two_stage_sound`, `flock_two_stage_full_profile`: **two-stage (b1) on Flock** is one-stage over the coarse
  partition, $\sup_{B \in \mathcal B} e_{L_1}(B) + 2^{-195.4}$, given the fine partition's lowering.

The hypotheses are BCHKS25 Theorem 4.6 and the lowerings.

**The compiled layer** (`Audit/Extraction.lean`, `Audit/FlockCompiled.lean`, `Audit/FlockBatched.lean`; DESIGN.md
§12):

- `extraction_audit_le`, `extraction_audit_count`, `extraction_audit_drawn`: the one-stage audit with a **randomized
  extractor**. Per drawn unit, `Pr[accept ∧ extraction fails]` and `Pr[accept ∧ extraction disagrees]` are joint
  probabilities over the fresh run and the extractor's reruns (`ExtractionAnalysis`), and the named bounds may depend
  on the prover's state. A deterministic extractor is the special case (`Analysis.toExtraction`, with
  `KnowledgeSound.toExtraction` and `LinkSound.toExtraction`);
- `flock_batched_knowledgeSound`: **ε_ks is discharged** for the compiled Flock audit in the protocol's batched session
  of several tables (`batchedSession`, `sessionB`), with the leaf layer registered. Each drawn unit is read from one
  table of the plan (`tab`). The knowledge term is, after each draw, the largest over the plan's tables of the table's
  $\varepsilon_c^-(\sigma) + K\,\mathrm{Adv}_0(\sigma) + N_0/(eK)$ computed in the session (`joint_le_B'`, from
  `session_knowledge_sound`), averaged over the draw (`ksAvgB`);
- `flock_compiled_knowledgeSound`: the same for one table on its own (`sessionC`), from
  `table_knowledge_sound_joint_tight`: $\mathbb E_\omega[\varepsilon_c^-(\sigma_\omega) + K\,\mathrm{Adv}_0(\sigma_\omega)
  + N_0/(eK)]$, which is $\varepsilon_c^- + t\cdot2^{-244.7}$ at the best $K$ under the generic collision bound. Both
  have BCHKS25 Theorem 4.6 as their only named assumption;
- `flock_batched_count`, `flock_batched_drawn` (and `flock_compiled_count`, `flock_compiled_drawn`):
  $\delta(K) = \mathrm{miss}_L(K) + \varepsilon_{ks}(\sigma) + \delta_{link}(\sigma)$, and drawn units up to
  $\varepsilon_{ks}(\sigma) + \delta_{link}(\sigma)$.

**The audit's circuit `C`** is abstract in these theorems. For a program it is the rows the verifier parses
(`Prog.circuit`, `Lowering.lean`, §1.5): each unit an instance of its template's `Rows`, one `Op.row` gate per
computed row. "Each template's rows compute its gates" in the program's Boolean circuit is L1 (§1.5), named and
tested. The decoder is **per drawn unit**: two drawn units that share a wire hold it in their own columns, and only
the link makes the readings agree.

| Named at the compiled layer | Meaning | Discharged by |
|---|---|---|
| `ExtractionAnalysis.LinkSound` (`analysisB`, `analysisC`) | `δ_link`: extraction succeeds, but every close satisfying candidate decodes, on the unit's committed wires, to values other than the committed transcript's | the link theorem under A2 (flock-soundness, planned), for an analysis whose extractor reruns until `k` reruns accept, in place of `Kr` reruns |
| `Partition.LoweringSoundB`, `Partition.LoweringSoundC` | a message packing a witness that satisfies the statement of a drawn unit's table decodes, for that unit, to values on which it is correct | `loweringSoundB_of_place`, `loweringSoundC_of_place` from each drawn unit's `UnitPlace` (`lowering_sound`, §1.5), given the statement builder's `Placement` (level 3) and L1 |

**Not yet:** the serving-tree term when the leaf layer is not registered; `LinkSound` itself; `Placement` for the
executable's statements.
