# Flock interactive soundness in Lean: what is proved

For the expert reviewer and for Table 1. This file is the list of what is proved. `ASSUMPTIONS.md`, beside it, states
what the theorems assume: every named assumption and hypothesis, in full, with the detail behind each in `assumptions/`.
The audit layer's theorems are in `FlockSoundness/Audit/README.md`. `DESIGN.md` holds the derivations: the hash-width
analysis, the compiled proof, the knowledge extractor, the spec deviations, the zero-knowledge plan, the a16z staging,
and the post-quantum and root-reuse analyses.

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

- **The radius** is `1 − √ρ − η` with the analysis slack `η = 1/200` (`Accounting.eta`), 0.2879 at rate 1/2. It was
  `1/50` (0.2729) until Sep 27; the protocol and queries are unchanged.
- **`tableError`** is a Lean function of the parameters (`Accounting/Bound.lean`), with one term per verifier coin. It
  is at most **2^-205 for 22 ≤ m ≤ 35**, proved in `Accounting/Numbers.lean`.
- **The statement shapes covered** have at most 2^27 witness bits per block, 1,024 public regions and 64 link-point
  coordinates. That is every statement of §16, with room to spare.
- **The link points' layout** is a named hypothesis (`Statement.LinkLayout`, `ASSUMPTIONS.md`).
- **At this layer committed tables are sent whole.** The theorem is statistical, so it holds against quantum provers
  too.
- **For the executable's arithmetic** (`Instance.lean`), `table_sound_exec` and its two `fast100` corollaries take
  `Arith.Correct` from `execArith_correct`, so they have no named assumption.
- **For the optimal prover** (`table_value_sound`): the same bound on `value`, which `table_sound` specializes to each
  strategy. This form composes across the tables of a batched session (`value_batch_le`, `FlockSoundness/Audit/README.md`).
- **With M1's padded level 0** (`table_sound_pad`, `SoundnessPad.lean`; the model is `Model/LigeritoPad.lean`). M1 (#123,
  `backends/flock/live/PROTOCOL.md` §7) pads every level-0 lane with `t` coefficients, so level 0's code is
  `RS[2^d, 2^cols + t]`, of rate `ρ' = (2^cols + t)/2^d`. Each rep also folds its own `e` extra lanes with fresh coins.
  - **The bound is a function of `t` and `e`** (`padTableError`). It is `tableError` with level 0's radius, list size,
    queries and MCA numerator taken at `ρ'`, plus one MCA line and one coin on the restarted claim per extra lane.
  - **At #123's values** (`8d621454`): `q₀ = 277, 242, 229` at `m = 25, 26, 27`, `t = 2q₀`, and a pair of extra lanes per
    rep. The bound is at most **2^-205 per table** (`table_sound_pad_m1`).
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
- **Under the generic bound** this is at most `2^-205 + t·2^{17.3−256} + t²/2^511` for a prover making `t` hash
  evaluations. That is at most 2^-128 up to `t ≈ 2^110.7` (m = 35) to `2^117` (m = 22).
- **No random oracle and no knowledge assumption** is used.

### 1.3 Knowledge soundness (`table_knowledge_sound`, `table_knowledge_sound_joint`; `registered_weights` to write)

It is needed for any claim about which data a proof used ("these proofs used the registered weights"), and for the
recursion track, whose outer proof must be a proof of knowledge (DESIGN.md §3).

**It is required for the partial-integrity composition of sampled audits** (§4). Halevi–Micali leaves are statistically
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
- **The expected-time extractor** (DESIGN.md §3) uses `k` reruns in which the session accepts:
  `ε·Pr[fails] ≤ ε_c⁻ + N₀ε/(ek) + √(k·Adv₀)` (`joint_le_accB`). `analysisBE`'s `cover` absorbs the middle term.
- **The link theorem** (`flock_batched_linkSoundE`, DESIGN.md §3): under A2 and `ValueBinding` (named), `δ_link` is
  at most `Q_s(2(1+k)t'/2^256 + …)/(1 − ρ)`, `Q_s ≤ N_s`. It dominates the audit's hash term.
- **Not yet:** the canonical HM96 values and `registered_weights`. Both need the SHA-512 leaf layout in the model,
  which also discharges `ValueBinding`.
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
| The padded bound at #123's values: 2^-205 per table at m = 25–27 | 1a | `table_sound_pad_m1` (`Accounting/PaddedNumbers.lean`) | proved |
| Merkle binding: two verified openings of one position with different columns give a SHA-512 collision | 1b | `Merkle.opening_binding` | proved |
| Rewinding: openings off the plurality table cost collisions, `#bad² ≤ N·Q·#conflict` | 1b | `Rewinding.bad_sq_le`, `prBad_le` | proved |
| Compilation step: the oracle bound plus the rewinding term, for a table committed before any coin or after a prefix | 1c | `Compiled.compile_step`, `compile_step_prefix` | proved |
| Session binding: the coin server keeps each round's bytes, so they are fixed before the round's coins | 1c | the `Game` model (`send` before `coin`) | by construction; no hash |
| The compiled table theorem (§1.2): the compiled game (caps, openings at the end of the proof), the lockstep coupling with the plurality prover, rewinding at every cap | 1c, interactive | `table_sound_compiled` (`Model/Compiled.lean`, `Game/Lock.lean`, `CompiledLock.lean`, `CompiledSound.lean`) | proved |
| Knowledge soundness (§1.3): extract the committed witness, then Halevi–Micali binding to registered data | 1c, 1e | `table_knowledge_sound`, `table_knowledge_sound_joint` (`Knowledge.lean`); `registered_weights` | the table theorems and the link theorem (A2) proved; `registered_weights` **to write**; required for sampled audits (§4) |
| The block R1CS means the named circuit (the lowering theorem) | 1d | level 3, another lane; its generic step `Lowering.lean` (§1.5) | the generic step proved; the placement and L1 remain |
| The executable Lean verifier refines the model | 2 | with `flock-verifier` | **to write** |
| Reference prover: completeness and zero knowledge | 3, 1e | after M1 (§3) | planned |
| Sampled audits: the integrity profile over any partition and draw law, two-stage and its relations, minimax, tightness and insecurity (`FlockSoundness/Audit/README.md`) | (composition) | `Audit/*` | proved; per-unit terms named |
| The Flock audit at the oracle layer: the batched session's projection to one table, `2^-205` (the audit layer) | 1a, composition | `Audit/Flock.lean`, `flock_session_sound` | proved, given the lowering |
| The compiled lowering: the audit's circuit is the rows the verifier parses, and a satisfied table makes every placed unit correct, for every template (§1.5) | 1a | `Lowering.lean`, `lowering_sound`, `Prog.isRowsUnit` | proved, given the placement |
| L1: each template's rows compute its gates (`ASSUMPTIONS.md`) | (meaning) | named | evidence: the IR comparison tests; discharge by a verified lowering |

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
  (the slots with Δ XORed in, spec §16.1) is part of the level-3 refinement (`assumptions/trusted-components.md`).
- **L1, each template's rows compute its gates,** is a named hypothesis (`ASSUMPTIONS.md`).

### 1.6 Proved from the libraries, not assumed

**A1, Reed–Solomon mutual correlated agreement up to the Johnson bound, is proved** (Sep 27). BCHKS25 (ePrint
2025/2055) Theorem 4.6 as printed, which ArkLib admits, is `mcaError_le_bchks25` (`JohnsonMCA.lean`) on `Fin n`
domains. The proof uses Dao–Kominers–Thaler (DKT26, ePrint 2026/2056), proved in ArkLib at our pin
(`ReedSolomon.mcaError_affineLine_johnson_le`, #907). With `ρ = (k − 1)/n` and `η = 1 − √ρ − δ`, its count is below
`(8/3)·n·t'³/ρ`, where `t' = max(⌈√ρ/(2η)⌉, 3) + ½ ≤ m + ½` for BCHKS25's `m = max(⌈√ρ/η⌉, 3)`, so below BCHKS25's
numerator `a` (`Level.mcaA`); for `k ≥ n` the code is every word. It bounds every Ligerito fold's coin (spec §13.3,
`prCoin_mca_le`). The accounting keeps `a`, so no number changes: DKT26's count is `2^8.2` to `2^12.3` smaller, on a
term below 2^-170.

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
  not to one witness. The list has at most 200 entries at rate 1/2 (ArkLib's Johnson bound, `1/(2ηρ)` with
  `η = 1/200`; PROTOCOL.md §15's 0.02 is a diagnostic).
  - "The committed values satisfy the circuit" means that some codeword of that list packs a satisfying witness
    (`Model.CommittedSatisfies`), which is exactly right for soundness.
  - Claims that use the committed values elsewhere (registered data, links to other proofs, recursion) need them
    extracted (§1.3).
- **The circuit.** The theorem is about the block R1CS the statement lowers to (spec §16.1), with the pinned constant
  column and the public regions. That this R1CS means the named circuit is the lowering theorem (level 3). Its generic
  step, for every template, is proved here (§1.5). The placement for the executable's statement and L1 remain.

## 3. Zero knowledge (planned; not yet proved)

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
  no random oracle (`N·2^-256` for N leaves with a fresh key, §4). They are binding under collision resistance of their
  hash, SHA-512 in the adopted variant.
- **Binding of the verifier's coin commitment** (§2: a SHA-512 Merkle root over Halevi–Micali coin commitments), under
  SHA-512 collision resistance. Against quantum verifiers it needs a collapse-binding commitment (DESIGN.md §9).
- **Uniformity of opened columns** of a padded Reed–Solomon codeword when the queries are at most the padding (the
  AHIV22 Lemma 4.15 condition). This is to be proved, not assumed.

## 4. One root across many proofs

The weights are committed once and every later proof opens that root. The analysis is DESIGN.md §10.

- **Soundness over N tables**, by union bound: `N·2^-205 + N·t·2^{17.3−256}`, plus `t²/2^513` once for the root. So
  2^-128 holds while `N·t ≤ 2^110.7`. This counts proofs as statements about committed values. **Decision, with
  Daniel:** state security per table, or as a lifetime budget.
- **Partial integrity counts audits, and needs knowledge soundness** (the lifetime-soundness desk study, §2–§5).
  - In a sampled audit only the first wrong sampled unit matters. So an audit costs one table's hash term plus one
    serving-tree term (DESIGN.md §1.2): about `t·2^-(222…238)` per audit.
  - Halevi–Micali leaves are statistically hiding, so plain soundness is vacuous there: a unit is wrong only relative
    to extracted values. Each audit therefore also pays the table's knowledge error and a link term (§1.3).
  - Under A2 that term is `t·Q_s(8N₀/e)/2^256` (audit B: 2^-129.4 at `t = 2^80`; strict time 2^-69.8), and it
    dominates. So 2^-128 over the lifetime holds while `N·t ≤ 2^78.4` to `2^90.4`, with `N` counting audits.
- **Binding** must hold for the root's whole life. **Design change:** use the SHA-512 leaf and tree scheme, since
  `hm96-sha256/v1` binds only up to `t ≈ 2^64.5`. Rotate on evidence, with a proof that both roots open to the same
  weights.
- **Hiding** is fixed by the leaf count, not by the number of proofs. **Design change:** draw the root's key at
  commitment, and never reuse salts.
- **Zero knowledge** holds per batched, serialized session (§2, §3). Throughput comes from proving in parallel inside
  a session, not from concurrency.
- **Statement leakage** accumulates over sessions. **Policy needed:** which predicates are public, and how many.
- **Linkability** is intended, and it needs knowledge soundness (§1.3).

## 5. What remains (by a16z stage)

1. **Knowledge soundness, the rest** (§1.3). The table theorems and the link theorem are proved. Remaining: the
   canonical `hm96` values, then `registered_weights`, which need the SHA-512 leaf layout (it discharges `ValueBinding`).
2. **Stage 2, refinement to `Flock.Verify`** (with lane `flock-verifier`): the executable verifier accepts only if the
   model accepts.
3. **Stages 1e and 3, zero knowledge and the reference prover** (§3), after M1's masking spec: first the special
   honest-verifier ZK lemma, then Goldreich–Kahan simulation of one batched session.
4. **Stage 1d, the lowering theorem**, which belongs to level 3 in another lane. Stage 1e then glues everything into
   one theorem from the verifier's acceptance to "the committed values satisfy the named circuit".
