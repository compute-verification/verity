# Flock interactive soundness in Lean: what is proved

For the expert reviewer and for Table 1. This file is the list of what is proved. `ASSUMPTIONS.md`, beside it, states
what the theorems assume: every named assumption and hypothesis, in full, with the detail behind each in `assumption-notes/`.
The audit layer's theorems are in `Audit/README.md`. `DESIGN.md` holds the derivations: the hash-width
analysis, the compiled proof, the knowledge extractor, the spec deviations, the zero-knowledge plan, the a16z staging,
and the post-quantum and root-reuse analyses.

The interactive mode only. The Lean development is this directory (namespace `FlockSoundness`). The verifier it is about
is specified in `../../../../verity/ml/flock/verifier/PROTOCOL.md`. The work is staged along a16z's zkVM security stages (DESIGN.md §7).

## 1. Theorems

### 1.1 Soundness at the oracle layer (`table_sound`, `table_sound_fast100`, `table_sound_fast100_34_35`)

Consider one table of a session, run as the verifier of record runs it: the `fast100` profile, run twice on one committed
root, with live coins, for any statement of PROTOCOL.md §16. For every prover strategy, the probability of this event is
at most `tableError(fast100 m, shape)`:

> the verifier accepts both reps, while no codeword within the level-0 proximity radius of the committed table packs a
> witness that satisfies the statement.

- **The radius** is `1 − √ρ − η` with the analysis slack `η = 1/200` (`Accounting.eta`), 0.2879 at rate 1/2.
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
  strategy. This form composes across the tables of a batched session (`value_batch_le`, `Audit/README.md`).
- **With M1's padded level 0** (`table_sound_pad`, `SoundnessPad.lean`; the model is `Definitions/Flock/LigeritoPad.lean`). M1
  (`verity/ml/flock/live/PROTOCOL.md` §7) pads every level-0 lane with `t` coefficients, so level 0's code is
  `RS[2^d, 2^cols + t]`, of rate `ρ' = (2^cols + t)/2^d`. Each rep also folds its own `e` extra lanes with fresh coins.
  - **The bound is a function of `t` and `e`** (`padTableError`). It is `tableError` with level 0's radius, list size,
    queries and MCA numerator taken at `ρ'`, plus one MCA line and one coin on the restarted claim per extra lane.
  - **At C-Flock ZK's values** (§7 item 6, `Zk.queries0`): `q₀ = 277, 242, 229` at `m = 25, 26, 27`, 219 at `m = 28..31`
    and 218 at `m = 32..35`, `t = 2q₀`, and a pair of extra lanes per rep. At every `m` the verifier serves the bound is
    at most **2^-205 per table** (`table_sound_pad_m1`).
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

### 1.3 Knowledge soundness (`table_knowledge_sound`, `table_knowledge_sound_joint`; `registered_weights`)

It is needed for any claim about which data a proof used ("these proofs used the registered weights"), and for the
recursion track, whose outer proof must be a proof of knowledge (DESIGN.md §3).

**It is required for the partial-integrity composition of sampled audits** (§4). Halevi–Micali leaves are statistically
hiding, so every commit string opens to essentially every value, and plain soundness says nothing about which values a
unit used. The audit bound defines a faulty proof block through extracted values, so it pays the table's knowledge error and a
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
    points are one sample, and then every table runs concurrently in any order. The rest of the session is arbitrary.
  - The extractor reruns everything after `Commit`.
  - `ε_c⁻` and `Adv₀` are the table's terms computed in the session (`epsS`, `adv₀S`).
  - It rests on `session_sound_of_table`, the compiled bound for any level-0 table in the session. The oracle
    prover plays the whole session there, and the table's value bounds it (`orcSess_value_le`).
- **Several tables in one session** (`SessionBatch.lean`). `sessionB` is the protocol's batch: `Commit` with every
  level-0 cap, one link-point sample, then every table concurrently.
  - For each table `j`, `joint_le_B'` bounds `Pr[the session accepts] · Pr[table j's extraction fails]` by table `j`'s
    `ε_c⁻ + K·Adv₀ + N₀/(eK)` in the session (`ksBoundB`).
  - `one_le_fail_add_B'` is the audit's `cover`, for any link event that commitment implies.
- **The expected-time extractor** (DESIGN.md §3) uses `k` reruns in which the session accepts:
  `ε·Pr[fails] ≤ ε_c⁻ + N₀ε/(ek) + √(k·Adv₀)` (`joint_le_accB`). `analysisBE`'s `cover` absorbs the middle term.
- **The link theorem** (`flock_batched_linkSoundE`, DESIGN.md §3): given `ValueBinding` (named), at each prover whose
  link finders satisfy A2 (`LinkCR`), `δ_link` is at most `Q_s(2(1+k)t'/2^256 + …)/(1 − ρ)`, `Q_s ≤ N_s`; at any other
  prover the bound is `⊤` (`linkBoundCR`). A2 is not asked of every prover at once: a prover can hard-code a collision.
  It dominates the audit's hash term.
- **The value binding** (`Binding/`). `ValueBinding` is built from the serving rows' `hm96-sha512/row/v1` leaves
  (`HmRows.binding`; a bit row's `/row/v2` leaves through `HmRows.v2`, whose `enc_inj` is `FlockLevel3.RowV2.enc_inj`):
  a commit string's commitment is its leaf `SHA-512(leaf_prefix ‖ b ‖ c)` under the executable's
  key, and `collide` is an explicit extractor (`ExplicitLeaf.hm96`, `hm96Pair_spec`): two different rows with one leaf
  give a SHA-512 collision of the outer hash, of the salt digest, or of the rows' own digests. What `ValueBinding`
  carried becomes `Assumptions.HmRowComputes` (the circuit's row hashing computes the commit string, phase 2i) and the
  layout's facts (`Layout`, `HmRows`). `flock_e2e_count_hm96`, `flock_e2e_drawn_hm96` and their `_exec_hm96` forms are
  the end-to-end theorems with that binding. The headline, `Prog.flock_headline` (`Headline.lean`), is the count curve
  with that binding on the statement's rows' circuit at the live sample under A3 (`ASSUMPTIONS.md`, The headline). Each
  is proved under the assumptions its footprint lists, every hypothesis with its kind: hardness (`cr/sha-512`,
  `ecr/sha-512`), platform (A3, and A6, the coin server's OS randomness), model, intended to prove (`hHm`, the placements
  `pp` or `dp`, `lay`, `rs`, `tr`, `hConst`, `hZero`; `assumption-notes/e2e-checklist.md`), or a numeric condition.
- **Over what the verifier accepts** (`Discharge/FailClosed`). `flock-verify verify` refuses every statement outside the
  forms the proof covers (the verifier's PROTOCOL.md §16.15; `FailClosed.scopeOk_eq`), so the executable's headlines
  (`zk_session_sound` with its `J`, hidden, custody and composed forms, and `flock_headline_exec` with its hidden and
  decoded forms) are restated over `Verifies` (accepted, and in scope) with no hypothesis on the circuit, the public file
  or the statement's form. The old forms, which take those hypotheses, are the lemmas they apply. The zero-knowledge
  views take the statement's avoidance of the mask slot from the setup (`Mask.lean`): `verify --zk` refuses a statement
  whose Δ or regions meet it (`Zk.maskFree`). `FailClosed.flock_verify_sound` (`One.lean`) is the one theorem: at every statement, with or without `--zk`, the
  headline of its case (public or hidden outputs; under `--zk`, one table or several).
- **`registered_weights`** (`Binding/Weights.lean`). At each of a sampled proof block's wires, the row a satisfying witness of
  its table holds is the registered row (any opening of the registered leaf: the weights and salt the registrant
  committed to), or `collide` of the two openings is a SHA-512 collision (`Layout.registered_weights`; from
  `HmRowComputes` for rows, `HmRows.registered_weights_hm96`; for the message the audit's extractor settles on,
  `registered_weights_extracted`).
- **Registered reads** (`Registered/`; PROTOCOL.md §16.10). A row a session reads from a value registered before opens
  the registered root the verifier holds, at the position the verifier's program reads (`Flock.Registered.check_ok`, in
  the verifier's package). Two different leaves the verifier accepts at one position under one root give an explicit
  SHA-512 collision (`TwoOpenings.collides`, from `Flock.Registered.opensLeaf_binding`), at most `q²/2^513` likely under
  `cr/sha-512` (`TwoOpenings.conflict_strict`). Where the check holds, a registered read's leaf is the registrant's, and
  the row a satisfying witness holds there is the row the registrant committed, or SHA-512 collides (`Reads.regLeaf_eq`,
  `Reads.registered_weights`). `flock_e2e_drawn_hm96_reads` and `flock_e2e_count_hm96_reads` add to the end-to-end
  bounds the event that a registered read's leaf under the prover's registration is not the registrant's, at
  `δ_reg = qR²/2^513` (`Reads.prob_off_le`). With `--registered F` at hidden outputs, sampled proof blocks' reads are
  checked: `SoundZHJR` (`--zk`), `SoundExecHR` (plain, `Discharge/PlainReg/`).

### 1.4 The pieces, by stage

| Piece | a16z stage | Lean | Status |
|---|---|---|---|
| Interaction model: live coins, prover strategies, acceptance probability, optimal prover | (framework) | `Game/Basic.lean` | proved |
| A live coin hits a fixed nonzero polynomial's zero with probability at most degree/|F| (Schwartz–Zippel) | (framework) | `Game/Poly.lean` | proved |
| Two reps on independent coins, any interleaving: the error squares | 1a | `Game/Interleave.lean`, `value_interleave_le` | proved |
| The bound as a function of the parameters, and its value for every `m` | 1a | `Accounting/*` | proved |
| The verifier, following PROTOCOL.md §9–§16 | 1a | `Model/*`, `Definitions/Flock/` | written (definitions) |
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
| The padded bound at C-Flock ZK's values: 2^-205 per table at m = 25–35 | 1a | `table_sound_pad_m1` (`Accounting/PaddedNumbers.lean`) | proved |
| Merkle binding: two verified openings of one position with different columns give a SHA-512 collision | 1b | `Merkle.opening_binding` | proved |
| Rewinding: openings off the plurality table cost collisions, `#bad² ≤ N·Q·#conflict` | 1b | `Rewinding.bad_sq_le`, `prBad_le` | proved |
| Compilation step: the oracle bound plus the rewinding term, for a table committed before any coin or after a prefix | 1c | `Compiled.compile_step`, `compile_step_prefix` | proved |
| Session binding: the coin server keeps each round's bytes, so they are fixed before the round's coins | 1c | the `Game` model (`send` before `coin`) | by construction; no hash |
| The compiled table theorem (§1.2): the compiled game (caps, openings at the end of the proof), the lockstep coupling with the plurality prover, rewinding at every cap | 1c, interactive | `table_sound_compiled` (`Model/Compiled.lean`, `Game/Lock.lean`, `CompiledLock.lean`, `CompiledSound.lean`; the definitions in `Definitions/Flock/Compiled.lean` and `Definitions/Flock/Game/Lock.lean`) | proved |
| Knowledge soundness (§1.3): extract the committed witness, then Halevi–Micali binding to registered data | 1c, 1e | `table_knowledge_sound`, `table_knowledge_sound_joint` (`Knowledge.lean`); `registered_weights` | the table theorems and the link theorem (A2) proved; `registered_weights` proved (`Binding/`) for rows under `HmRowComputes` (named, phase 2i) and the layout's facts; required for sampled audits (§4) |
| The block R1CS means the named circuit (the lowering theorem) | 1d | level 3; its generic step `Lowering.lean`, the placement `ExecSetup.lean`, the derived rows' placement `Compose.lean` (§1.5) | the generic step and the placement from `Stmt.setupH` proved; for derived rows, the placement from the block matrices proved; the link from `setupH`'s unit net to the derived rows remains |
| Zero knowledge of the masked model (§3): Lemma B per table and per session, Lemma A at every coin vector (the prover's refusals in the code's order), Lemma C against an adaptive verifier with its own tape | 1e | `table_shvzk_hm96`, `session_shvzk_hm96`, `session_prefinal_indep_cut`, `adaptive_prefinal_hm96_tape` (`ZK/`) | proved for the model `ZK/Table.lean`; Theorem GK over a model of the session's views, `gk_simulate_hm96` (`ZK/GK/`); the link to the code remains |
| Sampled audits: the integrity profile over any partition and sampling design, two-stage and its relations, minimax, tightness and insecurity (`Audit/README.md`) | (composition) | `Audit/*` | proved; per-unit terms named |
| The Flock audit at the oracle layer: the batched session's projection to one table, `2^-205` (the audit layer) | 1a, composition | `Audit/Flock.lean`, `flock_session_sound` | proved, given the lowering |
| The compiled lowering: the audit's circuit is the rows the verifier parses, and a satisfied table makes every placed unit correct, for every template (§1.5) | 1a | `Lowering.lean`, `lowering_sound`, `Prog.isRowsUnit`; `placement_of_setupH` | proved; the placement from the executable's accept step |
| The certificate: rows that are `GateRows.rows G` compute `G`, so a correct unit computes the definition `G` lowers (`ASSUMPTIONS.md`; no hypothesis, since the circuit is the statement's rows) | (meaning) | `IsRowsUnit.computes_of_cert` (`Certificate.lean`) | proved; the premises per template: the IR comparison tests on samples, and the typed layout's proof (`Types.Dag.layout_sound`, `TmplValue.defComputes_of_layout`) once a template's rows are a checked layout's (`assumption-notes/template-rows.md`) |

The proved results depend only on Lean's three standard axioms (`propext`, `Classical.choice`, `Quot.sound`). Run
`python tools/verity/lean/check.py Security/Proofs` to check every declaration (`lean-audit.json`).

### 1.5 The compiled lowering (`lowering_sound`, `Prog.isRowsUnit`; `Lowering.lean`)

- **The audit's circuit is the rows the verifier parses** (spec §16.5, `Flock/Net.lean`).
  - A template's unit is its input columns, its computed rows and its constant (`Rows`).
  - A computed row is `z_i = (⊕_{c∈a_i} z_c)·(⊕_{c∈b_i} z_c)`, reading earlier columns and the constant. It is one gate
    of the circuit (`Audit.Op.row`).
  - A program's circuit (`Prog.circuit`) is its units' rows, with each unit's input columns and constant wired to
    earlier gates. Several input columns may be wired to one gate (a unit reading one source on several inputs), and
    the constant to a gate no input column is wired to. Every unit is an instance of its template's rows
    (`Prog.isRowsUnit`).
  - **A pinned file read as `Rows`.**
    - Its inputs are its self rows (`A = B = [i]`).
    - Its padding columns (`A = B = 0` in the input region) are forced zero and read by no row, so they are not columns
      of `Rows`.
    - Its computed rows lie between the input region and the constant, its last column.
  - The parser checks the input rows, the constant and the row order (`Net.checkOrder`): every computed
    row reads only earlier columns or the constant, and never a padding column. So `Rows.topo` for a parsed net is
    proved, not assumed (`ExecRows.lean`):
    - `parse_rowOrder`: `Net.parse` accepted means the check passed on the very rows stored as `A` and `B`;
    - `ofRows_row`: the stored sparse rows read back as those rows (columns below `2^32`);
    - `Rows.ofNet`: the audit's `Rows` for a net `Net.ofRows` built (`Built`: a parsed net, or a typed flat class's
      `Typed.netOfD`), whose computed rows are `net.a.row` and `net.b.row`.

    `test_pinned_rows_read_earlier_columns` stays as a check on the pinned files.
  - **An instance's rows** (`Rows.stack`): a template instance is its VU's `upv` unit slots, so its `Rows` are `upv`
    copies of the net's, stacked in `Rows`' order. Each copy's constant is a computed row reading the shared constant,
    and precedes the copy's rows, which read it as their net rows read their slot's constant.
  - **Placing an instance** (`placement_stack`): `Placement` for the stacked rows at `stackPlace` (copy `u` at slot
    base `base u`, the shared constant at the pin) follows from three facts about the level-0 matrices: at each copy's
    computed rows they are the net's rows shifted to the slot, and each copy's constant row and the pin's own row read
    the pin. Those facts are what the verifier's matrices (L3-A's `placedA`/`placedB`) and Δ give, next.
  - **Δ's rows** (`delta_rows`, `ExecDelta.lean`): every row `HmRow.delta` writes is a slot constant other than the pin,
    or an input-port bit of a net slot (a row's first `cv`, an hm96 `pad`, a unit's leaf or leaf-cut bit, a wire's
    destination). That none is a unit's computed row follows from the layout checks (`delta_split`, `deltaIn_inBit`).
- **The theorem** (`lowering_sound`, for every template). Suppose a witness satisfies a statement in which a unit's rows
  are placed in a block. Then the trace decoded from that block makes the unit correct.
  - Placed (`Placement`) means two things:
    - the block matrices' rows at the unit's computed columns read exactly its rows' columns;
    - the rows at its constant read the pin, as every slot's constant is wired to it (spec §16.1).

    The decoded trace then carries `true` on the constant (`UnitPlace.decode_one`).
  - A unit's input columns on one gate must carry one value in every satisfying witness (`UnitPlace.aliased`), as
    they do when they copy one source's position, or forced-zero rows. The trace reads each gate off one of its
    columns.
  - `UnitPlace.correct` gives the audit's lowering hypothesis with one decoder per sampled proof block. Two sampled proof blocks that
    share a wire hold it in different columns. The link, not the lowering, makes them agree.
- **The placement, from the executable's accept step** (`placement_of_setupH`, `ExecSetup.lean`). When `Stmt.setupH`
  accepts, VU `g`'s stacked unit slots are placed at `ur.slot (g·upv + u)`, the shared constant at the pin, in the model
  statement of the matrices the verifier folds (flock-verifier's `placedA`/`placedB`; `Layout.model`).
  `unitPlace_of_setupH` is the `UnitPlace` that `loweringSoundB_of_place` takes, given the unit's instance in the audit
  circuit (`IsRowsUnit`, the circuit side) and `Layout.Aliased`: its inputs on one gate carry one value, from the flat
  statement's copies (open).
  - `Layout.placement` (`ExecPlacement.lean`) proves `Placement` from the layout facts `Layout` names. At a unit copy's
    computed rows only its slot has entries (pairwise-apart spans, `rangesEntry_of_covers`) and Δ has none. At its
    constant Δ's `(k, k)` cancels the row's own entry and `(k, pin)` reads the pin. The pin's row reads itself.
  - `setupH_layout` derives `Layout`:
    - `HmRow.parse` gives nets `Net.ofRows` built and lookup slots only (`parse_facts`);
    - `HmRow.check` gives the layout (`check_facts`);
    - `HmRow.pin` gives the pin's range;
    - `HmRow.delta`'s first loop writes exactly the slot constants' pairs, and every later pair is an input bit of a
      net slot (`delta_split`, `deltaIn_inBit`).
  - It needs two executable checks: `Net.parse` refuses a net whose input rows reach its constant, and
    `HmRow.pin` refuses a pin range with no slot. Otherwise an input port can cover a slot's constant column and Δ's
    copies rewrite that row, or the pin's row is empty or lands inside another range.
- **What a correct unit computes** is its certificate's, not a hypothesis: the
  end-to-end theorems are about the circuit of the statement's rows (`Prog.circuit`), and when a unit's rows are
  `GateRows.rows G L` and `G` computes `D`, a unit that is not faulty computes `D` (`IsRowsUnit.computes_of_cert`,
  `Certificate.lean`). A template's rows are tested against its primitives on samples; a checked layout's rows are proved
  to compute its type (`Types.Dag.layout_sound`, `assumption-notes/template-rows.md`).
- **The rows the verifier derives, read as `Rows`** (`Compose.lean`; verified-lowering phase 1, item 1d).
  - `Rows.compose` reads a unit's rows in logical order from a `Compose.Logical`: the columns in order, the input bits
    first, and each column's row as the block holds it, or a copy of `one` (the unit's constant, which Δ binds to the
    pin).
  - `placement_of_realizes` gives `Placement` when the block matrices, at each row's position, read its columns'
    positions, and the rows that copy `one` and the pin's row read the pin.
  - For a flat type, `derive`'s rows (`Compose.ofDerived`) are in that order (`compose_ordered`), and satisfied composed
    rows carry the type's outputs (`Rows.compose_eval`, from `Types.compose_sound`).
  - Over the type DAG (`ComposeDag.lean`), the unit's rows as the block holds them (`Compose.ofBlock`: `deriveChecked`'s
    order, Δ's constant copies copying `one`) are in that order (`ofBlock_ordered`, from `Types.Dag.order_sound`), and
    satisfied composed rows carry the unit type's outputs (`Rows.compose_eval_unit`, from `Types.Dag.unit_sound`).
  - **Placed** (`ComposePlace.lean`). `Compose.Realizes` is `placement_of_realizes`'s block facts as one proposition,
    and `Realizes.placement` the `Placement` it gives. `Compose.BlockFacts` is 1e's form of the same facts, over the
    unit's block rows (`blockRow`), and `BlockFacts.placement` gives a table class's placement at a slot for the unit's
    composed rows (`ofBlock`) from them. `Layout.realizes` proves `Realizes` at any unit slot of a flat statement
    `Stmt.setupH` accepted, for composed rows that are the unit net's (`Compose.NetRows`).
  - **A typed template's Δ** (`ExecTyped.lean`): `delta_typed` gives Δ_A and Δ_B as the flat entries
    (`delta_split`'s forms), then per VU the input bits' copies of their message bits, the parts' bound rows and the root
    rows' cross entries (`Typed.tail`). `CopyRow` and `ZeroRow` state a block row that reads exactly one column, or none;
    a satisfying witness carries the column's value there (`CopyRow.block_eq`), or 0 (`ZeroRow.block_eq`).
  - **A typed template's statement** (`ExecTemplate.lean`): what `setupH` accepted for `tags.typed`
    (`setupH_spec_typed`), `parseTyped`'s template and flat-class paths (`parseTyped_spec`), the template's parse
    (`parse_facts_tmpl`: its nets the root and placed layouts' circuits first, its block `blockOf t`, the rows' own wires,
    no leaf map or cut) and check (`check_facts_typed`).
  - **A template unit's parts** (`Types/Parts.lean`): `deriveChecked` also checks the unit's parts as its template
    statement reads them (`partsChecked`), giving the row facts a template's block rows rest on (`part_region`,
    `part_rows`, `delta_entry`, `part_reads`, `callee_prod` and the rest: the list is
    note:proofs/20261025T0738Z-report-flock-readme-parts).
  - **A template's block rows, placed** (`ExecTemplatePlace.lean`, T3). Given two facts structures,
    `TemplateLayout.blockFacts` proves `BlockFacts` at every VU `g`, at `Typed.col c tb g`, and
    `TemplateLayout.placement` gives a table class's `placed g`.
    - `TemplateLayout` is the accepted statement's layout: its ranges, the root's and the parts' nets and slots, and Δ.
      A part's net (`PartNet`) is its callee's derived rows, with a generated read's lookup from its held table.
    - `UnitShape` is the unit's shape: its constant row, its order within its own and its parts' columns, no Δ at an
      input, and a part's reads its callee's.
    - At a generated read's product row, the lookup reads what `readSide` reads (`lookupEntry_readSide`), so the block's
      `B` there is the row as the verifier folds it (`partB`, `PartNet.entryB`).
    - Supporting lemmas: unit columns sit at distinct positions across VUs (`col_inj`), and the typed tail at a unit
      column's position is its bound-input and cross entries (`tail_count`).
  - **From the accept step** (`ExecTemplateSetup.lean`). `setupH_templateLayout` derives `TemplateLayout` from what
    `Stmt.setupH` accepted for a template (`Typed.read`'s class and block). It uses the template's parse and check,
    `templateOf_spec`, and `blockOf_spec`, `slotOf_lt` and `slotOf_lt_of_lt`. `setupH_blockFacts` is then
    `BlockFacts` at every VU of the accepted statement.
    - The rows' circuits come after the root and the placed layouts' circuits, by `checkTyped`
      (`CheckFactsT.rows_after`). The unit's shape comes from `deriveChecked` (`unitShape_of_class`, with
      `unit_const_row` and `order_cols`).
    - A part may place a generated read: its net carries its own read and held table (`placedNet_gen`, `partNet_of`),
      and the held tables' words are their bytes read as `u32` (`read_held`). `part_reads` and `callee_prod`
      complete the unit's shape.
    - Supporting lemmas: the class's unit is one of its layouts (`unit_lt`), and Δ's other flat rows are input bits of
      the rows' circuits' slots (`deltaIn_rows`).
    - The unit's own check makes its inputs self rows with no Δ entry (`unit_inputs`), which discharges `UnitShape`'s
      input field (`unitShape_of`).
    - Each input of VU `g` copies its message bit: the block rows at its position read exactly `copySrc g w`
      (`input_rows`). As a model `CopyRow` (`input_copy`, and `setupH_inputCopy` for an accepted statement), that is a
      table class's `copy`. The message bit is a column of the block (`setupH_copySrc_lt`): it is in a `sha512x3`
      slot's message port (`msgCol_lt`), and the file has a row, since the instance's inputs are its rows' words
      (`CheckFactsT.ins`) and the unit's inputs its type's (`unit_inCols_size`, `read_template_ty`).
  - **A typed flat class's unit, placed** (`ExecFlatClass.lean`). A flat class's statement has the layout an untyped
    statement has (`setupH_flatLayout`): `parseTyped` puts the unit's net, `Typed.netOfD` of `derive`'s rows, where the
    untyped file has its `CIRCUIT unit` section, and `Layout` reads it as any net `Net.ofRows` built. The unit's composed
    rows are its net's (`netRows_flat`, `Compose.NetRows`), so `setupH_flatPlacement` is a table class's `placed` at
    each unit slot. That reads `PastInputs`, the order's columns after the unit's inputs past its net's input rows,
    which `Typed.read` checks (`Typed.flatNet`).
  - **A typed flat class's copies and zero rows** (`ExecFlatCopies.lean`). Δ is the slot constants' pairs, the rows'
    constant bits, then each unit copy's leaf, cut and wire pairs (`delta_flat`). `HmRow.check` wires each of the unit's
    input ports once (`CheckFacts.wired`, `wire_leaf`, `cut_leaf`), and `Typed.read` checks that the unit's inputs are
    its net's input port bits in order (`Typed.flatNet`), so those bits are distinct columns. So at each input the
    block rows read the one column Δ copies there (`Layout.placed_port`), and each unit slot's row past its useful rows
    is zero (`Layout.placed_zero`). `setupH_flatTableClass` is then the flat class's `TableClass`: its placement, its
    copies (`setupH_flatCopy`) and its forced-zero rows.
    - An input whose source lies outside the block has zero rows, and its copy position is its slot's forced-zero row.

### 1.6 Proved from the libraries, not assumed

**A1, Reed–Solomon mutual correlated agreement up to the Johnson bound, is proved.** BCHKS25 (ePrint
2025/2055) Theorem 4.6 as printed, which ArkLib admits, is `mcaError_le_bchks25` (`JohnsonMCA.lean`) on `Fin n`
domains. The proof uses Dao–Kominers–Thaler (DKT26, ePrint 2026/2056), proved in ArkLib at our pin
(`ReedSolomon.mcaError_affineLine_johnson_le`). With `ρ = (k − 1)/n` and `η = 1 − √ρ − δ`, its count is below
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

- **Interactive, live coins.** Each round's coins are flipped uniformly and independently after the prover has sent that
  round. Nothing is derived from a hash. Fiat–Shamir (C-fs) is out of scope.
- **Two layers.**
  - The *oracle layer* is where the statistical bound lives. There, a committed table is sent as a whole and the
    verifier reads the queried columns (§1.1).
  - The *compiled layer* replaces each table by its SHA-512 Merkle cap plus authentication paths, opened at the end of
    the proof. The coin server keeps each round's bytes before flipping that round's coins, and the replay compares
    bytes, so no hash binds the rounds (§1.2).
  - **The compiled layer is the adopted variant, not yet the recorded sessions'.** PROTOCOL.md specifies SHA-256 trees
    (§14) and round digests (§6) for the sessions it records, following upstream, and SHA-512 trees as its target. The
    compiled layer has SHA-512 trees and kept round bytes (DESIGN.md §1).
- **Sessions: batched and serialized** (DESIGN.md §11.1). There is exactly one verifier.
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
  step, for every template, is proved here (§1.5), and so is the placement for the executable's statement
  (`placement_of_setupH`). What a correct unit computes is its certificate's (`IsRowsUnit.computes_of_cert`).

## 3. Zero knowledge (Lemmas A, B and C proved for the model; the simulator over a model of the session's views)

The masked protocol and the reference prover are in DESIGN.md §6, and the session model in DESIGN.md §11.

**In Lean** (`ZK/`, pinned, the standard axioms only), for the model of one masked table `ZK/Table.lean`
(the reference prover `Table.view`, the simulator `Table.sim`) and sessions of `J` of them:

- **Lemma B**, special honest-verifier zero knowledge at a fixed coin vector where the prover refuses nothing:
  `table_shvzk`; with real hm96 leaves, within `2·N_hid·δ₁` under `Hm96Hiding`, `table_shvzk_hm96`; per session,
  `session_shvzk_hm96`.
- **Lemma A**, the pre-final view on two witnesses carrying the public region data (the dummy is one) has one
  distribution: `table_prefinal_indep` where the prover refuses nothing, and at every coin vector on what is sent before
  it refuses (a failing rank check, dependent `ρ^r`, `β = 0`), in the code's order, `session_prefinal_indep_cut`
  (`Definitions/Flock/ZK/Cut.lean`, `ZK/Refusal.lean`).
- **Lemma C**, the same for the pre-final transcript against a verifier that adapts its coin openings and aborts to the
  prover's messages and has its own tape, within `2·|L|·δ₁` for `L` the pre-final view's leaves:
  `adaptive_prefinal_hm96_tape` (`ZK/AdaptivePrefinal.lean`), from T5 (`card_run_eq`).
- M2's coin openings bind under SHA-512 collision resistance (T6, `coin_opening_binding_keyed`), and T7's numeric step
  (`gk_ratio_le_third`).
- **Theorem GK**, the Goldreich–Kahan simulator (`ZK/GK/`): `gk` makes a dummy run, estimates the completion rate from
  fresh dummy runs, and rewinds at most `4t/p̃` times with fresh dummy openings, under a cap `M`. `Model.gk_simulate`: for a
  deterministic `V*`, every test with values in `[0, 1]` of the simulator's output is within
  `3δ_C + 8t·δ_B + 3^{-t} + t/M + q²/2^513` of the real view's. `⊥_bind` is a SHA-512 collision the finder computes (T6's
  extraction as an algorithm, `bindColl_isColl` and `bindCollKeyed_isColl`, at most `2(|path| + 3 + n)` hashes), under
  `SHA512CRStrict` at the finder's budget `q`. `prefinalClose_hm96` (Lemma C, `δ_C = 2·|L|·δ₁`) and `rewindClose_hm96`
  (Lemma B, `δ_B = 2·N_hid·δ₁`) discharge the two distances, and `gk_simulate_hm96` is the bound with both on the keyed coin
  tree; `gk_simulate_avg` averages over `V*`'s tape. Expected polynomial time is stated as the expected number of `V*`'s
  runs: at most `1 + 5M` on every outcome (`sim_runs_le`), at most `1 + 5t` in expectation uniformly in `M`
  (`sim_runs_ev`), and `Pr[runs ≥ x] ≤ (1 + 5t)/x` (`sim_runs_tail`). The model of one session's views (`Model`: the real
  and dummy runs, the two rewinds, completion and the pre-final view, with its fields `comp_eq`, `inc_real`, `inc_dummy`
  and `bind_real`), its links to Lemmas B and C, the coin conflict (`hbad`) and Lemma B's non-degeneracy at the coins of
  every completed first run (`V*` picks them) are hypotheses.

**Not in Lean:** the link from `flock-circuit --zk` to the model (the Lean reference prover and its differential test,
DESIGN.md §6), which would instantiate `Model` and those hypotheses; ChaCha20 as uniform randomness for the masks and
padding. The leaf salts are the operating system's bytes, uniform under A3 (`Discharge/ZkView/OsSalts.lean`).

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
  for uniform masks and padding; the leaf salts are the operating system's (A3). The simulations of a session's tables compose, because their masks are independent.
- **Serialized sessions** then compose sequentially (Goldreich–Oren 1994).
- **Reserve, if concurrency is ever needed:** a single registered verifier key, or a Prabhakaran–Rosen–Sahai preamble
  (DESIGN.md §11.2).

**Expected assumptions** (to be confirmed against M1's spec):

- **Uniform prover randomness.** The masks, padding and leaf salts are uniform and independent. In production the
  leaf salts are the operating system's bytes, 192 a leaf (`getrandom`), which A3 makes
  uniform (`Discharge/ZkView/OsSalts.lean`), and the masks and padding are ChaCha20 output under a seed from the
  operating system. Treating that output as uniform is the one caveat to the lemma's statistical zero knowledge, and
  the operating system's randomness is A3.
- **Leaf hiding.** Halevi–Micali leaves are statistically hiding in the standard model, by the leftover hash lemma, with
  no random oracle (`N·2^-256` for N leaves with a fresh key, §4). They are binding under collision resistance of their
  hash, SHA-512 in the adopted variant.
- **Binding of the verifier's coin commitment** (§2: a SHA-512 Merkle root over Halevi–Micali coin commitments), under
  SHA-512 collision resistance. Against quantum verifiers it needs a collapse-binding commitment (DESIGN.md §9).
- **Uniformity of opened columns** of a padded Reed–Solomon codeword when the queries are at most the padding (the
  AHIV22 Lemma 4.15 condition). **Proved** (`ZK/Masking.lean`), with the standard axioms only:
  - `padded_openings_uniform`: at distinct points where the padding's factor doesn't vanish, every vector of opened
    values is reached by the same number of paddings, whatever the message is;
  - `level0_openings_uniform`: the same at M1's level 0, whose padding `(X_L + κ)·μ` combines the first `t_pad`
    novel-basis polynomials;
  - `card_translate_fiber`: any value masked by the image of uniform randomness under a surjective additive map is
    uniform. The pads, the mask slot, the extra-lane pairs and the paddings are all instances.

## 4. One root across many proofs

The weights are committed once and every later proof opens that root. The analysis is DESIGN.md §10.

- **Soundness over N tables**, by union bound: `N·2^-205 + N·t·2^{17.3−256}`, plus `t²/2^513` once for the root. So
  2^-128 holds while `N·t ≤ 2^110.7`. This counts proofs as statements about committed values. **Decision:** state security
  per table, or as a lifetime budget.
- **Partial integrity counts audits, and needs knowledge soundness** (the lifetime-soundness desk study, §2–§5).
  - In a sampled audit only the first faulty sampled unit matters. So an audit costs one table's hash term plus one
    serving-tree term (DESIGN.md §1.2): about `t·2^-(222…238)` per audit.
  - Halevi–Micali leaves are statistically hiding, so plain soundness is vacuous there: a unit is faulty only relative
    to extracted values. Each audit therefore also pays the table's knowledge error and a link term (§1.3).
  - Under A2 that term is `t·Q_s(8N₀/e)/2^256` (audit B: 2^-129.4 at `t = 2^80`; under strict time alone, proved
    for the non-ZK session, 2^-70.2), and it dominates. So 2^-128 over the lifetime holds while `N·t ≤ 2^78.4`
    to `2^90.4`, with `N` counting audits.
- **Binding** must hold for the root's whole life. **Design change:** use the SHA-512 leaf and tree scheme, since
  `hm96-sha256/v1` binds only up to `t ≈ 2^64.5`. Rotate on evidence, with a proof that both roots open to the same
  weights.
- **Hiding** is fixed by the leaf count, not by the number of proofs. **Design change:** sample the root's key at
  commitment, and never reuse salts.
- **Zero knowledge** holds per batched, serialized session (§2, §3). Throughput comes from proving in parallel inside
  a session, not from concurrency.
- **Statement leakage** accumulates over sessions. **Policy needed:** which predicates are public, and how many.
- **Linkability** is intended, and it needs knowledge soundness (§1.3).
