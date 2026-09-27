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

### 1.3 Knowledge soundness (`table_knowledge_sound`, `registered_weights`, to write)

It is needed for any claim about which data a proof used ("these proofs used the registered weights"), and for the
recursion track, whose outer proof must be a proof of knowledge (DESIGN.md §3).

**It is required for the partial-integrity composition of sampled audits** (§7). Halevi–Micali leaves are statistically
hiding, so every commit string opens to essentially every value, and plain soundness says nothing about which values a
unit used. The audit bound defines a wrong unit through extracted values, so it pays the table's knowledge error and a
link term per audit.

- **`table_knowledge_sound`.** An extractor that reruns any classical prover accepted with probability ε outputs a
  satisfying witness or a SHA-512 collision, with probability at least `ε − κ`. Here `κ` is the compiled soundness error
  plus extraction terms.
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
| Merkle binding: two verified openings of one position with different columns give a SHA-512 collision | 1b | `Merkle.opening_binding` | proved |
| Rewinding: openings off the plurality table cost collisions, `#bad² ≤ N·Q·#conflict` | 1b | `Rewinding.bad_sq_le`, `prBad_le` | proved |
| Compilation step: the oracle bound plus the rewinding term, for a table committed before any coin or after a prefix | 1c | `Compiled.compile_step`, `compile_step_prefix` | proved |
| Session binding: the coin server keeps each round's bytes, so they are fixed before the round's coins | 1c | the `Game` model (`send` before `coin`) | by construction; no hash |
| The compiled table theorem (§1.2): the compiled game (caps, openings at the end of the proof), the lockstep coupling with the plurality prover, rewinding at every cap | 1c, interactive | `table_sound_compiled` (`Model/Compiled.lean`, `Game/Lock.lean`, `CompiledLock.lean`, `CompiledSound.lean`) | proved |
| Knowledge soundness (§1.3): extract the committed witness, then Halevi–Micali binding to registered data | 1c, 1e | `table_knowledge_sound`, `registered_weights` | **to write**, after `table_sound_compiled` (DESIGN.md §3); required for sampled audits (§7) |
| The block R1CS means the named circuit (the lowering theorem) | 1d | level 3, another lane | not in this lane |
| The executable Lean verifier refines the model | 2 | with `flock-verifier` | **to write** |
| Reference prover: completeness and zero knowledge | 3, 1e | after M1 (§6) | planned |

The proved results depend only on Lean's three standard axioms (`propext`, `Classical.choice`, `Quot.sound`). Run
`lake env lean FlockSoundness/Check.lean` to check.

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
  column and the public regions. That this R1CS means the named circuit is the lowering theorem (level 3), outside
  this lane.

## 3. Named assumptions (mathematics taken without proof)

There is exactly one. It is a `Prop` (`Assumptions.BCHKS25Thm46`), and the theorems that use it take it as a hypothesis
(`hMCA`). The development declares no `axiom`. So a mis-stated assumption can make a theorem vacuous, but it can never
make Lean prove something false.

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

**The theorem** (to prove): single-rewind simulation of one batched session (§2). For every malicious verifier with
auxiliary input, a Goldreich–Kahan simulator produces a view indistinguishable from the verifier's view of one batched
session, from the tables' public statements alone. It makes one extraction run (an honest prover on a dummy witness),
one rewind to just after the coin commitment, and runs in expected polynomial time.

- **It rests on** special honest-verifier ZK of masked Flock: a simulator for every fixed coin sequence, which is VEIL's
  notion and the first zero-knowledge lemma to prove. The simulations of a session's tables compose, because their
  masks are independent.
- **Serialized sessions** then compose sequentially (Goldreich–Oren 1994).
- **Reserve, if concurrency is ever needed:** a single registered verifier key, or a Prabhakaran–Rosen–Sahai preamble
  (DESIGN.md §11.2).

**Expected assumptions** (to be confirmed against M1's spec):

- **Uniform prover randomness.** The masks, padding and leaf salts are uniform and independent. In production they come
  from the operating system and ChaCha; their quality is trusted.
- **Leaf hiding.** Halevi–Micali leaves are statistically hiding (`N·2^-256` for N leaves with a fresh key, §7) and
  binding under collision resistance of their hash, SHA-512 in the adopted variant.
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
    serving-tree term (DESIGN.md §1.2): about `t·2^-(222…238)` per audit, and 2^-128 over the lifetime holds while
    `N·t ≤ 2^94` to `2^110`, with `N` counting audits.
  - Halevi–Micali leaves are statistically hiding, so plain soundness is vacuous there: a unit is wrong only relative
    to extracted values. Each audit therefore also pays the table's knowledge error and a link term (§1.3).
  - The first estimate of that term, `√(t·2^-227)` per audit (2^-73.5 at `t = 2^80`), is negligible next to realistic
    sampling terms but not next to 2^-128. It is the number to sharpen.
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

1. **Knowledge soundness** (§1.3): `table_knowledge_sound`, then `registered_weights`. It is needed for claims about
   registered data and for recursion's outer proof, and it is required for the sampled-audit composition, with the link
   term (DESIGN.md §3).
2. **Stage 2, refinement to `Flock.Verify`** (with lane `flock-verifier`): the executable verifier accepts only if the
   model accepts.
3. **Stages 1e and 3, zero knowledge and the reference prover** (§6), after M1's masking spec: first the special
   honest-verifier ZK lemma, then single-rewind simulation of one batched session.
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
| `Analysis.LinkSound` | `δ_link` | extraction succeeds but differs from `X` on the unit's committed inputs and outputs | the link theorem, to write (flock-soundness review of the extraction analysis, §4d) |
| `Partition.AnchorsSound` | `δ_in` | the session accepts while the committed inputs differ from the anchors' values | `registered_weights` (§1.3); zero at the oracle layer (`RejectsBadInputs`) |
| `Partition.SessionSound` | `ε` | oracle layer only: a session with a drawn wrong unit accepts with probability at most `ε` | `table_sound_exec`, projected from the batched session to the first wrong drawn unit's table (to write) |

Both per-unit hypotheses are averages over the whole audit, for every prover and every rule that picks a drawn unit
before the session's first coin, not bounds state by state.

**Theorems, one-stage** (`Audit/OneStage.lean`), with $e_L(B)$ the probability that the draw misses $B$:

- `audit_le`: $\Pr[\text{accept} \wedge E(X, S)] \le \Pr_S[E(X, S) \wedge S \cap \mathrm{wrong}(X) = \emptyset] +
  \varepsilon_{ks} + \delta_{link}$ for every event $E$ on the committed transcript and the draw;
- `audit_profile`: $\Pr[\text{accept} \wedge \mathrm{wrong}(X) \in \mathcal B] \le \sup_{B \in \mathcal B} e_L(B) +
  \varepsilon_{ks} + \delta_{link}$;
- `audit_count`: $\delta(K) = \mathrm{miss}_L(K) + \varepsilon_{ks} + \delta_{link}$;
- `audit_drawn`: drawn units are correct up to $\varepsilon_{ks} + \delta_{link}$, even for a consumer who looks after
  the draw;
- `Partition.audit_cone`: for committed wires $T$ fixed before the draw, $\Pr[\text{accept} \wedge X_T \ne
  \mathrm{eval}_C(a)_T] \le \max_{u \in \mathrm{cone}(T)} (1 - \pi_u) + \varepsilon_{ks} + \delta_{link} + \delta_{in}$;
- `Partition.audit_cone_count`: at least $j$ wrong units in a cone;
- `Partition.audit_covered`: a cone inside the draw, for wires chosen even after the draw: $\varepsilon_{ks} +
  \delta_{link} + \delta_{in}$;
- `Partition.oracle_count`, `oracle_profile`, `oracle_cone`: the same at the oracle layer, from `SessionSound ε`, with
  $\delta_{link} = \delta_{in} = 0$.

**Theorems, laws** (`Audit/Law.lean`): `subset_escape`, the uniform $k$-subset's escape
$\binom{n-|B|}{k}/\binom{n}{k}$; `subset_miss`, its count curve $\binom{n-K}{k}/\binom{n}{k}$; `subset_minimax`, **uniform is
optimal**: no law that proves $k$ units on average has a smaller count curve, at any $K$.

**Theorems, two-stage** (`Audit/TwoStage.lean`):

- `twoStage_profile`, variant (b), nested sampling: the profile over the coarse partition with the effective escape
  $e_{\text{eff}}(B) = \mathbb E_{S_1}\big[\prod_{u \in B \cap S_1} \max_{v \subseteq u}(1 - \pi_{L_2 u}(v))\big]$, with
  $\varepsilon_{ks}$ and $\delta_{link}$ at the fine level;
- `effEscape_full`, `twoStage_full_profile`: **(b1) is one-stage over the coarse partition** ($e_{\text{eff}} = e_{L_1}$);
- `Partition.two_stage_a_coarse`: **(a) is one-stage over the fine partition**, and its curve bounds coarse counts;
- `Partition.Refines.exists_wrong_fine`: a wrong coarse unit has a wrong fine unit under every interior, the lemma
  behind (b).

**Composition** (`Audit/Circuit.lean`): `compose` (every unit correct and the inputs anchored give the circuit's
evaluation on every committed wire) and `compose_cone` (the same for a consumer's wires, from its cone alone).

**Not yet:** the (b2) closed form, tightness, and the two insecurity counterexamples (DESIGN.md §12.4); the Flock
instantiation of `SessionSound`, which needs the batched session's projection to one table.
