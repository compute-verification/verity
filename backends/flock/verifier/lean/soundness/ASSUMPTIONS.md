# Flock interactive soundness: assumptions and theorems

For the expert reviewer and for Table 1. This file is the concise list of what is proved, what is assumed and what is
trusted. `DESIGN.md`, beside it, holds the derivations: the hash-width analysis, the compiled proof, the knowledge
extractor, the spec deviations, the zero-knowledge plan, the a16z staging, and the post-quantum and root-reuse analyses.

Lane `flock-soundness`, level 4a of the formal-verification plan: the interactive mode only. The Lean development is
this directory (`FlockSoundness/`). The verifier it is about is specified in `../../PROTOCOL.md` (PR #85). The work is
staged along a16z's zkVM security stages (DESIGN.md §7).

## 1. Theorems

### 1.1 Soundness at the oracle layer (`table_sound`, `table_sound_fast100`)

Consider one table of a session, run as the verifier of record runs it: the `fast100` profile, run twice on one committed
root, with live coins, for any statement of PROTOCOL.md §16. For every prover strategy, the probability of this event is
at most `tableError(fast100 m, shape)`:

> the verifier accepts both reps, while no codeword within the level-0 proximity radius of the committed table packs a
> witness that satisfies the statement.

- **`tableError`** is a Lean function of the parameters (`Accounting/Bound.lean`), with one term per verifier coin. It
  is proved to be at most **2^-195.5 for 22 ≤ m ≤ 33** and **2^-195.4 for m = 34, 35** (`Accounting/Numbers.lean`).
- **The statement shapes covered** have at most 2^26 witness bits per block, 1,024 public regions and 64 link-point
  coordinates. That is every statement of §16, with room to spare.
- **At this layer committed tables are sent whole.** The theorem is statistical, so it holds against quantum provers
  too.

### 1.2 Compiled soundness (`table_sound_compiled`, to write)

This is the protocol as run, in the adopted variant: SHA-512 Merkle caps, openings at the end of the proof, and the coin
server keeping each round's bytes. For every classical prover `P*`, whose committed table `T*(P*)` is its plurality
table (at each level-0 position, the column `P*` opens there most often over the coins):

> Pr[accepted, and no codeword within the level-0 proximity radius of `T*(P*)` packs a satisfying witness]
> ≤ `tableError` + `Σ_ℓ √(N_ℓ·Q_ℓ·Adv_ℓ)` + `Adv_×`.

- **`Adv_ℓ` and `Adv_×`** are the success probabilities of explicit SHA-512 collision finders (DESIGN.md §2).
- **Under the generic bound** this is at most `2^-195.5 + t·2^{17.3−256} + t²/2^511` for a prover making `t` hash
  evaluations. That is at most 2^-128 up to `t ≈ 2^110.7` (m = 35) to `2^117` (m = 22).
- **No random oracle and no knowledge assumption** is used.

### 1.3 Knowledge soundness (`table_knowledge_sound`, `registered_weights`, to write)

It is needed for any claim about which data a proof used ("these proofs used the registered weights"), and for the
recursion track, whose outer proof must be a proof of knowledge (DESIGN.md §3).

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
| Opening phase: ring switching (seven coins per claim) and the batch, for any list size | 1a | `opening_phase_of_card` (`OpeningPhase.lean`) | proved; `opening_phase` follows from the list-size lemma |
| The list size: at most `L₀` codewords near the committed table (ArkLib's Johnson bound) | 1a | `closeMsgs_card_le` | **to prove** |
| Phase lemmas: zerocheck, lincheck, Ligerito | 1a | `zerocheck_phase`, `lincheck_phase`, `ligerito_phase` | **to prove** |
| The link points reduce region checks to opened claims | 1a | `link_sound` | **to prove** |
| Merkle binding: two verified openings of one position with different columns give a SHA-512 collision | 1b | `Merkle.opening_binding` | proved |
| Rewinding: openings off the plurality table cost collisions, `#bad² ≤ N·Q·#conflict` | 1b | `Rewinding.bad_sq_le`, `prBad_le` | proved |
| Compilation step: the oracle bound plus the rewinding term, for a table committed before any coin or after a prefix | 1c | `Compiled.compile_step`, `compile_step_prefix` | proved |
| Session binding: the coin server keeps each round's bytes, so they are fixed before the round's coins | 1c | the `Game` model (`send` before `coin`) | by construction; no hash |
| The compiled table theorem (§1.2) | 1c, interactive | `table_sound_compiled` | **to write**: the compiled game (openings at the end of the proof) and the coupling (DESIGN.md §2) |
| Knowledge soundness (§1.3): extract the committed witness, then Halevi–Micali binding to registered data | 1c, 1e | `table_knowledge_sound`, `registered_weights` | **to write**, after `table_sound_compiled` (DESIGN.md §3) |
| The block R1CS means the named circuit (the lowering theorem) | 1d | level 3, another lane | not in this lane |
| The executable Lean verifier refines the model | 2 | with `flock-verifier` | **to write** |
| Reference prover: completeness and zero knowledge | 3, 1e | after M1 (§6) | planned |

The proved results depend only on Lean's three standard axioms (`propext`, `Classical.choice`, `Quot.sound`). Run
`lake env lean FlockSoundness/Check.lean` to check. `rep_sound` and `table_sound` show `sorryAx` until the three remaining
phase lemmas, the list-size lemma and `link_sound` are proved.

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
- **`Arith.Correct`.** These are facts about the concrete GHASH arithmetic: Lagrange weights, the packing basis, the
  Reed–Solomon novel basis, the pinned constants, and the balance of query coins. They are hypotheses until proved for
  the instance the executable verifier defines. They are to be proved, not trusted.
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
  2^-128 holds while `N·t ≤ 2^110.7`. **Decision, with Daniel:** state security per table, or as a lifetime budget.
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

1. **Stage 1a, the phase lemmas of `rep_sound`.** The composition, the round lemmas and the opening phase are done.
   - **The shared list size** (`closeMsgs_card_le`): at most `L₀` codewords near the committed table. It follows from
     ArkLib's `irs_lambda_le_johnson_mds` once `Model.encode` is identified with ArkLib's interleaved Reed–Solomon code
     (`Arith.Correct.xhat_poly`).
   - **Zerocheck.**
   - **Lincheck,** including the pin.
   - **Ligerito,** level by level (fold, bind, query, batch; MCA commutes with list decoding from A1), and the final
     check.

   Each phase is a doomed-state transition bounded by the matching `Accounting` term. `Game.value_bind_le` chains them.
2. **Stage 1a, `link_sound`**: Schwartz–Zippel on each region's multilinear difference, at two independent points.
3. **`Arith.Correct`** for the GHASH instance, from CompPoly's definitions.
4. **Stages 1b and 1c (interactive), the compiled theorem** (§1.2). Merkle binding, the rewinding lemma and the
   compilation steps are proved. Remaining:
   - the compiled game: Ligerito with caps in place of oracles and query answers deferred to the end of the proof;
   - the coupling between a compiled run and the oracle-model run of the plurality prover `P̃` on the same coins;
   - `table_sound_compiled` from `table_sound`, `compile_step`, `compile_step_prefix` and a union bound.

   Session binding needs nothing, since the server keeps the round bytes.
5. **Knowledge soundness** (§1.3): `table_knowledge_sound`, then `registered_weights`. It is needed for claims about
   registered data and for recursion's outer proof.
6. **Stage 2, refinement to `Flock.Verify`** (with lane `flock-verifier`): the executable verifier accepts only if the
   model accepts.
7. **Stages 1e and 3, zero knowledge and the reference prover** (§6), after M1's masking spec: first the special
   honest-verifier ZK lemma, then single-rewind simulation of one batched session.
8. **Stage 1d, the lowering theorem**, which belongs to level 3 in another lane. Stage 1e then glues everything into
   one theorem from the verifier's acceptance to "the committed values satisfy the named circuit".
