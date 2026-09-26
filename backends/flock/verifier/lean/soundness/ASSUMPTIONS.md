# Flock interactive soundness: assumptions and trusted components

For the expert reviewer and for Table 1. Lane `flock-soundness`, level 4a of the formal-verification plan: the
interactive mode only. The Lean development is this directory (`FlockSoundness/`). The verifier it is about is
specified in `../../PROTOCOL.md` (PR #85).

The work is staged along a16z's zkVM security stages (Thaler, "The path to secure and efficient zkVMs", March 2025).
Stage 1 covers the protocol: 1a PIOP soundness, 1b commitment binding, 1c the compiled argument, 1d the constraint
system against its semantics, 1e the whole theorem including ZK. Stage 2 is the verifier implementation and Stage 3 the
prover implementation. §10 maps them onto this work, including where it deliberately differs. §11 sketches a future
Fiat–Shamir or SNARK variant. §12 covers quantum provers and verifiers, §13 one root reused across many proofs.

## 1. The claim, and its status

**The theorem** (`FlockSoundness.table_sound`, `table_sound_fast100` in `FlockSoundness/Soundness.lean`). Consider one table
of a session, run as the verifier of record runs it:

- the `fast100` profile, run twice on one committed root;
- live coins;
- any statement of PROTOCOL.md §16.

For every prover strategy, the probability of this event is at most `tableError(fast100 m, shape)`:

> the verifier accepts both reps, while no codeword within the level-0 proximity radius of the committed table packs a
> witness that satisfies the statement.

`tableError` is a Lean function of the parameters (`Accounting/Bound.lean`), with one term per verifier coin. It is
proved to be at most **2^-195.5 for 22 ≤ m ≤ 33** and **2^-195.4 for m = 34, 35**
(`Accounting/Numbers.lean`). The statement shapes covered are those with at most 2^26 witness bits per block,
1,024 public regions and 64 link-point coordinates. That covers every statement of §16 with room to spare.

That theorem is the oracle layer, where committed tables are sent whole. **The compiled theorem** (§4.3) is about the
protocol as run, with SHA-512 Merkle caps and the coin server keeping each round's bytes. It holds for every classical
prover, under plain collision resistance of SHA-512. It adds `Σ_ℓ √(N_ℓ·Q_ℓ·Adv_ℓ) + Adv_×` for explicit collision
finders, which is at most `t·2^{17.3−256}` for a prover making `t` hash evaluations, so the total stays below 2^-128 up to
`t ≈ 2^110.7`. Quantum provers are covered by §12.

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
| Phase lemmas: zerocheck, lincheck, opening, Ligerito | 1a | `zerocheck_phase`, `lincheck_phase`, `opening_phase`, `ligerito_phase` | **to prove** |
| The link points reduce region checks to opened claims | 1a | `link_sound` | **to prove** |
| Merkle binding: two verified openings of one position with different columns give a SHA-512 collision | 1b | `Merkle.opening_binding` | proved |
| Rewinding: openings off the plurality table cost collisions, `#bad² ≤ N·Q·#conflict` | 1b | `Rewinding.bad_sq_le`, `prBad_le` | proved |
| Compilation step: the oracle bound plus the rewinding term, for a table committed before any coin or after a prefix | 1c | `Compiled.compile_step`, `compile_step_prefix` | proved |
| Session binding: the coin server keeps each round's bytes, so they are fixed before the round's coins | 1c | the `Game` model (`send` before `coin`) | by construction; no hash |
| The compiled table theorem: SHA-512 collision resistance, every prover | 1c, interactive | `table_sound_compiled` | **to write**: the compiled game (openings at the end of the proof) and the coupling (§4.3) |
| Knowledge soundness: extract the committed witness, then Halevi–Micali binding to registered data | 1c, 1e | `table_knowledge_sound`, `registered_weights` | **to write**, after `table_sound_compiled` (§4.4) |
| The block R1CS means the named circuit (the lowering theorem) | 1d | level 3, another lane | not in this lane |
| The executable Lean verifier refines the model | 2 | with `flock-verifier` | **to write** |
| Reference prover: completeness and zero knowledge | 3, 1e | after M1 (§8) | planned |

The proved results depend only on Lean's three standard axioms (`propext`, `Classical.choice`, `Quot.sound`). Run
`lake env lean FlockSoundness/Check.lean` to check. `rep_sound` and `table_sound` show `sorryAx` until the phase lemmas and
`link_sound` are proved.

## 2. The model

- **Interactive, live coins.** Each round's coins are drawn uniformly and independently after the prover has sent that
  round. Nothing is derived from a hash. Fiat–Shamir (C-fs) is out of scope.
- **Two layers.**
  - The *oracle layer* is where the statistical bound lives. There, a committed table is sent as a whole and the
    verifier reads the queried columns.
  - The *compiled layer* replaces each table by its SHA-512 Merkle cap plus authentication paths, opened at the end of
    the proof. The coin server keeps each round's bytes before drawing that round's coins, and the replay compares
    bytes, so no hash binds the rounds.
  - The theorem above is the oracle layer. §4.3 is the compiled layer.
  - **The compiled layer is the adopted variant, not yet the spec.** PROTOCOL.md (PR #85) specifies SHA-256 trees
    (§14) and round digests (§6), following upstream. Daniel adopted SHA-512 trees and kept round bytes on
    2026-09-26 (§4.2a). The spec, upstream's Rust verifier used for cross-validation, and the provers must follow.
- **The committed values.** Flock runs Ligerito in the list-decoding regime, and no out-of-domain sample follows the
  commitment (spec §9.1; paper Remark 11). A committed table therefore binds the prover to a short list of codewords,
  not to one witness. The list has at most 50 entries at rate 1/2 (ArkLib's Johnson bound, `1/(2ηρ)` with η = 0.02).
  - "The committed values satisfy the circuit" means that some codeword of that list packs a satisfying witness
    (`Model.CommittedSatisfies`), which is exactly right for soundness.
  - Claims that use the committed values elsewhere (registered data, links to other proofs, recursion) need them
    extracted (§4.4).
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

- **Ring switching.** The spec's variant (§12, A9: seven coins per claim, then batching, with Ligerito's sumcheck in
  place of the standalone one) is not Diamond–Posen's. Its soundness follows from the paper's Appendix B argument
  (injective recombination over GF(2)) plus Schwartz–Zippel. ArkLib's ring switching (legacy layer, 18 admits, the
  other variant) is not used.
- **Merkle binding.** Two different openings of one position against one cap give a collision: `Merkle.opening_binding`,
  proved, as VCVio proves for its trees.
- **Johnson list bounds.** ArkLib's `irs_lambda_le_johnson_mds`, proved.
- **Sumcheck rounds and univariate steps.** Root counting in Mathlib (`Polynomial.card_roots'`,
  `MvPolynomial.schwartz_zippel_totalDegree`).
- **The pinned zerocheck coordinates.** The 128 equality weights are independent over GF(2) and none equals 1. These
  are facts about constants, proved for the GHASH instance (`Arith.Correct`).

## 4. Cryptographic assumptions, and which hash model the number lives in

### 4.1 Where hashing enters

In the adopted variant (SHA-512 trees, round bytes kept):

1. **Merkle openings** (spec §14, with SHA-512 leaves and nodes). Opened columns must be the committed ones. This is
   the only hash assumption of the theorem: collision resistance of SHA-512.
2. **Round bytes** (§6). The coin server keeps each round's bytes, then draws that round's coins, and the replay
   compares the proof's bytes with the kept ones. Nothing is assumed: a message cannot change after its coins. This
   replaces the SHA-256 round digests, whose collisions would have let a prover choose a round's message after seeing
   its coins. Keeping the bytes costs about the proof size, 0.7 MB per rep.
3. **Statement binding** (the statement digest, Σ and `link_sha256`). These are in round 0's kept bytes, fixed before
   any coin. Their own collision resistance matters to what the statement refers to (registered data, linked
   proofs), not to this theorem.

### 4.2 Why not SHA-256 (Daniel's question: does collision resistance clear 2^-128?)

With SHA-256 trees, two arguments cleared 2^-128, each for `t` up to about 2^64 hash evaluations:

- the straight-line collision-resistance reduction, which covers only provers that know the tables they commit to;
- the random-oracle model.

Rewinding, the only collision-resistance argument for every prover, gives `t·2^-113.6` at m = 33, which never clears
2^-128. The comparison, the alternatives weighed and the cost of SHA-512 are in
`note:20260926T2230Z-finding-sha512-merkle-choice`. In short: the prover costs about 1.5× per hashed byte on GPUs (under
2% of a proof), proofs grow about 36%, and verifier time stays about the same.

### 4.2a The rewinding bound as a function of hash width

Plain collision resistance is Daniel's preferred model: no random oracle, no knowledge assumption. For arbitrary
provers it needs the rewinding argument. Its cost is a square root on each Merkle-committed level, so its bound depends
on the hash's output width `n`.

- **The argument** (§4.3): openings off the plurality table cost collisions, found by running the prover twice.
- **The bound.** For a prover making `t` hash evaluations:
  - `Pr[inconsistent opening] ≤ Σ_levels √(N_ℓ·Q_ℓ·Adv_CR(2t)) ≤ t·√2·S_m / 2^{n/2}`;
  - `N_ℓ` is a level's codeword length;
  - `Q_ℓ` is its openings over the whole session (level 0 is committed once and opened by both reps);
  - `S_m = Σ_ℓ √(N_ℓ·Q_ℓ)`, with `log2(√2·S_m)` equal to 10.7 at m = 22, 16.3 at m = 33 and 17.3 at m = 35.
- **The square root is inherent to rewinding.** A prover holding one collision, opened either way when its position
  is queried (probability `q`), is inconsistent with probability about `q/2`, but two runs expose it only with about
  `q²/2`.

| Hash | Merkle rewinding term | ε ≤ 2^-128 up to |
|---|---|---|
| SHA-256 | `t·2^16.3 / 2^128` (m = 33) | never, for any `t ≥ 1` |
| SHA-384 | `t·2^16.3 / 2^192` | `t ≈ 2^47` to `2^53` (m = 35 to 22) |
| SHA-512 | `t·2^16.3 / 2^256` | **`t ≈ 2^110.7` to `2^117`** (m = 35 to 22) |

- **Width needed.** Clearing 2^-128 at budget `t` needs `n ≥ 2·(log2 t + 128 + log2(√2·S_m))`. That is 437–451 bits at
  `t = 2^80`, and 477–491 bits at `t = 2^100`. Only SHA-512 among the standard hashes qualifies.
- **More reps make it slightly worse.** Every rep opens the shared level-0 table, adding about 0.4 bit per extra rep.
- **Tight terms need `n ≥ 128 + 2·log2 t`** (straight-line reductions, the random-oracle bound, any long-lived data
  root): SHA-256 meets that only up to `t = 2^64.5`, so **at `t ≥ 2^65` SHA-256 misses 2^-128 in every model.**

**The route that avoids both the random oracle and the knowledge assumption** (adopted by Daniel, 2026-09-26):

1. Build the in-proof Merkle trees with SHA-512, leaves and nodes.
2. Make the round digests assumption-free by having the coin server keep each round's bytes (it already receives
   them).

The compiled bound is then 2^-195.5 plus `t·2^{17.3−256}` plus negligible terms, from plain collision resistance, for
every prover, up to `t ≈ 2^110`. §4.3 states it.

**Outside the verifier:** the statement's own data commitments (frame-v3 roots, row digests, Halevi–Micali leaves
hashed in the circuit) are tight collision-resistance terms. At `t ≥ 2^65` they too need a hash of at least 288 bits.
That is a separate commitment-scheme decision.

### 4.3 The compiled theorem: SHA-512 collision resistance, every prover

The protocol is §1's table (the `fast100` profile twice on one root, live `os` coins, a statement of §16) in the adopted
variant: SHA-512 Merkle caps, and the coin server keeping each round's bytes. No random oracle and no knowledge
assumption is used.

**The committed values of an arbitrary prover.** Let `P*` be any classical prover. Without loss of generality it is
deterministic, since a randomized prover is an average of deterministic ones. Its **committed table** is the plurality
table `T*(P*)`: at each level-0 column position, the column `P*` opens there most often over the coins, with a path
that verifies against its cap. `T*` is a function of `P*`, as in every rewinding-based binding. It is not computed by
anyone.

**Theorem** (`table_sound_compiled`, to write). For every classical prover `P*`,

> Pr[the table is accepted, and no codeword within the level-0 proximity radius of `T*(P*)` packs a satisfying witness]
> ≤ `tableError(fast100 m, shape)` + `Σ_ℓ √(N_ℓ · Q_ℓ · Adv_ℓ)` + `Adv_×`.

- **The sum** is over the Merkle-committed levels: level 0 once, and levels 1 to R once per rep.
- **`N_ℓ`** is the level's codeword length, its number of leaves. **`Q_ℓ`** is its openings in one run. Level 0 is
  opened by both reps, 2 × 218 at m = 33.
- **`Adv_ℓ`** is the probability that an explicit collision finder `B_ℓ` outputs a SHA-512 collision. `B_ℓ` runs
  `P*` up to level ℓ's commitment, continues it twice on independent coins, and applies Merkle binding at the first
  position the two runs open to different columns.
- **`Adv_×`** is the probability that a single run opens one level-0 position in both reps, with two different columns
  that both verify. That is a collision found in one run, with no rewinding.
- For a false statement, where no witness satisfies it at all, the event is simply "accepted".

**The number.** `B_ℓ` makes at most `2t` SHA-512 evaluations when `P*` makes `t`, plus two verifications. Under the
generic bound for SHA-512, `Adv ≤ (2t)²/2^513`, which gives

> ε ≤ 2^-195.5 + `t·√2·S_m / 2^256` + `t²/2^511`,

with `log2(√2·S_m)` = 10.7 at m = 22, 16.3 at m = 33 and 17.3 at m = 35 (§4.2a). That is at most 2^-128 for `t` up to
2^110.7 at m = 35 and 2^117 at m = 22. The hash term overtakes the statistical one from `t ≈ 2^43` (m = 35) to
`2^50` (m = 22).

**Proof, and where it stands in Lean.**

1. **The oracle-model prover `P̃`** sends the plurality table wherever `P*` sends a cap (at level ℓ ≥ 1, given the run
   so far), and otherwise plays as `P*`. `table_sound` bounds it by `tableError`: proved, modulo the phase lemmas and
   `link_sound`.
2. **Coupling.** On the same coins, if every opening of `P*` agrees with the plurality tables and the compiled verifier
   accepts, the oracle verifier accepts `P̃`. **To write**, with the compiled game: Ligerito with caps for oracles and
   query answers deferred to the end of the proof, where the spec puts the openings.
3. **Rewinding.** Runs with an opening off the plurality ("bad") have probability at most `√(N_ℓ·Q_ℓ·Pr[conflict_ℓ])`
   per level, where `conflict_ℓ` is two runs sharing the prefix up to the commitment opening one position differently.
   Proved: `Rewinding.prBad_le`, and `Compiled.compile_step_prefix` (Jensen over prefixes); `Compiled.compile_step`
   combines steps 1–3 for one table.
4. **Binding.** A conflict is a SHA-512 collision, so `Pr[conflict_ℓ] ≤ Adv_ℓ`. Proved: `Merkle.opening_binding`.
5. **Rounds.** The kept bytes fix every message before its coins: the `Game` model itself (`send` before `coin`).

`table_sound_compiled` then follows from `table_sound`, `compile_step`, `compile_step_prefix` and a union bound over
levels. Steps 3 and 4 depend only on Lean's standard axioms (`Check.lean`).

**Remark: the random-oracle version.** Modeling SHA-512 as a random oracle gives straight-line extraction of the
committed tables (VCVio's Merkle extraction) and a tight bound, `2^-195.5 + O(t²/2^512) + t·(cap entries)/2^512`,
without the square root. It is a cross-check only. The number above does not rest on it.

**Table 1** cites `computational-soundness` under `cr/sha-512`, `live-verifier` and `rs-proximity-johnson` (A1). The
theorem is classical; §12 covers quantum provers.

### 4.4 Knowledge soundness: extracting the committed witness

Soundness says the committed witness satisfies the statement. A claim about *which* data a proof used needs that witness
computed, because hiding leaves open to essentially any row. So "these proofs used the registered weights" (§13) needs
it, and so does the recursion track, whose outer proof must be a proof of knowledge. The long form, with the effort
table, is `note:20260926T2330Z-finding-knowledge-soundness`.

- **`table_knowledge_sound`** (to write). An extractor reruns any classical prover `P*` from the start, about
  `(N₀/Q₀)·ln(1/(ζβ))/ε` times, where ε is the probability that `P*` is accepted. That sees all but a fraction ζ of
  level 0's columns, except with probability β. Level 0 is committed in the first message, so every run opens the same
  cap.
  - Two runs that open one position differently give a SHA-512 collision.
  - Otherwise the extractor list-decodes the observed columns, treating the rest as erasures, and outputs a witness that
    satisfies the statement.
  - The claim: `Pr[witness or collision] ≥ ε − κ`, where `κ` is the compiled soundness error plus extraction terms.
- **`registered_weights`** (to write). Given the registrant's opening `(W, R)`, the extracted rows equal `W` wherever the
  statement links the root, or tree binding and then Halevi–Micali binding give a SHA-512 collision. So
  `Pr[accepted ∧ rows ≠ W] ≤ κ + Adv_CR(B_link)`.
- **The extraction is mostly straight-line,** since the extractor sees conflicts itself. The rewinding lemma bounds the
  positions where it saw only off-plurality values.
  - A first estimate of that term is `t·2^-227/ε`, so 2^-128 needs `t/ε ≤ 2^99`. This is to be sharpened.
  - Generic interactive-BCS bounds give only about 2^-100 to 2^-111 with SHA-512.
- **Effort:** about 1,250–2,000 lines of Lean, roughly half the development so far, plus 200–350 if recursion needs
  witness-extended emulation. Decoding is the largest piece and is provable: CompPoly's Guruswami–Sudan decoder is
  verified with no `sorry`, and ArkLib's Johnson bound covers the punctured interleaved code. Write and review the
  analysis on paper first. It follows `table_sound_compiled`; `registered_weights` waits for the SHA-512 leaf layout,
  and recursion for M1/M2.
- **Risks:** the concrete knowledge error; a thin decoding margin at level 0 (`η = 1/50` leaves under 1% after
  erasures, so coverage must be about 99%); bridging CompPoly, ArkLib and `Model`. It is classical only (§12).

### 4.5 Coins

- **`os` coins (the default).** Each round's coins are uniform and unknown to the prover before it sent the round. They
  come from the coin server's `/dev/urandom` (spec §7.1). The randomness is trusted, not proved. Nothing else is
  needed.
- **`seed` coins** (PR #83, `coin-commit/sha256/v1`). In addition:
  - the seed commitment hides the seed;
  - SHA-256 in counter mode is a pseudorandom function.

  These are needed only if the seed mode becomes the verifier of record's source.

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

## 6. Where the spec follows upstream rather than the paper (spec §19), and the bound

| Id | Change | How the model and bound cover it |
|---|---|---|
| A1 | 6 zerocheck coins squeezed, never read | Drawn and ignored in the model; no term |
| A2 | Last Ligerito batching coin | Charged like every batching coin |
| A3 | Vanishing on S imposed by interpolation; degree 127 | Skip term `127/|F|` (the paper has 126) |
| A4 | `final_c_eval` sent, must equal `Pc(z)` | Forced, so it is not a prover message in the model |
| A5 | `(G(1), G(∞))` encoding, `inv(0) = 0` | A sampled zerocheck coordinate equal to 1 would unbind that round. That event is charged once, `(m−13)/|F|`; the pinned coordinates are not 1. |
| A6 | Pinned coordinates at bits 6–12 | Needs their 128 weights independent over GF(2) (`Arith.Correct`) |
| A7 | Lincheck computes `α·A₀ᵀe + B₀ᵀe`; 64 skip values; no matrix claims | No matrix-claim term; final skip `63/|F|` |
| A8 | Constant-wire coin `β` | The pin must be 1 in every block. A pin failure survives only if its extension equals 1 at `ρ_out` (`(m−k_log)/|F|`) or at `β` (`1/|F|`). |
| A9 | Per-claim ring switch, then batching | `7/|F|` per claim, plus `1/|F|` for the batch |
| A10 | Packing on the 7 low address bits, `b_v = x^v` | `Arith.pack` / `unpack`; basis property in `Arith.Correct` |
| A11 | GF(2^256) folds; stratified queries; out-of-domain sample at level 0; clear residual | Fold terms use `2^256`. Stratified sampling misses a far word with probability at most `(1−γ)^Q`, exactly as uniform sampling does (AM–GM per stratum). The level-0 sample is one more batched claim (`L₀/|F|`). |
| A12 | Proof of work: nonces are messages, never checked | No grinding credit anywhere |
| A13 | Label, statement digest and cap absorbed before the first coin | The statement is fixed before any coin |
| A14 | Region claims at the link points, opened with `ab`, `c` | The link points are drawn once for both reps. Their term, `L₀·regions·(m_pts/2^128)^2`, is added once and not squared. |
| A15 | `root_F = Σ` | Session check; no soundness term |
| A16 | `m > 35` rejected | The schedule is data for `22 ≤ m ≤ 35` only |

## 7. Accounting findings

- **2^-195.5 holds for m ≤ 33 only.** At m = 34 and 35, `fast100` has a seventh Ligerito level. Its query term
  (2^-102.6) moves the table bound to 2^-195.43. PROTOCOL.md §15 quotes 2^-195.5 per table, from the m = 33 ledger.
  Both are far below 2^-128.
- **Terms flock-128's ledger does not charge,** none moving the total by 0.001 bit: the list factor `L₀` before
  Ligerito's binding (paper Appendix A), the `r = 1` set (A5), the pin (A8), the link points (A14), MCA and fold terms
  per folding round rather than per level, and consistency batching at each level's own list size.
- **List sizes.** ArkLib proves the Johnson list bound as `1/(2ηρ)` (50 at rate 1/2). Flock uses the sharper
  `1/(2η√ρ)` (35). The bound uses ArkLib's, and the difference is below 0.001 bit.

## 8. Zero knowledge (planned; not yet proved)

**Targets.** Two simulation theorems for the interactive mode:

1. **Honest-verifier ZK, statistical given uniform masks.** A simulator produces transcripts distributed like real ones,
   up to a statistical term, from the public statement alone (the circuit, the roots and the public parameters).
2. **ZK against a malicious verifier, with the verifier's coin seed committed at step 0.** The verifier's coins are
   fixed by its commitment before the first prover message. So it suffices to simulate for every fixed coin sequence
   ("special" honest-verifier ZK), together with the commitment's binding.

**The masked protocol** is lane `flock-netlist`'s M1/M2 (PR #83). It consists of VEIL-style masking of the sumchecks,
Reed–Solomon padding in Ligerito over GF(2^128), and Halevi–Micali leaf commitments. The model in `Model/*` is written
so that these slot in without restructuring:

- the prover's randomness becomes an extra input of the reference prover;
- masks are extra messages or extra committed columns in the existing phases;
- salted leaves change only the Merkle layer.

**Expected assumptions** (to be confirmed against M1's spec):

- **Uniform prover randomness.** The masks, padding and leaf salts are uniform and independent. In production they come
  from the operating system and ChaCha; their quality is trusted.
- **Leaf hiding.** Halevi–Micali leaves are statistically hiding (`N·2^-256` for N leaves with a fresh key, §13) and
  binding under collision resistance of their hash, SHA-512 in the adopted variant.
- **Binding of the coin-seed commitment**, under SHA-256 collision resistance. This is needed for malicious-verifier ZK
  only. Against quantum verifiers it needs a wider, collapse-binding commitment (§12).
- **Uniformity of opened columns** of a padded Reed–Solomon codeword when the queries are at most the padding (the
  AHIV22 Lemma 4.15 condition). This is to be proved, not assumed.

**The reference prover** (agreed with Daniel):

- The production prover (Rust and CUDA) cannot run in Lean, so a slow, spec-level Lean reference prover stands in for
  it. It is a deterministic function of (statement, witness, prover randomness, verifier coins).
- **Its theorems:** completeness (the Lean verifier accepts its honest output), and zero knowledge (the simulator
  theorem, over uniform prover randomness).
- **Differential tests** tie production to it byte for byte: on the same inputs, seeds and coins, the production prover
  must emit exactly its messages, which pins down mask placement.
- **Still trusted:** that equivalence beyond the tested inputs, and the quality of the operating system's and ChaCha's
  randomness.

## 9. What remains (by a16z stage)

1. **Stage 1a, the phase lemmas of `rep_sound`.** The composition and the round lemmas are done. Their shared first
   piece is the list size: at most `L₀` codewords near the committed table. It follows from ArkLib's
   `irs_lambda_le_johnson_mds` once `Model.encode` is identified with ArkLib's interleaved Reed–Solomon code
   (`Arith.Correct.xhat_poly`). The phases:
   - zerocheck;
   - lincheck (including the pin);
   - the opening (ring switching);
   - Ligerito, level by level (fold, bind, query, batch; MCA commutes with list decoding from A1);
   - the final check.

   Each phase is a doomed-state transition bounded by the matching `Accounting` term. `Game.value_bind_le` chains them.
2. **Stage 1a, `link_sound`**: Schwartz–Zippel on each region's multilinear difference, at two independent points.
3. **`Arith.Correct`** for the GHASH instance, from CompPoly's definitions.
4. **Stages 1b and 1c (interactive), the compiled theorem** (§4.3). Merkle binding, the rewinding lemma and the
   compilation steps are proved. Remaining:
   - the compiled game: Ligerito with caps in place of oracles and query answers deferred to the end of the proof;
   - the coupling between a compiled run and the oracle-model run of the plurality prover `P̃` on the same coins;
   - `table_sound_compiled` from `table_sound`, `compile_step`, `compile_step_prefix` and a union bound.

   Session binding needs nothing, since the server keeps the round bytes.
5. **Knowledge soundness** (§4.4): `table_knowledge_sound`, then `registered_weights`. It is needed for claims about
   registered data and for recursion's outer proof.
6. **Stage 2, refinement to `Flock.Verify`** (with lane `flock-verifier`): the executable verifier accepts only if the
   model accepts.
7. **Stages 1e and 3, zero knowledge and the reference prover** (§8), after M1's masking spec.
8. **Stage 1d, the lowering theorem**, which belongs to level 3 in another lane. Stage 1e then glues everything into
   one theorem from the verifier's acceptance to "the committed values satisfy the named circuit".

## 10. Staging against a16z's security stages

a16z's stages are written for zkVMs: SNARKs proving a virtual machine's execution, made non-interactive by
Fiat–Shamir. Our setting differs in three ways:

- it is interactive, with live coins;
- there is one Boolean-circuit prover;
- the verifier is written in Lean and runs offline.

We follow the staging and vocabulary, and differ where the setting does.

| a16z stage | What it asks | Here | Follows, or deliberately differs |
|---|---|---|---|
| **1a** PIOP soundness | A machine-checked soundness proof of the polynomial IOP | `table_sound` at the oracle layer: the PIOP with Ligerito as an interactive oracle commitment, information-theoretic, per table, against any prover. | Follows, modulo A1 (our level 4a); proving A1 is level 4b, not plannable at the Johnson bound (a unique-decoding profile would be, since ArkLib proves those gaps). |
| **1b** commitment binding, under a cryptographic assumption or an idealized model | Binding of the polynomial commitment | The Merkle layer: plain SHA-512 collision resistance, for every prover, by rewinding to the plurality table (§4.3). Ligerito's commitment binds to a list of at most L₀ codewords, not to one polynomial (§2). The random-oracle version is only a cross-check. | Follows. The model is named per result, as a16z asks. |
| **1c** the compiled argument | With Fiat–Shamir, security in the random-oracle model | **Deliberately different: no Fiat–Shamir.** The oracle protocol plus Merkle commitments (1b) plus session binding, which needs no assumption since the coin server keeps each round's bytes. | Differs. No random oracle for coins, avoiding the article's main caveat; a Fiat–Shamir variant is future work (§11). |
| **1d** constraint system against semantics | The constraint system is equivalent to the VM's semantics | No VM: 1d is the lowering theorem, that the block R1CS (spec §16) is satisfied exactly by correct evaluations of the named circuit (level 3, another lane). | Differs in content, not intent: "the circuit is the meaning". Circuit against hardware is checked by capture replay, outside formal scope. |
| **1e** glue, and ZK if intended | One theorem, including zero knowledge | One theorem from the verifier's acceptance to "the committed values satisfy the named circuit", for the compiled protocol. Zero knowledge is proved separately (§8). | Follows. **Recursion caveat:** triggered only by the recursion track. Its outer proof must be knowledge-sound (§4.4), and every component inside it verified. |
| **2** verifier implementation | The implemented verifier (Rust, Solidity, …) matches the Stage-1 protocol | `Flock.Verify` (lane `flock-verifier`) refines `Model`: if it accepts, the model accepts. | Differs: a Lean-to-Lean refinement; the compiler and runtime are trusted, mitigated by agreement tests (§5). |
| **3** prover implementation | The prover implementation is proved complete, and zero-knowledge if intended | A Lean reference prover with completeness and zero-knowledge theorems, tied to production by byte-for-byte differential tests (§8). | **Deliberately different: the production prover is not proved.** Untested equivalence is trusted; its failures cost liveness, not soundness. |

**Levels.** Our formal-verification levels (the Lean verifier study) map onto the stages as follows:

- level 4a is Stage 1a–1c (interactive) modulo A1;
- level 4b completes Stage 1 by proving A1;
- level 3 is Stage 2, plus the lowering theorem of 1d;
- levels 1–2 are the implementation that Stage 2 is about.

**Stating soundness.** a16z's concrete levels are 100 bits as the bare minimum and 128 bits as standard practice. We
target 2^-128, and state the number the way the stages suggest:

1. **The statistical part, per table:** `−log2` of the proven error against any prover at the oracle layer. That is
   195.5 bits for m ≤ 33 and 195.4 bits for m = 34, 35.
   - Only proven results are used: A1 is a published theorem, not a conjectured proximity gap.
   - No bits are credited to grinding (proof of work).
   - A session with several tables adds their errors.
2. **The cryptographic part, with its model named:** collision resistance of SHA-512, as
   `Σ_ℓ √(N_ℓ·Q_ℓ·Adv_ℓ) + Adv_×` for explicit collision finders. That is at most `t·2^{17.3−256}` under the generic
   bound (§4.3), for classical provers.
3. **Each assumption, at the stage that uses it:**
   - 1a: A1;
   - 1b: SHA-512 collision resistance;
   - 1c (interactive): uniform coins, and nothing for the rounds, whose bytes the server keeps;
   - Stage 2: the compiler's correctness, mitigated by tests.

**Performance stages** (proofs of at most 256 KB, verification in at most 16 ms) belong to the tables. `fast100` run
twice (1–2 MB, 20–50 ms) exceeds them, which matters only for a SNARK variant.

## 11. A Fiat–Shamir or SNARK variant (future extension)

A non-interactive variant ("C-fs") derives each round's coins from a hash of the transcript so far. It would need:

- **Round-by-round soundness of every component, and a Fiat–Shamir theorem** in the random-oracle model (a16z 1c).
  a16z's caveat applies: whether Fiat–Shamir with a real hash is secure is open. Systems without recursion are on
  firmer ground.
- **A new parameter set.** A cheating prover can re-hash to re-sample a round's coins (state restoration). So the error
  grows to roughly `t × (round-by-round error)`, and two reps no longer square against such a prover. flock-128 found
  `fast100` run twice at best 2^-75.6 under Fiat–Shamir. A SNARK profile would need one of:
  - the strict `Fast`/`Slim` profiles, with grinding credited in the random-oracle model;
  - GF(2^256) at the PIOP coins (flock-128's `f256` option).
- **Recursion,** if proofs must be small. That triggers a16z's recursion caveat: every component inside the recursion
  must be verified too.

**What we do now to keep the door open:**

- **Round-by-round first.** Each phase of `rep_sound` is a per-coin lemma: from a doomed state, whatever the prover
  sends, the next coin leaves it with probability at most ε. That is a round-by-round error; a Fiat–Shamir proof would
  reuse these lemmas.
- **One joint term, kept separately.** The two out-of-domain samples per level are charged jointly, `(μ/2^128)^2`,
  which is valid with live coins. Their round-by-round error, `pairs·μ/2^128` (about 2^-104.5 at m = 33), stays its own
  lemma.
- **The model is ready for a compiler.** `coin` nodes are the only randomness, and the tree carries the transcript
  prefix, so a Fiat–Shamir compiler would replace each `coin` by a random-oracle query on that prefix.
- **Tools.** VCVio has Fiat–Shamir for Σ-protocols and the forking lemma. ArkLib's BCS compiler and its Fiat–Shamir for
  IOPs (roadmap phase 6) are blocked, so a direct random-oracle proof is the alternative.

**The `seed` coin mode is not Fiat–Shamir** (PR #83, spec §7.2). Coins come from a seed the verifier commits at step
0, so soundness adds only seed hiding and SHA-256 as a PRF. It is the bridge to zero knowledge against a malicious
verifier (§8), not to non-interactive proofs.

## 12. Quantum provers and verifiers

The full analysis, with sources, is `note:20260926T2240Z-finding-post-quantum`.

**The theorem is classical, and only its compiled layer is affected.**

- **The oracle layer holds against quantum provers as it stands.** `table_sound` is statistical, and every message and
  coin is classical. A quantum prover is therefore one more randomized strategy, which `table_sound` already bounds.
- **The compiled layer does not carry over.** §4.3 rewinds `P*` from a snapshot taken at the commitment.
  - A quantum state cannot be copied, and measuring the first run's openings disturbs it. So there is no second
    independent run, and no plurality table.
  - Collision resistance is also the wrong property. Some hashes that resist quantum collision-finding give commitments
    that do not bind quantum adversaries (Ambainis–Rosmanis–Unruh 2014). The replacement is **collapsing** (Unruh
    2016).
  - Classical proof frameworks, ours and VCVio included, are unsound for quantum adversaries by design.

**What is known, and what applies to live coins.**

- Chiesa–Ma–Spooner–Zhandry (FOCS 2021) prove Kilian's argument post-quantum sound with a collapsing hash.
  Chiesa–Dall'Agnol–Di–Guan–Spooner (TCC 2025, ePrint 2025/947) extend it to our setting, interactive BCS for any
  public-coin semi-adaptive IOP with a collapse-position-binding vector commitment:
  `ε_ARG ≤ ε_IOP + Σ_i L_i·(q_i + 5)·ε_collapse + ϵ`.
- **Flock fits:** public coins; Merkle caps over a collapsing hash; Ligerito's queries depend only on coins
  (non-adaptive); `ε_IOP` is the oracle-layer bound; kept round bytes. So **if SHA-512 is collapsing, Flock's
  interactive protocol is post-quantum sound with error `tableError + negl(λ)`**, a paper-level corollary not yet
  written out.
- **Fiat–Shamir (§11).** The reference is the quantum random-oracle model: `O(t²·ε_rbr + t³/2^λ)` (Chiesa–Manohar–Spooner,
  TCC 2019).

**Is SHA-512 collapsing?** It is conjectured. Random oracles are collapsing (Unruh 2016), and Merkle–Damgård preserves
collapsing (Unruh, Asiacrypt 2016; Fehr, TCC 2018). So SHA-512 is collapsing if its compression function is. No attack
is known beyond generic collision search. Table 1 would need a new property for it (`collapsing/sha-512`).

**The loss makes the corollary asymptotic only.** The collapse adversary costs `poly(k·L_max/ϵ)` prover runs and its
advantage is multiplied by `Σ_i L_i·(q_i + 5) ≈ 2^30` (m = 33, `L_max = 2^21`, 150–200 rounds). A 2^-128 target needs
`ϵ ≤ 2^-128`, hence at least 2^157 runs; even then `q³/2^512 = 2^-41`, times 2^30, is 2^-11. **So no concrete
post-quantum bound follows at our parameters, for any prover size.** A number would need the quantum random-oracle model
(straight-line extraction by Zhandry's compressed oracle, target `ε_stat + O(t³/2^512)`, unproved for the interactive
case) or a tighter standard-model rewinding (open).

**Grover and BHT margins for SHA-512.**

| Attack | Quantum cost |
|---|---|
| Preimage or second preimage (Grover) | 2^256 queries, sequential; `P` machines gain only `√P` |
| Collision (Brassard–Høyer–Tapp) | 2^170.7 queries, and as much quantum-accessible memory |
| Collision with small quantum memory (Chailloux–Naya-Plasencia–Schrottenloher) | 2^204.8 time |
| Collision, generic limit (Zhandry) | success at most `O(q³/2^512)` |

- **The hash has ample margin**, the reduction's loss is what blocks a number. SHA-256 collisions cost 2^85.3 by BHT;
  NIST's post-quantum category 4 is SHA-384 collision search, and SHA-512 sits above it.
- **`seed` coins** (PR #83) sit at exactly 2^128: Grover on the 256-bit seed, and on SHA-256 as a PRF.

**Zero knowledge against quantum verifiers.**

- **Honest-verifier ZK is unaffected.** The view is classical and statistically close to the simulation, and
  statistical distance bounds quantum distinguishers too. Halevi–Micali hiding, masks and padding are statistical.
- **Malicious-verifier ZK needs quantum rewinding** of a verifier with quantum state. The closest result, Lombardi–Ma–
  Spooner (FOCS 2022), proves Goldreich–Kahan (committed challenges) post-quantum ZK, given a collapse-binding,
  statistically hiding verifier commitment. Constant-round black-box post-quantum ZK is impossible (Chia–Chung–Liu–
  Yamakawa, FOCS 2021); ours has 150–200 rounds, but adapting LMS22 is research. **Design consequence:** the coin-seed
  commitment needs a 128-bit quantum margin, so SHA-384 or SHA-512, not SHA-256.

**What a post-quantum statement would take:**

1. **The asymptotic corollary, on paper (small):** Flock is interactive BCS over a public-coin, non-adaptive IOP, so
   CDDGS25 applies under "SHA-512 is collapsing". It needs a quantum-cryptography reviewer and gives no number.
2. **A concrete number (research):** interactive BCS in the quantum random-oracle model with concrete terms, or tighter
   quantum rewinding.
3. **Lean.**
   - **What exists:** Lean 4 quantum-information libraries (Lean-QIT, QICLean): states, channels, measurements,
     entropy. They are outside Mathlib and not on our pins.
   - **What no proof assistant has:** quantum adversaries with superposition queries in a game model, collapsing,
     CMSZ21's rewinding engine, CDDGS25's multi-round argument, or Zhandry's compressed oracle. qrhl-tool and EasyPQC
     do quantum random-oracle proofs, but no rewinding.
   - **A full Lean statement** is a research program larger than this whole development, and not plannable now.
   - **A cheap intermediate:** CDDGS25's theorem as a named `Prop`, like A1, with only the application proved. It is
     honest, but adds little while the conclusion is asymptotic.

**Recommendation.** Keep the theorem classical, and claim no post-quantum number in Table 1. A fair statement is: the
statistical layer holds against quantum provers; the compiled protocol is post-quantum sound asymptotically if SHA-512
is collapsing (CDDGS25); no concrete post-quantum bound is known at these parameters.

## 13. One root, many proofs

The weights are committed once, as Halevi–Micali leaves under a Merkle root registered as the model's anchor, and every
later proof opens that root. Numbers and sources: `note:20260926T2310Z-finding-root-reuse`.

- **Soundness over N tables**, by union bound: `N·2^-195.5 + N·t·2^{17.3−256}`, plus the root's straight-line binding
  once, `t²/2^513`. The statistical part allows N up to 2^67.5; the rewinding term needs `N·t ≤ 2^110.7` (about 2^30
  tables at `t = 2^80`). So the per-table hash term runs out, not the root; the random-oracle version's is global.
  **Decision:** state security per table, or as a lifetime budget on `N·t`.
- **Binding for the root's life.** Proofs publish the leaves' commit strings and paths, so binding is straight-line
  against the adversary's cumulative work. It must hold as long as anyone relies on the root, disputes about old proofs
  included.
  - **Design change:** `hm96-sha256/v1` binds only up to `t ≈ 2^64.5`, so commit the root under the SHA-512 leaf and
    tree scheme.
  - Age doesn't weaken SHA-512. Rotate on a cryptanalytic advance, a salt or key exposure, or a hash upgrade, with a
    fresh key and salts and a one-time proof that both roots open to the same weights.
- **Hiding doesn't accumulate.** Every proof republishes the same commit strings, so the leaf count sets the bound:
  `N_leaves·2^-256` with a key drawn at commitment, or `N_leaves·2^-192` with the pinned key, except for a one-time
  `2^-64` fraction of keys.
  - **Design change:** draw the root's key at commitment, never reuse salts, and keep the salts as secret as the
    weights.
- **Zero knowledge.** Honest-verifier ZK composes sequentially and concurrently, with errors adding. Malicious-verifier
  ZK with the step-0 seed composes sequentially, by a hybrid.
  - Simultaneous sessions against a malicious verifier are not covered. The simulator rewinds each session to learn its
    seed, and interleaving defeats that; black-box concurrent ZK needs Ω̃(log n)-round preambles or setup.
  - **Decision**, one of:
    - serialize sessions (globally, if one party can hold many verifier identities);
    - take coins from a public beacon, which is slower and puts soundness on the beacon;
    - scope it out.
- **Statement leakage accumulates, and ZK doesn't bound it.** N sessions reveal N statements: workload shapes, timing,
  the shared root, and each proved predicate's truth. Weights never appear and outputs stay committed, but a predicate
  over committed outputs gives whoever sees it one bit per proof, an oracle to the model over chosen inputs. **Policy
  needed:** which predicates are public, and how many.
- **Linkability** is the point of one root, and it is also a leak: every proof visibly shares one model. Unlinkability
  would need a fresh commitment and an equality proof per run, which means re-hashing the weights each time.
  - **Knowledge soundness:** a statistically hiding leaf opens to essentially any row. So "these proofs used the registered
    weights" (even one proof) needs each proof's committed witness extracted, then Halevi–Micali binding. That is
    knowledge soundness, now planned (§4.4), and the circuit-privacy design's composition theorem needs it too.
