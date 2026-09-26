# Flock interactive soundness: assumptions and trusted components

For the expert reviewer and for Table 1. Lane `flock-soundness`, level 4a of the formal-verification plan: the
interactive mode only. The Lean development is this directory (`FlockSoundness/`). The verifier it is about is
specified in `../../PROTOCOL.md` (PR #85).

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

| Piece | Lean | Status |
|---|---|---|
| Interaction model: live coins, prover strategies, acceptance probability, optimal prover | `Game/Basic.lean` | proved |
| Two reps on independent coins, any interleaving: the error squares | `Game/Interleave.lean`, `value_interleave_le` | proved |
| The bound as a function of the parameters, and its value for every `m` | `Accounting/*` | proved |
| The verifier, following PROTOCOL.md §9–§16 | `Model/*` | written (definitions) |
| Composition: one table from one rep and the link points | `table_sound` | proved from the two lemmas below |
| One rep is sound from a doomed start | `rep_sound` | **to prove** |
| The link points reduce region checks to opened claims | `link_sound` | **to prove** |
| The Merkle layer (SHA-256 caps and paths) and session binding (round digests) | `Merkle` | **to write** (§4) |

The proved results depend only on Lean's three standard axioms (`propext`, `Classical.choice`, `Quot.sound`). Run
`lake env lean FlockSoundness/Check.lean` to check. `table_sound` shows `sorryAx` until the two lemmas are proved.

## 2. The model

- **Interactive, live coins.** Each round's coins are drawn uniformly and independently after the prover has sent that
  round. Nothing is derived from a hash. Fiat–Shamir (C-fs) is out of scope.
- **Two layers.**
  - The *oracle layer* is where the statistical bound lives. There, a committed table is sent as a whole and the
    verifier reads the queried columns.
  - The *compiled layer* replaces each table by its SHA-256 Merkle cap plus authentication paths. It also binds each
    round's bytes by the digest the coin server recorded before drawing the round's coins (§6 of the spec).
  - The theorem above is the oracle layer. §4 covers the compiled layer.
- **The committed values.** Flock runs Ligerito in the list-decoding regime, and no out-of-domain sample follows the
  commitment (spec §9.1; paper Remark 11). A committed table therefore binds the prover to a short list of codewords,
  not to one witness. The list has at most 50 entries at rate 1/2 (ArkLib's Johnson bound, `1/(2ηρ)` with η = 0.02).
  - "The committed values satisfy the circuit" therefore means that some codeword of that list packs a satisfying
    witness (`Model.CommittedSatisfies`).
  - For soundness of the statement this is exactly right.
  - Anything that uses the committed values outside this proof needs its own binding argument. Link claims to another
    proof system are an example.
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

- **Ring switching.** The spec's variant (§12, deviation A9) is different from the Diamond–Posen protocol:
  - each claim is switched with its own seven coins;
  - then all claims are batched;
  - Ligerito's sumcheck replaces the standalone one.

  Its soundness follows from the paper's Appendix B argument (injective recombination over GF(2)) plus Schwartz–Zippel.
  So it needs no Diamond–Posen theorem. ArkLib's ring switching is on its legacy layer with 18 admits and formalizes
  the other variant; it is not used.
- **Merkle binding.** Two different openings of one position against one root yield an explicit SHA-256 collision.
  This is VCVio's `getPutativeRootWithHash_binding_collision`, proved.
- **Johnson list bounds.** ArkLib's `irs_lambda_le_johnson_mds`, proved.
- **Sumcheck rounds and univariate steps.** Root counting in Mathlib (`Polynomial.card_roots'`,
  `MvPolynomial.schwartz_zippel_totalDegree`).
- **The pinned zerocheck coordinates.** The 128 equality weights are independent over GF(2) and none equals 1. These
  are facts about constants, proved for the GHASH instance (`Arith.Correct`).

## 4. Cryptographic assumptions, and which hash model the number lives in

### 4.1 Where SHA-256 enters

1. **Merkle openings** (spec §14). Opened columns must be the committed ones.
2. **Round digests** (§6). The coin server records SHA-256 of each round's bytes, then draws that round's coins. The
   replay checks the proof's bytes against the digests. A collision would let a prover choose a round's message after
   seeing its coins.
3. **Statement binding** (the statement digest, Σ and `link_sha256`). These are part of the bytes bound by item 2.

The round digests (item 2) could drop the assumption entirely if the server kept each round's bytes. It already receives
them, and storing them costs about the proof size, 0.6 MB per rep.

### 4.2 Does collision resistance clear 2^-128? (Daniel's question)

Three bounds for the compiled protocol. In each, `t` is the number of SHA-256 evaluations the adversary makes. The
statistical term is `ε_stat = tableError ≤ 2^-195.5`.

| Model | Covers | Bound | Clears 2^-128? |
|---|---|---|---|
| Collision resistance, straight-line reduction | Provers that know the tables they commit to: each commitment is the Merkle cap of a table the prover holds | `ε_stat + Adv_CR(B)`, with `B` one run of the prover plus a comparison of the opened columns with the tables. With the generic bound `Adv_CR ≤ t²/2^256`, this is `2^-195.5 + t²·2^-256`. | **Yes**, for `t` up to about 2^64 hash evaluations. The statistical margin is untouched. |
| Collision resistance, rewinding (the only known argument for arbitrary provers) | Every prover | Run the prover twice from the same commitment to extract a consistent table. This gives `Pr[inconsistent opening] ≤ √(N·Q·Adv_CR)` per committed level (`N` = codeword length, `Q` = queries). At level 0 for m = 33, `N·Q = 2^28.8`. So the bound is `t·2^-113.6` even with the most optimistic `Adv_CR ≤ t²/2^256`. | **No**, for every `t ≥ 1`. The square root sits on the hash term, where the 67.5-bit statistical margin cannot absorb it. |
| Random-oracle model (VCVio's Merkle extraction) | Every prover | `ε_stat + t²/2^257 + t·(cap entries)/2^256` | **Yes**, for `t` up to about 2^63.5 |

Round digests (item 2) get a tight collision-resistance bound for every prover. The reduction includes the coin server,
which sees the live bytes. So a mismatch with an equal digest is itself a collision.

### 4.3 The model the theorem uses

- **The main compiled theorem will be stated under collision resistance**, for provers that know their committed tables
  (the first row). Its hash term is exactly `Pr[B finds a SHA-256 collision]`, for an explicit, efficient `B`. It is
  not a hypothesis. Table 1 cites this as `cr/sha-256`.
- **"Knows its tables" is a real restriction.** A prover that sends a cap without holding a table it hashes to is not
  covered by collision resistance at 2^-128. For such provers the only route to 2^-128 is the random-oracle model
  (the third row). The number then lives in the random-oracle model, and Table 1 must say so.
- The oracle-layer theorem (§1) is information-theoretic and needs neither.

### 4.4 Coins

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
- **Lean's compiler, runtime and the C compiler, for the running binary.** In production only the Lean verifier runs.
  Upstream's Rust verifier is a cross-validation test in CI, not a second verifier that must also accept. Proofs are
  about Lean definitions, not the machine code.
  - **Mitigation:** the agreement tests of spec §17.1. An independent implementation (Rust, compiled by rustc) must
    reach the same verdict on the whole corpus and on fuzzed mutations. So a miscompilation that changes a tested
    verdict fails CI.
  - **Not mitigated:** inputs no test reaches.
  - **Also trusted:** any `@[extern]` or `@[implemented_by]` in the verification logic. There is none today.
- **The model against the executable verifier.** Until the refinement theorem, the theorem is about `Model`, not about
  `Flock.Verify`. The refinement theorem says that if the executable verifier accepts, the model accepts on the decoded
  proof and record.
  - Alignment rests on the executable verifier using the same arithmetic (`Model.Arith`) and the same order of steps.
  - The first refinement proofs belong to the level-3 plan.
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
- **Terms flock-128's ledger does not charge.** None changes the total by more than 0.001 bit:
  - the list factor `L₀` on every term before Ligerito's binding (paper Appendix A, "Security");
  - the `r = 1` exceptional set (A5);
  - pin enforcement (A8);
  - the link points (A14);
  - the MCA and fold-sumcheck terms, summed over every folding round rather than one per level;
  - consistency batching charged at the list size of the level it is about.
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
- **Leaf hiding.** Halevi–Micali commitments are statistically hiding (`2^-k` per leaf by HM96 Theorem 1, summed over
  every opened leaf) and binding under SHA-256 collision resistance.
- **Binding of the coin-seed commitment**, under SHA-256 collision resistance. This is needed for malicious-verifier ZK
  only.
- **Uniformity of opened columns** of a padded Reed–Solomon codeword when the queries are at most the padding (the
  AHIV22 Lemma 4.15 condition). This is to be proved, not assumed.

**The reference prover** (agreed with Daniel):

- The production prover (Rust and CUDA) cannot run in Lean. So a Lean reference prover is written at spec level. It is
  slow and never run at scale.
- It is a deterministic function of (statement, witness, prover randomness, verifier coins).
- Its theorems are:
  - **completeness**: the Lean verifier accepts its honest output;
  - **zero knowledge**: the simulator theorem, over uniform prover randomness.
- It is linked to production by byte-for-byte differential testing. For the same statement, witness, prover seed and
  verifier coins, the optimized prover must produce exactly the reference prover's messages. That pins down mask
  placement.
- **What stays trusted:**
  - that equivalence beyond the tested inputs;
  - the quality of the operating system's and ChaCha's randomness.

## 9. What remains

1. **`rep_sound`**, phase by phase:
   - zerocheck;
   - lincheck (including the pin);
   - the opening (ring switching);
   - Ligerito, level by level (fold, bind, query, batch; MCA commutes with list decoding from A1);
   - the final check.

   Each phase is a doomed-state transition bounded by the matching `Accounting` term. `Game.value_bind_le` chains them.
2. **`link_sound`**: Schwartz–Zippel on each region's multilinear difference, at two independent points.
3. **`Arith.Correct`** for the GHASH instance, from CompPoly's definitions.
4. **The compiled theorems** (§4):
   - collision resistance with explicit tables, using VCVio's binding theorem;
   - session binding through the round digests;
   - the random-oracle version for arbitrary provers.
5. **Refinement to `Flock.Verify`** (with lane `flock-verifier`): the executable verifier accepts only if the model
   accepts.
6. **Zero knowledge** (§8), after M1's masking spec.
