# Flock interactive soundness: assumptions and trusted components

For the expert reviewer and for Table 1. Lane `flock-soundness`, level 4a of the formal-verification plan: the
interactive mode only. The Lean development is this directory (`FlockSoundness/`). The verifier it is about is
specified in `../../PROTOCOL.md` (PR #85).

The work is staged along a16z's zkVM security stages (Thaler, "The path to secure and efficient zkVMs", March 2025).
Stage 1 covers the protocol: 1a PIOP soundness, 1b commitment binding, 1c the compiled argument, 1d the constraint
system against its semantics, 1e the whole theorem including ZK. Stage 2 is the verifier implementation and Stage 3 the
prover implementation. §10 maps them onto this work, including where it deliberately differs. §11 sketches a future
Fiat–Shamir or SNARK variant.

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

| Piece | a16z stage | Lean | Status |
|---|---|---|---|
| Interaction model: live coins, prover strategies, acceptance probability, optimal prover | (framework) | `Game/Basic.lean` | proved |
| A live coin hits a fixed nonzero polynomial's zero with probability at most degree/|F| (Schwartz–Zippel) | (framework) | `Game/Poly.lean` | proved |
| Two reps on independent coins, any interleaving: the error squares | 1a | `Game/Interleave.lean`, `value_interleave_le` | proved |
| The bound as a function of the parameters, and its value for every `m` | 1a | `Accounting/*` | proved |
| The verifier, following PROTOCOL.md §9–§16 | 1a | `Model/*` | written (definitions) |
| Composition: one table from one rep and the link points | 1a | `table_sound` | proved from the two lemmas below |
| One rep is sound from a doomed start | 1a | `rep_sound` | **to prove** |
| The link points reduce region checks to opened claims | 1a | `link_sound` | **to prove** |
| The Merkle layer (SHA-256 caps and paths) | 1b | `Merkle` | **to write** (§4) |
| Session binding (round digests) with live coins: the compiled interactive argument | 1c, interactive | `Merkle` | **to write** (§4) |
| The block R1CS means the named circuit (the lowering theorem) | 1d | level 3, another lane | not in this lane |
| The executable Lean verifier refines the model | 2 | with `flock-verifier` | **to write** |
| Reference prover: completeness and zero knowledge | 3, 1e | after M1 (§8) | planned |

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

### 4.2a The rewinding bound as a function of hash width

Plain collision resistance is Daniel's preferred model: no random oracle, no knowledge assumption. For arbitrary
provers it needs the rewinding argument. Its cost is a square root on each Merkle-committed level, so its bound depends
on the hash's output width `n`.

- **The argument.** For each committed level, define the prover's "committed" table as the plurality opening of each
  position over its continuations. If the prover opens a queried position differently from the plurality, running its
  continuation twice from the same commitment finds a collision often enough.
- **The bound.** For a prover making `t` hash evaluations:
  - `Pr[inconsistent opening] ≤ Σ_levels √(N_ℓ·Q_ℓ·Adv_CR(2t)) ≤ t·√2·S_m / 2^{n/2}`;
  - `N_ℓ` is a level's codeword length;
  - `Q_ℓ` is its openings over the whole session (level 0 is committed once and opened by both reps);
  - `S_m = Σ_ℓ √(N_ℓ·Q_ℓ)`, with `log2(√2·S_m)` equal to 10.7 at m = 22, 16.3 at m = 33 and 17.3 at m = 35.
- **The square root is inherent to rewinding.** Take a prover holding one collision at one position and using either
  side with probability ½ when that position is queried. It is inconsistent with probability about `q/2`, where `q` is
  the probability the position is queried. But two runs expose the collision only with probability about `q²/2`.

| Hash | Merkle rewinding term | ε ≤ 2^-128 up to |
|---|---|---|
| SHA-256 | `t·2^16.3 / 2^128` (m = 33) | never, for any `t ≥ 1` |
| SHA-384 | `t·2^16.3 / 2^192` | `t ≈ 2^47` to `2^53` (m = 35 to 22) |
| SHA-512 | `t·2^16.3 / 2^256` | **`t ≈ 2^110.7` to `2^117`** (m = 35 to 22) |

- **Width needed.** Clearing 2^-128 at budget `t` needs `n ≥ 2·(log2 t + 128 + log2(√2·S_m))`. That is 437–451 bits at
  `t = 2^80`, and 477–491 bits at `t = 2^100`. Only SHA-512 among the standard hashes qualifies.
- **More reps make it slightly worse.** Every rep opens the shared level-0 table, adding about 0.4 bit per extra rep.
- **The tight terms need `n ≥ 128 + 2·log2 t`.** These are the round digests, the straight-line reduction and the
  random-oracle bound. SHA-256 meets that up to `t = 2^64.5`, SHA-384 up to `t = 2^128.5`. So **at `t ≥ 2^65`, SHA-256
  misses 2^-128 in every model, the random-oracle model included.** Daniel's range, 2^80–2^100, needs wider hashes
  whichever model is chosen.
- **The project's convention agrees.** `verity_numerical.security.accounting.hash_collision_bound` charges `q²/2^h` at
  `q = 2^64`. A-GKR moved its Merkle tree to SHA-512 for this reason (`backends/gkr/gpu/sha512_cuda.py`).

**The route that avoids both the random oracle and the knowledge assumption:**

1. Build the in-proof Merkle trees with SHA-512, leaves and nodes.
2. Make the round digests assumption-free by having the coin server keep each round's bytes (it already receives
   them). Alternatively, move the digests to SHA-384 or SHA-512.

The compiled bound is then 2^-195.5 plus `t·2^{17.3−256}` plus negligible terms, from plain collision resistance, for
every prover, up to `t ≈ 2^110`.

**What the alternatives give:**

- **More repetitions:** no help, as above.
- **Rewinding more than twice:** a k-run reduction finds more collisions but runs k times longer. Under the generic
  bound the gain cancels.
- **Tight standard-model extraction** needs other primitives: somewhere-statistically-binding or somewhere-extractable
  hashes, built from LWE or DDH. They are impractical at this scale.
- **Sending the verifier the committed tables** makes binding information-theoretic, but costs about 2.3 GiB per table
  at m = 33, twice the witness, so the proof is no longer succinct.

**Cost of SHA-512 against SHA-256 in Flock.** Flock's in-proof Merkle trees are computed and checked natively, not in a
circuit.

- **Prover.**
  - The level-0 tree at m = 33 hashes about 2.1 GiB per table: 2^21 columns of 1 KiB, plus nodes.
  - On NVIDIA GPUs SHA-512 costs about 1.5× SHA-256 per byte, since 64-bit arithmetic is emulated. hashcat on an
    RTX 4090 does about 22 G SHA-256 compressions/s against 7.5 G SHA-512 compressions/s, and a SHA-512 compression
    covers twice the bytes.
  - That is a few milliseconds more per table, well under 2% of a 0.5–1 s GPU proof.
  - On CPUs, where SHA-256 has SHA-NI and SHA-512 usually has no instructions, it is about 2× per byte, so the Merkle
    share of a CPU proof roughly doubles.
- **Proof size.** 64-byte siblings double the hash part, about 36% of a proof: 6,126 hashes in the m = 32 example of
  spec §8.3. Proofs grow from about 544 KB to about 740 KB per rep.
- **Verifier.** Lean has no hash instructions either way. A SHA-512 compression costs about 1.25× a SHA-256 one and
  covers twice the bytes, so hashing time stays about the same (a few ms).
- **In-circuit.** This applies only if Merkle checks were ever verified inside a circuit (recursion, for a SNARK
  variant), or to the statement's own data hashes.
  - Per compression (Bristol Fashion): 22,573 against 57,947 AND gates, 2.6× per compression and 1.3× per input byte.
  - In Flock's power-of-two slots: 2^15 against 2^16 rows per compression, the same rows per input byte. Per
    Merkle node (two compressions each) that is twice the rows.
- **The statement's own data commitments** (frame-v3 roots, row digests, Halevi–Micali leaves hashed in the circuit)
  are tight collision-resistance terms. At `t ≥ 2^65` they too need a hash of at least 288 bits. That is a
  commitment-scheme decision, separate from the verifier's.

### 4.3 The model the theorem uses

- **The main compiled theorem will be stated under collision resistance**, for provers that know their committed tables
  (the first row). Its hash term is exactly `Pr[B finds a SHA-256 collision]`, for an explicit, efficient `B`. It is
  not a hypothesis. Table 1 cites this as `cr/sha-256`.
- **"Knows its tables" is a real restriction.** A prover that sends a cap without holding a table it hashes to is not
  covered by collision resistance at 2^-128. For such provers the only route to 2^-128 is the random-oracle model
  (the third row). The number then lives in the random-oracle model, and Table 1 must say so.
- **With SHA-512 Merkle trees** (§4.2a), plain collision resistance covers every prover up to about 2^110 hash
  evaluations. Neither the knowledge restriction nor the random-oracle model is needed. This is the recommended target
  if Daniel keeps collision resistance.
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

## 9. What remains (by a16z stage)

1. **Stage 1a, `rep_sound`**, phase by phase:
   - zerocheck;
   - lincheck (including the pin);
   - the opening (ring switching);
   - Ligerito, level by level (fold, bind, query, batch; MCA commutes with list decoding from A1);
   - the final check.

   Each phase is a doomed-state transition bounded by the matching `Accounting` term. `Game.value_bind_le` chains them.
2. **Stage 1a, `link_sound`**: Schwartz–Zippel on each region's multilinear difference, at two independent points.
3. **`Arith.Correct`** for the GHASH instance, from CompPoly's definitions.
4. **Stages 1b and 1c (interactive), the compiled theorems** (§4):
   - collision resistance with explicit tables, using VCVio's binding theorem;
   - session binding through the round digests;
   - the random-oracle version for arbitrary provers.
5. **Stage 2, refinement to `Flock.Verify`** (with lane `flock-verifier`): the executable verifier accepts only if the
   model accepts.
6. **Stages 1e and 3, zero knowledge and the reference prover** (§8), after M1's masking spec.
7. **Stage 1d, the lowering theorem**, which belongs to level 3 in another lane. Stage 1e then glues everything into
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
| **1a** PIOP soundness | A machine-checked soundness proof of the polynomial IOP | `table_sound` at the oracle layer. It covers the PIOP (zerocheck, lincheck, ring switching, batching) together with Ligerito as an interactive oracle commitment. It is information-theoretic, per table, against any prover. | Follows, with one gap: a16z's Stage 1 has no room for an unproved cited theorem. We reach "Stage 1a modulo A1" (our level 4a). Proving A1 is level 4b, which is not plannable at the Johnson bound. A unique-decoding profile would make it plannable, since ArkLib proves those gaps. |
| **1b** commitment binding, under a cryptographic assumption or an idealized model | Binding of the polynomial commitment | The Merkle layer: SHA-256 collision resistance, tight for provers that know their tables, or random-oracle extraction for every prover (§4.2). Ligerito's commitment binds to a list of at most L₀ codewords, not to one polynomial (§2). | Follows. The model is named per result, as a16z asks. |
| **1c** the compiled argument | With Fiat–Shamir, security in the random-oracle model | **Deliberately different: no Fiat–Shamir.** The compiled argument is the oracle protocol plus Merkle commitments (1b) plus session binding. Each round's bytes are bound by the digest the coin server recorded before drawing that round's coins. That binding is tight under collision resistance for every prover, since the server holds the live bytes. | Differs. The coins need no random oracle, which avoids the article's main caveat (Fiat–Shamir security with real hash functions is an open question). A Fiat–Shamir variant is future work (§11). |
| **1d** constraint system against semantics | The constraint system is equivalent to the VM's semantics | There is no VM: statements are circuits. 1d is the lowering theorem: the block R1CS the verifier builds (slots, Δ, pin, regions; spec §16) is satisfied exactly by correct evaluations of the named circuit. That is level 3, in another lane. | Differs in content, not in intent. a16z's "verified bytecode" caveat becomes our rule "the circuit is the meaning". That the circuit matches the hardware is checked empirically, by capture replay recorded as Store evidence, and stays outside formal scope. |
| **1e** glue, and ZK if intended | One theorem, including zero knowledge | One theorem from the verifier's acceptance to "the committed values satisfy the named circuit", for the compiled protocol. Zero knowledge is proved separately (§8). | Follows. **Recursion caveat:** not triggered. There is no recursion: the spec rejects forks and the union prover. |
| **2** verifier implementation | The implemented verifier (Rust, Solidity, …) matches the Stage-1 protocol | The executable Lean verifier (`Flock.Verify`, lane `flock-verifier`) refines `Model`: if it accepts, the model accepts. | Differs in the target. The implementation is itself Lean, so the refinement is Lean-to-Lean. The machine code is trusted: Lean's compiler and runtime, mitigated by agreement tests against upstream's Rust verifier in CI (§5). |
| **3** prover implementation | The prover implementation is proved complete, and zero-knowledge if intended | A Lean reference prover with completeness and zero-knowledge theorems. The production prover (Rust and CUDA) is linked to it by byte-for-byte differential tests (§8). | **Deliberately different: the production prover is not proved.** Equivalence beyond the tested inputs is trusted. Completeness failures cost liveness (a true statement that cannot be proved), not soundness. |

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
2. **The cryptographic part, with its model named:**
   - collision resistance: the probability that an explicit reduction finds a SHA-256 collision;
   - or the random-oracle model: `t²/2^257 + …` (§4.2).
3. **Each assumption, at the stage that uses it:**
   - 1a: A1;
   - 1b: SHA-256 collision resistance, or the random-oracle model;
   - 1c (interactive): SHA-256 collision resistance for the round digests, and uniform coins;
   - Stage 2: the compiler's correctness, mitigated by tests.

**Performance stages.** a16z also sets speed and memory stages, and verifier-cost limits: proofs of at most 256 KB,
verification in at most 16 ms. They belong to the tables, not to this lane. `fast100` run twice (1–2 MB of proof,
20–50 ms to verify, per flock-128) would exceed the verifier limits. That doesn't matter for an offline verifier, but it would matter
for a SNARK variant.

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

- **Round-by-round first.** Each phase of `rep_sound` is proved as a per-coin lemma: from a doomed state, whatever the
  prover sends, the next coin leaves it with probability at most ε. That is exactly a round-by-round error, and the
  live-coin theorem sums these lemmas. A Fiat–Shamir proof would reuse them.
- **One joint term, kept separately.** The two out-of-domain samples per level are charged jointly, `(μ/2^128)^2` over
  pairs, which is valid with live coins. Their round-by-round error is the unsquared `pairs·μ/2^128` (about 2^-104.5
  at the deepest level for m = 33, with ArkLib's list bound). It will be kept as its own lemma for a Fiat–Shamir accounting.
- **The model is ready for a compiler.** The `Game` model's `coin` nodes are the only randomness, and the tree
  already carries the transcript prefix. A Fiat–Shamir compiler would replace each `coin` by a random-oracle query on
  that prefix.
- **Tools.** VCVio has Fiat–Shamir for Σ-protocols and the forking lemma. ArkLib's compiler from oracle proofs to
  arguments (BCS) and its Fiat–Shamir for IOPs are roadmap phase 6, currently blocked. A direct random-oracle proof is
  the alternative.

**The `seed` coin mode is not Fiat–Shamir** (PR #83, spec §7.2). Coins come from a seed the verifier commits at step
0, so soundness adds only seed hiding and SHA-256 as a PRF. It is the bridge to zero knowledge against a malicious
verifier (§8), not to non-interactive proofs.
