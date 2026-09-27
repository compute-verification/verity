# Flock interactive soundness: design and derivations

This file holds the reasoning behind `ASSUMPTIONS.md`, which lists the theorems and assumptions themselves. Each section
derives or plans an entry there. The long forms, with sources, are lane notes, cited as `note:<id>`.

## 1. The hash: why SHA-512

### 1.1 Why not SHA-256 (Daniel's question: does collision resistance clear 2^-128?)

With SHA-256 trees, two arguments cleared 2^-128, each for `t` up to about 2^64 hash evaluations:

- the straight-line collision-resistance reduction, which covers only provers that know the tables they commit to;
- the random-oracle model.

Rewinding, the only collision-resistance argument for every prover, gives `t·2^-113.6` at m = 33, which never clears
2^-128. The comparison, the alternatives weighed and the cost of SHA-512 are in
`note:20260926T2230Z-finding-sha512-merkle-choice`. In short: the prover costs about 1.5× per hashed byte on GPUs (under
2% of a proof), proofs grow about 36%, and verifier time stays about the same.

### 1.2 The rewinding bound as a function of hash width

Plain collision resistance is Daniel's preferred model: no random oracle, no knowledge assumption. For arbitrary
provers it needs the rewinding argument. Its cost is a square root on each Merkle-committed level, so its bound depends
on the hash's output width `n`.

- **The argument** (§2): openings off the plurality table cost collisions, found by running the prover twice.
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
every prover, up to `t ≈ 2^110`. ASSUMPTIONS.md §1.2 states it, and §2 below derives it.

**Outside the verifier:** the statement's own data commitments (frame-v3 roots, row digests, Halevi–Micali leaves
hashed in the circuit) are tight collision-resistance terms. At `t ≥ 2^65` they too need a hash of at least 288 bits.
That is a separate commitment-scheme decision.

## 2. The compiled theorem: its terms, its number, its proof

The protocol is ASSUMPTIONS.md §1.1's table (the `fast100` profile twice on one root, live `os` coins, a statement of
§16) in the adopted variant: SHA-512 Merkle caps, and the coin server keeping each round's bytes.

**The committed values of an arbitrary prover.** Let `P*` be any classical prover. Without loss of generality it is
deterministic, since a randomized prover is an average of deterministic ones. Its **committed table** is the plurality
table `T*(P*)`: at each level-0 column position, the column `P*` opens there most often over the coins, with a path
that verifies against its cap. `T*` is a function of `P*`, as in every rewinding-based binding. It is not computed by
anyone.

**The terms of the bound** `tableError + Σ_ℓ √(N_ℓ·Q_ℓ·Adv_ℓ) + Adv_×`:

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

with `log2(√2·S_m)` = 10.7 at m = 22, 16.3 at m = 33 and 17.3 at m = 35 (§1.2). That is at most 2^-128 for `t` up to
2^110.7 at m = 35 and 2^117 at m = 22. The hash term overtakes the statistical one from `t ≈ 2^43` (m = 35) to
`2^50` (m = 22).

**The proof, and where it stands in Lean.**

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

## 3. Knowledge soundness: the extractor, its analysis, its cost

Soundness says the committed witness satisfies the statement. A claim about *which* data a proof used needs that witness
computed, because hiding leaves open to essentially any row. So "these proofs used the registered weights" (§10) needs
it, and so does the recursion track, whose outer proof must be a proof of knowledge. The long form, with the effort
table, is `note:20260926T2330Z-finding-knowledge-soundness`.

- **The extractor.** It reruns any classical prover `P*` from the start, about `(N₀/Q₀)·ln(1/(ζβ))/ε` times, where ε
  is the probability that `P*` is accepted. That sees all but a fraction ζ of level 0's columns, except with
  probability β. Level 0 is committed in the first message, so every run opens the same cap.
  - Two runs that open one position differently give a SHA-512 collision.
  - Otherwise the extractor list-decodes the observed columns, treating the rest as erasures, and outputs a witness that
    satisfies the statement.
- **Binding to registered data.** Given the registrant's opening `(W, R)`, the extracted rows equal `W` wherever the
  statement links the root, or tree binding and then Halevi–Micali binding give a SHA-512 collision.
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
  erasures, so coverage must be about 99%); bridging CompPoly, ArkLib and `Model`. It is classical only (§9).

## 4. Where the spec follows upstream rather than the paper (spec §19), and the bound

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

## 5. Accounting findings

- **2^-195.5 holds for m ≤ 33 only.** At m = 34 and 35, `fast100` has a seventh Ligerito level. Its query term
  (2^-102.6) moves the table bound to 2^-195.43. PROTOCOL.md §15 quotes 2^-195.5 per table, from the m = 33 ledger.
  Both are far below 2^-128.
- **Terms flock-128's ledger does not charge,** none moving the total by 0.001 bit: the list factor `L₀` before
  Ligerito's binding (paper Appendix A), the `r = 1` set (A5), the pin (A8), the link points (A14), MCA and fold terms
  per folding round rather than per level, and consistency batching at each level's own list size.
- **List sizes.** ArkLib proves the Johnson list bound as `1/(2ηρ)` (50 at rate 1/2). Flock uses the sharper
  `1/(2η√ρ)` (35). The bound uses ArkLib's, and the difference is below 0.001 bit.

## 6. Zero knowledge: the masked protocol and the reference prover

The targets and expected assumptions are ASSUMPTIONS.md §6.

**The masked protocol** is lane `flock-netlist`'s M1/M2 (PR #83). It consists of VEIL-style masking of the sumchecks,
Reed–Solomon padding in Ligerito over GF(2^128), and Halevi–Micali leaf commitments. The model in `Model/*` is written
so that these slot in without restructuring:

- the prover's randomness becomes an extra input of the reference prover;
- masks are extra messages or extra committed columns in the existing phases;
- salted leaves change only the Merkle layer.

**The reference prover** (agreed with Daniel):

- The production prover (Rust and CUDA) cannot run in Lean, so a slow, spec-level Lean reference prover stands in for
  it. It is a deterministic function of (statement, witness, prover randomness, verifier coins).
- **Its theorems:** completeness (the Lean verifier accepts its honest output), and zero knowledge (the simulator
  theorem, over uniform prover randomness).
- **Differential tests** tie production to it byte for byte: on the same inputs, seeds and coins, the production prover
  must emit exactly its messages, which pins down mask placement.
- **Still trusted:** that equivalence beyond the tested inputs, and the quality of the operating system's and ChaCha's
  randomness.

## 7. Staging against a16z's security stages

The work is staged along a16z's zkVM security stages (Thaler, "The path to secure and efficient zkVMs", March 2025).
Stage 1 covers the protocol: 1a PIOP soundness, 1b commitment binding, 1c the compiled argument, 1d the constraint
system against its semantics, 1e the whole theorem including ZK. Stage 2 is the verifier implementation and Stage 3 the
prover implementation.

a16z's stages are written for zkVMs: SNARKs proving a virtual machine's execution, made non-interactive by
Fiat–Shamir. Our setting differs in three ways:

- it is interactive, with live coins;
- there is one Boolean-circuit prover;
- the verifier is written in Lean and runs offline.

We follow the staging and vocabulary, and differ where the setting does.

| a16z stage | What it asks | Here | Follows, or deliberately differs |
|---|---|---|---|
| **1a** PIOP soundness | A machine-checked soundness proof of the polynomial IOP | `table_sound` at the oracle layer: the PIOP with Ligerito as an interactive oracle commitment, information-theoretic, per table, against any prover. | Follows, modulo A1 (our level 4a); proving A1 is level 4b, not plannable at the Johnson bound (a unique-decoding profile would be, since ArkLib proves those gaps). |
| **1b** commitment binding, under a cryptographic assumption or an idealized model | Binding of the polynomial commitment | The Merkle layer: plain SHA-512 collision resistance, for every prover, by rewinding to the plurality table (§2). Ligerito's commitment binds to a list of at most L₀ codewords, not to one polynomial (ASSUMPTIONS.md §2). The random-oracle version is only a cross-check. | Follows. The model is named per result, as a16z asks. |
| **1c** the compiled argument | With Fiat–Shamir, security in the random-oracle model | **Deliberately different: no Fiat–Shamir.** The oracle protocol plus Merkle commitments (1b) plus session binding, which needs no assumption since the coin server keeps each round's bytes. | Differs. No random oracle for coins, avoiding the article's main caveat; a Fiat–Shamir variant is future work (§8). |
| **1d** constraint system against semantics | The constraint system is equivalent to the VM's semantics | No VM: 1d is the lowering theorem, that the block R1CS (spec §16) is satisfied exactly by correct evaluations of the named circuit (level 3, another lane). | Differs in content, not intent: "the circuit is the meaning". Circuit against hardware is checked by capture replay, outside formal scope. |
| **1e** glue, and ZK if intended | One theorem, including zero knowledge | One theorem from the verifier's acceptance to "the committed values satisfy the named circuit", for the compiled protocol. Zero knowledge is proved separately (§6). | Follows. **Recursion caveat:** triggered only by the recursion track. Its outer proof must be knowledge-sound (§3), and every component inside it verified. |
| **2** verifier implementation | The implemented verifier (Rust, Solidity, …) matches the Stage-1 protocol | `Flock.Verify` (lane `flock-verifier`) refines `Model`: if it accepts, the model accepts. | Differs: a Lean-to-Lean refinement; the compiler and runtime are trusted, mitigated by agreement tests (ASSUMPTIONS.md §5). |
| **3** prover implementation | The prover implementation is proved complete, and zero-knowledge if intended | A Lean reference prover with completeness and zero-knowledge theorems, tied to production by byte-for-byte differential tests (§6). | **Deliberately different: the production prover is not proved.** Untested equivalence is trusted; its failures cost liveness, not soundness. |

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
   bound (§2), for classical provers.
3. **Each assumption, at the stage that uses it:**
   - 1a: A1;
   - 1b: SHA-512 collision resistance;
   - 1c (interactive): uniform coins, and nothing for the rounds, whose bytes the server keeps;
   - Stage 2: the compiler's correctness, mitigated by tests.

**Performance stages** (proofs of at most 256 KB, verification in at most 16 ms) belong to the tables. `fast100` run
twice (1–2 MB, 20–50 ms) exceeds them, which matters only for a SNARK variant.

## 8. A Fiat–Shamir or SNARK variant (future extension)

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
verifier (§6), not to non-interactive proofs.

## 9. Quantum provers and verifiers

The full analysis, with sources, is `note:20260926T2240Z-finding-post-quantum`.

**The theorem is classical, and only its compiled layer is affected.**

- **The oracle layer holds against quantum provers as it stands.** `table_sound` is statistical, and every message and
  coin is classical. A quantum prover is therefore one more randomized strategy, which `table_sound` already bounds.
- **The compiled layer does not carry over.** §2 rewinds `P*` from a snapshot taken at the commitment.
  - A quantum state cannot be copied, and measuring the first run's openings disturbs it. So there is no second
    independent run, and no plurality table.
  - Collision resistance is also the wrong property. Some hashes that resist quantum collision-finding give commitments
    that do not bind quantum adversaries (Ambainis–Rosmanis–Unruh 2014). The replacement is **collapsing** (Unruh
    2016).
  - Classical proof frameworks, ours and VCVio included, are unsound for quantum adversaries by design.

**What is known, and what applies to live coins.**

- **Kilian's protocol.** Chiesa–Ma–Spooner–Zhandry (FOCS 2021) prove Kilian's argument (one PCP commitment) post-quantum
  sound in the standard model, with a collapsing hash.
- **Our setting.** Chiesa–Dall'Agnol–Di–Guan–Spooner (TCC 2025, ePrint 2025/947) extend this to interactive BCS: any
  public-coin semi-adaptive IOP, a collapse-position-binding vector commitment, and openings at the end. Their Theorem
  6.1 gives `ε_ARG ≤ ε_IOP + Σ_i L_i·(q_i + 5)·ε_collapse + ϵ`.
- **Flock fits:**
  - the coins are public;
  - Merkle caps are a vector commitment, and hash trees over a collapsing hash are collapsing;
  - Ligerito's queries depend only on coins, so the IOP is non-adaptive, hence semi-adaptive;
  - `ε_IOP` is the oracle-layer bound;
  - the kept round bytes fix every message before its coins.

  So **if SHA-512 is collapsing, Flock's interactive protocol is post-quantum sound with error `tableError + negl(λ)`.**
  This is a paper-level corollary, not yet written out.
- **Fiat–Shamir (§8).** The reference is the quantum random-oracle model: `O(t²·ε_rbr + t³/2^λ)` (Chiesa–Manohar–Spooner,
  TCC 2019).

**Is SHA-512 collapsing?** It is conjectured. Random oracles are collapsing (Unruh 2016), and Merkle–Damgård preserves
collapsing (Unruh, Asiacrypt 2016; Fehr, TCC 2018). So SHA-512 is collapsing if its compression function is. No attack
is known beyond generic collision search. Table 1 would need a new property for it (`collapsing/sha-512`).

**The loss makes the corollary asymptotic only.** The reduction's collapse adversary costs `poly(k·L_max/ϵ)` prover runs,
and its advantage is multiplied by `Σ_i L_i·(q_i + 5)`. At m = 33, `L_max = 2^21`, that sum is about `2^30`, and a table
has 150–200 rounds.

- A 2^-128 target needs `ϵ ≤ 2^-128`, hence at least 2^157 prover runs.
- Even at 2^157 queries, the generic quantum collision probability `q³/2^512 = 2^-41`, times `2^30`, is `2^-11`.

**So no concrete post-quantum bound follows at our parameters, for any prover size.** A number would need one of:

- the quantum random-oracle model, with straight-line extraction by Zhandry's compressed oracle, and a target of
  `ε_stat + O(t³/2^512)`, which no one has proved for the interactive case;
- a tighter standard-model rewinding, which is open.

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

- **The special honest-verifier ZK lemma (§11) is unaffected.** For fixed coins the view is classical and
  statistically close to the simulation, and statistical distance bounds quantum distinguishers too. Halevi–Micali
  hiding, masks and padding are statistical.
- **Malicious-verifier ZK with a rewinding simulator** (§11's (a) and (d)) needs quantum rewinding of a verifier whose
  state may be quantum. The straight-line options, a beacon or registered keys, avoid it.
  - The closest result: Lombardi–Ma–Spooner (FOCS 2022) prove Goldreich–Kahan, where the verifier commits to its
    challenges first, post-quantum ZK. It needs the verifier's commitment to be collapse-binding and statistically
    hiding.
  - Constant-round black-box post-quantum ZK is impossible (Chia–Chung–Liu–Yamakawa, FOCS 2021). Ours has 150–200
    rounds, so that does not apply, but adapting LMS22 to coins revealed round by round is research.
  - **Design consequence:** the coin-seed commitment would need a 128-bit quantum margin, so SHA-384 or SHA-512, not
    SHA-256.

**What a post-quantum statement would take:**

1. **The asymptotic corollary, on paper (small).** Show that Flock is interactive BCS over a public-coin, non-adaptive
   IOP, and apply CDDGS25 under "SHA-512 is collapsing". It needs a quantum-cryptography reviewer, and it gives
   `tableError + negl(λ)`, not a number.
2. **A concrete number (research).** Either interactive BCS in the quantum random-oracle model with concrete terms, or
   tighter quantum rewinding.
3. **Lean.**
   - **What exists:** Lean 4 quantum-information libraries (Lean-QIT, 2026; QICLean) with states, channels,
     measurements, entropy and some QKD. They are outside Mathlib and not on our pins.
   - **What no proof assistant has:** quantum adversaries with superposition queries in a game model like ours,
     collapsing, CMSZ21's rewinding engine (Jordan's lemma, Marriott–Watrous), CDDGS25's multi-round argument, or
     Zhandry's compressed oracle. The nearest tools, qrhl-tool (Isabelle) and EasyPQC (EasyCrypt), handle quantum
     random-oracle proofs such as Fujisaki–Okamoto and Full Domain Hash, but no rewinding.
   - **A full Lean statement** is a research formalization program, larger than this whole classical development, and
     not plannable now.
   - **A cheap intermediate:** CDDGS25's theorem as a named `Prop`, like A1, with the application proved (Flock is
     public-coin and non-adaptive, with `ε_IOP = tableError`). It is honest, but adds little while the conclusion is
     asymptotic.

**Recommendation.** Keep the theorem classical, and claim no post-quantum number in Table 1. A fair statement is: the
statistical layer holds against quantum provers; the compiled protocol is post-quantum sound asymptotically if SHA-512
is collapsing (CDDGS25); no concrete post-quantum bound is known at these parameters.

## 10. One root, many proofs

The weights are committed once, as Halevi–Micali leaves under a Merkle root registered as the model's anchor, and every
later proof opens that root. Numbers and sources: `note:20260926T2310Z-finding-root-reuse`.

- **Soundness over N tables**, by union bound: `N·2^-195.5 + N·t·2^{17.3−256}`, plus the root's straight-line binding
  once, `t²/2^513`.
  - The statistical part allows N up to 2^67.5.
  - The rewinding term needs `N·t ≤ 2^110.7`, about 2^30 tables at `t = 2^80`.
  - So the per-table hash term runs out, not the root. The random-oracle version's hash term is global.
  - **Decision:** state security per table, or as a lifetime budget on `N·t`.
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
- **Zero knowledge** must hold against a malicious verifier, including concurrent sessions (Daniel, 2026-09-26). §11
  weighs the options and recommends one.
- **Statement leakage accumulates, and ZK doesn't bound it.** N sessions reveal N statements: workload shapes, timing,
  the shared root, and each proved predicate's truth.
  - Weights never appear, and outputs stay committed.
  - But a predicate over committed outputs gives whoever sees it one bit per proof: over chosen inputs, an oracle to the
    model.
  - **Policy needed:** which predicates are public, and how many.
- **Linkability** is the point of one root, and it is also a leak: every proof visibly shares one model. Unlinkability
  would need a fresh commitment and an equality proof per run, which means re-hashing the weights each time.
  - **Knowledge soundness:** a statistically hiding leaf opens to essentially any row. So "these proofs used the
    registered weights" (even one proof) needs each proof's committed witness extracted, then Halevi–Micali binding.
    That is knowledge soundness (§3), and the circuit-privacy design's composition theorem needs it too.

## 11. Zero knowledge against a malicious, concurrent verifier

**The target.** Daniel, 2026-09-26: the verifier is malicious (two nations checking each other), so honest-verifier zero
knowledge is not an acceptable fallback anywhere. The target is zero knowledge against a malicious verifier, including
many concurrent sessions.

**The lemma every option uses** is special honest-verifier ZK (SHVZK): a simulator for every fixed coin sequence.

- VEIL proves exactly that for its compiler: a simulator whose output equals the real view "for any value of verifier
  randomness" (perfect; Lemma 4.3). Its footnote names the notion "semi-malicious", and Succinct has machine-checked
  the paper's statements in Lean.
- Masked Flock (M1, VEIL-style) must keep that property on binary fields:
  - every leaked GF(2^128) value needs 128 uniform mask bits whose weights span;
  - ring switching must be covered;
  - adversarial out-of-domain points or repeated query positions must not reveal more than the padding covers.

  So the prover must reject degenerate coins, as VEIL's generators exclude a zero mask coefficient.
- What remains is stopping the verifier from choosing coins adaptively, and simulating without rewinding across
  sessions.

**(a) Serialize all sessions.**

- **Zero knowledge.** Coins are committed per round with HM96 at step 0, and a Goldreich–Kahan simulator makes one
  extraction run, then one rewind. Sequential composition of auxiliary-input zero knowledge (Goldreich–Oren 1994)
  covers session after session. Collision resistance only, no setup.
- **Cost.** One session at a time, each about 220 round trips (live, per the scoping estimate).
  - With the counterparty's coin server co-located with the prover (sub-millisecond round trips, as Phase B plans),
    that is 0.1–0.2 s per session, negligible next to proving.
  - Across a border (50–150 ms round trips) it is 11–33 s per session. Each session must then carry many tables in
    lockstep, up to a whole run's (the batch is bounded by prover memory), or throughput falls by the idle fraction.
- **Identities.** A counterparty can defeat per-identity serialization by running sessions under several identities and
  pooling their views. So serialization must be global: at most one session at a time with anyone who could collude with
  the counterparty, which in a bilateral setting means everyone.

**(b) Coins from a joint public beacon.**

- **Zero knowledge: yes, straight-line.** If every coin comes from a beacon that releases its values on schedule
  whatever the prover sends, the verifier chooses nothing. The simulator waits for the values and runs the SHVZK
  simulator. There is no rewinding, so concurrency costs nothing, and quantum verifiers are covered as well.
- **The beacon must not be the counterparty's.** One that can hold back a value until it has seen the prover's message
  forces rewinding again, and one the prover controls breaks soundness. "Joint" therefore means a third party or a
  threshold of them (drand's League of Entropy), or a VDF beacon.
- **Soundness then assumes:**
  - each round's bytes are recorded before that round's beacon value is released (the verifier's clock);
  - the beacon is unpredictable to the prover and unbiasable by it. For drand, that means fewer than a threshold of its
    members collude with the prover, plus BLS unforgeability over pairings: not collision resistance, and not
    post-quantum.
- **Cost.** Rounds pace to the beacon period: about 11 minutes per table at drand's 3 s.
- **Side effect.** With trusted timestamps, transcripts become transferable evidence rather than deniable ones. The
  weights stay hidden either way.

**(c) A straight-line-extractable coin commitment.**

- **Encryption to a key the simulator holds.** In the common-reference-string model, that key is a trapdoor someone
  generated: trusted setup, excluded.
- **Registered verifier keys** (the bare public-key model, Canetti–Goldreich–Goldwasser–Micali 2000).
  - Each verifier registers a public key once, with a proof of knowledge of its secret key. It then encrypts its
    per-round coins under that key, with perfectly binding encryption opened round by round.
  - The simulator extracts the secret key once, at registration, and afterwards decrypts every session's coins
    straight-line, so concurrency costs nothing.
  - There is no trusted party, but there is a public-key assumption (LWE, for post-quantum) and a registration phase
    before sessions.
- **The observable random-oracle model** (Pass 2003; Canetti–Jain–Scafuro 2014). The verifier commits with HM96 as
  today, and the simulator reads the coins off the verifier's SHA-512 queries. It is hash-only and setup-free, but zero
  knowledge then rests on the random-oracle model (soundness is unchanged), and quantum queries cannot be observed.

**(d) Standard concurrent-ZK techniques.**

- **Commitment first.** Public-coin protocols like Flock's cannot be black-box concurrent ZK as they stand
  (Pass–Tseng–Wikström 2009). Committing the verifier's coins first, as in (a), is the necessary first step.
- **Prabhakaran–Rosen–Sahai (2002).** A preamble of ω(log n) slots after the coin commitments lets a recursive rewinding
  simulator extract every session's coins in polynomial time. It needs collision resistance only and no setup, and
  gives real concurrency. The cost is a few dozen extra round trips, and a simulator whose proof and concrete bounds
  are heavy: a large formalization.
- **The timing model (Dwork–Naor–Sahai 1998).** Bounded delays and prover-imposed time-outs keep sessions from nesting.
  It needs collision resistance plus a timing assumption, and is paid for in latency.
- **Non-black-box simulation (Barak 2001)** is impractical here.

**Recommendation,** under Daniel's constraints (collision resistance only where possible, no trusted setup, a malicious
verifier):

1. **Now: (a), global serialization.** Co-locate the counterparty's coin server with the prover, and batch a run's
   tables into one session in lockstep. It needs only collision resistance (the HM96 coin commitments) and the
   Goldreich–Kahan simulator already planned. The cost is operational.
2. **If concurrency is required: (d), a Prabhakaran–Rosen–Sahai preamble.** It is the only option that keeps collision
   resistance only and no setup while allowing real concurrency. The price is extra rounds and a much harder proof.
3. **Otherwise:** (c) with registered keys is the cleanest concurrent option, if Daniel accepts one public-key
   assumption. (b) fits only if both nations accept a third-party beacon for soundness. Both are straight-line, so they
   also sidestep quantum rewinding (§9), which (a) and (d) would need.

All four rest on the SHVZK lemma for masked Flock, which is the first zero-knowledge theorem to prove.

### 11.1 Batching inside one serialized session (Daniel's route, 2026-09-27)

Daniel's constraints are exactly one verifier and no trusted third parties (so no beacon), and throughput matters. His
route: one session carries every table from a time window, and many GPUs or datacenters prove them in parallel. The
verifier commits once, at step 0, to the whole coin schedule. Sessions are serialized, and each is arbitrarily large.

1. **Zero knowledge needs only the sequential, single-rewind simulator.**
   - A session is one protocol with one commitment, however many tables and machines it spans. What forces repeated
     rewinding is several commitments interleaved in time.
   - The Goldreich–Kahan simulator makes one extraction phase: an honest prover on a dummy witness. That is
     indistinguishable from a real run until the final openings, because every earlier message is masked and those
     openings come after the last coin. It then rewinds once, to just after step 0, and simulates every table from the
     extracted coins by SHVZK. The tables' masks are independent, so their simulations compose. The simulator runs in
     expected polynomial time, handling aborts as Goldreich–Kahan do.
   - Serialized sessions compose sequentially.
   - **Conditions:** every prover machine checks each revealed coin against the same step-0 commitment before answering.
     No other interactive zero-knowledge protocol with the verifier (predicates, openings, recursion's outer proof) runs
     alongside a session, unless it is straight-line.
2. **Sharing coins across tables costs no soundness.** Each table's bound holds for every prover strategy, and the other
   tables are just part of that strategy. Each table's coins are still uniform, so the union bound gives
   `Σ tableError` over the session's tables, exactly as for separate sessions. Three conditions:
   - every table's statement and root is in round 0, before any coin;
   - a round's coins are released only after that round's bytes are recorded for every table that uses them. With
     identical coins that is a barrier across machines. Streams derived per table from the one commitment
     (domain-separated) avoid the barrier and keep the single rewind;
   - the two reps of one table keep independent coins, derived per (rep, round). Squaring 2^-97.8 into 2^-195.5 needs
     it.
3. **Latency.** A session lasts one machine's proving work plus about 220 round trips, paced by the slowest machine. With
   tables spread across GPUs, that work is about one table, 0.5–1 s.
   - That gives about 1–2 s with a co-located coin server, 3–16 s across datacenters in one region, and 11–33 s across a
     border.
   - Sessions run back to back, so the batching window equals the session length, and a table waits at most window plus
     session: twice the session length.
   - Throughput scales with machines, not concurrency, until prover memory limits the tables a machine holds in lockstep.
   - The one verifier can run a coin-server instance beside each datacenter, all opening the one committed schedule.
     That keeps round trips sub-millisecond everywhere.

**If batching failed: Prabhakaran–Rosen–Sahai costs.**

- **Structure.** The verifier commits to its secret σ and to 2k² shares of it (`σ⁰ᵢⱼ ⊕ σ¹ᵢⱼ = σ`). Then come k slots,
  each one round trip: the prover sends k random bits, and the verifier opens one share of each of k pairs.
- **Slots.** A session gets the simulator stuck with probability at most `2^-(k − O(h))`, where h ≈ log₂ of all slots in
  the schedule, union-bounded over the N concurrent sessions. So `k ≈ λ + log₂ N + O(log(N·k))` for simulation error
  2^-λ. Taking the hidden constant as 2, that is about 125 slots at λ = 80, N = 2^10, and 170–210 at λ = 128,
  N = 2^10–2^20: 55–95% more round trips than Flock's 220.
- **Growth.** k grows by about 3 slots per doubling of N, and does not depend on session size.
- **Per-session overhead.** 2k² commitments to σ, plus k² openings.
  - In Prabhakaran–Rosen–Sahai, σ is the verifier's challenge, and ours is a whole coin schedule: about 20 KB per
    table schedule, or 4–5 KB if coins are cut to the bits used. So the shares alone are 0.1–2 GB per session.
  - Protecting a short trapdoor instead means an FLS-style OR-branch in the proved circuit and a
    witness-indistinguishable argument for it: a redesign.
- **Lean.**
  - A malicious-verifier ZK framework for the `Game` model (views, simulators, rewinding access).
  - The Goldreich–Kahan simulator, about 1,000–1,500 lines, which batching needs too.
  - The recursive simulator and its random-tape counting argument, about 3,000–5,000 lines; Pass–Tseng–
    Venkitasubramaniam's simplified analysis is the one to formalize.

  That is several times the knowledge-soundness work.
- **"Exactly one verifier"** does not simplify it, because the concurrent model already has one adversarial verifier
  controlling every session. It does make global serialization enforceable, with no identities to multiply. It also
  makes registered keys trivial: one key, registered once with no concurrency, then straight-line simulation, at the
  price of a public-key assumption (LWE).

**Recommendation: batching.** It meets throughput with serialized sessions, and needs only collision resistance plus the
single-rewind simulator we need anyway. Prabhakaran–Rosen–Sahai's rounds, bandwidth and proof stay off the table. If
real concurrency is ever needed, a single registered verifier key is cheaper than Prabhakaran–Rosen–Sahai, if LWE is
acceptable.
