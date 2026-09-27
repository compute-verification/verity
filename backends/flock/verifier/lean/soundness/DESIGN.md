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
- **Tight terms need `n ≥ 128 + 2·log2 t`** (straight-line reductions, the random-oracle bound, a long-lived data root
  checked against a registrant's opening): SHA-256 meets that only up to `t = 2^64.5`, so **at `t ≥ 2^65` SHA-256
  misses 2^-128 in every model.**

**The route that avoids both the random oracle and the knowledge assumption** (adopted by Daniel, 2026-09-26):

1. Build the in-proof Merkle trees with SHA-512, leaves and nodes.
2. Make the round digests assumption-free by having the coin server keep each round's bytes (it already receives
   them).

The compiled bound is then 2^-195.5 plus `t·2^{17.3−256}` plus negligible terms, from plain collision resistance, for
every prover, up to `t ≈ 2^110`. ASSUMPTIONS.md §1.2 states it, and §2 below derives it.

**Outside the verifier,** the statement's data commitments (frame-v3 roots, row digests, Halevi–Micali leaves hashed
in the circuit) fall into two cases (the lifetime-soundness desk study, §1 and §3):

- **Tight against a registrant's opening.** Where someone other than the prover holds an opening, as the registrant
  holds `(W, R)` for the weights root, binding is a straight-line reduction: `t²/2^513` once (`registered_weights`, §3).
  At `t ≥ 2^65` such roots need a hash of at least 288 bits.
- **Not tight for serving roots in a sampled audit.** Serving registers its roots before the draw and nobody else holds
  an opening, so against arbitrary provers they need the rewinding argument too. `Rewinding.bad_sq_le` applies unchanged
  with the unit draw in place of the proof coins: `δ_tree ≤ √(N_s·Q_s·Adv) ≤ t·√(2·N_s·Q_s)/2^256`. Here `N_s` counts
  the leaves under every serving root registered before the draw, and `Q_s` the leaves the sampled units open.
  - `log2 √(2·N_s·Q_s)` is 17.4 to 34.0 over the study's audits, so this term exceeds the Flock proof term by up to
    2^17.
  - Clearing 2^-128 per audit needs `n ≥ 2·(log2 t + 128 + log2 √(2·N_s·Q_s))`: 475 bits at `t = 2^80` for a #23-sized
    audit. SHA-512 serving roots are therefore required for the audit composition. With SHA-256 roots the tree term is
    `t·2^-101.3` for a #101-sized audit with per-value leaves.

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
- **`Adv_×`** is the probability that a single run opens one position two ways that both verify under one cap: level 0
  across the two reps, or a position two queries of one level share. That is a collision found in one run, with no
  rewinding.
- For a false statement, where no witness satisfies it at all, the event is simply "accepted".

**The number.** `B_ℓ` makes at most `2t` SHA-512 evaluations when `P*` makes `t`, plus two verifications. Under the
generic bound for SHA-512, `Adv ≤ (2t)²/2^513`, which gives

> ε ≤ 2^-195.5 + `t·√2·S_m / 2^256` + `t²/2^511`,

with `log2(√2·S_m)` = 10.7 at m = 22, 16.3 at m = 33 and 17.3 at m = 35 (§1.2). That is at most 2^-128 for `t` up to
2^110.7 at m = 35 and 2^117 at m = 22. The hash term overtakes the statistical one from `t ≈ 2^43` (m = 35) to
`2^50` (m = 22).

**The proof, all of it in Lean** (`table_sound_compiled`, `Game/Lock.lean`, `CompiledLock.lean`, `CompiledSound.lean`):

1. **The oracle-model prover `P̃`** sends the plurality table wherever `P*` sends a cap (at a rep's cap, given the run
   so far), and otherwise plays as `P*` (`Lock.oracleStrat`, `plurC`, `C0star`).
2. **Coupling.** The compiled game runs in lockstep with the oracle game (`Game.Lock`): the same coins and messages,
   a cap for each table, the openings as a last message. Along it, the compiled claim evaluated at openings that agree
   with the fixed tables at every requested position equals the oracle claim (`StRel`), so on such runs compiled
   acceptance is oracle acceptance, and `table_sound` bounds `P̃` (`compiled_le_bad`).
3. **Rewinding.** A bad run clashes within the run or opens a position off the table fixed at some cap (`bad_split`).
   Per cap, the second costs `√(N_ℓ·Q_ℓ·Pr[conflict_ℓ])`, where `conflict_ℓ` is two runs sharing the prefix up to the
   cap opening one position differently: the rewinding lemma for any run distribution (`Rewinding.expect_bad_le`) at
   the cap, and Jensen over the prefixes above it (`Lock.leafE_off_le`).
4. **Binding.** A conflict, like a self-clash, is a SHA-512 collision (`Merkle.opening_binding`), so
   `Pr[conflict_ℓ] ≤ Adv_ℓ`.
5. **Rounds.** The kept bytes fix every message before its coins: the `Game` model itself (`send` before `coin`).

Every step depends only on Lean's standard axioms (`Check.lean`).

**Remark: the random-oracle version.** Modeling SHA-512 as a random oracle gives straight-line extraction of the
committed tables (VCVio's Merkle extraction) and a tight bound, `2^-195.5 + O(t²/2^512) + t·(cap entries)/2^512`,
without the square root. It is a cross-check only. The number above does not rest on it.

## 3. Knowledge soundness: the extractor, its analysis, its cost

Soundness says the committed witness satisfies the statement. A claim about *which* data a proof used needs that witness
computed, because hiding leaves open to essentially any row. So "these proofs used the registered weights" (§10) needs
it, and so does the recursion track, whose outer proof must be a proof of knowledge. The analysis and its review are
`note:20260927T0535Z-draft-knowledge-soundness-analysis` and `note:20260927T0600Z-review-knowledge-soundness`. The
adopted form is the review's "fill and decode"; it is proved (`Knowledge.lean`).

- **The extractor.** Level 0 is committed in the first message, so every rerun after it opens under the same cap. The
  extractor:
  - reruns `P*` after its level-0 cap `K` times;
  - fills each level-0 position with the first verifying opening it saw, and a fixed row elsewhere;
  - decodes the full table at the level-0 radius `δ`, over at most `L₀` candidates (`closeMsgs`). The table theorem
    decodes non-constructively; Guruswami–Sudan is needed only by reductions that run the extractor.
- **Why it works.** For any level-0 table `T`, the compiled bound gives
  `Pr[accepted ∧ T uncommitted] ≤ ε_c⁻ + Pr[a run opens a level-0 position off T]`
  (`table_sound_compiled_of_table`), with one `ε_c⁻` for every `T`. A fresh run opens a position off the extractor's
  table in one of two ways, with these costs over the extractor's `K` runs:
  - it conflicts with the extractor run that supplied the row: `K·Adv₀`, a union over the `K` fresh–extractor pairs,
    each a two-run collision finder;
  - it opens a position no extractor run opened: `Σ_p q_p(1 − q_p)^K ≤ N₀/(eK)`.

  One Markov step gives `Pr[extraction fails] ≤ (K·Adv₀ + N₀/(eK))/(ε − ε_c⁻)` (`table_knowledge_sound`).
- **The numbers** (m = 33).
  - With `K = 2^21.6/(ε − ε_c⁻)`, failure is at most `1/4 + (t·2^-244.7/(ε − ε_c⁻))²`. The hash term is squared, so
    `κ = ε_c⁻ + t·2^-243.7`.
  - The product form the audit consumes needs no condition on `ε`:
    `Pr[accepted ∧ extraction fails] ≤ ε_c⁻ + K·Adv₀ + N₀/(eK)` (`table_knowledge_sound_joint_tight`). Rerun list by rerun
    list, `ε·1[fails] ≤ ε_c⁻ + mass`, so no Markov step enters. At `K = √(N₀/(e·Adv₀))` this is
    `ε_c⁻ + t·2^-244.7`, which is `2^-164.7` at `t = 2^80` above `ε_c⁻`.
  - For comparison, the draft's form (fresh runs against `T*`, with a rewinding square root) gives a hash term of
    `t·2^-233.4/(ε − ε_c)` and `κ = ε_c + t·2^-231.4`.
- **Not knowing ε.** For the audit, the extractor is an analysis device, and `K` may depend on `ε` and `Adv₀`. An
  efficient extractor, as recursion needs, doubles `K` until a satisfying witness is found, which is a decidable stop.
  An exponential tail comes from disjoint blocks: the unseen set is the intersection of the blocks' unseen sets, so `r`
  blocks of `2N₀/(e(ε − ε_c⁻))` runs fail with probability at most `2^-r`.
- **Canonical values.** The extractor should output the values behind the `hm96` commit strings, not "one satisfying
  witness". Two satisfying candidates that differ behind a commit string are an `hm96` collision. This, and
  `registered_weights`, need the SHA-512 leaf layout in the model.
- **Binding to registered data.** Given the registrant's opening `(W, R)`, the extracted rows equal `W` wherever the
  statement links the root, or tree binding and then Halevi–Micali binding give a SHA-512 collision.
- **Sampled audits need the link term `δ_link` as well** (the lifetime-soundness desk study, §2 and §3).
  - Its collision finder runs the extractor, so the extractor's run count enters the collision-resistance bound.
  - The analysis is below and in `note:20260927T0705Z-draft-link-term`; the Lean plan is
    `note:20260927T1510Z-draft-expected-time-link-plan`.
- **Risks:** `δ_link`; bridging CompPoly, ArkLib and `Model` for the efficient decoder. It is classical only (§9).

**The link term: expected-time collision resistance, of record** (Daniel, Sep 27). The audit's value layer rests on
`ASSUMPTIONS.md` A2. One audit's knowledge-plus-link term, of record and, for comparison, under the other two
assumptions. $N_s$ is the leaves under the serving roots, $N_0 = 2^{21}$ the level-0 codeword length (m = 33), and $t$
the prover's SHA-512 evaluations. The audits are the lifetime doc's A–D.

| Assumption | Knowledge + link | A | B | C | D |
|---|---|---|---|---|---|
| **Expected-time collision resistance** (A2, of record) | $t\cdot N_s\,(8N_0/e)/2^{256.5}$ | $t\cdot2^{-215.2}$ | $t\cdot2^{-205.6}$ | $t\cdot2^{-199.6}$ | $t\cdot2^{-193.9}$ |
| Strict-time collision resistance (comparison) | $2\sqrt{t\cdot 4N_0\sqrt{N_s/e}/(e\,2^{256.5})}$ | $2\sqrt{t\cdot2^{-226.3}}$ | $2\sqrt{t\cdot2^{-221.5}}$ | $2\sqrt{t\cdot2^{-218.5}}$ | $2\sqrt{t\cdot2^{-215.7}}$ |
| Random oracle (comparison; tree, link and knowledge together) | $t^2/2^{511}$ | $t^2/2^{511}$ | $t^2/2^{511}$ | $t^2/2^{511}$ | $t^2/2^{511}$ |

The first two rows are the terms beyond $\varepsilon_c^-$ and $\delta_{\rm tree}$, which the record doubles (below).

The same terms in $\log_2$ for audit B, next to the rest of its hash bound ($\delta_{\rm Flock}+\delta_{\rm tree}=t\cdot2^{-229.3}$):

| Audit B | $t=2^{64}$ | $2^{80}$ | $2^{100}$ | $2^{110}$ |
|---|---|---|---|---|
| Expected-time (of record) | −141.6 | −125.6 | −105.6 | −95.6 |
| Strict-time | −77.8 | −69.8 | −59.8 | −54.8 |
| Random oracle | −383 | −351 | −311 | −291 |
| $\delta_{\rm Flock}+\delta_{\rm tree}$ | −165.3 | −149.3 | −129.3 | −119.3 |

At $t=2^{80}$ the audits read −135.2, −125.6, −119.6 and −113.9 of record, and −72.2, −69.8, −68.3 and −66.8 under
strict time.

- **Expected-time collision resistance, of record.**
  - **The assumption** (A2; Chiesa–Dall'Agnol–Guan–Spooner–Yogev, ePrint 2024/1434, Theorem 2): an algorithm whose
    *expected* number of SHA-512 evaluations is $T$ finds a collision with probability at most $T/2^{256.5}$.
  - **Why linear in $T$.** This is SHA-512's generic bound for such finders, since
    $\min(1,q^2/2^{513})\le q/2^{256.5}$. It is linear rather than quadratic because a finder can gamble, making
    $2^{256.5}$ evaluations with small probability. The quadratic form $T^2/2^{513}$ would be false for this reason.
  - **The extractor** runs the session once. Only if that run accepts does it rerun, until $k=cN_0/e$ reruns have
    accepted: $(1+k)\,t$ evaluations in expectation, whatever $\varepsilon$, and no estimate of it. It fills level 0
    from the accepted reruns, and the compiled bound's deviation term needs only accepted runs. So with
    $q_p=\Pr[\text{a run accepts and opens }p]$ the missing term is $\sum_p q_p(1-q_p/\varepsilon)^k\le\varepsilon/c$.
    The conflict term is $(k/\varepsilon)\mathrm{Adv}_0$, and bounding the failure by $\min(\varepsilon,\cdot)$ turns
    it into $\sqrt{k\,\mathrm{Adv}_0}\approx t\cdot2^{-244}$.
  - **The value layer** is the plurality of what extractions recover behind each commit string (lifetime doc §2), a
    function of the state at registration. Behind one string, disagreeing with the plurality is at most as likely as
    two independent recoveries disagreeing: $r-p_{\max}\le r-\sum_v p_v^2/r$.
  - **The link finder** picks one of the $N_s$ strings uniformly and runs an extraction. If that recovered the string,
    it runs extractions until one recovers it again, $1/r$ of them in expectation. That is $2(1+k)\,t$ evaluations in
    expectation (Wald), so $\delta_{\rm link}\le2N_s(1+k)\,t/2^{256.5}$, plus one extraction's worth for two drawn
    units that read one string differently.
  - **The bound is relative.** $u^*$ is wrong, so acceptance itself is the bad event:
    $\Pr[\text{accept}\wedge\text{hit}]\,(1-1/c)\le\delta_{\rm tree}+\varepsilon_c^-+\sqrt{k\,\mathrm{Adv}_0}+\delta_{\rm link}$.
    The square root goes.
  - **The constant.** $c=2$ minimizes $c^2/(c-1)$, giving the record's $t\cdot N_s(8N_0/e)/2^{256.5}$ up to lower-order
    terms. It doubles the other terms, to $2\delta_{\rm tree}+2\varepsilon_c^-$, one bit on terms 23 bits below.
  - **Half of it,** $4N_0/e$, would come from a value layer drawn from a sequence of auxiliary extractions (the first
    recovery behind each string, $N_s$ extractions by Wald), with the committed transcript then chosen by averaging
    over that sequence. The record keeps $8N_0/e$, the form the Lean plan proves.
  - **The status.** On paper; the Lean plan is `note:20260927T1510Z-draft-expected-time-link-plan`.
- **Strict time** (comparison). Balancing the knowledge term's $N_0/(eK)$ against the link finder's cost, which grows
  linearly with the extractor's $K$ reruns, leaves a square root. The review's first estimate was $2^{-65.3}$ at
  $t=2^{80}$, audit B. Improving substantially on the strict-time loss would improve the analysis of Schnorr's
  protocol (CDGSY24).
- **The random-oracle column** (comparison, lifetime doc §7). The extractor reads the commit strings, the `hm96`
  openings and the tables off the prover's queries, so `δ_tree`, `δ_link` and the knowledge term collapse into one
  global $t^2/2^{511}$. `ASSUMPTIONS.md` §4 keeps the random oracle as a cross-check only.
- **Either way,** the table theorems (`table_sound_compiled`, `table_knowledge_sound_joint`) and `δ_tree` stand as
  proved under strict-time collision resistance. A2 enters only the value layer.

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

The full analysis, with sources and the Lean landscape, is `note:20260926T2240Z-finding-post-quantum`.

**The theorem is classical, and only its compiled layer is affected.**

- **The oracle layer holds against quantum provers as it stands.** `table_sound` is statistical, and every message and
  coin is classical, so a quantum prover is one more randomized strategy.
- **The compiled layer does not carry over.** §2 rewinds `P*`, but a quantum state can't be copied, or measured without
  disturbing it. Collision resistance is also the wrong property: the replacement is **collapsing** (Unruh 2016).
  Classical frameworks, ours and VCVio included, are unsound for quantum adversaries by design.

**What applies to live coins.** Chiesa–Dall'Agnol–Di–Guan–Spooner (TCC 2025, ePrint 2025/947, Theorem 6.1) prove
interactive BCS post-quantum sound: `ε_ARG ≤ ε_IOP + Σ_i L_i·(q_i + 5)·ε_collapse + ϵ`, for a public-coin semi-adaptive
IOP with a collapse-position-binding vector commitment and openings at the end.

- Flock fits: its coins are public and its commitments are Merkle caps. Its queries depend only on coins, `ε_IOP` is
  the oracle-layer bound, and the kept round bytes fix every message before its coins.
- So **if SHA-512 is collapsing, Flock is post-quantum sound with error `tableError + negl(λ)`.** This is a paper-level
  corollary, not yet written out.
- SHA-512 is collapsing if its compression function is (Unruh 2016; Fehr 2018). Table 1 would need a new property,
  `collapsing/sha-512`.

**No concrete bound follows at our parameters, for any prover size.**

- The reduction's collapse adversary costs `poly(k·L_max/ϵ)` prover runs, and its advantage is multiplied by
  `Σ_i L_i·(q_i + 5) ≈ 2^30` at m = 33.
- A 2^-128 target needs `ϵ ≤ 2^-128`, hence at least 2^157 runs. At that many queries even the generic `q³/2^512` is
  2^-41, which is 2^-11 after the loss.
- A number needs straight-line extraction in the quantum random-oracle model (Zhandry's compressed oracle), or tighter
  rewinding. Both are open for the interactive case.
- The hash itself has ample margin: collisions cost 2^170.7 queries by Brassard–Høyer–Tapp. `seed` coins (PR #83) sit
  at 2^128, from Grover on the 256-bit seed.

**Zero knowledge against quantum verifiers.**

- Special honest-verifier ZK (§11) is statistical, so it is unaffected.
- Malicious-verifier ZK with the single-rewind simulator needs quantum rewinding. The registered-verifier-key route
  (§11.2) is straight-line.
- A coin-seed commitment that must hold against quantum verifiers needs SHA-384 or SHA-512, not SHA-256.

**Recommendation.** Keep the theorem classical, and claim no post-quantum number in Table 1. The statistical layer
holds against quantum provers. The compiled protocol is post-quantum sound asymptotically if SHA-512 is collapsing
(CDDGS25), and no concrete bound is known at these parameters. A Lean statement would be a research program, because no
proof assistant has quantum rewinding. The cheap intermediate is CDDGS25 as a named `Prop`, with Flock's application
proved.

## 10. One root, many proofs

The weights are committed once, as Halevi–Micali leaves under a Merkle root registered as the model's anchor, and every
later proof opens that root. Numbers and sources: `note:20260926T2310Z-finding-root-reuse`.

- **Soundness over N tables**, by union bound: `N·2^-195.5 + N·t·2^{17.3−256}`, plus the root's straight-line binding
  once, `t²/2^513`.
  - The statistical part allows N up to 2^67.5.
  - The rewinding term needs `N·t ≤ 2^110.7`, about 2^30 tables at `t = 2^80`.
  - So the per-table hash term runs out, not the root. The random-oracle version's hash term is global.
  - **Decision:** state security per table, or as a lifetime budget on `N·t`.
  - For sampled audits the count is audits, not tables, and the composition needs knowledge soundness
    (ASSUMPTIONS.md §7).
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
- **Zero knowledge** must hold against a malicious verifier. §11 states the decided model: batched, serialized
  sessions.
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

## 11. Zero knowledge against a malicious verifier: batched, serialized sessions

**The target.** The verifier is malicious (two nations checking each other), so honest-verifier zero knowledge is not an
acceptable fallback anywhere (Daniel, 2026-09-26). There is exactly one verifier and no trusted third party, and
throughput matters. Daniel decided the model on 2026-09-27: batched, serialized sessions (§11.1). The options weighed
before that decision (serialization, a public beacon, straight-line-extractable commitments, concurrent-ZK preambles)
are in `note:20260927T0015Z-finding-malicious-verifier-zk-options`.

**The lemma** is special honest-verifier ZK (SHVZK): a simulator for every fixed coin sequence.

- VEIL proves exactly that for its compiler: a simulator whose output equals the real view "for any value of verifier
  randomness" (perfect; Lemma 4.3). Its footnote names the notion "semi-malicious", and Succinct has machine-checked
  the paper's statements in Lean.
- Masked Flock (M1, VEIL-style) must keep that property on binary fields:
  - every leaked GF(2^128) value needs 128 uniform mask bits whose weights span;
  - ring switching must be covered;
  - adversarial out-of-domain points or repeated query positions must not reveal more than the padding covers.

  So the prover must reject degenerate coins, as VEIL's generators exclude a zero mask coefficient.

### 11.1 The model: batched, serialized sessions

A session carries every table from a time window, and many GPUs or datacenters prove them in parallel. Sessions are
serialized, and each is arbitrarily large. `ASSUMPTIONS.md` §2 states the rules. Why they give zero knowledge and keep
soundness:

1. **Zero knowledge needs only one rewind point** (the "single-rewind" simulator of §9).
   - A session has one coin commitment, however many tables and machines it spans. What forces rewinding to several
     points is several commitments interleaved in time.
   - The Goldreich–Kahan simulator makes one extraction phase: an honest prover on a dummy witness. That is
     indistinguishable from a real run until the final openings, because every earlier message is masked and those
     openings come after the last coin. It then rewinds only to just after step 0, and simulates every table from the
     extracted coins by SHVZK. The tables' masks are independent, so their simulations compose. Its rewinds, aborts and
     completion estimate are as `ASSUMPTIONS.md` §6 states them, in expected polynomial time.
   - Serialized sessions compose sequentially (auxiliary-input zero knowledge, Goldreich–Oren 1994).
2. **Coins committed under one root, per (table, repetition, round).** The commitment is a SHA-512 Merkle root over
   Halevi–Micali commitments to each coin, so each table's stream is a subtree. That keeps soundness to statistical
   hiding and zero knowledge to collision resistance. Deriving streams from one seed with `derive()` would add a PRF
   assumption, and the prover could not check a coin without the seed.
3. **Shared commitment, per-table streams: soundness unchanged.** Each table's bound holds for every prover strategy,
   and the other tables are just part of that strategy. Its coins are uniform and released only after its own round's
   bytes are recorded. So the union bound gives `Σ tableError` over the session's tables, and there is no barrier
   across machines. Squaring 2^-97.8 into 2^-195.5 needs independent coins per repetition.
4. **Latency.** A session lasts one machine's proving work plus about 220 round trips, paced by the slowest stream.
   - With tables spread across GPUs and a coin server co-located per datacenter, that is about 1–2 s.
   - Sessions run back to back, so a table waits at most one window plus one session: twice the session length.
   - Throughput scales with machines, until prover memory limits the tables a machine holds in lockstep.

### 11.2 Reserve notes

If real concurrency is ever needed:

- **A single registered verifier key** (the bare public-key model) is the cheaper route.
  - The verifier registers one public key once, with a proof of knowledge of its secret key, and encrypts its coins
    under it with perfectly binding encryption, opened round by round.
  - The simulator extracts the key once, then simulates every session straight-line.
  - Exactly one verifier makes the registration trivial. The price is a public-key assumption (LWE, for post-quantum).
- **Prabhakaran–Rosen–Sahai.**
  - **Structure.** The verifier commits to its secret σ and to 2k² shares of it. Then come k slots, each one round trip.
  - **Slots.** A session gets the simulator stuck with probability at most `2^-(k − O(h))`, where h ≈ log₂ of all slots
    in the schedule, union-bounded over N concurrent sessions. So `k ≈ λ + log₂ N + O(log(N·k))`. Taking the hidden
    constant as 2, that is about 125 slots at λ = 80, N = 2^10, and 170–210 at λ = 128: 55–95% more round trips than
    Flock's 220. k grows by about 3 per doubling of N.
  - **Per-session overhead.** σ is a whole coin schedule (4–20 KB), so the shares cost 0.1–2 GB per session. Protecting a
    short trapdoor instead means an FLS-style OR-branch in the circuit.
  - **Lean.** About 3,000–5,000 lines on top of the Goldreich–Kahan simulator. Pass–Tseng–Venkitasubramaniam's
    simplified analysis is the one to formalize.
  - Exactly one verifier does not simplify it.

## 12. The audit layer: the integrity profile

`docs/audit-protocols.md` §4 is the plan and `ASSUMPTIONS.md` §9 the list of theorems. The full rationale is in
`note:audit-lean/20260927T0736Z-finding-audit-layer-design`. The points a reviewer needs:

- **Strategies, not `value`.** `value` can exceed every strategy's `prob` when a branch has no strategy. So the audit
  theorems take per-strategy hypotheses and give per-strategy conclusions (`Game/Prob.lean`), as `table_sound_exec`
  does.
- **Knowledge soundness and the link term.** Halevi–Micali leaves open to essentially any value, so a unit is wrong
  relative to a committed transcript defined by extraction. A drawn wrong unit is charged to one of two named events:
  extraction fails, or it disagrees with the transcript (`cover`, from `correct_congr`). The oracle layer is the special
  case where the registration is the transcript (`Partition.oracle`).
- **Transcripts are total** (`Fin C.N → Bool`), and the committed set is derived. `Circuit.run` is the one evaluator,
  and every composition lemma is a strong induction over gate indices.
- **Two-stage.** Second-stage coins are drawn for every coarse unit, and only the drawn units' coordinates are used.
  Tightness and the dilution counterexample hold for any first-stage law. Minimax uses the supporting line of
  `s ↦ C(n − s, K)` in `ℕ`, where real convexity fails.
- **The Flock instantiation, oracle layer** (`Audit/Flock.lean`). `table_sound` splits into its `value` form
  (`table_value_sound`), which projects a batched session onto one table through `value_interleave_le`. The lowering
  is a named hypothesis.
- **The compiled layer** (`Audit/Extraction.lean`, `Audit/FlockCompiled.lean`, `Audit/FlockBatched.lean`). The
  table's extractor is randomized and its bound depends on the prover, so the audit carries, per drawn unit, the joint
  probabilities `ks = Pr[accept ∧ extraction fails]` and `link = Pr[accept ∧ extraction disagrees]` over the fresh run
  and the reruns, with `cover: Pr[accept] ≤ ks + link` at a wrong drawn unit, and bounds that are functions of the
  prover's state (the form agreed with flock-soundness). A deterministic extractor is the special case
  (`Analysis.toExtraction`), so `extraction_audit_le` subsumes `audit_le`.
  - Several tables (`FlockBatched.lean`): the protocol's batched session (`sessionB`); each drawn unit is read from
    one table of the plan, `ks` is `Pr[accept]` times that table's extraction failure (the product is the joint
    probability, by independence), and ε_ks is the largest table's `joint_le_B'` bound averaged over the draw.
  - One table on its own (`FlockCompiled.lean`): `ks` is `table_knowledge_sound_joint_tight`'s left side.
  - The decoder is per drawn unit, since two drawn units that share a wire hold it in their own columns. The circuit
    `C` stays abstract; for a program it is the rows the verifier parses (`Prog.circuit`), and
    `loweringSoundB_of_place` discharges the lowering from each unit's `UnitPlace`, given L1 and the placement.
  - The link event is the weakest one (every satisfying candidate disagrees). `δ_link` stays named until the link
    theorem lands. Expected-time collision resistance is decided (§3), and its extractor replaces the fixed `Kr` reruns.
