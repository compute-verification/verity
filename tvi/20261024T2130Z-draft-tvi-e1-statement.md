---
id: training-20261024T2130Z-draft-tvi-e1-statement
campaign: training-vs-inference
lane: training
kind: draft
status: open
repo: compute-verification/verity
origin: E1 (statement), worker of @training (bc-b43da88b-5d25-4463-9b8c-da4df6f84207), phase 1 of crux 6
---

# E1: the production covering lemma, its proof obligations, a counterexample per assumption, and a check of the console notes' formulas

Crux 6 of note:console/20261024T1900Z-draft-training-vs-inference-program, in the definitions of
note:console/20261024T1810Z-finding-production-bound-red-team ("the red-team note" below; "the program" is the first).
Math only, no Lean and no compute. Every number below is reproduced by `tvi/e1_arith.py` on branch
`cursor/tvi-e1-4207` (plain Python, no inputs). Deck formulas are listed for Daniel in section 5: the deck
(`website-gpu-lenses`) is not reachable from this VM.

## Verdict

1. **One statement covers all three regimes** (section 1.3). Every regime reduces to one number, `W(S)`: the
   unregistered bits per training token that the blocks read, summed over the blocks. Then
   `rho(S) = max{1, eta * kappa * W(S) / h}` with `kappa = phi*h/(6P)`. `kappa` is honest inference compute per token
   over training compute per token: 1/3 for prompt-only traffic, 0.40 for the red-team note's own `phi` and
   DeepSeek-V4-Pro, and about 4 for reasoning traffic. The theorem bounds `W(S)` from below in each regime under (U),
   (L), (C), (D) and (E). An explicit covering attains each bound, up to a factor `k*/k` in regime III.
2. **The cliff formula is right at `S = k+`, but its 2/3 is two factors.** It is
   `kappa * 2 * X * F/h`, where `kappa = 1/3` (prompt-only traffic) and the 2 comes from each block spending half its
   `k` on state. It reproduces the program's 1e3 and 20 (1.1e3 and 21). The value is not flat above `k`: it is 2.2e3 at
   `S = 1.5k`, 3.3e3 at `2k`, and grows about linearly, `8 eta kappa (S/k) F/h`. It meets the full-training value at
   `S_III ~ beta P k / (8 d_eff F)`, about 24k for a full-rank fresh residual and about 1300k at rank 64.
3. **The cap condition is 4x looser than stated, and the narrowest map sets it.** The exact threshold is
   `k* = 4 d_in^2 beta_w beta_x / beta_v`, by AM-GM over a tile's rows and tokens, against the note's
   `d_in^2 beta_w beta_x / beta_v`. At `beta_x = beta_v/10` that is 1.0 MB, not 245 kB, for `d_in = 7168`, and 189 kB
   for an expert down-projection (`d_in = 3072`). With all betas equal, a square map never breaks the cap, and an
   up-projection breaks it at `4 d^2 beta`, about 10 MB, not at `d^2 beta`, about 2.5 MB.
4. **Two rows of the rho table are wrong, and two need a different regime.**
   - Head-only training: 4.8e4 at 10 kB, not about 5e5. At 1 MB the cheap-input tile attack costs about 30, because
     the cap fails above 61 kB.
   - FP8 training: 4.2e6 and 4.2e4 at `beta = 0.1`, not about 1e7 and 1e5.
   - LoRA: the `r/d` arithmetic is right, but in the per-value regime the factor is about `3r/d`. At 1 MB a block holds
     several layers of a rank-8 or rank-16 adapter, so those rows are cliff-regime and need measured `F`.
   - TinyLoRA's "about 1" holds for data generated inside the block. On fresh SFT data it is `eta*kappa`, the inference
     tax: about 83 at the 10 kB headline `c`.
5. **The accumulator corollary holds as a lower bound, and its lemma is already proved.** The reduction-tree lemma it
   rests on is the wiring lemma `Nci.Cover.Guarantees.PartialSumsRead`, proved on `main` with no premise. Adding one
   named premise (a block that finishes a token's products reads `h_eff` bits about that token) gives
   `rho >= c h_eff b_ps / (6 k^2)`. That is 6.2e7 at 10 kB and 6.2e3 at 1 MB with `b_ps = 1` gate per partial-sum
   crossing, against the note's 5e7 and 5e3.
6. **Most of the deck's statement already has a proved Lean counterpart.**
   - Regime I is where Lean's depth hypothesis cannot hold: the bound is silent there.
   - Regime II has the shape of `CoveringBoundPriced` (branch) and `CoveringBoundAdditive` (main):
     `rho >= gamma * eta * kappa * (S/k) * (c_a/h)`, with `gamma = 1` priced and 4 additive. The cliff's height is
     the step from Lean's per-token cut `c_a` (one activation, at most 16 bits) to the fresh residual `F` (about
     14 kbit).
   - In regime III the proved forms (bf16, `c_a = 16`) beat the production bound at 10 kB and lose to it at 1 MB.
     `CoveringBoundAdditive` at a uniform `c_w = 2.3` gives 33x the production bound at 10 kB and 0.33x at 1 MB;
     `CoveringBoundPriced` at the mean price 6.7 gives 24x and 0.24x.
   - The production bound itself, the crossing lemma and the locality lemma are not in Lean.
7. **Three definitional fixes.**
   - The red-team note's theorem `|T| >= c w(V)/(k + O(log|T|))` should read `|𝓘| >= w(V)/(k + O(log|T|))`.
   - `beta` and `F` must be fresh relative to `theta`'s own activation on the same token. The in-hand pythia activation
     numbers (3.1-4.1 bits per value at step 1) are not that, since fresh content at step 1 is about 0.
   - `beta` is relative to the registered `theta`, so it grows with the steps since registration. The deck needs the
     run's average `beta`, and the ratchet (re-registering derived weights) resets it.

## 0. Notation, and the one identity

Console symbols, which the deck uses:

| Symbol | Meaning |
|---|---|
| `c` | a block's compute budget (FLOPs); 1.4e17 in the headline |
| `k` | a block's unregistered input bits; 10 kB = 8e4 bits, 1 MB = 8e6 bits |
| `T`, `\|T\|` | the reference training circuit and its compute, about `6P` per token (dense) |
| `P` | active parameters per token (49B for DeepSeek-V4-Pro) |
| `𝓘` | the covering's blocks; `rho = c\|𝓘\|/\|T\|` |
| `h` | bits per raw token, about 17 |
| `phi` | honest inference FLOPs per unregistered input bit; `eta = c/(k phi)` |
| `kappa` | `phi h / (6P)` = honest inference FLOPs per token / training FLOPs per token = `(1+g)/3`, `g` the generated:prompt token ratio |
| `beta`, `beta_w`, `beta_x`, `beta_v` | fresh bits per counted value, per updated weight, per input coordinate, per output value |
| `d_eff` | `P / V_tok`, with `V_tok = sum over active maps of min(d_in, d_out)` per token; FLOPs per counted value are `6 d_eff` |
| `S` | bits of unregistered training state every forward pass depends on, decodable within `t` (the program's definition) |
| `F(S)` | fresh bits per token per crossing: `K^t(x \| theta, token, state the receiving block holds)` for an activation `x` crossing into a block |
| `t`, `lambda` | `t = O~(c)`, the per-block time bound; `lambda = O(log\|T\|)` for indices |

Lean's symbols collide with these. Lean's `s` is the console's `c`. Lean's `c` (`= r*b`), `c_w`, `c_a` and `p i` are
bits per weight or per token cut, so they are the console's `beta`-like quantities. Lean's `k` is `k`. Lean's slowdown
`s n / |C|` is the console's `rho`.

**The identity.** Let `W` be the unregistered bits per training token read by a covering's blocks, summed over the
blocks. If the blocks are bound by input bits, then `|𝓘| = N_tok W / k`, and

$$\rho = \frac{c\,|\mathcal I|}{|T|} = \frac{c}{k}\cdot\frac{W}{6P} = \eta\,\varphi\,\frac{W}{6P} = \eta\,\kappa\,\frac{W}{h}.$$

If the blocks are bound by compute instead (each does `c` of useful work), `rho` is about 1. So in general
`rho = max{1, eta kappa W/h}`. Each regime below is a statement about `W`.

## 1. The statement

### 1.1 Objects

- `T` is the training circuit of a fixed recipe, with registered parameters `theta` and unregistered data `D`.
- A run declares a circuit `C` and a covering `𝓘` of it. A covering is blocks `I`, sets of gates that may overlap and
  recompute, with `|I| <= c` and `|in(I) \ theta| <= k`, which together compute every non-input gate of `C`.
- The counted values `V` of `T` are the outputs of parameterized maps: `min(d_in, d_out)` per token per map,
  deduplicated by dependency cone (the red-team note's section 4 rule). A weight `w(v) >= 0` makes `w` additive on
  sets: `w(A) = sum over v in A of w(v)`.
- The information measure is `K^t(. | theta)`, time-bounded conditional Kolmogorov complexity, applied block by block.
  Section 3.1 shows that nothing else works.

### 1.2 The named assumptions

- **(E) Exactness.** `C` computes `T`'s recipe bit for bit, so each `v` in `V` is a function of `C`'s wires.
  - Strong form, which Lean uses: containment, where each `v` is a word of gates of `C`.
  - Capability form: replace `w(V)` by its minimum over recipes that reach the target capability. The attacker picks
    the recipe, the precision and the state size `S`.
- **(U) Uniformity.** There is a fixed program `G`, of size `O(1)` given the recipe, such that `G(theta, i, in(I_i))`
  outputs block `I_i`'s gate list (gates, gate types, and wiring into `theta`) in time `t`. Block structure is thus a
  checked function of `theta` and the block's own counted inputs. Data-dependent structure, such as MoE routing, must be
  computed inside the block from counted inputs, or read as counted input.
- **(L) Locality.** Each `v` in `V` is decodable by a fixed decoder, in time `t`, from the wires of a single block. The
  locality lemma (section 1.5) replaces (L) with a charge at the nonlinearity that consumes a split value.
- **(C) Cap.** No admissible block computes, by an explicit strategy, a set of counted values of weight above
  `k + lambda` from at most `k` bits. Its checkable form has two clauses.
  - **(C1) Tiles.** Take every parameterized map `d_in -> d_out`. A tile is `R <= d_out` rows of the map's update and
    `Q` tokens' inputs, with `d_in (R beta_w + Q beta_x) <= k`. The condition is `max over tiles of R Q beta_v <= k`.
    Solving it exactly (AM-GM, then the `R <= d_out` cap) gives the threshold:
    - `k* = 4 d_in^2 beta_w beta_x / beta_v`, when `beta_x/beta_v <= d_out/(2 d_in)`;
    - `k* = infinity` (never fails), when `d_out beta_v <= d_in beta_x`;
    - `k* = d_in d_out^2 beta_w beta_v / (d_out beta_v - d_in beta_x)`, otherwise.

    The cap holds for `k <= min over maps of k*`, so the narrowest `d_in` sets it.
  - **(C2) Held subnetworks.** For every subnetwork whose update fits in `k`, the weight of its counted values per
    token is at most its fresh input per token. This is the clause the cliff regime runs into.

  (C) follows from (D), since every tile or held subnetwork is a compressor a block can run. So (C) adds no logical
  strength. It is named because it can be checked before (D), and because it marks where the per-value weight must
  come down.
- **(D) Decoding, as computational depth.** For every `A` contained in `V` with `w(A) > k + lambda`,
  `K^t(A | theta) > k + lambda`.
  - Training is a function of `theta` and the data, so `K(V | theta) <= |D| + O(1)`, and even less when the data is
    seeded. (D) says the counted values are nevertheless `t`-deep: no description of fewer than `w(A) - lambda` bits
    yields them within one block's time.
  - The cliff regime uses this at one value per crossing, **(D_F)**: the activation crossing into a block has
    `K^t >= F(S)` given `theta`, the token and the state the block holds. Parts shared across tokens, such as the basis
    of a common subspace, are counted in `S`.
  - **Two ways to state it.** As a named assumption (recommended, `training-depth`), with measured `beta` as its
    parameter and B2's shortcut attempts as the evidence against it. Or folded into the weight, `w(v) :=` marginal
    `K^t`: then (D) holds by definition and the theorem is unconditional, but `w` can only be bounded from above (by
    exhibiting compressors), so every number computed from it is an upper bound on the bound. The deck should say which
    one it uses. Today it implicitly uses the first.

### 1.3 Theorem (the production covering lemma)

Assume (E), (U), (L) and (D).

**(a) Covering.** Every covering has `|𝓘| >= w(V) / (k + lambda)`, so

$$\rho \;\ge\; \frac{c\,w(V)}{|T|\,(k+\lambda)} \;=\; \eta\,\kappa\,\frac{w(V)/N_{\rm tok}}{h}\,\Big(1+\frac{\lambda}{k}\Big)^{-1}.$$

**(b) The three regimes.** Use the largest weight that (C) allows at state size `S`. Then
`rho(S) = max{1, eta kappa W(S)/h}`, with `W(S)` as follows.

| Regime | `W(S)` (bits per token) | `rho(S)` at `eta = 1`, `kappa = 1/3` |
|---|---|---|
| I. `S < k`, fresh data | `h k/(k - S)` | `max{1, k/(3(k - S))}`: 1 up to `S = 2k/3`, 3.3 at 0.9k, 33 at 0.99k |
| I. `S < k`, data generated in the block (RL rollouts, procedural tasks) | state only, `S/Q` | 1, whatever `eta` is |
| II. `k <= S <= S_III` | `min over m > S/k of [m h + 2(m-1) F(S)] / (1 - S/(mk))` | `F` = 14 kbit: 1.1e3 at `k+`, 2.2e3 at 1.5k, 3.3e3 at 2k, 7.7e3 at 4k, 2.1e4 at 10k. `F` = 256 bits: 22, 42, 63, 146, 395 |
| III. `S >= S_III`, `k <= k*` | `beta P / d_eff` | `eta phi beta / (6 d_eff)` = 5.4e4 at `beta = 0.4` (6.5e4 at `kappa = 0.40`) |
| III. `k > k*` | the tile count `4 P d_in beta_w beta_x / k`; cheap-input layers use the accumulator bound | `(2/3) eta phi d_in beta_w beta_x / k`, and `rho >= c h_eff b_ps / (6 k^2)` |

Two shorthands for regime II:
- At `S = k+` (`m = 2`): `W = 4(h + F)`, so `rho = 4 eta kappa (1 + F/h)`. At `kappa = 1/3` that is the program's
  `(2/3) * 2 crossings * F/h`.
- For `S >> k` (best `m` about `2S/k`): `W ~ 8 (S/k) F`.

Regime II gives way to regime III at `S_III ~ beta P k / (8 d_eff F)`. That is about `24k` for a full-rank fresh
residual (`F` about 14 kbit) and `1300k` at rank 64 (`F` about 256 bits). At `S_III` the two `W`s agree: 5.2e4
against 5.4e4.

**(c) Attained.** In regimes I and II, an explicit covering (one block per batch and step, or the state pipelined
over `m` blocks) attains `W(S)` up to rounding `m`. In regime III the best known attack, the tile, costs
`(k*/k) * W_III`, so (a) is tight only at `k = k*`.

### 1.4 Proofs

**(a)**

1. By (L), assign each `v` to a block `I(v)` that decodes it, and let `A_I` be the values assigned to block `I`.
2. Given `theta`, a description of `(i, in(I_i) \ theta, decoder index)`, at most `k + lambda` bits, yields `A_I` in
   time `t`. Rebuild the block's gate list with `G` (U), evaluate it (at most `c` work), then run the decoder. So
   `K^t(A_I | theta) <= k + lambda`.
3. By (D), applied in the contrapositive, `w(A_I) <= k + lambda`.
4. Summing over blocks, `w(V) <= sum over I of w(A_I) <= |𝓘| (k + lambda)`. ∎

(E) is what makes the assigned values the recipe's own, so that `w(V)` is the recipe's weight. The time bound is per
block, and it is what resolves the red-team note's global paradox: the whole run is decodable from `theta` and `D`, but
only in time `|T|`, not `t`.

**(b) Regime I**

- *Upper bound.* Use one block per batch and step. It holds the state (`S` bits), reads `Q` tokens and writes the new
  state, which the next block reads. Then `Q = min{(k - S)/h, c/(6P)}`, and `c/(6P) = eta kappa k/h`. So
  `rho = c/(6 P Q) = max{1, eta kappa k/(k - S)}`.
- *Lower bound.* Every bit of `D` is read by some block, so `|𝓘| >= |D|/k`, and `rho >= eta kappa` for raw tokens. The
  bound and the construction meet up to the factor `k/(k - S)`, which is the state's amortization over a block's
  tokens.
- *Data generated in the block.* `|D|` is about the seed, the construction is compute-bound, and `rho` is about 1.

**(b) Regime II: the crossing lemma (sketch, under (D_F) and (L))**

1. Fix a token `j`. Each block that computes a counted value of `j` decodes some piece of the state.
2. `j`'s loss depends on all `S` bits, and no block holds more than `k` minus its other inputs. So `j`'s forward
   chain changes block at least `m - 1` times, where `m` is the number of distinct blocks it visits.
3. At each change, the receiving block reads at least `F` bits about `j`, by (D_F). The gradient into each piece
   depends on state outside it, so the backward chain changes block at least `m - 1` times too, each at `F`.
4. Each visited block also reads the token (`h`) or a full activation, which costs more.
5. Spread each block's state over its `Q_I` tokens and minimize over the split. The minimum is at equal pieces, and
   that gives `W_II`.
6. Pipelining the state over `m` blocks attains it.

The open step is ordering. A covering may hold non-contiguous layers or interleave blocks, so the proof counts changes
of block along the chain rather than intervals of layers. That is at least the number of distinct blocks minus one.
(C2) is the statement that forces this regime: inside a held piece, per-value weights cannot add up to more than the
piece's fresh input.

**(b) Regime III**

- (a) applies with `w = beta_v` on counted values, which (C1) allows exactly when `k <= k*`. Then
  `w(V)/N_tok = beta V_tok = beta P/d_eff`.
- The tile attack shows the gap: `R = k/(2 d_in beta_w)` rows and `Q = k/(2 d_in beta_x)` tokens give
  `rho_tile = (2/3) eta phi d_in beta_w beta_x / k`, which is `(k*/k) * rho_III`.

**Accumulator corollary (frozen prefix, layer 1, linearized training)**

When a layer's inputs are cheap (the token alone, `h` bits), count the weight-gradient entries `G = sum over j of
delta_j x_j^T`.

- *Reduction-tree lemma, wiring only.* If an entry's products are finished in `m_E` blocks, at least `m_E - 1` gates of
  its accumulation are inputs of those blocks.

  Proof: label each accumulation from its root down. A block that computes a gate has all its predecessors, so each
  predecessor either keeps the label or is an input that takes a fresh label. Each fresh label is a partial sum that
  some block is handed, and accumulations are disjoint. ∎ (`PartialSumsRead` proves this in Lean for forward products;
  the proof transfers to gradient accumulators unchanged.)
- *Premise (R).* A block that finishes a product of token `j` reads at least `h_eff` bits about `j`: `h`, plus the
  fresh backward signal `F_delta` when there is one.
- *Count.* Per step, products total at least `P N`. Block `I` finishes at most `|E_I| Q_I` of them, with
  `Q_I h_eff <= k`. So `sum over I of |E_I| >= P N h_eff / k`. Partial-sum reads total at least
  `(sum over I of |E_I| - P) b_ps`, and all reads total at most `|𝓘| k`. Hence
  `|𝓘| >= P N h_eff b_ps / k^2 - P b_ps / k`, and

$$\rho \;\ge\; \frac{c\,h_{\rm eff}\,b_{ps}}{6\,k^2}\cdot\frac{6P}{|T|_{\rm tok}}.$$

- *Attained.* Leaves take every entry for `k/h_eff` tokens each, and a reduction tree sums their partial sums. So the
  bound is tight, with `b_ps` the bits of a partial sum.

### 1.5 The locality lemma: (L) replaced by communication at the nonlinearity

**Statement.** Drop (L). For a value `v` that no single block decodes, let `g_v` be the first nonlinearity that
consumes `v`, or consumes a value computed linearly from it. Assume:

- **(L')** every protocol between the two sides of the split that computes `g_v`'s output exchanges at least
  `IC(g_v)` bits, where `IC` is information complexity under the actual distribution of the split;
- those bits are charged additively with the receiving blocks' (D) charges.

Then (a) holds with `w(v)` replaced by `min{w(v), IC(g_v)}` for every split value.

**Sketch.**

1. Wires between blocks are inputs of the receiving block.
2. The final outputs are local by (E). By induction back from them, each split value's consumer has a local output.
3. The split's transcript is a set of input bits of blocks on the consumer's side, at least `IC(g_v)` of them.
4. Additivity lets the transcript be added to the sum over blocks of `w(A_I)`. ∎

(`min` is the most one can charge: either side can always send its share's fresh content.)

**What the charge is for a transformer's nonlinearities.**

- *Linear maps* cost nothing: `W(a + b) = Wa + Wb`. That is why the charge sits at the nonlinearity and never at the
  split.
- *RMSNorm*: `||x_A + x_B||^2` needs `<x_A, x_B>`, an inner product, whose randomized communication complexity is
  `Omega(d)` even mod 2 (Chor-Goldreich). That is about `d` bits per token per norm, where `d` is the normed width:
  7,168 for V4-Pro, and up to 4x that if a norm runs over the hyper-connected `4d` residual. After that one scalar, the
  normalized vector stays split for free.
- *SwiGLU* (`silu(g) * u`) is a per-coordinate nonlinearity, so it costs about `g`'s precision per coordinate.
- *ReLU or sign gating* on additive shares needs only comparisons, `O(log b)` bits per coordinate, which is still
  above `beta_v` of about 0.4.
- *Softmax of `q.k`* has bilinear cross terms for every query-key pair.

For the cliff, a crossing costs `min{F, about d}` at the next norm. For V4-Pro, `F` of about 14 kbit is cut at most to
about 7 kbit, so the cliff stays within 2x.

## 2. Relation to the Lean family

| Assumption | Lean premise or construct | Notes |
|---|---|---|
| (E) | `Nci.Covering.Covers` (blocks compute every non-input gate of the reference circuit). `Nci.Cover.Guarantees.CoverTraining`'s `Containment c.D d.D` and `BaselineRegistered` (the declared circuit contains the reference circuit, with its baseline registered) | Containment is strong (E), and it also gives (L) for free |
| (U) | None needed | Lean's premises count input gates of every subcircuit of a fixed circuit: "nothing here reads what a gate computes, only the wiring" (`Definitions/Nci/Cover.lean`). So structure is quantified over rather than charged as advice. The `K^t` form needs (U) to make the same per-block statement. Lean does (U)'s job by stating the premises on the honest reference circuit `c` and pulling blocks back through `Containment`. MoE routing is muxes in a fixed circuit, so routing-shaped blocks are subcircuits the premises must hold for; the realism note measured 53.3 bits per token per layer of routing input |
| (L) | None needed | Each product has a single output gate `o_{i,j}` that some block computes. A block that computes it without completing the product reads one of its gates (`completes_or_meets` in `Proofs/Nci/Covering.lean`). That is the "+k" term: splitting costs one input gate per split product. **The product (gate) form is robust to bit-slicing**, which is a reason to state the deck's theorem in it |
| (C) | The two-cost and additive forms themselves | The tile attack's count is `CoveringBoundAdditive`'s `k^2/(4 c_w c_a)` products at `c_w = beta_w`, `c_a = d_in beta_x`. So (C1) says exactly that the per-value cap `k/beta_v` is at least the additive count |
| (D), weights' side | `ParameterHard`, `ParameterHardAt` (`c_w`), `ParameterHardPricedAt` (prices `p i`, branch `cursor/nci-covering-priced-4207`), `ConditionalParameterHardAt` (additive), `UpdateHardness` | Gate-count forms of (D): a block of at most `s` gates cannot rebuild updated weights from fewer input gates than their price |
| (D), activations' side | `ActivationChainedAt` (cut width `c_a`), `ValueReadHardness` (`f` per value), `ActivationReads`, `ColumnsIncompressible` | `c_a` is one activation's width, at most 16. (D_F) is the same premise at the width of the whole fresh residual `F` |
| Regime I | The depth hypothesis `c_w D > k` (priced form: any `D` weights cost more than `k`) | When the whole update costs at most `k`, no such `D` exists, and Lean's bound is silent exactly in regime I |
| Regime II | `CoveringBoundPriced` (branch) and `CoveringBoundTwoCosts` (main) | Prices sum to `S`, and a block holds at most `k` of them and reads at least `c_a` per token. In console units that is `rho >= eta kappa (S/k)(c_a/h)`, and `CoveringBoundAdditive` gives 4x that. This is regime II's shape with `8F` replaced by `c_a` or `4 c_a`. **The cliff's height is the step from `c_a` (16 bits) to `F` (about 14 kbit)**, and the fan-out tier (`(d - E) f`, about 1.5 kbit for Kimi K2) is in between |
| Regime III | `CoveringBoundTwoCosts`, `CoveringBoundAdditive`, `CoveringBoundPriced` | In console units, `rho >= gamma eta phi c_w c_a / (6k)`, with `gamma = 1` for two costs and priced, and 4 for additive. Against the production bound the ratio is `gamma d c_w c_a / (beta k)` (`d = 7168`, `beta = 0.4`, `c_a = 16` at bf16, 8 at FP8). At 10 kB and 1 MB: two costs at a uniform `c_w = 2.3` (the cheapest 1% at bf16), 8.2 and 0.08; additive at a uniform `c_w = 2.3`, 33 and 0.33; priced at the bf16 mean 6.7, 24 and 0.24; priced at the FP8 mean 2.8, 5.0 and 0.05. Additive and priced together is not a stated theorem. Under (C) with `c_a = d_in beta_x`, the additive form would always beat the production bound. That `c_a` is not proved; the fan-out and partial-sum tiers are the proved steps toward it |
| Accumulators | `Nci.Cover.Guarantees.PartialSumsRead` | The reduction-tree lemma, proved with no premise, for forward products |

**What the deck can cite as proved.** All of these are proved on `main`, with each proof named in
`Security/Proofs/Nci/`:

- `Nci.Guarantees.CoveringBound`;
- `Nci.Guarantees.CoveringBoundTwoCosts`;
- `Nci.Cover.Guarantees.CoveringBoundAdditive`;
- `FanOutCoverBound`, `FanOutReadCoverBound` and `FanOutReadCoverBoundStructural`;
- `PartialSumsRead`, `PartialSumCoverBound` and `PartialSumKeyedCoverBound`;
- `CoverTraining` and `CoverTrainingAligned`.

`CoveringBoundPriced` is proved on its branch, not on `main`. None of these is the production bound, the crossing
lemma or the locality lemma. A citation should carry the units caveat from `CoverAdditive`'s docstring: Lean's `s` and
`alpha_0` are counted in Boolean gates, and converting to the FLOP-based `c` and `|T|` scales them by serving's over
training's gates per multiply-add.

## 3. Counterexamples: the conclusion fails without each assumption

### 3.1 The information measure: PRG stretch

- **HILL or metric entropy.** Train on data `G(sigma)`, a PRG of a 128-bit seed, with state `S <= k`.
  - Every token and every activation has HILL entropy near its width, so the HILL weight per token is about
    `beta P/d_eff`, roughly 2.7e6 bits. Theorem (a) would then claim regime III, about 6.5e4 at matched `c`.
  - One block holds `sigma` and the state and generates its own data, so `rho` is about 1. The bound fails by about
    6.5e4.
  - The compression step (`K^t(A_I | theta) <= k`) has no HILL analogue: a `k`-bit seed stretches.
- **Shannon entropy under a prior on the data.** The same family, read as self-distillation on `theta`'s own seeded
  samples. The attacker chooses the distribution, so `H(D | theta)` is the seed's 128 bits, and decoding (D) fails.
- **Unbounded `K`.** `K(V | theta) <= |D| + O(1)`, so only the data bound `|𝓘| >= |D|/k` survives, which is regime
  I's lower bound for every `S`. This is the global paradox. Only `K^t` with `t` about `c`, per block, separates the
  regimes.

### 3.2 Uniformity: MoE routing

- **Family.** Fine-tune only the router of an MoE whose experts stay registered, and choose blocks by routing, one per
  token path. Each block's gate list wires the token's routed experts at every layer.
- **What breaks.** Without (U) that wiring is free advice, about `c log|T|` bits a block. The block reads only the
  token (`h` bits) and computes its whole forward pass.
- **How badly.** The routing that shapes the circuit carries up to 53.3 bits per token per layer (the realism note),
  about 3.3e3 bits per token over 61 layers. That is up to about 190 `h`, all of it uncharged, so the bound fails by up
  to about 190x on routing-dependent values.
- **Generic form.** Wiring encodes constants. `theta_a XOR theta_a` and `theta_a XNOR theta_a` give 0 and 1 with no
  source gate, so a block outputs about `c` bits of a precomputed update from no unregistered input.
- **The fix.** With (U), the routing must be computed in the block from a counted router state, or read as input at 53
  bits per token per layer.

### 3.3 Locality: bit-slicing

- **The proof gap.** Compute a value `v` as low bits (with the carry) in block A and high bits in block B, or as
  additive shares `v = a + b`. No block decodes `v`, so step 1 of proof (a) assigns it nowhere.
- **Where the conclusion fails.** Through linear maps, slices and shares propagate for free: `W(a + b) = Wa + Wb`, and
  a radix decomposition `x = sum over i of 2^i x_i` gives `W x = sum over i of 2^i W x_i`. So every counted value
  consumed only linearly can stay split, or be re-associated, and never be decoded.
  - Extreme instance: a stretch of `L'` linear maps (a deep linear network, the tangent of linearized training, chained
    adapters) computed as one product `M = W_{L'} ... W_1` applied to each token. The `L' d` counted values per token
    never exist, and the per-value count overstates the bound by `L'`, without limit.
- **The fix.** Charge at the consuming nonlinearity (section 1.5): for a transformer, `Theta(d)` per token per
  RMSNorm, and about full precision at SwiGLU.
- **Lean's product form never needs (L)** (section 2). This counterexample is a reason to state the deck's theorem
  over products (gates), not over values.

### 3.4 The cap: linearized training (and layer 1, frozen prefixes, head-only)

- **Family.** Training linearized at `theta` uses only `theta`'s activations, so every token costs `h` bits.
- **What breaks.** The tile is `R` update rows and `Q` tokens, with `R d beta_w + Q h <= k`. It computes
  `k^2/(4 d beta_w h)` values against the cap's `k/beta_v`, so (C1) fails for `k > 4 d h beta_w/beta_v`, which is
  61 kB.
- **How badly.** At 1 MB the tile attack costs `rho = (2/3) c beta_w h / k^2`, about 9.9e3, against the claimed
  1.7e5: 17x below.
- **Head-only training.** The same holds: about 30 at 1 MB, where the table says 5e3.
- **(C2)'s family.** A block holding a 4-layer slice of a rank-8 LoRA (about 200 kB per layer) whose counted outputs
  outweigh its fresh input.
- **The fix.** Count on the accumulators (section 1.4), or lower `w` until (C) holds.

### 3.5 Decoding: seeded RL

- **Family.** GRPO on a procedurally generated task: a generator, reward code and a 64-bit seed, a few kbit in all.
  - `theta'_t = Train(theta, seed, t)` and `K(V | theta)` is a few kbit. The bound stands only if no block can
    shortcut the trajectory within `c`.
  - The compressors B1 and B2 run don't know the seed, so they measure `beta` as large.
- **Where the conclusion fails.** Whenever the dynamics have a fast-forward:
  - full-batch gradient descent on a quadratic objective, in exact (integer or rational) arithmetic:
    `theta_t = M^t theta_0 + (sum over s < t of M^s) b` by repeated squaring, in `O(d^3 log t)` against `t d^2`;
  - a run whose update direction does not rotate (`theta + alpha(t) v`, with `v` the first seeded gradient);
  - self-distillation to a computable fixed point.

  Then one block holds `(seed, t)`, the run is regime I with `rho` about 1, and the bound from measured `beta` claims
  regime III.
- **The caveat.** Bit-exact floating point (E) defeats the algebraic fast-forwards, since rounding differs. That is
  evidence for (D), not a proof: a proof would be a time-bounded Kolmogorov lower bound for an explicit string, which
  is out of reach.

### 3.6 Exactness (the capability form)

The prover reaches the capability with another recipe whose state is at most `k`: TinyLoRA, a rank-1 LoRA for
policy-gradient RL, or a distilled adapter. The exact-recipe bound for full fine-tuning (regime III) does not bind it.
This is the program's crux 1. It is also why `beta` must be minimized over precisions: the attacker picks the lowest
one that still trains.

## 4. Check of every formula in the two console notes

Arithmetic: `tvi/e1_arith.py`. Units: 1 kB = 8e3 bits, `c` = 1.4e17, `d` = 7000 (7168 where marked), `h` = 17,
`P` = 49B active.

### The headline and `eta`

| Formula (note) | Verdict | Check and fix |
|---|---|---|
| `rho ~ c beta/(6 d k)`: 1.7e7 at 10 kB, 1.7e5 at 1 MB (red-team, introduction) | **Correct** within its domain | 1.67e7 and 1.67e5 (1.63e7 and 1.63e5 at `d = 7168`). Valid only for `k <= k*` and `S >= S_III`, and `d` must be `d_eff = P/V_tok`, which is about `d_max` under the section 4 counting rule |
| `rho = eta phi beta/(6d)`, `eta = c/(k phi)` (both notes) | **Correct**, an identity | `eta` = 250 at 10 kB and 2.5 at 1 MB for `phi` = 7e9 |
| "`c` is about 1.3M tokens of compute; forward passes over 4.7k tokens use 0.7% of it" (section 5) | **Minor** | 1.18M tokens at `phi h` per token, 1.43M at `2P`. 10 kB is 4,706 tokens. Their forward uses 0.40% of `c`, and forward plus backward 1.2%, so neither gives 0.7% |
| `rho_free = phi beta/(6d)` "about 6e4-1.1e5 at `beta = 0.4`" (section 6) | **Top end not reproduced** | The listed non-reasoning `phi` (6.6e9-8.9e9) give 6.3e4-8.5e4. 1.1e5 needs `phi` = 1.16e10 |
| "Reasoning-heavy traffic, 10-100 generated tokens per input token, reaches 1e6-1e7" (section 6) | **High by about 1.5x** | `phi` = 6.9e10 at 10x gives 6.6e5, so 100x gives about 6.6e6 |
| Dense `rho` "5e3-3e4" at matched `c`, `beta` = 0.03-0.1; "1e6-1e7" at 1.4e17 and 10 kB (section 6) | **Slightly high** | 4.7e3-2.1e4 with the listed `phi`; 1.25e6-4.2e6 at the headline `c` |
| "With `c` matched, `rho ~ eta phi beta/(6 d_eff)`" (program) | **Correct** | It is regime III of section 1.3, with `d_eff := P/V_tok`. Holds for `k <= k*` and `S >= S_III` |
| "About 1e5 for full training" (program) | **High by about 1.5-2x** | 5.4e4 at `kappa = 1/3` and 6.5e4 at `phi` = 7e9 (`beta = 0.4`, `d = 7168`). 1e5 needs `beta` of about 0.6 |
| "5e3 and 1e6 both clear 1000x" (program) | **Correct** | 5e3 is `beta = 0.03`; 1e6 is reasoning `phi` at `beta` of about 0.6 |

### The cap condition and `k*`

| Formula (note) | Verdict | Check and fix |
|---|---|---|
| Cheap inputs: "`k^2/(4 d beta_w h)` values, exceeds `k/beta` once `k > 4dh` (about 60 kB), 17x at 1 MB" (section 2.1) | **Correct** | `4dh` = 59.5 kB (60.9 kB at `d = 7168`); 1 MB is 16.8x that. In general the threshold is `4 d h beta_w/beta_v` |
| "Claim holds iff `min(d_out, k/(d_in beta_w)) <= d_in beta_x/beta_v`" (section 3) | **Wrong, too strict** | It is the condition that every tile carries its full nominal weight (exact per-tile decoding), which the theorem does not need. The threshold form of (D) needs only (C1): `max over tiles of R Q beta_v <= k` with the tile's description at most `k`. Exact solution: section 1.2 |
| "All betas equal gives `k <= d^2 beta`, about 2.4 MB" (section 3) | **Wrong** | The note's own formula gives "never fails" for square maps. For an up-projection (`d_out >= 2 d_in`) the exact threshold is `4 d_in^2 beta`, about 10.3 MB, not `d_in^2 beta`, about 2.5 MB |
| `k* = d_in^2 beta_w beta_x/beta_v`: "240 kB at `beta_x = beta_v/10`" (section 3) | **Off by 4x, and must be minimized over maps** | `k* = 4 d_in^2 beta_w beta_x / beta_v`: 980 kB at `d_in = 7000` (1.03 MB at 7168). For V4-Pro's expert down-projection (`d_in = 3072`) it is 189 kB, and for compressed-attention latents (`d_in` about 512-1536) it is about 5-50 kB, so those set `k*` |
| "Dense full-parameter training with `k <~ d^2 beta`: `rho ~ c beta_v/(6 d_max k)`" (section 4) | **Condition wrong** | Replace `k <~ d^2 beta` with `k <= min over maps of 4 d_in^2 beta_w beta_x/beta_v`, and add `S >= S_III` |
| Theorem "`\|T\| >= c w(V)/(k + O(log\|T\|))`" (section 4) | **Typo** | It should be `\|𝓘\| >= w(V)/(k + O(log\|T\|))`, hence `rho >= c w(V)/(\|T\|(k + O(log\|T\|)))` |
| "The cap per block `k/beta_v` is tight" (section 4) | **True of values, not of products** | Reading compressed values attains `k/beta_v`, but those values were computed elsewhere. Lean counts finished products, and the attack that matters under (C) is the tile, which falls below `k/beta_v` by `k/k*` |

### The cliff

| Formula (note) | Verdict | Check and fix |
|---|---|---|
| `rho ~ (2/3) * crossings * F/h`: about 1e3 at `F` = 14 kbit with two crossings, about 20 at rank 64 (program) | **Correct at `S = k+` with `kappa = 1/3`, otherwise not** | Exact form: `eta kappa [m + 2(m-1)F/h]/(1 - S/(mk))`, minimized over `m`. At `m = 2`, `S -> k`: `4 eta kappa (1 + F/h)`, which is 1.10e3 and 21 at `kappa = 1/3`. The 2/3 is `kappa * 2`: the 2 is blocks spending half their `k` on state, and `kappa = (1+g)/3` is a traffic parameter (0.40 at the note's `phi`, giving 1.3e3 and 26; 4.0 for reasoning traffic). The value grows above `k`: 3.3e3 at `2k`, 2.1e4 at `10k`, about `8 eta kappa (S/k) F/h`. Also, `F` should be `min{F, about d}` per norm (section 1.5) and measured over the 4d hyper-connected residual |

### The rho table (red-team note, section 5)

| Row | Verdict | Check and fix |
|---|---|---|
| TinyLoRA, evolution strategies on procedural tasks, seeded subspace, FedKSeed: "about 1" | **Correct for data generated in the block** | On fresh SFT data `rho = max{1, eta kappa k/(k - S)}`: about 83 at the 10 kB headline `c`, about 1 at matched `c` |
| MeZO with a growing list: 150-200 at 10 kB, 1.5-2 at 1 MB | **Not checkable** (no derivation given) | The list grows `b_seed + b_scalar` bits per step (about 48), so `S <= k` up to about 1.7e3 steps at 10 kB and 1.7e5 at 1 MB. At 1 MB, OPT-13B's 20k steps are regime I, so 1.5-2 against normal MeZO is plausible. At 10 kB the crossing is per-weight partial sums of `sum over s of scalar_s z_s`, outside the cliff formula's form; it needs a derivation |
| LoRA rank 16: 3.8e4 and 380. Rank 256: 6.1e5 and 6.1e3 ("`rho` scales by `r/d`") | **Arithmetic correct, scaling and regime wrong** | `1.67e7 * 16/7168` = 3.7e4, and so on. Per value, an adapted matrix contributes `2r` counted values against `d`, and a frozen backbone's `\|T\|` is about `4P` rather than `6P`, so the factor is about `3r/d`: 1.1e5 and 1.1e3 at `r = 16`. At 1 MB a block holds about 4 layers of a rank-8 adapter (200 kB per layer, 12 MB in all), so these rows are regime II and need measured `F` |
| Head, a few layers or a few experts only: about 5e5 and 5e3 | **Wrong** | Head-only: its inputs are `theta`'s activations (cheap, `h` per token) and `\|T\|` is about `2P + 6dV` per token. Per value, 4.8e4 at 10 kB. At 1 MB the cap fails (above 61 kB) and the tile attack costs about 30. The "few layers" rows: the first trained layer is the cheap-input case, so they need the accumulator form |
| FP8 training: about 1e7 and 1e5 | **High by about 2.4x** | At `beta = 0.1`, the note's own FP8 figure (section 2.5), it is 4.2e6 and 4.2e4. The `beta` here is per step against a `theta` registered at the run's start, so the deck needs the run average |
| Dense fine-tuning, distillation, full-rank linearized: "full bound" | **Linearized is wrong** | Linearized training is the cap counterexample (section 3.4): `rho` is about 9.9e3 at 1 MB, not the full bound of 1.7e5. Use the accumulator row |
| "Join bound tighter below `k* = b^2 d/(beta r^2)`: 360 bytes at `beta = 0.4`, `r = 10`, and 36 kB at `beta = 0.004`" (section 6) | **Correct at `b` = 4 (FP4)** | 358 bytes and 35.8 kB at `b = 4`, `d = 7168`, with `r` a compression ratio (Lean's `r` is a rate, `c = r b`). At `b = 16` it is 5.7 kB |
| Sizes: rank-8 LoRA 200 kB per layer, 12 MB whole; one dense matrix `d^2 beta` about 2.4 MB | **Correct** | 200.7 kB, 12.2 MB (2 bits per parameter), 2.57 MB |

### The accumulator corollary

| Formula (note) | Verdict | Check and fix |
|---|---|---|
| "Counted on gradient accumulators: about 5e7 at 10 kB and 5e3 at 1 MB; the reduction-tree lemma is not proved" (section 4) | **Correct to about 25% as a lower bound, and the lemma is proved** | The lower bound is `rho >= c h_eff b_ps/(6 k^2) * 6P/\|T\|_tok` (section 1.4). That is 6.2e7 and 6.2e3 at `h_eff = 17`, `b_ps = 1` gate and `\|T\| = 6P`, and 4.6e7 and 4.6e3 if linearized training costs `8P` per token. The reduction-tree lemma is `PartialSumsRead` (proved on `main`, wiring only, forward products); its instance on weight gradients has the same proof. The one premise beyond wiring is (R). With `b_ps` set to a partial sum's fresh bits rather than one gate, the bound scales by `b_ps`. The leaf-and-tree covering attains it |

## 5. Deck formulas Daniel must re-check

The deck is `apps/web/components/talks/covering-bound/` in `website-gpu-lenses`, which this VM cannot reach. Each
item names what to check against the corrections above.

1. **The theorem's conclusion.** It should be `|𝓘| >= w(V)/(k + O(log|T|))`, not `|T| >= ...`.
2. **The cap condition and `k*`** in Definition (inference-like) or Theorem (training slowdown).
   - `k* = 4 d_in^2 beta_w beta_x/beta_v`, minimized over maps.
   - With all betas equal, the threshold is `4 d^2 beta` (up-projections) or none (square maps), not `d^2 beta`.
   - The "whole layer" condition should become `k <= min over maps of k*`.
3. **The cliff formula.** If the deck has `(2/3) * crossings * F/h`, write it as
   `4 eta kappa (1 + F/h)` at `S = k+`, with `kappa = phi h/(6P)` named and its traffic value stated. Also show that it
   grows as about `8 eta kappa (S/k) F/h` above `k`.
4. **`rho(S)` across the regimes**, if the deck plots or tabulates it: the table in section 1.3(b), including the
   ramp `k/(k - S)` just below `k`.
5. **The attack table.** Fix head-only (4.8e4 and about 30), FP8 (4.2e6 and 4.2e4 at `beta = 0.1`), LoRA (about
   `3r/d` per value, and the cliff regime at 1 MB), linearized (the accumulator row, not the full bound), and TinyLoRA
   on fresh data (`eta kappa`, not 1).
6. **`rho_free`** at matched `c`: 6.3e4-8.5e4 at `beta = 0.4`, not up to 1.1e5. Reasoning traffic: 6.6e5-6.6e6.
7. **The accumulator corollary.** It can now cite `PartialSumsRead` for the reduction tree, with premise (R) named.
   Numbers: 6.2e7 and 6.2e3.
8. **The assumption list.** It should match section 1.2, with (D) named as the computational-depth assumption, or
   folded into `w`, saying which.
9. **Any slide comparing with the Lean covering bound.** In regime III the proved forms are stronger than the
   production bound at 10 kB and weaker at 1 MB (bf16): additive at a uniform `c_w = 2.3`, 33x and 0.33x; priced
   (branch) at the mean 6.7, 24x and 0.24x. They cross over between about 80 kB (two costs) and 330 kB (additive).
   The regime II shape is proved with `c_a`
   in place of `F`.

## 6. What phase 2 should bring, to revise the theorem

- **`F(S)`** at the cut, measured fresh relative to `theta`'s own activation on the same token (it is about 0 at step
  1), over the hyper-connected residual. Also `F_delta` for the backward crossing.
- **`beta(S)`** on counted values (linear-map outputs), at the attacker's precision, averaged over steps since
  registration.
- **`kappa` from D1's traffic** (`g`, the generated:prompt ratio).
- **The attack curves `rho_attack(S, k, c)`**, to place against `W(S)` in section 1.3. A point below the regime II
  formula with `F` measured means (D_F) or (L') fails; a point below regime III at `k <= k*` means (D) or (C) fails.
- **Capability against `S`.** This decides which regime each training goal sits in; it is the deck's scope question,
  not the theorem's.
