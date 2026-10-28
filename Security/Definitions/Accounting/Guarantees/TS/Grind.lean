import Definitions.Pouw.RelativeSecurity

/-!
# The grinding premium under hashed seeds (trusted)

GRIND: a unit's seed is `SHA-512(tag ‖ salt ‖ position ‖ root(A) ‖ root(B))`: the salt is a live coin, the roots are
hiding commitments, and a cheater re-rolls a root for free through its blinds, so `T` hash queries give it `T` fresh
seeds (the random-oracle model, `random-oracle`). These are the lemmas that price that, each `def G : Prop` proved as
`Accounting.SecurityProofs.TS.G` in `Proofs.Accounting.TS.Grind`. They are stated over abstract per-query bounds, so a
service statement cites them without these definitions reaching into its own:

* `ExchangeTail`: a gain whose exponential moment at rate `θ` is at most `M` exceeds `π` with probability at most
  `M·e^{-θπ}` (Chernoff, one query).
* `GrindTail`: over `T` queries, where query `q`'s bad event reads every oracle answer, its own one included, and each
  query's event has probability at most `τ` over its own answer with every other answer fixed, some query is bad with
  probability at most `T·τ`. Its hypothesis is per query, given all other answers: that is the adaptivity across
  stages (a later stage's roots may read every earlier seed).
* `GrindPremium`: the two together, with the commitment's binding a parameter. `R q` is what the cheater realizes
  through query `q`, and it equals the oracle's gain `g q` off an event `Break` of probability at most `δb` (opening a
  root two ways after its seed, at the same `T`). Some query realizes a gain of at least `π` with probability at most
  `T·M·e^{-θπ} + δb`. It is gap (7)'s steer bound at `Steer := {π ≤ gain}`, `μ^m := M·e^{-θπ}`, relative to the stage's
  savings.
* `GrindPremiumAt`: `GrindPremium` at the premium `π* = ln(T·M/δ)/θ`, so the grind is priced into ε by `π*` a seed group
  and the bound holds outside `δ + δb`. With `M = 1` and `θ = h·ln 2` a unit of work, `π* = log₂(T/δ)/h` units.
* `GrindMass`: the expectation form. The `T` queries' moments sum to at most `T·M` in mean, so by Markov and Jensen
  `n` committed seed groups realize at most `ln(T·M/(n·δ))/θ` each on average outside `δ`, and `ln(T·M)/θ` in mean.

The per-query moment `M` for a deployed template, with the gain of a scope's skips at exchange rate `κ`: a unit `u`
skipping `w_u` of work gains `w_u` when lucky (its receipt passes) and `-κ·w_u` when not, so `L - κ·U` with `L` and `U`
the lucky and unlucky savings. Outside the premium, savings are at most `(1 + κ)·U + π*`, and `U` is caught at the
zero-luck sampled rate; at `κ = p/(1 - p)` that is exactly the live analysis' charge at luck `p`.

* `LuckMoment`: one unit with luck at most `p`: moment at most `p·e^{θw} + (1 - p)·e^{-κθw}`.
* `PrivateMoment`: units whose gains read shared coins `R` and each its own private coins multiply their moments.
* `TileLuck`: the deployed tile (h1tile's `r × c`, one seed a tile). Its `n` columns share the rows' coins `Ω₀` and each
  has its own coins `E`, read by its `r` units. A unit is dead (its skipped value doesn't read its column's coins: the
  row dither cancels the operand on its skip set) with probability at most `q` over `Ω₀`, and otherwise lucky with
  probability at most `p` over its column's coins (`HMu.CoinPointMassHalf`: `1/2`). If every unit meets
  `q·e^{n·rθw} + (1 - q)·(p·e^{rθw} + (1 - p)·e^{-κ·rθw})^n ≤ M`, the tile's moment is at most `M`, however the dead
  events and the lucky events correlate (AM-GM over the columns and over each column's units).
* `TemplateCondition` and `TemplateMoment`: `TileLuck` at h1tile's segment (32 columns, words of at most 32 skipped
  products, a one-bit dither so `q = 2^{-w}`, `p = 1/2`) with `M = 1` at `κ = 4` and `r·θ = 1/256`: `θ = 1/(256·r)` a
  product, and a seed group's premium at most `256·r·ln(T/δ)` products. The condition's largest `r·θ` at `κ = 4` is
  about `0.0099` (`benchmarks/pouw/grind_price.py`), 2.5 times this proved point.

What these don't give, and stays a premise: the skip decision is fixed by the query (E3, `HMu`'s no-leak premise:
a cheater that sees its luck after the seed and picks which units to skip pays `C(n, w)` and needs more than
`log₂(e·n/w)` bits a unit, which no one-bit coin gives, under live coins as much as here), the joint view of words that
share a coin (`HMu`), and the binding term `δb` itself, which is per prover (SHA-512 frame-v3 roots: about `T²/2^513`
for a prover that holds no SHA-512 collision, and `1` for one that does).
-/

namespace Accounting.Guarantees.TS

open Pouw.RelativeSecurity

/-- **The exchange tail (one query).** A gain `f` whose moment at rate `θ ≥ 0` is at most `M` reaches `π` with
probability at most `M·e^{-θπ}`. -/
def ExchangeTail : Prop :=
  ∀ {B : Type} [Fintype B] (f : B → ℝ) (θ M π : ℝ), 0 ≤ θ → 0 ≤ M →
    ∑ b, Real.exp (θ * f b) ≤ M * Fintype.card B → prob (fun b => π ≤ f b) ≤ M * Real.exp (-(θ * π))

/-- **The grinding union.** On coins `Ω₀` and `T` oracle answers in `B`, uniform, if each query's bad event has
probability at most `τ` over its own answer, every other answer fixed, some query is bad with probability at most
`T·τ`. -/
def GrindTail : Prop :=
  ∀ {Ω₀ B : Type} [Fintype Ω₀] [Fintype B] {T : ℕ} (Bad : Fin T → Ω₀ → (Fin T → B) → Prop) (τ : ℝ), 0 ≤ τ →
    (∀ q ω₀ a, prob (fun b : B => Bad q ω₀ (Function.update a q b)) ≤ τ) →
    prob (fun ω : Ω₀ × (Fin T → B) => ∃ q, Bad q ω.1 ω.2) ≤ T * τ

/-- **The grinding premium, binding a parameter.** If each query's gain `g q` has moment at most `M` at rate `θ` over
its own answer, every other answer fixed, and what the cheater realizes through query `q` is `g q` off `Break`
(probability at most `δb`), some query realizes at least `π` with probability at most `T·M·e^{-θπ} + δb`. -/
def GrindPremium : Prop :=
  ∀ {Ω₀ B : Type} [Fintype Ω₀] [Fintype B] {T : ℕ} (g R : Fin T → Ω₀ → (Fin T → B) → ℝ)
    (Break : Ω₀ × (Fin T → B) → Prop) (θ M π δb : ℝ), 0 ≤ θ → 0 ≤ M →
    (∀ q ω₀ a, ∑ b : B, Real.exp (θ * g q ω₀ (Function.update a q b)) ≤ M * Fintype.card B) →
    prob Break ≤ δb → (∀ ω q, ¬Break ω → R q ω.1 ω.2 = g q ω.1 ω.2) →
    prob (fun ω : Ω₀ × (Fin T → B) => ∃ q, π ≤ R q ω.1 ω.2) ≤ T * M * Real.exp (-(θ * π)) + δb

/-- **The premium priced into ε.** `GrindPremium` at `π* = ln(T·M/δ)/θ`: some query realizes `π*` with probability at
most `δ + δb`. -/
def GrindPremiumAt : Prop :=
  ∀ {Ω₀ B : Type} [Fintype Ω₀] [Fintype B] {T : ℕ} (g R : Fin T → Ω₀ → (Fin T → B) → ℝ)
    (Break : Ω₀ × (Fin T → B) → Prop) (θ M δ δb : ℝ), 0 < θ → 0 < M → 0 < δ →
    (∀ q ω₀ a, ∑ b : B, Real.exp (θ * g q ω₀ (Function.update a q b)) ≤ M * Fintype.card B) →
    prob Break ≤ δb → (∀ ω q, ¬Break ω → R q ω.1 ω.2 = g q ω.1 ω.2) →
    prob (fun ω : Ω₀ × (Fin T → B) => ∃ q, Real.log (T * M / δ) / θ ≤ R q ω.1 ω.2) ≤ δ + δb

/-- **The grinding mass (expectation form).** Under `GrindPremium`'s per-query moment, the `T` queries' moments sum
to at most `T·M` in mean. -/
def GrindMass : Prop :=
  ∀ {Ω₀ B : Type} [Fintype Ω₀] [Fintype B] {T : ℕ} (g : Fin T → Ω₀ → (Fin T → B) → ℝ) (θ M : ℝ),
    (∀ q ω₀ a, ∑ b : B, Real.exp (θ * g q ω₀ (Function.update a q b)) ≤ M * Fintype.card B) →
    ∑ ω : Ω₀ × (Fin T → B), ∑ q, Real.exp (θ * g q ω.1 ω.2) ≤ T * M * Fintype.card (Ω₀ × (Fin T → B))

open Classical in
/-- **One unit's moment.** A unit skipping `w ≥ 0`, lucky with probability at most `p`, gains `w` when lucky and
`-κ·w` when not; its moment at rate `θ ≥ 0` is at most `p·e^{θw} + (1 - p)·e^{-κθw}`. -/
def LuckMoment : Prop :=
  ∀ {Ω : Type} [Fintype Ω] (lucky : Ω → Prop) (p w θ κ : ℝ), 0 ≤ w → 0 ≤ θ → 0 ≤ κ → prob lucky ≤ p →
    ∑ ω, Real.exp (θ * (w * ((1 + κ) * (if lucky ω then 1 else 0) - κ))) ≤
      (p * Real.exp (θ * w) + (1 - p) * Real.exp (-(κ * θ * w))) * Fintype.card Ω

/-- **Private coins multiply moments.** Gains `g i` that read shared coins `R` and each its own coins `E i`: if each
has moment at most `M i` over its own coins, whatever the shared ones, their sum has moment at most `∏ M i`. -/
def PrivateMoment : Prop :=
  ∀ {R ι : Type} {E : ι → Type} [Fintype R] [Fintype ι] [DecidableEq ι] [∀ i, Fintype (E i)]
    (g : ∀ i, R → E i → ℝ) (θ : ℝ) (M : ι → ℝ),
    (∀ i r, ∑ x, Real.exp (θ * g i r x) ≤ M i * Fintype.card (E i)) →
    ∑ ω : R × (∀ i, E i), Real.exp (θ * ∑ i, g i ω.1 (ω.2 i)) ≤ (∏ i, M i) * Fintype.card (R × (∀ i, E i))

open Classical in
/-- **The tile's moment (the template's entropy).** `n` columns share coins `Ω₀`, and column `k` has its own coins in
`E`, read by its `r` units. Unit `(k, i)` skips `w k i ≥ 0`; it is dead with probability at most `q k i` over `Ω₀`, and
otherwise lucky with probability at most `p k i ∈ [0, 1]` over its column's coins. If every unit meets
`q·e^{n·rθw} + (1 - q)·(p·e^{rθw} + (1 - p)·e^{-κ·rθw})^n ≤ M`, the tile's gain at exchange rate `κ` has moment at
most `M` at rate `θ`. -/
def TileLuck : Prop :=
  ∀ {Ω₀ E : Type} [Fintype Ω₀] [Fintype E] {n r : ℕ} (w : Fin n → Fin r → ℝ) (dead : Fin n → Fin r → Ω₀ → Prop)
    (lucky : Fin n → Fin r → Ω₀ → E → Prop) (p q : Fin n → Fin r → ℝ) (θ κ M : ℝ),
    0 < n → 0 < r → 0 ≤ θ → 0 ≤ κ → (∀ k i, 0 ≤ w k i) → (∀ k i, 0 ≤ p k i ∧ p k i ≤ 1) →
    (∀ k i ω₀, ¬dead k i ω₀ → prob (lucky k i ω₀) ≤ p k i) → (∀ k i, prob (dead k i) ≤ q k i) →
    (∀ k i, q k i * Real.exp (n * (r * θ * w k i)) +
        (1 - q k i) * (p k i * Real.exp (r * θ * w k i) + (1 - p k i) * Real.exp (-(κ * (r * θ) * w k i))) ^ n ≤ M) →
    ∑ ω : Ω₀ × (Fin n → E),
        Real.exp (θ * ∑ k, ∑ i, w k i * ((1 + κ) * (if lucky k i ω.1 (ω.2 k) then 1 else 0) - κ)) ≤
      M * Fintype.card (Ω₀ × (Fin n → E))

/-- **The template's condition at one point.** h1tile's segment at `κ = 4`: a word of `w ≤ 32` skipped products over
32 columns, dead with probability at most `2^{-w}` (a one-bit dither a row entry) and otherwise lucky with probability
at most `1/2`, meets `TileLuck`'s condition with `M = 1` at `r·θ = 1/256`. -/
def TemplateCondition : Prop :=
  ∀ w : ℕ, w ≤ 32 →
    (1 / 2 : ℝ) ^ w * Real.exp (32 * (1 / 256 * w)) +
      (1 - (1 / 2 : ℝ) ^ w) * (1 / 2 * Real.exp (1 / 256 * w) + (1 - 1 / 2) * Real.exp (-(4 * (1 / 256) * w))) ^ 32
        ≤ 1

open Classical in
/-- **The template's moment (its θ).** `TileLuck` at h1tile's segment: 32 columns share the rows' coins `Ω₀`, column
`k` has its own coins in `E`, and each of the `r` rows' words skips `w k i ≤ 32` products, dead with probability at most
`2^{-w}` and otherwise lucky with probability at most `1/2`. At `κ = 4` the segment's gain has moment at most 1 at rate
`θ = 1/(256·r)` a product, so a seed group's premium is at most `256·r·ln(T/δ)` products. -/
def TemplateMoment : Prop :=
  ∀ {Ω₀ E : Type} [Fintype Ω₀] [Fintype E] {r : ℕ} (w : Fin 32 → Fin r → ℕ) (dead : Fin 32 → Fin r → Ω₀ → Prop)
    (lucky : Fin 32 → Fin r → Ω₀ → E → Prop), 0 < r → (∀ k i, w k i ≤ 32) →
    (∀ k i ω₀, ¬dead k i ω₀ → prob (lucky k i ω₀) ≤ 1 / 2) → (∀ k i, prob (dead k i) ≤ (1 / 2) ^ w k i) →
    ∑ ω : Ω₀ × (Fin 32 → E),
        Real.exp (1 / (256 * r) * ∑ k, ∑ i, (w k i : ℝ) * ((1 + 4) * (if lucky k i ω.1 (ω.2 k) then 1 else 0) - 4)) ≤
      Fintype.card (Ω₀ × (Fin 32 → E))

end Accounting.Guarantees.TS
