import Mathlib.Combinatorics.SimpleGraph.Clique
import Mathlib.Combinatorics.SimpleGraph.Finite
import Mathlib.Data.ZMod.Basic
import Definitions.Pous.Accounting.LemmaA
import Definitions.Pous.Accounting.Params
import Definitions.Pous.Accounting.PubMeets
import Definitions.Pous.Accounting.GradedMeets
import Definitions.Pous.Accounting.Rebuild
import Specs.Pous.Assumptions
import Definitions.Pous.Model.HashChain
import Definitions.Pous.Model.Dense
import Definitions.Pous.Model.Inc
import Definitions.Pous.Model.Invertible
import Definitions.Pous.Model.Labelling
import Definitions.Pous.Model.M1
import Definitions.Pous.Model.P3Chain
import Definitions.Pous.Model.BandChain
import Definitions.Pous.Model.Pebbling
import Definitions.Pous.Model.SampledAudit
import Specs.Pous.Guarantees.P2
import Specs.Pous.Guarantees.SecureErasure
import Definitions.Pous.Model.BandGraph
import Definitions.Pous.Model.SegLift
import Definitions.Pous.Model.Continuous

/-!
# Guarantees

The statements `grade.sh` grades against. A submission proves exactly one of them, as
`theorem solution : Pous.Guarantees.<Name>`. Named assumptions (`verity/Security/Specs/Pous/Assumptions.lean`) appear as
hypotheses of the pinned `Prop`, never as axioms. "Pinned as drafted" means the binders and conclusion are the draft's,
verbatim.
-/

namespace Pous.Guarantees

open scoped ENNReal

/-- **The idealized audit floor** (problem statement, "Sufficient target"): `k = 86` is the least `k` with
`ρ^k ≤ δ`. The grader's positive control. -/
def KFloor : Prop :=
  IsLeast {k : ℕ | Params.ρ ^ k ≤ Params.δ} 86

/-- **Report Theorem 1(ii), untimed.** `INC(β, π)` implies that a product-form responder with an `S`-bit state
passes `k` uniform challenges with probability at most `π + p₀^k`, `p₀ = 1 − (β − S − B)/(B·ℓ)`. -/
def Theorem1ii : Prop :=
  ∀ (M : IdealModel) (sch : Scheme M) (β S k : ℕ) (π : ℝ≥0∞), INC sch β π →
    ∀ (W : Bits sch.n) (A₁ : Preprocessor sch S) (A₂ : ProdResponder sch S),
      pr (prodAccepts sch k W A₁ A₂) ≤ π + ENNReal.ofReal (p₀ β S sch.B sch.ℓ) ^ k

/-- **Report Theorem 1(ii), timed.** The same bound for `(D, Q)`-bounded product-form responders, from `INC(β, π)`
against expanders that rebuild each block within `(D, Q)`. -/
def Theorem1iiTimed : Prop :=
  ∀ (M : IdealModel) (sch : Scheme M) (D Q β S k : ℕ) (π : ℝ≥0∞), TimedINC sch D Q β π →
    ∀ (W : Bits sch.n) (A₁ : Preprocessor sch S) (A₂ : TimedProdResponder sch S D Q),
      pr (timedProdAccepts sch k W A₁ A₂) ≤ π + ENNReal.ofReal (p₀ β S sch.B sch.ℓ) ^ k

/-! ## The report's reductions in the timed game -/

/-- **Pinned as drafted** (`PousTargets.theorem1_i_timed`).

Report Theorem 1(i), timed: a sequential audit bound gives `TimedINC` at the state size. -/
def Theorem1iTimed : Prop :=
  ∀ {M : IdealModel} (sch : Scheme M) (S D Q k : ℕ) (δ ε : ℝ≥0∞)
    (h : AuditSecure sch .sequential S D Q k δ ε),
    TimedINC sch D Q S (δ + ε)

/-- **Pinned as drafted** (`PousTargets.theorem1_ii_seq`).

Theorem 1(ii) for stateful sequential responders: one compressor per history of fewer than `k` indices, then a
union bound (the plug-in of tradeoff Theorem 1).

The coefficient `Σ_{j<k} B^j` is the union over histories; the tighter `k·π` is plausible but unproved. -/
def Theorem1iiSeq : Prop :=
  ∀ {M : IdealModel} (sch : Scheme M) (D Q β S k : ℕ) (π : ℝ≥0∞)
    (hinc : TimedINC sch D Q β π) (W : Bits sch.n) (A₁ : Preprocessor sch S) (A₂ : SeqResponder sch S D Q),
    pr (seqAccepts sch k W A₁ A₂) ≤
      (∑ j ∈ Finset.range k, (sch.B : ℝ≥0∞) ^ j) * π + ENNReal.ofReal (p₀ β S sch.B sch.ℓ) ^ k

/-- **Pinned as drafted** (`PousTargets.theorem1_iii_timed`).

Report Theorem 1(iii), timed, simultaneous reveal, by the sampling compressor. The compressor's seed is fixed
inside each deterministic `(cmp, exp)` pair, so it costs no bits (the report charges `λ` bits for it).

`lam` is the report's miss parameter `λ'`, not the seed length. The seed costs nothing because `TimedINC`
bounds no compressor; if `INC` ever limited the compressor's size, a built-in seed would be non-uniform advice. -/
def Theorem1iiiTimed : Prop :=
  ∀ {M : IdealModel} (sch : Scheme M) (D Q β S k lam R : ℕ) (π : ℝ≥0∞) (η : ℝ)
    (hη0 : 0 < η) (hη1 : η < 1) (hR : R = ⌈((lam : ℝ) + Real.logb 2 sch.B) * Real.log 2 / η⌉₊)
    (hRℓ : Real.logb 2 R < sch.ℓ) (hinc : TimedINC sch D Q β π)
    (W : Bits sch.n) (A₁ : Preprocessor sch S) (A₂ : SimResponder sch S D Q k),
    pr (simAccepts sch W A₁ A₂) ≤
      k * π + ENNReal.ofReal ((1 - ((β : ℝ) - S - sch.B) / (sch.B * sch.ℓ)) * sch.ℓ / (sch.ℓ - Real.logb 2 R)) ^ k
        + k * ENNReal.ofReal η + k * (2⁻¹ : ℝ≥0∞) ^ lam

/-- **Pinned as drafted** (`PousTargets.fibre_count`).

Fibre counting (report Proposition 9(a), audit §5 item 1): untimed `INC(β, π)` for a correct scheme forces
`(1 − π)·2^(n + β) ≤ 2^(t + B·ℓ)`. Within the space bound this is `β ≲ 0.05·n`, far below `ρ|C|`, so the untimed
`INC` hypothesis of `Guarantees.Theorem1ii` cannot hold at the operating point in this model. -/
def FibreCount : Prop :=
  ∀ {M : IdealModel} (sch : Scheme M) (hc : sch.Correct) (β : ℕ) (π : ℝ≥0∞) (h : INC sch β π),
    (1 - π) * 2 ^ (sch.n + β) ≤ 2 ^ (sch.t + sch.B * sch.ℓ)

/-! ## Track A (construction paths §4; sms5 ideal-model review) -/

/-- **Pinned as drafted** (`PousTargets.a1_compression_card`).

A1. Distinct descriptions of at most `L` bits: fewer than `2^(L+1)` outcomes. -/
def A1CompressionCard : Prop :=
  ∀ {α : Type} [Fintype α] (E : α → Prop) [DecidablePred E] (L : ℕ)
    (enc : α → List Bool) (hlen : ∀ x, E x → (enc x).length ≤ L)
    (hinj : ∀ x y, E x → E y → enc x = enc y → x = y),
    (Finset.univ.filter E).card < 2 ^ (L + 1)

/-- **Pinned as drafted** (`PousTargets.a1_compression_pr`).

A1, probability form. -/
def A1CompressionPr : Prop :=
  ∀ {α : Type} [Fintype α] (E : α → Prop) (L : ℕ) (enc : α → List Bool)
    (hlen : ∀ x, E x → (enc x).length ≤ L) (hinj : ∀ x y, E x → E y → enc x = enc y → x = y),
    pr E ≤ 2 ^ (L + 1) / Fintype.card α

/-- **Pinned as drafted** (`PousTargets.a2_lemmaA_m1`).

A2. Lemma A in M1 (tradeoff §1.2) for a per-block responder, with exact binomials: the blocks it gets right
number at least `g` with probability at most `2^S·C(B, g)·Σ_{h ≤ g} C((g+1)·Q, h) / 2^(ℓ·g)`. -/
def A2LemmaAM1 : Prop :=
  ∀ (B ℓ S D Q g : ℕ) [NeZero B] (W : Bits (B * ℓ)) (A₁ : Preprocessor (m1 B ℓ) S)
    (A₂ : TimedProdResponder (m1 B ℓ) S D Q),
    pr (fun ω : (perms B ℓ).Ω =>
        let pp := ((m1 B ℓ).enc W ω ()).1
        let C := ((m1 B ℓ).enc W ω ()).2
        g ≤ (Finset.univ.filter fun i =>
          (A₂.prog W pp (A₁ W pp C ω) i).run ((perms B ℓ).answer ω) = C i).card)
      ≤ 2 ^ S * (B.choose g) * (∑ h ∈ Finset.range (g + 1), (((g + 1) * Q).choose h : ℝ≥0∞)) / 2 ^ (ℓ * g)

open Assumptions in
/-- **Pinned as drafted** (`lean/submissions/track-a/A3Prime.lean`, `a3_prime`).

**A3′** (sms5-review-ideal, Track A; the target game's class (b): the state is `S` bits at every challenge and
each answer gets `(D, Q)`, so `T := Q`). Lemma A for each of the `|Hist| = Σ_{j<k} B^j` histories, a union bound,
and the conditional product give `Pr[pass] ≤ ((g − 1)/B)^k + |Hist|·Γ + Pr[not G]`. -/
def A3Prime : Prop :=
  ∀ {M : IdealModel} (sch : Scheme M) (S D Q k g : ℕ) (Γ : ℝ≥0∞)
    (G : Bits sch.n → M.Ω × sch.Coins → Prop) (hA : LemmaA sch S D Q g Γ G)
    (W : Bits sch.n) (A₁ : Preprocessor sch S) (A₂ : SeqResponder sch S D Q),
    pr (seqAccepts sch k W A₁ A₂) ≤
      (((g - 1 : ℕ) : ℝ≥0∞) / sch.B) ^ k + (∑ j ∈ Finset.range k, (sch.B : ℝ≥0∞) ^ j) * Γ +
        pr (fun a => ¬ G W a)

open Assumptions in
/-- **Pinned as drafted** (`lean/submissions/track-a/A3Prime.lean`, `a3_prime_lam`).

A3′ in the review's form: if `|Hist|·Γ ≤ 2^-λ'`, then `Pr[pass] ≤ ((g − 1)/B)^k + 2^-λ' + Pr[not G]`. -/
def A3PrimeLam : Prop :=
  ∀ {M : IdealModel} (sch : Scheme M) (S D Q k g lam : ℕ) (Γ : ℝ≥0∞)
    (G : Bits sch.n → M.Ω × sch.Coins → Prop) (hA : LemmaA sch S D Q g Γ G)
    (hΓ : (∑ j ∈ Finset.range k, (sch.B : ℝ≥0∞) ^ j) * Γ ≤ (2⁻¹ : ℝ≥0∞) ^ lam)
    (W : Bits sch.n) (A₁ : Preprocessor sch S) (A₂ : SeqResponder sch S D Q),
    pr (seqAccepts sch k W A₁ A₂) ≤
      (((g - 1 : ℕ) : ℝ≥0∞) / sch.B) ^ k + (2⁻¹ : ℝ≥0∞) ^ lam + pr (fun a => ¬ G W a)

/-- **Pinned as drafted** (`PousTargets.a5_rabin_gcd`).

A5. The Rabin regeneration leg: two square roots that differ other than by sign split `N`. -/
def A5RabinGcd : Prop :=
  ∀ (N : ℕ) (x y : ZMod N) (h : x ^ 2 = y ^ 2) (hxy : x ≠ y) (hxy' : x ≠ -y),
    1 < Nat.gcd (x - y).val N ∧ Nat.gcd (x - y).val N < N

/-- **Pinned as drafted** (`PousTargets.a6_rederive_untimed`).

A6, untimed. If `C = R(W, pp, z)` for some `κ`-bit advice `z` (any function of the setup), `INC(κ, π)` fails
for every `π < 1`. -/
def A6RederiveUntimed : Prop :=
  ∀ {M : IdealModel} (sch : Scheme M) (κ : ℕ) (π : ℝ≥0∞)
    (z : Bits sch.n → M.Ω → sch.Coins → Bits κ) (R : Bits sch.n → Bits sch.t → Bits κ → M.Ω → sch.Code)
    (hR : ∀ W ω r, R W (sch.enc W ω r).1 (z W ω r) ω = (sch.enc W ω r).2) (h : INC sch κ π),
    1 ≤ π

/-- **Pinned as drafted** (`PousTargets.a6_rederive_timed`).

A6, timed. If every block is re-derived from `κ`-bit advice by a `(D, Q)`-bounded program, a `κ`-bit adversary
passes every sequential audit with probability 1 (the timed form of `redesign.md` Proposition 1). -/
def A6RederiveTimed : Prop :=
  ∀ {M : IdealModel} (sch : Scheme M) (κ D Q k : ℕ)
    (z : Bits sch.n → M.Ω → sch.Coins → Bits κ)
    (R : Bits sch.n → Bits sch.t → Bits κ → Fin sch.B → Prog M (Bits sch.ℓ))
    (hRb : ∀ W pp s i, (R W pp s i).Bounded D Q)
    (hR : ∀ W ω r i, (R W (sch.enc W ω r).1 (z W ω r) i).run (M.answer ω) = (sch.enc W ω r).2 i),
    ∃ (A₁ : Preprocessor sch κ) (A₂ : SeqResponder sch κ D Q), ∀ W, pr (seqAccepts sch k W A₁ A₂) = 1

/-- **A7′** (sms5 review, restating the A7 draft), in certified log form: at `B = 2^28` blocks,
`S = ⌊(18/19)·2056·B⌋` (RW–SMS records of `w + 8 = 2056` bits), `λ' = 128` and `log₂ M ≥ 2048 − 2.8·10⁻⁶`
(`M = φ(N)`), the history union closes and `k` challenges reach 1%: `k = 117` (M1, `T = 2^20`), `142` (M1, `2^34`),
`175` (M2, `2^20`) and `238` (M2, `2^34`). The review's `g` (258,033,293; 259,824,025; 261,446,274; 263,285,578) are
witnesses, with 5,700–45,500 blocks to spare below the `((g − 1)/B)^k` limit. The M1 rows are Conjecture-B1′
numbers for RW–SMS. -/
def A7Params : Prop :=
  A7Row logHistGammaM1 (2 ^ 28) ⌊(18 / 19 : ℚ) * (2056 * 2 ^ 28)⌋₊ (2 ^ 20) 117 ∧
    A7Row logHistGammaM1 (2 ^ 28) ⌊(18 / 19 : ℚ) * (2056 * 2 ^ 28)⌋₊ (2 ^ 34) 142 ∧
    A7Row logHistGammaM2 (2 ^ 28) ⌊(18 / 19 : ℚ) * (2056 * 2 ^ 28)⌋₊ (2 ^ 20) 175 ∧
    A7Row logHistGammaM2 (2 ^ 28) ⌊(18 / 19 : ℚ) * (2056 * 2 ^ 28)⌋₊ (2 ^ 34) 238

/-- **`m1_meets`, conditional on Conjecture B1′** (sms5 review; red team). In the ideal permutation model M1, the
scheme with `2^28` blocks of 2048 bits meets the requirement: 117 sequential challenges, `2^20` queries per answer,
any depth, `ε = 2^-128`. It follows from A2, A3′ and A7′'s first row. B1′ is a hypothesis so that the claim cannot
be read as one about RW–SMS; in M1 it follows from A2 and A3′.

**It does not transfer to the real RW–SMS scheme at `λ = 128`** (sms5 F1): a 2048-bit `N` factors in about
`2^112 < 2^128` preprocessing, after which two root extractions answer any challenge within the deadline. -/
def M1Meets : Prop :=
  ∀ D : ℕ,
    Assumptions.B1Prime (m1 (2 ^ 28) 2048) (2 ^ 2048) ((m1 (2 ^ 28) 2048).stateBound Params.ρ) D (2 ^ 20) 117 →
    Meets (m1 (2 ^ 28) 2048) .sequential D (2 ^ 20) 117 Params.εMax

/-! ## Track B -/

/-- **Pinned as drafted** (`PousTargets.b1_indep_set`).

B1. A graph of maximum degree `Δ` on `n` vertices has an independent set of at least `n/(Δ + 1)` vertices. -/
def B1IndepSet : Prop :=
  ∀ (n Δ : ℕ) (G : SimpleGraph (Fin n)) [DecidableRel G.Adj] (hΔ : ∀ v, G.degree v ≤ Δ),
    ∃ s : Finset (Fin n), G.IsIndepSet (s : Set (Fin n)) ∧ n ≤ (Δ + 1) * s.card

open DRG in
/-- **Pinned as drafted** (`lean/submissions/pebbling/B2StackingDraft.lean`, which proves it in this
strengthened form: unpebbled green paths).

**B2** (Fisch 2018/702, Claim 12). In `ℓ ≥ 1/(1 − α − δ)` stacked `(n, α, β)`-depth-robust layers joined by
superconcentrators, any `(1 − ε)n` black pebbles plus at most `δn` red pebbles per level, with `δ < ε/2`, leave an
unpebbled path of at least `βn` nodes ending in the last layer `V_t`, each of whose nodes has an unpebbled green path
to a node of `V_t`. -/
def B2Stacking : Prop :=
  ∀ {ℓ n : ℕ} {V : Type} [Fintype V] [DecidableEq V] (α β ε δ : ℝ)
    (node : Fin ℓ → Fin n → V) (level : V → Fin ℓ) (drg : Fin ℓ → Fin n → Fin n → Prop) (green : V → V → Prop)
    (hnode : ∀ i, Function.Injective (node i)) (hlevel : ∀ i a, level (node i a) = i)
    (hdag : ∃ r : V → ℕ, ∀ u v, StackedEdge node drg green u v → r u < r v)
    (hdr : ∀ i, DepthRobust n (drg i) α β)
    (hsc : ∀ i j : Fin ℓ, (j : ℕ) = i + 1 → DisjointPaths node level green i j)
    (hn : 0 < n) (hε : 0 < ε) (hδ : δ < ε / 2) (hℓ : 1 ≤ (1 - α - δ) * ℓ)
    (black red : Finset V) (hblack : (black.card : ℝ) ≤ (1 - ε) * n)
    (hred : ∀ i, ((red.filter fun v => level v = i).card : ℝ) ≤ δ * n),
    ∀ t : Fin ℓ, (t : ℕ) + 1 = ℓ →
      ∃ P : List V,
        P.IsChain (StackedEdge node drg green) ∧ P.Nodup ∧ β * n ≤ P.length ∧
        (∀ v ∈ P, Unpebbled black red v) ∧
        (∃ b, P.getLast? = some (node t b)) ∧
        ∀ v ∈ P, ∃ g : List V, g.head? = some v ∧ g.IsChain green ∧ (∀ w ∈ g, Unpebbled black red w) ∧
          ∃ b, g.getLast? = some (node t b)

open HashChain in
/-- **Pinned as drafted** (`PousTargets.b3_chain`).

B3. Hash-chain sequentiality: `D` rounds of at most `q` random-oracle queries output the end `x_{D+1}` of the
tweaked chain with probability at most `(D·q + 1)/m`. (The memo's `(D + 1)·q/m` fails at `q = 0`, and fails for an
untweaked chain `x_{i+1} = H(x_i)`, which can cycle back onto queried points.) -/
def B3Chain : Prop :=
  ∀ (m D q : ℕ) [NeZero m] (x₀ : Fin m)
    (P : Prog (IdealModel.randomOracle (Fin (D + 1) × Fin m) (Fin m)) (Fin m)) (hP : P.RoundBounded D q),
    pr (fun H : Fin (D + 1) × Fin m → Fin m => P.run H = chain H x₀ (D + 1)) ≤ ((D * q + 1 : ℕ) : ℝ≥0∞) / m

/-! ## Fisch Claims 13–14 and B5 -/

open DRG in
/-- **Pinned as drafted** (`lean/submissions/pebbling/FischClaims1314Draft.lean`, `claim13`).

**Claim 13** (Fisch 2018/702). Under B2's hypotheses with `δ < ε/4`, at least `εn/2` nodes of the last layer
`V_t` end an unpebbled path of at least `βn` nodes, each of whose nodes has an unpebbled green path to `V_t`. -/
def FischClaim13 : Prop :=
  ∀ {ℓ n : ℕ} {V : Type} [Fintype V] [DecidableEq V] (α β ε δ : ℝ)
    (node : Fin ℓ → Fin n → V) (level : V → Fin ℓ) (drg : Fin ℓ → Fin n → Fin n → Prop) (green : V → V → Prop)
    (hnode : ∀ i, Function.Injective (node i)) (hlevel : ∀ i a, level (node i a) = i)
    (hdag : ∃ r : V → ℕ, ∀ u v, StackedEdge node drg green u v → r u < r v)
    (hdr : ∀ i, DepthRobust n (drg i) α β)
    (hsc : ∀ i j : Fin ℓ, (j : ℕ) = i + 1 → DisjointPaths node level green i j)
    (hn : 0 < n) (hε : 0 < ε) (hδ : δ < ε / 4) (hℓ : 1 ≤ (1 - α - δ) * ℓ)
    (black red : Finset V) (hblack : (black.card : ℝ) ≤ (1 - ε) * n)
    (hred : ∀ i, ((red.filter fun v => level v = i).card : ℝ) ≤ δ * n),
    ∀ t : Fin ℓ, (t : ℕ) + 1 = ℓ →
      ∃ E : Finset (Fin n), ε / 2 * n ≤ E.card ∧ ∀ b ∈ E,
        ∃ P : List V,
          P.IsChain (StackedEdge node drg green) ∧ P.Nodup ∧ β * n ≤ P.length ∧
          (∀ v ∈ P, Unpebbled black red v) ∧ P.getLast? = some (node t b) ∧
          ∀ v ∈ P, ∃ g : List V, g.head? = some v ∧ g.IsChain green ∧ (∀ w ∈ g, Unpebbled black red w) ∧
            ∃ b', g.getLast? = some (node t b')

open DRG in
/-- **Pinned as drafted** (`lean/submissions/pebbling/FischClaims1314Draft.lean`, `claim14`).

**Claim 14** (Fisch 2018/702). In the white/green pebbling game, an unpebbled path each of whose nodes has an
unpebbled green path to the last layer `V_t` is pebbled in order: its `k`-th node carries no pebble before round
`k + 1`. -/
def FischClaim14 : Prop :=
  ∀ {ℓ n : ℕ} {V : Type} (node : Fin ℓ → Fin n → V) (drg : Fin ℓ → Fin n → Fin n → Prop) (green : V → V → Prop)
    (t : Fin ℓ) (htop : ∀ b u, ¬ green (node t b) u)
    (black red : Finset V) (C : ℕ → Finset V) (hC0 : ∀ v, Unpebbled black red v → v ∉ C 0)
    (hC : IsPebbling node drg green C)
    (P : List V) (hP : P.IsChain (StackedEdge node drg green)) (hunp : ∀ v ∈ P, Unpebbled black red v)
    (hgreen : ∀ v ∈ P, ∃ g : List V, g.head? = some v ∧ g.IsChain green ∧ (∀ w ∈ g, Unpebbled black red w) ∧
      ∃ b, g.getLast? = some (node t b)),
    ∀ (k : ℕ) (hk : k < P.length) (r : ℕ), P[k] ∈ C r → k < r

open Labelling in
/-- **Pinned as drafted** (`lean/submissions/pebbling/B5ExPostFactoDraft.lean`, `b5_ex_post_facto_bounded`).

**B5** (Pietrzak 2018/194, Theorem 10 with §8's data-carrying labels, data fixed before the oracle), in the
`(D, Q)` model of the audit game. If `Φ(G, VC)` is `(s, D, k)`-hard, then an `m`-bit preprocessor `A₁` and
`(D, Q)`-bounded online programs `A₂` answer at least `k` challenges with probability at most
`N · 2^m · ((|VC|·(Q + 1) + N)² / 2^w)^(s+1)`.

Scoped to one-way labels: the standard (forward-only) pebbling game. It gives no bound for invertible,
locally decodable labels (P3), whose decoding is Fisch's reverse move. -/
def B5ExPostFactoBounded : Prop :=
  ∀ (N w m s D Q k : ℕ) (G : TopoDAG N) (VC : Finset (Fin N)) (d : Fin N → Bits w)
    (hard : PebblingHard G VC s D k)
    (A₁ : (LabelQry N w → Bits w) → Bits m)
    (A₂ : Bits m → Fin N → Prog (IdealModel.randomOracle (LabelQry N w) (Bits w)) (Bits w))
    (hA₂ : ∀ σ c, (A₂ σ c).Bounded D Q),
    pr (fun H : LabelQry N w → Bits w =>
        k ≤ (VC.filter fun c => (A₂ (A₁ H) c).run H = label G d H c).card) ≤
      N * 2 ^ m * ((((VC.card * (Q + 1) + N) ^ 2 : ℕ) : ℝ≥0∞) / 2 ^ w) ^ (s + 1)

open Labelling in
/-- **Pinned as drafted** (`lean/submissions/pebbling/B5ExPostFactoDraft.lean`, `b5_ex_post_facto`).

**B5** in the parallel-query model (`Prog.RoundBounded D q`, at most `D·q` queries per challenge).

Scoped to one-way labels: the standard (forward-only) pebbling game. It gives no bound for invertible,
locally decodable labels (P3), whose decoding is Fisch's reverse move. -/
def B5ExPostFacto : Prop :=
  ∀ (N w m s D q k : ℕ) (G : TopoDAG N) (VC : Finset (Fin N)) (d : Fin N → Bits w)
    (hard : PebblingHard G VC s D k)
    (A₁ : (LabelQry N w → Bits w) → Bits m)
    (A₂ : Bits m → Fin N → Prog (IdealModel.randomOracle (LabelQry N w) (Bits w)) (Bits w))
    (hA₂ : ∀ σ c, (A₂ σ c).RoundBounded D q),
    pr (fun H : LabelQry N w → Bits w =>
        k ≤ (VC.filter fun c => (A₂ (A₁ H) c).run H = label G d H c).card) ≤
      N * 2 ^ m * ((((VC.card * (D * q + 1) + N) ^ 2 : ℕ) : ℝ≥0∞) / 2 ^ w) ^ (s + 1)

/-! ## N10: invertible labels (PIEs ePrint 2018/684 Conjecture 1 at L = 2) -/

open Pous.Invertible in
/-- **Pinned as drafted** (`lean/submissions/pebbling/InvertibleExPostFactoDraft.lean`, `transpose_mixing`).

 **Transpose mixing (proved).** Fix a lower node `u`. Its label `c¹_u` collects one component from every top
label: component `v` of `c¹_u` is a function of `c²_v` alone (`decodeLower … u v` depends on `c2 v`), and every `v`
contributes. Hence if the top labels are known only on a proper subset `S ⊊ univ`, the lower label `c¹_u` is
undetermined: there are two top configurations agreeing on `S` whose `u`-th lower labels differ. So no adversary
recovers any lower label without all `n` top labels — the naive partial-storage shortcut fails. -/
def TransposeMixing : Prop :=
  ∀ {n wc : ℕ} (P : Equiv.Perm (Lab n wc)) (k2 : Fin n → Lab n wc) (hwc : 0 < wc)
    (u : Fin n) (S : Finset (Fin n)) (hS : S ≠ Finset.univ),
    ∃ c2 c2' : Fin n → Lab n wc, (∀ v ∈ S, c2 v = c2' v) ∧ decodeLower P c2 k2 u ≠ decodeLower P c2' k2 u

open Pous.Invertible in
/-- **Pinned as drafted** (`lean/submissions/pebbling/InvertibleExPostFactoDraft.lean`, `reverse_needs_all_children`).

 **Reverse move needs all green children (proved).** Unfolding the trusted reverse rule for the `K_{n,n}`
connector: to place a pebble on a layer-0 node `u` by a reverse (decode) move from configuration `C`, every layer-1
node must already carry a pebble. This is the labeling-side reason a decode step is not free — it is the whole top
layer, matching `transpose_mixing`. -/
def ReverseNeedsAllChildren : Prop :=
  ∀ {n : ℕ} (G : Labelling.TopoDAG n) (C : Finset (Fin 2 × Fin n)) (a : Fin n)
    (hrev : ¬ (∀ x, DRG.StackedEdge node2 (drg2 G) green2 x (node2 0 a) → x ∈ C))
    (havail : DRG.Available node2 (drg2 G) green2 C (node2 0 a)),
    ∀ b : Fin n, node2 1 b ∈ C

open Pous.Invertible in
/-- **Pinned as drafted** (`lean/submissions/pebbling/InvertibleExPostFactoDraft.lean`, `inv_ex_post_facto_two_layer`).

 **N10 as drafted (false, see `N10Refuted`).** If the two-layer P3 graph is white/green `(s, D, k)`-hard, then an
`m`-bit preprocessor and `(D, Q)`-bounded online programs — which may query `P⁻¹` — answer at least `k` top-label
challenges with probability at most the B5 bound at width `n·wc` and `Q_tot = n·(Q + 1) + 2n`.

Kept only as the statement `N10Refuted` negates; it is no longer a target. -/
def InvExPostFactoTwoLayer : Prop :=
  ∀ (n wc m s D Q k : ℕ) (G : Labelling.TopoDAG n) (W : Fin n → Lab n wc)
    (hwc : 0 < wc)
    (hard : WhiteGreenHard G s D k)
    (A₁ : (p3Model n wc).Ω → Bits m)
    (A₂ : Bits m → Fin n → Prog (p3Model n wc) (Lab n wc))
    (hA₂ : ∀ σ c, (A₂ σ c).Bounded D Q),
    pr (fun ω : (p3Model n wc).Ω =>
        k ≤ (Finset.univ.filter fun v => (A₂ (A₁ ω) v).run ((p3Model n wc).answer ω) = topLabel G W ω v).card) ≤
      (n : ℝ≥0∞) * 2 ^ m * ((((n * (Q + 1) + 2 * n) ^ 2 : ℕ) : ℝ≥0∞) / 2 ^ (n * wc)) ^ (s + 1)

/-- **Pinned as proved** (`lean/submissions/p3-forward/Refutation.lean`, `not_invExPostFactoTwoLayer`).

 **N10 is false.** White/green hardness charges a missing lower label a whole label, but top label `v` reads only
column `v` of the lower layer, so an adversary pays for lower labels a component at a time (by storing components,
`docs/p3-cryptanalysis.md` §1, or by guessing them). -/
def N10Refuted : Prop := ¬ InvExPostFactoTwoLayer

/-! ## Sampled audits and the dense scheme -/

/-- **Pinned as drafted** (`lean/submissions/b5/SampledAudit.lean`, `sampled_audit_prod`).

 **Product form** (simultaneous reveal, each answer from the state and its own index: `TimedProdResponder`).
If, except with probability `p` over the setup, fewer than `k` blocks are answerable, a `t`-sample audit with
replacement passes with probability at most `p + ((k − 1)/B)^t`. -/
def SampledAuditProd : Prop :=
  ∀ {M : IdealModel} (sch : Scheme M) (S D Q t k : ℕ) (p : ℝ≥0∞) (W : Bits sch.n)
    (A₁ : Preprocessor sch S) (A₂ : TimedProdResponder sch S D Q)
    (h : pr (fun a : M.Ω × sch.Coins => k ≤ (Finset.univ.filter fun i =>
      (A₂.prog W (sch.enc W a.1 a.2).1 (A₁ W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1) i).run
        (M.answer a.1) = (sch.enc W a.1 a.2).2 i).card) ≤ p),
    pr (timedProdAccepts sch t W A₁ A₂) ≤ p + (((k - 1 : ℕ) : ℝ≥0∞) / sch.B) ^ t

open Pous.Sampled in
/-- **Pinned as drafted** (`lean/submissions/b5/SampledAudit.lean`, `sampled_audit_seq`).

 **Sequential reveal.** If after every history of fewer than `t` indices, except with probability `p` over the
setup, fewer than `k` blocks are answerable in the next round, a `t`-round audit passes with probability at most
`((k − 1)/B)^t + |Hist|·p`, where `|Hist| = Σ_{j<t} B^j`. -/
def SampledAuditSeq : Prop :=
  ∀ {M : IdealModel} (sch : Scheme M) (S D Q t k : ℕ) (p : ℝ≥0∞) (W : Bits sch.n)
    (A₁ : Preprocessor sch S) (A₂ : SeqResponder sch S D Q)
    (h : ∀ hist : List (Fin sch.B), hist.length < t →
      pr (fun a : M.Ω × sch.Coins => k ≤ (Finset.univ.filter fun i =>
        ((A₂.step W (sch.enc W a.1 a.2).1 hist.length
          (seqState (M.answer a.1) (A₂.step W (sch.enc W a.1 a.2).1) 0 hist
            (A₁ W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1)) i).run (M.answer a.1)).1 =
          (sch.enc W a.1 a.2).2 i).card) ≤ p),
    pr (seqAccepts sch t W A₁ A₂) ≤
      (((k - 1 : ℕ) : ℝ≥0∞) / sch.B) ^ t + (∑ j ∈ Finset.range t, (sch.B : ℝ≥0∞) ^ j) * p

/-- **Pinned as drafted** (`lean/submissions/b5/SampledAudit.lean`, `sampled_audit_sim`).

 **General simultaneous reveal** (the answer at position `j` may depend on the whole challenge tuple). As in
`theorem1_iii_timed`, the answerable set at position `j` is `L_j = {i : Pr_I[answer_j(I[j := i]) = c_i] ≥ η}`. If,
except with probability `p` over the setup, every `|L_j| < k`, the audit passes with probability at most
`p + ((k − 1)/B)^t + t·η`. -/
def SampledAuditSim : Prop :=
  ∀ {M : IdealModel} (sch : Scheme M) (S D Q t k : ℕ) (p η : ℝ≥0∞) (W : Bits sch.n)
    (A₁ : Preprocessor sch S) (A₂ : SimResponder sch S D Q t)
    (h : pr (fun a : M.Ω × sch.Coins => ∃ j : Fin t, k ≤ (Finset.univ.filter fun i =>
      η ≤ pr (fun I : Fin t → Fin sch.B =>
        (A₂.prog W (sch.enc W a.1 a.2).1 (A₁ W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1)
          (Function.update I j i)).run (M.answer a.1) j = (sch.enc W a.1 a.2).2 i)).card) ≤ p),
    pr (simAccepts sch W A₁ A₂) ≤ p + ((((k - 1 : ℕ) : ℝ≥0∞) / sch.B) ^ t + t * η)

open DRG Labelling in
/-- **Pinned as drafted** (`lean/submissions/b5/SampledAudit.lean`, `stacked_audit_seq`).

 **Sequential audit of the stacked one-way graph** (B5's game). The challenges are `t` uniform last-layer
indices, the responder carries an `S`-bit state and answers each within `(⌊βn⌋ − 1, Q)`, and it passes iff every
answer is the challenged label. Each history's next answers come from a preprocessor (the state after that
history) and a per-challenge program, so `b5_stacked` bounds them, and the chain rule plus the union over
histories finish.

Conditional on its depth-robustness (`hdr`) and superconcentrator (`hsc`) hypotheses, which are inherent. -/
def StackedAuditSeq : Prop :=
  ∀ {ℓ n N : ℕ} {V : Type} [Fintype V] [DecidableEq V] (α β ε : ℝ)
    (node : Fin ℓ → Fin n → V) (level : V → Fin ℓ) (drg : Fin ℓ → Fin n → Fin n → Prop) (green : V → V → Prop)
    (hnode : ∀ i, Function.Injective (node i)) (hlevel : ∀ i a, level (node i a) = i)
    (hdr : ∀ i, DepthRobust n (drg i) α β)
    (hsc : ∀ i j : Fin ℓ, (j : ℕ) = i + 1 → DisjointPaths node level green i j)
    (hn : 0 < n) (hε : 0 < ε) (hε1 : ε ≤ 1) (hℓ : 1 ≤ (1 - α) * ℓ) (hβ : 1 ≤ β * n)
    (e : V ≃ Fin N) (G : TopoDAG N) (hG : ∀ u v, e u ∈ G.par (e v) ↔ StackedEdge node drg green u v)
    (t : Fin ℓ) (ht : (t : ℕ) + 1 = ℓ)
    (w S Q tt : ℕ) (d : Fin N → Bits w)
    (A₁ : (LabelQry N w → Bits w) → Bits S)
    (step : ℕ → Bits S → Fin n → Prog (IdealModel.randomOracle (LabelQry N w) (Bits w)) (Bits w × Bits S))
    (hstep : ∀ j σ b, (step j σ b).Bounded (⌊β * n⌋₊ - 1) Q),
    pr (fun x : (LabelQry N w → Bits w) × (Fin tt → Fin n) =>
        seqAnswers x.1 step 0 (List.ofFn x.2) (A₁ x.1) = List.ofFn fun j => label G d x.1 (e (node t (x.2 j)))) ≤
      (((n - ⌈ε / 2 * n⌉₊ : ℕ) : ℝ≥0∞) / n) ^ tt +
        (∑ j ∈ Finset.range tt, (n : ℝ≥0∞) ^ j) *
          (N * 2 ^ S * ((((n * (Q + 1) + N) ^ 2 : ℕ) : ℝ≥0∞) / 2 ^ w) ^ (⌊(1 - ε) * n⌋₊ + 1))

open DRG Labelling in
/-- **Pinned as drafted** (`lean/submissions/b5/SampledAudit.lean`, `stacked_audit_one_percent`).

 **At the example parameters**: `n = 2^16` last-layer blocks of `w = 8192` bits (1 KB), `N = 2^17` nodes,
`Q = 2^20` queries per answer, an `S = ⌊(18/19)·n·w⌋ = 508614548`-bit state (`ρ` of the last layer) and `ε = 1/23`,
a `t = 210`-round sequential audit passes with probability at most `1% + 2^-128`.

**Conditional** on a concrete depth-robust stacked graph (the `hdr`, `hsc` hypotheses; construction path N4, not yet
proved), and it covers the **audit clause only** of the stacked one-way scheme, not a full `Meets`. Its `ε = 1/23` is
the stacked graph's pebbling deficit, distinct from `ρ`'s `1/19`. -/
def StackedAuditOnePercent : Prop :=
  ∀ {ℓ : ℕ} {V : Type} [Fintype V] [DecidableEq V] (α β : ℝ)
    (node : Fin ℓ → Fin (2 ^ 16) → V) (level : V → Fin ℓ) (drg : Fin ℓ → Fin (2 ^ 16) → Fin (2 ^ 16) → Prop)
    (green : V → V → Prop)
    (hnode : ∀ i, Function.Injective (node i)) (hlevel : ∀ i a, level (node i a) = i)
    (hdr : ∀ i, DepthRobust (2 ^ 16) (drg i) α β)
    (hsc : ∀ i j : Fin ℓ, (j : ℕ) = i + 1 → DisjointPaths node level green i j)
    (hℓ : 1 ≤ (1 - α) * ℓ) (hβ : 1 ≤ β * (2 ^ 16 : ℕ))
    (e : V ≃ Fin (2 ^ 17)) (G : TopoDAG (2 ^ 17)) (hG : ∀ u v, e u ∈ G.par (e v) ↔ StackedEdge node drg green u v)
    (t : Fin ℓ) (ht : (t : ℕ) + 1 = ℓ) (d : Fin (2 ^ 17) → Bits 8192)
    (A₁ : (LabelQry (2 ^ 17) 8192 → Bits 8192) → Bits 508614548)
    (step : ℕ → Bits 508614548 → Fin (2 ^ 16) →
      Prog (IdealModel.randomOracle (LabelQry (2 ^ 17) 8192) (Bits 8192)) (Bits 8192 × Bits 508614548))
    (hstep : ∀ j σ b, (step j σ b).Bounded (⌊β * (2 ^ 16 : ℕ)⌋₊ - 1) (2 ^ 20)),
    pr (fun x : (LabelQry (2 ^ 17) 8192 → Bits 8192) × (Fin 210 → Fin (2 ^ 16)) =>
        seqAnswers x.1 step 0 (List.ofFn x.2) (A₁ x.1) = List.ofFn fun j => label G d x.1 (e (node t (x.2 j)))) ≤
      1 / 100 + (2⁻¹ : ℝ≥0∞) ^ 128

/-- **Pinned as drafted** (`lean/submissions/dense/DenseMeetsDraft.lean`, `PousDense.dense_meets`; the scheme is
in `Pous.Dense`).

 **The first end-to-end `Meets` with no named assumption** (`docs/p3-scheme.md` §4). The dense scheme on
`B = 2^16` blocks of 1 KB meets the requirement with sequential reveal, `D = 64` rounds and `Q = 2^20` queries per
answer, `k = 107` challenges and `ε_crypto = 2^-128`: it is correct, at rate 1, and every `⌊ρ|C|⌋`-bit adversary is
accepted with probability at most `1% + 2^-128`.

The first `Meets` with no named assumption. Scope: security, correctness and rate only, not decode cost, which is
`Θ(B)` per byte (decoding hashes every earlier block). -/
def DenseMeets : Prop :=
  Pous.Meets (Pous.Dense.denseScheme (2 ^ 16) 8192) .sequential 64 (2 ^ 20) 107 ((2⁻¹ : ENNReal) ^ 128)

/-! ## The overwrite chain: P3 with a concrete `H`, the band, and the dense chain scheme -/

/-! ### P3 with a concrete `H`: the overwrite chain on `Π₂` (`Definitions.Pous.Model.P3Chain`)

Added on 28 Sep 2026. Accepted on 29 Sep 2026: red team GO. The proofs are in this package
(`Pous.SecurityProofs.P3Concrete*`, from `verity/Security/Proofs/Pous/Chain/Flat.lean`).

The model is `Pous.Sponge.p3ChainModel 220 2384`: a tweakable ideal permutation `P` of 64 KB labels (an independent
uniform permutation per tweak `(tag, layer, node)`) and an untweaked ideal permutation `Π₂` of `2·524480 + 512` bits,
queried both ways. `H` is the overwrite chain on `Π₂` (`Pous.Sponge.ovKey`), so `D` and `Q` count `P^{±1}` and
`Π₂^{±1}` calls. The labels are P3's two layers on the band graph `B(220, k)` (`Pous.P3Column.bandGraph`): 220 labels
of `220·2384` bits (64 KB) per segment, 14.4 MB. The salt is 24 bytes, and segment `s` is keyed by `salt ‖ u64(s)`.

Scope: the ideal two-permutation model only. Lean does not prove anything about a concrete permutation (no Keccak or
other instantiation of `P` or `Π₂`), decode cost, the wall-clock-to-`Q` calibration, authentication, accounting or
fresh setup. As for every `Meets` here, `W` is fixed before the primitives are drawn. -/

/-- **Unconditional (no hypothesis): P3 with a concrete `H`, 64 KB labels, one segment (14.4 MB).**
`p3SchemeC 2384 B(220, 12)` meets the requirement with sequential reveal, `D = 10` rounds, `Q = 2^20` queries per
answer, `k = 106` challenges and `ε = 2^-128`: it is correct, `|C| ≥ |W|`, within the space bound, and every
`⌊ρ|C|⌋`-bit adversary is accepted with probability at most `δ + 2^-128`. Pinned as drafted
(`lean/submissions/p3-concrete-multi/finals/P3ConcreteFinals.lean`; graded with the namespaces `PousX` renamed `Pous.X`, `grading/`, `PousP3ConcreteMulti.p3_meets_concrete_64_D10_unconditional`; here `Pous.SecurityProofs.P3ConcreteMeets64`).
It rests on `PousP3Concrete.p3SegCharges` (`lean/submissions/p3-concrete/P3ChainSegCharges.lean`) through the ex post
facto target `P3ChainEPF`, and on the proved chain certificate `p3ChainCert` (`X = 1` at `D = 10`). Red team:
`bc-22298e90-fd61-5062-a836-0b7a423cab8a`, GO (`docs/lean-trusted-layer-review.md` §§50–51). Scope: as the section
header. -/
def P3ConcreteMeets64 : Prop :=
  Meets (Pous.P3Concrete.p3SchemeC 2384 (Pous.P3Column.bandGraph 220 12)) .sequential 10 (2 ^ 20) 106
    ((2⁻¹ : ℝ≥0∞) ^ 128)

/-- **Unconditional (no hypothesis): P3 with a concrete `H`, 64 KB labels, 14.005 GB (`N = 971` segments).**
`p3SchemeNC 2384 B(220, 12) 971` meets the requirement with sequential reveal, `D = 10`, `Q = 2^20`, `k = 106` and
`ε = 2^-128`. Security is joint over all 971 segments: one advice string for the whole store, and segment-by-segment
bounds do not combine to this (`not_jointClassWeight`). Pinned as drafted
(`lean/submissions/p3-concrete-multi/finals/P3ConcreteFinals.lean`; graded with the namespaces `PousX` renamed `Pous.X`, `grading/`,
`PousP3ConcreteMulti.p3_meets_concrete_multi_64_D10_unconditional`; here `Pous.SecurityProofs.P3ConcreteMeets14GB`). It rests on `p3SegCharges` through
`P3ChainEPFMulti` and the joint bridge (`PousP3ConcreteMulti.p3ChainEPFMulti_of_segCharges`), and on `p3ChainCert`.
Red team: as `P3ConcreteMeets64` (GO, §50). Scope: as the section header. -/
def P3ConcreteMeets14GB : Prop :=
  Meets (Pous.P3Concrete.p3SchemeNC 2384 (Pous.P3Column.bandGraph 220 12) 971) .sequential 10 (2 ^ 20) 106
    ((2⁻¹ : ℝ≥0∞) ^ 128)

/-- **Unconditional (no hypothesis): the fallback, P3 with a concrete `H` at `D = 12` on `B(220, 16)`, one
segment.** `p3SchemeC 2384 B(220, 16)` meets the requirement with sequential reveal, `D = 12`, `Q = 2^20`, `k = 106`
and `ε = 2^-128`. Pinned as drafted (`lean/submissions/p3-concrete-multi/finals/P3ConcreteFinals.lean`; graded with the namespaces `PousX` renamed `Pous.X`, `grading/`,
`PousP3ConcreteMulti.p3_meets_concrete_64_D12_band16_unconditional`; here `Pous.SecurityProofs.P3ConcreteMeets64D12`), from `P3ChainEPF` and the certificate
`p3ChainCert_band16`. Red team: as `P3ConcreteMeets64` (GO, §50). Scope: as the section header. -/
def P3ConcreteMeets64D12 : Prop :=
  Meets (Pous.P3Concrete.p3SchemeC 2384 (Pous.P3Column.bandGraph 220 16)) .sequential 12 (2 ^ 20) 106
    ((2⁻¹ : ℝ≥0∞) ^ 128)

/-! ### The band scheme and the dense chain scheme over the overwrite chain (`Definitions.Pous.Model.BandChain`)

Added on 29 Sep 2026. Red team: `bc-22298e90-fd61-5062-a836-0b7a423cab8a`, GO
(`docs/lean-trusted-layer-review.md` §§52–57). The proofs
are in this package (`Pous.SecurityProofs.Band*`, `Pous.SecurityProofs.ChainDense*`, from
`verity/Security/Proofs/Pous/Chain/Flat.lean`), and each was graded PASS
(`lean/submissions/sponge-dense/grading/GRADE.txt`).

The model is `Pous.Sponge.idealPerm (2·524288 + 512)`: one uniform permutation `Π₂` of `1049088` bits, queried both
ways, so `D` and `Q` count `Π₂^{±1}` calls. `H` is the overwrite chain on `Π₂` (`Pous.Sponge.ovKey`), labels are 64 KiB
(`ℓ = 524288`) with capacity 512, and there is no `pp` and there are no coins.

Scope: the ideal-permutation model only. Lean does not prove anything about a concrete permutation, decode cost, the
wall-clock-to-`Q` calibration, authentication, accounting or fresh setup. As for every `Meets` here, `W` is fixed
before `Π₂` is drawn, and the salt is a parameter every statement quantifies over. The band needs in-degree `d ≥ D`:
with `d + 1 ≤ D` it is broken (`BandGapAttack`, `lean/submissions/band-chain/BandChain.lean`). -/

/-- **Unconditional: the band scheme at 64 KiB.** For every in-degree `d ≥ 5` and salt, `chainBand (2^9) d` (512
blocks of 64 KiB labels) meets the requirement with sequential reveal, `D = 5` rounds, `Q = 2^20` queries per answer,
`k = 111` challenges and `ε = 2^-128`. Proof: `lean/submissions/sponge-dense/ChainEPFV6FinalBand64.lean`,
`PousBandChain.band_meets_64_final`; here `Pous.SecurityProofs.BandChainMeets64`. -/
def BandChainMeets64 : Prop :=
  ∀ d : ℕ, 5 ≤ d → ∀ salt : Bits 256,
    Meets (Pous.BandChain.chainBand (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) .sequential 5 (2 ^ 20) 111
      ((2⁻¹ : ℝ≥0∞) ^ 128)

/-- **Unconditional: the band at the deployment in-degree `d = 5`** (`band_meets_64_d5_final`; here `Pous.SecurityProofs.BandChainMeets64D5`). -/
def BandChainMeets64D5 : Prop :=
  ∀ salt : Bits 256,
    Meets (Pous.BandChain.chainBand (2 ^ 9) 5 524288 512 (Pous.Sponge.dom salt)) .sequential 5 (2 ^ 20) 111
      ((2⁻¹ : ℝ≥0∞) ^ 128)

/-- **Unconditional: the band at the recommended in-degree `d = 12`** (`band_meets_64_d12_final`; here `Pous.SecurityProofs.BandChainMeets64D12`). -/
def BandChainMeets64D12 : Prop :=
  ∀ salt : Bits 256,
    Meets (Pous.BandChain.chainBand (2 ^ 9) 12 524288 512 (Pous.Sponge.dom salt)) .sequential 5 (2 ^ 20) 111
      ((2⁻¹ : ℝ≥0∞) ^ 128)

/-- **Unconditional: the band at 14 GiB.** For every `d ≥ 5`, the 448-segment band `chainBandSeg 448 (2^9) d`
(229,376 blocks of 64 KiB) meets the requirement at `D = 5`, `Q = 2^20`, `k = 111`, `ε = 2^-128`, both with the
implementation's per-segment tags (`segDom salt`, segment `s` keyed by `salt ‖ u64(s)`, any 192-bit salt) and with one
global salt and global indices (`dom salt`). One advice string covers the whole store. Proof:
`ChainEPFV6FinalBand14.lean`, `PousBandMulti.band_meets_14_final`; here `Pous.SecurityProofs.BandMultiMeets14`. -/
def BandMultiMeets14 : Prop :=
  ∀ d : ℕ, 5 ≤ d →
    (∀ salt : Bits 192,
      Meets (Pous.BandMulti.chainBandSeg 448 (2 ^ 9) d 524288 512 (Pous.BandMulti.segDom salt (2 ^ 9))) .sequential
        5 (2 ^ 20) 111 ((2⁻¹ : ℝ≥0∞) ^ 128)) ∧
    (∀ salt : Bits 256,
      Meets (Pous.BandMulti.chainBandSeg 448 (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) .sequential 5 (2 ^ 20) 111
        ((2⁻¹ : ℝ≥0∞) ^ 128))

/-- **Unconditional: the band at 14 GiB with 106 challenges**, per-segment tags (`band_meets_14_k106_final`; here `Pous.SecurityProofs.BandMultiMeets14K106`). -/
def BandMultiMeets14K106 : Prop :=
  ∀ d : ℕ, 5 ≤ d → ∀ salt : Bits 192,
    Meets (Pous.BandMulti.chainBandSeg 448 (2 ^ 9) d 524288 512 (Pous.BandMulti.segDom salt (2 ^ 9))) .sequential 5
      (2 ^ 20) 106 ((2⁻¹ : ℝ≥0∞) ^ 128)

/-- **Unconditional: the band store at every segment count from 1 to 2^64.** For every `N` with `1 ≤ N ≤ 2^64` and
every `d ≥ 5`, the `N`-segment band `chainBandSeg N (2^9) d` (`N·512` blocks of 64 KiB, `N·32` MiB) meets the
requirement at `D = 5`, `Q = 2^20`, `k = 111`, `ε = 2^-128`, both with the implementation's per-segment tags
(`segDom salt`, segment `s` keyed by `salt ‖ u64(s)`, any 192-bit salt) and with one global salt and global indices
(`dom salt`). One advice string covers the whole store. `2^64` is where `u64(s)` stops being injective, and the domains
must be injective across segments: with a shared domain every `N ≥ 2` breaks (the lane's `shared_domain_breaks_fam`).
At `N = 448` this is `BandMultiMeets14`, and at `N = 1` its global layout is `BandChainMeets64`. Added on 29 Sep 2026.
Red team: `bc-22298e90`, GO (the POUS store's `docs/lean-trusted-layer-review.md`
§58), and `bc-9e381a9e`, for the band lane, GO (the POUS store's
`internal/p3-instantiation/band-family-pin-redteam-verdict.md`). Proof: the POUS store's
`lean/submissions/band-multi/BandMultiFamilyFinal.lean`, `PousBandMulti.band_meets_family_final` (graded:
`band-multi/family-pin/GRADE.txt`); here `Pous.SecurityProofs.BandMultiMeetsFamily`. -/
def BandMultiMeetsFamily : Prop :=
  ∀ (N : ℕ) [NeZero N], N ≤ 2 ^ 64 → ∀ d : ℕ, 5 ≤ d →
    (∀ salt : Bits 192,
      Meets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.BandMulti.segDom salt (2 ^ 9))) .sequential
        5 (2 ^ 20) 111 ((2⁻¹ : ℝ≥0∞) ^ 128)) ∧
    (∀ salt : Bits 256,
      Meets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) .sequential 5 (2 ^ 20) 111
        ((2⁻¹ : ℝ≥0∞) ^ 128))

/-- **Unconditional: the overwrite-chain dense scheme at 64 KiB.** `chainDense (2^9)` (512 blocks of 64 KiB on the
complete DAG) meets the requirement at `D = 5`, `Q = 2^20`, `k = 111`, `ε = 2^-128`. Not `DenseMeets` (the dense
scheme with `H` a random oracle), and not `SpongeDense`'s `ChainDenseMeetsPinned` (`2^16` blocks of 1 KiB, `D = 64`).
Proof: `ChainEPFV6FinalDense.lean`, `PousSpongeChain.chain_meets_64_final`; here
`Pous.SecurityProofs.ChainDenseMeets64`. -/
def ChainDenseMeets64 : Prop :=
  ∀ salt : Bits 256,
    Meets (Pous.Sponge.chainDense (2 ^ 9) 524288 512 (Pous.Sponge.dom salt)) .sequential 5 (2 ^ 20) 111
      ((2⁻¹ : ℝ≥0∞) ^ 128)

/-! ### The band at `D = 2`, `D = 1` and `D = 0` rounds

Added on 3 Oct 2026, approved by Daniel (3 Oct 2026, 05:27Z; `D = 0` approved 3 Oct 2026). Each
statement is a band pin above with only the round budget `D`, the challenge count `k` and the
bound on the in-degree `d` changed; the model, the scope and `d ≥ D` (`BandGapAttack`) are as there (`D = 0` puts no
bound on `d`, since the responder makes no `Π₂` call). The proofs are
`verity/Security/Proofs/Pous/Chain/FamilyRounds.lean`'s (`Pous.BandMulti.Rounds`): `Family.lean`'s certificate with `D`
and `k` as parameters. -/

/-- **Unconditional: the band store at `D = 2` rounds and `k = 97`, at every segment count from 1 to 2^64.**
`BandMultiMeetsFamily` at `D = 2` and `k = 97`, for every `d ≥ 2`: for every `N` with `1 ≤ N ≤ 2^64` and every
`d ≥ 2`, the `N`-segment band `chainBandSeg N (2^9) d` meets the requirement at `D = 2`, `Q = 2^20`, `k = 97`,
`ε = 2^-128`, both with the implementation's per-segment tags (`segDom salt`, any 192-bit salt) and with one global
salt and global indices (`dom salt`). Proof: `Pous.BandMulti.Rounds.band_meets_family_d2`; here
`Pous.SecurityProofs.BandMultiMeetsFamilyD2`. -/
def BandMultiMeetsFamilyD2 : Prop :=
  ∀ (N : ℕ) [NeZero N], N ≤ 2 ^ 64 → ∀ d : ℕ, 2 ≤ d →
    (∀ salt : Bits 192,
      Meets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.BandMulti.segDom salt (2 ^ 9))) .sequential
        2 (2 ^ 20) 97 ((2⁻¹ : ℝ≥0∞) ^ 128)) ∧
    (∀ salt : Bits 256,
      Meets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) .sequential 2 (2 ^ 20) 97
        ((2⁻¹ : ℝ≥0∞) ^ 128))

/-- **Unconditional: the band store at `D = 1` round and `k = 93`, at every segment count from 1 to 2^64.**
`BandMultiMeetsFamily` at `D = 1` and `k = 93`, for every `d ≥ 1`, both domain layouts. Proof:
`Pous.BandMulti.Rounds.band_meets_family_d1`; here `Pous.SecurityProofs.BandMultiMeetsFamilyD1`. -/
def BandMultiMeetsFamilyD1 : Prop :=
  ∀ (N : ℕ) [NeZero N], N ≤ 2 ^ 64 → ∀ d : ℕ, 1 ≤ d →
    (∀ salt : Bits 192,
      Meets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.BandMulti.segDom salt (2 ^ 9))) .sequential
        1 (2 ^ 20) 93 ((2⁻¹ : ℝ≥0∞) ^ 128)) ∧
    (∀ salt : Bits 256,
      Meets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) .sequential 1 (2 ^ 20) 93
        ((2⁻¹ : ℝ≥0∞) ^ 128))

/-- **Unconditional: the band store at `D = 0` rounds and `k = 89`, at every segment count from 1 to 2^64.**
`BandMultiMeetsFamily` at `D = 0` and `k = 89`, for every `d` with no bound on the in-degree: at `D = 0` the responder
makes no `Π₂` call and answers from its state alone, so the band needs no bound on `d`. Both domain layouts. Approved
by Daniel (3 Oct 2026). Proof: `Pous.BandMulti.Rounds.band_meets_family_d0`; here
`Pous.SecurityProofs.BandMultiMeetsFamilyD0`. -/
def BandMultiMeetsFamilyD0 : Prop :=
  ∀ (N : ℕ) [NeZero N], N ≤ 2 ^ 64 → ∀ d : ℕ,
    (∀ salt : Bits 192,
      Meets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.BandMulti.segDom salt (2 ^ 9))) .sequential
        0 (2 ^ 20) 89 ((2⁻¹ : ℝ≥0∞) ^ 128)) ∧
    (∀ salt : Bits 256,
      Meets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) .sequential 0 (2 ^ 20) 89
        ((2⁻¹ : ℝ≥0∞) ^ 128))

/-- **Unconditional: the band scheme at 64 KiB, `D = 2` rounds and `k = 97`.** `BandChainMeets64` at `D = 2` and
`k = 97`, for every `d ≥ 2` and salt (the suffix counts rounds, where `BandChainMeets64D5`'s names the in-degree).
Proof: `Pous.BandMulti.Rounds.band_meets_64_d2`, `BandMultiMeetsFamilyD2`'s global layout at `N = 1`; here
`Pous.SecurityProofs.BandChainMeets64Rounds2`. -/
def BandChainMeets64Rounds2 : Prop :=
  ∀ d : ℕ, 2 ≤ d → ∀ salt : Bits 256,
    Meets (Pous.BandChain.chainBand (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) .sequential 2 (2 ^ 20) 97
      ((2⁻¹ : ℝ≥0∞) ^ 128)

/-- **Unconditional: the band scheme at 64 KiB, `D = 1` round and `k = 93`.** `BandChainMeets64` at `D = 1` and
`k = 93`, for every `d ≥ 1` and salt. Proof: `Pous.BandMulti.Rounds.band_meets_64_d1`; here
`Pous.SecurityProofs.BandChainMeets64Rounds1`. -/
def BandChainMeets64Rounds1 : Prop :=
  ∀ d : ℕ, 1 ≤ d → ∀ salt : Bits 256,
    Meets (Pous.BandChain.chainBand (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) .sequential 1 (2 ^ 20) 93
      ((2⁻¹ : ℝ≥0∞) ^ 128)

/-! ### The band with a public encoder (`Definitions.Pous.Model.PubEncoder`, `Definitions.Pous.Accounting.PubMeets`)

Added on 29 Sep 2026, with the band pins. Red team: `bc-22298e90`, GO
(`docs/lean-trusted-layer-review.md` §§53, 55). The codeword is
produced by an untrusted server and accepted by the exact check. Each holds under the four deployment conditions of the
review's §48, which are in its docstring. The proofs are in this package (`Pous.SecurityProofs.BandPub*`). -/

/-- **Unconditional: the band at 64 KiB with a public encoder.** For every `d ≥ 5` and salt, `PubMeets` for
`chainBand (2^9) d` with the exact check, at `D = 5`, `Q = 2^20`, `k = 111`, `ε = 2^-128`: an untrusted server that
encodes the data is exactly as secure as verifier-side encoding (`PubMeets ↔ Meets`: `band_pub_iff_64`,
`lean/submissions/public-encoder/PubEncoderBand.lean`). Deployment conditions (§48): (1) the check is the exact,
full-decode check (a sampled or spot check is not covered); (2) the salt is fixed by the verifier or the protocol, not
the server; (3) `W` is fixed before setup, and `Π₂` is independent of `W` and the salt; (4) the commitment to `W` and
the binding of `C*` are idealized. Proof: `ChainEPFV6FinalPub.lean`, `PousPub.band_pub_meets_64_final`; here
`Pous.SecurityProofs.BandPubMeets64`. -/
def BandPubMeets64 : Prop :=
  ∀ d : ℕ, 5 ≤ d → ∀ salt : Bits 256,
    PubMeets (Pous.BandChain.chainBand (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) (exactCheck _) .sequential 5
      (2 ^ 20) 111 ((2⁻¹ : ℝ≥0∞) ^ 128)

/-- **Unconditional: the band at 14 GiB with a public encoder, per-segment tags** (`PubMeets ↔ Meets`:
`band_pub_iff_14`), under the same four deployment conditions (§48). Proof: `PousPub.band_pub_meets_14_final`; here
`Pous.SecurityProofs.BandPubMeets14`. -/
def BandPubMeets14 : Prop :=
  ∀ d : ℕ, 5 ≤ d → ∀ salt : Bits 192,
    PubMeets (Pous.BandMulti.chainBandSeg 448 (2 ^ 9) d 524288 512 (Pous.BandMulti.segDom salt (2 ^ 9)))
      (exactCheck _) .sequential 5 (2 ^ 20) 111 ((2⁻¹ : ℝ≥0∞) ^ 128)

/-- **Unconditional: the band at 14 GiB with a public encoder, one global salt** (`PubMeets ↔ Meets`:
`band_pub_iff_14_global`), under the same four deployment conditions (§48). Proof:
`PousPub.band_pub_meets_14_global_final`; here `Pous.SecurityProofs.BandPubMeets14Global`. -/
def BandPubMeets14Global : Prop :=
  ∀ d : ℕ, 5 ≤ d → ∀ salt : Bits 256,
    PubMeets (Pous.BandMulti.chainBandSeg 448 (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) (exactCheck _) .sequential
      5 (2 ^ 20) 111 ((2⁻¹ : ℝ≥0∞) ^ 128)

/-- **Unconditional: the band store with a public encoder at every segment count from 1 to 2^64.** For every `N` with
`1 ≤ N ≤ 2^64` and every `d ≥ 5`, `PubMeets` for the `N`-segment band `chainBandSeg N (2^9) d` with the exact check, at
`D = 5`, `Q = 2^20`, `k = 111`, `ε = 2^-128`, both with the implementation's per-segment tags (`segDom salt`, segment
`s` keyed by `salt ‖ u64(s)`, any 192-bit salt) and with one global salt and global indices (`dom salt`). It is
`BandMultiMeetsFamily` with an untrusted server encoding the data (`PubMeets ↔ Meets` at every `N`: `band_pub_iff_seg`,
`band_pub_iff_seg_global`), under the same four deployment conditions (§48). At `N = 448` it is `BandPubMeets14` and
`BandPubMeets14Global`. Added on 3 Oct 2026. Proof:
`PousPub.band_pub_meets_family_final` (`verity/Security/Proofs/Pous/Chain/PubFamily.lean`); here
`Pous.SecurityProofs.BandPubMeetsFamily`. -/
def BandPubMeetsFamily : Prop :=
  ∀ (N : ℕ) [NeZero N], N ≤ 2 ^ 64 → ∀ d : ℕ, 5 ≤ d →
    (∀ salt : Bits 192,
      PubMeets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.BandMulti.segDom salt (2 ^ 9)))
        (exactCheck _) .sequential 5 (2 ^ 20) 111 ((2⁻¹ : ℝ≥0∞) ^ 128)) ∧
    (∀ salt : Bits 256,
      PubMeets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) (exactCheck _)
        .sequential 5 (2 ^ 20) 111 ((2⁻¹ : ℝ≥0∞) ^ 128))

/-- **Unconditional: the band store with a public encoder at `D = 2` rounds and `k = 97`, at every segment count from
1 to 2^64.** `BandPubMeetsFamily` at `D = 2` and `k = 97`, for every `d ≥ 2`: it is `BandMultiMeetsFamilyD2` with an
untrusted server encoding the data and the exact check (`PubMeets ↔ Meets` at every `N`), both domain layouts, under
the same four deployment conditions (§48). Added on 3 Oct 2026. Proof:
`PousPub.band_pub_meets_family_d2` (`verity/Security/Proofs/Pous/Chain/PubRounds.lean`); here
`Pous.SecurityProofs.BandPubMeetsFamilyD2`. -/
def BandPubMeetsFamilyD2 : Prop :=
  ∀ (N : ℕ) [NeZero N], N ≤ 2 ^ 64 → ∀ d : ℕ, 2 ≤ d →
    (∀ salt : Bits 192,
      PubMeets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.BandMulti.segDom salt (2 ^ 9)))
        (exactCheck _) .sequential 2 (2 ^ 20) 97 ((2⁻¹ : ℝ≥0∞) ^ 128)) ∧
    (∀ salt : Bits 256,
      PubMeets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) (exactCheck _)
        .sequential 2 (2 ^ 20) 97 ((2⁻¹ : ℝ≥0∞) ^ 128))

/-- **Unconditional: the band store with a public encoder at `D = 1` round and `k = 93`, at every segment count from
1 to 2^64.** `BandPubMeetsFamily` at `D = 1` and `k = 93`, for every `d ≥ 1`: `BandMultiMeetsFamilyD1` with a public
encoder and the exact check, both domain layouts, under the same four deployment conditions (§48). Added on
3 Oct 2026. Proof: `PousPub.band_pub_meets_family_d1`; here
`Pous.SecurityProofs.BandPubMeetsFamilyD1`. -/
def BandPubMeetsFamilyD1 : Prop :=
  ∀ (N : ℕ) [NeZero N], N ≤ 2 ^ 64 → ∀ d : ℕ, 1 ≤ d →
    (∀ salt : Bits 192,
      PubMeets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.BandMulti.segDom salt (2 ^ 9)))
        (exactCheck _) .sequential 1 (2 ^ 20) 93 ((2⁻¹ : ℝ≥0∞) ^ 128)) ∧
    (∀ salt : Bits 256,
      PubMeets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) (exactCheck _)
        .sequential 1 (2 ^ 20) 93 ((2⁻¹ : ℝ≥0∞) ^ 128))

/-! ### The dense deployment over the overwrite chain

Re-pinned on 29 Sep 2026 from `k = 107` to `k = 111`: the dense chain scheme at the deployment point was pinned in the
lane at `k = 107` (`SpongeDense`'s `ChainDenseMeetsPinned`), which the ex post facto argument does not reach, and is
pinned here at `k = 111`, which is proved. Red team: `bc-22298e90`, GO
(`docs/lean-trusted-layer-review.md` §§53–54). -/

/-- **Unconditional: the dense chain scheme at the deployment point, `k = 111`.** For every salt,
`chainDense (2^16) 8192 512 (dom salt)` (`2^16` blocks of 1 KiB labels on the complete DAG, `H` the overwrite chain on
one ideal permutation) meets the requirement with sequential reveal, `D = 64` rounds, `Q = 2^20` queries per answer,
`k = 111` challenges and `ε = 2^-128`. It is strictly weaker than the `k = 107` target (`ChainDenseMeetsPinned`), which
stays open: it costs four more challenges. Not `DenseMeets`, which is the dense scheme with `H` a random oracle at
`k = 107`. Proof: `lean/submissions/sponge-dense/ChainEPFV6FinalDense111.lean`,
`PousSpongeChain.chainDenseMeets111_final`; here `Pous.SecurityProofs.ChainDenseMeets111`. -/
def ChainDenseMeets111 : Prop :=
  ∀ salt : Bits 256,
    Meets (Pous.Sponge.chainDense (2 ^ 16) 8192 512 (Pous.Sponge.dom salt)) .sequential 64 (2 ^ 20) 111
      ((2⁻¹ : ℝ≥0∞) ^ 128)

/-! ### The band with grace `t` (`Definitions.Pous.Model.Graded`, `Definitions.Pous.Accounting.GradedMeets`)

Added on 3 Oct 2026 (PR #905). The game's requirements on the protocol are in
`verity/Security/Definitions/Pous/Model/Graded.lean` and `PROTOCOL.md`, "The graded audit". -/

/-- **Unconditional: the band store with up to `t` late answers, at every segment count from 1 to 2^64.** For every `N`
with `1 ≤ N ≤ 2^64`, every `d ≥ 5`, every hard bound `(D_late, Q_late)` with `5 ≤ D_late ≤ d`, `D_late ≤ 12` and
`2^20 ≤ Q_late`, and each `(t, k)` of `(0, 111)`, `(1, 160)`, `(2, 203)`, `(3, 243)`, `GradedMeets` for the `N`-segment
band `chainBandSeg N (2^9) d` at `D = 5`, `Q = 2^20`, `ε = 2^-128`, both with the implementation's per-segment tags
(`segDom salt`, any 192-bit salt) and with one global salt and global indices (`dom salt`). Each `k` is the least with
`P[Bin(k, 95923/100000) ≥ k − t] ≤ 1%`; at `t = 0` and `k = 111` the row is `BandMultiMeetsFamily`'s
(`PousGraded.gradedMeets_zero_iff`).

The hypotheses on `D_late` are the deployment condition, necessary for the game's `S`-bit state to be a model of
storage at rest: the audit's window must be too short to re-encode `C`. Re-encoding a segment from `W` takes
`R(d) = Σ_{v < 2^9} (min(v, d) + 1) ≥ R(5) = 3,057` sequential `Π₂` calls. Every row's rounds, `k − t` at `D = 5` and
`t` at `D_late`, last under `240·6 + 3·13 = 1,479` call-times, and the memory challenger's gaps between them add to
the window, so it is bounded by the memory challenger's limit on the audit's duration that `BackgroundRebuildExcluded`
requires, `1,505` call-times. `D_late ≤ d` is the band's cliff applied to `Δ_late`; at the deployed `d = 12` both caps
on `D_late` are `Δ_late + RTT < 13 × 241.5 µs ≈ 3.14 ms`. That window argument and its wall-clock reading count whole
`Π₂` calls, so they hold in the ideal-permutation model only: at the reference Feistel-SHAKE256 `Π₂` a prover re-encodes
nothing, rebuilding any label from 1.53% of `C` in 481 Keccak-f, less than one call (red team, 4 Oct 2026,
art:7617c8017e80). The statement itself is about the ideal model and is unaffected. Proof:
`PousGraded.band_graded_meets_family` (`verity/Security/Proofs/Pous/Graded/Band.lean`); here
`Pous.SecurityProofs.BandGradedMeetsFamily`. -/
def BandGradedMeetsFamily : Prop :=
  ∀ (N : ℕ) [NeZero N], N ≤ 2 ^ 64 → ∀ d : ℕ, 5 ≤ d →
    ∀ D_late Q_late : ℕ, 5 ≤ D_late → D_late ≤ d → D_late ≤ 12 → 2 ^ 20 ≤ Q_late →
    ∀ tk ∈ ([(0, 111), (1, 160), (2, 203), (3, 243)] : List (ℕ × ℕ)),
      (∀ salt : Bits 192,
        GradedMeets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.BandMulti.segDom salt (2 ^ 9))) 5
          (2 ^ 20) D_late Q_late tk.2 tk.1 ((2⁻¹ : ℝ≥0∞) ^ 128)) ∧
      (∀ salt : Bits 256,
        GradedMeets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) 5 (2 ^ 20) D_late
          Q_late tk.2 tk.1 ((2⁻¹ : ℝ≥0∞) ^ 128))

/-! ### The continuous audit (`Definitions.Pous.Model.Continuous`)

Added on 3 Oct 2026. Red team: needed. The memory challenger enforces the three timing
rules: it reveals the next index only after the last round closed, rejects an answer past the hard deadline, and counts
every answer past `Δ` as a miss, with every rejected or wrong one. Neither statement takes a hypothesis about them, and
both hold at every hard deadline `(D_late, Q_late)` and every arrival schedule `arr`
(`verity/Security/Definitions/Pous/Model/Continuous.lean`).
-/

/-- **The per-answer bound in the continuous audit.** From `TimedINC sch D Q β π`, for every machine, every arrival
schedule, every hard deadline `(D_late, Q_late)` and every history `h` (the indices sent before): off a setup event of
probability at most `(β + 1)·π`, the round after `h` is a hit with probability at most `1 − ε + ε_proof` over its
uniform index, where `ε` is the freed fraction of the responder's memory at that round's send time. The `β + 1` is a
union over the memory's length. Proof: `PousContinuous.per_answer`; here `Pous.SecurityProofs.PerAnswer`. -/
def PerAnswer : Prop :=
  ∀ {M : IdealModel} (sch : Scheme M) (D Q D_late Q_late β : ℕ) (π : ℝ≥0∞), TimedINC sch D Q β π →
    ∀ (W : Bits sch.n) (mach : ContMachine sch) (arr : ℕ → ℕ) (h : List (Fin sch.B)),
      pr (fun a : M.Ω × sch.Coins =>
          let pp := (sch.enc W a.1 a.2).1
          let C := (sch.enc W a.1 a.2).2
          let s := contAfter mach D_late W pp C a.1 arr h
          ¬ pr (fun i => (contRecord mach D Q D_late Q_late W pp C a.1 arr s i).verdict = .hit) ≤
            ENNReal.ofReal (1 - contFreed sch (contMem mach W pp C a.1 arr s).length + epsProof sch β))
        ≤ (β + 1) * π

/-- **The continuous audit's guarantee** (the explorer note, §1). From `TimedINC sch D Q β π`, for every machine,
every arrival schedule, every hard deadline `(D_late, Q_late)`, every `f` and every `βe > 0` (the explorer's `β`), over
the setup and `n` uniform indices: the memory challenger accepts with tolerance `f` and `Σ_j max(ε_j − ε_proof, 0)`
exceeds `G(f) = inf_{θ > 0} (θ·f + ln(1/βe)) / (1 − e^−θ)` with probability at most `βe + n·(β + 1)·π`. Here `ε_j` is
the freed fraction of the responder's memory at challenge `j`'s send time. The positive part keeps a memory above `|C|`
at one challenge (`ε_j < 0`) from offsetting a shortfall at another. Dividing by `n`:
`Pr[accept(f₀) ∧ ε̄ > g(f₀)] ≤ βe + n·(β + 1)·π`, with `ε̄ = (1/n)·Σ_j max(ε_j − ε_proof, 0)` and `g = G/n`. When
`ε_proof ≥ 0` (whenever the bound is not vacuous), `max(ε_j − ε_proof, 0) ≥ max(ε_j, 0) − ε_proof`, so the same bound
holds for the average of `max(ε_j, 0)` exceeding `ε_proof + g(f₀)`. The scope is the module docstring's: conditional
on the arrivals, storage at send times, and `g` normalized by the realized number of challenges, never by `λT`. Proof:
`PousContinuous.continuous_audit`; here `Pous.SecurityProofs.ContinuousAudit`. -/
def ContinuousAudit : Prop :=
  ∀ {M : IdealModel} (sch : Scheme M) (D Q D_late Q_late β n : ℕ) (π : ℝ≥0∞), TimedINC sch D Q β π →
    ∀ (W : Bits sch.n) (mach : ContMachine sch) (arr : ℕ → ℕ) (f βe : ℝ), 0 < βe →
      pr (fun x : Outcome sch n =>
          let pp := (sch.enc W x.1 x.2.1).1
          let C := (sch.enc W x.1 x.2.1).2
          let rs := contRecords mach D Q D_late Q_late W pp C x.1 arr .start (List.ofFn x.2.2)
          contAccepts f rs ∧
            (⨅ θ : Set.Ioi (0 : ℝ), ((θ : ℝ) * f + Real.log (1 / βe)) / (1 - Real.exp (-(θ : ℝ)))) <
              (rs.map fun r => max (contFreed sch r.held - epsProof sch β) 0).sum)
        ≤ ENNReal.ofReal βe + n * (β + 1) * π

/-! ### The background-rebuild exclusion (`Definitions.Pous.Accounting.Rebuild`)

Added on 3 Oct 2026. Red team: GO at `f6aaddc36` (4 Oct 2026). -/

/-- **Under the memory challenger's cap on an audit's duration: no graded audit lasts long enough to re-encode a whole
band segment from `W` alone**, with no state about `Π₂`. It is the deployment condition of `BandGradedMeetsFamily`
(`PROTOCOL.md`, "Isolation over the whole graded window"), as counts and as a bound.
1. **The counts.** Re-encoding a segment of `chainBand (2^9) d` from `W` costs `R(d) = rebuildCalls (2^9) d` sequential
   `Π₂` calls: `R(5) = 3,057`, `R(12) = 6,578`, and `R(d) ≥ 3,057` for every `d ≥ 5`. The rounds of each row `(t, k)`
   of `BandGradedMeetsFamily` last under `gradedRoundSpan 5 D_late k t ≤ 240·6 + 3·13 = 1,479` call-times when
   `D_late ≤ 12`: a round answered at its deadline lasts just under `D + 1` call-times, not `D`.
2. **The bound.** For every `d ≥ 5`, every domain map injective on the segment's `2^9` indices, every `W`, `Q ≤ 2^128`
   and every `T ≤ auditCap = 1,505`, programs that know `W` and the domains but nothing about `Π₂` beyond their own
   queries, each within `T` rounds and `Q` queries, output all `2^9` labels of `chainBand (2^9) d 524288 512 dom` with
   probability at most `2^-128` over `Π₂`. Segment `s` of the deployed store is such a segment, with domains `dom tag_s`
   under the per-segment tags and `j ↦ dom salt (s·2^9 + j)` under one global salt (`Pous.BandMulti.segFaithfulCode`,
   `Pous.BandMulti.segFaithful`).

**The precondition is the memory challenger's.** A rebuild runs beside the audit on the wall clock, through the rounds
and the memory challenger's gaps between them, and no round deadline bounds the gaps. So the bound covers a graded
audit only if the memory challenger aborts, without accepting, every audit that lasts more than `auditCap · call_ns` (≈
363.5 ms at 241.5 µs per call) from its first challenge's send to its last answer's receipt. If every round of the worst
row answers at its deadline, that leaves its 242 gaps `1,505 − 1,479 = 26` call-times (≈ 6.3 ms). The bound rests on the
cap, which is the proof's reach, not on `R(d)`, which exceeds it. The milliseconds read call-times as whole `Π₂` calls
of at least 241.5 µs, which holds in the ideal-permutation model only: at the reference Feistel-SHAKE256 `Π₂` a prover
answers from 1.53% of `C` in 481 Keccak-f without re-encoding anything (red team, 4 Oct 2026, art:7617c8017e80).

**What it does not exclude.** It excludes only re-encoding a whole segment from `W` with no state about `Π₂`. Within the
cap, `W` alone rebuilds a segment's first 253 labels (49%) at `d = 5` and its first 121 (24%) at `d = 12`. `d`
consecutive held labels restart the chain, so two held runs (10 labels, 2.0% of the segment) at `d = 5`, or four
(48 labels, 9.4%) at `d = 12`, let a prover re-encode a whole segment within the cap. That gap between the storage held
at rest and what is present when a challenge arrives is not bounded here. Proof: `PousRebuild.background_rebuild_excluded`
(`verity/Security/Proofs/Pous/Rebuild/`); here `Pous.SecurityProofs.BackgroundRebuildExcluded`. -/
def BackgroundRebuildExcluded : Prop :=
  Pous.Rebuild.rebuildCalls (2 ^ 9) 5 = 3057 ∧ Pous.Rebuild.rebuildCalls (2 ^ 9) 12 = 6578 ∧
  (∀ d : ℕ, 5 ≤ d → 3057 ≤ Pous.Rebuild.rebuildCalls (2 ^ 9) d) ∧
  (∀ D_late : ℕ, D_late ≤ 12 → ∀ tk ∈ ([(0, 111), (1, 160), (2, 203), (3, 243)] : List (ℕ × ℕ)),
    Pous.Rebuild.gradedRoundSpan 5 D_late tk.2 tk.1 ≤ 1479) ∧
  ∀ d : ℕ, 5 ≤ d → ∀ dom : ℕ → Bits 512, Function.Injective (fun i : Fin (2 ^ 9) => dom i) →
    ∀ (W : Bits (2 ^ 9 * 524288)) (Q : ℕ), Q ≤ 2 ^ 128 → ∀ T : ℕ, T ≤ Pous.Rebuild.auditCap →
    ∀ A : Fin (2 ^ 9) → Prog (Pous.Sponge.idealPerm (524288 + (524288 + 512))) (Bits 524288),
      (∀ i, (A i).Bounded T Q) →
      pr (fun π : Equiv.Perm (Bits (524288 + (524288 + 512))) =>
          ∀ i, (A i).run ((Pous.Sponge.idealPerm (524288 + (524288 + 512))).answer π) =
            ((Pous.BandChain.chainBand (2 ^ 9) d 524288 512 dom).enc W π ()).2 i)
        ≤ (2⁻¹ : ℝ≥0∞) ^ 128

/-! ### Completeness of the continuous audit

Added on 3 Oct 2026. Red team: needed. `ContinuousAudit` bounds what the memory challenger
of `verity/Security/Definitions/Pous/Model/Continuous.lean` accepts; `ContinuousComplete` bounds what the same memory
challenger rejects, for a responder whose rounds miss at a bounded rate. It takes no named assumption: the miss rate `q`
is a parameter.
-/

/-- **Completeness of the continuous audit.** Fix a configuration first: the scheme, the data `W`, the machine, the
number of challenges `n`, the arrival schedule `arr`, the deadline `(D, Q)` and the hard deadline `(D_late, Q_late)`.
Suppose that in this configuration the machine's rounds miss at rate at most `q`: at every setup and after every
history `h` of fewer than `n` indices, the round after `h` is not a hit (it is rejected, late or wrong) for at most a
`q` fraction of its uniform index. Then for every tolerance `f₀`, over the setup and `n` uniform indices, the memory
challenger rejects (more than `f₀` rounds are not hits) with probability at most the binomial upper tail
`α = P[Bin(n, q) > f₀] = Σ_{f₀ < m ≤ n} C(n, m)·q^m·(1 − q)^(n − m)`.

`q` is a parameter, not an assumption, and the statement holds for every machine, but a `q` is a rate in one
configuration. A tighter deadline needs a larger `q`: at a fixed hard deadline the rounds and the histories do not
depend on `(D, Q)`, and a smaller `(D, Q)` turns some hits into late answers. A machine is deterministic, so the only
randomness the theorem bounds is the setup and the index draw, and `q` is the worst miss rate over the next index, at
the worst setup and after the worst history. The model's honest responder, which keeps `C` and answers by lookup, hits
every round: `q = 0` (`PousContinuous.honest_complete`).

A real responder also misses on its host's clock (scheduling, pauses, the network), which the model leaves out. Those
misses do not depend on the indices, so conditioning on the challenges before only gives back their run-average rate,
which is not a valid `q`. A valid `q` is the worst per-round miss probability over the histories the clock can
produce, such as the rate right after a miss. Bursty or correlated misses set the two far apart: a host that misses 1%
of its rounds in bursts of mean length 10 (a miss follows a miss with probability `0.9`) has run-average rate `0.01`,
but at `n = 1000` and `f₀ = 25` it is rejected with probability `0.123`, against `α = 1.56·10⁻⁵` at `q = 0.01`; its
valid `q` is about `0.9`. Carrying the tail bound from the index draw to the clock's miss process is the same tail
rule applied outside the Lean model, and is not proved here.

`1 − q` is truncated at `0`, so above `q = 1` the bound is `q^n ≥ 1` when `f₀ < n`, and `0` otherwise, where the
memory challenger cannot reject. Proof: `PousContinuous.continuous_complete`; here
`Pous.SecurityProofs.ContinuousComplete`. -/
def ContinuousComplete : Prop :=
  ∀ {M : IdealModel} (sch : Scheme M) (D Q D_late Q_late n : ℕ) (q : ℝ≥0∞) (W : Bits sch.n) (mach : ContMachine sch)
    (arr : ℕ → ℕ),
    (∀ (a : M.Ω × sch.Coins) (h : List (Fin sch.B)), h.length < n →
      let pp := (sch.enc W a.1 a.2).1
      let C := (sch.enc W a.1 a.2).2
      pr (fun i =>
          (contRecord mach D Q D_late Q_late W pp C a.1 arr (contAfter mach D_late W pp C a.1 arr h) i).verdict ≠ .hit)
        ≤ q) →
    ∀ f₀ : ℝ,
      pr (fun x : Outcome sch n =>
          let pp := (sch.enc W x.1 x.2.1).1
          let C := (sch.enc W x.1 x.2.1).2
          ¬ contAccepts f₀ (contRecords mach D Q D_late Q_late W pp C x.1 arr .start (List.ofFn x.2.2)))
        ≤ ∑ m ∈ (Finset.range (n + 1)).filter (fun m : ℕ => f₀ < m),
            (n.choose m : ℝ≥0∞) * q ^ m * (1 - q) ^ (n - m)

/-! ### The continuous audit, time-averaged, with arrivals drawn by the memory challenger

Added on 3 Oct 2026. Red team: needed. The memory challenger of `ContinuousAudit`,
unchanged, run on arrivals it draws itself: one coin per slot of one unit
(`verity/Security/Definitions/Pous/Model/Continuous.lean`, "Arrivals drawn by the memory challenger"). Arrivals queue
behind an open round (rule 1). The memory is read at every idle slot, not only at send times, and the bound is
normalized by the expected number of challenges `λT`, not the realized one. -/

/-- **The continuous audit's time-averaged bound.** From `TimedINC sch D Q β π`, for every machine, every hard
deadline `(D_late, Q_late)`, every `f` and every `βe > 0`, over the setup and the memory challenger's `T` slot draws
`x t ∈ Fin b × Fin B` (a challenge arrives at slot `t` iff its coin is below `a`, probability `p = a/b`; its index is
the slot's): the memory challenger, run on these arrivals, accepts with tolerance `f` and
`p·Σ_{t idle} max(ε_t − ε_proof, 0)` exceeds `G(f) = inf_{θ > 0} (θ·f + ln(1/βe)) / (1 − e^−θ)` with probability at most
`βe + T·(β + 1)·π`. Here `ε_t` is the freed fraction of the responder's memory at slot `t` (`contHeldAt`), read at every
idle slot (`contIdle`: every earlier arrival has closed) whether or not a challenge arrives there, and a busy slot
counts as `0`. Dividing by `pT`, the expected number of challenges (`λ·T` in seconds):
`Pr[accept(f₀) ∧ (1/T)·Σ_{t idle} max(ε_t − ε_proof, 0) > g(f₀)] ≤ βe + T·(β + 1)·π` with `g = G/(pT)`, the explorer's
`g` at `n = λT`. In an accepted run off that event, the idle slots with `ε_t > X` are at most a `g/(X − ε_proof)`
fraction of the `T` slots (Markov).

This is an average over the idle slots, not over the whole period: it does not bound the explorer's
`ε̄ = (1/T)·Σ_t ε_t`. The memory is not read at a busy slot, and the responder chooses which slots are busy
(`verity/Security/Definitions/Pous/Model/Continuous.lean`, "The responder chooses the busy slots"). In an accepted run
off the event, with `0 < a` and `ε_proof ≥ 0`, `ε̄ ≤ ε_proof + g + (busy slots)/T` whatever the memory at a busy slot,
and a responder that pads every hit to `D` units makes about 2.5% of the slots busy at `D = 2` and 9.9% at `D = 5`
(100 challenges/s on 241.5 µs slots) without a miss. Neither is proved here. `a ≤ b` keeps `a/b` a probability,
and `0 < b` makes the draw space nonempty. The
scope is the model's: the arrival coins are the memory challenger's own randomness, independent of the setup and the
answers; memory is read once per slot, so a dip shorter than one slot is not seen; time is in units of one round, which
the step floor converts to seconds in the ideal-permutation model only (at the reference Feistel-SHAKE256 `Π₂` a prover
answers from 1.53% of `C` in 481 Keccak-f, under one call: red team, 4 Oct 2026, art:7617c8017e80). Proof:
`PousContinuous.continuous_time_average`; here `Pous.SecurityProofs.ContinuousTimeAverage`. -/
def ContinuousTimeAverage : Prop :=
  ∀ {M : IdealModel} (sch : Scheme M) (D Q D_late Q_late β T a b : ℕ) (π : ℝ≥0∞), 0 < b → a ≤ b →
    TimedINC sch D Q β π →
    ∀ (W : Bits sch.n) (mach : ContMachine sch) (f βe : ℝ), 0 < βe →
      pr (fun x : M.Ω × sch.Coins × (Fin T → Fin b × Fin sch.B) =>
          let pp := (sch.enc W x.1 x.2.1).1
          let C := (sch.enc W x.1 x.2.1).2
          let rs := contRecords mach D Q D_late Q_late W pp C x.1 (arrAt a x.2.2) .start (histAt a x.2.2 T)
          contAccepts f rs ∧
            (⨅ θ : Set.Ioi (0 : ℝ), ((θ : ℝ) * f + Real.log (1 / βe)) / (1 - Real.exp (-(θ : ℝ)))) <
              ((a : ℝ) / b) * ∑ t : Fin T,
                if contIdle mach D_late W pp C x.1 a x.2.2 t then
                  max (contFreed sch (contHeldAt mach D_late W pp C x.1 a x.2.2 t) - epsProof sch β) 0
                else 0)
        ≤ ENNReal.ofReal βe + T * (β + 1) * π

/-- **Unconditional: the continuous audit's time-averaged bound on the band store at `D = 0, 1, 2, 5` rounds, at
every segment count from 1 to 2^64.** `ContinuousTimeAverage` instantiated on the `N`-segment band
`chainBandSeg N (2^9) d 524288 512 dom`, for every `N` with `1 ≤ N ≤ 2^64`, both domain layouts (the
implementation's per-segment tags `segDom salt`, any 192-bit salt, and one global salt `dom salt`), every `d ≥ D`
(every `d` at `D = 0`), at `Q = 2^20` and each `(D, k)` of `(0, 89)`, `(1, 93)`, `(2, 97)`, `(5, 111)`. `β` is the
compression width of the sequential certificate at `(D, k)` (`BandMultiMeetsFamilyD0`, `D1`, `D2`,
`BandMultiMeetsFamily`): `524288·(511 − D) − 69·(512 − D) − 1 − (9k + 129)` at `N = 1` and
`523898·(512 − D)·N − 532588` at `N ≥ 2`. The tail is at most `2^-(9k + 99)` per slot: `2^-900`, `2^-936`, `2^-972`,
`2^-1098` (at `N ≥ 2` it is below `2^-8206`). So `ε_proof = 1 − (β − 512·N)/(N·2^28)` (`epsProof`) is

| `D` | `k` | `N = 1` | `N = 2` | `N = 448` | `N = 2240` |
|---|---|---|---|---|---|
| 0 | 89 | 0.2090% | 0.1738% | 0.0750% | 0.0747% |
| 1 | 93 | 0.4043% | 0.3689% | 0.2702% | 0.2698% |
| 2 | 97 | 0.5996% | 0.5641% | 0.4654% | 0.4650% |
| 5 | 111 | 1.1855% | 1.1496% | 1.0509% | 1.0505% |

These slacks are for the idle slots only (`ContinuousTimeAverage`). At `D = 2` and `D = 5` a responder that pads its
hits to `D` units makes about 2.5% and 9.9% of the slots busy without a miss (100 challenges/s), and the bound reads
nothing there; at `D_late ≤ 1` no slot is busy.

Proof: `PousContinuous.band_family_continuous_time_average`; here
`Pous.SecurityProofs.BandFamilyContinuousTimeAverage`. -/
def BandFamilyContinuousTimeAverage : Prop :=
  ∀ (N : ℕ) [NeZero N], N ≤ 2 ^ 64 → ∀ Dk ∈ ([(0, 89), (1, 93), (2, 97), (5, 111)] : List (ℕ × ℕ)), ∀ d : ℕ,
    Dk.1 ≤ d →
    ∀ dom : ℕ → Bits 512,
      ((∃ salt : Bits 192, dom = Pous.BandMulti.segDom salt (2 ^ 9)) ∨ ∃ salt : Bits 256, dom = Pous.Sponge.dom salt) →
    let sch := Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 dom
    let β := if N = 1 then 524288 * (511 - Dk.1) - 69 * (512 - Dk.1) - 1 - (9 * Dk.2 + 129)
      else 523898 * ((512 - Dk.1) * N) - 532588
    ∀ (D_late Q_late T a b : ℕ), 0 < b → a ≤ b →
    ∀ (W : Bits sch.n) (mach : ContMachine sch) (f βe : ℝ), 0 < βe →
      pr (fun x : (Pous.Sponge.idealPerm (524288 + (524288 + 512))).Ω × sch.Coins × (Fin T → Fin b × Fin sch.B) =>
          let pp := (sch.enc W x.1 x.2.1).1
          let C := (sch.enc W x.1 x.2.1).2
          let rs := contRecords mach Dk.1 (2 ^ 20) D_late Q_late W pp C x.1 (arrAt a x.2.2) .start
            (histAt a x.2.2 T)
          contAccepts f rs ∧
            (⨅ θ : Set.Ioi (0 : ℝ), ((θ : ℝ) * f + Real.log (1 / βe)) / (1 - Real.exp (-(θ : ℝ)))) <
              ((a : ℝ) / b) * ∑ t : Fin T,
                if contIdle mach D_late W pp C x.1 a x.2.2 t then
                  max (contFreed sch (contHeldAt mach D_late W pp C x.1 a x.2.2 t) - epsProof sch β) 0
                else 0)
        ≤ ENNReal.ofReal βe + T * (2⁻¹ : ℝ≥0∞) ^ (9 * Dk.2 + 99)

/-! ### Proved results cited outside the package

The types of ten theorems that the protocol's documents and code cite, written out so that each citation names a
guarantee. Each is proved in `Pous.SecurityProofs` under its own name, from the theorem its docstring names. -/

open DRG Labelling in
/-- **The one-way bound for the stacked graph** (B2's graph under B5, with the bridge's hardness at
`s = ⌊(1 − ε)n⌋`, `D = ⌊βn⌋ − 1`, `k = n − ⌈εn/2⌉ + 1`). In `ℓ ≥ 1/(1 − α)` stacked `(n, α, β)`-depth-robust layers
joined by superconcentrators, numbered topologically by `e`, with one-way data-carrying labels, an `m`-bit
preprocessor and `(⌊βn⌋ − 1, Q)`-bounded online programs answer at least `n − ⌈εn/2⌉ + 1` of the `n` last-layer
challenges with probability at most `N · 2^m · ((n(Q + 1) + N)² / 2^w)^(⌊(1 − ε)n⌋ + 1)`. Proof:
`PousBridge.b5_stacked`. -/
def B5Stacked : Prop :=
  ∀ {ℓ n N : ℕ} {V : Type} [Fintype V] [DecidableEq V] (α β ε : ℝ)
    (node : Fin ℓ → Fin n → V) (level : V → Fin ℓ) (drg : Fin ℓ → Fin n → Fin n → Prop) (green : V → V → Prop)
    (hnode : ∀ i, Function.Injective (node i)) (hlevel : ∀ i a, level (node i a) = i)
    (hdr : ∀ i, DepthRobust n (drg i) α β)
    (hsc : ∀ i j : Fin ℓ, (j : ℕ) = i + 1 → DisjointPaths node level green i j)
    (hn : 0 < n) (hε : 0 < ε) (hε1 : ε ≤ 1) (hℓ : 1 ≤ (1 - α) * ℓ) (hβ : 1 ≤ β * n)
    (e : V ≃ Fin N) (G : TopoDAG N) (hG : ∀ u v, e u ∈ G.par (e v) ↔ StackedEdge node drg green u v)
    (t : Fin ℓ) (ht : (t : ℕ) + 1 = ℓ)
    (w m Q : ℕ) (d : Fin N → Bits w)
    (A₁ : (LabelQry N w → Bits w) → Bits m)
    (A₂ : Bits m → Fin N → Prog (IdealModel.randomOracle (LabelQry N w) (Bits w)) (Bits w))
    (hA₂ : ∀ σ c, (A₂ σ c).Bounded (⌊β * n⌋₊ - 1) Q),
    pr (fun H : LabelQry N w → Bits w =>
        n - ⌈ε / 2 * n⌉₊ + 1 ≤ ((Finset.univ.image fun b => e (node t b)).filter fun c =>
          (A₂ (A₁ H) c).run H = label G d H c).card) ≤
      N * 2 ^ m * ((((n * (Q + 1) + N) ^ 2 : ℕ) : ℝ≥0∞) / 2 ^ w) ^ (⌊(1 - ε) * n⌋₊ + 1)

open Dense in
/-- **The dense scheme meets the requirement at the 64 KB point**, random-oracle `H`, with sequential reveal, `D = 5`
rounds, `Q = 2^20`, `k = 106` challenges and `ε = 2^-128`: correct, rate 1, and every `⌊ρ|C|⌋`-bit adversary is
accepted with probability at most `1% + 2^-128`. Proof: `PousDenseReCert.dense_meets_64`. -/
def DenseMeets64 : Prop :=
  Pous.Meets (denseScheme (2 ^ 9) 524288) .sequential 5 (2 ^ 20) 106 ((2⁻¹ : ENNReal) ^ 128)

open Dense PousDenseReCert in
/-- **The segmented dense scheme meets the requirement at 14 GB** (`N = 448` segments of `2^9` blocks of 64 KB),
random-oracle `H`, sequential reveal, `D = 5`, `Q = 2^20`, `k = 106`, `ε = 2^-128`. Same `k` as one segment: the audit
soundness is per-block, so it does not degrade with the number of segments. Proof: `PousDenseReCert.seg_meets_14`. -/
def SegMeets14 : Prop :=
  Pous.Meets (segScheme 448 (2 ^ 9) 524288) .sequential 5 (2 ^ 20) 106 ((2⁻¹ : ENNReal) ^ 128)

open Dense PousDenseReCert in
/-- **The 14 GB theorem for the implementation's keying.** The tagged segmented dense scheme (448 segments of `2^9`
blocks of 64 KB, oracle `H(salt ‖ u64 s, local query)`) meets the requirement for every salt, random-oracle `H`,
sequential reveal, `D = 5`, `Q = 2^20`, `k = 106`, `ε = 2^-128`. Proof: `PousDenseReCert.segTag_meets_14`. -/
def SegTagMeets14 : Prop :=
  ∀ salt : Bits 192,
    Pous.Meets (segTagScheme 448 (2 ^ 9) 524288 salt) .sequential 5 (2 ^ 20) 106 ((2⁻¹ : ENNReal) ^ 128)

open Dense Labelling in
/-- **The tagged scheme is as secure as the global one.** For a query map `f` that is injective and sends segment
`s`'s queries to their global form, every sequential-audit bound of `segScheme` holds for `segTagScheme`: a tagged
adversary, with its queries sent through `f`, is a global adversary accepted with the same probability. Proof:
`PousDenseReCert.segTag_auditSecure`. -/
def SegTagAuditSecure : Prop :=
  ∀ {N B ℓ : ℕ} [NeZero (N * B)] (salt : Bits 192)
    (f : Bits (192 + 64) × LabelQry B ℓ → LabelQry (N * B) ℓ) (hf : Function.Injective f)
    (htag : ∀ (s : Fin N) i p,
      f (tag salt s.val, (i, p)) = (segIndexEquiv N B (s, i), PousDenseReCert.liftPar N B ℓ s p))
    {S D Q k : ℕ} {δ ε : ℝ≥0∞} (h : AuditSecure (segScheme N B ℓ) .sequential S D Q k δ ε),
    AuditSecure (segTagScheme N B ℓ salt) .sequential S D Q k δ ε

open Dense in
/-- **Distinct tags give the same domain separation as global indices.** The per-segment tagged keying
`(s, i) ↦ (tag_s, i)` is the global-index keying (`segIndexEquiv`) composed with an injective relabeling `t`, for
`N ≤ 2^64` segments. So keying by `tag_s = salt ‖ u64 s` separates the `N·B` blocks exactly as global indices do.
Proof: `PousDenseReCert.domainSep_equiv`. -/
def DomainSepEquiv : Prop :=
  ∀ (N B : ℕ) (hN : N ≤ 2 ^ 64) (salt : Bits 192),
    ∃ t : Fin (N * B) → Bits (192 + 64) × Fin B, Function.Injective t ∧
      ∀ p : Fin N × Fin B, (tag salt p.1.val, p.2) = t (segIndexEquiv N B p)

open Dense in
/-- **Satisfiability witness at 14 GB:** the implementation's tags `salt ‖ u64 s` are distinct across the 448
segments. Proof: `PousDenseReCert.tag_injective_14`. -/
def TagInjective14 : Prop :=
  ∀ salt : Bits 192, Function.Injective (fun s : Fin 448 => tag salt s.val)

open ColumnGame in
/-- **`B(220, 12)` at `d = 10`** (`D = 20` oracle rounds), in the two-directional column game: saving at most 3
labels. Proof: `PousP3Column.band_twoWay_220_12`. -/
def BandTwoWay220_12 : Prop :=
  ColumnHard (PousP3Column.bandGraph 220 12) 10 220 3

open ColumnGame in
/-- **`B(220, 12)` at `d = 11`** (`D ≤ 22`), in the two-directional column game: saving at most 3 labels. Proof:
`PousP3Column.band_twoWay_220_12_d11`. -/
def BandTwoWay220_12D11 : Prop :=
  ColumnHard (PousP3Column.bandGraph 220 12) 11 220 3

open ColumnGame in
/-- **`B(220, 16)` at `d = 12`** (`D ≤ 24`), in the two-directional column game: saving at most 3 labels. Proof:
`PousP3Column.band_twoWay_220_16`. -/
def BandTwoWay220_16 : Prop :=
  ColumnHard (PousP3Column.bandGraph 220 16) 12 220 3

end Pous.Guarantees
