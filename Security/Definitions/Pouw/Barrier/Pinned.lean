import Definitions.Pouw.Barrier.Defs

/-!
# The degenerate-input barrier: pinned statements (trusted)
-/

namespace Pouw.Barrier.Pinned

open Finset

/-- **The generic barrier, in probability.** In every cost model, for every protocol and domain: let `𝒜` be within
`(c₀, q)` on an in-domain workload `U` with `W = ∑_u W_ref(u) > 0`. Then for every `γ < 1 − c₀/W`, `G_γ` with error
`η` bounds by `η(q, N)` the probability that `𝒜` commits every unit of `U` correct. -/
def GenericPr : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (U : Workload) (A : Adversary CM) (c₀ : ℝ)
    (q : ℕ),
    U.InDomain D → 0 ≤ c₀ → q ≤ 2 ^ 64 → A.Bounded U c₀ q → 0 < totalWref P U.layout →
    ∀ γ : ℝ, γ < 1 - c₀ / totalWref P U.layout → ∀ η : ℕ → ℕ → ℝ, Gγ CM P D γ η →
      pr (fun x : (Q → R) × S => correctSet P U x.1 x.2 (A.transcript U x.1 x.2) = range U.layout.N)
        ≤ η q U.layout.N

/-- **Class (a) in the abstract model: `γ ≥ 1 − 1/p`.** For every noise source, domain and `W_ref ≥ p·W_mm`
(`p ≥ 1`), on every in-domain workload with positive work: the adversary that commits `A = 0`, computes
`(0 + E)(B + F)` with one product and outputs the constant `0` wins with probability 1 at every `γ < 1 − 1/p`. A
decode by plain products that costs one product more than the checked one (`p = 2`) gives `γ ≥ 1/2`; the standard
decode `C' − A·F − E·(B + F)` (`p = 3`) gives `γ ≥ 2/3`. -/
def Additive : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Nonempty R] [Fintype S] [Nonempty S]
    (N : NoiseSrc Q R S) (W : Layout → ℕ → ℕ) (p : ℝ) (D : Layout → Prop) (U : Workload),
    1 ≤ p → U.InDomain D → 0 < ∑ u ∈ range U.layout.N, ((U.layout.shape u).Wmm : ℝ) →
    (∀ u < U.layout.N, p * ((U.layout.shape u).Wmm : ℝ) ≤ (W U.layout u : ℝ)) →
    ∀ γ : ℝ, γ < 1 - 1 / p → ∀ η : ℕ → ℕ → ℝ, Gγ (gemmModel N) (addProto N W) D γ η → 1 ≤ η 0 U.layout.N

/-- **Class (a) is a real protocol** (satisfiability of `Additive`'s hypotheses on the protocol side): for every noise
source and domain, `addProto` is complete, and the abstract model runs a complete `H_ref` at `W_ref = 2·W_mm` (the
checked product and `A·B`) and at `W_ref = 3·W_mm` (the checked product and the standard decode). -/
def AdditiveWitness : Prop :=
  ∀ (Q R S : Type) (N : NoiseSrc Q R S) (D : Layout → Prop),
    (addProto N fun L u => 2 * (L.shape u).Wmm).Complete D ∧
    AdmitsRef (gemmModel N) (addProto N fun L u => 2 * (L.shape u).Wmm) D (fun _ => 0) ∧
    AdmitsRef (gemmModel N) (addProto N fun L u => 3 * (L.shape u).Wmm) D (fun _ => 0)

/-- **Class (b) in the abstract model: the one-relation leak, `γ ≥ 1/t`.** For `t ≥ 1` checked products under the
relation condition with a unit coefficient on the last (`c_{t−1}² = 1`), and `W_ref ≥ t·W_mm`, on every in-domain
workload with positive work: the adversary that commits `A = 0`, computes the first `t − 1` products and derives the
last one from the relation (at `A = 0` the useful output is `0`) wins with probability 1 at every `γ < 1/t`. -/
def Relation : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Nonempty R] [Fintype S] [Nonempty S]
    (t : ℕ) (c : ℕ → ℤ) (N : NoiseSrc Q R S) (W : Layout → ℕ → ℕ) (D : Layout → Prop) (U : Workload),
    1 ≤ t → c (t - 1) * c (t - 1) = 1 → RelNoise t c N → U.InDomain D →
    0 < ∑ u ∈ range U.layout.N, ((U.layout.shape u).Wmm : ℝ) →
    (∀ u < U.layout.N, (t : ℝ) * ((U.layout.shape u).Wmm : ℝ) ≤ (W U.layout u : ℝ)) →
    ∀ γ : ℝ, γ < 1 / t → ∀ η : ℕ → ℕ → ℝ, Gγ (gemmModel N) (relProto t c N W) D γ η → 1 ≤ η 0 U.layout.N

/-- **The relation condition is met by the lead's noise with output-only checks**: for all factor sources, the
noise source of the lead's form satisfies `RelNoise` with `t = 3`, `c = (1, 1, −1)`, and the last coefficient is a
unit. So the lead's noise with its three products checked only at their final values is class (b): `γ ≥ 1/3`. -/
def RelationWitness : Prop :=
  ∀ (Q R S : Type) (e f r : (Q → R) → S → Workload → ℕ → Mat → Mat),
    RelNoise 3 (fun i => if i = 2 then -1 else 1) (cancelNoise e f r) ∧
      (fun i : ℕ => if i = 2 then (-1 : ℤ) else 1) 2 * (fun i : ℕ => if i = 2 then (-1 : ℤ) else 1) 2 = 1

/-- **Class (b) with the lead's noise is a real protocol** (satisfiability of `Relation`'s hypotheses on the protocol
side): for all factor sources and domains, `relProto` with `t = 3`, `c = (1, 1, −1)` and the lead's noise
is complete, and the abstract model runs a complete `H_ref` at `W_ref = 3·W_mm` (the three checked products, and the
useful output as their free combination). -/
def RelationAdmits : Prop :=
  ∀ (Q R S : Type) (e f r : (Q → R) → S → Workload → ℕ → Mat → Mat) (D : Layout → Prop),
    (relProto 3 (fun i => if i = 2 then -1 else 1) (cancelNoise e f r) fun L u => 3 * (L.shape u).Wmm).Complete D ∧
    AdmitsRef (gemmModel (cancelNoise e f r))
      (relProto 3 (fun i => if i = 2 then -1 else 1) (cancelNoise e f r) fun L u => 3 * (L.shape u).Wmm) D
      (fun _ => 0)

/-- **Class (c) in the abstract model: one segment, `γ ≥ 1/L = d/(L·d)`.** For the running sums at depth `d` of one
product of inner dimension `L·d` (`L ≥ 1`) with linear operand maps (`α 0 = 0`, `β 0 = 0`), noise that cancels at the
zero input (the last running sum of `E·F` is `0` on the unit's entries) and `W_ref ≥ m·(L·d)·n`: on every in-domain
layout with zero weights and positive work, the adversary that commits `A = 0`, computes the first `L − 1` running
sums step by step and commits the known last one (`0`) wins with probability 1 at every `γ < 1/L`. -/
def Segment : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Nonempty R] [Fintype S] [Nonempty S]
    (L d : ℕ) (α β : Mat → Mat) (N : NoiseSrc Q R S) (W : Layout → ℕ → ℕ) (D : Layout → Prop) (Lay : Layout),
    1 ≤ L → α 0 = 0 → β 0 = 0 →
    (∀ (H : Q → R) (s : S) (U : Workload) (u : ℕ), ∀ a < (U.layout.shape u).m, ∀ b < (U.layout.shape u).n,
      mm (L * d) (N H s U u 0 0) (N H s U u 0 1) a b = 0) →
    (⟨Lay, fun _ => 0⟩ : Workload).InDomain D →
    0 < ∑ u ∈ range Lay.N, (((Lay.shape u).m * d * (Lay.shape u).n : ℕ) : ℝ) →
    (∀ u < Lay.N, (((Lay.shape u).m * (L * d) * (Lay.shape u).n : ℕ) : ℝ) ≤ (W Lay u : ℝ)) →
    ∀ γ : ℝ, γ < 1 / L → ∀ η : ℕ → ℕ → ℝ, Gγ (gemmModel N) (segProto L d α β N W) D γ η → 1 ≤ η 0 Lay.N

/-- **The hypotheses of `Generic` are jointly satisfiable, with a complete protocol**: in the abstract model there
are a protocol whose `H_ref` is complete and admitted at `W_ref`, an in-domain workload and a cheap point at a cost
`c₀` strictly below `W`, so the range `γ < 1 − c₀/W` contains `γ = 0`. -/
def GenericWitness : Prop :=
  ∃ (N : NoiseSrc Unit Unit Unit) (P : Protocol Unit Unit Unit) (D : Layout → Prop) (U : Workload)
    (A : Adversary (gemmModel N)) (c₀ : ℝ),
    P.Complete D ∧ AdmitsRef (gemmModel N) P D (fun _ => 0) ∧ CheapPoint (gemmModel N) P D U A c₀ 0 ∧
      0 < totalWref P U.layout ∧ c₀ < totalWref P U.layout

/-- **Entrywise encodings with a cheap correction are affine** (class (a′); `escapes-encodings.md` route 1e, from
worker `escapes-enc`'s `int7_tables_affine`): if an entrywise encoding `x ↦ s x`, `y ↦ t y` of int7 operands admits
a separable correction `s x · t y = α·x·y + g x + h y` with `α ≠ 0` on int7, then both tables are affine on int7, so
the encoding is an affine one, in class (a). -/
def TablesAffine : Prop :=
  ∀ (s t g h : ℤ → ℤ) (α : ℤ), α ≠ 0 →
    (∀ x y, InInt7 x → InInt7 y → s x * t y = α * x * y + g x + h y) →
    (∀ x, InInt7 x → s x = s 0 + x * (s 1 - s 0)) ∧ (∀ y, InInt7 y → t y = t 0 + y * (t 1 - t 0))

/-- **`TablesAffine`'s hypothesis is satisfiable** (the identity tables with `α = 1` and no correction). -/
def TablesAffineWitness : Prop :=
  ∃ (s t g h : ℤ → ℤ) (α : ℤ), α ≠ 0 ∧ ∀ x y, InInt7 x → InInt7 y → s x * t y = α * x * y + g x + h y

end Pouw.Barrier.Pinned
