module

public import Definitions.Pouw.Barrier.Defs

/-! The degenerate-input barrier's guarantees. -/

@[expose] public section

namespace Pouw.Barrier.Pinned

open Finset

def GenericPr : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (U : Workload) (A : Adversary CM) (c₀ : ℝ)
    (q : ℕ),
    U.InDomain D → 0 ≤ c₀ → q ≤ 2 ^ 64 → A.Bounded U c₀ q → 0 < totalWref P U.layout →
    ∀ g : ℝ, g < 1 - c₀ / totalWref P U.layout → ∀ δ : ℕ → ℕ → ℝ, GGap CM P D g δ →
      pr (fun x : (Q → R) × S =>
          correctSet P U x.1 x.2 (A.trace U x.1 x.2) = range U.layout.N ∧ ¬ A.Breaks U x.1 x.2)
        ≤ δ q U.layout.N

def Additive : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Nonempty R] [Fintype S] [Nonempty S]
    (N : NoiseSrc Q R S) (W : Layout → ℕ → ℕ) (p : ℝ) (D : Layout → Prop) (U : Workload),
    1 ≤ p → U.InDomain D → 0 < ∑ u ∈ range U.layout.N, ((U.layout.shape u).Wmm : ℝ) →
    (∀ u < U.layout.N, p * ((U.layout.shape u).Wmm : ℝ) ≤ (W U.layout u : ℝ)) →
    ∀ g : ℝ, g < 1 - 1 / p → ∀ δ : ℕ → ℕ → ℝ, GGap (gemmModel N) (addProto N W) D g δ → 1 ≤ δ 0 U.layout.N

/-- The witness holds at `W_ref = 2·W_mm` (the checked product and `A·B`) and at `W_ref = 3·W_mm` (the standard decode). -/
def AdditiveWitness : Prop :=
  ∀ (Q R S : Type) (N : NoiseSrc Q R S) (D : Layout → Prop),
    (addProto N fun L u => 2 * (L.shape u).Wmm).Complete D ∧
    AdmitsRef (gemmModel N) (addProto N fun L u => 2 * (L.shape u).Wmm) D (fun _ => 0) ∧
    AdmitsRef (gemmModel N) (addProto N fun L u => 3 * (L.shape u).Wmm) D (fun _ => 0)

def Relation : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Nonempty R] [Fintype S] [Nonempty S]
    (t : ℕ) (c : ℕ → ℤ) (N : NoiseSrc Q R S) (W : Layout → ℕ → ℕ) (D : Layout → Prop) (U : Workload),
    1 ≤ t → c (t - 1) * c (t - 1) = 1 → RelNoise t c N → U.InDomain D →
    0 < ∑ u ∈ range U.layout.N, ((U.layout.shape u).Wmm : ℝ) →
    (∀ u < U.layout.N, (t : ℝ) * ((U.layout.shape u).Wmm : ℝ) ≤ (W U.layout u : ℝ)) →
    ∀ g : ℝ, g < 1 / t → ∀ δ : ℕ → ℕ → ℝ, GGap (gemmModel N) (relProto t c N W) D g δ → 1 ≤ δ 0 U.layout.N

/-- The witness: `cancelNoise` with `t = 3` and `c = (1, 1, −1)`, its products checked only at their final values. -/
def RelationWitness : Prop :=
  ∀ (Q R S : Type) (e f r : (Q → R) → S → Workload → ℕ → Mat → Mat),
    RelNoise 3 (fun i => if i = 2 then -1 else 1) (cancelNoise e f r) ∧
      (fun i : ℕ => if i = 2 then (-1 : ℤ) else 1) 2 * (fun i : ℕ => if i = 2 then (-1 : ℤ) else 1) 2 = 1

/-- The witness: `relProto` with `t = 3`, `c = (1, 1, −1)` and `cancelNoise`, with `H_ref` at `W_ref = 3·W_mm`. -/
def RelationAdmits : Prop :=
  ∀ (Q R S : Type) (e f r : (Q → R) → S → Workload → ℕ → Mat → Mat) (D : Layout → Prop),
    (relProto 3 (fun i => if i = 2 then -1 else 1) (cancelNoise e f r) fun L u => 3 * (L.shape u).Wmm).Complete D ∧
    AdmitsRef (gemmModel (cancelNoise e f r))
      (relProto 3 (fun i => if i = 2 then -1 else 1) (cancelNoise e f r) fun L u => 3 * (L.shape u).Wmm) D
      (fun _ => 0)

def Segment : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Nonempty R] [Fintype S] [Nonempty S]
    (L d : ℕ) (α β : Mat → Mat) (N : NoiseSrc Q R S) (W : Layout → ℕ → ℕ) (D : Layout → Prop) (Lay : Layout),
    1 ≤ L → α 0 = 0 → β 0 = 0 →
    (∀ (H : Q → R) (s : S) (U : Workload) (u : ℕ), ∀ a < (U.layout.shape u).m, ∀ b < (U.layout.shape u).n,
      mm (L * d) (N H s U u 0 0) (N H s U u 0 1) a b = 0) →
    (⟨Lay, fun _ => 0⟩ : Workload).InDomain D →
    0 < ∑ u ∈ range Lay.N, (((Lay.shape u).m * d * (Lay.shape u).n : ℕ) : ℝ) →
    (∀ u < Lay.N, (((Lay.shape u).m * (L * d) * (Lay.shape u).n : ℕ) : ℝ) ≤ (W Lay u : ℝ)) →
    ∀ g : ℝ, g < 1 / L → ∀ δ : ℕ → ℕ → ℝ, GGap (gemmModel N) (segProto L d α β N W) D g δ → 1 ≤ δ 0 Lay.N

/-- The witness is in the abstract model, at a cheap point with `c₀ < W`, so the range `g < 1 − c₀/W` contains `g = 0`. -/
def GenericWitness : Prop :=
  ∃ (N : NoiseSrc Unit Unit Unit) (P : Protocol Unit Unit Unit) (D : Layout → Prop) (U : Workload)
    (A : Adversary (gemmModel N)) (c₀ : ℝ),
    P.Complete D ∧ AdmitsRef (gemmModel N) P D (fun _ => 0) ∧ CheapPoint (gemmModel N) P D U A c₀ 0 ∧
      0 < totalWref P U.layout ∧ c₀ < totalWref P U.layout

def TablesAffine : Prop :=
  ∀ (s t g h : ℤ → ℤ) (α : ℤ), α ≠ 0 →
    (∀ x y, InInt7 x → InInt7 y → s x * t y = α * x * y + g x + h y) →
    (∀ x, InInt7 x → s x = s 0 + x * (s 1 - s 0)) ∧ (∀ y, InInt7 y → t y = t 0 + y * (t 1 - t 0))

/-- The witness: the identity tables with `α = 1` and no correction. -/
def TablesAffineWitness : Prop :=
  ∃ (s t g h : ℤ → ℤ) (α : ℤ), α ≠ 0 ∧ ∀ x y, InInt7 x → InInt7 y → s x * t y = α * x * y + g x + h y

end Pouw.Barrier.Pinned
