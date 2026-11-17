module

public import Definitions.Pouw.NCP.Protocol
public import Definitions.Pouw.NCP.Witness
public import Definitions.Pouw.Assumptions.NCP.Assumptions
public import Definitions.Pouw.Barrier.Defs
public import Definitions.Pouw.Game.Params

/-! NCP's game instance: its completeness, satisfiability witnesses and the one-segment adversary. -/

@[expose] public section

namespace Pouw.NCP.GamePinned

open Finset

/-- **The three-block int32 bound**, a sanity bound for the symmetric noise range. The checked values are taken mod
`2^32`, so completeness does not use it. -/
def ThreeSegment : Prop :=
  ∀ (k : ℕ) (A B E1 F1 R : Mat) (i j τ : ℕ), k ≤ 107524 → SignedPerm k R →
    (∀ l < k, InInt7 (A i l)) → (∀ l < k, InInt7 (B l j)) →
    (∀ l < k, InSmall (E1 i l)) → (∀ l < k, InSmall (F1 l j)) → τ ≤ 3 * k →
    InInt32 (mm τ (stackX k A E1 R) (stackY k B R F1) i j)

/-- **Usefulness**: the last checked value, mod `2^32`, is `A·B`. -/
def LastChecked : Prop :=
  ∀ (Q R S : Type) (P : Params Q R S) (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) (i j : ℕ), 0 < P.d →
    (U.layout.shape u).k ≤ 2 ^ 19 - 1 → (∀ l < (U.layout.shape u).k, InInt7 (A i l)) →
    (∀ l < (U.layout.shape u).k, InInt7 (U.fixed (U.layout.fixedId u) l j)) →
    P.Y H s U u A (D1.nSegments P.d (3 * (U.layout.shape u).k) - 1) i j
      = mm (U.layout.shape u).k A (U.fixed (U.layout.fixedId u)) i j

def LeadComplete : Prop :=
  ∀ (Q R S : Type) (P : Params Q R S), P.toProtocol.Complete leadDomain

/-- **`G_g` from TT_NCP**, with credited work `3·m·k·n`: for every cost model, protocol, segment length `d` and
domain, if `W_ref(u) ≤ Ω_max·3·W_mm(u)` and `g₀ ≥ d/(3k)` on the domain and TT_NCP(g₀) holds with error `δ`, then
`G_g` holds with `g = 1 − (1 − g₀)/Ω_max` and the same failure probability `δ`. -/
def GapFromTTNCP : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (d : ℕ) (D : Layout → Prop) (g₀ Ωmax : ℝ) (δ : ℕ → ℕ → ℝ),
    g₀ < 1 → 0 < Ωmax →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N,
      (P.Wref U.layout u : ℝ) ≤ Ωmax * (3 * ((U.layout.shape u).Wmm : ℝ))) →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N, (d : ℝ) ≤ g₀ * (3 * (U.layout.shape u).k)) →
    Assumptions.TTNCP CM P d D g₀ δ →
    GGap CM P D (reductionGap g₀ Ωmax) δ

/-- **Satisfiability witness for TT_NCP** at every `g₀ ∈ [0, 1)`, in the output-priced model at `3·m·k·n`, a query model. -/
def TTNCPWitness : Prop :=
  ∀ (Q R : Type) [Fintype Q] [DecidableEq Q] [Fintype R] (P : Params Q R Salt) (c : ℕ) (g₀ : ℝ),
    0 ≤ g₀ → g₀ < 1 → P.QueryBound c →
    (∀ U : Workload, U.InDomain leadDomain → ∀ u < U.layout.N, 3 * (U.layout.shape u).Wmm ≤ P.Wref U.layout u) →
    Assumptions.TTNCP (outputPriced3 Q R Salt) P.toProtocol P.d leadDomain g₀ (fun _ _ => 0) ∧
    AdmitsRef (outputPriced3 Q R Salt) P.toProtocol leadDomain (fun U => U.layout.N * c)

/-- **The one-segment adversary**, in the abstract model `Pouw.Barrier.gemmModel`: why TT_NCP needs `g₀ ≥ d/(3k)`. -/
def OneSegment : Prop :=
  ∀ (Q R S : Type) (P : Params Q R S) (Lay : Layout), 0 < P.d → P.NoiseRange → P.RSigned →
    (∀ u < Lay.N, P.d ∣ (Lay.shape u).k) →
    (⟨Lay, fun _ => 0⟩ : Workload).InDomain leadDomain →
    ∃ A : Adversary (Barrier.gemmModel P.nz),
      Barrier.CheapPoint (Barrier.gemmModel P.nz) P.toProtocol leadDomain ⟨Lay, fun _ => 0⟩ A
        (∑ u ∈ range Lay.N, (3 * ((Lay.shape u).Wmm : ℝ) - (Lay.shape u).m * P.d * (Lay.shape u).n)) 0 ∧
      ∀ g₀ : ℝ, g₀ < 1 → 0 < Lay.N → (∀ u < Lay.N, g₀ * (3 * (Lay.shape u).k) < P.d) →
        ∀ (H : Q → R) (s : S),
          (∑ u ∈ range Lay.N, (3 * ((Lay.shape u).Wmm : ℝ) - (Lay.shape u).m * P.d * (Lay.shape u).n)) / (1 - g₀)
            < ∑ u ∈ checkedSet P.toProtocol ⟨Lay, fun _ => 0⟩ H s (A.trace ⟨Lay, fun _ => 0⟩ H s),
                3 * ((Lay.shape u).Wmm : ℝ)

/-- **`Barrier.Segment`'s hypotheses hold jointly with the lead's noise**, so `Segment` applies with `1/L = 16/(3k)`. -/
def SegmentLead : Prop :=
  ∀ (Q R S : Type) (e f : (Q → R) → S → Workload → ℕ → Mat → Mat) (k : ℕ), 2 ^ 10 ≤ k → k ≤ 2 ^ 16 → 64 ∣ k →
    ∃ (Rm : Mat) (Lay : Layout), Admissible k 16 Rm ∧ 1 ≤ 3 * k / 16 ∧ 3 * k / 16 * 16 = 3 * k ∧
      stackX k 0 0 0 = 0 ∧ stackY k 0 0 0 = 0 ∧
      (∀ (H : Q → R) (s : S) (U : Workload) (u : ℕ), ∀ a < (U.layout.shape u).m, ∀ b < (U.layout.shape u).n,
        mm (3 * k / 16 * 16) (Barrier.stackNoise k Rm e f H s U u 0 0) (Barrier.stackNoise k Rm e f H s U u 0 1) a b
          = 0) ∧
      (⟨Lay, fun _ => 0⟩ : Workload).InDomain (fun L => leadDomain L ∧ ∀ u < L.N, (L.shape u).k = k) ∧
      0 < ∑ u ∈ range Lay.N, (((Lay.shape u).m * 16 * (Lay.shape u).n : ℕ) : ℝ) ∧
      (∀ u < Lay.N, (((Lay.shape u).m * (3 * k / 16 * 16) * (Lay.shape u).n : ℕ) : ℝ)
        ≤ ((3 * (Lay.shape u).Wmm : ℕ) : ℝ)) ∧
      (Barrier.segProto (3 * k / 16) 16 (fun A => stackX k A 0 0) (fun B => stackY k B 0 0)
        (Barrier.stackNoise k Rm e f) (fun L u => 3 * (L.shape u).Wmm)).Complete
        (fun L => leadDomain L ∧ ∀ u < L.N, (L.shape u).k = k)

/-- **The lead's instance, jointly**: the hypotheses of `LeadComplete`, `TTNCPWitness`, `GapFromTTNCP` (at
`Ω_max = 1`) and `OneSegment` hold together at one instance. Its `R`, the cyclic shift by 24, is a witness, not a
proposal. -/
def LeadWitness : Prop :=
  ∃ (P : Params Unit Unit Salt) (Lay : Layout), P.IsLead ∧ P.RSigned ∧ P.QueryBound 0 ∧
    (∀ (L : Layout) (u : ℕ), P.Wref L u = 3 * (L.shape u).Wmm) ∧
    P.toProtocol.Complete leadDomain ∧
    AdmitsRef (outputPriced3 Unit Unit Salt) P.toProtocol leadDomain (fun _ => 0) ∧
    (∀ g₀ : ℝ, 0 ≤ g₀ → g₀ < 1 →
      Assumptions.TTNCP (outputPriced3 Unit Unit Salt) P.toProtocol P.d leadDomain g₀ (fun _ _ => 0)) ∧
    (⟨Lay, fun _ => 0⟩ : Workload).InDomain leadDomain ∧ 0 < Lay.N ∧ ∀ u < Lay.N, P.d ∣ (Lay.shape u).k

def RangeLead : Prop :=
  ∀ (k : ℕ) (A B E1 F1 R : Mat), SignedPerm k R →
    (∀ i, ∀ l < k, InNoise6 (E1 i l)) → (∀ l < k, ∀ j, InNoise6 (F1 l j)) →
    (∀ i, ∀ l < k, InInt7 (A i l)) → (∀ l < k, ∀ j, InInt7 (B l j)) →
    (∀ i l, l < 3 * k → -127 ≤ stackX k A E1 R i l ∧ stackX k A E1 R i l ≤ 127) ∧
      (∀ l j, l < 3 * k → -127 ≤ stackY k B R F1 l j ∧ stackY k B R F1 l j ≤ 127)

end Pouw.NCP.GamePinned
