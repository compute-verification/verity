import Definitions.Pouw.NCP.Protocol
import Definitions.Pouw.NCP.Witness
import Definitions.Pouw.Assumptions.NCP.Assumptions
import Definitions.Pouw.Barrier.Defs
import Definitions.Pouw.Game.Params

/-!
# NCP's game instance: pinned statements (trusted)
-/

namespace Pouw.NCP.GamePinned

open Finset

/-- **The three-block int32 bound**, a sanity bound for the symmetric noise range: for a
signed-permutation `R`, int7 `A` and `B`, `E₁` and `F₁` in `[−31, 31]` and `k ≤ 107,524`, every running sum of the
stacked product (inner depth `τ ≤ 3k`) is in int32. Blocks 1 and 2 give at most `2·95²·k`; in block 3 the running sum
is `A·B` minus the rest of block 3, at most `64²·k + 126²·k`. The checked values are taken mod `2^32` (NA2), so
completeness does not use it. -/
def ThreeBlock : Prop :=
  ∀ (k : ℕ) (A B E1 F1 R : Mat) (i j τ : ℕ), k ≤ 107524 → SignedPerm k R →
    (∀ l < k, InInt7 (A i l)) → (∀ l < k, InInt7 (B l j)) →
    (∀ l < k, InSmall (E1 i l)) → (∀ l < k, InSmall (F1 l j)) → τ ≤ 3 * k →
    InInt32 (mm τ (stackX k A E1 R) (stackY k B R F1) i j)

/-- **Usefulness**: for every NCP instance with `d > 0` and every noise, the last checked value (mod `2^32`) is
`A·B` at every entry whose row of `A` and column of `B` are int7, when `k ≤ 2^19 − 1`. -/
def LastChecked : Prop :=
  ∀ (Q R S : Type) (P : Params Q R S) (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) (i j : ℕ), 0 < P.d →
    (U.layout.shape u).k ≤ 2 ^ 19 - 1 → (∀ l < (U.layout.shape u).k, InInt7 (A i l)) →
    (∀ l < (U.layout.shape u).k, InInt7 (U.weight (U.layout.wid u) l j)) →
    P.Y H s U u A (D1.nBlocks P.d (3 * (U.layout.shape u).k) - 1) i j
      = mm (U.layout.shape u).k A (U.weight (U.layout.wid u)) i j

/-- **NCP's honest reference is complete** on the lead's domain, for every noise and every `R`: its int32
accumulator is the checked values mod `2^32` by definition, and its useful output equals `A·B` (`WrapInt7`). -/
def LeadComplete : Prop :=
  ∀ (Q R S : Type) (P : Params Q R S), P.toProtocol.Complete leadDomain

/-- **`G_γ` from TT_NCP**, with credited work `3·m·k·n`: for every cost model, protocol, segment length `d` and
domain, if `W_ref(u) ≤ Ω_max·3·W_mm(u)` and `γ₀ ≥ d/(3k)` on the domain and TT_NCP(γ₀) holds with error `ε`, then
`G_γ` holds with `γ = 1 − (1 − γ₀)/Ω_max` and `η = ε`. -/
def GammaFromTTNCP : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (d : ℕ) (D : Layout → Prop) (γ₀ Ωmax : ℝ) (ε : ℕ → ℕ → ℝ),
    γ₀ < 1 → 0 < Ωmax →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N,
      (P.Wref U.layout u : ℝ) ≤ Ωmax * (3 * ((U.layout.shape u).Wmm : ℝ))) →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N, (d : ℝ) ≤ γ₀ * (3 * (U.layout.shape u).k)) →
    Assumptions.TTNCP CM P d D γ₀ ε →
    Gγ CM P D (reductionGamma γ₀ Ωmax) ε

/-- **Satisfiability witness for TT_NCP** at every `γ₀ ∈ [0, 1)`: for every NCP instance with at most `c` queries per
unit and `W_ref ≥ 3·W_mm`, in the output-priced model at `3·m·k·n` (a query model) TT_NCP(γ₀) holds with error 0, and
one program runs `H_ref` at cost `≤ W_ref` with `N·c` oracle calls. -/
def TTNCPWitness : Prop :=
  ∀ (Q R : Type) [Fintype Q] [DecidableEq Q] [Fintype R] (P : Params Q R Salt) (c : ℕ) (γ₀ : ℝ),
    0 ≤ γ₀ → γ₀ < 1 → P.QueryBound c →
    (∀ U : Workload, U.InDomain leadDomain → ∀ u < U.layout.N, 3 * (U.layout.shape u).Wmm ≤ P.Wref U.layout u) →
    Assumptions.TTNCP (outputPriced3 Q R Salt) P.toProtocol P.d leadDomain γ₀ (fun _ _ => 0) ∧
    AdmitsRef (outputPriced3 Q R Salt) P.toProtocol leadDomain (fun U => U.layout.N * c)

/-- **The one-segment adversary** (class (c) for NCP, and why TT_NCP needs `γ₀ ≥ d/(3k)`): for every NCP instance
with `d > 0`, noise in `[−31, 32]` and signed-permutation `R` (so that its exact running sums at zero input, at most
`3k·64²`, are their own values mod `2^32`), and every in-domain layout with `d ∣ k` on every unit, at zero weights, in the abstract model
`Pouw.Barrier.gemmModel` (plain products priced `m·k·n`, everything else free), there is an adversary that commits
`A = 0`, computes the first `3k/d − 1` running sums step by step and commits the known last one (`A·B = 0`): it is a
cheap point at cost `∑_u (3·m·k·n − m·d·n)` with no oracle calls, and for every `γ₀ < 1` with `γ₀·3k < d` on every unit
(and at least one unit) the TT_NCP event at that cost holds for every oracle and salt. -/
def OneSegment : Prop :=
  ∀ (Q R S : Type) (P : Params Q R S) (Lay : Layout), 0 < P.d → P.NoiseRange → P.RSigned →
    (∀ u < Lay.N, P.d ∣ (Lay.shape u).k) →
    (⟨Lay, fun _ => 0⟩ : Workload).InDomain leadDomain →
    ∃ A : Adversary (Barrier.gemmModel P.nz),
      Barrier.CheapPoint (Barrier.gemmModel P.nz) P.toProtocol leadDomain ⟨Lay, fun _ => 0⟩ A
        (∑ u ∈ range Lay.N, (3 * ((Lay.shape u).Wmm : ℝ) - (Lay.shape u).m * P.d * (Lay.shape u).n)) 0 ∧
      ∀ γ₀ : ℝ, γ₀ < 1 → 0 < Lay.N → (∀ u < Lay.N, γ₀ * (3 * (Lay.shape u).k) < P.d) →
        ∀ (H : Q → R) (s : S),
          (∑ u ∈ range Lay.N, (3 * ((Lay.shape u).Wmm : ℝ) - (Lay.shape u).m * P.d * (Lay.shape u).n)) / (1 - γ₀)
            < ∑ u ∈ checkedSet P.toProtocol ⟨Lay, fun _ => 0⟩ H s (A.transcript ⟨Lay, fun _ => 0⟩ H s),
                3 * ((Lay.shape u).Wmm : ℝ)

/-- **`Barrier.Segment`'s hypotheses hold jointly with the lead's noise**: for all factor sources and
every inner dimension `k` of the default domain, at `L = 3k/16`, `d = 16`, `α(A) = [A | A | −A]`, `β(B) = [B; B; B]`
and the noise `Barrier.stackNoise` with an `R` that passes the rejection rule: `L ≥ 1`, `L·d = 3k`, `α 0 = 0`,
`β 0 = 0`, the noise cancels at the zero input, a layout is in the domain (`leadDomain` with inner dimension `k`) at
zero weights with positive work, `W_ref = 3·W_mm ≥ m·(L·d)·n` on it, and the protocol is complete there. So `Segment`
applies, with `1/L = 16/(3k)`. -/
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

/-- **The lead's instance, jointly**: an NCP instance with `IsLead` (`d = 16`, noise in
`[−31, 32]`, the rejection rule), signed-permutation `R` on domain shapes, no oracle queries and `W_ref = 3·W_mm`; its
protocol is complete on `leadDomain`, the output-priced model at `3·m·k·n` runs its `H_ref` with no oracle calls, and
TT_NCP(γ₀) holds there with error 0 at every `γ₀ ∈ [0, 1)`; and a layout with at least one unit and `d ∣ k` on every
unit is in the domain at zero weights. So the hypotheses of `LeadComplete`, `TTNCPWitness`, `GammaFromTTNCP` (at
`Ω_max = 1`) and `OneSegment` hold together at one instance. Its `R` is the cyclic shift by 24, which passes the rule
for `k ≥ 1024` with `16 ∣ k` (each step's image is disjoint from it and straddles two steps, and `R²` is the shift by
48, not `±I`); it is a witness, not a proposal. -/
def LeadWitness : Prop :=
  ∃ (P : Params Unit Unit Salt) (Lay : Layout), P.IsLead ∧ P.RSigned ∧ P.QueryBound 0 ∧
    (∀ (L : Layout) (u : ℕ), P.Wref L u = 3 * (L.shape u).Wmm) ∧
    P.toProtocol.Complete leadDomain ∧
    AdmitsRef (outputPriced3 Unit Unit Salt) P.toProtocol leadDomain (fun _ => 0) ∧
    (∀ γ₀ : ℝ, 0 ≤ γ₀ → γ₀ < 1 →
      Assumptions.TTNCP (outputPriced3 Unit Unit Salt) P.toProtocol P.d leadDomain γ₀ (fun _ _ => 0)) ∧
    (⟨Lay, fun _ => 0⟩ : Workload).InDomain leadDomain ∧ 0 < Lay.N ∧ ∀ u < Lay.N, P.d ∣ (Lay.shape u).k

/-- **The int8 range at the lead's noise**: for `R` a signed permutation, `E₁`, `F₁` in `[−31, 32]`
and int7 `A`, `B`, every entry of both stacked operands lies in `[−127, 127]`. -/
def RangeLead : Prop :=
  ∀ (k : ℕ) (A B E1 F1 R : Mat), SignedPerm k R →
    (∀ i, ∀ l < k, InNoise6 (E1 i l)) → (∀ l < k, ∀ j, InNoise6 (F1 l j)) →
    (∀ i, ∀ l < k, InInt7 (A i l)) → (∀ l < k, ∀ j, InInt7 (B l j)) →
    (∀ i l, l < 3 * k → -127 ≤ stackX k A E1 R i l ∧ stackX k A E1 R i l ≤ 127) ∧
      (∀ l j, l < 3 * k → -127 ≤ stackY k B R F1 l j ∧ stackY k B R F1 l j ≤ 127)

end Pouw.NCP.GamePinned
