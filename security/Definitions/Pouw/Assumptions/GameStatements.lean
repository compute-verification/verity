module

public import Definitions.Pouw.M1
public import Definitions.Pouw.Witness.Model
public import Definitions.Pouw.Witness.Copy
public import Definitions.Pouw.Assumptions.Game
public import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-! The PoUW game's headline statements over its named assumptions, each a `Prop` a proof inhabits. -/

@[expose] public section

namespace Pouw.GameStatements

open Finset

/-- **(a) `G_g` from TT.** For every cost model, protocol and domain: if `W_ref(u) ≤ Ω_max·W_mm(u)` on the domain
and TT(g₀) holds with error `δ(q, N)`, then `G_g` holds with `g = 1 − (1 − g₀)/Ω_max` and the same failure probability `δ`. `K ⊆ K'` is
built into the definitions (`Correct = CheckedOK ∧ OutOK`); no random-oracle term is needed because TT is stated in
the same game, and no collision term because both bound their event with no break (`Adversary.Breaks`). -/
def GapFromTT : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (g₀ Ωmax : ℝ) (δ : ℕ → ℕ → ℝ),
    g₀ < 1 → 0 < Ωmax →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N,
      (P.Wref U.layout u : ℝ) ≤ Ωmax * ((U.layout.shape u).Wmm : ℝ)) →
    Assumptions.TT CM P D g₀ δ →
    GGap CM P D (reductionGap g₀ Ωmax) δ

/-- Whether the milestone meets `Ω_max ≤ 199/198` is a count of `H_ref`'s M_4090 schedule, not proved here. -/
def MilestoneRequirement : Prop :=
  ∀ (CM : CostModel D1.M1Q D1.M1R Salt) (Wref : Layout → ℕ → ℕ) (qref : Workload → ℕ) (Ωmax : ℝ)
    (δ : ℕ → ℕ → ℝ),
    0 < Ωmax → Ωmax ≤ 199 / 198 →
    (∀ U : Workload, U.InDomain D1.m1Domain → ∀ u < U.layout.N,
      (Wref U.layout u : ℝ) ≤ Ωmax * ((U.layout.shape u).Wmm : ℝ)) →
    AdmitsRef CM (D1.m1 Wref).toProtocol D1.m1Domain qref →
    Assumptions.TT CM (D1.m1 Wref).toProtocol D1.m1Domain Params.gap0 δ →
    Requirement CM D1.m1Domain (reductionGap Params.gap0 Ωmax) δ ∧ reductionGap Params.gap0 Ωmax ≤ 1 / 100

def MilestoneExhaustion : Prop :=
  ∀ (CM : CostModel D1.M1Q D1.M1R Salt) (Wref : Layout → ℕ → ℕ) (Ωmax : ℝ) (δ : ℕ → ℕ → ℝ),
    0 < Ωmax →
    (∀ U : Workload, U.InDomain D1.m1Domain → ∀ u < U.layout.N,
      (Wref U.layout u : ℝ) ≤ Ωmax * ((U.layout.shape u).Wmm : ℝ)) →
    Assumptions.TT CM (D1.m1 Wref).toProtocol D1.m1Domain Params.gap0 δ →
    ∀ U : Workload, U.InDomain D1.m1Domain → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 →
    ∀ A : Adversary CM, A.Bounded U T q → ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      pr (fun x : (D1.M1Q → D1.M1R) × Salt × (Fin t → Ticket (Wref U.layout) U.layout.N) =>
          AuditAccepts (correctSet (D1.m1 Wref).toProtocol U x.1 x.2.1 (A.trace U x.1 x.2.1)) x.2.2 ∧
          T < (1 - reductionGap Params.gap0 Ωmax) * (1 - εs) * totalWref (D1.m1 Wref).toProtocol U.layout ∧
          ¬ A.Breaks U x.1 x.2.1)
        ≤ (1 - εs) ^ t + δ q U.layout.N

def GapBudget : Prop :=
  ∀ Ωmax : ℝ, 0 < Ωmax → (reductionGap Params.gap0 Ωmax ≤ 1 / 100 ↔ Ωmax ≤ 199 / 198)

def Usefulness : Prop :=
  ∀ (k r : ℕ) (A B EL ER FL FR : Mat),
    mm k (A + mm r EL ER) (B + mm r FL FR) - mm r (mm k A FL) FR - mm r EL (mm k ER (B + mm r FL FR))
      = mm k A B

def Int32Trace : Prop :=
  ∀ (k : ℕ) (A B : Mat) (i j : ℕ), k ≤ 133144 →
    (∀ l < k, -127 ≤ A i l ∧ A i l ≤ 126) → (∀ l < k, -127 ≤ B l j ∧ B l j ≤ 126) →
    mmW k A B i j = mm k A B i j ∧ InInt32 (mm k A B i j)

def NoisedKBound : Prop :=
  (∀ (k : ℕ) (a b : ℕ → ℤ), k ≤ 133144 → (∀ l < k, -127 ≤ a l ∧ a l ≤ 126) → (∀ l < k, -127 ≤ b l ∧ b l ≤ 126) →
      InInt32 (∑ l ∈ range k, a l * b l)) ∧
  ¬ InInt32 (∑ _l ∈ range 133145, (-127 : ℤ) * (-127))

def Int8KBound : Prop :=
  (∀ (k : ℕ) (a b : ℕ → ℤ), k ≤ 2 ^ 17 - 1 → (∀ l < k, InInt8 (a l)) → (∀ l < k, InInt8 (b l)) →
      InInt32 (∑ l ∈ range k, a l * b l)) ∧
  ¬ InInt32 (∑ _l ∈ range (2 ^ 17), (-128 : ℤ) * (-128))

def Int32Decode : Prop :=
  ∀ (k r : ℕ) (A B EL ER FL FR : Mat) (i j : ℕ), k ≤ 2 ^ 19 - 1 →
    (∀ l < k, InInt7 (A i l)) → (∀ l < k, InInt7 (B l j)) →
    wrap32 (wrap32 (mmW k (fun i' l => wrap32 (A i' l + mmW r EL ER i' l))
          (fun l j' => wrap32 (B l j' + mmW r FL FR l j')) i j
        - mmW r (mmW k A FL) FR i j)
      - mmW r EL (mmW k ER (fun l j' => wrap32 (B l j' + mmW r FL FR l j'))) i j) = mm k A B i j

def D1Complete : Prop :=
  ∀ (Q R S : Type) (P : D1.Params Q R S), P.NoiseRange → P.toProtocol.Complete (Layout.ShapesIn Shape.InDefaultDomain)

/-- The witness holds in the output-priced model (a query model), at `D1.m1` with `W_ref ≥ W_mm`, with error 0. -/
def TTWitness : Prop :=
  ∀ Wref : Layout → ℕ → ℕ,
    (∀ U : Workload, U.InDomain D1.m1Domain → ∀ u < U.layout.N, (U.layout.shape u).Wmm ≤ Wref U.layout u) →
    Assumptions.TT (outputPriced D1.M1Q D1.M1R Salt) (D1.m1 Wref).toProtocol D1.m1Domain Params.gap0 (fun _ _ => 0) ∧
    AdmitsRef (outputPriced D1.M1Q D1.M1R Salt) (D1.m1 Wref).toProtocol D1.m1Domain
      (fun U => U.layout.N * D1.m1Queries)

def MilestoneInstance : Prop :=
  ∀ Wref : Layout → ℕ → ℕ,
    (D1.m1 Wref).IsMilestone ∧ (D1.m1 Wref).NoiseRange ∧ (D1.m1 Wref).QueryBound D1.m1Queries ∧
    (∀ (s s' : Salt) (u u' : ℕ) (A A' : Mat) (w w' : ℕ), u < 2 ^ 64 → u' < 2 ^ 64 →
      ∀ x ∈ D1.m1ELQ s u A w, x ∈ (D1.m1 Wref).qAct s' u' A' w' ++ (D1.m1 Wref).qWt s' u' w' → u = u') ∧
    ∃ (H : D1.M1Q → D1.M1R) (s : Salt) (U : Workload) (A : Mat), U.InDomain D1.m1Domain ∧ 0 < U.layout.N ∧
      mm (D1.m1 Wref).r ((D1.m1 Wref).actF H s U 0 A).1 ((D1.m1 Wref).actF H s U 0 A).2 0 0 ≠ 0

/-- That M_4090 runs `H_ref` within the `m1Wref` schedule (`AdmitsRef`) is not proved. -/
def MilestoneWref : Prop :=
  (∀ U : Workload, U.InDomain D1.m1Domain → ∀ u < U.layout.N,
    (U.layout.shape u).Wmm ≤ D1.m1Wref U.layout u ∧
      (D1.m1Wref U.layout u : ℝ) ≤ D1.m1Omega * ((U.layout.shape u).Wmm : ℝ)) ∧
  D1.m1Omega ≤ 199 / 198 ∧ reductionGap Params.gap0 D1.m1Omega = 14017 / 1644225

def HonestComplete : Prop :=
  ∀ (Q R S : Type) (P : Protocol Q R S) (D : Layout → Prop) (U : Workload) (H : Q → R) (s : S) (act : ℕ → Mat),
    P.Complete D → U.InDomain D → (∀ u < U.layout.N, ActOK (U.layout.shape u) (act u)) →
    correctSet P U H s (honestTrace P H s U act) = range U.layout.N ∧
    ∀ g : ℝ, 0 ≤ g → g < 1 → ¬ Wins P U g (totalWref P U.layout) (honestTrace P H s U act) H s

def GameCanBeLost : Prop :=
  ∀ (Q R : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Nonempty R] (P : D1.Params Q R Salt) (c : ℕ) (Ω : ℝ),
    P.NoiseRange → P.QueryBound c → 16 * c ≤ 2 ^ 64 → 1 ≤ Ω →
    (∀ U : Workload, U.InDomain D1.m1Domain → ∀ u < U.layout.N,
      Ω * ((U.layout.shape u).Wmm : ℝ) ≤ (P.Wref U.layout u : ℝ)) →
    (∀ g : ℝ, g < 1 → ∀ δ : ℕ → ℕ → ℝ,
      GGap (freeModel Q R Salt) P.toProtocol D1.m1Domain g δ → 1 ≤ δ (16 * c) 16) ∧
    (∀ g : ℝ, g < 1 - 1 / Ω → ∀ δ : ℕ → ℕ → ℝ,
      GGap (outputPriced Q R Salt) P.toProtocol D1.m1Domain g δ → 1 ≤ δ (16 * c) 16)

def CopyBound : Prop :=
  ∀ (Ω V : Type) [Fintype Ω] (n : ℕ → ℕ) (X : Ω → ℕ → ℕ → V) (N : ℕ) (A : Copy.Adversary) (g₀ : ℝ),
    0 ≤ g₀ → g₀ < 1 → Copy.Nz n X →
    pr (fun ω => (Copy.cost n N A : ℝ) / (1 - g₀) < 16 * ∑ u ∈ Copy.correctSet n X N A ω, (n u : ℝ)) ≤
      ∑ u ∈ range N, (2 : ℝ) ^ (-(g₀ * n u))

/-- The witness: values uniform and independent on `{0, 1}`, `M` positions per unit. -/
def CopyNzWitness : Prop :=
  ∀ M : ℕ, Copy.Nz (fun _ => M) (fun (ω : Fin M → Bool) (_ : ℕ) => Copy.bits ω)

end Pouw.GameStatements
