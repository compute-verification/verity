import Definitions.Pouw.Game.Params
import Definitions.Pouw.M1
import Definitions.Pouw.Witness.Model
import Definitions.Pouw.Witness.Copy
import Definitions.Pouw.Assumptions.Game

/-!
# The game's statements (trusted)

The game's headline statements over its named assumptions (`Pouw.Assumptions.TT`), each a `Prop`: `G_γ` from TT, the
milestone's requirement and exhaustion, the overhead budget, the decode's exactness over the integers and in int32, D1's
completeness, the witnesses for TT and Nz, and that the game can be lost. Their proofs are the theorems
`Pouw.Proofs.* : GameStatements.X`. (a) is the guarantee `Pouw.Guarantees.GammaFromTT`; (b), (c) and (d) are guarantees in
`Pouw.Guarantees.Game`.
-/

namespace Pouw.GameStatements

open Finset

/-- **(a) `G_γ` from TT.** For every cost model, protocol and domain: if `W_ref(u) ≤ Ω_max·W_mm(u)` on the domain
and TT(γ₀) holds with error `ε(q, N)`, then `G_γ` holds with `γ = 1 − (1 − γ₀)/Ω_max` and `η = ε`. `K ⊆ K'` is
built into the definitions (`Correct = CheckedOK ∧ OutOK`); no random-oracle term is needed because TT is stated in
the same game. -/
def GammaFromTT : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (γ₀ Ωmax : ℝ) (ε : ℕ → ℕ → ℝ),
    γ₀ < 1 → 0 < Ωmax →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N,
      (P.Wref U.layout u : ℝ) ≤ Ωmax * ((U.layout.shape u).Wmm : ℝ)) →
    Assumptions.TT CM P D γ₀ ε →
    Gγ CM P D (reductionGamma γ₀ Ωmax) ε

/-- **(d′) The milestone requirement, in budget form**. For the named instance `D1.m1` on its domain
and any `W_ref`: if `W_ref ≤ Ω_max·W_mm` with `Ω_max ≤ 199/198`, the cost model runs `H_ref` at cost `≤ W_ref`
(`AdmitsRef`), and TT(0.5%) holds with error `ε`, then §3.3's requirement holds with `γ = 1 − (199/200)/Ω_max ≤ 1%`
and `η = ε`. Whether the milestone meets `Ω_max ≤ 199/198` is a count of `H_ref`'s M_4090 schedule, not proved here. -/
def MilestoneRequirement : Prop :=
  ∀ (CM : CostModel D1.M1Q D1.M1R Salt) (Wref : Layout → ℕ → ℕ) (qref : Workload → ℕ) (Ωmax : ℝ)
    (ε : ℕ → ℕ → ℝ),
    0 < Ωmax → Ωmax ≤ 199 / 198 →
    (∀ U : Workload, U.InDomain D1.m1Domain → ∀ u < U.layout.N,
      (Wref U.layout u : ℝ) ≤ Ωmax * ((U.layout.shape u).Wmm : ℝ)) →
    AdmitsRef CM (D1.m1 Wref).toProtocol D1.m1Domain qref →
    Assumptions.TT CM (D1.m1 Wref).toProtocol D1.m1Domain Params.γ0 ε →
    Requirement CM D1.m1Domain (reductionGamma Params.γ0 Ωmax) ε ∧ reductionGamma Params.γ0 Ωmax ≤ 1 / 100

/-- **(d″) Milestone resource exhaustion**: (d) for `D1.m1` on its domain, at `γ = 1 − (199/200)/Ω_max`. -/
def MilestoneExhaustion : Prop :=
  ∀ (CM : CostModel D1.M1Q D1.M1R Salt) (Wref : Layout → ℕ → ℕ) (Ωmax : ℝ) (ε : ℕ → ℕ → ℝ),
    0 < Ωmax →
    (∀ U : Workload, U.InDomain D1.m1Domain → ∀ u < U.layout.N,
      (Wref U.layout u : ℝ) ≤ Ωmax * ((U.layout.shape u).Wmm : ℝ)) →
    Assumptions.TT CM (D1.m1 Wref).toProtocol D1.m1Domain Params.γ0 ε →
    ∀ U : Workload, U.InDomain D1.m1Domain → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 →
    ∀ A : Adversary CM, A.Bounded U T q → ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      pr (fun x : (D1.M1Q → D1.M1R) × Salt × (Fin t → Ticket (Wref U.layout) U.layout.N) =>
          AuditAccepts (correctSet (D1.m1 Wref).toProtocol U x.1 x.2.1 (A.transcript U x.1 x.2.1)) x.2.2 ∧
          T < (1 - reductionGamma Params.γ0 Ωmax) * (1 - εs) * totalWref (D1.m1 Wref).toProtocol U.layout)
        ≤ (1 - εs) ^ t + ε q U.layout.N

/-- **The overhead budget**: under TT(0.5%) the reduction's gap is at most 1% exactly when `Ω_max ≤ 199/198`
(about 0.505% of arithmetic overhead). -/
def GammaBudget : Prop :=
  ∀ Ωmax : ℝ, 0 < Ωmax → (reductionGamma Params.γ0 Ωmax ≤ 1 / 100 ↔ Ωmax ≤ 199 / 198)

/-- **(e1) Usefulness over the integers**: the decode `C' − (A·F_L)·F_R − E_L·(E_R·B')` with `C' = A'·B'`,
`A' = A + E_L·E_R`, `B' = B + F_L·F_R`, equals `A·B`, for all integer matrices and dimensions. -/
def Usefulness : Prop :=
  ∀ (k r : ℕ) (A B EL ER FL FR : Mat),
    mm k (A + mm r EL ER) (B + mm r FL FR) - mm r (mm k A FL) FR - mm r EL (mm k ER (B + mm r FL FR))
      = mm k A B

/-- **(e2) int32 exactness of the transcript**: with noised operands in `[−127, 126]` (int7 plus noise in
`[−63, 63]`) and `k ≤ 133144`, the int32 accumulator equals the exact partial sum, which lies in int32. -/
def Int32Transcript : Prop :=
  ∀ (k : ℕ) (A B : Mat) (i j : ℕ), k ≤ 133144 →
    (∀ l < k, -127 ≤ A i l ∧ A i l ≤ 126) → (∀ l < k, -127 ≤ B l j ∧ B l j ≤ 126) →
    mmW k A B i j = mm k A B i j ∧ InInt32 (mm k A B i j)

/-- **(e3) The exact `k` bound for noised operands**: every sum of `k ≤ 133144` products of entries in
`[−127, 126]` lies in int32, and `133145` products `(−127)·(−127)` do not. (For full-range int8 the bound is
`2^17 − 1`: `Int8KBound`.) -/
def NoisedKBound : Prop :=
  (∀ (k : ℕ) (a b : ℕ → ℤ), k ≤ 133144 → (∀ l < k, -127 ≤ a l ∧ a l ≤ 126) → (∀ l < k, -127 ≤ b l ∧ b l ≤ 126) →
      InInt32 (∑ l ∈ range k, a l * b l)) ∧
  ¬ InInt32 (∑ _l ∈ range 133145, (-127 : ℤ) * (-127))

/-- **(e3′) The problem statement's int8 bound, checked**: `k ≤ 2^17 − 1` suffices for int8 products, and `k = 2^17`
products `(−128)·(−128)` overflow. -/
def Int8KBound : Prop :=
  (∀ (k : ℕ) (a b : ℕ → ℤ), k ≤ 2 ^ 17 - 1 → (∀ l < k, InInt8 (a l)) → (∀ l < k, InInt8 (b l)) →
      InInt32 (∑ l ∈ range k, a l * b l)) ∧
  ¬ InInt32 (∑ _l ∈ range (2 ^ 17), (-128 : ℤ) * (-128))

/-- **(e4) int32 decode**: `H_ref`'s decode, with every multiply-add and subtraction wrapping in int32 and arbitrary
integer factors, returns exactly `A·B` for int7 `A`, `B` and `k ≤ 2^19 − 1`. -/
def Int32Decode : Prop :=
  ∀ (k r : ℕ) (A B EL ER FL FR : Mat) (i j : ℕ), k ≤ 2 ^ 19 - 1 →
    (∀ l < k, InInt7 (A i l)) → (∀ l < k, InInt7 (B l j)) →
    wrap32 (wrap32 (mmW k (fun i' l => wrap32 (A i' l + mmW r EL ER i' l))
          (fun l j' => wrap32 (B l j' + mmW r FL FR l j')) i j
        - mmW r (mmW k A FL) FR i j)
      - mmW r EL (mmW k ER (fun l j' => wrap32 (B l j' + mmW r FL FR l j'))) i j) = mm k A B i j

/-- **(e5) D1's honest reference is complete** on the §3.2 default domain when the factor distribution keeps noise
in `[−63, 63]`: its int32 checked values equal the exact transcript, and its int32 decode equals `A·B`. -/
def D1Complete : Prop :=
  ∀ (Q R S : Type) (P : D1.Params Q R S), P.NoiseRange → P.toProtocol.Complete (Layout.ShapesIn Shape.InDefaultDomain)

/-- **Satisfiability witness for TT** at the chain's parameters: for `D1.m1` with any `W_ref ≥ W_mm`, in the
output-priced model (a query model), TT(0.5%) holds with error 0, and one program runs `H_ref` at cost `≤ W_ref`
with `N·m1Queries` oracle calls. -/
def TTWitness : Prop :=
  ∀ Wref : Layout → ℕ → ℕ,
    (∀ U : Workload, U.InDomain D1.m1Domain → ∀ u < U.layout.N, (U.layout.shape u).Wmm ≤ Wref U.layout u) →
    Assumptions.TT (outputPriced D1.M1Q D1.M1R Salt) (D1.m1 Wref).toProtocol D1.m1Domain Params.γ0 (fun _ _ => 0) ∧
    AdmitsRef (outputPriced D1.M1Q D1.M1R Salt) (D1.m1 Wref).toProtocol D1.m1Domain
      (fun U => U.layout.N * D1.m1Queries)

/-- **The named instance**: for every `W_ref`, `D1.m1` has the milestone's rank and depth, the
weight noise shared per (epoch, weight) and `E_R` per (epoch, weight, group of 16) (`IsMilestone`), noise in
`[−63, 63]`, `m1Queries` queries per unit, `E_L` queries that no other unit below `2^64` asks, and
noise that is not identically zero on an in-domain workload. -/
def MilestoneInstance : Prop :=
  ∀ Wref : Layout → ℕ → ℕ,
    (D1.m1 Wref).IsMilestone ∧ (D1.m1 Wref).NoiseRange ∧ (D1.m1 Wref).QueryBound D1.m1Queries ∧
    (∀ (s s' : Salt) (u u' : ℕ) (A A' : Mat) (w w' : ℕ), u < 2 ^ 64 → u' < 2 ^ 64 →
      ∀ x ∈ D1.m1ELQ s u A w, x ∈ (D1.m1 Wref).qAct s' u' A' w' ++ (D1.m1 Wref).qWt s' u' w' → u = u') ∧
    ∃ (H : D1.M1Q → D1.M1R) (s : Salt) (U : Workload) (A : Mat), U.InDomain D1.m1Domain ∧ 0 < U.layout.N ∧
      mm (D1.m1 Wref).r ((D1.m1 Wref).actF H s U 0 A).1 ((D1.m1 Wref).actF H s U 0 A).2 0 0 ≠ 0

/-- **The milestone's `W_ref`**: the explicit `H_ref` schedule `D1.m1Wref` lies between `W_mm` and
`Ω_M1·W_mm` on every in-domain unit, with `Ω_M1 = 65769/65536 ≤ 199/198`, and the reduction's gap there is
`14017/1644225 ≈ 0.8525%`. With (d′) this gives §3.3's requirement at that gap from `AdmitsRef` at `m1Wref` (that
M_4090 runs `H_ref` within the schedule, not proved) and TT(0.5%). -/
def MilestoneWref : Prop :=
  (∀ U : Workload, U.InDomain D1.m1Domain → ∀ u < U.layout.N,
    (U.layout.shape u).Wmm ≤ D1.m1Wref U.layout u ∧
      (D1.m1Wref U.layout u : ℝ) ≤ D1.m1Omega * ((U.layout.shape u).Wmm : ℝ)) ∧
  D1.m1Omega ≤ 199 / 198 ∧ reductionGamma Params.γ0 D1.m1Omega = 14017 / 1644225

/-- **The honest reference is complete and does not win**: for a complete protocol, every unit of the honest
transcript is correct, and at budget `T = W` the honest reference does not win for `0 ≤ γ < 1`. -/
def HonestComplete : Prop :=
  ∀ (Q R S : Type) (P : Protocol Q R S) (D : Layout → Prop) (U : Workload) (H : Q → R) (s : S) (act : ℕ → Mat),
    P.Complete D → U.InDomain D → (∀ u < U.layout.N, ActOK (U.layout.shape u) (act u)) →
    correctSet P U H s (honestTranscript P H s U act) = range U.layout.N ∧
    ∀ γ : ℝ, 0 ≤ γ → γ < 1 → ¬ Wins P U γ (totalWref P U.layout) (honestTranscript P H s U act) H s

/-- **The game can be lost** (non-vacuity in the other direction), for every D1 instance with noise in `[−63, 63]`
and at most `c` queries per unit on the milestone domain, with `W_ref ≥ Ω·W_mm` for some `Ω ≥ 1`: in the free model
`G_γ` fails for every `η(16c, 16) < 1`, and in the witness model it fails below the witness's tight gap, for every
`γ < 1 − 1/Ω`. -/
def GameCanBeLost : Prop :=
  ∀ (Q R : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Nonempty R] (P : D1.Params Q R Salt) (c : ℕ) (Ω : ℝ),
    P.NoiseRange → P.QueryBound c → 16 * c ≤ 2 ^ 64 → 1 ≤ Ω →
    (∀ U : Workload, U.InDomain D1.m1Domain → ∀ u < U.layout.N,
      Ω * ((U.layout.shape u).Wmm : ℝ) ≤ (P.Wref U.layout u : ℝ)) →
    (∀ γ : ℝ, γ < 1 → ∀ η : ℕ → ℕ → ℝ,
      Gγ (freeModel Q R Salt) P.toProtocol D1.m1Domain γ η → 1 ≤ η (16 * c) 16) ∧
    (∀ γ : ℝ, γ < 1 - 1 / Ω → ∀ η : ℕ → ℕ → ℝ,
      Gγ (outputPriced Q R Salt) P.toProtocol D1.m1Domain γ η → 1 ≤ η (16 * c) 16)

/-- **R_copy's bound** (tt-attacks §6; the second, noise-dependent witness): under Nz, a copy adversary's correct
units are worth more than its cost at TT's rate, `∑_{u ∈ K'} 16·N_u > cost/(1 − γ₀)`, with probability at most
`∑_u 2^(−γ₀·N_u)`. -/
def CopyBound : Prop :=
  ∀ (Ω V : Type) [Fintype Ω] (n : ℕ → ℕ) (X : Ω → ℕ → ℕ → V) (N : ℕ) (A : Copy.Adversary) (γ₀ : ℝ),
    0 ≤ γ₀ → γ₀ < 1 → Copy.Nz n X →
    pr (fun ω => (Copy.cost n N A : ℝ) / (1 - γ₀) < 16 * ∑ u ∈ Copy.correctSet n X N A ω, (n u : ℝ)) ≤
      ∑ u ∈ range N, (2 : ℝ) ^ (-(γ₀ * n u))

/-- **Nz is satisfiable**: values uniform and independent on `{0, 1}`, `M` positions per unit. -/
def CopyNzWitness : Prop :=
  ∀ M : ℕ, Copy.Nz (fun _ => M) (fun (ω : Fin M → Bool) (_ : ℕ) => Copy.bits ω)

end Pouw.GameStatements
