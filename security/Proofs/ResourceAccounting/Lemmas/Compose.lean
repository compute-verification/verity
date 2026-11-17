module

public import Definitions.Compose.Guarantees.Compose

/-! Steps of `Definitions.Compose.Guarantees.Compose`'s proofs, each a `def G : Prop` proved by the `SecurityProofs`
theorem of the same name and pinned by no record. -/

@[expose] public section

namespace ResourceAccounting.Guarantees.Compose

open ResourceAccounting.TS ResourceAccounting.Compose ResourceAccounting.Assumptions.TS ResourceAccounting.Assumptions.Compose Pouw.RelativeSecurity

def ServiceBudget : Prop :=
  ∀ {C Ξ : Type} [Fintype C] [Fintype Ξ] {N M : ℕ} (stg : Fin N → ℕ) (W : SplitStages C (Fin N) M)
    (w lk lk0 : Fin N → ℝ) (K : ℕ) (ρ p p0 a b wmax Λ : ℝ),
    (∀ v, 0 ≤ w v) → 0 < ρ → (∀ v, ρ * N * w v ≤ 1) → (∀ j, stageShare stg w j ≤ wmax) →
    (∀ v, 0 < w v → lk v ≤ p ∧ lk0 v ≤ p0) → W.LuckAt lk lk0 stg → 0 ≤ a → 0 ≤ b →
    SplitRate K ρ p p0 a b wmax Λ → RoundGame.BudgetAt Λ (serviceGame (Ξ := Ξ) stg W w a b K)

def SplitRateClosed : Prop :=
  ∀ (K : ℕ) (ρ p p0 a b wmax : ℝ), 1 ≤ K → 0 < ρ → 0 ≤ wmax → ρ * wmax ≤ 1 → 0 ≤ p → p < 1 → 0 ≤ p0 → p0 < 1 →
    0 < a → 0 < b →
    SplitRate K ρ p p0 a b wmax
      (max (a * (1 + K * ρ * wmax) / (ρ * K * (1 - p))) (b * (1 + K * ρ * wmax) / (ρ * K * (1 - p0))))

def SplitRateExp : Prop :=
  ∀ (K : ℕ) (ρ p p0 a b wmax Λ : ℝ), 1 ≤ K → 0 < ρ → 0 < wmax → ρ * wmax ≤ 1 →
    0 ≤ p → p < 1 → 0 ≤ p0 → p0 < 1 → 0 < a → 0 < b →
    Λ = max (a * wmax / -Real.log (p + (1 - p) * Real.exp (-(K * ρ * wmax))))
      (b * wmax / -Real.log (p0 + (1 - p0) * Real.exp (-(K * ρ * wmax)))) →
    SplitRateAt K ρ p p0 a b wmax Λ ((Real.exp (a * wmax / Λ) - 1) / ((1 - p) * wmax))
      ((Real.exp (b * wmax / Λ) - 1) / ((1 - p0) * wmax))

def SplitRateMerged : Prop :=
  ∀ (K : ℕ) (ρ p p0 q a b wmax : ℝ), 1 ≤ K → 0 < ρ → 0 ≤ wmax → ρ * wmax ≤ 1 → 0 ≤ q → p ≤ q → p0 ≤ q → q < 1 →
    0 < a → a ≤ b → SplitRate K ρ p p0 a b wmax (b * (1 + K * ρ * wmax) / (ρ * K * (1 - q)))

/-- It reads no committed value and no rightness, so the run's modes may read every round's coin. -/
def ServiceCovers : Prop :=
  ∀ {ι K T L U C Ξ V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype C] [Fintype Ξ]
    (Sc : Scheme K) (Cd : Declared ι) {Np Kd : ℕ} (hr : 0 < Cd.rounds) (hK : Kd ≤ Np)
    (R : JointRun (Paths C Cd.rounds) (Fin Np)) (Pl : Play (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) 1 K T L U)
    (pr : ProofRun C (Equiv.Perm (Fin Np) × Ξ) Cd.rounds Kd Np V),
    Cd.Wf → LayerCommitted Cd → Pl.game = jointGame (Ξ := Ξ) R (tileCone Sc Cd Np) 1 1 Kd →
    (Cd.layer hr hK pr).Covers Pl (jointMode R) (fun _ t => (tileWords Sc Cd Np t : ℝ)) (wordsOf Sc Cd : ℝ)

open Classical in
/-- It assumes no hash: a prover whose fork finder beats `ecr` gets `⊤`, so it is the per-prover `δ_bind` the
statements take. -/
def BindingOfECR : Prop :=
  ∀ {J Q Pos V : Type} [Nonempty V] (C B : Type) [Fintype C] [Fintype B] (S m s : ℕ)
    (pf : SampledLayer C B S m s J Q Pos V) (w : Fin S) (L : Finset Pos) (H : Claims.Msg → Claims.Msg)
    (F : Claims.Finder) (κ δ' : ℝ), (∀ j k, pf.pos w j k ∈ L) → 0 < δ' →
    prob (fun p : Paths (Paths C m × B) S × Paths (Paths C m × B) S => ∃ ℓ ∈ L, pf.Forked ℓ p.1 p.2) ≤
      F.Pr H + κ →
    ENNReal.ofReal (prob (fun x => pf.MovedIn x w)) ≤
      ENNReal.ofReal δ' + ENNReal.ofReal ((L.card : ℝ) ^ 2 / (4 * δ')) *
        ((if Claims.ExpectedCR H F then ENNReal.ofReal (Claims.Ex (fun c => (F.cost c : ℝ)) / 2 ^ (256 : ℝ))
          else (⊤ : ENNReal)) + ENNReal.ofReal κ)

def WholeLuckOfEntropy : Prop :=
  ∀ {B V : Type} [Fintype B] [Fintype V] {Tq M : ℕ} (rd : V → ℕ) (R : JointRun (Paths (Fin Tq → B) M) V)
    (w h : V → ℝ) (κ0 : ℝ) (θ : ℕ → ℝ) (sq : Paths (Fin Tq → B) M → Fin M → Option (Fin Tq)) (δb : ℝ),
    0 ≤ κ0 → (∀ v, 0 ≤ w v) → (∀ r, 0 ≤ θ r) →
    (∀ v, Real.exp (-h v) * Real.exp (θ (rd v) * w v) +
      (1 - Real.exp (-h v)) * Real.exp (-(κ0 * (θ (rd v) * w v))) ≤ 1) →
    WholeEntropyAt rd R h sq δb → SeedLuckAt rd R w .whole κ0 θ sq δb

end ResourceAccounting.Guarantees.Compose
