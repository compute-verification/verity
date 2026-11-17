module

public import Proofs.Properties.Verity.Sampled.Target
public import Proofs.Flock.SamplingGeneric.Guarantees

/-! SPOT's target form by form, and the two-stage form's subprotocol from `Live₂.KS`; the two-stage parameters
are a predicate (`TwoAt`) because the component's two-stage parse is not written. -/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open scoped ENNReal

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

def TargetOn (V : Vocab d α) (k : Verity.Registration.Composite) (P : ByteArray → Prop) : Prop :=
  ∀ (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)), P θ →
    Verity.Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ V.Holds k θ F c σ .sampled o) (V.game θ F c) σ ≤
      V.error .sampled θ F c σ

theorem spotSound_of_forms (V : Vocab d α) (k : Verity.Registration.Composite) (P₁ P₂ : ByteArray → Prop)
    (h₁ : TargetOn V k P₁) (h₂ : TargetOn V k P₂) (hoff : TargetOn V k fun θ => ¬ P₁ θ ∧ ¬ P₂ θ) :
    V.SpotSound k := by
  intro θ F c σ
  by_cases a : P₁ θ
  · exact h₁ θ F c σ a
  by_cases b : P₂ θ
  · exact h₂ θ F c σ b
  exact hoff θ F c σ ⟨a, b⟩

structure Live₂ where
  h : SpotFlock.Held
  δ : ℝ
  ε : ℝ

namespace Live₂

noncomputable abbrev L (A : Live₂) :=
  SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (SpotFlock.codeConnected A.h)
    SpotFlock.iroots (SpotFlock.twoConnected A.h) (SpotFlock.rsConnected A.h) (SpotFlock.cap A.h)

/-- The subprotocol takes it as a hypothesis; it holds where `Flock.Guarantees.SpotKSTwo`'s premises do
(`Live₂.ks_of_connected`), and `SpotFlock.Addresses` cannot be met at a held code where the second stage reads a
checked unit's port off the interior record. -/
def KS (A : Live₂) : Prop := A.L.KS FlockDeploy.sha512

theorem ks_of_connected (A : Live₂)
    (hB : SpotFlock.BuiltWired A.h (SpotFlock.wireMap A.h) (SpotFlock.returned A.h))
    (hA : SpotFlock.AddressesConnected A.h) : A.KS :=
  _root_.Flock.SecurityProofs.SpotKSTwo A.h hB hA

noncomputable abbrev R (A : Live₂) :=
  Claims.Sampling.RTwo FlockDeploy.sha512 A.L.commit A.L.parse A.L.partition A.L.two A.δ

noncomputable def errAt (A : Live₂) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) : ℝ :=
  A.L.κ A.δ x + Real.sqrt (A.L.γ x * PrB FlockDeploy.sha512 A.L.V P x) +
    (Claims.Sampling.Bext FlockDeploy.sha512 A.L.V A.L.E Ω P x).Pr FlockDeploy.sha512

theorem prBad_le (A : Live₂) (hKS : A.KS) (x : Msg) (hs : A.L.Scope x) (hδ : 0 ≤ A.δ) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) : PrBad FlockDeploy.sha512 A.L.V A.R A.L.E x P ≤ A.errAt x P := by
  have h := hKS A.δ hδ x hs Ω P
  simpa [errAt] using h

end Live₂

def CouplingTwoAt (V : Vocab d α) (k : Verity.Registration.Composite) (A : Live₂) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (t : ℝ≥0∞) : Prop :=
  Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ = 0 ∨
    ∃ (x : Msg) (Ω : Type) (_ : Fintype Ω) (_ : Nonempty Ω) (P : Prover Ω),
      A.L.Scope x ∧
      Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ ≤
          ENNReal.ofReal (PrBad FlockDeploy.sha512 A.L.V A.R A.L.E x P) + t ∧
        (V.finderCR .sampled θ F c σ → V.premises .sampled θ F c σ →
          ENNReal.ofReal (A.errAt x P) + t ≤ V.delta θ .sampled (V.cost θ F c σ))

def CouplingTargetTwo (V : Vocab d α) (k : Verity.Registration.Composite) (A : Live₂) (θ : ByteArray)
    (F : Array ByteArray) (terms : (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) → ℝ≥0∞) : Prop :=
  ∀ (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)),
    ¬ ResourceAccounting.IntegrityHolds (V.partsOf θ F) (V.Opens θ F c) (V.extract θ F c σ) (V.certOf θ).ε →
      CouplingTwoAt V k A θ F c σ (terms c σ)

def CoversTwo (V : Vocab d α) (k : Verity.Registration.Composite) (TwoAt : ByteArray → Prop)
    (S₂ : ByteArray → Array ByteArray → Option Live₂) : Prop :=
  ∀ θ F, S₂ θ F = none → TwoAt θ → ∀ (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)),
    V.finderCR .sampled θ F c σ →
      Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ ≤ V.delta θ .sampled (V.cost θ F c σ)

def TwoSubprotocol (V : Vocab d α) (k : Verity.Registration.Composite) (TwoAt : ByteArray → Prop)
    (S₂ : ByteArray → Array ByteArray → Option Live₂) : Prop :=
  (∀ θ F A, S₂ θ F = some A → A.KS ∧ 0 ≤ A.δ) → CoversTwo V k TwoAt S₂ →
    (∀ θ F A, S₂ θ F = some A → ∃ terms, CouplingTargetTwo V k A θ F terms) → TargetOn V k TwoAt

theorem twoSubprotocol (V : Vocab d α) (k : Verity.Registration.Composite) (TwoAt : ByteArray → Prop)
    (S₂ : ByteArray → Array ByteArray → Option Live₂) : TwoSubprotocol V k TwoAt S₂ := by
  intro hKS hCov hC θ F c σ hP
  unfold Vocab.error ResourceAccounting.condError
  beta_reduce
  split_ifs with hE
  swap
  · exact le_top
  obtain ⟨hfs, hpr, _, _⟩ := hE
  by_cases hI : ResourceAccounting.IntegrityHolds (V.partsOf θ F) (V.Opens θ F c) (V.extract θ F c σ) (V.certOf θ).ε
  · calc Verity.Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ V.Holds k θ F c σ .sampled o) (V.game θ F c) σ
          ≤ Verity.Game.prob (fun _ => False) (V.game θ F c) σ :=
            Verity.Game.prob_mono (fun o ho => ho.2 hI) _ σ
      _ = 0 := Verity.Game.prob_false _ σ
      _ ≤ _ := zero_le
  have hver : Verity.Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ V.Holds k θ F c σ .sampled o) (V.game θ F c) σ ≤
      Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ :=
    Verity.Game.prob_mono (fun o ho => ho.1) _ σ
  refine hver.trans ?_
  refine le_trans ?_ (le_add_right (le_add_right le_rfl))
  cases hS : S₂ θ F with
  | none => exact hCov θ F hS hP c σ hfs
  | some A =>
    obtain ⟨hks, hδ⟩ := hKS θ F A hS
    obtain ⟨terms, hT⟩ := hC θ F A hS
    have hT' := hT c σ hI
    unfold CouplingTwoAt at hT'
    obtain h0 | ⟨x, Ω, _, _, P, hs, hle, hdelta⟩ := hT'
    · rw [h0]
      exact zero_le
    · calc _ ≤ ENNReal.ofReal (PrBad FlockDeploy.sha512 A.L.V A.R A.L.E x P) + terms c σ := hle
        _ ≤ ENNReal.ofReal (A.errAt x P) + terms c σ :=
            add_le_add (ENNReal.ofReal_le_ofReal (A.prBad_le hks x hs hδ P)) le_rfl
        _ ≤ _ := hdelta hfs hpr

end Verity.SampledTwo
