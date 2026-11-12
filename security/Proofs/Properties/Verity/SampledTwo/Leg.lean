module

public import Proofs.Properties.Verity.Sampled.Target
public import Proofs.Flock.SamplingGeneric.Guarantees

/-!
# SPOT's target at the two-stage form: the leg's statements

`Verity.Vocab.SpotSound` reads the composite's sampled component, whose parameters name a form. The one-stage leg
(`Verity.Sampled.spotSound_of_targets`) proves the target at the one-stage form and reads every other form as refused
(`Verity.Sampled.recursiveForm_refused`). A two-stage form of the component makes that split three-way: this module
states the target form by form and the two-stage form's leg, from the two-stage verifier's knowledge soundness at the
auditor's held copies (`Flock.Guarantees.SpotKSTwo`'s conclusion, `Live₂.KS`) to SPOT's target on the parameters that
name the two-stage form.

* **The target on a set of parameters** (`TargetOn`): SPOT's target asked only at the parameters `P` picks. SPOT's
  target is its conjunction over the forms (`spotSound_of_forms`): the one-stage form's, the two-stage form's, and the
  parameters that name neither.
* **The two-stage session of record** (`Live₂`): the auditor's held copies (`SpotFlock.Held`), the run's statement (the
  boundary registration's record, as bytes), the confidence `δ` the knowledge soundness is read at, and the claim's
  share `ε`. Its verifier is the compiled two-stage verifier at SPOT[Flock]'s code (`Live₂.L`), its relation the
  two-stage relation at `δ` (`Live₂.R`).
* **The session's error at a prover** (`Live₂.errAt`): what `CompiledTwo.KS` bounds `PrBad` by at a prover `P`,
  `κ_δ(x) + √(γ(x)·Pr[B^P])` plus the binding finder's collision (`Live₂.prBad_le`).
* **The coupling at the two-stage form** (`CouplingTargetTwo`): where the claim fails on the vocabulary's transcript,
  some prover of the two-stage verifier's run makes the sampled verdict at most as likely as `PrBad` at the session's
  relation plus `terms` (the device's coin trees, its log's read, its signature and the messages' two-continuation
  finders), and where `error`'s premises hold, the session's error at that prover plus `terms` is at most the published
  `δ`. It holds three pieces in one statement: the projection of the period's game onto the two-stage verifier's
  public-coin run, the claim's bridge from the vocabulary's transcript to the per-run witness the extractor gives
  (the claim's units at the two-stage form, and the witness fixed at registration), and the bound's bookkeeping.
* **The deployment's sessions cover the form** (`CoversTwo`): where the deployment holds no session and the parameters
  name the two-stage form, the sampled verdict accepts with probability at most the published `δ`.
* **The leg** (`TwoLeg`, `twoLeg`): from the knowledge soundness at each session of record in scope, the cover and the
  coupling at each session, SPOT's target on the two-stage form's parameters.

Statements and their bookkeeping only: no module imports this one, and no guarantee reads it. The two-stage form's
parameters are a predicate here (`TwoAt`) because the component's two-stage parse is not written.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **SPOT's target at the parameters `P` picks**: at every `θ` with `P θ`, all committed files and commitments and
every strategy, the probability that the sampled verdict accepts while its claim fails is at most its error. -/
def TargetOn (V : Vocab d α) (k : Verity.Registration.Composite) (P : ByteArray → Prop) : Prop :=
  ∀ (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)), P θ →
    Verity.Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ V.Holds k θ F c σ .sampled o) (V.game θ F c) σ ≤
      V.error .sampled θ F c σ

/-- **SPOT's target, form by form**: its target on the parameters naming one form (`P₁`), on those naming the other
(`P₂`), and on those naming neither. -/
theorem spotSound_of_forms (V : Vocab d α) (k : Verity.Registration.Composite) (P₁ P₂ : ByteArray → Prop)
    (h₁ : TargetOn V k P₁) (h₂ : TargetOn V k P₂) (hoff : TargetOn V k fun θ => ¬ P₁ θ ∧ ¬ P₂ θ) :
    V.SpotSound k := by
  intro θ F c σ
  by_cases a : P₁ θ
  · exact h₁ θ F c σ a
  by_cases b : P₂ θ
  · exact h₂ θ F c σ b
  exact hoff θ F c σ ⟨a, b⟩

/-- **The two-stage session of record**: the auditor's held copies, fixed before the run (`h`); the run's statement,
the boundary registration's record as bytes (`x`); the confidence the knowledge soundness is read at (`δ`); and the
claim's share (`ε`). -/
structure Live₂ where
  h : SpotFlock.Held
  x : Msg
  δ : ℝ
  ε : ℝ

namespace Live₂

/-- **The session's verifier**: the compiled two-stage verifier at SPOT[Flock]'s code at the held copies, its sessions
Flock's masked verifier over the readable relation, the interior registrations' roots, the two-stage design, the
replay sample of record and the replay cap, as `Flock.Guarantees.SpotKSTwo` states it. -/
noncomputable abbrev L (A : Live₂) :=
  SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (SpotFlock.code A.h)
    SpotFlock.iroots (SpotFlock.two A.h) (SpotFlock.rs A.h) (SpotFlock.cap A.h)

/-- **The session's knowledge soundness**: `CompiledTwo.KS` at SHA-512, `Flock.Guarantees.SpotKSTwo`'s conclusion at
the held copies. -/
def KS (A : Live₂) : Prop := A.L.KS FlockDeploy.sha512

/-- **The session's relation**: the two-stage relation at the session's `δ` (`Claims.Sampling.RTwo`): at some
interiors, the extracted boundary trace's footprint is in `M̄_δ`. -/
noncomputable abbrev R (A : Live₂) :=
  Claims.Sampling.RTwo FlockDeploy.sha512 A.L.commit A.L.parse A.L.partition A.L.two A.δ

/-- **The session's error at a prover `P`**: `κ_δ(x) + √(γ(x)·Pr[B^P(x)])` plus the binding finder's collision. -/
noncomputable def errAt (A : Live₂) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) : ℝ :=
  A.L.κ A.δ A.x + Real.sqrt (A.L.γ A.x * PrB FlockDeploy.sha512 A.L.V P A.x) +
    (Claims.Sampling.Bext FlockDeploy.sha512 A.L.V A.L.E Ω P A.x).Pr FlockDeploy.sha512

/-- **The session's knowledge soundness at its statement**: in scope and at `δ ≥ 0`, every prover's `PrBad` at the
session's relation is at most the session's error at it. -/
theorem prBad_le (A : Live₂) (hKS : A.KS) (hs : A.L.Scope A.x) (hδ : 0 ≤ A.δ) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) : PrBad FlockDeploy.sha512 A.L.V A.R A.L.E A.x P ≤ A.errAt P := by
  have h := hKS A.δ hδ A.x hs Ω P
  simpa [errAt] using h

end Live₂

/-- **The coupling at the two-stage form, at the session of record `A`**: where the claim fails on the vocabulary's
transcript at a strategy, some prover of the two-stage verifier's run makes the sampled verdict at most as likely as
`PrBad` at the session's relation plus `terms`, and where `error`'s premises hold the session's error at that prover
plus `terms` is at most the published `δ` at the strategy's calls. -/
def CouplingTargetTwo (V : Vocab d α) (k : Verity.Registration.Composite) (A : Live₂) (θ : ByteArray)
    (F : Array ByteArray) (terms : (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) → ℝ≥0∞) : Prop :=
  ∀ (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)),
    ¬ Accounting.IntegrityHolds (V.partsOf θ F) (V.Opens θ F c) (V.extract θ F c σ) (V.certOf θ).ε →
      ∃ (Ω : Type) (_ : Fintype Ω) (_ : Nonempty Ω) (P : Prover Ω),
        Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ ≤
            ENNReal.ofReal (PrBad FlockDeploy.sha512 A.L.V A.R A.L.E A.x P) + terms c σ ∧
          (V.finderCR .sampled θ F c σ → V.premises .sampled θ F c σ →
            ENNReal.ofReal (A.errAt P) + terms c σ ≤ V.delta θ .sampled (V.cost θ F c σ))

/-- **The deployment's sessions `S₂` cover the two-stage form at composite `k`**: where `S₂` holds no session and the
parameters name the two-stage form (`TwoAt`), each strategy that meets the further finders' premises makes the sampled
verdict accept with probability at most the published `δ` at its calls. -/
def CoversTwo (V : Vocab d α) (k : Verity.Registration.Composite) (TwoAt : ByteArray → Prop)
    (S₂ : ByteArray → Array ByteArray → Option Live₂) : Prop :=
  ∀ θ F, S₂ θ F = none → TwoAt θ → ∀ (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)),
    V.finderCR .sampled θ F c σ →
      Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ ≤ V.delta θ .sampled (V.cost θ F c σ)

/-- **The two-stage leg**: where each session of record `S₂` holds is knowledge sound, in scope and at `δ ≥ 0`, the
sessions cover the two-stage form, and the coupling holds at each session at some `terms`, SPOT's target holds on the
two-stage form's parameters. -/
def TwoLeg (V : Vocab d α) (k : Verity.Registration.Composite) (TwoAt : ByteArray → Prop)
    (S₂ : ByteArray → Array ByteArray → Option Live₂) : Prop :=
  (∀ θ F A, S₂ θ F = some A → A.KS ∧ A.L.Scope A.x ∧ 0 ≤ A.δ) → CoversTwo V k TwoAt S₂ →
    (∀ θ F A, S₂ θ F = some A → ∃ terms, CouplingTargetTwo V k A θ F terms) → TargetOn V k TwoAt

/-- **The two-stage leg holds**: its bookkeeping, at the coupling and the cover. -/
theorem twoLeg (V : Vocab d α) (k : Verity.Registration.Composite) (TwoAt : ByteArray → Prop)
    (S₂ : ByteArray → Array ByteArray → Option Live₂) : TwoLeg V k TwoAt S₂ := by
  intro hKS hCov hC θ F c σ hP
  unfold Vocab.error Accounting.condError
  beta_reduce
  split_ifs with hE
  swap
  · exact le_top
  obtain ⟨hfs, hpr, _, _⟩ := hE
  by_cases hI : Accounting.IntegrityHolds (V.partsOf θ F) (V.Opens θ F c) (V.extract θ F c σ) (V.certOf θ).ε
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
    obtain ⟨hks, hs, hδ⟩ := hKS θ F A hS
    obtain ⟨terms, hT⟩ := hC θ F A hS
    obtain ⟨Ω, _, _, P, hle, hdelta⟩ := hT c σ hI
    calc _ ≤ ENNReal.ofReal (PrBad FlockDeploy.sha512 A.L.V A.R A.L.E A.x P) + terms c σ := hle
      _ ≤ ENNReal.ofReal (A.errAt P) + terms c σ :=
          add_le_add (ENNReal.ofReal_le_ofReal (A.prBad_le hks hs hδ P)) le_rfl
      _ ≤ _ := hdelta hfs hpr

end Verity.SampledTwo
