module

public import Proofs.Properties.Verity.Sampled.Window.Subprotocol
public import Proofs.Properties.Verity.Sampled.Binding
public import Proofs.Properties.Verity.Sampled.Window.FilesSent
public import Proofs.Properties.Verity.Sampled.Digest

/-! SPOT's subprotocol over a window, assembled from its pieces. -/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open FlockSoundness.Discharge.Tables (SubmitJ LiveOutJ)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

noncomputable def ExecutionPhase.srcCoinsAt (m : ℕ) (W : ExecutionPhase) : W.Ω → Fin m → Fin 512 → Bool :=
  if h : W.Lb 0 = m * 64 then byteCoins W.src h else fun _ _ _ => false

theorem ExecutionPhase.avg_srcCoinsAt {m : ℕ} {W : ExecutionPhase} (hL : W.Lb 0 = m * 64) (H : (Fin m → Fin 512 → Bool) → ℝ≥0∞) :
    avg (fun ω => H (W.srcCoinsAt m ω)) = avg H := by
  have h : W.srcCoinsAt m = byteCoins W.src hL := by
    unfold ExecutionPhase.srcCoinsAt
    exact dif_pos hL
  rw [h]
  exact W.avg_byteCoins hL H

variable (V : Vocab d α) (k : Verity.Registration.Composite)

noncomputable def executionPhaseBindFinder (W : ExecutionPhase) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Strategy (V.game θ F c)) : Claims.Finder :=
  (bindCut V k θ F c).finder σ (2 * V.cost θ F c σ + 2 * W.h)

theorem executionPhaseBindFinder_calls : ∀ (W : ExecutionPhase) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Strategy (V.game θ F c)) (x : (executionPhaseBindFinder V k W θ F c σ).S),
    ((executionPhaseBindFinder V k W θ F c σ).cost x : ℝ) ≤ W.qM (V.cost θ F c σ) := fun W θ F c σ _ => by
  show ((2 * V.cost θ F c σ + 2 * W.h : ℕ) : ℝ) ≤ W.qM (V.cost θ F c σ)
  simp [ExecutionPhase.qM]

noncomputable def ExecutionPhaseConstructions.withBinding {V : Vocab d α} (K : ExecutionPhaseConstructions V)
    (T : PinnedTables) (dev : Exchange → Properties.Challenger.History) (k : Verity.Registration.Composite) :
    ExecutionPhaseConstructions V :=
  { K with
    readFinder := executionPhaseReadFinder V T dev
    forger := Binding.forger V dev
    bindFinder := executionPhaseBindFinder V k
    bind_calls := executionPhaseBindFinder_calls V k }

theorem executionPhaseSpotSound_assembled (e : 𝔽 ≃ (Fin 128 → Bool)) (rd : ExecutionPhaseReadings) (K : ExecutionPhaseConstructions V)
    (T : PinnedTables) (dev : Exchange → Properties.Challenger.History)
    (D : (W : ExecutionPhase) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      ExecutionPhaseDecoders V θ F c W σ)
    (label : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → String) (nc : ℕ)
    {S : ByteArray → Array ByteArray → Option ExecutionPhase}
    (hD : ExecutionPhaseDeployed V ((K.withBinding T dev k).withProject V rd D
      (executionPhaseSampleMaps e V rd label (afterAt V k) fun W => W.srcCoinsAt nc)) S)
    (hCov : ExecutionPhaseCovers V k S)
    (I : (W : ExecutionPhase) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      (W.extend (executionPhaseSampleMaps e V rd label (afterAt V k) (fun W => W.srcCoinsAt nc) W θ F c σ).widening).L.Ω →
        Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → ℕ → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → Prop)
    (hP : ∀ θ F W, S θ F = some W → ∀ o : Exchange, o.params = θ → Lines.PrintsLog (dev o) o.log)
    (hS : ∀ θ F W, S θ F = some W → ∀ c, Binding.SignsPlay V dev θ F c)
    (hL : ∀ θ F W, S θ F = some W → W.Lb 0 = nc * 64)
    (hR : ∀ θ F W, S θ F = some W → ∀ (c : ByteArray) j pre fr, (rd W θ F c).round j pre fr →
      ∃ i s n, fr = .opening i s n)
    (hm : ∀ θ F W, S θ F = some W → ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      MemberSteps V θ F c k
        (fun o => (Binding.Logged Binding.readsV2 o ∧ TablesSent T o.sent ∧ FilesSent V o) ∧
          SampledDigest k o.sent (afterAt V k θ F c σ))
        W (V.cost θ F c σ) (rd W θ F c) σ (D W θ F c σ)
        (executionPhaseSampleMaps e V rd label (afterAt V k) (fun W => W.srcCoinsAt nc) W θ F c σ)
        (fun _ _ => linkEmptyAt e (finish (V.signer θ) θ F c σ.1))
        (fun _ => roundEmptyAt e (finish (V.signer θ) θ F c σ.1)) (I W θ F c σ)) :
    V.SpotSound k :=
  executionPhaseSpotSound_of_sample V k e rd (K.withBinding T dev k) D label (afterAt V k) (fun W => W.srcCoinsAt nc) hD hCov
    (fun o => Binding.Logged Binding.readsV2 o ∧ TablesSent T o.sent ∧ FilesSent V o)
    (fun θ F c σ o => SampledDigest k o.sent (afterAt V k θ F c σ)) I
    (fun θ F W hW H => ExecutionPhase.avg_srcCoinsAt (hL θ F W hW) H)
    (fun θ F W hW c σ => logged_files_le V T k dev θ F c σ
      (Binding.binds_of_prints V k dev θ (hP θ F W hW)) (hS θ F W hW c))
    (fun θ F W _ c σ => prob_not_sampledDigest_le V k θ F c σ (2 * V.cost θ F c σ + 2 * W.h))
    hR hm

end Verity.Sampled
