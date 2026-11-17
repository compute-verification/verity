module

public import Proofs.Properties.Verity.Sampled.Window.Logged
public import Proofs.Properties.Verity.Sampled.Window.Charge

/-! SPOT's subprotocol over a window: the subprotocol from the walk's premises. -/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open FlockSoundness.Discharge.Tables (SubmitJ LiveOutJ)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

variable (V : Vocab d α) (k : Verity.Registration.Composite)

theorem executionPhaseSpotSound_of_logged (rd : ExecutionPhaseReadings) (K : ExecutionPhaseConstructions V)
    (D : (W : ExecutionPhase) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      ExecutionPhaseDecoders V θ F c W σ)
    (M : (W : ExecutionPhase) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      ExecutionPhaseCoinMap (V.game θ F c) σ W.Ω W.rounds)
    {S : ByteArray → Array ByteArray → Option ExecutionPhase} (hD : ExecutionPhaseDeployed V (K.withProject V rd D M) S)
    (hCov : ExecutionPhaseCovers V k S) (P : Exchange → Prop)
    (Q : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → Exchange → Prop)
    (BL : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → ExecutionPhaseLinkHold)
    (B : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → ExecutionPhaseRoundHold)
    (ha : ∀ (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Strategy (V.game θ F c)),
      prob (fun o => V.Verdict k .sampled o ∧ ¬ P o) (V.game θ F c) σ ≤
        ENNReal.ofReal ((K.readFinder θ F c σ).Pr sha512Msg) +
          ENNReal.ofReal ((K.forger θ F c σ).Pr Verity.lmsVerify))
    (hq : ∀ θ F W, S θ F = some W → ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      prob (fun o => V.Verdict k .sampled o ∧ ¬ Q θ F c σ o) (V.game θ F c) σ ≤
        ENNReal.ofReal (Real.sqrt ((K.bindFinder W θ F c σ).Pr sha512Msg)))
    (hC : ∀ θ F W, S θ F = some W → ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      ExecutionPhaseCouples V θ F c W (V.cost θ F c σ) (rd W θ F c) σ (D W θ F c σ) (M W θ F c σ) (BL θ F c σ) (B θ F c σ))
    (hDec : ∀ θ F W, S θ F = some W → ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      ExecutionPhaseDecodes V θ F c k (fun o => P o ∧ Q θ F c σ o) W (V.cost θ F c σ) (rd W θ F c) σ (D W θ F c σ)
        (M W θ F c σ) (BL θ F c σ) (B θ F c σ)) :
    V.SpotSound k :=
  spotSound_of_executionPhase V k hD hCov fun θ F W hW =>
    executionPhaseCouplingTarget_of_logged V k rd K D M W P θ F (Q θ F) (BL θ F) (B θ F) (ha θ F) (hq θ F W hW)
      (hC θ F W hW) (hDec θ F W hW)

theorem executionPhaseSpotSound_of_members (rd : ExecutionPhaseReadings) (K : ExecutionPhaseConstructions V)
    (D : (W : ExecutionPhase) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      ExecutionPhaseDecoders V θ F c W σ)
    (M : (W : ExecutionPhase) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      ExecutionPhaseCoinMap (V.game θ F c) σ W.Ω W.rounds)
    {S : ByteArray → Array ByteArray → Option ExecutionPhase} (hD : ExecutionPhaseDeployed V (K.withProject V rd D M) S)
    (hCov : ExecutionPhaseCovers V k S) (P : Exchange → Prop)
    (Q : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → Exchange → Prop)
    (BL : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → ExecutionPhaseLinkHold)
    (B : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → ExecutionPhaseRoundHold)
    (I : (W : ExecutionPhase) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      (W.extend (M W θ F c σ).widening).L.Ω → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → ℕ →
        Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → Prop)
    (ha : ∀ (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Strategy (V.game θ F c)),
      prob (fun o => V.Verdict k .sampled o ∧ ¬ P o) (V.game θ F c) σ ≤
        ENNReal.ofReal ((K.readFinder θ F c σ).Pr sha512Msg) +
          ENNReal.ofReal ((K.forger θ F c σ).Pr Verity.lmsVerify))
    (hq : ∀ θ F W, S θ F = some W → ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      prob (fun o => V.Verdict k .sampled o ∧ ¬ Q θ F c σ o) (V.game θ F c) σ ≤
        ENNReal.ofReal (Real.sqrt ((K.bindFinder W θ F c σ).Pr sha512Msg)))
    (hC : ∀ θ F W, S θ F = some W → ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      ExecutionPhaseCouples V θ F c W (V.cost θ F c σ) (rd W θ F c) σ (D W θ F c σ) (M W θ F c σ) (BL θ F c σ) (B θ F c σ))
    (hm : ∀ θ F W, S θ F = some W → ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      MemberDecodes V θ F c k (fun o => P o ∧ Q θ F c σ o) W (V.cost θ F c σ) (rd W θ F c) σ (D W θ F c σ)
        (M W θ F c σ) (BL θ F c σ) (B θ F c σ) (I W θ F c σ)) :
    V.SpotSound k :=
  executionPhaseSpotSound_of_logged V k rd K D M hD hCov P Q BL B ha hq hC fun θ F W hW c σ =>
    executionPhaseDecodes_of_members V θ F c k (fun o => P o ∧ Q θ F c σ o) W (V.cost θ F c σ) (rd W θ F c) σ (D W θ F c σ)
      (M W θ F c σ) (BL θ F c σ) (B θ F c σ) (I W θ F c σ) (hm θ F W hW c σ)

theorem executionPhaseSpotSound_of_sample (e : 𝔽 ≃ (Fin 128 → Bool)) (rd : ExecutionPhaseReadings) (K : ExecutionPhaseConstructions V)
    (D : (W : ExecutionPhase) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      ExecutionPhaseDecoders V θ F c W σ)
    (label : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → String)
    (after : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → ByteArray) {nc : ℕ}
    (κ : (W : ExecutionPhase) → W.Ω → Fin nc → Fin 512 → Bool) {S : ByteArray → Array ByteArray → Option ExecutionPhase}
    (hD : ExecutionPhaseDeployed V (K.withProject V rd D (executionPhaseSampleMaps e V rd label after κ)) S)
    (hCov : ExecutionPhaseCovers V k S) (P : Exchange → Prop)
    (Q : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → Exchange → Prop)
    (I : (W : ExecutionPhase) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      (W.extend (executionPhaseSampleMaps e V rd label after κ W θ F c σ).widening).L.Ω →
        Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → ℕ → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → Prop)
    (hκ : ∀ θ F W, S θ F = some W → ∀ H : (Fin nc → Fin 512 → Bool) → ℝ≥0∞, avg (fun ω => H (κ W ω)) = avg H)
    (ha : ∀ θ F W, S θ F = some W → ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      prob (fun o => V.Verdict k .sampled o ∧ ¬ P o) (V.game θ F c) σ ≤
        ENNReal.ofReal ((K.readFinder θ F c σ).Pr sha512Msg) +
          ENNReal.ofReal ((K.forger θ F c σ).Pr Verity.lmsVerify))
    (hq : ∀ θ F W, S θ F = some W → ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      prob (fun o => V.Verdict k .sampled o ∧ ¬ Q θ F c σ o) (V.game θ F c) σ ≤
        ENNReal.ofReal (Real.sqrt ((K.bindFinder W θ F c σ).Pr sha512Msg)))
    (hR : ∀ θ F W, S θ F = some W → ∀ (c : ByteArray) j pre fr, (rd W θ F c).round j pre fr →
      ∃ i s n, fr = .opening i s n)
    (hm : ∀ θ F W, S θ F = some W → ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      MemberSteps V θ F c k (fun o => P o ∧ Q θ F c σ o) W (V.cost θ F c σ) (rd W θ F c) σ (D W θ F c σ)
        (executionPhaseSampleMaps e V rd label after κ W θ F c σ) (fun _ _ => linkEmptyAt e (finish (V.signer θ) θ F c σ.1))
        (fun _ => roundEmptyAt e (finish (V.signer θ) θ F c σ.1)) (I W θ F c σ)) :
    V.SpotSound k :=
  spotSound_of_executionPhase V k hD hCov fun θ F W hW =>
    executionPhaseCouplingTarget_of_sample_at e V k rd K D label after κ W P θ F (Q θ F) (hκ θ F W hW) (ha θ F W hW)
      (hq θ F W hW) fun c σ =>
        executionPhaseDecodes_of_members V θ F c k (fun o => P o ∧ Q θ F c σ o) W (V.cost θ F c σ) (rd W θ F c) σ
          (D W θ F c σ) (executionPhaseSampleMaps e V rd label after κ W θ F c σ) _ _ _
          (memberDecodes_of_stops V θ F c k (fun o => P o ∧ Q θ F c σ o) W (V.cost θ F c σ) (rd W θ F c) σ
            (D W θ F c σ) (executionPhaseSampleMaps e V rd label after κ W θ F c σ)
            (fun _ _ => linkEmptyAt e (finish (V.signer θ) θ F c σ.1))
            (fun _ => roundEmptyAt e (finish (V.signer θ) θ F c σ.1)) (I W θ F c σ) (hR θ F W hW c)
            (hm θ F W hW c σ))

end Verity.Sampled
