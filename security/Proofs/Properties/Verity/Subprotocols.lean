module

public import Properties.Verity.Subprotocols
public import Proofs.Properties.Verity.Soundness
public import Proofs.Core.Game.Prob

/-! Each subprotocol's target from its bound at its compiled component, by monotonicity, and soundness from those
bounds. -/

@[expose] public section

namespace Verity

open ResourceAccounting Core
open scoped ENNReal

namespace Vocab

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α)

theorem target_eq (k : Verity.Registration.Composite) (a : Subprotocol) :
    V.Target k a = V.TargetFor (V.Verdict k a) k a := rfl

theorem sound_of (k : Verity.Registration.Composite) (a : Subprotocol) (h : V.ComponentBound k a) : V.Target k a :=
  fun θ F c σ =>
    le_trans (Game.prob_mono (p := fun o => V.Verdict k a o ∧ ¬ V.Holds k θ F c σ a o)
      (q := fun o => ComponentAccepts k a o.sent ∧ ¬ V.Holds k θ F c σ a o)
      (fun _ ⟨⟨_, _, hc⟩, hh⟩ => ⟨hc, hh⟩) _ σ) (h θ F c σ)

/-- **At the honest deployment**: at registered files, where subprotocol `a`'s honest lemma at composite `k` holds, the
deployment's honest prover meets `ComponentBound`'s event with probability `0`, within `a`'s error, and the target's
event with probability `0`. -/
theorem sound_of_honest (k : Verity.Registration.Composite) (a : Subprotocol) (h : V.HonestHolds k a)
    (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (hreg : V.Registered θ F) :
    Game.prob (fun o => ComponentAccepts k a o.sent ∧ ¬ V.Holds k θ F c (V.honest θ F c) a o) (V.game θ F c)
        (V.honest θ F c) ≤ V.error a θ F c (V.honest θ F c) ∧
      Game.prob (fun o => V.Verdict k a o ∧ ¬ V.Holds k θ F c (V.honest θ F c) a o) (V.game θ F c)
        (V.honest θ F c) = 0 := by
  have h0 := h θ F c hreg
  have hc : Game.prob (fun o => ComponentAccepts k a o.sent ∧ ¬ V.Holds k θ F c (V.honest θ F c) a o)
      (V.game θ F c) (V.honest θ F c) = 0 :=
    le_antisymm (le_trans (Game.prob_mono
      (p := fun o => ComponentAccepts k a o.sent ∧ ¬ V.Holds k θ F c (V.honest θ F c) a o)
      (q := fun o => ¬ V.Holds k θ F c (V.honest θ F c) a o) (fun _ ho => ho.2) _ _) h0.le) zero_le
  refine ⟨hc.le.trans zero_le, le_antisymm (le_trans (Game.prob_mono
    (p := fun o => V.Verdict k a o ∧ ¬ V.Holds k θ F c (V.honest θ F c) a o)
    (q := fun o => ¬ V.Holds k θ F c (V.honest θ F c) a o) (fun _ ho => ho.2) _ _) h0.le) zero_le⟩

/-- **The target fails for a verifier that accepts everything**: at a strategy on whose every play subprotocol `a`'s
claim at composite `k` fails, with its error there below one (so its finder meets `FinderCR` and its forger
`Unforgeable`, and its `ε_forge` and collision terms are below one). -/
theorem sound_of_nontrivial (k : Verity.Registration.Composite) (a : Subprotocol) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (V.game θ F c))
    (hfail : ∀ o, ¬ V.Holds k θ F c σ a o) (hlt : V.error a θ F c σ < 1) :
    ¬ V.TargetFor (fun _ => True) k a := by
  intro h
  have h1 := prob_play_eq_one (V.families θ) (D := V.signer θ) θ F c
    (fun o => (fun _ : Exchange => True) o ∧ ¬ V.Holds k θ F c σ a o) (fun o _ => ⟨trivial, hfail o⟩) σ
  exact lt_irrefl (1 : ℝ≥0∞) ((h1.symm.trans_le (h θ F c σ)).trans_lt hlt)

section

variable {k : Verity.Registration.Composite}

theorem spotSound_of (h : V.ComponentBound k .sampled) : V.SpotSound k := V.sound_of k .sampled h

theorem pouwSound_of (h : V.ComponentBound k .pouw) : V.PouwSound k := V.sound_of k .pouw h

theorem inferenceOnlySound_of (h : V.ComponentBound k .inferenceOnly) : V.InferenceOnlySound k :=
  V.sound_of k .inferenceOnly h

theorem spotSound_of_honest (h : V.HonestHolds k .sampled) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (hreg : V.Registered θ F) :
    Game.prob (fun o => ComponentAccepts k .sampled o.sent ∧ ¬ V.Holds k θ F c (V.honest θ F c) .sampled o)
        (V.game θ F c) (V.honest θ F c) ≤ V.error .sampled θ F c (V.honest θ F c) ∧
      Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ V.Holds k θ F c (V.honest θ F c) .sampled o) (V.game θ F c)
        (V.honest θ F c) = 0 :=
  V.sound_of_honest k .sampled h θ F c hreg

theorem pouwSound_of_honest (h : V.HonestHolds k .pouw) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (hreg : V.Registered θ F) :
    Game.prob (fun o => ComponentAccepts k .pouw o.sent ∧ ¬ V.Holds k θ F c (V.honest θ F c) .pouw o)
        (V.game θ F c) (V.honest θ F c) ≤ V.error .pouw θ F c (V.honest θ F c) ∧
      Game.prob (fun o => V.Verdict k .pouw o ∧ ¬ V.Holds k θ F c (V.honest θ F c) .pouw o) (V.game θ F c)
        (V.honest θ F c) = 0 :=
  V.sound_of_honest k .pouw h θ F c hreg

theorem inferenceOnlySound_of_honest (h : V.HonestHolds k .inferenceOnly) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (hreg : V.Registered θ F) :
    Game.prob (fun o => ComponentAccepts k .inferenceOnly o.sent ∧
          ¬ V.Holds k θ F c (V.honest θ F c) .inferenceOnly o)
        (V.game θ F c) (V.honest θ F c) ≤ V.error .inferenceOnly θ F c (V.honest θ F c) ∧
      Game.prob (fun o => V.Verdict k .inferenceOnly o ∧ ¬ V.Holds k θ F c (V.honest θ F c) .inferenceOnly o)
        (V.game θ F c) (V.honest θ F c) = 0 :=
  V.sound_of_honest k .inferenceOnly h θ F c hreg

theorem spotSound_of_nontrivial (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Game.Strategy (V.game θ F c))
    (hfail : ∀ o, ¬ V.Holds k θ F c σ .sampled o) (hlt : V.error .sampled θ F c σ < 1) :
    ¬ V.TargetFor (fun _ => True) k .sampled :=
  V.sound_of_nontrivial k .sampled θ F c σ hfail hlt

theorem pouwSound_of_nontrivial (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Game.Strategy (V.game θ F c))
    (hfail : ∀ o, ¬ V.Holds k θ F c σ .pouw o) (hlt : V.error .pouw θ F c σ < 1) :
    ¬ V.TargetFor (fun _ => True) k .pouw :=
  V.sound_of_nontrivial k .pouw θ F c σ hfail hlt

theorem inferenceOnlySound_of_nontrivial (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Game.Strategy (V.game θ F c))
    (hfail : ∀ o, ¬ V.Holds k θ F c σ .inferenceOnly o) (hlt : V.error .inferenceOnly θ F c σ < 1) :
    ¬ V.TargetFor (fun _ => True) k .inferenceOnly :=
  V.sound_of_nontrivial k .inferenceOnly θ F c σ hfail hlt

end

end Vocab

/-- **Soundness at every composite, from its subprotocols' bounds at their components**: each bound gives its
subprotocol's target (`Vocab.sound_of`), and the targets the union bound over the composite's subprotocols
(`soundness_at_of_targets`). -/
theorem soundness_at_of_componentBounds : SoundnessOfComponents := by
  intro ι α _ d V k h
  exact soundness_at_of_targets ι α d V k fun a ha => V.sound_of k a (h a ha)

namespace Idle

theorem componentBound_unbounded (k : Verity.Registration.Composite) (a : Subprotocol) :
    unbounded.ComponentBound k a := by
  intro θ F c σ
  rw [error_unbounded]
  exact le_top

end Idle

/-- **`SoundnessOfComponents`' honest lemma**: its hypotheses hold at the idle deployment's compiled remote composite,
SPOT, PoUW and the inference-only policy each meeting its bound at its component. -/
theorem soundness_at_of_componentBounds_honest :
    Idle.unbounded.ComponentBound .remote .sampled ∧ Idle.unbounded.ComponentBound .remote .pouw ∧
      Idle.unbounded.ComponentBound .remote .inferenceOnly :=
  ⟨Idle.componentBound_unbounded _ _, Idle.componentBound_unbounded _ _, Idle.componentBound_unbounded _ _⟩

/-- **`SoundnessOfComponents`' nontrivial lemma**: its conclusion fails at remote for a verifier that accepts
everything, at the idle deployment whose errors are all `0`. -/
theorem soundness_at_of_componentBounds_nontrivial : ¬ Idle.exact.SoundnessFor .remote fun _ => True :=
  Idle.nontrivial_at .remote (Verity.mem_subprotocols.2 (by decide))

end Verity
