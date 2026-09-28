import Proofs.Flock.Soundness.Discharge.PrivateCircuit.Statements

/-!
# The session ran the registered program, under `cr/sha-512`

`ran_registered`: the union of protocol 1's bad event and a program other than the registered one; on the second, the
session's opening and the registrant's are two openings of one root to two programs, so the extractor's pair collides,
which `SHA512CRStrict` bounds by `q²/2^513`.
-/

namespace FlockSoundness.Discharge.PrivateCircuit

open FlockSoundness.Game
open scoped ENNReal

theorem ran_registered : RanRegistered := by
  intro Root P Op com hbind rt p op₀ h₀ β G σ acc Q prog op hacc B hB cost q hcost hCR
  have hle : prob (fun b => acc b ∧ prog b ≠ p) G σ ≤
      prob (fun b => ∃ x y, some (com.ext (op b) op₀) = some (x, y) ∧ x ≠ y ∧ Refine.H512 x = Refine.H512 y) G σ :=
    prob_mono (fun b ⟨ha, hne⟩ =>
      have hc := hbind rt _ _ _ _ (hacc b ha) h₀ hne
      ⟨_, _, congrArg some Prod.mk.eta.symm, hc.1, hc.2⟩) G σ
  have hfin : prob (fun b => ∃ x y, some (com.ext (op b) op₀) = some (x, y) ∧ x ≠ y ∧ Refine.H512 x = Refine.H512 y)
      G σ ≠ ⊤ := ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _)
  have hcoll := hle.trans ((ENNReal.le_ofReal_iff_toReal_le hfin (by positivity)).2 (hCR hcost))
  calc prob (fun b => acc b ∧ (Q b ∨ prog b ≠ p)) G σ
      ≤ prob (fun b => (acc b ∧ Q b) ∨ (acc b ∧ prog b ≠ p)) G σ :=
        prob_mono (fun _ ⟨ha, h⟩ => h.imp (⟨ha, ·⟩) (⟨ha, ·⟩)) G σ
    _ ≤ prob (fun b => acc b ∧ Q b) G σ + prob (fun b => acc b ∧ prog b ≠ p) G σ := prob_or_le _ _ G σ
    _ ≤ B + ENNReal.ofReal (q ^ 2 / 2 ^ (513 : ℝ)) := add_le_add hB hcoll

end FlockSoundness.Discharge.PrivateCircuit
