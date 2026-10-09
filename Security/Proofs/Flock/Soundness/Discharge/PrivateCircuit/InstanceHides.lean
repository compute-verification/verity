import Proofs.Flock.Soundness.Discharge.PrivateCircuit.InstanceDefs
import Proofs.Flock.Soundness.Discharge.PrivateCircuit.Hides
import Proofs.Flock.Soundness.ZK.RealLeaves

/-!
# The hm96 registration hides the program

`hm96Reg_hides`: under `Hm96Hiding My saltHash δ₁` (`hash-derived-key` at `δ₁ = 2^-193`; at `δ₁` the hiding gap it holds
outright, `CROnly.hm96Hiding_gap`), the commit strings of an `L`-row program at uniform salts are within `L·δ₁` of `L`
ideal commit strings, which read no program (`ZK.ideal_leaves_swap_fintype`), and those within `L·δ₁` of any other
program's (`ZK.real_leaves_swap_fintype`): the registration hides up to `2·L·δ₁`.
-/

namespace FlockSoundness.Discharge.PrivateCircuit

open Flock FlockSoundness.Game FlockSoundness.Discharge.ZkView
open scoped ENNReal

theorem prCoin_snd_unit {α : Type} [Fintype α] (f : α → Prop) : prCoin (fun q : Unit × α => f q.2) = prCoin f :=
  ZK.prCoin_congr_equiv (Equiv.punitProd α) f

/-- **The hm96 registration hides the program** up to `2·L·δ₁`, under `Hm96Hiding My saltHash δ₁`. -/
theorem hm96Reg_hides {Prg : Type} {L : ℕ} (x : Prg → Fin L → Bits) {δ₁ : ℝ≥0∞}
    (hKey : Assumptions.Hm96Hiding My saltHash δ₁) : Hides (hm96Reg x) (2 * L * δ₁) := by
  intro p p' E
  have h1 := ZK.ideal_leaves_swap_fintype My saltHash hKey id (Ω := Unit) (fun _ => x p) (fun _ v => E v)
  have h2 := ZK.real_leaves_swap_fintype My saltHash hKey id (Ω := Unit) (fun _ => x p') (fun _ v => E v)
  rw [prCoin_snd_unit (fun r : Fin L → Salt => E fun i => id (x p i + My (r i), saltHash (r i)))] at h1
  rw [prCoin_snd_unit (fun r : Fin L → Salt => E fun i => id (x p' i + My (r i), saltHash (r i)))] at h2
  simp only [Fintype.card_fin, id] at h1 h2
  calc prCoin (fun r => E (hm96Reg x p r))
      ≤ prCoin (fun q : Unit × (Fin L → Bits × Salt) => E fun i => ((q.2 i).1, saltHash (q.2 i).2)) + L * δ₁ := h1
    _ ≤ prCoin (fun r => E (hm96Reg x p' r)) + L * δ₁ + L * δ₁ := add_le_add h2 le_rfl
    _ = prCoin (fun r => E (hm96Reg x p' r)) + 2 * L * δ₁ := by ring

end FlockSoundness.Discharge.PrivateCircuit
