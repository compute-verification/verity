import Definitions.Core.Blocks
import Proofs.Core.Game.Basic

/-!
# A wipe that meets the boundary's definitions

`Wipe.blank` clears the device's whole state, sends nothing out and always accepts. It separates
(`Wipe.blank_separates`), and meets the survivor bound at the constant `2^-128` for every `foreignBits ≥ 128`
(`Wipe.blank_survivorBound`), so `Wipe.Separates` and `Wipe.SurvivorBound` hold together on some wipe of every device.
-/

namespace Verity

open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {C : Circuit} {B : ℕ} {s : Blocks C B} {b : Fin B} {Ω Ch : Type}

/-- The wipe that clears the device's state: the fill covers the whole store and leaves every bit of the device's state
false, nothing leaves the device, the challenger always accepts, and the spot reads change nothing. -/
def Wipe.blank (d : Accounting.Device ι) (s : Blocks C B) (b : Fin B) (Ω Ch : Type) : Wipe d s b Ω Ch where
  fill := d.mem
  fill_le := le_rfl
  egress := 0
  carry _ _ := 0
  filled _ _ := (⟨fun _ => false, fun _ => false⟩, fun _ => false)
  accepts _ _ _ := True
  after σ _ _ := σ

theorem Wipe.blank_separates {Y : Type} :
    (Wipe.blank d s b Ω Ch).Separates fun σ ω c (_ : Y) => (Wipe.blank d s b Ω Ch).start σ ω c :=
  ⟨fun st _ _ => st, fun _ _ _ _ => rfl⟩

theorem Wipe.blank_survivorBound [Fintype Ω] [Nonempty Ω] [Fintype Ch] [Nonempty Ch] {foreignBits : ℕ}
    (h : 128 ≤ foreignBits) : (Wipe.blank d s b Ω Ch).SurvivorBound foreignBits fun _ _ _ => 2⁻¹ ^ 128 := by
  classical
  intro S hS hold keep
  show _ ≤ (2⁻¹ : ℝ≥0∞) ^ 128
  have hS' : 128 ≤ S := by omega
  set P : (Fin S → Bool) × Ω × Ch → Prop := fun x =>
    (Wipe.blank d s b Ω Ch).accepts (hold x.1) x.2.1 x.2.2 ∧
      keep ((Wipe.blank d s b Ω Ch).filled (hold x.1) x.2.1).1 ((Wipe.blank d s b Ω Ch).filled (hold x.1) x.2.1).2
        x.2.2 = x.1
  have hcard : (Finset.univ.filter P).card ≤ Fintype.card (Ω × Ch) := by
    rw [← Finset.card_univ]
    refine Finset.card_le_card_of_injOn Prod.snd (fun _ _ => Finset.mem_univ _) ?_
    rintro ⟨z, ω, c⟩ hx ⟨z', ω', c'⟩ hy hxy
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj hxy
    have hx := (Finset.mem_filter.1 hx).2.2
    have hy := (Finset.mem_filter.1 hy).2.2
    simp only [Wipe.blank] at hx hy
    rw [← hx, ← hy]
  have hm : (Fintype.card (Ω × Ch) : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have hX : (Fintype.card ((Fin S → Bool) × Ω × Ch) : ℝ≥0∞) = 2 ^ S * Fintype.card (Ω × Ch) := by
    simp [Fintype.card_prod]
  rw [show Game.draw ((Fin S → Bool) × Ω × Ch) = .coin _ fun x => .ret (id x) from rfl, Game.value_coin_ret,
    Game.prCoin_eq_card]
  calc ((Finset.univ.filter P).card : ℝ≥0∞) / Fintype.card ((Fin S → Bool) × Ω × Ch)
      ≤ 1 * Fintype.card (Ω × Ch) / (2 ^ S * Fintype.card (Ω × Ch)) := by
        rw [one_mul, hX]
        gcongr
    _ = (2 ^ S)⁻¹ := by rw [ENNReal.mul_div_mul_right _ _ hm (ENNReal.natCast_ne_top _), one_div]
    _ = 2⁻¹ ^ S := ENNReal.inv_pow
    _ ≤ 2⁻¹ ^ 128 := pow_le_pow_right_of_le_one' (by norm_num) hS'

end Verity
