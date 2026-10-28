import Definitions.Core.Blocks
import Proofs.Core.Game.Prob

/-!
# A wipe that meets the boundary's definitions, and its survivor bound on a run

`Wipe.blank` clears the device's whole state, sends nothing out and always accepts. It separates
(`Wipe.blank_separates`), and meets the survivor bound at the constant `2^-128` for every `foreignBits ≥ 128`
(`Wipe.blank_survivorBound`), so `Wipe.Separates` and `Wipe.SurvivorBound` hold together on some wipe of every device.

On a run whose view at the boundary is distributed as the wipe's game draws it (`Wipe.OfRun.Preserves`), the run's event
that the wipe accepts while foreign content past the survivor threshold survives (`Wipe.OfRun.Kept`) is the preimage of
the wipe's game's, so the survivor bound bounds it at the run's moves (`Wipe.OfRun.prob_kept_le`). A run that, after
any game, draws the foreign content, the seed and the spot reads at the boundary meets `Preserves` at every strategy,
for every wipe (`Wipe.OfRun.drawn_preserves`); on it the blank wipe's event has probability at most `2^-128`
(`Wipe.OfRun.blank_drawn`).
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

/-- **The survivor bound on a run.** If the wipe `w` meets its survivor bound at `foreignBits` and `ε`, and the run's
view at the boundary is distributed as the wipe's game draws it at the strategy `t`, then on the run the probability
that the wipe accepts while foreign content past the survivor threshold survives is at most `ε` at the run's moves. -/
theorem Wipe.OfRun.prob_kept_le [Fintype Ω] [Nonempty Ω] [Fintype Ch] [Nonempty Ch] {w : Wipe d s b Ω Ch}
    {foreignBits : ℕ}
    {ε : (S_φ : ℕ) → ((Fin S_φ → Bool) → Device.State d) →
      (Device.State d → (Fin w.egress → Bool) → Ch → Fin S_φ → Bool) → ℝ≥0∞}
    (hw : w.SurvivorBound foreignBits ε) {O : Type} {G : Game O} (R : w.OfRun G) (t : Game.Strategy G)
    (hR : R.Preserves t) :
    Game.prob (fun o => R.accepts o ∧ R.Kept foreignBits o) G t ≤ ε R.S_φ R.hold R.keep := by
  refine (hR fun x => w.accepts (R.hold x.1) x.2.1 x.2.2 ∧ (foreignBits + w.unfilled ≤ R.S_φ ∧
    R.keep (w.filled (R.hold x.1) x.2.1).1 (w.filled (R.hold x.1) x.2.1).2 x.2.2 = x.1)).le.trans ?_
  by_cases hS : foreignBits + w.unfilled ≤ R.S_φ
  · refine (Game.value_mono ?_ _).trans (hw R.S_φ hS R.hold R.keep)
    rintro x ⟨ha, -, hk⟩
    exact ⟨ha, hk⟩
  · calc _ ≤ Game.value (fun _ => False) (Game.draw ((Fin R.S_φ → Bool) × Ω × Ch)) := by
          refine Game.value_mono ?_ _
          rintro x ⟨-, h1, -⟩
          exact hS h1
      _ = 0 := Game.value_false _
      _ ≤ _ := zero_le

/-- A continuation whose event has the same probability `V` from every output, whatever the prover does, gives the
composition probability `V`. -/
theorem Wipe.OfRun.prob_bind_eq {α β : Type} (f : α → Game β) (p : β → Prop) (V : ℝ≥0∞)
    (hf : ∀ a t, Game.prob p (f a) t = V) :
    ∀ (g : Game α) (t : Game.Strategy (g.bind f)), Game.prob p (g.bind f) t = V
  | .ret a, t => hf a t
  | .send _ k, t => Wipe.OfRun.prob_bind_eq f p V hf (k t.1) t.2
  | @Game.coin _ _ _ _ k, t =>
      (congrArg Game.avg (funext fun c => Wipe.OfRun.prob_bind_eq f p V hf (k c) (t c))).trans (Game.avg_const V)

/-- **The run that draws its foreign content with the wipe's coins**: after the game `g`, whatever was played there,
the run draws the foreign content of `S_φ` bits, the seed and the spot reads, and the view reads them; `hold` and `keep`
are the run's moves at the boundary. -/
def Wipe.OfRun.drawn [Fintype Ω] [Nonempty Ω] [Fintype Ch] [Nonempty Ch] (w : Wipe d s b Ω Ch) {O₀ : Type}
    (g : Game O₀) (S_φ : ℕ) (hold : (Fin S_φ → Bool) → Device.State d)
    (keep : Device.State d → (Fin w.egress → Bool) → Ch → Fin S_φ → Bool) :
    w.OfRun (g.bind fun o₀ => (Game.draw ((Fin S_φ → Bool) × Ω × Ch)).map (Prod.mk o₀)) where
  S_φ := S_φ
  hold := hold
  keep := keep
  view := Prod.snd

/-- **The map exists at every wipe** on a run that draws its foreign content with the wipe's coins: its view is
distributed as the wipe's game draws it at every strategy. -/
theorem Wipe.OfRun.drawn_preserves [Fintype Ω] [Nonempty Ω] [Fintype Ch] [Nonempty Ch] (w : Wipe d s b Ω Ch)
    {O₀ : Type} (g : Game O₀) (S_φ : ℕ) (hold : (Fin S_φ → Bool) → Device.State d)
    (keep : Device.State d → (Fin w.egress → Bool) → Ch → Fin S_φ → Bool)
    (t : Game.Strategy (g.bind fun o₀ => (Game.draw ((Fin S_φ → Bool) × Ω × Ch)).map (Prod.mk o₀))) :
    (Wipe.OfRun.drawn w g S_φ hold keep).Preserves t := fun P =>
  Wipe.OfRun.prob_bind_eq _ (fun o => P o.2) (Game.value P (Game.draw _)) (fun _ _ => rfl) g t

/-- **The blank wipe on the run that draws its foreign content**: for every `foreignBits ≥ 128`, the run's event that
the wipe accepts while foreign content past the survivor threshold survives has probability at most `2^-128`, so the
premises of a statement over runs (`Wipe.SurvivorBound` and `Wipe.OfRun.Preserves`) hold together. -/
theorem Wipe.OfRun.blank_drawn [Fintype Ω] [Nonempty Ω] [Fintype Ch] [Nonempty Ch] {foreignBits : ℕ}
    (h : 128 ≤ foreignBits) {O₀ : Type} (g : Game O₀) (S_φ : ℕ) (hold : (Fin S_φ → Bool) → Device.State d)
    (keep : Device.State d → (Fin (Wipe.blank d s b Ω Ch).egress → Bool) → Ch → Fin S_φ → Bool)
    (t : Game.Strategy (g.bind fun o₀ => (Game.draw ((Fin S_φ → Bool) × Ω × Ch)).map (Prod.mk o₀))) :
    Game.prob (fun o => (Wipe.OfRun.drawn (Wipe.blank d s b Ω Ch) g S_φ hold keep).accepts o ∧
        (Wipe.OfRun.drawn (Wipe.blank d s b Ω Ch) g S_φ hold keep).Kept foreignBits o) _ t ≤ 2⁻¹ ^ 128 :=
  Wipe.OfRun.prob_kept_le (Wipe.blank_survivorBound h) _ t (Wipe.OfRun.drawn_preserves _ g S_φ hold keep t)

end Verity
