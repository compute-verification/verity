import Mathlib.Data.Fintype.Prod
import Mathlib.Algebra.BigOperators.Fin
import Definitions.Core.Guarantees.Transport

/-! # Proofs of the transport's guarantees, each with an honest instance and one that fails without its premise -/

namespace Verity.Transport

open Classical
open scoped ENNReal

theorem close_refl (a η : ℝ≥0∞) : Close a a η :=
  ⟨le_self_add, le_self_add⟩

theorem sum_inv_card {α : Type} [Fintype α] [Nonempty α] (x : ℝ≥0∞) :
    ∑ _a : α, (Fintype.card α : ℝ≥0∞)⁻¹ * x = x := by
  rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, ← mul_assoc,
    ENNReal.mul_inv_cancel (by exact_mod_cast Fintype.card_ne_zero) (ENNReal.natCast_ne_top _), one_mul]

theorem value_map_draw {α β : Type} [Fintype α] [Nonempty α] (p : β → Prop) (f : α → β) :
    Game.value p ((Game.draw α).map f) = ∑ a, (Fintype.card α : ℝ≥0∞)⁻¹ * if p (f a) then 1 else 0 :=
  rfl

theorem value_bind_draw {α β : Type} [Fintype α] [Nonempty α] (p : β → Prop) (g : Game β) :
    Game.value p ((Game.draw α).bind fun _ => g) = Game.value p g := by
  show ∑ _a : α, (Fintype.card α : ℝ≥0∞)⁻¹ * Game.value p g = _
  exact sum_inv_card _

theorem value_map_draw_true {α β : Type} [Fintype α] [Nonempty α] {p : β → Prop} {f : α → β} (h : ∀ a, p (f a)) :
    Game.value p ((Game.draw α).map f) = 1 := by
  rw [value_map_draw]
  simpa [h] using sum_inv_card (α := α) (1 : ℝ≥0∞)

theorem value_map_draw_false {α β : Type} [Fintype α] [Nonempty α] {p : β → Prop} {f : α → β} (h : ∀ a, ¬ p (f a)) :
    Game.value p ((Game.draw α).map f) = 0 := by
  simp [value_map_draw, h]

theorem seen_none {Node W : Type} {C reads : Node → Prop} (h : ¬ ∃ o, C o ∧ reads o) (w : W) :
    seen C reads w = none := by
  simp [seen, h]

theorem seen_some {Node W : Type} {C reads : Node → Prop} (h : ∃ o, C o ∧ reads o) (w : W) :
    seen C reads w = some w := by
  simp [seen, h]

theorem not_read_of_pathPrivate {Node : Type} {s : Roles Node} {party : Node} {reads : Node → Prop} (C : Node → Prop)
    (hp : PathPrivate s party reads) : ¬ ∃ o, outside s party C o ∧ reads o := by
  rintro ⟨o, ⟨_, hne, hd, hconf⟩, hr⟩
  rcases hp o hr with h | h | h
  · exact hne h
  · exact hd h
  · exact hconf h

end Verity.Transport

namespace Verity.SecurityProofs

open Verity.Transport Verity.Assumptions

theorem TransportPassiveZk : Verity.Guarantees.TransportPassiveZk := by
  intro Node Ω M W Rec _ _ s c C wire record fill η m hprem E
  unfold ViewIsRecord
  rw [value_bind_draw]
  by_cases hr : ∃ o, outside s c.party C o ∧ c.readsPrivate o
  · rcases hprem with hp | hh
    · exact absurd hr (not_read_of_pathPrivate C hp)
    · have h := hh (fun w => E (some w))
      simp only [ChannelHides, value_map_draw] at h
      simpa only [seen_some hr, value_map_draw] using h
  · simp only [seen_none hr]
    exact close_refl _ _

theorem ImprintRootHidden : Verity.Guarantees.ImprintRootHidden := by
  intro Node R K _ _ s c C root hp E
  unfold ViewIsRecord
  rw [value_bind_draw]
  simp only [seen_none (not_read_of_pathPrivate C hp)]
  exact close_refl _ _

theorem TransportDelivers : Verity.Guarantees.TransportDelivers := by
  intro Run Frame Body sent accepted forgery body h r
  exact (h r).imp (fun hp => hp.map body) id

end Verity.SecurityProofs

namespace Verity.Transport.Instances

open Verity.Transport Verity.Assumptions
open scoped ENNReal

/-- The cable: its two ends are the only readers, the auditor device and the prover's server. -/
def cable : Channel Bool where
  party := false
  readsPrivate := fun o => o = false ∨ o = true
  readsImprint := fun o => o = false ∨ o = true

/-- The cable's roles: the server (`true`) is the prover's, and no other party is on it. -/
def cableRoles : Roles Bool where
  developer := fun o => o = true
  Confined := fun _ => False

theorem cable_witnessPathPrivate : WitnessPathPrivate cableRoles cable := by
  intro o _
  cases o <;> simp [cable, cableRoles]

theorem cable_imprintPathPrivate : ImprintPathPrivate cableRoles cable := by
  intro o _
  cases o <;> simp [cable, cableRoles]

/-- A one-bit pad hides one bit against every test, at `η = 0`. -/
theorem pad_hides (E : Bool → Prop) (m : Bool) :
    ChannelHides (fun (m k : Bool) => xor m k) (fun _ => false) E 0 m := by
  unfold ChannelHides
  rw [value_map_draw, value_map_draw]
  cases m
  · exact close_refl _ _
  · rw [Fintype.sum_bool, Fintype.sum_bool, add_comm]
    exact close_refl _ _

/-- A tapped cable: the device (`0`) and the prover's server (`1`), and a third node (`2`) that reads it too. -/
def tapped : Channel (Fin 3) where
  party := 0
  readsPrivate := fun _ => True
  readsImprint := fun _ => True

/-- The tapped cable's roles: the server is the prover's, and no party is `Confined`. -/
def tappedRoles : Roles (Fin 3) where
  developer := fun o => o = 1
  Confined := fun _ => False

theorem tap_outside : outside tappedRoles tapped.party (fun o => o = 2) 2 :=
  ⟨rfl, by decide, by simp [tappedRoles], by simp [tappedRoles]⟩

theorem tap_sees {W : Type} (w : W) :
    seen (outside tappedRoles tapped.party (fun o => o = 2)) tapped.readsPrivate w = some w ∧
      seen (outside tappedRoles tapped.party (fun o => o = 2)) tapped.readsImprint w = some w :=
  ⟨seen_some ⟨2, tap_outside, trivial⟩ w, seen_some ⟨2, tap_outside, trivial⟩ w⟩

theorem tapped_not_private : ¬ WitnessPathPrivate tappedRoles tapped ∧ ¬ ImprintPathPrivate tappedRoles tapped := by
  constructor <;> intro h <;> rcases h 2 trivial with h | h | h <;> simp_all [tapped, tappedRoles]

/-- `TransportPassiveZk`'s conclusion on the tapped cable carrying its bit in the clear (`wire m ω = m`, an empty
record, blank traffic `false`), for the bit `true`: the tap sees it, so no simulator matches at any `η < 1`. -/
theorem clear_fails {η : ℝ≥0∞} (hη : η < 1) :
    ¬ ViewIsRecord (Ω := Unit) (Ω' := Unit)
      (fun _ => seen (outside tappedRoles tapped.party (fun o => o = 2)) tapped.readsPrivate true) (fun _ => ())
      (fun _ _ => seen (outside tappedRoles tapped.party (fun o => o = 2)) tapped.readsPrivate false)
      (fun v => v = some true) η := by
  rintro ⟨h, -⟩
  rw [value_bind_draw, value_map_draw_true (fun _ => (tap_sees _).1),
    value_map_draw_false (fun _ => by simp [(tap_sees _).1]), zero_add] at h
  exact absurd h (not_le.mpr hη)

/-- `ImprintRootHidden`'s conclusion on the tapped cable: the tap sees the exchange and so the root it derives, and no
fresh root matches it. -/
theorem imprint_read_fails :
    ¬ ViewIsRecord (Ω := Bool × Bool) (Ω' := Bool × Bool)
      (fun x => (seen (outside tappedRoles tapped.party (fun o => o = 2)) tapped.readsImprint x, xor x.1 x.2))
      (fun _ => ()) (fun _ x' => ((none : Option (Bool × Bool)), xor x'.1 x'.2))
      (fun v => ∃ x : Bool × Bool, v.1 = some x ∧ v.2 = xor x.1 x.2) 0 := by
  rintro ⟨h, -⟩
  rw [value_bind_draw, value_map_draw_true (fun x => ⟨x, (tap_sees x).2, rfl⟩),
    value_map_draw_false (fun _ => by simp), zero_add] at h
  exact absurd h (by norm_num)

/-- A receiver that accepts a frame its peer never sent, in a run with no forgery, delivers it. -/
theorem injected_fails :
    ¬ ((([true] : List Bool).map id) <+: (([] : List Bool).map id) ∨ False) := by
  simp

end Verity.Transport.Instances
