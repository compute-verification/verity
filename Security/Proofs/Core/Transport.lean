import Mathlib.Data.Fintype.Prod
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Definitions.Core.Guarantees.Transport

/-! # Proofs of the transport's guarantees, each with an honest instance and one that fails without its premise -/

namespace Verity.Transport

open Classical

theorem pr_snd {α β : Type} [Fintype α] [Fintype β] [Nonempty α] (P : β → Prop) :
    Pr (fun x : α × β => P x.2) = Pr P := by
  unfold Pr
  have hα : (Fintype.card α : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr Fintype.card_ne_zero
  have h : (Finset.univ.filter fun x : α × β => P x.2) = Finset.univ ×ˢ Finset.univ.filter P := by
    ext x
    simp
  rw [h, Finset.card_product, Finset.card_univ, Fintype.card_prod, Nat.cast_mul, Nat.cast_mul,
    mul_div_mul_left _ _ hα]

theorem pr_comp_equiv {α β : Type} [Fintype α] [Fintype β] (e : α ≃ β) (P : β → Prop) :
    Pr (fun a => P (e a)) = Pr P := by
  unfold Pr
  have h := Fintype.card_congr (e.subtypeEquiv fun _ => Iff.rfl : {a // P (e a)} ≃ {b // P b})
  rw [Fintype.card_subtype, Fintype.card_subtype] at h
  rw [h, Fintype.card_congr e]

theorem seen_none {Node W : Type} {C reads : Node → Prop} (h : ¬ ∃ o, C o ∧ reads o) (w : W) :
    seen C reads w = none := by
  simp [seen, h]

theorem seen_some {Node W : Type} {C reads : Node → Prop} (h : ∃ o, C o ∧ reads o) (w : W) :
    seen C reads w = some w := by
  simp [seen, h]

theorem not_read_of_pathPrivate {Node : Type} {s : Site Node} {party : Node} {reads : Node → Prop} (C : Node → Prop)
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
  intro Node Ω M W Rec _ _ s c C wire record fill η m hη hprem E
  unfold ViewIsRecord
  rw [pr_snd (fun ω' => E (seen (outside s c.party C) c.readsPrivate (wire (fill (record m)) ω')))]
  by_cases hr : ∃ o, outside s c.party C o ∧ c.readsPrivate o
  · rcases hprem with hp | hh
    · exact absurd hr (not_read_of_pathPrivate C hp)
    · simp only [seen_some hr]
      exact hh (fun w => E (some w))
  · simp only [seen_none hr, sub_self, abs_zero]
    exact hη

theorem ImprintRootHidden : Verity.Guarantees.ImprintRootHidden := by
  intro Node R K _ _ s c C root hp E
  unfold ViewIsRecord
  rw [pr_snd (fun x' : R × R => E ((none : Option (R × R)), root x'.1 x'.2))]
  simp only [seen_none (not_read_of_pathPrivate C hp), sub_self, abs_zero, le_refl]

theorem TransportDelivers : Verity.Guarantees.TransportDelivers := by
  intro Run Frame Body sent accepted forgery body h r
  exact (h r).imp (fun hp => hp.map body) id

end Verity.SecurityProofs

namespace Verity.Transport.Instances

open Verity.Transport Verity.Assumptions

/-- The cable: its two ends are the only readers, the auditor device and the prover's server. -/
def cable : Channel Bool where
  party := false
  readsPrivate := fun o => o = false ∨ o = true
  readsImprint := fun o => o = false ∨ o = true

/-- The cable's site: the server (`true`) is the prover's, and no other party is on it. -/
def cableSite : Site Bool where
  developer := fun o => o = true
  Confined := fun _ => False

theorem cable_witnessPathPrivate : WitnessPathPrivate cableSite cable := by
  intro o _
  cases o <;> simp [cable, cableSite]

theorem cable_imprintPathPrivate : ImprintPathPrivate cableSite cable := by
  intro o _
  cases o <;> simp [cable, cableSite]

/-- A one-bit pad hides one bit against every test, at `η = 0`. -/
theorem pad_hides (E : Bool → Prop) (m : Bool) :
    ChannelHides (fun (m k : Bool) => xor m k) (fun _ => false) E 0 m := by
  unfold ChannelHides
  cases m
  · simp
  · have h : (fun k => E (xor true k)) = fun k => E (Function.Involutive.toPerm _ Bool.not_not k) := by
      funext k
      cases k <;> rfl
    rw [h, pr_comp_equiv]
    simp

/-- A tapped cable: the device (`0`) and the prover's server (`1`), and a third node (`2`) that reads it too. -/
def tapped : Channel (Fin 3) where
  party := 0
  readsPrivate := fun _ => True
  readsImprint := fun _ => True

/-- The tapped cable's site: the server is the prover's, and no party is `Confined`. -/
def tappedSite : Site (Fin 3) where
  developer := fun o => o = 1
  Confined := fun _ => False

theorem tap_outside : outside tappedSite tapped.party (fun o => o = 2) 2 :=
  ⟨rfl, by decide, by simp [tappedSite], by simp [tappedSite]⟩

theorem tap_sees {W : Type} (w : W) :
    seen (outside tappedSite tapped.party (fun o => o = 2)) tapped.readsPrivate w = some w ∧
      seen (outside tappedSite tapped.party (fun o => o = 2)) tapped.readsImprint w = some w :=
  ⟨seen_some ⟨2, tap_outside, trivial⟩ w, seen_some ⟨2, tap_outside, trivial⟩ w⟩

theorem tapped_not_private : ¬ WitnessPathPrivate tappedSite tapped ∧ ¬ ImprintPathPrivate tappedSite tapped := by
  constructor <;> intro h <;> rcases h 2 trivial with h | h | h <;> simp_all [tapped, tappedSite]

/-- `TransportPassiveZk`'s conclusion on the tapped cable carrying its bit in the clear (`wire m ω = m`, an empty
record, blank traffic `false`), for the bit `true`: the tap sees it, so no simulator matches at any `η < 1`. -/
theorem clear_fails {η : ℝ} (hη : η < 1) :
    ¬ ViewIsRecord (Ω := Unit) (Ω' := Unit)
      (fun _ => seen (outside tappedSite tapped.party (fun o => o = 2)) tapped.readsPrivate true) (fun _ => ())
      (fun _ _ => seen (outside tappedSite tapped.party (fun o => o = 2)) tapped.readsPrivate false)
      (fun v => v = some true) η := by
  simp only [ViewIsRecord, (tap_sees _).1, Pr]
  norm_num
  linarith

/-- `ImprintRootHidden`'s conclusion on the tapped cable: the tap sees the exchange and so the root it derives, and no
fresh root matches it. -/
theorem imprint_read_fails :
    ¬ ViewIsRecord (Ω := Bool × Bool) (Ω' := Bool × Bool)
      (fun x => (seen (outside tappedSite tapped.party (fun o => o = 2)) tapped.readsImprint x, xor x.1 x.2))
      (fun _ => ()) (fun _ x' => ((none : Option (Bool × Bool)), xor x'.1 x'.2))
      (fun v => ∃ x : Bool × Bool, v.1 = some x ∧ v.2 = xor x.1 x.2) 0 := by
  simp only [ViewIsRecord, (tap_sees _).2, Pr]
  norm_num

/-- A receiver that accepts a frame its peer never sent, in a run with no forgery, delivers it. -/
theorem injected_fails :
    ¬ ((([true] : List Bool).map id) <+: (([] : List Bool).map id) ∨ False) := by
  simp

end Verity.Transport.Instances
