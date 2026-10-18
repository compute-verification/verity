import Proofs.Population.Transfer
import Definitions.Population.AddressMap

/-!
# The converse: a corruptible address needs the padding's cones

`Transfer` gives a program every population theorem with the cones of its padding over the assignments holding the
public values. This file shows that nothing smaller works.

* `AddressMap.closure_expresses` and `AddressMap.mem_of_expresses`, **the padding is the least wiring that expresses
  every program with a given address map**: the closure decides every compatible program's correctness, and any static
  wiring that does contains every slot a unit reads in some assignment holding the public values, apart from the
  public slots and the unit's own writes (`closureReads_sub_of_expresses`). The unit function that witnesses it
  compares the slot it may read with its value in one assignment, so a wiring without that slot cannot tell a correct
  unit from a wrong one. `DynPopulation.addressMap_closureReads`: for a program, the closure is `padOver`'s read sets.
* `vm_not_addressedBy_inputs`: the pointer machine's read sets depend on more than all its inputs, since the pointer is
  a unit's write.
* `vm_pad_pub_secret_up`: in its padding over the assignments holding the public zero region, every secret word is
  upstream of the egress, so `card_egress_le_pad` says nothing about the secrets.
* `vm_honest_wiring_fails`, **the honest wiring is not enough**: in every honest run no secret is upstream of the
  egress, yet with the one-bit pointer step wrong the egress takes `2 ^ N` values, more than the `|V| ^ |writes B| = 2`
  the capacity theorem would give were the honest run's upstream the public one.
-/

namespace Population.AddressMap

open Finset

variable {U S V : Type} (A : AddressMap U S V)

theorem mem_closureReads [Fintype S] {Pub : Finset S} {a : S → V} {u : U} {s : S} :
    s ∈ A.closureReads Pub a u ↔ ∃ σ : S → V, (∀ t ∈ Pub, σ t = a t) ∧ s ∈ A.reads u σ := by
  classical
  simp only [closureReads, mem_filter, mem_univ, true_and]

/-- **The closure expresses the address map**: every compatible program's correctness, among the assignments holding
`a` on `Pub`, depends only on the closure's reads and the unit's writes. -/
theorem closure_expresses [Fintype S] (Pub : Finset S) (a : S → V) : A.Expresses Pub a (A.closureReads Pub a) := by
  intro f hf u σ τ hσ _ hR hW
  have hr : ∀ s ∈ A.reads u σ, σ s = τ s := fun s hs => hR s ((A.mem_closureReads).2 ⟨σ, hσ, hs⟩)
  constructor
  · intro h s hs
    rw [← hW s hs, h s hs]
    exact hf u σ τ hr s hs
  · intro h s hs
    rw [hW s hs, h s hs]
    exact (hf u σ τ hr s hs).symm

end Population.AddressMap

namespace Population.AddressMap

open Finset

variable {U S : Type} (A : AddressMap U S Bool)

/-- **Any wiring that expresses the address map wires every slot its units can read**: if unit `u` writes something and
reads `x` in some assignment holding `a` on `Pub`, with `x` neither public nor written by `u`, then `x ∈ R u`. -/
theorem mem_of_expresses [DecidableEq U] [DecidableEq S] {Pub : Finset S} {a : S → Bool} {R : U → Finset S}
    (hR : A.Expresses Pub a R) {u : U} {w x : S} (hw : w ∈ A.writes u) {σ0 : S → Bool}
    (h0 : ∀ t ∈ Pub, σ0 t = a t) (hx : x ∈ A.reads u σ0) (hxP : x ∉ Pub) (hxw : x ∉ A.writes u) : x ∈ R u := by
  by_contra hxR
  have hcond : ∀ σ τ : S → Bool, (∀ s ∈ A.reads u σ, σ s = τ s) →
      ((x ∈ A.reads u σ ∧ σ x = σ0 x) ↔ (x ∈ A.reads u τ ∧ τ x = σ0 x)) := by
    intro σ τ h
    rw [← A.self u σ τ h]
    constructor
    · rintro ⟨hm, he⟩
      exact ⟨hm, (h x hm).symm.trans he⟩
    · rintro ⟨hm, he⟩
      exact ⟨hm, (h x hm).trans he⟩
  let g : (S → Bool) → Bool := fun σ => if x ∈ A.reads u σ ∧ σ x = σ0 x then σ0 w else !σ0 w
  let f : U → (S → Bool) → S → Bool := fun u' σ s => if u' = u ∧ s = w then g σ else σ0 s
  have hf : A.Compatible f := by
    intro u' σ τ h s _
    by_cases hu : u' = u ∧ s = w
    · rw [hu.1] at h
      show (if u' = u ∧ s = w then g σ else σ0 s) = (if u' = u ∧ s = w then g τ else σ0 s)
      rw [if_pos hu, if_pos hu]
      exact if_congr (hcond σ τ h) rfl rfl
    · show (if u' = u ∧ s = w then g σ else σ0 s) = (if u' = u ∧ s = w then g τ else σ0 s)
      rw [if_neg hu, if_neg hu]
  let τ0 : S → Bool := Function.update σ0 x (!σ0 x)
  have hτP : ∀ t ∈ Pub, τ0 t = a t := fun t ht => by
    have hne : t ≠ x := fun e => hxP (e ▸ ht)
    simp [τ0, hne, h0 t ht]
  have hsame : ∀ s ∈ R u, σ0 s = τ0 s := fun s hs => by
    have hne : s ≠ x := fun e => hxR (e ▸ hs)
    simp [τ0, hne]
  have hsamew : ∀ s ∈ A.writes u, σ0 s = τ0 s := fun s hs => by
    have hne : s ≠ x := fun e => hxw (e ▸ hs)
    simp [τ0, hne]
  have hc0 : A.Correct f σ0 u := by
    intro s _
    show σ0 s = (if u = u ∧ s = w then g σ0 else σ0 s)
    by_cases e : s = w
    · rw [if_pos (show u = u ∧ s = w from ⟨rfl, e⟩), e]
      show σ0 w = (if x ∈ A.reads u σ0 ∧ σ0 x = σ0 x then σ0 w else !σ0 w)
      rw [if_pos (show x ∈ A.reads u σ0 ∧ σ0 x = σ0 x from ⟨hx, rfl⟩)]
    · rw [if_neg (show ¬ (u = u ∧ s = w) from fun h => e h.2)]
  have hc1 : ¬ A.Correct f τ0 u := by
    intro hc
    have h1 : τ0 w = f u τ0 w := hc w hw
    have hwx : w ≠ x := fun e => hxw (e ▸ hw)
    have hτw : τ0 w = σ0 w := by simp [τ0, hwx]
    have hτx : τ0 x = !σ0 x := by simp [τ0]
    have hfw : f u τ0 w = !σ0 w := by
      show (if u = u ∧ w = w then g τ0 else σ0 w) = !σ0 w
      rw [if_pos (show u = u ∧ w = w from ⟨rfl, rfl⟩)]
      show (if x ∈ A.reads u τ0 ∧ τ0 x = σ0 x then σ0 w else !σ0 w) = !σ0 w
      refine if_neg (show ¬ (x ∈ A.reads u τ0 ∧ τ0 x = σ0 x) from fun h => ?_)
      have h2 := h.2
      rw [hτx] at h2
      cases hb : σ0 x <;> simp [hb] at h2
    rw [hτw, hfw] at h1
    cases hb : σ0 w <;> simp [hb] at h1
  exact hc1 ((hR f hf u σ0 τ0 h0 hτP hsame hsamew).1 hc0)

/-- **The closure is the least expressing wiring**: every slot of the closure is wired by any expressing `R`, public,
or the unit's own write. -/
theorem closureReads_sub_of_expresses [Fintype S] [DecidableEq U] [DecidableEq S] {Pub : Finset S}
    {a : S → Bool} {R : U → Finset S} (hR : A.Expresses Pub a R) {u : U} (hw : (A.writes u).Nonempty) :
    ∀ x ∈ A.closureReads Pub a u, x ∈ R u ∨ x ∈ Pub ∨ x ∈ A.writes u := by
  intro x hx
  obtain ⟨σ0, h0, hxr⟩ := (A.mem_closureReads).1 hx
  by_cases hxP : x ∈ Pub
  · exact Or.inr (Or.inl hxP)
  by_cases hxw : x ∈ A.writes u
  · exact Or.inr (Or.inr hxw)
  obtain ⟨w, hw⟩ := hw
  exact Or.inl (A.mem_of_expresses hR hw h0 hxr hxP hxw)

end Population.AddressMap

namespace Population.DynPopulation

open Finset

variable {U S V : Type} (D : DynPopulation U S V)

/-- A program's own unit functions are compatible with its address map. -/
theorem addressMap_compatible (hac : D.AddressClosed) : (D.addressMap hac).Compatible D.f := D.reads_only

/-- **For a program, the closure is the padding over the assignments holding the public values.** -/
theorem addressMap_closureReads [Fintype S] (hac : D.AddressClosed) (Pub : Finset S) (a : S → V) (u : U) :
    (D.addressMap hac).closureReads Pub a u = (D.padOver {σ | ∀ t ∈ Pub, σ t = a t}).reads u := by
  ext s
  rw [AddressMap.mem_closureReads, D.mem_padOver_reads]
  rfl

end Population.DynPopulation

namespace Population

open Finset

/-- The pointer machine's public inputs: the zero region. -/
def vmPub (N : ℕ) : Finset (Slot N) := univ.filter fun s => s.isInput && !s.isSecret

/-- **The pointer machine's read sets depend on more than its inputs**: copy step `i` reads the zero word or the secret
word as the pointer, a unit's write, says. -/
theorem vm_not_addressedBy_inputs {N : ℕ} (i : Fin N) :
    ¬ (vm N).AddressedBy (univ.filter fun s => s.isInput) := by
  intro ho
  have h := ho (some i) (fun _ => false) (fun s => decide (s = .ptr)) (fun s hs => by cases s <;> simp_all)
  have hmem : Slot.secret i ∈ (vm N).reads (some i) (fun s => decide (s = .ptr)) := by
    simp [vm, src]
  rw [← h] at hmem
  simp [vm, src] at hmem

/-- **In the padding over the public values every secret word is upstream of the egress.** -/
theorem vm_pad_pub_secret_up (N : ℕ) (i : Fin N) :
    Slot.secret i ∈ ((vm N).padOver {σ | ∀ t ∈ vmPub N, σ t = false}).up (univ.filter fun s => s.isOut) := by
  refine ⟨.out i, by simp, .single ⟨some i, ?_, ?_⟩⟩
  · have hC : (fun s => decide (s = .ptr)) ∈ {σ : Slot N → Bool | ∀ t ∈ vmPub N, σ t = false} := by
      intro t ht
      cases t <;> simp_all [vmPub]
    exact (DynPopulation.mem_padOver_reads _).2 ⟨_, hC, by simp [vm, src]⟩
  · show Slot.out i ∈ (vm N).writes (some i)
    simp [vm]

/-- **The honest wiring is not enough**: no honest run has a secret upstream of the egress, yet one wrong one-bit unit
gives `2 ^ N` egress values over the secrets, more than the `|V| ^ |writes B|` the capacity theorem gives a static
population whose secrets lie outside the egress's upstream cone. -/
theorem vm_honest_wiring_fails (N : ℕ) (hN : 2 ≤ N) :
    (∀ σ, (∀ u, (vm N).Correct σ u) → ∀ i, Slot.secret i ∉ (vm N).up σ (univ.filter fun s => s.isOut)) ∧
      Fintype.card Bool ^ ((vm N).writesOf {none}).card <
        ((vm N).egress (fun _ => false) (univ.filter fun s => s.isSecret) (univ.filter fun s => s.isOut)
          {none}).card := by
  refine ⟨fun σ hσ i => vm_honest_up N hσ i, ?_⟩
  have hw : ((vm N).writesOf {none}).card = 1 := by
    simp [DynPopulation.writesOf, vm]
  rw [hw, Fintype.card_bool]
  calc 2 ^ 1 < 2 ^ 2 := by decide
    _ ≤ 2 ^ N := Nat.pow_le_pow_right (by decide) hN
    _ ≤ _ := vm_leak N

end Population
