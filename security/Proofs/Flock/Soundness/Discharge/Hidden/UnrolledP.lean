module

public import Proofs.Flock.Soundness.Discharge.Hidden.Unrolled
public import Proofs.Flock.Soundness.Discharge.Aliased.KeysP

/-! The committed circuit's public input ports, public file and public words are the statement circuit's (`Unrolled.lean` at the verifier's public inputs). -/

@[expose] public section

namespace FlockSoundness.Discharge.Hidden

open Flock

variable {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {cs : Circuit}

theorem Committed.unrolled_inWords (hty : tags.typed = false) (h : Committed tags cf tables cs) :
    (HmOut.unrolled cs).unit.inWords = cs.unit.inWords := by
  rcases h with h | h
  · rw [unrolled_flat (parse_tmpl_none hty h)]
  · obtain ⟨n, _pos, hsz, heu, -, -, hw, -⟩ := tmpl_unit_at h
    rw [heu]
    show ((cs.nets.set! cs.unitNet ("unit", n)).getD cs.unitNet default).2.inWords = _
    rw [getD_set_unit _ _ _ _ hsz, ite_eq_left rfl, hw]

theorem parseIn_commit (hty : tags.typed = false) (h : Committed tags cf tables cs) :
    HmIn.parse (HmOut.unrolled cs) = HmIn.parse cs := by
  obtain ⟨nets, he⟩ := h.unrolled_eq hty
  have hg := h.unrolled_inGroups hty
  rw [he] at hg ⊢
  unfold HmIn.parse HmIn.words
  rw [hg]
  rfl

theorem loadPublicIn_commit (hty : tags.typed = false) (h : Committed tags cf tables cs) (t : Tags) (pi : HmIn.Ports)
    (pf : ByteArray) : HmIn.loadPublic t (HmOut.unrolled cs) pi pf = HmIn.loadPublic t cs pi pf := by
  obtain ⟨_nets, he⟩ := h.unrolled_eq hty
  rw [he]
  rfl

theorem pubLeaf_commit (hty : tags.typed = false) (h : Committed tags cf tables cs) (pi : HmIn.Ports) (x : ℕ) :
    Aliased.pubLeaf (HmOut.unrolled cs) pi x = Aliased.pubLeaf cs pi x := by
  unfold Aliased.pubLeaf
  rw [h.unrolled_inWords hty, h.unrolled_inGroups hty]

theorem srcKeyP_eq_commit (hty : tags.typed = false) (h : Committed tags cf tables cs) (C : CheckFacts cs)
    {pi : HmIn.Ports} {x x' : ℕ}
    (hk : Aliased.srcKeyP (HmOut.unrolled cs) pi x = Aliased.srcKeyP (HmOut.unrolled cs) pi x') :
    Aliased.srcKeyP cs pi x = Aliased.srcKeyP cs pi x' := by
  unfold Aliased.srcKeyP at hk ⊢
  simp only [pubLeaf_commit hty h, h.unrolled_inWords hty, h.unrolled_inGroups hty] at hk
  by_cases h1 : 0 ≤ Aliased.pubLeaf cs pi x <;> by_cases h2 : 0 ≤ Aliased.pubLeaf cs pi x' <;>
    simp only [h1, h2, ↓reduceIte, reduceCtorEq, Aliased.KeyP.src.injEq] at hk ⊢
  all_goals first | exact hk | exact srcKey_eq_commit hty h C hk

theorem srcKeyP_pub_commit (hty : tags.typed = false) (h : Committed tags cf tables cs) {pi : HmIn.Ports} {x q t : ℕ}
    (hk : Aliased.srcKeyP (HmOut.unrolled cs) pi x = .pub q t) : Aliased.srcKeyP cs pi x = .pub q t := by
  unfold Aliased.srcKeyP at hk ⊢
  simp only [pubLeaf_commit hty h, h.unrolled_inWords hty, h.unrolled_inGroups hty] at hk
  by_cases h1 : 0 ≤ Aliased.pubLeaf cs pi x <;> simp only [h1, ↓reduceIte, reduceCtorEq] at hk ⊢
  all_goals exact hk

end FlockSoundness.Discharge.Hidden
