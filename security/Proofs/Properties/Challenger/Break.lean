module

public import Properties.Challenger

/-! The fixed finder `findBreak` outputs a break on every forgery and every collision among the strings it hashes. -/

@[expose] public section

namespace Properties.Challenger

open Lean (Json)
open Spot.Challenger

local instance lawfulBEqByteArrayBreak : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := by show (a.data == a.data) = true; exact beq_self_eq_true _

variable {dev : History} {key : ChallengerKey} {label : String} {coins : Json}

theorem collisionIn_valid {xs : List ByteArray} {x y : ByteArray} (h : collisionIn xs = some (x, y)) :
    x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y := by
  unfold collisionIn at h
  obtain ⟨⟨x', hx'⟩, hmx, hfs⟩ := List.exists_of_findSome?_eq_some h
  obtain ⟨⟨y', hy'⟩, hfind, he⟩ := Option.map_eq_some_iff.1 hfs
  have hp := List.find?_some hfind
  have hmy := List.mem_of_find?_eq_some hfind
  obtain ⟨a, -, ha⟩ := List.mem_map.1 hmx
  obtain ⟨b, -, hb⟩ := List.mem_map.1 hmy
  simp only [Prod.mk.injEq] at ha hb
  obtain ⟨rfl, rfl⟩ := ha
  obtain ⟨rfl, rfl⟩ := hb
  simp only [Prod.mk.injEq] at he
  obtain ⟨rfl, rfl⟩ := he
  simp only [Bool.and_eq_true, Bool.not_eq_true', beq_iff_eq, beq_eq_false_iff_ne, ne_eq] at hp
  exact ⟨fun e => hp.2 e.symm, hp.1.symm⟩

theorem findBreak_valid {w : Finding} (h : findBreak dev key label coins = some w) : w.Valid dev key := by
  unfold findBreak at h
  split at h
  · next m sig hf =>
    cases h
    have hp := List.find?_some hf
    simp only [Bool.and_eq_true, Bool.not_eq_true'] at hp
    refine ⟨hp.1, fun hs => ?_⟩
    have hc : (signedBy dev key.pub).contains m = true := List.contains_iff_mem.2 hs
    rw [hc] at hp
    exact absurd hp.2 (by decide)
  · obtain ⟨⟨x, y⟩, hc, he⟩ := Option.map_eq_some_iff.1 h
    subst he
    exact collisionIn_valid hc

theorem break_iff : Break dev key label coins ↔ findBreak dev key label coins ≠ none := by
  constructor
  · rintro ⟨w, hw, -⟩ h
    rw [h] at hw
    cases hw
  · intro h
    cases hf : findBreak dev key label coins with
    | none => exact absurd hf h
    | some w => exact ⟨w, hf, findBreak_valid hf⟩

theorem collisionIn_ne_none {xs : List ByteArray} {x y : ByteArray} (hx : x ∈ xs) (hy : y ∈ xs) (hne : x ≠ y)
    (he : Flock.Sha512.hash x = Flock.Sha512.hash y) : collisionIn xs ≠ none := by
  intro h
  unfold collisionIn at h
  have hmx : (x, Flock.Sha512.hash x) ∈ xs.map fun x => (x, Flock.Sha512.hash x) := List.mem_map_of_mem hx
  have hmy : (y, Flock.Sha512.hash y) ∈ xs.map fun x => (x, Flock.Sha512.hash x) := List.mem_map_of_mem hy
  have h1 := List.findSome?_eq_none_iff.1 h _ hmx
  simp only [Option.map_eq_none_iff] at h1
  apply List.find?_eq_none.1 h1 _ hmy
  simp [he, Ne.symm hne]

theorem break_of_forgery {m sig : ByteArray} (hm : (m, sig) ∈ signatures key coins) (hv : LmsVerifies key m sig)
    (hs : ¬ Signed dev key.pub m) : Break dev key label coins := by
  refine break_iff.2 ?_
  unfold findBreak
  split
  · exact Option.some_ne_none _
  · next hf =>
    exfalso
    apply List.find?_eq_none.1 hf _ hm
    have hc : (signedBy dev key.pub).contains m = false := by
      cases h : (signedBy dev key.pub).contains m
      · rfl
      · exact absurd (List.contains_iff_mem.1 h) hs
    show (lmsVerifies key m sig && !((signedBy dev key.pub).contains m)) = true
    rw [show lmsVerifies key m sig = true from hv, hc]
    rfl

theorem break_of_collision {x y : ByteArray} (hx : x ∈ readHashed key label coins ++ deviceHashed dev key.pub)
    (hy : y ∈ readHashed key label coins ++ deviceHashed dev key.pub) (hne : x ≠ y)
    (he : Flock.Sha512.hash x = Flock.Sha512.hash y) : Break dev key label coins := by
  refine break_iff.2 ?_
  unfold findBreak
  split
  · exact Option.some_ne_none _
  · intro h
    exact collisionIn_ne_none hx hy hne he (Option.map_eq_none_iff.1 h)

theorem signed_or_break {m sig : ByteArray} (hm : (m, sig) ∈ signatures key coins) (hv : LmsVerifies key m sig) :
    Signed dev key.pub m ∨ Break dev key label coins := by
  by_cases hs : Signed dev key.pub m
  · exact .inl hs
  · exact .inr (break_of_forgery hm hv hs)

theorem eq_or_break {x y : ByteArray} (hx : x ∈ readHashed key label coins ++ deviceHashed dev key.pub)
    (hy : y ∈ readHashed key label coins ++ deviceHashed dev key.pub)
    (he : Flock.Sha512.hash x = Flock.Sha512.hash y) : x = y ∨ Break dev key label coins := by
  by_cases e : x = y
  · exact .inl e
  · exact .inr (break_of_collision hx hy e he)

end Properties.Challenger
