import Properties.Proofs.NetworkCertifier.Check

/-!
# The extractors find a break wherever SHA-512 or LMS fails the run

`findCollision` returns only collisions (`findCollision_some`), and where it returns none SHA-512 is injective on its
list (`findCollision_none`); `runStrings` lists every string the run hashes (`mem_runStrings`), so where the run yields
no collision SHA-512 has none on `RunStrings` (`cf_run`). Likewise `findForgery` returns only forgeries
(`findForgery_some`), so where the run yields none, each certificate's signature that verifies over its message is over
a message the device signed (`signed_of_not`).
-/

open NetworkCheck hiding Params
open NetworkCheck.Check
open Flock
open NetworkCertifier (ClockSync)

set_option maxRecDepth 20000

namespace Properties.NetworkCertifier

open Tree CheckProofs

theorem toOption_eq_some {ε α : Type} {x : Except ε α} {a : α} : x.toOption = some a ↔ x = .ok a := by
  cases x <;> simp [Except.toOption]

/-! ## Collisions -/

theorem findCollision_some : ∀ {l : List ByteArray} {x y : ByteArray}, findCollision l = some (x, y) →
    x ≠ y ∧ Sha512.hash x = Sha512.hash y
  | [], _, _, h => by simp [findCollision] at h
  | a :: xs, x, y, h => by
    unfold findCollision at h
    split at h
    · rename_i y' hf
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj h)
      have hp := List.find?_some hf
      cases hb : (a == y')
      · rw [hb, Bool.not_false, Bool.true_and] at hp
        exact ⟨fun e => (by rw [e, ba_beq.mpr rfl] at hb; cases hb), ba_beq.mp hp⟩
      · rw [hb] at hp; cases hp
    · exact findCollision_some h

theorem findCollision_none {l : List ByteArray} (h : findCollision l = none) :
    ∀ x ∈ l, ∀ y ∈ l, Sha512.hash x = Sha512.hash y → x = y := by
  induction l with
  | nil => intro x hx; simp at hx
  | cons a xs ih =>
    intro x hx y hy hxy
    unfold findCollision at h
    split at h
    · cases h
    · rename_i hf
      have no : ∀ z ∈ xs, Sha512.hash a = Sha512.hash z → a = z := fun z hz hh => by
        have hn := List.find?_eq_none.mp hf z hz
        cases hb : (a == z)
        · rw [hb, Bool.not_false, Bool.true_and, ba_beq.mpr hh] at hn; exact absurd rfl hn
        · exact ba_beq.mp hb
      rcases List.mem_cons.mp hx with rfl | hx' <;> rcases List.mem_cons.mp hy with rfl | hy'
      · rfl
      · exact no y hy' hxy
      · exact (no x hx' hxy.symm).symm
      · exact ih h x hx' y hy' hxy

theorem cf_of_not {l : List ByteArray} (h : ¬ IsCollision (findCollision l)) :
    ∀ x ∈ l, ∀ y ∈ l, Sha512.hash x = Sha512.hash y → x = y := by
  cases hf : findCollision l with
  | none => exact findCollision_none hf
  | some xy =>
    obtain ⟨x, y⟩ := xy
    rw [hf] at h
    exact (h (findCollision_some hf)).elim

/-! ## The run's strings -/

theorem mem_headStrings {h : RecordHead} {x : ByteArray} (hx : HeadString h x) : x ∈ headStrings h := by
  obtain ⟨enc, he, rfl⟩ := hx
  simp [headStrings, he, Except.toOption]

theorem mem_framesStrings {g : Grid} {fr : Array ByteArray} {x : ByteArray} (hx : FramesString g fr x) :
    x ∈ framesStrings g fr := by
  unfold framesStrings
  rcases hx with ⟨f, hf, hl⟩ | rfl | ⟨l, hl, rfl⟩ | ⟨ls, hls, lv, hlv, hx⟩
  · exact List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _
      (List.mem_filterMap.mpr ⟨f, hf, toOption_eq_some.mpr hl⟩)))
  · exact List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_singleton_self _)))
  · exact List.mem_append_left _ (List.mem_append_right _ (List.mem_map.mpr ⟨l, List.mem_range.mpr hl, rfl⟩))
  · refine List.mem_append_right _ ?_
    rw [hls]
    exact List.mem_flatMap.mpr ⟨lv, List.mem_range.mpr hlv, hx⟩

theorem mem_treeStrings {g : Grid} {grid : Array (Array (Option ByteArray))} {x : ByteArray}
    (hx : TreeString g grid x) : x ∈ treeStrings g grid := by
  obtain ⟨fr, hfr, hx⟩ := hx
  unfold treeStrings
  rw [hfr]
  exact mem_framesStrings hx

theorem declared_eq {W : Nat} {link : Link} {syncs : Fin W → ClockSync} (h : Declares link syncs) :
    syncs = declared W link :=
  funext fun v => by rw [declared, Array.getD_eq_getD_getElem?, h v]; rfl

theorem mem_runStrings {p : Params} {c : Commitment} {e : Evidence} {dev : Device p} {eg x : ByteArray}
    (hx : RunStrings p c e dev eg x) : x ∈ runStrings p c e dev eg := by
  unfold runStrings
  rcases hx with ⟨r, hr, hx | hx⟩ | ⟨l, link, hm, w, hw, syncs, hd, hx | ⟨h, hh, hx⟩⟩ |
      ⟨i, hi, hx | ⟨dg, hdg, hx⟩⟩ | ⟨ℓ, w, hw, hx⟩
  · exact List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _
      (List.mem_flatMap.mpr ⟨r, hr, List.mem_append_left _ (mem_headStrings hx)⟩)))
  · refine List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _
      (List.mem_flatMap.mpr ⟨r, hr, List.mem_append_right _ ?_⟩)))
    rw [hx]
    exact List.mem_singleton_self _
  · obtain rfl := declared_eq hd
    exact List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _
      (List.mem_flatMap.mpr ⟨(l, link), hm, List.mem_flatMap.mpr ⟨w, List.mem_range.mpr hw,
        List.mem_append_left _ (mem_treeStrings hx)⟩⟩)))
  · obtain rfl := declared_eq hd
    refine List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _
      (List.mem_flatMap.mpr ⟨(l, link), hm, List.mem_flatMap.mpr ⟨w, List.mem_range.mpr hw,
        List.mem_append_right _ (List.mem_flatMap.mpr ⟨h, ?_, mem_headStrings hx⟩)⟩⟩)))
    rw [hh]
    exact List.mem_singleton_self _
  · exact List.mem_append_left _ (List.mem_append_right _
      (List.mem_flatMap.mpr ⟨i, hi, List.mem_append_left _ (mem_headStrings hx)⟩))
  · refine List.mem_append_left _ (List.mem_append_right _
      (List.mem_flatMap.mpr ⟨i, hi, List.mem_append_right _ ?_⟩))
    rw [hdg]
    show x ∈ (Except.toOption _).toList
    rw [hx]
    exact List.mem_singleton_self _
  · exact List.mem_append_right _ (List.mem_flatMap.mpr ⟨ℓ, List.mem_finRange ℓ,
      List.mem_flatMap.mpr ⟨w, List.mem_range.mpr hw, mem_treeStrings hx⟩⟩)

/-- Where the run yields no collision, SHA-512 has none on the strings the run hashes. -/
theorem cf_run {p : Params} {c : Commitment} {e : Evidence} {dev : Device p} {eg : ByteArray}
    (h : ¬ IsCollision (findCollision (runStrings p c e dev eg))) : CF (RunStrings p c e dev eg) :=
  fun x y hx hy hxy => cf_of_not h x (mem_runStrings hx) y (mem_runStrings hy) hxy

/-! ## Forgeries -/

theorem lmsAccepts_of {p : Params} {m sig : ByteArray} (h : LmsVerifies p m sig) : lmsAccepts p m sig = true := by
  obtain ⟨s, hs, hv⟩ := h
  simp [lmsAccepts, hs, hv, Except.toOption]

theorem lmsVerifies_of {p : Params} {m sig : ByteArray} (h : lmsAccepts p m sig = true) : LmsVerifies p m sig := by
  unfold lmsAccepts at h
  split at h
  · rename_i s hs
    refine ⟨s, hs, ?_⟩
    cases hv : Lms.verify p.lms p.key m s with
    | error _ => simp [hv, Except.toOption] at h
    | ok b => simp [hv, Except.toOption] at h; rw [h]
  · cases h

theorem mem_signedMessages {p : Params} {dev : Device p} {m : ByteArray} :
    m ∈ signedMessages dev ↔ Signed dev m := by
  simp only [signedMessages, List.mem_filterMap, Signed, Option.bind_eq_some_iff, toOption_eq_some]

theorem findForgery_some {p : Params} {c : Commitment} {dev : Device p} {m σ : ByteArray}
    (h : findForgery c dev = some (m, σ)) : IsForgery dev (some (m, σ)) := by
  obtain ⟨r, -, hr⟩ := List.exists_of_findSome?_eq_some h
  obtain ⟨m', -, hif⟩ := Option.bind_eq_some_iff.mp hr
  split at hif
  · rename_i hc
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hif)
    rw [Bool.and_eq_true] at hc
    refine ⟨lmsVerifies_of hc.1, fun hs => ?_⟩
    have hany : (signedMessages dev).any (· == m') = true :=
      List.any_eq_true.mpr ⟨m', mem_signedMessages.mpr hs, ba_beq.mpr rfl⟩
    rw [hany] at hc
    cases hc.2
  · cases hif

/-- Where the run yields no forgery, each certificate whose signature verifies over its message is over a message the
device signed. -/
theorem signed_of_not {p : Params} {c : Commitment} {dev : Device p} (h : ¬ IsForgery dev (findForgery c dev)) :
    ∀ r ∈ c.records.toList, ∀ m, certMessage r = some m → LmsVerifies p m r.certificate.signature →
      Signed dev m := by
  intro r hr m hm hv
  cases hf : findForgery c dev with
  | some ms =>
    obtain ⟨m', σ'⟩ := ms
    rw [hf] at h
    exact (h (findForgery_some hf)).elim
  | none =>
    have hn := List.findSome?_eq_none_iff.mp hf r hr
    rw [hm, Option.bind_some] at hn
    split at hn
    · cases hn
    · rename_i hc
      rw [lmsAccepts_of hv, Bool.true_and, Bool.not_eq_true', Bool.not_eq_false] at hc
      obtain ⟨m'', hm'', he⟩ := List.any_eq_true.mp hc
      rw [ba_beq.mp he] at hm''
      exact mem_signedMessages.mp hm''

end Properties.NetworkCertifier
