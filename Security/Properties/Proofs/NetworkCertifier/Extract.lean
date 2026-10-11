import Properties.Proofs.NetworkCertifier.Check

/-!
# The strings the check hashes, listed

`checkStrings` lists every string the check hashes (`mem_checkStrings`): what a run without a SHA-512 collision among
the strings it hashes rules out (`Properties.Proofs.CertifierNetwork.Runs`).
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

/-! ## The run's strings -/

theorem mem_headStrings {h : RecordHead} {x : ByteArray} (hx : HeadString h x) : x ∈ headStrings h := by
  obtain ⟨enc, he, hx⟩ := hx
  simp [headStrings, he, hx, Except.toOption]

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

theorem mem_checkStrings {p : Params} {c : Commitment} {e : Evidence} {eg x : ByteArray}
    (hx : CheckString p c e eg x) : x ∈ checkStrings p c e eg := by
  unfold checkStrings
  rcases hx with ⟨r, hr, hx | hx⟩ | ⟨l, link, hm, w, hw, syncs, hd, hx | ⟨h, hh, hx⟩⟩
  · exact List.mem_append_left _ (List.mem_flatMap.mpr ⟨r, hr, List.mem_append_left _ (mem_headStrings hx)⟩)
  · refine List.mem_append_left _ (List.mem_flatMap.mpr ⟨r, hr, List.mem_append_right _ ?_⟩)
    rw [hx]
    exact List.mem_singleton_self _
  · obtain rfl := declared_eq hd
    exact List.mem_append_right _ (List.mem_flatMap.mpr ⟨(l, link), hm, List.mem_flatMap.mpr
      ⟨w, List.mem_range.mpr hw, List.mem_append_left _ (mem_treeStrings hx)⟩⟩)
  · obtain rfl := declared_eq hd
    refine List.mem_append_right _ (List.mem_flatMap.mpr ⟨(l, link), hm, List.mem_flatMap.mpr ⟨w,
      List.mem_range.mpr hw, List.mem_append_right _ (List.mem_flatMap.mpr ⟨h, ?_, mem_headStrings hx⟩)⟩⟩)
    rw [hh]
    exact List.mem_singleton_self _

end Properties.NetworkCertifier
