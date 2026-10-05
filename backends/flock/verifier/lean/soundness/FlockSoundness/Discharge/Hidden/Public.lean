import FlockSoundness.Discharge.Hidden.Facts
import FlockSoundness.Discharge.Layout.OutSite
import FlockSoundness.Protocol.Discharge.Hidden.Public
import FlockSoundness.Discharge.Layout.Site

/-!
# What a hidden-output statement takes from the public file (`PublicExposes`)

`HmOut.loadPublic` and `HmOut.drawn` are `HmRow`'s on the circuit without output ports, so a public file's output words
are `n · 0` of them (`loadPublic_outputs_nil`, `drawn_outputs_nil`). `HmOut.regions` is one `mkRegion` per row, region
`p` row `p`'s `b ‖ c` at its `hm96` slot, as `HmRow.regions`' region "digest p" (`regions_get`).
-/

set_option Elab.async false

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

open Refine

/-- A hidden-output population holds no output word. -/
theorem loadPublic_outputs_nil {tags : Tags} {c : Circuit} {pf : ByteArray} {pop : Public}
    (h : HmOut.loadPublic tags c pf = .ok pop) : pop.outputs = #[] := by
  have hs := (Layout.loadPublic_outputs (loadPublic_nil h)).1
  rw [show ({ c with outPorts := #[] } : Circuit).outWords = 0 from rfl, Nat.mul_zero] at hs
  exact Array.size_eq_zero_iff.1 hs

theorem foldl_extract_zero (a : Array ℕ) (w : ℕ) (hw : w = 0) (L : Array ℕ) :
    L.foldl (fun acc u => acc ++ a.extract (u * w) (u * w + w)) #[] = #[] := by
  subst hw
  rw [← Array.foldl_toList]
  suffices ∀ (l : List ℕ) (acc : Array ℕ), l.foldl (fun acc u => acc ++ a.extract (u * 0) (u * 0 + 0)) acc = acc from
    this _ _
  intro l
  induction l with
  | nil => intro acc; rfl
  | cons x l ih => intro acc; rw [List.foldl_cons, ih]; simp

/-- A hidden-output statement's drawn public file holds no output word. -/
theorem drawn_outputs_nil {c : Circuit} {pop pub : Public} {d : Lean.Json} (h : HmOut.drawn c pop d = .ok pub) :
    pub.outputs = #[] := by
  obtain ⟨D, -, hout⟩ := Layout.drawn_outputs h
  rw [hout]
  exact foldl_extract_zero _ _ rfl _

theorem public_exposes' : PublicExposes := by
  intro tags cf pf coins tables draw partition program st h
  have s := setupHidden_spec' h
  obtain ⟨hsz, hall⟩ := regions_get s.regions
  refine ⟨?_, hsz, fun p hp => ?_⟩
  · obtain ⟨pop, hl, hn, hd⟩ := s.pub
    cases draw with
    | none => rw [hn rfl]; exact loadPublic_outputs_nil hl
    | some d => exact drawn_outputs_nil (hd d rfl)
  · have hp' : p < st.regions.size := hsz ▸ hp
    exact ⟨st.regions[p], Array.getElem?_eq_getElem hp', hall p hp'⟩

end FlockSoundness.Discharge.Hidden
