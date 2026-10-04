import FlockSoundness.Discharge.HiddenExec.HeadlineExec
import FlockSoundness.Discharge.ScopeW.Sites
import FlockSoundness.Discharge.ScopeG.Places

set_option Elab.async false

/-!
# What is open of the VU program's binding at the hidden tables

`VuBindH` (`Headline.lean`) is the program's places, its constant and zero columns, and a binding of registered rows.
From a placement of the VU's rows with its inputs on their unit slots (`VuPlace`), `Places.lean` gives the places, the
constant and the zero columns, with every input group's aliasing. What is left is that placement, at the sites of the
one-table hidden plan (`sitesG`: `ScopeW.sitesAtHW`, whose acceptances take any row of the four shapes), and a binding of
registered rows at the places it gives (`VuRowsH`), each a named obligation on a `sorry` stub:

* `vuPlaceRowsH_nonempty`: at `scopeOkG`, the unit returning the outputs;
* `vuPlaceRowsHT_nonempty`: at `scopeOkGT`, a tail's last stage returning them included.

They are one obligation each, not a placement and a binding apart: a binding reads the places of the units' committed
columns, which `VuPlace` pins only at the inputs.
-/

namespace FlockSoundness.Discharge.ScopeG

open Game Model Accounting Refine Audit Audit.Partition Flock
open FlockSoundness.Discharge.Exec FlockSoundness.Discharge.HiddenExec

/-- The four row shapes, from `scopeOkGT`. -/
theorem scopeOkGT_rows {c : Flock.Circuit} (hs : scopeOkGT c = true) : ∀ p ∈ c.ports.toList, ScopeW.rowShapeW p = true :=
  (scopeOkGT_spec hs).2.2.2.1

variable {n : ℕ} {I : Inputs} {mPts : ℕ} {c : Flock.Circuit}
  (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}
  (ht : I.tags.typed = false) (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c)

/-- **The sites of the one-table hidden plan at `scopeOkGT`** (`ScopeW.sitesAtHW`). -/
noncomputable abbrev sitesG (hs : scopeOkGT c = true) : SitesH (planAtH (Reg := Reg) dj x₀) (tabAtH dj x₀) :=
  ScopeW.sitesAtHW ht hc (scopeOkGT_rows hs) dj x₀

/-- One circuit per session (`ScopeW.sitesAtHW_parse`). -/
theorem sitesG_parse (hs : scopeOkGT c = true) :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      HmRow.parse ((sitesG dj x₀ ht hc hs).acc S R u h).tags ((sitesG dj x₀ ht hc hs).acc S R u h).cf
        ((sitesG dj x₀ ht hc hs).acc S R u h).tables = .ok c :=
  ScopeW.sitesAtHW_parse ht hc (scopeOkGT_rows hs) dj x₀

/-- **A binding of registered rows at the VU program's places `pp`**: `VuBindH`'s `lay`, `rs` and `hHm`. -/
structure VuRowsH (hs : scopeOkGT c = true) {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
    {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
    (pp : ProgPlaces (vuProg c hs n) (vuOutsOf c hs n) plan tab) where
  lay : Binding.Layout ((vuProg c hs n).partition (vuOutsOf c hs n)) plan tab
    (((vuProg c hs n).partition (vuOutsOf c hs n)).placeDecoder plan tab pp.derived.place)
  rs : Binding.HmRows lay
  hHm : rs.Computes

/-- The VU program's places at the hidden sites, from a placement `P`. -/
noncomputable abbrev ppG (hs : scopeOkGT c = true) (P : VuPlace c hs (sitesG (Reg := Reg) dj x₀ ht hc hs)) :
    ProgPlaces (vuProg c hs n) (vuOutsOf c hs n) (planAtH (Reg := Reg) dj x₀) (tabAtH dj x₀) :=
  P.pp (sitesG_parse dj x₀ ht hc hs) (vuOutsOf c hs n)

/-! ## The obligations -/

/-- **The VU's rows are placed with their inputs on their unit slots, and bound to registered rows there, when the unit
returns the outputs**: at a hidden-output session's tables of an untyped circuit at `scopeOkG` whose public file has no
public tables. -/
theorem vuPlaceRowsH_nonempty (hscope : scopeOkG c = true) {pub : Public}
    (hpub : HmOut.loadPublic I.tags c I.publicFile = .ok pub) (hpt : pub.tables = none) :
    Nonempty ((P : VuPlace c (scopeOkG_GT hscope) (sitesG (Reg := Reg) dj x₀ ht hc (scopeOkG_GT hscope))) ×'
      VuRowsH (scopeOkG_GT hscope) (ppG dj x₀ ht hc (scopeOkG_GT hscope) P)) := sorry

/-- **`vuPlaceRowsH_nonempty` at `scopeOkGT`**: a tail's last stage returning the outputs included, the output rows'
copies from `HmOut.outCol`'s stage columns. -/
theorem vuPlaceRowsHT_nonempty (hs : scopeOkGT c = true) {pub : Public}
    (hpub : HmOut.loadPublic I.tags c I.publicFile = .ok pub) (hpt : pub.tables = none) :
    Nonempty ((P : VuPlace c hs (sitesG (Reg := Reg) dj x₀ ht hc hs)) ×' VuRowsH hs (ppG dj x₀ ht hc hs P)) := sorry

end FlockSoundness.Discharge.ScopeG
