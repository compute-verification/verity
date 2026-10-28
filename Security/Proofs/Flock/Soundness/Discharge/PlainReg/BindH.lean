import Proofs.Flock.Soundness.Discharge.HiddenExec.HeadlineExec

set_option Elab.async false

/-!
# The rows at the hidden tables, with or without shared rows

`HiddenExec.rsAtH` takes `pub.tables = none`, which `verify --registered F` does not give (its registered check accepts
only a shared-row public file), and reads it nowhere: `sitesAtH_bcSite` is `tableH_bcSiteS` at every drawn table. So
here are `sitesAtH_bcSiteS`, `rsAtHS` and `rsAtHS_computes`, the same without it, at the layout `HiddenExec.layAtH`.
-/

namespace FlockSoundness.Discharge.PlainReg

open Game Model Accounting Refine Audit Audit.Partition Flock
open FlockSoundness.Discharge.Exec FlockSoundness.Discharge.HiddenExec

variable {n : ℕ} {I : Inputs} {mPts : ℕ} (ht : I.tags.typed = false) {c : Flock.Circuit}
  (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) (hscope : Layout.scopeOkH c = true)
  (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) (hU : Built c.unit)
  {pub : Public} (hpub : HmOut.loadPublic I.tags c I.publicFile = .ok pub) {Reg : Type}

include hU hpub in
/-- **`BcSiteH` at every satisfiable site** of the one-table hidden plan, with or without shared rows: at a draw
flock-verify sets up by `tableH_bcSiteS`; at a refused draw no message satisfies the table (`tableAtH_unsat`). -/
theorem sitesAtH_bcSiteS :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S)
      (M : Binding.Msg (planAtH dj x₀) (tabAtH dj x₀) S R u), Binding.Sat (planAtH dj x₀) (tabAtH dj x₀) S R u M →
        BcSiteH ((sitesAtH ht hc (rowsFit_of_scopeH ht hc hscope) dj x₀).acc S R u h) pub
          ((sitesAtH ht hc (rowsFit_of_scopeH ht hc hscope) dj x₀).o S R u h)
          ((sitesAtH ht hc (rowsFit_of_scopeH ht hc hscope) dj x₀).vu S R u h) u.val := by
  intro S R u h M hM
  show BcSiteH ((tableAtH dj x₀ S).acc ht hc (rowsFit_of_scopeH ht hc hscope)) pub ((tableAtH dj x₀ S).o u h)
    ((tableAtH dj x₀ S).vu u h) u.val
  by_cases hS : Nonempty (DrawSetupH I mPts S (dj S))
  · have e : tableAtH dj x₀ S = (Classical.choice hS).table := by
      unfold tableAtH
      rw [dite_eq_left_of_eq_true (eq_true hS)]
    rw [e]
    exact tableH_bcSiteS ht hc hscope dj hU hpub _ h
  · exact absurd hM (tableAtH_unsat dj x₀ hS _)

/-- **`rs` at the hidden tables, with or without shared rows**: `rsAtH` with `hsite` from `sitesAtH_bcSiteS`. -/
noncomputable abbrev rsAtHS : Binding.HmRowsX (layAtH ht hc hscope dj x₀ hU (pub := pub) (Reg := Reg)) :=
  keyRowsH (sitesAtH_parse ht hc (rowsFit_of_scopeH ht hc hscope) dj x₀) (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOkH_spec hU hscope).2.1 (rowsFit_of_scopeH ht hc hscope) (Layout.outsOut_outsOf c hU n)
    (sitesAtH_bcSiteS ht hc hscope dj x₀ hU hpub)
    (Layout.keysRow_of_parse (parse_untyped ht hc).1 (Layout.scopeOkH_spec hU hscope).1
      (Layout.scopeOkH_spec hU hscope).2.2)

/-- **`hHm` at the hidden tables, with or without shared rows** (`keyRowsH_computes`). -/
theorem rsAtHS_computes : (rsAtHS ht hc hscope dj x₀ hU hpub (Reg := Reg)).Computes :=
  keyRowsH_computes (sitesAtH_parse ht hc (rowsFit_of_scopeH ht hc hscope) dj x₀) (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOkH_spec hU hscope).2.1 (rowsFit_of_scopeH ht hc hscope) (Layout.outsOut_outsOf c hU n)
    (sitesAtH_bcSiteS ht hc hscope dj x₀ hU hpub)
    (Layout.keysRow_of_parse (parse_untyped ht hc).1 (Layout.scopeOkH_spec hU hscope).1
      (Layout.scopeOkH_spec hU hscope).2.2)

end FlockSoundness.Discharge.PlainReg
