import Proofs.Flock.Soundness.Discharge.ZkHidden.DefsJ
import Proofs.Flock.Soundness.Registered.ReadsX

set_option Elab.async false

/-!
# The rows at the hidden-output `--zk` tables of `J` tables, with or without shared rows

`ZkHidden.rsZHJ` takes `pub.tables = none`, as `ZkHidden.rsZH` does at one table (`BindH.lean`), so here are `rsShHJ`,
`rsZHJ` and their `Computes` without it: `tableShHJ_bcSiteS` is `ZkHidden.tableShHJ_bcSite` from
`HiddenExec.tableH_bcSiteS`, at table `j`'s inputs. `layShHJ` is the shadow layout `layZHJ` moves to the `--zk` tables,
where the registered reads (`Registered.ReadsX`) are read.
-/

namespace FlockSoundness.Discharge.ZkReg

open Flock FlockLevel3 Game Model Accounting Refine Audit ZkSession ZkLink ZkExec ZkBind ZkHidden HiddenExec Placed
  Composed
open FlockSoundness.Discharge.Exec (Inputs)
open scoped ENNReal

variable {n : ℕ} {I : Inputs} {mPts : ℕ} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
  (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) {Reg : Type}
  (ht : I.tags.typed = false) {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
  (hscope : Layout.scopeOkH c = true)
  (hU : Built c.unit) {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)

include hU hpub in
/-- `BcSiteH` at a table set up at table `j`'s inputs, with or without shared rows (`tableH_bcSiteS`). -/
theorem tableShHJ_bcSiteS {S' : Finset (Fin n)} (Tb : (I' : Inputs) × TableH I' mPts S')
    (p₁ : Tb.1.tags.typed = false) (p₂ : HmOut.parse Tb.1.tags Tb.1.circuitFile Tb.1.tables = .ok c)
    (hX : ∃ (j : ℕ) (d : Lean.Json) (x : DrawSetupH (inputsJ I j) mPts S' d), Tb = ⟨inputsJ I j, x.table⟩)
    {u : Fin n} (h : u ∈ S') :
    BcSiteH (Tb.2.acc p₁ p₂ (rowsFit_of_scopeH p₁ p₂ hscope)) pub (Tb.2.o u h) (Tb.2.vu u h) u.val := by
  obtain ⟨j, d, x, rfl⟩ := hX
  exact tableH_bcSiteS p₁ p₂ hscope (fun _ => d) hU ((loadPublicH_tagsAt _ _ _ _ _).trans hpub) x h

include hU hpub in
/-- **`BcSiteH` at every satisfiable public site** at `J` tables, with or without shared rows. -/
theorem sitesShHJ_bcSiteS :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S)
      (M : Binding.Msg (planShHJ dj y₀) (tabShHJ dj y₀) S R u),
      Binding.Sat (planShHJ dj y₀) (tabShHJ dj y₀) S R u M →
        BcSiteH ((sitesShHJ dj y₀ ht hc hscope).acc S R u h) pub ((sitesShHJ dj y₀ ht hc hscope).o S R u h)
          ((sitesShHJ dj y₀ ht hc hscope).vu S R u h) u.val := by
  intro S R u h M hM
  refine bcSiteH_recast _ (congrArg TabSpec.S (tabOfShHJ_eq dj y₀ S R u)).symm _ _
    (congrArg (fun T : TabSpec mPts => 2 ^ T.S.nbl) (tabOfShHJ_eq dj y₀ S R u)).symm _ _ ?_
  by_cases hS : Nonempty (DrawSetupZHJ (inputsJ I (partOf I (dj S) u.val)) mPts
    (partS I (dj S) S (partOf I (dj S) u.val)) (drawJ I (dj S) (partOf I (dj S) u.val)))
  · exact tableShHJ_bcSiteS hscope hU hpub _ _ _
      ⟨_, _, (Classical.choice hS).x, tableShHJ_set dj y₀ hS⟩ (mem_partS h)
  · have hns : ∀ z, ¬ (Partition.tabOf (planShHJ (Reg := Reg) dj y₀) (tabShHJ dj y₀) S R u).S.Satisfies z := by
      rw [tabOfShHJ_eq, tableShHJ_refused dj y₀ hS]
      exact y₀.x.refused_unsat _
    exact absurd hM (hns _)

/-- **`rs` at the public plan of `J` tables, with or without shared rows**: `rsShHJ` with `hsite` from
`sitesShHJ_bcSiteS`. -/
noncomputable abbrev rsShHJS : Binding.HmRowsX (layShHJ dj y₀ ht hc hscope hU (pub := pub) (Reg := Reg)) :=
  keyRowsH (sitesShHJ_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOkH_spec hU hscope).2.1 (rowsFit_of_scopeH (zkTyped ht) hc hscope) (Layout.outsOut_outsOf c hU n)
    (sitesShHJ_bcSiteS dj y₀ ht hc hscope hU hpub)
    (Layout.keysRow_of_parse (parse_untyped (zkTyped ht) hc).1 (Layout.scopeOkH_spec hU hscope).1
      (Layout.scopeOkH_spec hU hscope).2.2)

theorem rsShHJS_computes : (rsShHJS dj y₀ ht hc hscope hU hpub (Reg := Reg)).Computes :=
  keyRowsH_computes (sitesShHJ_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOkH_spec hU hscope).2.1 (rowsFit_of_scopeH (zkTyped ht) hc hscope) (Layout.outsOut_outsOf c hU n)
    (sitesShHJ_bcSiteS dj y₀ ht hc hscope hU hpub)
    (Layout.keysRow_of_parse (parse_untyped (zkTyped ht) hc).1 (Layout.scopeOkH_spec hU hscope).1
      (Layout.scopeOkH_spec hU hscope).2.2)

/-- **`rs` at the hidden-output `--zk` tables of `J` tables, with or without shared rows**: `rsShHJS` moved along
`planZHJ_S`. -/
noncomputable abbrev rsZHJS : HmRowsZX (layZHJ dj y₀ ht hc hscope hU (pub := pub) (Reg := Reg)) :=
  HmRowsZX.ofShadow (planZHJ_S dj y₀) (decAgreeHJ dj y₀ ht hc hscope hU) (rsShHJS dj y₀ ht hc hscope hU hpub)

/-- **`hHm` at the hidden-output `--zk` tables' rows of `J` tables, with or without shared rows** (`rsZHJ_computes`). -/
theorem rsZHJS_computes : (rsZHJS dj y₀ ht hc hscope hU hpub (Reg := Reg)).Computes :=
  HmRowsZX.ofShadow_computes (planZHJ_S dj y₀) (decAgreeHJ dj y₀ ht hc hscope hU) _
    (rsShHJS_computes dj y₀ ht hc hscope hU hpub)

end FlockSoundness.Discharge.ZkReg
