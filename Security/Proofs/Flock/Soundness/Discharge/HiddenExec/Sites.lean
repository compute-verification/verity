import Proofs.Flock.Soundness.Discharge.HiddenExec.Placed

/-!
# The sites of a one-table hidden-output session

Integrate's `sitesAt` and `sitesAt_parse` (`Integrate/Sites.lean`) at the hidden tables (`planAtH`, `tabAtH`), at the
session's statement circuit `c` (`HmOut.parseStmt`, a templated file's committed circuit unrolled). A drawn table's
acceptance (`TableH.accS`) is its `Stmt.setupHidden` run, with the layout `Hidden.SpecC.layout` gives, its unit slots'
rows its flat twin's (`unitRowsFlatC`, at rows that fit), and the table's statement `stmtOf` of it, which is the
layout's block R1CS (`Hidden.SpecC.model_eq`).

`sitesAtHS` holds each drawn unit's acceptance, template instance and block; every acceptance's statement circuit is the
session's (`sitesAtHS_parse`). `TableH.acc`, `sitesAtH` and `sitesAtH_parse` are them at a flat session's `HmOut.parse`.
So `ppH`, `hConstH` and `hZeroH` are `ProgPlaces`, `hConst` and `hZero` for `Placed.keyProg` at the hidden tables.
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Model Refine Audit
open FlockSoundness.Discharge.Exec (Inputs)

variable {n : ℕ} {I : Inputs} {mPts : ℕ}
  (ht : I.tags.typed = false) {c : Flock.Circuit} (hcS : HmOut.parseStmt I.tags I.circuitFile I.tables = .ok c)
  (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c)
  (hv1 : Hidden.Any.RowsFitS c)

include hc in
/-- A flat session's circuit is its statement circuit (`HmOut.parse` refuses templated tags). -/
theorem parseStmt_of_parseH : HmOut.parseStmt I.tags I.circuitFile I.tables = .ok c :=
  parseStmt_of_parse (HmOut.parse_isTmpl hc) hc

/-- A committed circuit's output rows are its statement circuit's: `HmOut.unrolled` changes only its nets and template. -/
theorem outRowsS_of_unrolled {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {cs : Flock.Circuit}
    (hty : tags.typed = false) (h : Hidden.Committed tags cf tables cs) (hr : Hidden.Any.OutRowsS (HmOut.unrolled cs)) :
    Hidden.Any.OutRowsS cs := by
  obtain ⟨_nets, he⟩ := h.unrolled_eq hty
  rw [he] at hr
  exact hr

namespace TableH

variable {S : Finset (Fin n)} (x : TableH I mPts S)

include hc in
/-- A drawn table's statement is on the session's one circuit. -/
theorem c_eq : x.st.c = c := setupHidden_c_of_parse hc x.h

include ht hcS in
/-- **A drawn table's statement circuit is the session's**: its committed circuit unrolls to `c`. -/
theorem unrolled_eq : HmOut.unrolled x.st.c = c :=
  Except.ok.inj (((Hidden.setupHidden_specCC x.h).parseStmt_eq ht).symm.trans hcS)

/-- **A drawn table's acceptance**, at the session's statement circuit `c` (`HmOut.parseStmt`): its `Stmt.setupHidden`
run, with the layout `Hidden.SpecC.layout` gives and its unit slots' rows its flat twin's (`unitRowsFlatC`); the
table's statement is the layout's block R1CS at the table's regions (`Hidden.SpecC.model_eq`, and `TableH.hT`). -/
noncomputable def accS : AcceptedH x.T.S where
  tags := I.tags
  cf := I.circuitFile
  pf := I.publicFile
  coins := I.coins
  tables := I.tables
  draw := some x.d
  partition := I.partition
  program := I.program
  st := x.st
  ht := ht
  h := Hidden.setupHidden_specCC x.h
  L := (Hidden.setupHidden_specCC x.h).layout ht
  hR := unitRowsFlatC ht (Hidden.setupHidden_specCC x.h)
    (outRowsS_of_unrolled ht (Hidden.setupHidden_specCC x.h).committed (by rw [x.unrolled_eq ht hcS]; exact hv1.1))
  regs := x.regs
  hT := ((Hidden.setupHidden_specCC x.h).model_eq ht _ x.hwf x.regs).trans x.hT.symm

/-- **A drawn table's acceptance** at a flat session's circuit (`accS` at `HmOut.parse`'s). -/
noncomputable abbrev acc : AcceptedH x.T.S := x.accS ht (parseStmt_of_parseH hc) hv1

/-- **A drawn table's statement circuit is the session's**: `unrolled_eq` at its acceptance. -/
theorem accS_unrolled : HmOut.unrolled (x.accS ht hcS hv1).st.c = c := x.unrolled_eq ht hcS

/-- **A drawn table's statement circuit is the session's one circuit**, at a flat session's (`accS_unrolled`). -/
theorem acc_unrolled : HmOut.unrolled (x.acc ht hc hv1).st.c = c := x.unrolled_eq ht (parseStmt_of_parseH hc)

end TableH

variable (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}

/-- **The sites of the one-table hidden plan**, at the session's statement circuit: each drawn unit's acceptance,
template instance and block are its draw's table's. -/
noncomputable def sitesAtHS : SitesH (planAtH (Reg := Reg) dj x₀) (tabAtH dj x₀) where
  acc S _ _ _ := (tableAtH dj x₀ S).accS ht hcS hv1
  vu S _ u h := (tableAtH dj x₀ S).vu u h
  hvu S _ u h := (tableAtH dj x₀ S).hvu u h
  o S _ u h := (tableAtH dj x₀ S).o u h

/-- **One circuit per session**: every site's statement circuit is the session's statement circuit `c`. -/
theorem sitesAtHS_parse :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      HmOut.unrolled ((sitesAtHS ht hcS hv1 dj x₀).acc S R u h).st.c = c :=
  fun S _ _ _ => (tableAtH dj x₀ S).unrolled_eq ht hcS

/-- **The sites of the one-table hidden plan** at a flat session's circuit (`sitesAtHS` at `HmOut.parse`'s). -/
noncomputable abbrev sitesAtH : SitesH (planAtH (Reg := Reg) dj x₀) (tabAtH dj x₀) :=
  sitesAtHS ht (parseStmt_of_parseH hc) hv1 dj x₀

/-- **One circuit per session**, at a flat session's circuit (`sitesAtHS_parse`). -/
theorem sitesAtH_parse :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      HmOut.unrolled ((sitesAtH ht hc hv1 dj x₀).acc S R u h).st.c = c :=
  sitesAtHS_parse ht (parseStmt_of_parseH hc) hv1 dj x₀

/-- Every site's committed circuit is the session's one circuit `c` (the flat-only `instProg` path). -/
theorem sitesAtH_c :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S), ((sitesAtH ht hc hv1 dj x₀).acc S R u h).st.c = c :=
  fun S _ _ _ => (tableAtH dj x₀ S).c_eq hc

variable (hU : Built c.unit)
  (outs : Finset (Fin (Placed.gates (Placed.inGates n (Placed.circuitRows c hU).nIn) (Placed.circuitRows c hU) n)))

/-- **`pp` at the hidden tables**: `SitesH.progPlaces_keyProg` at `sitesAtH`. -/
noncomputable abbrev ppH : ProgPlaces (Placed.keyProg c hU n) outs (planAtH (Reg := Reg) dj x₀) (tabAtH dj x₀) :=
  SitesH.progPlaces_keyProg hU (sitesAtH_parse ht hc hv1 dj x₀) outs

/-- **`hConst` at the hidden tables** (`constCols_keyProgH`): only a unit's constant column sits on the one gate. -/
theorem hConstH :
    ((Placed.keyProg c hU n).partition outs).ConstCols (planAtH (Reg := Reg) dj x₀) (tabAtH dj x₀)
      (ppH ht hc hv1 dj x₀ hU outs).derived.place
      {Fin.castLE (Placed.le_gates _ _ n) (Placed.oneGate n (Placed.circuitRows c hU).nIn)} :=
  constCols_keyProgH hU (sitesAtH_parse ht hc hv1 dj x₀) outs

/-- **`hZero` at the hidden tables** (`zeroCols_keyProgH`): every column on the zero gate carries 0 in every satisfying
witness. -/
theorem hZeroH :
    ((Placed.keyProg c hU n).partition outs).ZeroCols (planAtH (Reg := Reg) dj x₀) (tabAtH dj x₀)
      (ppH ht hc hv1 dj x₀ hU outs).derived.place
      {Fin.castLE (Placed.le_gates _ _ n) (Placed.zeroGate n (Placed.circuitRows c hU).nIn)} :=
  zeroCols_keyProgH hU (sitesAtH_parse ht hc hv1 dj x₀) outs

end FlockSoundness.Discharge.HiddenExec
