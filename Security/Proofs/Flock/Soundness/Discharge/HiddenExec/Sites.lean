import Proofs.Flock.Soundness.Discharge.HiddenExec.Placed

/-!
# The sites of a one-table hidden-output session

Integrate's `sitesAt` and `sitesAt_parse` (`Integrate/Sites.lean`) at the hidden tables (`planAtH`, `tabAtH`). A drawn
table's acceptance (`TableH.acc`) is its `Stmt.setupHidden` run, with the layout `Discharge/Hidden`'s
`Hidden.hidden_layout`
gives, its unit slots' rows its flat twin's (`unitRowsFlat`, from `Hidden.hidden_delta'`, at rows that fit), and
the table's statement `stmtOf` of it, which is the layout's block R1CS (`Integrate.model_eq`).

`sitesAtH` holds each drawn unit's acceptance, template instance and block; every acceptance parses the session's one
circuit file (`sitesAtH_parse`). So `ppH`, `hConstH` and `hZeroH` are `ProgPlaces`, `hConst` and `hZero` for
`Placed.keyProg` at the hidden tables.
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Model Refine Audit
open FlockSoundness.Discharge.Exec (Inputs)

variable {n : ℕ} {I : Inputs} {mPts : ℕ}
  (ht : I.tags.typed = false) {c : Flock.Circuit} (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c)
  (hv1 : Hidden.Any.RowsFitS c)

namespace TableH

variable {S : Finset (Fin n)} (x : TableH I mPts S)

include hc in
/-- A drawn table's statement is on the session's one circuit. -/
theorem c_eq : x.st.c = c := Except.ok.inj ((setupHidden_parseC x.h).symm.trans hc)

/-- **A drawn table's acceptance**: its `Stmt.setupHidden` run, with the layout `Hidden.hidden_layout` gives and its
unit slots' rows its flat twin's (`unitRowsFlat`); the table's statement is the layout's block R1CS at the table's
regions (`Integrate.model_eq`, and `TableH.hT`). -/
noncomputable def acc : AcceptedH x.T.S where
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
  h := x.h
  L := Hidden.hidden_layout ht x.h
  hR := unitRowsFlat ht x.h (by rw [x.c_eq hc]; exact hv1.1)
  regs := x.regs
  hT := (Integrate.model_eq (parse_facts (parse_untyped ht (setupHidden_parseC x.h)).1) _
    (Hidden.setupHidden_spec x.h).leaf x.hwf x.regs).trans x.hT.symm

end TableH

variable (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}

/-- **The sites of the one-table hidden plan**: each drawn unit's acceptance, template instance and block are its draw's
table's. -/
noncomputable def sitesAtH : SitesH (planAtH (Reg := Reg) dj x₀) (tabAtH dj x₀) where
  acc S _ _ _ := (tableAtH dj x₀ S).acc ht hc hv1
  vu S _ u h := (tableAtH dj x₀ S).vu u h
  hvu S _ u h := (tableAtH dj x₀ S).hvu u h
  o S _ u h := (tableAtH dj x₀ S).o u h

/-- **One circuit per session**: every site's acceptance parses the session's one circuit file to `c`. -/
theorem sitesAtH_parse :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      HmRow.parse ((sitesAtH ht hc hv1 dj x₀).acc S R u h).tags ((sitesAtH ht hc hv1 dj x₀).acc S R u h).cf
        ((sitesAtH ht hc hv1 dj x₀).acc S R u h).tables = .ok c :=
  fun _ _ _ _ => (parse_untyped ht hc).1

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
