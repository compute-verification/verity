import FlockSoundness.Discharge.HiddenExec.Placed
import FlockSoundness.Protocol.Discharge.HiddenExec.Sites

/-!
# The sites of a one-table hidden-output session

Integrate's `sitesAt` and `sitesAt_parse` (`Integrate/Sites.lean`) at the hidden tables (`planAtH`, `tabAtH`). A drawn
table's acceptance (`TableH.acc`) is its `Stmt.setupHidden` run, with the layout [wave2-hidden]'s `Hidden.hidden_layout`
gives, its unit slots' rows its flat twin's (`unitRowsFlat`, from `Hidden.hidden_delta'`, for `sha512/row/v1` rows), and
the table's statement `stmtOf` of it, which is the layout's block R1CS (`Integrate.model_eq`).

`sitesAtH` holds each drawn unit's acceptance, VU and block; every acceptance parses the session's one circuit file
(`sitesAtH_parse`). So `ppH`, `hConstH` and `hZeroH` are `ProgPlaces`, `hConst` and `hZero` for `Placed.keyProg` at the
hidden tables.
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Model Refine Audit
open FlockSoundness.Discharge.Exec (Inputs)

variable {n : ℕ} {I : Inputs} {mPts : ℕ}
  (ht : I.tags.typed = false) {c : Flock.Circuit} (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c)
  (hv1 : ∀ p ∈ c.ports.toList, p.v2 = false)

namespace TableH

variable {S : Finset (Fin n)} (x : TableH I mPts S)

end TableH

variable (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}

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
