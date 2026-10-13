import Proofs.Flock.C3.Reps.RepsRead
import Proofs.Flock.C3.Reps.OpensRead

/-!
# `SrcOK` at the reps' normalized tied source

The source is `tiedSrc (normSrc (repsOfVals stp sh rep vals))`: the step source `stp` with its reps read off the
values (`repsOfVals`), its salts and its openings from stratum 3 on set to defaults (`normSrc`), and its stages'
outputs tied (`tiedSrc`). `srcOK_repsOfVals` is `srcOK_tied_norm` there: the reps' part is read by
`repsMsgsRead_repsOfVals` (from `CapRead` and `TsRead`), and the openings at strata 0 and 1 by
`opensReadAt_repsOfVals` (from `OpensOfLayers`). The other hypotheses are `srcOK_tied_norm`'s at
`tiedSrc (repsOfVals stp sh rep vals)`: the committed rounds' read facts, the openings at R's stratum, `LigWhole`, and
the shapes' types. The reps' tie carries over to the same source by `srcRepTie_norm`.
-/

namespace FlockC3.Reps

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose

local notation "𝔽" => FlockLevel3.GF128

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **`SrcOK` at the reps' normalized tied source** `tiedSrc (normSrc (repsOfVals stp sh rep vals))`, from the cap's
and the reps' rounds' reads (`CapRead`, `TsRead`), the values' openings read off the layers (`OpensOfLayers`), and
`srcOK_tied_norm`'s other hypotheses at `tiedSrc (repsOfVals stp sh rep vals)`. -/
theorem srcOK_repsOfVals {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {sh : Flock.Shape} {rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
    {vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals}
    {W : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)}
    {rd : Fin 2 → ℕ → ℕ → ((j : Fin S) → (Vs j).Bits) → List 𝔽 × List Dig}
    {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
    (hleaf : HLeafC3 zs dg N k Rw)
    (hcoins : AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun t u =>
      StageCoins (CT := CT) (t.map Prod.snd) u)
    (hc : CapRead zs dg N k Rw stp own sel) (hts : TsRead zs dg N k Rw (repsOfVals stp sh rep vals) own sel)
    (hsp : SparseMsgsRead zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) own sel)
    (hop : OpenMsgsRead zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) own sel)
    (hlg : LigMsgsRead zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) own sel)
    (hv : OpensOfLayers zs dg N k Rw W rd vals)
    (hos₂ : OpensReadAt zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) 2)
    (hw : LigWhole zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)))
    (hcap : ∃ j, T j = Cap Dig) (hbool : ∃ j, T j = Bool) (hty : ∃ j, T j = (Opens GF128 Dig × Salts (List UInt8)))
    (hT : AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun _ u => ShapeTys T (holoAt st hp) u)
    (hplay : AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun _ u => StagesPlay (holoAt st hp) u) :
    SrcOK zs dg N k Rw (tiedSrc (normSrc (repsOfVals stp sh rep vals))) own sel :=
  srcOK_tied_norm hleaf hcoins (repsMsgsRead_repsOfVals hc hts) hsp hop hlg
    (fun s hs => by
      rcases Nat.lt_or_ge s 2 with h2 | h2
      · exact opensReadAt_repsOfVals (nm := fun u => u) (fun _ _ => rfl) hv ⟨s, h2⟩
      · obtain rfl : s = 2 := by omega
        exact hos₂)
    hw hcap hbool hty hT hplay

end

end FlockC3.Reps
