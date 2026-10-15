import Proofs.Flock.C3.Compose.LaterHolo
import Proofs.Flock.C3.Sparse.SessionZero

/-!
# `stageCoins_holo_held` at a family, the record's decode from check 5

`recordDecodes_family`: at every accepted outer proof of a family's sessions the record at the history decodes
(`FlockC3.Sparse.record_decodes_of_accepted`): check 5 passes where a session accepts (`CoinsAccepted`), the family
has a sparse statement (`hs`), and check 5's sources are the files of the record they name (`hfiles`).
`stageCoins_holo_family`: `stageCoins_holo_held` at the family's sessions (`F.zs`, `F.ish`), its `hrec` that. The
run's points stay the history's first coin (`hpts`, `FlockC3.HPort.PointsFromHistory`).
-/

namespace FlockC3.Compose

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Reps (SrcRepTie repsOfVals UnitVals ProofOfVals SetupOfVals FramesOfVals RepRoundsHeld RepsDrawn)
open Flock.Guarantees (holoCTs holoRounds)

variable {nT : ℕ} {T : Fin nT → Type} {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ}
  {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => holoInnerAt T (holoCTs st hp) st hp

/-- **The record decodes at every accepted outer proof of a family** (`record_decodes_of_accepted`), at any source. -/
theorem recordDecodes_family (F : C3Family T (holoCTs st hp) st hp xR L Reg Cm Reg' Reg₂)
    {rows : Option (Array Flock.Registered.Port)} {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json}
    {coins : String → Option Flock.RecCoins.Source} (hchk : FlockC3.Sparse.CoinsAccepted F rows wr reg coins)
    (hsi : FlockC3.VStarList.SparseIs F.ish hp) (hell : hp.EllOk (execClassAt st).cls.kLog)
    {s : Fin (vstarCount F.ish)} {shS : Flock.RecSparse.Shape} {hR : ℕ}
    (hs : F.stmts[(s : ℕ)]? = some (.sparse shS hR))
    (hfiles : ∀ r src, coins r = some src → ∃ f, src = .files f ∧ f.record = r) {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount F.ish)) → (F.Vs j).Bits) →
      C3Run (holoAt st hp)) :
    AtSrc F.zs dg N k Rw src fun t _ => ∃ spec sess, Flock.Record.decode spec (wr t).2 = .ok sess :=
  fun _ _ _ _ _ _ hA => FlockC3.Sparse.record_decodes_of_accepted F hchk hsi hell hs hfiles hA

/-- **`stageCoins_holo_held` at a family**: its `hrec` from check 5 (`recordDecodes_family`). -/
theorem stageCoins_holo_family (F : C3Family T (holoCTs st hp) st hp xR L Reg Cm Reg' Reg₂)
    {stmt : Fin (vstarCount F.ish) → VStar.Stmt} {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount F.ish)) → (F.Vs j).Bits) → C3Run (holoAt st hp)}
    {sh : Flock.Shape}
    {rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount F.ish)) → (F.Vs j).Bits) → Fin 2 → RepData}
    {vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount F.ish)) → (F.Vs j).Bits) → Fin 2 → UnitVals}
    {wr : List (Cm × (ℐ).Coin) → Lean.Json × String}
    {held : Lean.Json × String → Except String RecVStar.InnerCommits} {sparse : Option RecSparse.Shape}
    {tbl unit : String} {bc : Cm → String → Nat → List UInt8} {s : Fin 2 → String}
    {ix : Fin 2 → ℕ → ℕ} {rr rs : Array RecVStar.SchedRound} {sp : RecSparse.Shape} {shR : RecVStar.StackedShape}
    {M : FlockC3.Reps.StackedMaps stmt F.Vs} {osh : Flock.Shape}
    {c3 : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount F.ish)) → (F.Vs j).Bits) → RecCoins.C3}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount F.ish)) → (F.Vs j).Bits) → Array F128}
    {kap : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount F.ish)) → (F.Vs j).Bits) →
      Array (Array (Array F128))}
    {e0 : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount F.ish)) → (F.Vs j).Bits) → ℕ → ByteArray}
    {sb : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount F.ish)) → (F.Vs j).Bits) → ℕ → ℕ → ℕ →
      ByteArray}
    {e : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount F.ish)) → (F.Vs j).Bits) → ℕ → ℕ → ByteArray}
    {cap₀ : Cap Dig} {rows : Option (Array Flock.Registered.Port)} {reg : Lean.Json}
    {coins : String → Option Flock.RecCoins.Source} (hchk : FlockC3.Sparse.CoinsAccepted F rows wr reg coins)
    (hsi : FlockC3.VStarList.SparseIs F.ish hp) {sI : Fin (vstarCount F.ish)} {shS : Flock.RecSparse.Shape} {hR : ℕ}
    (hsS : F.stmts[(sI : ℕ)]? = some (.sparse shS hR))
    (hfiles : ∀ r src, coins r = some src → ∃ f, src = .files f ∧ f.record = r)
    (hleaf : HLeafC3 F.zs dg N k Rw)
    (halg : AlgebraDecodes (I := ℐ) F.zs dg N k Rw sh rep) (hacc : AccChains (I := ℐ) F.zs dg N k Rw sh rep)
    (hpr : ProofOfVals F.zs dg N k Rw sh rep vals)
    (hsv : SetupOfVals F.zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) sh rep vals)
    (hf : FramesOfVals F.zs dg N k Rw sh rep vals)
    (hcus : FlockSoundness.Assumptions.InnerRecordCustody wr held sparse
      (Flock.Guarantees.holoMap st hp Cm F.ish tbl bc))
    (hheldR : RepRoundsHeld F.zs dg N k Rw sh rep wr held s ix)
    (hdr : RepsDrawn F.zs dg N k Rw sh rep vals (holoSelOne F.ish tbl) s ix)
    (hrt : SrcRepTie F.zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)))
    (hst : FlockC3.Sparse.SrcSparseTie F.zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)))
    (hot : AtSrc F.zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun _ u => ∃ sh w ops,
      FlockC3.Opening.OpenTie sh w ops u ∧ OpeningReadAt (runSteps u) u)
    (hell : hp.EllOk (execClassAt st).cls.kLog)
    (hosh : RecOpening.shape? (holoAt st hp).cls.shape.mReg 2 = some osh)
    (hmaps : MapsAt F.zs dg N k Rw M osh unit (repsOfVals stp sh rep vals) c3 w kap e0 sb e cap₀)
    (hc3 : C3Held wr held F.ish c3) (hrep : holoRounds F.ish.m F.ish.kLog F.ish.claims = some rr)
    (hsp : F.ish.sparse = some sp) (hsh : F.ish.stacked = some shR) (hcl : shR.claims = RecVStar.STACKED_CLAIMS)
    (hrs : RecVStar.stackedRounds shR = some rs) (hc : 0 < F.ish.claims)
    (hA : repFormsAt (holoAt st hp) = repForms F.ish.m F.ish.kLog F.ish.claims)
    (hshape : (holoAt st hp).cls.shape = sparseOf sp) (htblS : tbl ≠ s!"{RecVStar.SPARSE}/")
    (htbl : tbl ≠ s!"{RecVStar.STACKED}/")
    (hpts : AtSrc F.zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun t u =>
      FlockC3.HPort.PointsFromHistory F.ish tbl bc t u)
    (hheld : AtSrc F.zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun t _ => HeldOk wr held t)
    (hcnt : AtSrc F.zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun t _ => SparseCountedAt wr held sp t)
    (hss : AtSrc F.zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun t u =>
      SparseStagesAt wr held F.ish unit sp t u)
    (hx : AtSrc F.zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun t _ => StackedHeldAt wr held rs t)
    (hts : AtSrc F.zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun _ u => RepStagesCounted rr u) :
    AtSrc F.zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun t u =>
      StageCoins (CT := holoCTs st hp) (t.map Prod.snd) u :=
  stageCoins_holo_held hleaf halg hacc hpr hsv hf hcus hheldR hdr hrt hst hot hell hosh hmaps hc3 hrep hsp hsh hcl
    hrs hc hA hshape htblS htbl (recordDecodes_family F hchk hsi hell hsS hfiles _) hpts hheld hcnt hss hx hts

end

end FlockC3.Compose
