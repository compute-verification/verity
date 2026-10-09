import Proofs.Flock.VBridge.Family.Join
import Proofs.Flock.VBridge.Family.Registered
import Proofs.Flock.VBridge.Family.Encode
import Proofs.Flock.VBridge.Family.Structure

/-!
# VBridge G at V*'s family, assembled from the lanes' definitions

`assembled` is a `VStarFamily` whose every field the lanes define is their definition:

* `rep₀ := repOf coinFiles` (ha's `repOf` at the history's coins, `Join.lean`), `alpha := alphaOf` (ho);
* `rows := rowsOf`, `acc := accOf` (ho: the openings read off the layers);
* `Sk := skOf own sel`, `cmt := cmtOf bc`, `wmsg := wmsgOf dec`, `msg := msgOf dec` (hc);
* `files := filesOf spec st cap lig` (hf).

`vbridgeRead_of_lanes` is `VBridgeRead` at any family whose fields satisfy the defining equations, its obligations
discharged by the lanes' theorems; `vbridgeRead_assembled` is `VBridgeRead` at `assembled`, every defining equation by
`rfl`. Their hypotheses are the lanes' named facts, each a `Prop` whose docstring names the check or lane that
establishes it.
-/

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages
open Flock.RecResiduals (portSizes groups)
open Flock.RecAlgebra (structureOf verifierRows el)

section family

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 leaf512 cls regs (schedOf sch)

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ)
variable (spec : SessionSpec) (st : Reg → L.Ω → Setup)
    (cap : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Array ByteArray)
    (lig : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Fin 2 → LigeritoProof)
    (own : Fin F.stmts.size → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop)
    (bc : (j : Fin F.stmts.size) → Cm → PosK (F.Vs j).c (F.Vs j).hU (F.Vs j).nv → List UInt8)

/-- **`VBridgeRead` at a family whose fields are the lanes' definitions**: `rep₀` is `repOf coinFiles`, `rows` and `acc`
the layers' (`rowsOf`, `accOf`), `files` `filesOf`'s, `Sk`, `cmt`, `wmsg` and `msg` `skOf`'s, `cmtOf`'s, `wmsgOf`'s and
`msgOf`'s. Each obligation is a lane's theorem: `HStruct` (`hStruct_of`, at struct's `hSomePart`), `HFit` (`hFit_of`,
`claimsIs_of`), `HShape` (`hShape_of`), `HResL` (`hResL_of`, at struct's `hCover_of` and reg's `hOne_of`,
`hRowKeys_of`, `hRowsAgree_of`), `HUL` (`hUL_of`, at struct's `hOpenUnits_of` and reg's `hOneOpen_of`), `HReads`
(`hReads_of`, at struct's `nOpen_pos`), `HSch` (`hSch_of`), `HFiles` (`hFiles_of`), `HRuns` (`hRuns_of`, at
`extrasAre_of`, `coins_of`, `combFolds_at`), `Encodes` (enc's `hEnc_of`), `HCommits` (`hCommits_of`), `VDecodesRead`
(`vDecodesRead_of`), `VFullRead` (`vFullRead_of`). -/
theorem vbridgeRead_of_lanes (dec : ℕ → ((j : Fin F.stmts.size) → (F.Vs j).Bits) → (ℐ).Msg)
    (hrep : F.rep₀ = F.repOf (F.coinFiles spec st cap lig F.wmsg)) (hrows : F.rows = F.rowsOf)
    (hacc : F.acc = F.accOf) (hF : F.files = F.filesOf spec st cap lig) (hSk : F.Sk = skOf F.Vs own sel)
    (hcmt : F.cmt = cmtOf F.Vs bc) (hw : F.wmsg = wmsgOf F.Vs F.Sk (ℐ).r dec) (hm : F.msg = msgOf F.Vs F.Sk dec)
    (hne : Nonempty (Reg × L.Ω × Reg₂)) (honame : F.HOpenName) (hcount : F.HPartCount) (hname : F.HUnitName)
    (hv : F.HVCoins dg N k Rw)
    (hlay : F.OneInLayer dg N k Rw) (hrr : F.RowsRegistered dg N k Rw) (hrck : F.RowColKeys dg N k Rw)
    (hsame : F.SameRow dg N k Rw) (hshared : F.RowsShared dg N k Rw) (hz : F.HZeroL dg N k Rw)
    (hcoef : F.HCoefCols dg N k Rw) (V : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → ℕ → Array F128)
    (hV : F.HAccRows dg N k Rw V) (hE : F.HEnds dg N k Rw V)
    (hset : ∀ R ω, SetupRealizes (st R ω) ((F.x R ω).toStatement regs) sch cls.ptLocal cls.mPts)
    (hshp : ∀ R ω, ShapeIs F.sh (st R ω) sch) (hcl : F.ClaimsAt) (hfold : ∀ R ω, FoldsOk (st R ω) F.sh)
    (hco : ∀ R ω, CoinsAgree (st R ω) F.sh sch (2 * ((F.x R ω).toStatement regs).regions.length))
    (hcapS : F.AtAccepted dg N k Rw fun R ω _ l => CapShaped (st R ω) sch (cap l))
    (hligS : F.AtAccepted dg N k Rw fun R ω _ l => ∀ r, LigShaped (st R ω) sch (lig l r))
    (hregs : ∀ R ω, ((F.x R ω).toStatement regs).regions.length < 2 ^ 48)
    (htr : F.AtAccepted dg N k Rw fun R ω t l =>
      (F.tableSrc spec st cap lig R ω t l).TranscriptIs T CT cls ((F.wmsg l).zip (t.map Prod.snd)))
    (hopen : F.AtAccepted dg N k Rw fun R ω t l => ∀ r,
      OpenedAll (st R ω) sch F.sh ((F.tableSrc spec st cap lig R ω t l).proofAt r) (F.rep R ω t l r))
    (hpub : F.PubCommits bc) (hsel : SelRounds sel) (hS : F.LawsExec) {dj : Fin F.stmts.size → Lean.Json}
    {d : Fin F.stmts.size → Flock.Draw.UnitDraw} (hu4 : F.U4Passes own dj d) :
    VBridgeRead (I := ℐ) F.zs dg N k Rw F.x F.wmsg F.Sk F.cmt F.msg := by
  obtain ⟨hsome, hcov, hU, hn⟩ := F.structure_of hne honame hcount
  obtain ⟨GS, gs, hst⟩ := F.hStruct_of hne hsome hname
  have hfit := F.hFit_of dg N k Rw spec st cap lig hrep hst hset (F.claimsIs_of hcl)
  have hxs : F.AtAccepted dg N k Rw fun R ω t l =>
      (F.tableSrc spec st cap lig R ω t l).ExtrasAre T CT cls ((F.wmsg l).zip (t.map Prod.snd)) :=
    fun R ω t _ _ _ _ => F.extrasAre_of spec st cap lig hrep hset R ω t _
  have henc := F.hEnc_of dg N k Rw spec st cap lig hset hshp hfit hxs hcapS hligS hregs
  exact vbridgeRead_of_family F dg N k Rw hst (F.hShape_of dg N k Rw hrep hst hfit)
    (F.hResL_of dg N k Rw hrep hst hcov hname (F.hRowKeys_of dg N k Rw hrr hrck)
      (F.hRowsAgree_of dg N k Rw hrr hrck hsame hshared) (F.hOne_of dg N k Rw hlay) hv hz)
    (F.hUL_of dg N k Rw hrows hacc hU (F.hOneOpen_of dg N k Rw hlay) hcoef V hV)
    (F.hReads_of dg N k Rw hacc hn V hV hE)
    (F.hSch_of spec st cap lig hF) (F.hFiles_of dg N k Rw spec st cap lig hF hset henc htr)
    (F.hRuns_of dg N k Rw spec st cap lig hF hset hshp henc hxs
      (fun R ω t _ _ _ _ => F.coins_of spec st cap lig hrep hco R ω t _)
      (fun R ω t _ _ _ _ => F.combFolds_at spec st cap lig hrep hfold R ω t _) hopen)
    (F.hCommits_of own sel bc hSk hcmt hpub) (F.vDecodesRead_of dec hw hm)
    (F.vFullRead_of own sel hSk hsel hS hu4)

/-- **The family at the lanes' definitions**: `F`'s statements, sessions and statement family, with `Sk`, `cmt`, `wmsg`
and `msg` hc's at `own`, `sel`, `bc` and `dec`, `rep₀` `repOf coinFiles` at those messages, `rows` and `acc` the
layers', then `alpha` `alphaOf` at that `rep₀`, then `files` `filesOf` at that `rep`. -/
noncomputable def assembled (dec : ℕ → ((j : Fin F.stmts.size) → (F.Vs j).Bits) → (ℐ).Msg) :
    VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂ :=
  let F₁ : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂ :=
    { F with
      rep₀ := F.repOf (F.coinFiles spec st cap lig (wmsgOf F.Vs (skOf F.Vs own sel) (ℐ).r dec))
      rows := F.rowsOf
      acc := F.accOf
      wmsg := wmsgOf F.Vs (skOf F.Vs own sel) (ℐ).r dec
      Sk := skOf F.Vs own sel
      cmt := cmtOf F.Vs bc
      msg := msgOf F.Vs (skOf F.Vs own sel) dec }
  let F₂ : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂ := { F₁ with alpha := F₁.alphaOf }
  { F₂ with files := F₂.filesOf spec st cap lig }

/-- **VBridge G's read conjunct at the lanes' definitions**: `VBridgeRead` at `skOf`'s commit strings, `cmtOf`'s
digests and `wmsgOf`'s and `msgOf`'s messages, at the family `assembled` (`vbridgeRead_of_lanes`, every defining
equation by `rfl`). The hypotheses that read a field `assembled` defines (`rep`, `coef`, `ops`, `files`, `Sk`) are read
at `assembled`; the others at `F`, whose statements, sessions and registrations `assembled` keeps. -/
theorem vbridgeRead_assembled (dec : ℕ → ((j : Fin F.stmts.size) → (F.Vs j).Bits) → (ℐ).Msg)
    (hne : Nonempty (Reg × L.Ω × Reg₂)) (honame : F.HOpenName) (hcount : F.HPartCount) (hname : F.HUnitName)
    (hv : (F.assembled spec st cap lig own sel bc dec).HVCoins dg N k Rw)
    (hlay : F.OneInLayer dg N k Rw) (hrr : F.RowsRegistered dg N k Rw) (hrck : F.RowColKeys dg N k Rw)
    (hsame : F.SameRow dg N k Rw) (hshared : F.RowsShared dg N k Rw) (hz : F.HZeroL dg N k Rw)
    (hcoef : (F.assembled spec st cap lig own sel bc dec).HCoefCols dg N k Rw)
    (V : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → ℕ → Array F128) (hV : F.HAccRows dg N k Rw V)
    (hE : (F.assembled spec st cap lig own sel bc dec).HEnds dg N k Rw V)
    (hset : ∀ R ω, SetupRealizes (st R ω) ((F.x R ω).toStatement regs) sch cls.ptLocal cls.mPts)
    (hshp : ∀ R ω, ShapeIs F.sh (st R ω) sch) (hcl : F.ClaimsAt) (hfold : ∀ R ω, FoldsOk (st R ω) F.sh)
    (hco : ∀ R ω, CoinsAgree (st R ω) F.sh sch (2 * ((F.x R ω).toStatement regs).regions.length))
    (hcapS : F.AtAccepted dg N k Rw fun R ω _ l => CapShaped (st R ω) sch (cap l))
    (hligS : F.AtAccepted dg N k Rw fun R ω _ l => ∀ r, LigShaped (st R ω) sch (lig l r))
    (hregs : ∀ R ω, ((F.x R ω).toStatement regs).regions.length < 2 ^ 48)
    (htr : F.AtAccepted dg N k Rw fun R ω t l =>
      ((F.assembled spec st cap lig own sel bc dec).tableSrc spec st cap lig R ω t l).TranscriptIs T CT cls
        ((wmsgOf F.Vs (skOf F.Vs own sel) (ℐ).r dec l).zip (t.map Prod.snd)))
    (hopen : F.AtAccepted dg N k Rw fun R ω t l => ∀ r,
      OpenedAll (st R ω) sch F.sh
        (((F.assembled spec st cap lig own sel bc dec).tableSrc spec st cap lig R ω t l).proofAt r)
        ((F.assembled spec st cap lig own sel bc dec).rep R ω t l r))
    (hpub : (F.assembled spec st cap lig own sel bc dec).PubCommits bc) (hsel : SelRounds sel) (hS : F.LawsExec)
    {dj : Fin F.stmts.size → Lean.Json} {d : Fin F.stmts.size → Flock.Draw.UnitDraw} (hu4 : F.U4Passes own dj d) :
    VBridgeRead (I := ℐ) F.zs dg N k Rw F.x (wmsgOf F.Vs (skOf F.Vs own sel) (ℐ).r dec) (skOf F.Vs own sel)
      (cmtOf F.Vs bc) (msgOf F.Vs (skOf F.Vs own sel) dec) :=
  (F.assembled spec st cap lig own sel bc dec).vbridgeRead_of_lanes dg N k Rw spec st cap lig own sel bc dec
    rfl rfl rfl rfl rfl rfl rfl rfl hne honame hcount hname hv hlay hrr hrck hsame hshared hz hcoef V hV hE hset hshp
    hcl hfold hco hcapS hligS hregs htr hopen hpub hsel hS hu4

end VStarFamily

end family

end FlockVBridge.Family
