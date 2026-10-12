import Proofs.Flock.VBridge.Family.Transcript
import Proofs.Flock.VBridge.Family.Registered
import Proofs.Flock.VBridge.Family.Registrants
import Proofs.Flock.VBridge.Family.Encode
import Proofs.Flock.VBridge.Family.Structure

/-!
# VBridge G at V*'s family, assembled from the lanes' definitions

`assembled` is a `VStarFamily` whose every field the lanes define is their definition:

* `rep₀ := repOf coinFiles` (ha's `repOf` at the history's coins, `Join.lean`), `alpha := alphaOf` (ho);
* `rows := rowsOf`, `acc := accOf` (ho: the openings read off the layers);
* `Sk := skOf own sel`, `cmt := cmtOf bc`, `wmsg := wmsgOf dec`, `msg := msgOf dec` (hc);
* `files := filesOf spec st cap lig` (hf).

`HResL` and `HZeroL` (`Rep.lean`) read the parts' outputs in the layer, as the openings' are (`HUL`);
`vbridgeRead_of_layers` is `vbridgeRead_of_family` at `HFilesR`: the files are valid once they run (`hFilesR_of`), so
the runs come first.

`vbridgeRead_of_lanes` is `VBridgeRead` at any family whose fields satisfy the defining equations, its obligations
discharged by the lanes' theorems; `vbridgeRead_assembled` is `VBridgeRead` at `assembled` and the decoder `decOf`,
every defining equation by `rfl` and `hmsg` by `msgsAre_assembled`. Their hypotheses are the lanes' named facts, each a
`Prop` whose docstring names the check or lane that establishes it.
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

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

/-- **`v-bridge` at the read wires, the parts' and openings' outputs read in the layers**: `vbridgeRead_of_family` with
`HFilesR`, valid once they run, in place of `HFiles`. -/
theorem vbridgeRead_of_layers (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray)
    (N k Rw : ℕ) {GS : Flock.RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hst : F.HStruct GS gs)
    (hsh : F.HShape dg N k Rw GS) (hres : F.HResL dg N k Rw gs) (hu : F.HUL dg N k Rw) (hrd : F.HReads dg N k Rw)
    (hsch : F.HSch) (hfiles : F.HFilesR dg N k Rw) (hruns : F.HRuns dg N k Rw) (hcm : F.HCommits)
    (hdec : F.VDecodesRead) (hfull : F.VFullRead) :
    VBridgeRead (I := ℐ) F.zs dg N k Rw F.x F.wmsg F.Sk F.cmt F.msg :=
  ⟨fun R ω t R₂ hc τ o h₀ => by
    have h : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o := ⟨h₀, hc⟩
    have hr := hruns R ω t R₂ τ o h fun r => by
        obtain ⟨⟨v, info, kappa, hv, hvs⟩, hw, hc⟩ := hsh R ω t R₂ τ o h r
        exact ⟨FlockVBridge.OfVStar.algebraHolds_of_decodeL (F.zs R ω t R₂) (layers F.zs dg N k Rw R ω t R₂ τ)
            (fun s => (h.1 s).2) hst.1 (Flock.RecAlgebra.structureOf_bitsOk hst.1) hv hw hc hvs hst.2 F.sess
            (fun p => F.un p r) (fun p => cp p r) (hres R ω t R₂ τ o h r),
          FlockVBridge.AccChain.accHolds_of_units
            (unitSums_of_opensAtL (F.zs R ω t R₂) (layers F.zs dg N k Rw R ω t R₂ τ) F.sh _ _ F.sessO
              (fun ℓ j => F.unO ℓ r j) (fun ℓ j => cpO ℓ r j) (hu R ω t R₂ τ o h r))
            (F.coefsAre R ω t _ r) (F.consAre R ω t _ r) (hrd R ω t R₂ τ o h r)⟩
    exact (F.files R ω t (layers F.zs dg N k Rw R ω t R₂ τ)).accepts regs (F.x R ω) T CT sch (hsch R ω t _) _
      (hfiles R ω t R₂ τ o h hr) hr,
    hcm, hdec, hfull⟩

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
`hRowKeys_of`, `hRowsAgree_of` at `rowsShared` and `ReadsUnsegmented`), `HUL` (`hUL_of`, at struct's `hOpenUnits_of` and reg's `hOneOpen_of`), `HReads`
(`hReads_of`, at struct's `nOpen_pos`), `HSch` (`hSch_of`), `HFilesR` (`hFilesR_of`, at `MsgsAre`, `HLeaf` and
`CoinTys`), `HRuns` (`hRuns_of`, at `extrasAre_of` and `ptsOf_at`, `coins_of`, `combFolds_at`), `Encodes` (enc's
`hEnc_of`), `HCommits` (`hCommits_of`), `VDecodesRead` (`vDecodesRead_of`), `VFullRead` (`vFullRead_of`). -/
theorem vbridgeRead_of_lanes (dec : ℕ → ((j : Fin F.stmts.size) → (F.Vs j).Bits) → (ℐ).Msg)
    (hrep : F.rep₀ = F.repOf (F.coinFiles spec st cap lig)) (hrows : F.rows = F.rowsOf)
    (hacc : F.acc = F.accOf) (hF : F.files = F.filesOf spec st cap lig) (hSk : F.Sk = skOf F.Vs own sel)
    (hcmt : F.cmt = cmtOf F.Vs bc) (hw : F.wmsg = wmsgOf F.Vs F.Sk (ℐ).r dec) (hm : F.msg = msgOf F.Vs F.Sk dec)
    (hne : Nonempty (Reg × L.Ω × Reg₂)) (honame : F.HOpenName) (hcount : F.HPartCount) (hname : F.HUnitName)
    (hv : F.HVCoins dg N k Rw)
    (hlay : F.OneInLayer dg N k Rw) (hrr : F.RowsRegistered dg N k Rw) (hrck : F.RowColKeys dg N k Rw)
    (hsame : F.SameRow dg N k Rw) (hseg : F.ReadsUnsegmented dg N k Rw) (hz : F.HZeroL dg N k Rw)
    (hcoef : F.HCoefCols dg N k Rw) (V : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → ℕ → Array F128)
    (hV : F.HAccRows dg N k Rw V) (hE : F.HEnds dg N k Rw V)
    (hset : ∀ R ω, SetupRealizes (st R ω) ((F.x R ω).toStatement regs) sch cls.ptLocal cls.mPts)
    (hshp : ∀ R ω, ShapeIs F.sh (st R ω) sch) (hcl : F.ClaimsAt) (hfold : ∀ R ω, FoldsOk (st R ω) F.sh)
    (hco : ∀ R ω, CoinsAgree (st R ω) F.sh sch (2 * ((F.x R ω).toStatement regs).regions.length))
    (hcapS : F.AtAccepted dg N k Rw fun R ω _ l => CapShaped (st R ω) sch (cap l))
    (hligS : F.AtAccepted dg N k Rw fun R ω _ l => ∀ r, LigShaped (st R ω) sch (lig l r))
    (hregs : ∀ R ω, ((F.x R ω).toStatement regs).regions.length < 2 ^ 48)
    (hCT : CoinTys cls.mPts F.sh CT) (hleaf : F.HLeaf dg N k Rw) (hmsgs : F.MsgsAre dg N k Rw spec st cap lig)
    (hopen : F.AtAccepted dg N k Rw fun R ω t l => ∀ r,
      OpenedAll (st R ω) sch F.sh ((F.tableSrc spec st cap lig R ω t l).proofAt r) (F.rep R ω t l r))
    (hpub : F.PubCommits bc) :
    VBridgeRead (I := ℐ) F.zs dg N k Rw F.x F.wmsg F.Sk F.cmt F.msg := by
  obtain ⟨hsome, hcov, hU, hn⟩ := F.structure_of hne honame hcount
  obtain ⟨GS, gs, hst⟩ := F.hStruct_of hne hsome hname
  have hfit := F.hFit_of dg N k Rw spec st cap lig hrep hst hset (F.claimsIs_of hcl)
  have hxs : F.AtAccepted dg N k Rw fun R ω t l =>
      (F.tableSrc spec st cap lig R ω t l).ExtrasAre T CT cls ((F.wmsg l).zip (t.map Prod.snd)) :=
    fun R ω t _ _ _ h => F.extrasAre_of spec st cap lig hrep hset R ω t _
      (F.ptsOf_at dg N k Rw spec st cap lig hCT hleaf hmsgs h)
  have henc := F.hEnc_of dg N k Rw spec st cap lig hset hshp hfit hxs hcapS hligS hregs
  exact vbridgeRead_of_layers F dg N k Rw hst (F.hShape_of dg N k Rw hrep hst hfit)
    (F.hResL_of dg N k Rw hrep hst hcov hname (F.hRowKeys_of dg N k Rw hrr hrck)
      (F.hRowsAgree_of dg N k Rw hrr hrck hsame (F.rowsShared dg N k Rw hseg)) (F.hOne_of dg N k Rw hlay) hv hz)
    (F.hUL_of dg N k Rw hrows hacc hU (F.hOneOpen_of dg N k Rw hlay) hcoef V hV)
    (F.hReads_of dg N k Rw hacc hn V hV hE)
    (F.hSch_of spec st cap lig hF) (F.hFilesR_of dg N k Rw spec st cap lig hF hrep hset henc hCT hleaf hmsgs)
    (F.hRuns_of dg N k Rw spec st cap lig hF hset hshp henc hxs
      (fun R ω t _ _ _ _ => F.coins_of spec st cap lig hrep hco R ω t _)
      (fun R ω t _ _ _ _ => F.combFolds_at spec st cap lig hrep hfold R ω t _) hopen)
    (F.hCommits_of own sel bc hSk hcmt hpub) (F.vDecodesRead_of dec hw hm)
    F.vFullRead_of

/-- **The family at the lanes' definitions**: the coins' family (`coinFamily`: `rep₀` `repOf coinFiles`, `rows` and
`acc` the layers', `alpha` `alphaOf`), with `Sk`, `cmt`, `wmsg` and `msg` hc's at `own`, `sel`, `bc` and the decoder
`decOf`, then `files` `filesOf` at that `rep`. -/
noncomputable def assembled : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂ :=
  let F₂ : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂ :=
    { F.coinFamily spec st cap lig with
      wmsg := wmsgOf F.Vs (skOf F.Vs own sel) (ℐ).r (F.decOf dg N k Rw spec st cap lig own sel)
      Sk := skOf F.Vs own sel
      cmt := cmtOf F.Vs bc
      msg := msgOf F.Vs (skOf F.Vs own sel) (F.decOf dg N k Rw spec st cap lig own sel) }
  { F₂ with files := F₂.filesOf spec st cap lig }

/-- **`hmsg` at `assembled`**: one message per inner round, round `i`'s the table's (`decOf_spec`). -/
theorem msgsAre_assembled (hread : F.MsgsRead dg N k Rw spec st cap lig own sel)
    (htys : F.MsgTys dg N k Rw spec st cap lig) :
    (F.assembled dg N k Rw spec st cap lig own sel bc).MsgsAre dg N k Rw spec st cap lig := by
  intro R ω t R₂ τ o h
  refine ⟨?_, fun i hi => ?_⟩
  · show (wmsgOf F.Vs (skOf F.Vs own sel) (ℐ).r (F.decOf dg N k Rw spec st cap lig own sel) _).length = _
    simp [wmsgOf]
  · show ((wmsgOf F.Vs (skOf F.Vs own sel) (ℐ).r (F.decOf dg N k Rw spec st cap lig own sel)
        (layers F.zs dg N k Rw R ω t R₂ τ))[i]?).map (roundMsgs T) =
      some (((F.coinFamily spec st cap lig).tableMsgs spec st cap lig R ω t
        (layers F.zs dg N k Rw R ω t R₂ τ))[i]?.getD [])
    rw [wmsgOf, List.getElem?_map, List.getElem?_range hi, Option.map_some]
    exact congrArg some (F.decOf_spec dg N k Rw spec st cap lig own sel hread htys h i)

/-- **VBridge G's read conjunct at the lanes' definitions**: `VBridgeRead` at `skOf`'s commit strings, `cmtOf`'s
digests and `wmsgOf`'s and `msgOf`'s messages at the decoder `decOf`, at the family `assembled` (`vbridgeRead_of_lanes`,
every defining equation by `rfl`, `hmsg` by `msgsAre_assembled`). The hypotheses that read a field `assembled` defines
(`rep`, `coef`, `ops`, `files`, `Sk`) are read at `assembled`; the others at `F`, whose statements, sessions and
registrations `assembled` keeps. -/
theorem vbridgeRead_assembled
    (hne : Nonempty (Reg × L.Ω × Reg₂)) (honame : F.HOpenName) (hcount : F.HPartCount) (hname : F.HUnitName)
    (hv : (F.assembled dg N k Rw spec st cap lig own sel bc).HVCoins dg N k Rw)
    (hlay : F.OneInLayer dg N k Rw) (hrr : F.RowsRegistered dg N k Rw) (hrck : F.RowColKeys dg N k Rw)
    (hsame : F.SameRow dg N k Rw) (hseg : F.ReadsUnsegmented dg N k Rw) (hz : F.HZeroL dg N k Rw)
    (hcoef : (F.assembled dg N k Rw spec st cap lig own sel bc).HCoefCols dg N k Rw)
    (V : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → ℕ → Array F128) (hV : F.HAccRows dg N k Rw V)
    (hE : (F.assembled dg N k Rw spec st cap lig own sel bc).HEnds dg N k Rw V)
    (hset : ∀ R ω, SetupRealizes (st R ω) ((F.x R ω).toStatement regs) sch cls.ptLocal cls.mPts)
    (hshp : ∀ R ω, ShapeIs F.sh (st R ω) sch) (hcl : F.ClaimsAt) (hfold : ∀ R ω, FoldsOk (st R ω) F.sh)
    (hco : ∀ R ω, CoinsAgree (st R ω) F.sh sch (2 * ((F.x R ω).toStatement regs).regions.length))
    (hcapS : F.AtAccepted dg N k Rw fun R ω _ l => CapShaped (st R ω) sch (cap l))
    (hligS : F.AtAccepted dg N k Rw fun R ω _ l => ∀ r, LigShaped (st R ω) sch (lig l r))
    (hregs : ∀ R ω, ((F.x R ω).toStatement regs).regions.length < 2 ^ 48)
    (hCT : CoinTys cls.mPts F.sh CT) (hleaf : F.HLeaf dg N k Rw)
    (hread : F.MsgsRead dg N k Rw spec st cap lig own sel) (hT : TysIn T sch.ext)
    (hopen : F.AtAccepted dg N k Rw fun R ω t l => ∀ r,
      OpenedAll (st R ω) sch F.sh
        (((F.assembled dg N k Rw spec st cap lig own sel bc).tableSrc spec st cap lig R ω t l).proofAt r)
        ((F.assembled dg N k Rw spec st cap lig own sel bc).rep R ω t l r))
    (hpub : (F.assembled dg N k Rw spec st cap lig own sel bc).PubCommits bc) :
    VBridgeRead (I := ℐ) F.zs dg N k Rw F.x
      (wmsgOf F.Vs (skOf F.Vs own sel) (ℐ).r (F.decOf dg N k Rw spec st cap lig own sel)) (skOf F.Vs own sel)
      (cmtOf F.Vs bc) (msgOf F.Vs (skOf F.Vs own sel) (F.decOf dg N k Rw spec st cap lig own sel)) :=
  (F.assembled dg N k Rw spec st cap lig own sel bc).vbridgeRead_of_lanes dg N k Rw spec st cap lig own sel bc
    (F.decOf dg N k Rw spec st cap lig own sel) rfl rfl rfl rfl rfl rfl rfl rfl
    hne honame hcount hname hv hlay hrr hrck hsame hseg
    hz hcoef V hV hE hset hshp hcl hfold hco hcapS hligS hregs hCT hleaf
    (F.msgsAre_assembled dg N k Rw spec st cap lig own sel bc hread (F.msgTys_of dg N k Rw spec st cap lig hT))
    hopen hpub

end VStarFamily

end family

end FlockVBridge.Family
