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

`HRes` and `HZero` are false at an accepted outer proof with a part (`hRes_false`, `hZero_false`: no output column is a
registered read), so the parts' outputs are read in the layer, as the openings' are (`HUL`):

* `HResL`: `HRes` at `ResZeroAtL`; `hResL_of` gives it as `hRes_of` gives `HRes`, from `HZeroL` in place of `HZero`;
* `vbridgeRead_of_layers`: `vbridgeRead_of_family` at `HResL` (`algebraHolds_of_decodeL`) and `HUL`.

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
open FlockVBridge.OfVStar (rowCol vBase partRep partBase)
open FlockVBridge.Decode (RegBit sub vOf)
open FlockVBridge.PartS (wIdx)

section family

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 leaf512 cls regs (schedOf sch)

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ)

/-- **`hres` with the outputs read in the layer**: at every accepted outer proof, rep `r`'s instance of part `p` holds
part `p` at the rep's values, its outputs 0 in the layer (`ResZeroAtL`). -/
def HResL (gs : Array (Array ℕ × Array ℕ × Array ℕ)) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r, ∀ p < gs.size,
    ResZeroAtL (F.zs R ω t R₂ (F.sess p)) (layers F.zs dg N k Rw R ω t R₂ τ (F.sess p)) (F.un p r) (cp p r) F.sh gs p
      (el (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).w)
      (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).rounds (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).extras
      (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).comb

/-- **The parts' outputs are 0 in the layer.** At an accepted outer proof, every output column of rep `r`'s copy of
part `p` is 0 in the layer (`copyVal`). V* registers each part's `out` as reads of the zero row `rec-residuals-zero`
(`rec_vstage.py`), but a registered read is a row of an input port and no output column is one (`not_regOut`), so
`HZero`, which asks for a registered 0 there, is false (`hZero_false`), and acceptance (`Accepted`) does not reach
these columns. What makes them 0 in the layer is the registration lane's to supply: a conjunct of `ZkOuter.Good` at the
output gates, as `PubAgree` is at single gates, or V*'s residual unit with `out` as an input row. -/
def HZeroL : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r p, p < (partsOf F.sh).size →
    ∀ S ns nv, RecResiduals.part F.sh p = .ok (S, ns, nv) → ∀ hj : cp p r < (F.Vs (F.sess p)).c.upv,
      ∀ e < 16 * RecResiduals.RES_OUT_WORDS,
        copyVal (keyInst (F.Vs (F.sess p)).hU (F.Vs (F.sess p)).nv
            (Layout.outsOf (F.Vs (F.sess p)).c (F.Vs (F.sess p)).hU (F.Vs (F.sess p)).nv) (F.un p r))
          (layers F.zs dg N k Rw R ω t R₂ τ (F.sess p)) hj (partBase S ns nv + e) = false

/-- **`HResL` at `repOf`'s `w`**: `hRes_of`'s argument, the outputs 0 in the layer (`HZeroL`) in place of registered
zero. -/
theorem hResL_of {tf : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin F.stmts.size) → (F.Vs j).Bits) → TableFiles cls}
    (hrep : F.rep₀ = F.repOf tf) {GS : Flock.RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hst : F.HStruct GS gs) (hcov : F.HCover) (hname : F.HUnitName) (hseen : F.HRowSeen)
    (hkeys : F.HRowKeys dg N k Rw) (hagree : F.HRowsAgree dg N k Rw) (hone : F.HOne dg N k Rw)
    (hv : F.HVCoins dg N k Rw) (hz : F.HZeroL dg N k Rw) : F.HResL dg N k Rw gs := by
  intro R ω t R₂ τ o h r p hp
  have hP : partsOf F.sh = gs := partsOf_eq hst.1 hst.2
  have hpP : p < (partsOf F.sh).size := by rw [hP]; exact hp
  obtain ⟨j, nm, pt, hj⟩ := hcov p hpP
  have hsj : F.sess p = j := F.hsess p j ⟨nm, pt, hj⟩
  have hjs : F.stmts[F.sess p] = .inner p nm pt := by subst hsj; exact hj
  have hnm : (F.Vs (F.sess p)).c.unit.name = RecResiduals.unitName F.vsh.m F.vsh.kLog F.vsh.claims p :=
    (hname _ p nm pt hjs).trans (statements_name F.hstmts _ hjs)
  have hchk : RecResiduals.check (F.Vs (F.sess p)).c = .ok () :=
    (parse_facts (F.zs R ω t R₂ (F.sess p)).hc).recResiduals
  have hpar := resParams_of_check hchk hnm
  have hupv : ∀ p', 0 < (F.Vs (F.sess p')).c.upv := fun p' =>
    upv_pos (F.zs R ω t R₂ (F.sess p')).hc (F.Vs (F.sess p')).hscope
  obtain ⟨S, ns, nv, hpart, -⟩ := FlockVBridge.OfVStar.check_part hchk hpar F.hsh
  obtain ⟨-, hns, hnv⟩ := FlockVBridge.OfVStar.part_eq hst.1 hst.2 hp hpart
  have hgp : (partsOf F.sh)[p]! = gs[p] := by rw [hP, getElem!_pos gs p hp]
  have hnsp : nsOf F.sh p = ns := by rw [nsOf, hgp, hns]
  have hnvp : nvOf F.sh p = nv := by rw [nvOf, hgp, hnv]
  have hwI : wIOf F.sh p = wIdx (portSizes F.sh) gs[p].2.1 := by rw [wIOf, hgp]
  have hW : (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).w = F.wOf (layers F.zs dg N k Rw R ω t R₂ τ) r := by
    show (F.rep₀ R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).w = _
    rw [hrep]
    rfl
  refine ⟨hp, hupv p, hone R ω t R₂ τ o h r p hpP, hchk, unitLog_le (F.Vs (F.sess p)).hscope, _, _, _, hpar, F.hsh,
    S, ns, nv, hpart, ?_, ?_, ?_⟩
  · intro q hq
    show RegBit (F.zs R ω t R₂ (F.sess p)) (F.un p r) 0 _ _
    rw [hW, ← hwI]
    have hsz : (wIOf F.sh p).size = ns.sum := by rw [hwI, FlockVBridge.OfVStar.wIdx_size, hns]
    have ha : q / 128 < (wIOf F.sh p).size := by rw [hsz]; omega
    have he : q % 128 < 128 := Nat.mod_lt _ (by norm_num)
    have hi : (wIOf F.sh p)[q / 128] < nwOf F.sh := by
      rw [nwOf_eq hst.1, ← part_nw hst.1 hpart]
      exact wIdx_lt (portSizes F.sh) ((partsOf F.sh)[p]!.2.1) ha
    have hex : ∃ pk, F.Seen r (wIOf F.sh p)[q / 128] (q % 128) pk :=
      hseen r _ _ he ⟨p, q / 128, hpP, ha, rfl⟩
    obtain ⟨hp0, ha0, hi0, hio0⟩ := F.locOf_spec hex
    have hsz0 : (wIOf F.sh (F.locOf r (wIOf F.sh p)[q / 128] (q % 128)).1).size =
        (nsOf F.sh (F.locOf r (wIOf F.sh p)[q / 128] (q % 128)).1).sum := by
      rw [wIOf, nsOf, FlockVBridge.OfVStar.wIdx_size]
    obtain ⟨b0, hb0⟩ := hkeys R ω t R₂ τ o h r _ hp0
      (128 * (F.locOf r (wIOf F.sh p)[q / 128] (q % 128)).2 + q % 128) (by rw [← hsz0]; omega)
    have hdec := bitAt_of_regBit (F.zs R ω t R₂ _) (h _).2.2 hb0 hio0 (hupv _)
    have hb := hagree R ω t R₂ τ o h r _ p _ (q / 128) (q % 128) b0 hp0 hpP ha0 ha hi0 he hb0
    have hval : FlockVBridge.wbit (sub (wIOf F.sh p) (el (F.wOf (layers F.zs dg N k Rw R ω t R₂ τ) r))) q = b0 := by
      unfold FlockVBridge.wbit
      rw [FlockVBridge.Decode.sub_lt _ _ _ ha, F.el_wOf _ r hi, FlockVBridge.bitOf_ofBits _ he]
      exact hdec
    rw [hval]
    have hq : 128 * (q / 128) + q % 128 = q := Nat.div_add_mod q 128
    rw [hq, rowColOf, hnsp, hnvp] at hb
    exact hb
  · intro q hq
    have hq' : q < 128 * nvOf F.sh p := by rw [hnvp]; exact hq
    have := hv R ω t R₂ τ o h r p hpP q hq'
    rw [hnsp, hnvp, hgp] at this
    exact this
  · intro e he
    exact hz R ω t R₂ τ o h r p hpP S ns nv hpart (hupv p) e he

end VStarFamily

/-- **`v-bridge` at the read wires, the parts' and openings' outputs read in the layers**: `vbridgeRead_of_family` with
`HResL` in place of `HRes` (`algebraHolds_of_decodeL`) and `HUL` in place of `HU` (`unitSums_of_opensAtL`). -/
theorem vbridgeRead_of_layers (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray)
    (N k Rw : ℕ) {GS : Flock.RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hst : F.HStruct GS gs)
    (hsh : F.HShape dg N k Rw GS) (hres : F.HResL dg N k Rw gs) (hu : F.HUL dg N k Rw) (hrd : F.HReads dg N k Rw)
    (hsch : F.HSch) (hfiles : F.HFiles dg N k Rw) (hruns : F.HRuns dg N k Rw) (hcm : F.HCommits)
    (hdec : F.VDecodesRead) (hfull : F.VFullRead) :
    VBridgeRead (I := ℐ) F.zs dg N k Rw F.x F.wmsg F.Sk F.cmt F.msg :=
  ⟨fun R ω t R₂ τ o h =>
    (F.files R ω t (layers F.zs dg N k Rw R ω t R₂ τ)).accepts regs (F.x R ω) T CT sch (hsch R ω t _) _
      (hfiles R ω t R₂ τ o h) (hruns R ω t R₂ τ o h fun r => by
        obtain ⟨⟨v, info, kappa, hv, hvs⟩, hw, hc⟩ := hsh R ω t R₂ τ o h r
        exact ⟨algebraHolds_of_decodeL (F.zs R ω t R₂) (layers F.zs dg N k Rw R ω t R₂ τ)
            (fun s => (h s).2) hst.1 (Flock.RecAlgebra.structureOf_bitsOk hst.1) hv hw hc hvs hst.2 F.sess
            (fun p => F.un p r) (fun p => cp p r) (hres R ω t R₂ τ o h r),
          FlockVBridge.AccChain.accHolds_of_units
            (unitSums_of_opensAtL (F.zs R ω t R₂) (layers F.zs dg N k Rw R ω t R₂ τ) F.sh _ _ F.sessO
              (fun ℓ j => F.unO ℓ r j) (fun ℓ j => cpO ℓ r j) (hu R ω t R₂ τ o h r))
            (F.coefsAre R ω t _ r) (F.consAre R ω t _ r) (hrd R ω t R₂ τ o h r)⟩),
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
`hRowKeys_of`, `hRowsAgree_of`), `HUL` (`hUL_of`, at struct's `hOpenUnits_of` and reg's `hOneOpen_of`), `HReads`
(`hReads_of`, at struct's `nOpen_pos`), `HSch` (`hSch_of`), `HFiles` (`hFiles_of`), `HRuns` (`hRuns_of`, at
`extrasAre_of`, `coins_of`, `combFolds_at`), `Encodes` (enc's `hEnc_of`), `HCommits` (`hCommits_of`), `VDecodesRead`
(`vDecodesRead_of`), `VFullRead` (`vFullRead_of`). -/
theorem vbridgeRead_of_lanes (dec : ℕ → ((j : Fin F.stmts.size) → (F.Vs j).Bits) → (ℐ).Msg)
    (hrep : F.rep₀ = F.repOf (F.coinFiles spec st cap lig F.wmsg)) (hrows : F.rows = F.rowsOf)
    (hacc : F.acc = F.accOf) (hF : F.files = F.filesOf spec st cap lig) (hSk : F.Sk = skOf F.Vs own sel)
    (hcmt : F.cmt = cmtOf F.Vs bc) (hw : F.wmsg = wmsgOf F.Vs F.Sk (ℐ).r dec) (hm : F.msg = msgOf F.Vs F.Sk dec)
    (hne : Nonempty (Reg × L.Ω × Reg₂)) (honame : F.HOpenName) (hcount : F.HPartCount) (hname : F.HUnitName)
    (hseen : F.HRowSeen) (hv : F.HVCoins dg N k Rw)
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
  exact vbridgeRead_of_layers F dg N k Rw hst (F.hShape_of dg N k Rw hrep hst hfit)
    (F.hResL_of dg N k Rw hrep hst hcov hname hseen (F.hRowKeys_of dg N k Rw hrr hrck)
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
    (hseen : F.HRowSeen) (hv : (F.assembled spec st cap lig own sel bc dec).HVCoins dg N k Rw)
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
    rfl rfl rfl rfl rfl rfl rfl rfl hne honame hcount hname hseen hv hlay hrr hrck hsame hshared hz hcoef V hV hE hset
    hshp hcl hfold hco hcapS hligS hregs htr hopen hpub hsel hS hu4

end VStarFamily

end family

end FlockVBridge.Family
