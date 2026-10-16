import Proofs.Flock.C3.Sparse.SessionZero
import Proofs.Flock.C3.Sparse.SessionKeys
import Proofs.Flock.C3.RegLevels.Reached

/-!
# `sparseSessionAt_of_zero` from the verifier's checks

`sparseSessionAt_of_checks`: rep `i`'s sparse phase in V*'s sessions at copy 0 of unit `i` of the RecSparse session `s`,
with every premise the checks give discharged: the session has two units (`SessStmt`, `VStarParsesScoped`, RecSparse's
two instances), so unit `i` exists and its words are `wSparseAt`'s; `hkeys` from `hkeys_of_rows` (the circuit's keys,
`sparseColKeys_of_check`; the reads file, `checkOwn` in the statement check; the rows registered, `RowsRegisteredC3`);
`htau` is vacuous (check 5's shape has one type, `tLog_zero_of_checkInnerStmt`); and `hout`, the session's `out` read,
from check 5's `checkZero` (the reads file names `out`, `own_of_checkRows`), `RowsRegisteredC3`, `ReadsOwnAt`
(`readOne_of`) and `readsUnsegC3`. What stays: the layer's constant (`hone`, G's `OneInLayer` at `s`), `v`'s registered
row (`hv`), the unit's output layout (`hlaid`, `SparseOutsLaid`, which `RecSparse.check` does not read) and being off
`ValueColl` (`hnc`).
-/

namespace FlockC3.Sparse

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled FlockC3.Compose
open FlockVBridge (wbit)
open FlockVBridge.Decode (RegBit)
open FlockVBridge.Sparse (SCoins verifierRow vBaseS)

/-- Rep `i`'s words in session `s`: copy 0 of unit `i` at the message columns (`wSparse`), none past its units. -/
def wSparseAt {S : ℕ} (Vs : Fin S → VStmt) (sh : SparseShape) (mp : List (String × ℕ))
    (Lb : (j : Fin S) → (Vs j).Bits) (s : Fin S) (i : Fin 2) : Array F128 :=
  if h : i.val < (Vs s).nv then wSparse Vs sh mp Lb s ⟨i.val, h⟩ else #[]

section Family

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

/-- **(n7) Rep `i`'s sparse phase from the verifier's checks** (`sparseSessionAt_of_zero` at unit `i` of session `s`;
module doc): at an accepted outer proof off `ValueColl`, where the layer carries 1 at the constant gate (`hone`), the
unit's `v` row is registered as the verifier's row at `c` (`hv`) and its output is laid as V* lays it (`hlaid`). -/
theorem sparseSessionAt_of_checks (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂)
    {rows : Option (Array Flock.Registered.Port)} {wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String}
    {reg : Lean.Json} {coins : String → Option Flock.RecCoins.Source} (hchk : CoinsAccepted F rows wr reg coins)
    (hca : FlockC3.PubRows.ChecksAccepted F rows wr reg) {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    (hro : FlockC3.HPort.ReadsOwnAt F dg N k Rw) (hrr : F.RowsRegisteredC3 dg N k Rw)
    (hst : SessStmt F.ish F.bind F.zs dg N k Rw) (hsp : FlockC3.VStarList.SparseIs F.ish hp)
    (hell : hp.EllOk (execClassAt st).cls.kLog) (hi : RecVStar.checkInnerStmt F.ish st = .ok ())
    {s : Fin (vstarCount F.ish)} {sh : Flock.RecSparse.Shape} {hR : ℕ} (hs : F.stmts[(s : ℕ)]? = some (.sparse sh hR))
    (hlaid : SparseOutsLaid (F.Vs s).c (unitShape (holoAt st hp)) 0)
    {R : Reg} {ω : L.Ω} {t : List (Cm × (holoInnerAt T CT st hp).Coin)} {R₂ : Reg₂}
    {τ : (j : Fin (vstarCount F.ish)) → Strategy ((F.zs R ω t R₂ j).game dg N)}
    {o : (j : Fin (vstarCount F.ish)) → Reg' × Finset (Fin (F.Vs j).nv) × Exec.Transcript}
    (hA : Accepted (I := holoInnerAt T CT st hp) F.zs dg N k Rw R ω t R₂ τ o)
    (hnc : ¬ ValueColl F.zs (valueSrcAll (fun t => coins (wr t).2)
      (fun t => Flock.Guarantees.c3Of F.ish (innerCommitsAt (wr t).1 (wr t).2 st reg).toOption)) R ω t R₂)
    (hone : layers F.zs dg N k Rw R ω t R₂ τ s
      (Layout.oneG (c := (F.Vs s).c) (hU := (F.Vs s).hU) (n := (F.Vs s).nv)) = true)
    (i : Fin 2) {C : Flock.RecCoins.C3}
    (hC : ∀ cs, innerCommitsAt (wr t).1 (wr t).2 st reg = .ok cs → Flock.Guarantees.c3Of F.ish (some cs) = some C)
    {c : SCoins} (hcs : c.cs = csOf C (Flock.RecSparse.unitName (unitShape (holoAt st hp))) i.val)
    (hv : ∀ u : Fin (F.Vs s).nv, u.val = i.val → ∀ q < 128 * Flock.RecSparse.nv (unitShape (holoAt st hp)),
      RegBit (F.zs R ω t R₂ s) u 0 (vBaseS (unitShape (holoAt st hp)) + q)
        (wbit ((verifierRow (holoAt st hp).cls.shape c)[·]!) q)) :
    SparseSessionAt (F.zs R ω t R₂) (layers F.zs dg N k Rw R ω t R₂ τ) (holoAt st hp) C
        (Flock.RecSparse.unitName (unitShape (holoAt st hp))) i
        (wSparseAt F.Vs (holoAt st hp).cls.shape (Flock.RecSparse.messagePorts (unitShape (holoAt st hp)))
          (layers F.zs dg N k Rw R ω t R₂ τ) s i) c ∧
      ∃ (s' : Fin (vstarCount F.ish)) (un : Fin (F.Vs s').nv) (j : ℕ) (hj : j < (F.Vs s').c.upv), un.val = i.val ∧
        SparseCopyAt (F.zs R ω t R₂) (layers F.zs dg N k Rw R ω t R₂ τ) (holoAt st hp)
          (wSparseAt F.Vs (holoAt st hp).cls.shape (Flock.RecSparse.messagePorts (unitShape (holoAt st hp)))
            (layers F.zs dg N k Rw R ω t R₂ τ) s i) c s' un j hj := by
  have hnv : (F.Vs s).nv = 2 := by
    obtain ⟨ss, sv, cf, hss, hj, -, hcf, hc⟩ := hst R ω t R₂ τ o hA s
    obtain rfl : ss = F.stmts := Except.ok.inj (hss.symm.trans F.hstmts)
    rw [hs] at hj
    cases hj
    exact (F.hob s _ _ cf _ F.hstmts hs hcf hc (F.Vs s).hscope).2
  have hu : i.val < (F.Vs s).nv := by
    rw [hnv]
    exact i.isLt
  have hshS : sh = unitShape (holoAt st hp) := by
    rw [unitShape_holoAt hell]
    exact Option.some.inj ((FlockC3.PubRows.sparse_of_name F.hstmts (Array.mem_of_getElem? hs) rfl).symm.trans hsp)
  subst hshS
  obtain ⟨sj, f, c3, hc, -, -, hck, hpar⟩ := sparseCheck_at F hchk hsp hell hs ⟨(o s).2.2, (hA.1 s).1⟩
  have hname : (F.Vs s).c.unit.name = Flock.RecSparse.unitName (unitShape (holoAt st hp)) :=
    (FlockVBridge.Sparse.unitName_params hpar).symm
  have hckS := (parse_facts (F.zs R ω t R₂ s).hc).recSparse
  obtain ⟨-, -, -, hn⟩ := FlockVBridge.Sparse.check_sparse hckS hpar
  have hn' : Flock.HmNets.netOf (Flock.RecSparse.unitName (unitShape (holoAt st hp)))
      (Flock.RecSparse.unitLaid (unitShape (holoAt st hp))) = .ok (F.Vs s).c.unit := hn
  have hown : OwnNamesMsgs (F.zs R ω t R₂ s).own (unitShape (holoAt st hp)) := by
    obtain ⟨_, -, -, _, _, hset, hfm⟩ := hca R ω t R₂ s ⟨_, (hA.1 s).1⟩
    obtain ⟨h0, -⟩ := setup_zero_c (F.zs R ω t R₂ s).ht (F.zs R ω t R₂ s).hTags.1 (F.zs R ω t R₂ s).hcH hset
    obtain ⟨-, -, ss, sv, hss₁, hj₁, -, -, -, ho, -⟩ := checkStmt_ok (array_forM_ok hfm 0 h0)
    obtain rfl : ss = F.stmts := Except.ok.inj (hss₁.symm.trans F.hstmts)
    rw [hs] at hj₁
    cases hj₁
    exact FlockC3.Sparse.sparseOwnNames_of_checkOwn ho
  have hns : (unitShape (holoAt st hp)).stacked = false := by
    have htl : (unitShape (holoAt st hp)).tLog = 0 := by
      rw [unitShape_holoAt hell]
      exact FlockC3.VStarList.tLog_zero_of_checkInnerStmt hi hsp
    unfold Flock.RecSparse.Shape.stacked
    rw [htl]
    rfl
  have hout : ∀ hq : HmOut.nIn (F.Vs s).c < (F.Vs s).c.ports.size,
      (F.zs R ω t R₂ s).rd.isRead (.inl (⟨i.val, hu⟩, ⟨_, hq⟩)) ∧
      (∀ pt ∈ (F.zs R ω t R₂ s).own, pt.name = "out" →
        pt = (F.zs R ω t R₂ s).rd.port (.inl (⟨i.val, hu⟩, ⟨_, hq⟩))) ∧
      ((F.zs R ω t R₂ s).rd.port (.inl (⟨i.val, hu⟩, ⟨_, hq⟩))).reads[i.val]? =
        some (some ((F.zs R ω t R₂ s).rd.readAt (.inl (⟨i.val, hu⟩, ⟨_, hq⟩)))) ∧
      ((F.Vs s).c.ports[HmOut.nIn (F.Vs s).c]!).segs = none := by
    intro hq
    obtain ⟨-, hnm, -⟩ := hlaid.2.2
    have hnm' : ((F.Vs s).c.ports[(⟨HmOut.nIn (F.Vs s).c, hq⟩ : Fin (F.Vs s).c.ports.size)]).name = "out" := hnm
    obtain ⟨pt, hpt, hpn⟩ : ∃ pt ∈ (F.zs R ω t R₂ s).own, pt.name = "out" := by
      obtain ⟨inn, -, hck'⟩ := sparseCheck_of_check hck (by rw [hc, hname]; exact stackedParams_sparse _)
        (by rw [hc, hname]; exact openingParams_sparse _) (by rw [hc]; exact hpar)
      obtain ⟨-, -, -, -, hz⟩ := sparse_of_check hck'
      exact FlockC3.RegLevels.own_of_checkRows hz
    have hr := hrr R ω t R₂ τ o hA s ⟨i.val, hu⟩ ⟨_, hq⟩ ⟨pt, hpt, hpn.trans hnm'.symm⟩
    obtain ⟨hone', hpos⟩ := FlockC3.RegLevels.readOne_of hro hca hA s hr
    refine ⟨hr, fun pt' hpt' hpn' => hone' pt' hpt' ?_, hpos,
      FlockC3.HPort.readsUnsegC3 F dg N k Rw R ω t R₂ τ o hA s _ _ hr⟩
    rw [hpn', getElem!_pos (F.Vs s).c.ports _ hq]
    exact hnm'.symm
  have hw : wSparseAt F.Vs (holoAt st hp).cls.shape (Flock.RecSparse.messagePorts (unitShape (holoAt st hp)))
      (layers F.zs dg N k Rw R ω t R₂ τ) s i =
      wSparse F.Vs (holoAt st hp).cls.shape (Flock.RecSparse.messagePorts (unitShape (holoAt st hp)))
        (layers F.zs dg N k Rw R ω t R₂ τ) s ⟨i.val, hu⟩ := dif_pos hu
  rw [hw]
  exact sparseSessionAt_of_zero F hchk hsp hell hs hA i ⟨i.val, hu⟩ rfl hC hcs
    (by rw [FlockVBridge.Family.oneAt_keyProg]; exact hone)
    (hkeys_of_rows F hrr hA hn' (sparseColKeys_of_check (F.Vs s).hU hckS hpar
      (FlockC3.Reps.upv_of_outer (F.zs R ω t R₂ s))) hown ⟨i.val, hu⟩)
    (hv _ rfl) (fun h1 => absurd (hns.symm.trans h1) Bool.false_ne_true) hlaid hout hnc

end Family

end

end FlockC3.Sparse
