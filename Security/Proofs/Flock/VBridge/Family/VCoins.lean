import Proofs.Flock.VBridge.Family.Registered

/-!
# VBridge G at V*'s family: the parts' `v` and `out` columns (`HVCoins`, `HZeroL`)

`HVCoins` and `HZeroL` (`Family/Rep.lean`) split into what the parts' circuits key at their `v` and `out` columns and
what the part sessions' reads files and registrants hold there:

* `vColOf_lt`: each `v` column is an input (the verifier's `RecResiduals.check`, as `rowColOf_lt`);
* `PartVCols`, `OwnNamesV`: each `v` column is keyed by its bit of the part's `v` port (the parse of the parts'
  circuits), and each part's reads file names `v` (`FlockC3.RegLevels.own_of_checkValue` at check 5's `checkValue`);
* `VRowsAre`: the registrant's bits at a part's `v` read are `vOf`'s at the rep (`Recursion.vRows_tie`: `value_tie_coll`
  at check 5's `rec-residuals` rows, off `ValueColl`);
* `hVCoins_of`: `HVCoins` from those and `RowsRegistered`;
* `outCol_committed`, `outColAt_stackCol`: a copy's gate at a returned output's column is committed (it is an output of
  the program's circuit, `Layout.outsOf`), the committed conjunct of `RegOut`;
* `PartOutCols`, `ZeroReads`: each part's `out` columns are its leaf's returned output (the parse), and the gates there
  are registered reads whose registrant's bit is 0 (`rec-residuals-zero`, `Recursion.zero_tie` off `ValueColl`);
* `hZeroL_of`: `HZeroL` from those (`copyVal_regOut` in the accepted `Good` layer).
-/

namespace FlockVBridge.OfVStar

open Flock.HmNets hiding and
open Flock FlockSoundness.Discharge.Hm
open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge

/-- **A part's `v` columns are inputs**: the laid-out part's inputs end past its `v` port (`rowCol_lt_nIn`'s, at the
`v` port's bits). -/
theorem vCol_lt_nIn {c : Flock.Circuit} (hU : Built c.unit) {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    {name : String} (hn : netOf name (RecResiduals.unitLaid S ns nv) = .ok c.unit) {q : ℕ} (hq : q < 128 * nv) :
    vBase (partRep ns nv) + q < (Rows.ofNet hU).nIn := by
  have hN : (partIn ns nv).foldl (· + ·) 0 = innerIn (partRep ns nv) := by
    rw [partIn, foldl_replicate, innerIn_partRep]
    exact Nat.mul_comm _ _
  have hi := innerIn_eq (partRep ns nv)
  have hp := le_portWords nv
  have hnv : (partRep ns nv).gf.nv = nv := rfl
  rw [hnv] at hi
  rw [Layout.ofNet_nIn_eq, inWords_unitLaid hn]
  refine lt_of_lt_of_le ?_ (Flock.HmNets.le_groupWords _)
  rw [hN]
  omega

end FlockVBridge.OfVStar

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages
open FlockVBridge.OfVStar (vBase partRep partBase)
open FlockVBridge.Decode (RegBit sub vOf)

section Copy

variable {c : Flock.Circuit} {hU : Built c.unit} {nv : ℕ}

/-- **A returned output's column, stacked**: copy `j`'s column `outCol j' t`, a computed column, is `OutColAt`'s. -/
theorem outColAt_stackCol {j : ℕ} (hj : j < c.upv) {col : ℕ} (hcol : col < (Rows.ofNet hU).w) {j' t : ℕ}
    (hoc : col = Layout.outCol c j' t) (hj' : j' < (c.leavesOut.getD j #[]).size) (ht : t < 16)
    (hlo : (Rows.ofNet hU).nIn ≤ col) (hhi : col < (Rows.ofNet hU).nIn + (Rows.ofNet hU).nComp) :
    Layout.OutColAt c hU ((Rows.ofNet hU).stackCol c.upv hj ⟨col, hcol⟩).val j j' t := by
  subst hoc
  refine ⟨hj, hj', ht, hlo, hhi, ?_⟩
  have h1 : ¬ (⟨Layout.outCol c j' t, hcol⟩ : Fin (Rows.ofNet hU).w).val < (Rows.ofNet hU).nIn := Nat.not_lt.mpr hlo
  have h2 : (⟨Layout.outCol c j' t, hcol⟩ : Fin (Rows.ofNet hU).w).val <
      (Rows.ofNet hU).nIn + (Rows.ofNet hU).nComp := hhi
  unfold FlockSoundness.Rows.stackCol
  split
  next hc => exact absurd hc h1
  next => rfl

/-- **A copy's gate at a returned output's column is committed**: it is unit `u`'s instance's gate there, which
`Layout.outsOf` returns, an output of the program's circuit. -/
theorem outCol_committed (u : Fin nv) {j : ℕ} (hj : j < c.upv) {col : ℕ} (hcol : col < (Rows.ofNet hU).w)
    {j' t : ℕ} (h : Layout.OutColAt c hU ((Rows.ofNet hU).stackCol c.upv hj ⟨col, hcol⟩).val j j' t) :
    (keyInst hU nv (Layout.outsOf c hU nv) u).wire ((Rows.ofNet hU).stackCol c.upv hj ⟨col, hcol⟩) ∈
      ((Placed.keyProg c hU nv).partition (Layout.outsOf c hU nv)).committed := by
  refine Partition.mem_committed_of_output _ ?_
  show _ ∈ Layout.outsOf c hU nv
  unfold Layout.outsOf
  exact (@Finset.mem_filter _ _ (Classical.decPred _) _ _).mpr ⟨Finset.mem_univ _, u, _,
    (Const.Prog.castW_wire (Placed.rows_circuitProg c hU nv _ _ u)
      (Const.Prog.rowsWireC (Placed.keyProg c hU nv) u) _).symm, j, j', t, h⟩

end Copy

section family

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ)

/-! ## `v` -/

/-- **A part's `v` columns are inputs**: as `rowColOf_lt`, at the `v` port's bits (`vCol_lt_nIn`). -/
theorem vColOf_lt {GS : Flock.RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hst : F.HStruct GS gs)
    (hname : F.HUnitName) (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (R₂ : Reg₂) {p : ℕ} {j : Fin F.stmts.size}
    {nm : String} {pt : VStar.Part} (hj : F.stmts[j] = .inner p nm pt) (hp : p < (partsOf F.sh).size) {q : ℕ}
    (hq : q < 128 * nvOf F.sh p) : vBase (partRep (nsOf F.sh p) (nvOf F.sh p)) + q < (Rows.ofNet (F.Vs j).hU).nIn := by
  have hP : partsOf F.sh = gs := partsOf_eq hst.1 hst.2
  have hpg : p < gs.size := by rw [← hP]; exact hp
  have hnm : (F.Vs j).c.unit.name = RecResiduals.unitName F.vsh.m F.vsh.kLog F.vsh.claims p :=
    (hname _ p nm pt hj).trans (statements_name F.hstmts _ hj)
  have hchk : RecResiduals.check (F.Vs j).c = .ok () := (parse_facts (F.zs R ω t R₂ j).hc).recResiduals
  obtain ⟨S, ns, nv, hpart, hn⟩ :=
    FlockVBridge.OfVStar.check_part hchk (resParams_of_check hchk hnm) F.hsh (F.hplain j)
  obtain ⟨-, hns, hnv⟩ := FlockVBridge.OfVStar.part_eq hst.1 hst.2 hpg hpart
  have hgp : (partsOf F.sh)[p]! = gs[p] := by rw [hP, getElem!_pos gs p hpg]
  have hnsp : nsOf F.sh p = ns := by rw [nsOf, hgp, hns]
  have hnvp : nvOf F.sh p = nv := by rw [nvOf, hgp, hnv]
  rw [hnsp, hnvp]
  exact OfVStar.vCol_lt_nIn (F.Vs j).hU hn (by rw [← hnvp]; exact hq)

/-- Obligation: discharged by the parse of the part's circuit (`parse_circuitFile` with `circuitFile`'s emit and parse
round trip, the lane that proves `HUnitName`), as `PartRowCols`. **Each `v` column of a part is keyed by its bit of the
part's `v` port.** In the circuit of part `p`'s statement, a port `qv` is named `v`, and the Δ key (`Aliased.srcKey`,
copy 0) of the unit's `v` column `vBase + q` is message bit `q` of `qv`, within its row. V*'s `InnerRepCheck` circuit
takes `v` as its input port after the row ports (`VStar.Parts`, `rec_residuals.py:396`). -/
def PartVCols : Prop :=
  ∀ (p : ℕ) (j : Fin F.stmts.size) (nm : String) (pt : VStar.Part), F.stmts[j] = .inner p nm pt →
    ∃ qv : Fin (F.Vs j).c.ports.size, (F.Vs j).c.ports[qv].name = "v" ∧
      ∀ q < 128 * nvOf F.sh p, ∃ _ : q < Layout.portRowW (F.Vs j).c qv,
        Aliased.srcKey (F.Vs j).c (0 * (Rows.ofNet (F.Vs j).hU).nIn + (vBase (partRep (nsOf F.sh p) (nvOf F.sh p)) + q)) =
          .msg qv.val (q / 8) (q % 8)

/-- Obligation: discharged by check 5 (`ChecksC3`'s `RecCoins.check` at the part's `rec-residuals` unit:
`Recursion.own_v_out`, `own_of_checkValue` at its `checkValue` at `v`). **Each part's reads file names `v`.** -/
def OwnNamesV : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ p, p < (partsOf F.sh).size →
    ∃ pt ∈ (F.zs R ω t R₂ (F.sess p)).own, pt.name = "v"

/-- Obligation: discharged at each read by `Recursion.vRows_tie`, off `ValueColl` (the guarantee's collision
disjunct): at the part's `rec-residuals` unit, check 5 accepted `checkValue` at `v` of `residualRows`' rows,
`value_tie_coll` there gives the registrant's bits as the rows', and `rows_vOf` the rows' as `vOf`'s, where the inner
session's coins, extra claims and comb are the rep's (the lane that proves `HFiles`). **The registrant's `v` is the verifier's at the rep.** At an accepted outer proof,
the registrant's bit `b` at the `v` read of rep `r`'s unit of part `p` is `wbit` of part `p`'s elements of `vOf` at the
rep's coins, extra claims and comb. -/
def VRowsAre : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r p, p < (partsOf F.sh).size →
    ∀ q' : Fin (F.Vs (F.sess p)).c.ports.size, (F.Vs (F.sess p)).c.ports[q'].name = "v" →
      ∀ b < 128 * nvOf F.sh p, ∀ hb : b < Layout.portRowW (F.Vs (F.sess p)).c q',
        (Decode.regVal (F.zs R ω t R₂ (F.sess p)) (.inl (F.un p r, q'))).2
            ⟨b, lt_of_lt_of_le hb ((Finset.le_sup (Finset.mem_univ q')).trans (Nat.le_succ _))⟩ =
          FlockVBridge.wbit (sub (partsOf F.sh)[p]!.2.2
            (vOf F.sh (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).rounds
              (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).extras
              (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).comb)) b

/-- **`HVCoins` from the parts' circuits, reads files and registrants** (`PartVCols`, `OwnNamesV`, `VRowsAre`, at the
part's statement `HCover` gives; the columns inputs by `vColOf_lt`): the column's key names the `v` port, a registered
read (`RowsRegistered`), and its bit is the registrant's. -/
theorem hVCoins_of {GS : Flock.RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hst : F.HStruct GS gs)
    (hname : F.HUnitName) (hcov : F.HCover) (hreg : F.RowsRegistered dg N k Rw) (hcols : F.PartVCols)
    (hown : F.OwnNamesV dg N k Rw) (hrows : F.VRowsAre dg N k Rw) : F.HVCoins dg N k Rw := by
  intro R ω t R₂ τ o ha r p hp q hq
  obtain ⟨j, nm, pt, hj⟩ := hcov p hp
  have hsj : F.sess p = j := F.hsess p j ⟨nm, pt, hj⟩
  subst hsj
  obtain ⟨qv, hqv, hk⟩ := hcols p _ nm pt hj
  obtain ⟨hkq, hkey⟩ := hk q hq
  obtain ⟨pv, hpv, hpvn⟩ := hown R ω t R₂ τ o ha p hp
  have hread := hreg R ω t R₂ τ o ha (F.sess p) (F.un p r) qv ⟨pv, hpv, hpvn.trans hqv.symm⟩
  exact ⟨F.vColOf_lt hst hname R ω t R₂ hj hp hq, qv, q, hkq, hkey, hread,
    hrows R ω t R₂ τ o ha r p hp qv hqv q hq hkq⟩

/-! ## `out` -/

/-- Obligation: discharged by the parse of the part's circuit (`parse_circuitFile`, the lane that proves `HUnitName`)
and its layout (`inWords_unitLaid`, `inRows_le_outStart`, as `partCols_of_resZeroAt`'s `hbase`). **A part's `out`
columns are its leaf's returned output.** In the circuit of part `p`'s statement, at every copy, column
`partBase + e` (`e < 16 · RES_OUT_WORDS`) is a computed column, `outCol (e / 16) (e % 16)` of the copy's leaf's output
`e / 16`. -/
def PartOutCols : Prop :=
  ∀ (p : ℕ) (j : Fin F.stmts.size) (nm : String) (pt : VStar.Part), F.stmts[j] = .inner p nm pt →
    ∀ S ns nv, RecResiduals.part F.sh p = .ok (S, ns, nv) → ∀ cpy < (F.Vs j).c.upv,
      ∀ e < 16 * RecResiduals.RES_OUT_WORDS,
        partBase S ns nv + e = Layout.outCol (F.Vs j).c (e / 16) (e % 16) ∧
          e / 16 < ((F.Vs j).c.leavesOut.getD cpy #[]).size ∧ (Rows.ofNet (F.Vs j).hU).nIn ≤ partBase S ns nv ∧
          partBase S ns nv + e < (Rows.ofNet (F.Vs j).hU).nIn + (Rows.ofNet (F.Vs j).hU).nComp

/-- Obligation: discharged, off `ValueColl`, by check 5's `checkZero` (`checkResiduals` holds the part's `out` to the
zero rows `rec-residuals-zero` under `zeroSalt`, `rec_residuals.zero_salts`): `Recursion.zero_tie` gives the
registrant's bits at the `out` read `(u, q)` as 0, and the read is the gate's at its position (`rowOfH`, open).
**A part's `out` gates are registered reads of zero bits.** At an accepted outer proof, the gate of rep `r`'s copy of
part `p` at each `out` column is at a registered read, and the registrant's bit there is 0. -/
def ZeroReads : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r p, p < (partsOf F.sh).size →
    ∀ S ns nv, RecResiduals.part F.sh p = .ok (S, ns, nv) → ∀ hj : cp p r < (F.Vs (F.sess p)).c.upv,
      ∀ e < 16 * RecResiduals.RES_OUT_WORDS, ∀ hcol : partBase S ns nv + e < (Rows.ofNet (F.Vs (F.sess p)).hU).w,
        (F.zs R ω t R₂ (F.sess p)).rd.isRead ((F.zs R ω t R₂ (F.sess p)).lay.pos
          ((keyInst (F.Vs (F.sess p)).hU (F.Vs (F.sess p)).nv
              (Layout.outsOf (F.Vs (F.sess p)).c (F.Vs (F.sess p)).hU (F.Vs (F.sess p)).nv) (F.un p r)).wire
            ((Rows.ofNet (F.Vs (F.sess p)).hU).stackCol (F.Vs (F.sess p)).c.upv hj ⟨partBase S ns nv + e, hcol⟩))) ∧
        (F.zs R ω t R₂ (F.sess p)).lay.bit ((F.zs R ω t R₂ (F.sess p)).rd.wv ((F.zs R ω t R₂ (F.sess p)).lay.pos
          ((keyInst (F.Vs (F.sess p)).hU (F.Vs (F.sess p)).nv
              (Layout.outsOf (F.Vs (F.sess p)).c (F.Vs (F.sess p)).hU (F.Vs (F.sess p)).nv) (F.un p r)).wire
            ((Rows.ofNet (F.Vs (F.sess p)).hU).stackCol (F.Vs (F.sess p)).c.upv hj ⟨partBase S ns nv + e, hcol⟩))))
          ((keyInst (F.Vs (F.sess p)).hU (F.Vs (F.sess p)).nv
              (Layout.outsOf (F.Vs (F.sess p)).c (F.Vs (F.sess p)).hU (F.Vs (F.sess p)).nv) (F.un p r)).wire
            ((Rows.ofNet (F.Vs (F.sess p)).hU).stackCol (F.Vs (F.sess p)).c.upv hj ⟨partBase S ns nv + e, hcol⟩)) =
          false

/-- **`HZeroL` from the parts' `out` columns and their reads** (`PartOutCols`, `ZeroReads`, at the part's statement
`HCover` gives): each column's gate is committed (`outCol_committed`), so `RegOut … false` holds there, and
`copyVal_regOut` in the accepted `Good` layer gives 0. -/
theorem hZeroL_of (hcov : F.HCover) (hcols : F.PartOutCols) (hz : F.ZeroReads dg N k Rw) : F.HZeroL dg N k Rw := by
  intro R ω t R₂ τ o ha r p hp S ns nv hpart hj e he
  obtain ⟨j, nm, pt, hjs⟩ := hcov p hp
  have hsj : F.sess p = j := F.hsess p j ⟨nm, pt, hjs⟩
  subst hsj
  obtain ⟨hoc, hlv, hlo, hhi⟩ := hcols p _ nm pt hjs S ns nv hpart _ hj e he
  have hcol : partBase S ns nv + e < (Rows.ofNet (F.Vs (F.sess p)).hU).w := Nat.lt_succ_of_lt hhi
  obtain ⟨hread, hbit⟩ := hz R ω t R₂ τ o ha r p hp S ns nv hpart hj e he hcol
  have hG := (ha.1 (F.sess p)).2
  exact AccChain.copyVal_regOut _ hG.2 (keyProg_correct _ _ hG.1 _) hj (by omega)
    ⟨hcol, outCol_committed _ hj hcol (outColAt_stackCol hj hcol hoc hlv (Nat.mod_lt _ (by decide)) (by omega) hhi),
      hread, hbit⟩

end VStarFamily

end family

end FlockVBridge.Family
