module

public import Proofs.Flock.C3.Opening.SessionAcc
public import Proofs.Flock.VBridge.Family.Opened
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt

/-! `Opened` at R's consistency checks, from the `rec-open-stacked` units, whose name `RecOpen.params` does
not parse. -/

@[expose] public section

namespace FlockC3.RegLevels

open Flock
open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open FlockVBridge FlockVBridge.AccChain
open FlockSoundness.Discharge.Hm (bitL)
open FlockVBridge.Decode (RegBit)
open FlockSoundness.Refine (capAt openAt)

theorem climb_node {ms ms' : MerkleScheme} (h : ∀ l r, ms.node l r = ms'.node l r) :
    ∀ (ss : List ByteArray) (acc : ByteArray) (idx : ℕ), climb ms acc idx ss = climb ms' acc idx ss
  | [], _, _ => rfl
  | s :: rest, acc, idx => by
    simp only [climb, h]
    exact climb_node h rest _ _

theorem sibsOf_length {s : Array ByteArray} {q w j : ℕ} (hs : s.size = q * w) (hj : j < q) :
    (sibsOf s w j).length = w := by
  have hle : j * w + w ≤ q * w := by
    have := Nat.mul_le_mul_right w hj
    rwa [Nat.succ_mul] at this
  simp [sibsOf, hs]
  omega

/-- G's `Conss.ligerito_conss` is proved of the symbolic run (`LM Lin`), and a run over F does not give one
over `Lin`; R's opening runs Ligerito over F. -/
def ConssAt (sch : Flock.Schedule) (lg : RecAlgebra.LgPts) : Prop :=
  lg.conss.toList.map (fun c => (c.start, c.pos.size)) =
    (List.range sch.levels.size).map fun k => (k, (sch.levels[k]!).q)

section Copy

variable {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : SamplingDesign nv} (zo : ZkOuter mPts nv J Reg' c hU hscope Lv)

/-- The row and the siblings are in the copy's own input columns, not registered: R's units register neither. -/
def OpensTopAt (bits : Fin (Layout.keyN c hU nv) → Bool) (u : Fin nv) (j LANES : ℕ) (r a cf A : Array F128)
    (H pos : ℕ) (ss : List ByteArray) (top : ByteArray) : Prop :=
  ∃ (hj : j < c.upv), bits (Const.Prog.oneAt (Placed.keyProg c hU nv) u) = true ∧
  ∃ (ms : MerkleScheme), (∀ l r, ms.node l r = Sha512.hash (l ++ r)) ∧
  ∃ (p : Ports) (outBits : Array ℕ) (name : String) (rest : ByteArray),
    H ≤ 32 ∧ Flock.HmNets.netOf name (recOpenLaid LANES H p outBits) = .ok c.unit ∧ c.unit.unitLog ≤ 32 ∧
    (recOpenOuts LANES H p).length ≤ outBits.foldl (· + ·) 0 ∧
    r.size = LANES ∧ a.size = 2 ∧ ss.length = H ∧
    p.row = 8 * (MerkleScheme.rowBytes r).size ∧ 512 * H ≤ p.aux ∧
    p.acc = 8 * (MerkleScheme.rowBytes a ++ rest).size ∧ p.coef = 8 * (MerkleScheme.rowBytes cf).size ∧ H ≤ p.dirs ∧
    (∀ col < p.row + 512 * H, col < (Rows.ofNet hU).nIn ∧ ∀ (hc : col < (Rows.ofNet hU).w),
      (keyInst hU nv (Layout.outsOf c hU nv) u).wire ((Rows.ofNet hU).stackCol c.upv hj ⟨col, hc⟩) ∈
        ((Placed.keyProg c hU nv).partition (Layout.outsOf c hU nv)).io u) ∧
    ColsHold (copyVal (keyInst hU nv (Layout.outsOf c hU nv) u) bits hj) 0 (MerkleScheme.rowBytes r) ∧
    (∀ i (hi : i < ss.length), ss[i].size = 64 ∧
      ColsHold (copyVal (keyInst hU nv (Layout.outsOf c hU nv) u) bits hj) (p.row + 512 * i) ss[i]) ∧
    (∀ t < 8 * (MerkleScheme.rowBytes a ++ rest).size,
      RegBit zo u j (p.row + p.aux + t) (bitL (MerkleScheme.rowBytes a ++ rest).data.toList t)) ∧
    (∀ t < 8 * (MerkleScheme.rowBytes cf).size,
      RegBit zo u j (p.row + p.aux + p.acc + t) (bitL (MerkleScheme.rowBytes cf).data.toList t)) ∧
    (∀ h < H, RegBit zo u j (p.row + p.aux + p.acc + p.coef + h) (pos.testBit h)) ∧
    (∀ t < 256, RegOut zo u hj (Hm.outStart ((recOpenNet LANES H p).run Flock.HmNets.St.empty).2 p.bits outBits +
      1024 + t) (bitL (MerkleScheme.rowBytes A).data.toList t)) ∧
    top.size = 64 ∧ ∀ t < 512, RegOut zo u hj
      (Hm.outStart ((recOpenNet LANES H p).run Flock.HmNets.St.empty).2 p.bits outBits + t) (bitL top.data.toList t)

theorem colsHold_fill_of_io {bits : Fin (Layout.keyN c hU nv) → Bool} (hA : zo.RegAgree bits) {u : Fin nv} {j : ℕ}
    (hj : j < c.upv) {base : ℕ} {B : ByteArray}
    (hio : ∀ t < 8 * B.size, base + t < (Rows.ofNet hU).nIn ∧ ∀ (hc : base + t < (Rows.ofNet hU).w),
      (keyInst hU nv (Layout.outsOf c hU nv) u).wire ((Rows.ofNet hU).stackCol c.upv hj ⟨base + t, hc⟩) ∈
        ((Placed.keyProg c hU nv).partition (Layout.outsOf c hU nv)).io u)
    (h : ColsHold (copyVal (keyInst hU nv (Layout.outsOf c hU nv) u) bits hj) base B) :
    ColsHold (copyVal (keyInst hU nv (Layout.outsOf c hU nv) u) (Decode.fill zo bits) hj) base B := by
  intro t ht
  obtain ⟨hin, hw⟩ := hio t ht
  refine Eq.trans ?_ (h t ht)
  refine (copyVal_input _ _ hj hin).trans (Eq.trans ?_ (copyVal_input _ _ hj hin).symm)
  exact Decode.fill_io zo hA (hw _)

theorem decode_recOpenCols {bits : Fin (Layout.keyN c hU nv) → Bool} (hA : zo.RegAgree bits) (u : Fin nv) {j : ℕ}
    (hj : j < c.upv) (hone : bits (Const.Prog.oneAt (Placed.keyProg c hU nv) u) = true)
    (ms : MerkleScheme) (hms : ∀ l r, ms.node l r = Sha512.hash (l ++ r)) (LANES H : ℕ) (hH : H ≤ 32)
    (p : Ports) (outBits : Array ℕ) {name : String}
    (hl : Flock.HmNets.netOf name (recOpenLaid LANES H p outBits) = .ok c.unit)
    (hu : c.unit.unitLog ≤ 32) (hout : (recOpenOuts LANES H p).length ≤ outBits.foldl (· + ·) 0)
    (r a cf : Array F128) (rest : ByteArray) (idx : ℕ) (ss : List ByteArray)
    (hr : r.size = LANES) (ha : a.size = 2) (hss : ss.length = H)
    (hrowN : p.row = 8 * (MerkleScheme.rowBytes r).size) (hauxN : 512 * H ≤ p.aux)
    (haccN : p.acc = 8 * (MerkleScheme.rowBytes a ++ rest).size)
    (hcoefN : p.coef = 8 * (MerkleScheme.rowBytes cf).size) (hdirsN : H ≤ p.dirs)
    (hio : ∀ col < p.row + 512 * H, col < (Rows.ofNet hU).nIn ∧ ∀ (hc : col < (Rows.ofNet hU).w),
      (keyInst hU nv (Layout.outsOf c hU nv) u).wire ((Rows.ofNet hU).stackCol c.upv hj ⟨col, hc⟩) ∈
        ((Placed.keyProg c hU nv).partition (Layout.outsOf c hU nv)).io u)
    (hrow : ColsHold (copyVal (keyInst hU nv (Layout.outsOf c hU nv) u) bits hj) 0 (MerkleScheme.rowBytes r))
    (haux : ∀ i (hi : i < ss.length), ss[i].size = 64 ∧
      ColsHold (copyVal (keyInst hU nv (Layout.outsOf c hU nv) u) bits hj) (p.row + 512 * i) ss[i])
    (hacc : ∀ t < 8 * (MerkleScheme.rowBytes a ++ rest).size,
      RegBit zo u j (p.row + p.aux + t) (bitL (MerkleScheme.rowBytes a ++ rest).data.toList t))
    (hcoef : ∀ t < 8 * (MerkleScheme.rowBytes cf).size,
      RegBit zo u j (p.row + p.aux + p.acc + t) (bitL (MerkleScheme.rowBytes cf).data.toList t))
    (hdir : ∀ h < H, RegBit zo u j (p.row + p.aux + p.acc + p.coef + h) (idx.testBit h)) :
    ColsHold (copyVal (keyInst hU nv (Layout.outsOf c hU nv) u) bits hj)
      (Hm.outStart ((recOpenNet LANES H p).run Flock.HmNets.St.empty).2 p.bits outBits)
      ((climb ms (Sha512.hash (MerkleScheme.rowBytes r)) idx ss ++ ⟨Array.replicate 64 0⟩) ++
        (MerkleScheme.rowBytes #[FlockVBridge.consTerm LANES r a cf 0, FlockVBridge.consTerm LANES r a cf 1] ++
          rest)) := by
  have hone' : Decode.fill zo bits (Const.Prog.oneAt (Placed.keyProg c hU nv) u) = true := by
    rw [← keyInst_one hU nv (Layout.outsOf c hU nv) u, Decode.fill_one zo hA _ hj, keyInst_one]
    exact hone
  have hO := keyProg_recOpen hU nv (Layout.outsOf c hU nv) (Decode.fill zo bits) u hj hone' ms hms LANES H hH p
    outBits hl hu hout r a cf rest idx ss hr ha hss hrowN hauxN haccN hcoefN hdirsN
    (colsHold_fill_of_io zo hA hj (fun t ht => hio _ (by omega)) hrow)
    (fun i hi => ⟨(haux i hi).1, colsHold_fill_of_io zo hA hj
      (fun t ht => hio _ (by have := (haux i hi).1; omega)) (haux i hi).2⟩)
    (Decode.colsHold_fill zo bits hj _ _ hacc) (Decode.colsHold_fill zo bits hj _ _ hcoef)
    (fun h hh => Decode.copyVal_regBit zo bits hj (hdir h hh))
  refine Decode.colsHold_of_fill zo hA hj ?_ hO
  rw [Layout.ofNet_nIn_eq, inWords_recOpen hl]
  exact Decode.inRows_le_outStart _ _ _

theorem climb_of_opensTop {bits : Fin (Layout.keyN c hU nv) → Bool} (hG : zo.Good bits) {u : Fin nv}
    {j LANES H pos : ℕ} {r a cf A : Array F128} {ss : List ByteArray} {top : ByteArray}
    (h : OpensTopAt zo bits u j LANES r a cf A H pos ss top) (msId : String) :
    climb (MerkleScheme.sha512 msId) (Sha512.hash (MerkleScheme.rowBytes r)) pos ss = top := by
  obtain ⟨hj, hone, ms, hms, p, outBits, name, rest, hH, hl, hu, hout, hr, ha, hss, hrowN, hauxN, haccN, hcoefN,
    hdirsN, hio, hrow, haux, hacc, hcoef, hdir, -, htop, hT⟩ := h
  have hO := decode_recOpenCols zo hG.2 u hj hone ms hms LANES H hH p outBits hl hu hout r a cf rest pos ss hr ha
    hss hrowN hauxN haccN hcoefN hdirsN hio hrow haux hacc hcoef hdir
  have hnIn : (Rows.ofNet hU).nIn ≤
      Hm.outStart ((recOpenNet LANES H p).run Flock.HmNets.St.empty).2 p.bits outBits := by
    rw [Layout.ofNet_nIn_eq, inWords_recOpen hl]
    exact Decode.inRows_le_outStart _ _ _
  have hX := Family.bytesAt_of_cols (Family.colsHold_left (Family.colsHold_left hO))
  rw [climb_size_of ms hms ss pos _ (FlockSoundness.Refine.sha512_size _)] at hX
  have hT' : ColsHold (copyVal (keyInst hU nv (Layout.outsOf c hU nv) u) bits hj)
      (Hm.outStart ((recOpenNet LANES H p).run Flock.HmNets.St.empty).2 p.bits outBits) top := fun t ht =>
    copyVal_regOut zo hG.2 (keyProg_correct hU nv hG.1 u) hj (by omega) (hT t (by omega))
  have hY := Family.bytesAt_of_cols hT'
  rw [htop] at hY
  exact (climb_node (ms := MerkleScheme.sha512 msId) (ms' := ms) (fun l r => (hms l r).symm) ss _ pos).trans
    (hX.symm.trans hY)

end Copy

theorem opened_of_climbs {msId : String} {sch : Flock.Schedule} {cap : Array ByteArray} {p : LigeritoProof}
    {M : RecAlgebra.Messages F128} {lg : RecAlgebra.LgPts} (hcs : ConssAt sch lg) (hcaps : Family.CapsAt sch cap p)
    (rows : ℕ → Array (Array F128)) (sibs : ℕ → Array ByteArray)
    (hO : ∀ ℓ < sch.levels.size, openAt p (sch.levels.size - 1) ℓ = (rows ℓ, sibs ℓ, #[]))
    (hrs : ∀ ℓ < sch.levels.size, (rows ℓ).size = (sch.levels[ℓ]!).q)
    (hss : ∀ ℓ < sch.levels.size,
      (sibs ℓ).size = (sch.levels[ℓ]!).q * ((sch.levels[ℓ]!).d - (sch.levels[ℓ]!).cap))
    (h64 : ∀ ℓ < sch.levels.size, ∀ x ∈ sibs ℓ, x.size = 64)
    (hrow : ∀ ℓ < sch.levels.size, ∀ j < (sch.levels[ℓ]!).q, ((rows ℓ)[j]!).size = 2 ^ (sch.levels[ℓ]!).lanes)
    (hclimb : ∀ ℓ < sch.levels.size, ∀ j < (sch.levels[ℓ]!).q,
      climb (MerkleScheme.sha512 msId) (Sha512.hash (MerkleScheme.rowBytes (rows ℓ)[j]!)) ((lg.conss[ℓ]!).pos[j]!)
          (sibsOf (sibs ℓ) ((sch.levels[ℓ]!).d - (sch.levels[ℓ]!).cap) j) =
        (capAt cap p ℓ).getD ((lg.conss[ℓ]!).pos[j]! >>> ((sch.levels[ℓ]!).d - (sch.levels[ℓ]!).cap))
          ByteArray.empty)
    (hcons : ∀ ℓ < sch.levels.size, M.cons ℓ ((eqTable (lg.conss[ℓ]!).alpha).extract 0 (sch.levels[ℓ]!).q) =
      (Array.range (2 ^ (sch.levels[ℓ]!).lanes)).map
        (RecAlgebra.consAt (rows ℓ) (sch.levels[ℓ]!).q (lg.conss[ℓ]!).alpha)) :
    ∀ cs ∈ lg.conss.toList, RecAlgebra.Opened (MerkleScheme.sha512 msId) sch cap p M cs := by
  intro cs hmem
  obtain ⟨hℓ, hcs0, hpsz⟩ := Family.Conss.conss_mem hcs hmem
  have hpsz' : cs.pos.size = (sch.levels[cs.start]!).q := hpsz
  obtain ⟨hc1, hc2⟩ := hcaps cs.start hℓ
  have hcl := hclimb cs.start hℓ
  have hco := hcons cs.start hℓ
  rw [hcs0] at hcl hco
  unfold RecAlgebra.Opened
  rw [Family.Conss.getD_default, hO cs.start hℓ]
  refine ⟨Family.Conss.merkleCheck_of (rows := rows cs.start) (sibs := sibs cs.start) (salts := #[]) hc1
    (by rw [hrs cs.start hℓ, hpsz']) (by rw [hss cs.start hℓ, hpsz']) rfl hc2 (h64 cs.start hℓ) fun j hj => ?_, hco⟩
  have hj' : j < (sch.levels[cs.start]!).q := by
    rw [← hpsz']
    exact hj
  unfold opens
  simp only [Bool.and_eq_true, decide_eq_true_eq]
  exact ⟨⟨⟨hrow cs.start hℓ j hj', sibsOf_length (hss cs.start hℓ) hj'⟩, by simp; rfl⟩, hcl j hj'⟩

theorem decodedO_cons (sh : Flock.Shape) (w : Array F128) (cons : ℕ → ℕ → F128) (ℓ : ℕ) (ea : Array F128) :
    (Opening.decodedO sh w cons).cons ℓ ea = (Array.range (2 ^ (sh.sched.levels[ℓ]!).lanes)).map (cons ℓ) := by
  show ((Array.range (2 ^ (sh.sched.levels[ℓ]!).lanes)).map (RecAlgebra.Lin.var ∘ RecAlgebra.Key.cons ℓ)).map
    (RecAlgebra.Lin.eval (Opening.valO sh w cons)) = _
  rw [Array.map_map]
  congr 1
  funext i
  exact RecAlgebra.Lin.eval_var _ _

section Units

variable {nS : ℕ} {Vs : Fin nS → VStmt} {Reg' : Type}

theorem opened_of_units (zs : (s : Fin nS) → (Vs s).Outer Reg') (Lb : (s : Fin nS) → (Vs s).Bits)
    (hG : ∀ s, (zs s).Good (Lb s)) {sh : Flock.Shape} {w : Array F128} {ops : FlockVBridge.AccChain.Openings}
    {msId : String} {cap : Array ByteArray} {p : LigeritoProof} {lg : RecAlgebra.LgPts} (hcs : ConssAt sh.sched lg)
    (hcaps : Family.CapsAt sh.sched cap p) (sibs : ℕ → Array ByteArray)
    (hO : ∀ ℓ < sh.sched.levels.size, openAt p (sh.sched.levels.size - 1) ℓ = (ops.rows ℓ, sibs ℓ, #[]))
    (hrs : ∀ ℓ < sh.sched.levels.size, (ops.rows ℓ).size = (sh.sched.levels[ℓ]!).q)
    (hss : ∀ ℓ < sh.sched.levels.size,
      (sibs ℓ).size = (sh.sched.levels[ℓ]!).q * ((sh.sched.levels[ℓ]!).d - (sh.sched.levels[ℓ]!).cap))
    (h64 : ∀ ℓ < sh.sched.levels.size, ∀ x ∈ sibs ℓ, x.size = 64)
    (hq : ∀ ℓ < sh.sched.levels.size, ops.q ℓ = (sh.sched.levels[ℓ]!).q)
    (halpha : ∀ ℓ < sh.sched.levels.size, ops.alpha ℓ = (lg.conss[ℓ]!).alpha)
    (sess : ℕ → Fin nS) (un : (ℓ : ℕ) → ℕ → Fin (Vs (sess ℓ)).nv) (cp : ℕ → ℕ → ℕ)
    (hu : ∀ ℓ < sh.sched.levels.size, ∀ j < (sh.sched.levels[ℓ]!).q, ∃ a cf A,
      OpensTopAt (zs (sess ℓ)) (Lb (sess ℓ)) (un ℓ j) (cp ℓ j) (levelLanes sh ℓ) (ops.rows ℓ)[j]! a cf A
        ((sh.sched.levels[ℓ]!).d - (sh.sched.levels[ℓ]!).cap) ((lg.conss[ℓ]!).pos[j]!)
        (sibsOf (sibs ℓ) ((sh.sched.levels[ℓ]!).d - (sh.sched.levels[ℓ]!).cap) j)
        ((capAt cap p ℓ).getD ((lg.conss[ℓ]!).pos[j]! >>> ((sh.sched.levels[ℓ]!).d - (sh.sched.levels[ℓ]!).cap))
          ByteArray.empty)) :
    ∀ cs ∈ lg.conss.toList,
      RecAlgebra.Opened (MerkleScheme.sha512 msId) sh.sched cap p (Opening.decodedO sh w (Opening.consO ops)) cs := by
  refine opened_of_climbs hcs hcaps ops.rows sibs hO hrs hss h64 (fun ℓ hℓ j hj => ?_) (fun ℓ hℓ j hj => ?_)
    fun ℓ hℓ => ?_
  · obtain ⟨a, cf, A, h⟩ := hu ℓ hℓ j hj
    obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, hr, _⟩ := h
    exact hr
  · obtain ⟨a, cf, A, h⟩ := hu ℓ hℓ j hj
    exact climb_of_opensTop (zs (sess ℓ)) (hG (sess ℓ)) h msId
  · rw [decodedO_cons, ← hq ℓ hℓ, ← halpha ℓ hℓ]
    rfl

end Units

end FlockC3.RegLevels
