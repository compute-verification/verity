module

public import Proofs.Flock.C3.Opening.SessionParts
public import Proofs.Flock.VBridge.ResLayer
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
import Proofs.Flock.VBridge.AlgHolds
import Proofs.Flock.VBridge.OpeningUnit

/-! `OpeningZeroAtL`: the opening's parts' zero columns, at the columns the rows read. -/

@[expose] public section

namespace FlockVBridge.OfVStar

open Flock.HmNets hiding and
open Flock FlockSoundness.Discharge.Hm
open Flock.RecResiduals (groups partS)
open FlockVBridge.PartS (wIdx)
open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open FlockVBridge.Decode
open FlockVBridge.AccChain (RegOut copyVal_regOut)

variable {mPts nU J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : SamplingDesign nU} (zo : ZkOuter mPts nU J Reg' c hU hscope Lv)

theorem decode_openingR {bits : Fin (Layout.keyN c hU nU) → Bool} (hA : zo.RegAgree bits) (u : Fin nU) {j : ℕ}
    (hj : j < c.upv) (hone : bits (Const.Prog.oneAt (Placed.keyProg c hU nU) u) = true)
    (hc : RecOpening.check c = .ok ()) {m cl p : ℕ} (hp : RecOpening.params c.unit.name = some (m, cl, p))
    {sh : Flock.Shape} (hsh : RecOpening.shape? m cl = some sh) {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (hpart : RecOpening.part sh p = .ok (S, ns, nv)) (hu : c.unit.unitLog ≤ 32)
    (w vv : ℕ → Flock.F128) (wI V : Array ℕ) (hwI : wI.size ≤ ns.sum) (hV : V.size ≤ nv)
    (hw : ∀ q < 128 * ns.sum, ReadCol (Rows.ofNet hU) (rowCol (partRep ns nv).ports 0 q) →
      RegBit zo u j (rowCol (partRep ns nv).ports 0 q) (wbit (sub wI w) q))
    (hv : ∀ q < 128 * nv, RegBit zo u j (vBase (partRep ns nv) + q) (wbit (sub V vv) q)) :
    ∃ w' v' : ℕ → Flock.F128, (∀ k (hk : k < wI.size), w' k = w wI[k]) ∧
      (∀ k (hk : k < V.size), v' k = vv V[k]) ∧
      PartCols S w' v' (copyVal (keyInst hU nU (Layout.outsOf c hU nU) u) bits hj) (partBase S ns nv) := by
  obtain ⟨S', ns', nv', hpart', hn⟩ := check_opening hc hp hsh
  rw [hpart] at hpart'
  cases hpart'
  exact ⟨sub wI w, sub V vv, sub_lt wI w, sub_lt V vv,
    decode_resUnitR zo hA u hj hone S ns nv (opening_small hpart) hn hu _ _ (sub_ge w hwI) (sub_ge vv hV) hw hv⟩

/-- Row ports are asked for `w` only at the columns some computed row reads: first-fit packing leaves elements
of a touched port that no computed row reads, which nothing registers. -/
def OpeningZeroAtL (bits : Fin (Layout.keyN c hU nU) → Bool) (u : Fin nU) (j : ℕ) (sh : Shape)
    (gs : Array (Array ℕ × Array ℕ × Array ℕ)) (p : ℕ) (w v : ℕ → F128) : Prop :=
  ∃ (hpg : p < gs.size) (hj : j < c.upv), bits (Const.Prog.oneAt (Placed.keyProg c hU nU) u) = true ∧
    RecOpening.check c = .ok () ∧ c.unit.unitLog ≤ 32 ∧
    ∃ m cl, RecOpening.params c.unit.name = some (m, cl, p) ∧ RecOpening.shape? m cl = some sh ∧
    ∃ S ns nv, RecOpening.part sh p = .ok (S, ns, nv) ∧
      (∀ q < 128 * ns.sum, ReadCol (Rows.ofNet hU) (rowCol (partRep ns nv).ports 0 q) →
        RegBit zo u j (rowCol (partRep ns nv).ports 0 q) (wbit (sub (wIdx (RecOpening.portSizes sh) gs[p].2.1) w) q)) ∧
      (∀ q < 128 * nv, RegBit zo u j (vBase (partRep ns nv) + q) (wbit (sub gs[p].2.2 v) q)) ∧
      ∀ t < 16 * RecResiduals.RES_OUT_WORDS, RegOut zo u hj (partBase S ns nv + t) false

theorem openingZeroAtL_of {bits : Fin (Layout.keyN c hU nU) → Bool} {u : Fin nU} {j : ℕ} {sh : Shape}
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} {p : ℕ} {w v : ℕ → F128} (h : OpeningZeroAt zo bits u j sh gs p w v) :
    OpeningZeroAtL zo bits u j sh gs p w v :=
  let ⟨hpg, hj, hone, hc, hu, m, cl, hp, hsh, S, ns, nv, hpart, hw, hv, hz⟩ := h
  ⟨hpg, hj, hone, hc, hu, m, cl, hp, hsh, S, ns, nv, hpart, fun q hq _ => hw q hq, hv, hz⟩

theorem partCols_of_openingZeroAtL {bits : Fin (Layout.keyN c hU nU) → Bool} (hG : zo.Good bits) {u : Fin nU}
    {j : ℕ} {sh : Shape} {S0 : RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hS : RecOpening.structureOf sh = .ok S0) (hg : groups S0 (RecOpening.portSizes sh) = .ok gs) {p : ℕ}
    {w v : ℕ → F128} (h : OpeningZeroAtL zo bits u j sh gs p w v) :
    ∃ hpg : p < gs.size, ∃ S', partS S0 (RecOpening.portSizes sh) gs[p].1 gs[p].2.1 gs[p].2.2 = .ok S' ∧
      ∃ (w' v' : ℕ → F128) (col : ℕ → Bool) (base : ℕ),
        (∀ k (hk : k < (wIdx (RecOpening.portSizes sh) gs[p].2.1).size),
          w' k = w (wIdx (RecOpening.portSizes sh) gs[p].2.1)[k]) ∧
        (∀ k (hk : k < gs[p].2.2.size), v' k = v gs[p].2.2[k]) ∧
        PartCols S' w' v' col base ∧ ∀ t < 128 * S'.res.length, col (base + t) = false := by
  obtain ⟨hpg, hj, hone, hc, hu, m, cl, hp, hsh, S, ns, nv, hpart, hw, hv, hz⟩ := h
  obtain ⟨hS', hns, hnv⟩ := part_eqOpening hS hg hpg hpart
  obtain ⟨w', v', hw', hv', hcols⟩ := decode_openingR zo hG.2 u hj hone hc hp hsh hpart hu w v _ _
    (Nat.le_of_eq (by rw [wIdx_size, hns])) (Nat.le_of_eq hnv.symm) hw hv
  obtain ⟨S1, ns1, nv1, hpart1, hn⟩ := check_opening hc hp hsh
  rw [hpart] at hpart1
  cases hpart1
  have hbase : (Rows.ofNet hU).nIn ≤ partBase S ns nv := by
    rw [Layout.ofNet_nIn_eq, inWords_unitLaid hn]
    exact inRows_le_outStart _ _ _
  have hsmall := opening_small hpart
  exact ⟨hpg, S, hS', w', v', _, _, hw', hv', hcols, fun t ht =>
    copyVal_regOut zo hG.2 (keyProg_correct hU nU hG.1 u) hj (by omega) (hz t (by omega))⟩

end FlockVBridge.OfVStar
