module

public import Proofs.Flock.C3.Compose.HoloCoinTys
public import Proofs.Flock.C3.Compose.LaterHeld
import all Init.Data.Repr
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

/-! The verifier's map's link points and coins are the history's, where a stream round takes one history round
(a sparse phase's round of `n > 1` coins does not). -/

@[expose] public section

namespace FlockC3.Compose

open Flock Flock.RecVStar Flock.Guarantees FlockSoundness.Discharge.Exec FlockSoundness.Discharge.PlaysCoin
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockLevel3
open FlockC3.Reps (CoinsEncoded PointsEncoded repStart)

def holoPts (mPts : ℕ) (A : Array (Array F128)) : Fin 2 → Fin mPts → GF128 :=
  fun a b => GF128.of ((A[a.val]!)[b.val]!)

theorem holoPts_ofFn {mPts : ℕ} (c : Fin 2 → Fin mPts → GF128) :
    holoPts mPts (Array.ofFn fun x : Fin 2 => Array.ofFn fun y : Fin mPts => GF128.val (c x y)) = c := by
  funext a b
  have ha : a.val < (Array.ofFn fun x : Fin 2 => Array.ofFn fun y : Fin mPts => GF128.val (c x y)).size := by
    rw [Array.size_ofFn]; exact a.isLt
  have hb : b.val < (Array.ofFn fun y : Fin mPts => GF128.val (c a y)).size := by
    rw [Array.size_ofFn]; exact b.isLt
  show GF128.of (((Array.ofFn fun x : Fin 2 => Array.ofFn fun y : Fin mPts => GF128.val (c x y))[a.val]!)[b.val]!) =
    c a b
  rw [getElem!_pos (Array.ofFn fun x : Fin 2 => Array.ofFn fun y : Fin mPts => GF128.val (c x y)) a.val ha,
    Array.getElem_ofFn, getElem!_pos (Array.ofFn fun y : Fin mPts => GF128.val (c a y)) b.val hb,
    Array.getElem_ofFn]
  rfl

theorem pointsEncoded_holoMap (st : Flock.Stmt) (hp : HoloParams) (Cm : Type) (ish : RecVStar.Inner) (tbl : String)
    (bc : Cm → String → Nat → List UInt8) :
    PointsEncoded (CT := holoCTs st hp) (holoPts (holoAt st hp).cls.mPts) (holoMap st hp Cm ish tbl bc).points := by
  intro h c
  exact congrArg (fun p : Fin 2 → Fin (holoAt st hp).cls.mPts → GF128 => (⟨_, p⟩ : FlockSoundness.Game.Msg))
    (holoPts_ofFn (c ⟨0, h⟩))

def holoSelOne (ish : RecVStar.Inner) (tbl : String) (i : ℕ) (s : String) (q : ℕ) : Prop :=
  ∃ L e, holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked = some L ∧
    L.find? (fun e => e.1.name tbl == s) = some e ∧ e.1.group (e.2.getD q default).1 = 1 ∧ i = roundAt L e.1 q

theorem selC_of_holoSelOne {st : Flock.Stmt} {hp : HoloParams} {Cm : Type} {ish : RecVStar.Inner} {tbl : String}
    {bc : Cm → String → Nat → List UInt8} {i : ℕ} {s : String} {q : ℕ} (h : holoSelOne ish tbl i s q) :
    (holoMap st hp Cm ish tbl bc).selC i s q := by
  obtain ⟨L, e, hS, hf, hg, rfl⟩ := h
  exact ⟨L, e, hS, hf, by rw [hg]; exact Nat.one_pos, by rw [hg]⟩

theorem roundWords_take {st : Flock.Stmt} {hp : HoloParams}
    (cs : List ((k : Fin (1 + (kindCoinTys (holoKinds (holoAt st hp))).length)) → holoCTs st hp k)) {n j : ℕ}
    (h : j < n) : roundWords (cs.take n) j = roundWords cs j := by
  unfold roundWords
  rw [List.getElem?_take_of_lt h]

theorem coinsEncoded_holoMap {st : Flock.Stmt} {hp : HoloParams} {Cm : Type} {ish : RecVStar.Inner} {tbl : String}
    {bc : Cm → String → Nat → List UInt8} :
    CoinsEncoded (CT := holoCTs st hp) (holoSelOne ish tbl) (holoMap st hp Cm ish tbl bc).coin := by
  intro i hi s q cs hc hsel
  obtain ⟨L', e, hS, hf, hg, rfl⟩ := hsel
  have hcoin : (holoMap st hp Cm ish tbl bc).coin (cs.take (roundAt L' e.1 q + 1)) s q =
      ((List.range (e.1.group (e.2.getD q default).1)).flatMap fun j =>
        (roundWords (cs.take (roundAt L' e.1 q + 1)) (roundAt L' e.1 q + j)).toList).toArray := by
    simp only [holoMap, hS, hf]
  rw [hcoin, hg, List.range_one, List.flatMap_cons, List.flatMap_nil, List.append_nil, Array.toArray_toList,
    Nat.add_zero (roundAt L' e.1 q), roundWords_take cs (show roundAt L' e.1 q < roundAt L' e.1 q + 1 from Nat.lt_succ_self _)]
  exact coinArr_roundWords cs _ hi hc (holoForms_getD_ne_one st hp _)

theorem roundIn_cons_self {s : HoloStream} {rs : Array SchedRound} {L : List (HoloStream × Array SchedRound)} {k : ℕ} :
    roundIn ((s, rs) :: L) s k = heldBefore s rs k :=
  ite_eq_left rfl

theorem roundIn_cons_ne {s' s : HoloStream} {rs : Array SchedRound} {L : List (HoloStream × Array SchedRound)} {k : ℕ}
    (h : ¬s'.idx = s.idx) : roundIn ((s', rs) :: L) s k = heldBefore s' rs rs.size + roundIn L s k :=
  ite_eq_right h

theorem roundAt_reps (rs : Array SchedRound) (L : List (HoloStream × Array SchedRound)) (i : Fin 2) (q : ℕ) :
    roundAt ((.rep 0, rs) :: (.rep 1, rs) :: L) (.rep i) q =
      repStart (heldBefore (.rep 0) rs rs.size) i + heldBefore (.rep i) rs q := by
  rcases i with ⟨_ | _ | n, hn⟩
  · exact congrArg (1 + ·) roundIn_cons_self
  · show 1 + roundIn _ (.rep 1) q = 1 + heldBefore (.rep 0) rs rs.size + heldBefore (.rep 1) rs q
    rw [roundIn_cons_ne (s' := .rep 0) (s := .rep 1) (by decide), roundIn_cons_self]
    omega
  · exact absurd hn (by omega)

theorem holoSelOne_rep {ish : RecVStar.Inner} {tbl s : String} {rs : Array SchedRound}
    {L : List (HoloStream × Array SchedRound)} {i : Fin 2} {q : ℕ}
    (hS : holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked = some ((.rep 0, rs) :: (.rep 1, rs) :: L))
    (hf : ((HoloStream.rep 0, rs) :: (HoloStream.rep 1, rs) :: L).find? (fun e => e.1.name tbl == s) =
      some (.rep i, rs))
    (hq : 0 < (rs.getD q default).1) :
    holoSelOne ish tbl (repStart (heldBefore (.rep 0) rs rs.size) i + heldBefore (.rep i) rs q) s q :=
  ⟨_, _, hS, hf, Nat.min_eq_right hq, (roundAt_reps rs L i q).symm⟩

theorem repName_ne (t : String) : (HoloStream.rep 0).name t ≠ (HoloStream.rep 1).name t := by
  intro h
  have h := congrArg String.toList h
  simp only [HoloStream.name, String.toList_append, toString, Nat.toList_repr, List.append_assoc] at h
  exact absurd (List.append_cancel_left (List.append_cancel_left h)) (by decide)

theorem find?_repName (tbl : String) (rs : Array SchedRound) (L : List (HoloStream × Array SchedRound)) (i : Fin 2) :
    ((HoloStream.rep 0, rs) :: (HoloStream.rep 1, rs) :: L).find? (fun e => e.1.name tbl == (HoloStream.rep i).name tbl) =
      some (.rep i, rs) := by
  rcases i with ⟨_ | _ | n, hn⟩
  · exact List.find?_cons_of_pos (beq_iff_eq.2 rfl)
  · exact (List.find?_cons_of_neg
      (p := fun e : HoloStream × Array SchedRound => e.1.name tbl == (HoloStream.rep ⟨0 + 1, hn⟩).name tbl)
      (a := (HoloStream.rep 0, rs)) (l := (HoloStream.rep 1, rs) :: L)
      fun h => repName_ne tbl (beq_iff_eq.1 h)).trans (List.find?_cons_of_pos (beq_iff_eq.2 rfl))
  · exact absurd hn (by omega)

theorem holoSelOne_repName {ish : RecVStar.Inner} {tbl : String} {rs : Array SchedRound}
    {L : List (HoloStream × Array SchedRound)} {i : Fin 2} {q : ℕ}
    (hS : holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked = some ((.rep 0, rs) :: (.rep 1, rs) :: L))
    (hq : 0 < (rs.getD q default).1) :
    holoSelOne ish tbl (repStart (heldBefore (.rep 0) rs rs.size) i + heldBefore (.rep i) rs q)
      ((HoloStream.rep i).name tbl) q :=
  holoSelOne_rep hS (find?_repName tbl rs L i) hq

end FlockC3.Compose
