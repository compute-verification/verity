module

public import Proofs.Flock.Recursive.HoloSched
public import Proofs.Flock.Recursive.HoloCoins
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

/-! The verifier's map of candidate 3's inner session (`holoMap`): where the challenger's live-coin part's records hold each round of the history. -/

@[expose] public section

namespace Flock.Guarantees

open Flock Flock.RecVStar FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Recursion FlockLevel3

def HoloStream.idx : HoloStream → Nat
  | .rep x => x.val
  | .sparse x => 2 + x.val
  | .stacked => 4

def HoloStream.group : HoloStream → Nat → Nat
  | .sparse _, n => n
  | _, n => min n 1

def heldBefore (s : HoloStream) (rs : Array SchedRound) (k : Nat) : Nat :=
  ((List.range (min k rs.size)).map fun q => s.group (rs.getD q default).1).sum

def roundIn : List (HoloStream × Array SchedRound) → HoloStream → Nat → Nat
  | [], _, _ => 0
  | (s', rs) :: L, s, k => if s'.idx = s.idx then heldBefore s rs k else heldBefore s' rs rs.size + roundIn L s k

def roundAt (L : List (HoloStream × Array SchedRound)) (s : HoloStream) (k : Nat) : Nat := 1 + roundIn L s k

open Classical in
noncomputable def coinWords (C : Type) (c : C) : Array F128 :=
  if h : ∃ a : Array F128, FlockSoundness.Discharge.PlaysCoin.coinArr FlockSoundness.Refine.fOf a = ⟨C, c⟩ then
    h.choose
  else if h : ∃ a : Array F128, (⟨Fin a.size → GF128, fun i => FlockSoundness.Refine.fOf a[i.val]!⟩ :
      FlockSoundness.Game.Msg) = ⟨C, c⟩ then h.choose
  else #[]

noncomputable def roundWords {st : Flock.Stmt} {hp : HoloParams}
    (cs : List ((k : Fin (1 + (kindCoinTys (holoKinds (holoAt st hp))).length)) → holoCTs st hp k)) (j : Nat) :
    Array F128 :=
  match cs[j]? with
  | some c => if h : j < 1 + (kindCoinTys (holoKinds (holoAt st hp))).length then coinWords _ (c ⟨j, h⟩) else #[]
  | none => #[]

noncomputable def holoMap (st : Flock.Stmt) (hp : HoloParams) (Cm : Type) (ish : RecVStar.Inner) (tbl : String)
    (bc : Cm → String → Nat → List UInt8) :
    FlockSoundness.Assumptions.InnerRecordMap Cm
      ((k : Fin (1 + (kindCoinTys (holoKinds (holoAt st hp))).length)) → holoCTs st hp k) where
  sel i v x := ∃ L rep s k, holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked = some L ∧
    holoRounds ish.m ish.kLog ish.claims = some rep ∧
    valueRound rep ish.sparse (ish.stacked.bind stackedRounds) v x = some (s, k) ∧ roundAt L s k = i
  bc := bc
  selC i s q := ∃ L e, holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked = some L ∧
    L.find? (fun e => e.1.name tbl == s) = some e ∧ 0 < e.1.group (e.2.getD q default).1 ∧
    i + 1 = roundAt L e.1 q + e.1.group (e.2.getD q default).1
  coin cs s q := match holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked with
    | some L => match L.find? fun e => e.1.name tbl == s with
      | some e => ((List.range (e.1.group (e.2.getD q default).1)).flatMap fun j =>
          (roundWords cs (roundAt L e.1 q + j)).toList).toArray
      | none => #[]
    | none => #[]
  points c := Array.ofFn fun a : Fin 2 => Array.ofFn fun b : Fin (holoAt st hp).cls.mPts =>
    GF128.val ((c ⟨0, by omega⟩ : Fin 2 → Fin (holoAt st hp).cls.mPts → GF128) a b)

end Flock.Guarantees
