import Proofs.Flock.Recursive.HoloSched
import Proofs.Flock.Recursive.HoloCoins
import Proofs.Flock.Soundness.Assumptions.Inner
import Proofs.Flock.Soundness.Discharge.PlaysCoin.Statements

/-!
# Candidate 3's inner record map

`Flock.Guarantees.holoMap`: the verifier's map of candidate 3's inner session (`InnerRecordMap` at the coins of their
coin kinds, `holoCTs`), which says where the coin server's records hold each round of the history, read off the
verifier's own inner shape (`RecVStar.Inner`) through the session's schedule (`HoloSched`):

* the history lays the streams out in `holoStreams`' order, as `holoKinds` lays their coins: round 0 draws the link
  points, then rep 0's stream rounds, rep 1's, both sparse phases' and R's take theirs in turn (`roundAt`). A stream
  round draws its coins as one history coin, but a sparse phase's draws one coin per history round
  (`HoloStream.group`: ζ_c's six, ζ_p's `pLog` and ρ_t's `tLog`, each after the stream round's one message, as
  `sparseKinds` has them);
* round `i` fixes row `x` of a value where the transcript holds that row at a stream round (`valueRound`) whose message
  round `i` commits: the history round of the stream round's first coin, or, at a round of no coins, of the next coin
  the history draws (`sel`);
* round `i` draws a stream round's coins where it is the last history round the stream round takes (`selC`), and they
  are the words of those history rounds' coins, in order (`coin`, at `coinWords`);
* the session's link points are the first coin's words (`points`).

The table's prefix `tbl` (`<table>/`, which names the reps' streams in the inner transcript) and the `b ‖ c` of a
round's commitment at a value's row (`bc`, at the generic commitment type) are the map's parameters.
-/

namespace Flock.Guarantees

open Flock Flock.RecVStar FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Recursion FlockLevel3

/-- The stream's place in `holoStreams`' order. -/
def HoloStream.idx : HoloStream → Nat
  | .rep x => x.val
  | .sparse x => 2 + x.val
  | .stacked => 4

/-- **The history rounds a stream round of `n` coins takes**: a sparse phase's draws one coin per round, every other
stream's draws its round's coins as one coin, and a round of no coins takes none. -/
def HoloStream.group : HoloStream → Nat → Nat
  | .sparse _, n => n
  | _, n => min n 1

/-- The history rounds stream `s`'s rounds `rs` before round `k` take. -/
def heldBefore (s : HoloStream) (rs : Array SchedRound) (k : Nat) : Nat :=
  ((List.range (min k rs.size)).map fun q => s.group (rs.getD q default).1).sum

/-- The history rounds the streams `L` take before stream `s`'s round `k`, in `L`'s order. -/
def roundIn : List (HoloStream × Array SchedRound) → HoloStream → Nat → Nat
  | [], _, _ => 0
  | (s', rs) :: L, s, k => if s'.idx = s.idx then heldBefore s rs k else heldBefore s' rs rs.size + roundIn L s k

/-- **The history round of stream `s`'s round `k`** in the streams `L`: one past the history rounds before it
(`roundIn`), round 0 drawing the link points. It draws the stream round's first coin or, at a round of no coins, the
next coin the history draws; past the last coin, it is the last round, which draws none. -/
def roundAt (L : List (HoloStream × Array SchedRound)) (s : HoloStream) (k : Nat) : Nat := 1 + roundIn L s k

open Classical in
/-- **A coin's words**: the array a run reads as the coin step `c` (`Discharge.PlaysCoin.coinArr` at `fOf`, the one
word of a scalar or the words of a vector), else the array whose words are the coordinates of the vector `c` (a vector
of one coordinate, which `coinArr` reads as a scalar), empty where neither reads as `c`. -/
noncomputable def coinWords (C : Type) (c : C) : Array F128 :=
  if h : ∃ a : Array F128, FlockSoundness.Discharge.PlaysCoin.coinArr FlockSoundness.Refine.fOf a = ⟨C, c⟩ then
    h.choose
  else if h : ∃ a : Array F128, (⟨Fin a.size → GF128, fun i => FlockSoundness.Refine.fOf a[i.val]!⟩ :
      FlockSoundness.Game.Msg) = ⟨C, c⟩ then h.choose
  else #[]

/-- **The words of history round `j`'s coin** in the history's coins `cs`: coordinate `j` of its coin, the one round `j`
draws (`schedSteps`); empty past `cs`. -/
noncomputable def roundWords {st : Flock.Stmt} {hp : HoloParams}
    (cs : List ((k : Fin (1 + (kindCoinTys (holoKinds (holoAt st hp))).length)) → holoCTs st hp k)) (j : Nat) :
    Array F128 :=
  match cs[j]? with
  | some c => if h : j < 1 + (kindCoinTys (holoKinds (holoAt st hp))).length then coinWords _ (c ⟨j, h⟩) else #[]
  | none => #[]

/-- **The verifier's map of candidate 3's inner session**, at its inner shape `ish`, the table's prefix `tbl` and the
`b ‖ c` of a round's commitment at a value's row `bc`, with `L` the session's streams at `ish` (`holoStreams`):

* round `i` fixes row `x` of `v` where the transcript holds it at a stream round (`valueRound`) of history round `i`
  (`roundAt`);
* round `i` draws round `q` of the stream named `s` where that stream round draws coins and `i` is the last history
  round it takes;
* a stream round's coins are the words of the history rounds it takes, in order (`roundWords`);
* the link points are the first coin's words. -/
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
