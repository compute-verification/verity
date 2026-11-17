module

public import Flock.RecVStar

/-! Mirrors of `rec_live`'s schedule of candidate 3's inner session, and where its transcript holds each fixed value, which `holoMap` reads. -/

@[expose] public section

namespace Flock.Guarantees

open Flock Flock.RecVStar

/-- `rec_live.zp_round`: z_partial's round in `repRounds`. -/
def zpRound (m kLog : Nat) : Nat := m + kLog - 7

/-- `rec_live.holo_rounds`, recording only whether a round absorbs, not how many elements: h's round absorbs 64 in
`rec_live`, and no guarantee or assumption record reads `holoRounds`. -/
def holoRounds (m kLog claims : Nat) : Option (Array SchedRound) := do
  let out ← repRounds m kLog claims
  let k := zpRound m kLog
  if out[k]? != some ((1, #[], 0, true) : SchedRound) then none
  return out.extract 0 k ++ #[((0, #[], 0, true) : SchedRound), (1, #[], 0, true)] ++ out.extract (k + 1) out.size

/-- `rec_live.sparse_rounds`: a rep's sparse phase at `sh`, in `rec_sparse.prove`'s samples. -/
def sparseRounds (sh : RecSparse.Shape) : Array SchedRound :=
  let one := fun (n : Nat) (absorbs : Bool) => ((n, #[], 0, absorbs) : SchedRound)
  #[one RecSparse.PRE_ROUNDS false] ++ Array.replicate RecSparse.PRE_ROUNDS (one 1 true) ++ #[one 0 true] ++
    (sh.degrees.map fun _ => one 1 true).toArray ++ #[one sh.pLog true] ++
    (if sh.stacked then #[one sh.tLog true] else #[]) ++ #[one 0 true]

inductive HoloStream
  | rep (x : Fin 2)
  | sparse (x : Fin 2)
  | stacked

def HoloStream.name (t : String) : HoloStream → String
  | .rep x => s!"{t}rep{x.val}"
  | .sparse x => s!"{SPARSE}/rep{x.val}"
  | .stacked => s!"{STACKED}/rep0"

/-- `rec_live.schedule_for(statement, holo=True, stacked, sparse)`'s streams in its order, each with its rounds. -/
def holoStreams (m kLog claims : Nat) (sparse : Option RecSparse.Shape) (stacked : Option StackedShape) :
    Option (List (HoloStream × Array SchedRound)) := do
  let rep ← holoRounds m kLog claims
  let sp : List (HoloStream × Array SchedRound) := match sparse with
    | some sh => [(HoloStream.sparse 0, sparseRounds sh), (HoloStream.sparse 1, sparseRounds sh)]
    | none => []
  let r : List (HoloStream × Array SchedRound) ← match stacked with
    | some sh => (stackedRounds sh).map fun rs => [(HoloStream.stacked, rs)]
    | none => some []
  return [(HoloStream.rep 0, rep), (HoloStream.rep 1, rep)] ++ sp ++ r

def msgRounds (rs : Array SchedRound) : List Nat :=
  (List.range rs.size).filter fun k => (rs.getD k default).2.2.2

def coinRounds (rs : Array SchedRound) : List Nat :=
  (List.range rs.size).filter fun k => 0 < (rs.getD k default).1

def topRounds (rs : Array SchedRound) : List (Nat × Nat) :=
  (List.range rs.size).flatMap fun k => (rs.getD k default).2.1.toList.map fun c => (k, c)

def valueRound (rep : Array SchedRound) (sparse : Option RecSparse.Shape) (stacked : Option (Array SchedRound))
    (v : String) (x : Nat) : Option (HoloStream × Nat) :=
  match cIndex v, openingIndex v, stackedTopLevel v, sparsePort v with
  | some i, _, _, _ => if h : x < 2 then ((msgRounds rep)[i]?).map (HoloStream.rep ⟨x, h⟩, ·) else none
  | _, some i, _, _ => if x = 0 then (stacked.bind fun rs => (msgRounds rs)[i]?).map (HoloStream.stacked, ·) else none
  | _, _, some l, _ => (stacked.bind fun rs => (topRounds rs)[l]?).bind fun (k, c) =>
      if x < c then some (HoloStream.stacked, k) else none
  | _, _, _, some p =>
      if h : x < 2 then
        (sparse.bind fun sh => ((sparseMessages sh).idxOf? p).bind fun i => (msgRounds (sparseRounds sh))[i]?).map
          (HoloStream.sparse ⟨x, h⟩, ·)
      else none
  | none, none, none, none => none

end Flock.Guarantees
