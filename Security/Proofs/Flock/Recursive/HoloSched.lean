import Flock.RecVStar

/-!
# Candidate 3's inner schedule, and where its transcript holds each fixed value

Mirrors of `rec_live`'s schedule of candidate 3's inner session (`schedule_for` with `holo`, `sparse` and `stacked`),
which the guarantee's fixed record map (`InnerRecordMap`'s `sel` and `selC`) is to read:
* a rep's rounds with hidden unit circuits (`holoRounds`, `rec_live.holo_rounds`), a rep's sparse phase's
  (`sparseRounds`, `rec_live.sparse_rounds`) and R's (`RecVStar.stackedRounds`), and the session's streams in
  `schedule_for`'s order (`holoStreams`);
* a stream's message rounds (`msgRounds`), the rounds that draw coins (`coinRounds`), and the tops it commits
  (`topRounds`);
* the stream and round at which the inner transcript holds row `x` of each value `RecVStar.fixedRow` names
  (`valueRound`): `rec-c<i>` at rep `x`'s `i`-th message round, `rec-opening-c<i>` at R's stream's `i`-th (row 0),
  `rec-top-stacked-L<l>` at the round committing R's `l`-th top, and `rec-sparse-<p>` and `rec-claim` at `sparse/rep<x>`'s
  message round of `p`'s place in `RecVStar.sparseMessages` (`claim` last).

A round's `SchedRound` is `rec_live`'s tuple with its absorbed elements read as whether it absorbs a message, as
`RecVStar.repRounds` has it.
-/

namespace Flock.Guarantees

open Flock Flock.RecVStar

/-- `rec_live.zp_round`: z_partial's round in `repRounds`. -/
def zpRound (m kLog : Nat) : Nat := m + kLog - 7

/-- `rec_live.holo_rounds`: `repRounds` with hidden unit circuits. z_partial's round squeezes nothing, and a round of its
own absorbs h and squeezes r′. None where round `zpRound` is not z_partial's.
A round here records only whether it absorbs (`SchedRound`'s flag), not how many elements: h's round absorbs 64 in
`rec_live`, and no guarantee or assumption record reads `holoRounds`. -/
def holoRounds (m kLog claims : Nat) : Option (Array SchedRound) := do
  let out ← repRounds m kLog claims
  let k := zpRound m kLog
  if out[k]? != some ((1, #[], 0, true) : SchedRound) then none
  return out.extract 0 k ++ #[((0, #[], 0, true) : SchedRound), (1, #[], 0, true)] ++ out.extract (k + 1) out.size

/-- `rec_live.sparse_rounds`: a rep's sparse phase at `sh`, in `rec_sparse.prove`'s draws: ζ_c, H2's rounds, v2's (no
coins), H3's, u's (ζ_p), at T > 1 vt's (ρ_t), and last the registered claim's (no coins). Each absorbs one message but
ζ_c's. -/
def sparseRounds (sh : RecSparse.Shape) : Array SchedRound :=
  let one := fun (n : Nat) (absorbs : Bool) => ((n, #[], 0, absorbs) : SchedRound)
  #[one RecSparse.PRE_ROUNDS false] ++ Array.replicate RecSparse.PRE_ROUNDS (one 1 true) ++ #[one 0 true] ++
    (sh.degrees.map fun _ => one 1 true).toArray ++ #[one sh.pLog true] ++
    (if sh.stacked then #[one sh.tLog true] else #[]) ++ #[one 0 true]

/-- A stream of candidate 3's inner session: rep `x` of the circuit table, rep `x`'s sparse phase, or R's. -/
inductive HoloStream
  | rep (x : Fin 2)
  | sparse (x : Fin 2)
  | stacked

/-- The stream's name in the inner transcript, `t` the circuit table's prefix (`<table>/`). -/
def HoloStream.name (t : String) : HoloStream → String
  | .rep x => s!"{t}rep{x.val}"
  | .sparse x => s!"{SPARSE}/rep{x.val}"
  | .stacked => s!"{STACKED}/rep0"

/-- `rec_live.schedule_for(statement, holo=True, stacked, sparse)`'s streams in its order, each with its rounds: both
reps' at `claims` claims (two and two per region, `holoRounds`); with `sparse` both reps' sparse phases
(`sparseRounds`), each declared at the end of its rep's stream; with `stacked` R's (`stackedRounds`). None where a
schedule is none. -/
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

/-- The rounds of `rs` that absorb a message, in order: the `i`-th is the stream's `i`-th message round, whose `c_k` the
inner transcript records. -/
def msgRounds (rs : Array SchedRound) : List Nat :=
  (List.range rs.size).filter fun k => (rs.getD k default).2.2.2

/-- The rounds of `rs` that draw coins, in order. -/
def coinRounds (rs : Array SchedRound) : List Nat :=
  (List.range rs.size).filter fun k => 0 < (rs.getD k default).1

/-- The tops `rs` commits, in order, each with its round and node count: the `l`-th is the stream's `l`-th top. -/
def topRounds (rs : Array SchedRound) : List (Nat × Nat) :=
  (List.range rs.size).flatMap fun k => (rs.getD k default).2.1.toList.map fun c => (k, c)

/-- **Where the inner transcript holds row `x` of a value `fixedRow` names**, at the reps' rounds `rep`, the RecSparse
shape `sparse` and R's rounds `stacked`: the stream and its round. `rec-c<i>`: rep `x`'s `i`-th message round;
`rec-opening-c<i>`: R's `i`-th, at row 0; `rec-top-stacked-L<l>`: the round committing R's `l`-th top, node `x` of it;
`rec-sparse-<p>` and `rec-claim`: `sparse/rep<x>`'s message round of `p`'s place in `sparseMessages`. None for any other
value, or where the streams have no such round. -/
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
