/-
What a part holds whole after the link is the coin server's message there (`RecDeploy.cmt_of_rep`, `held_cmt`): at the
plain session's inner shape, on a record whose every `c_k` differs (`node j` at its `j`-th round) and that
`checkReads` passes, rep `x`'s `kc`-th `c_k` as `innerCommits` holds it is the record's message `x · R + k + 1`, at each
commit round `k` with no salted tops, the `kc`-th commit round of the rep schedule.
-/
import Flock.Test.RecReads

namespace Flock.Test.RecHeld

open Lean Flock.RecVStar Flock.RecRecord Flock.RecSetup Flock.Test.RecReads

/-- `honest` with its `j`-th round's `c_k`, where it commits one, `node j`. -/
def distinct : Record :=
  { honest with
    rounds := honest.rounds.mapIdx fun j x => if x.commits.isEmpty then x else { x with commits := [node j] } }

def trD : Json := (Json.parse (text distinct)).toOption.getD Json.null

/-- Each rep's `c_k` as `innerCommits` holds them, off `distinct`. -/
def csD : InnerCommits :=
  { reps := #[0, 1].map fun (x : Nat) =>
      ((distinct.rounds.filter (·.stream == bs s!"circuit/rep{x}")).filterMap fun r =>
        if r.commits.isEmpty then none else some (r.k, ByteArray.mk (r.commits.headD []).toArray)).toArray }

def rs : Array SchedRound := (repRounds inner.m inner.kLog inner.claims).getD #[]

/-- The commit rounds with no salted tops, as `(kc, k)`: the `kc`-th commit round is round `k`. -/
def bare : List (Nat × Nat) :=
  ((commitRounds rs).zipIdx.map fun (k, kc) => (kc, k)).filter fun (_, k) => (rs.getD k default).2.1.isEmpty

/-- `cmt_of_rep`'s conclusion at rep `x`, at every bare commit round. -/
def held (x : Nat) : Bool :=
  bare.all fun (kc, k) =>
    ((csD.reps.getD x #[])[kc]?.map fun (kk, c) => (kk, c.toList)) ==
      some (k, (distinct.msgs.map Prod.fst).getD (x * rs.size + k + 1) [])

#guard (checkReads inner (text distinct) trD record csD matches .ok ())
#guard readFull (bs (text distinct)) == some distinct
#guard bare.length > 0 && (bare.map (·.2)).eraseDups.length == bare.length
#guard (((csD.reps.getD 0 #[]).toList ++ (csD.reps.getD 1 #[]).toList).map (·.2.toList)).eraseDups.length ==
  (csD.reps.getD 0 #[]).size + (csD.reps.getD 1 #[]).size
#guard held 0 && held 1
#guard !(bare.all fun (kc, k) => ((csD.reps.getD 0 #[])[kc]?.map (·.2.toList)) ==
  some ((distinct.msgs.map Prod.fst).getD (1 * rs.size + k + 1) []))

end Flock.Test.RecHeld
