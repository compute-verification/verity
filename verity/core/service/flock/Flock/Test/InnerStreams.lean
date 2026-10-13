/-
`RecVStar.fixedRow` on an inner session of two reps of two message rounds each and R's stream (one message round, a level-0
top of two nodes and a level-1 top of one): what row `x` of each value the inner transcript fixes is.
-/
import Flock.RecVStar

namespace Flock.Test.InnerStreams

open Flock.RecVStar

def b (n : Nat) : ByteArray := ByteArray.mk (Array.replicate 128 n.toUInt8)

def ic : InnerCommits :=
  { reps := #[#[(2, b 1), (5, b 2)], #[(2, b 3), (5, b 4)]],
    streams := #[{ name := "stacked/rep0", commits := #[(0, b 5)], tops := #[#[b 6, b 7], #[b 8]] }] }

/-- `none` for a value the transcript does not fix, `some none` for a row it has not, else the row's bytes. -/
def row (cs : InnerCommits) (value : String) (x : Nat) : Option (Option (List UInt8)) :=
  (fixedRow cs value x).map fun r => r.toOption.map (·.2.data.toList)

def ok (n : Nat) : Option (Option (List UInt8)) := some (some (b n).data.toList)
def gone : Option (Option (List UInt8)) := some none

#guard row ic "rec-c1" 0 == ok 2 && row ic "rec-c0" 1 == ok 3
#guard row ic "rec-opening-c0" 0 == ok 5
#guard row ic "rec-top-stacked-L0" 1 == ok 7 && row ic "rec-top-stacked-L1" 0 == ok 8
#guard row ic "rec-c2" 0 == gone && row ic "rec-c0" 2 == gone
-- R's opening is one rep's: row 1 of its message round is no row the transcript fixes
#guard row ic "rec-opening-c0" 1 == gone && row ic "rec-opening-c1" 0 == gone
#guard row ic "rec-top-stacked-L1" 1 == gone && row ic "rec-top-stacked-L2" 0 == gone
-- a session of one table's reps holds no stream of R's
#guard row { reps := ic.reps } "rec-opening-c0" 0 == gone
#guard row ic "rec-acc" 0 == none && row ic "rec-top-L1" 0 == none && row ic "rec-sparse-h" 0 == none
-- rec-claim is the sparse phase's: without the verifier's RecSparse shape it is fixed by no row
#guard row ic "rec-claim" 0 == gone

/-! RecSparse at NU 1, ELL 6, S_LOG 1: its message ports `pre0`..`pre5`, `v2`, `sc0`, `sc1`, `u`, `vt` only at T > 1, then
the claim round's `claim` (eleven rounds at T = 1). -/

def sp : Flock.RecSparse.Shape := ⟨1, 6, 1, 0⟩
def sp4 : Flock.RecSparse.Shape := ⟨1, 6, 1, 2⟩

#guard sparseMessages sp == ["pre0", "pre1", "pre2", "pre3", "pre4", "pre5", "v2", "sc0", "sc1", "u", "claim"]
#guard sparseMessages sp4 == ["pre0", "pre1", "pre2", "pre3", "pre4", "pre5", "v2", "sc0", "sc1", "u", "vt", "claim"]

def sparseStream (x base n : Nat) : InnerStream :=
  { name := s!"sparse/rep{x}", commits := (Array.range n).map fun i => (2 * i + 1, b (base + i)), tops := #[] }

def ics : InnerCommits := { reps := ic.reps, streams := #[sparseStream 0 20 11, sparseStream 1 40 11] }

def srow (cs : InnerCommits) (value : String) (x : Nat) (sh : Option Flock.RecSparse.Shape := some sp) :
    Option (Option (List UInt8)) :=
  (fixedRow cs value x sh).map fun r => r.toOption.map (·.2.data.toList)

-- row x of rec-sparse-<p> is sparse/rep<x>'s message round at p's place
#guard srow ics "rec-sparse-pre0" 0 == ok 20 && srow ics "rec-sparse-pre5" 1 == ok 45
#guard srow ics "rec-sparse-v2" 0 == ok 26 && srow ics "rec-sparse-sc1" 1 == ok 48 && srow ics "rec-sparse-u" 0 == ok 29
-- no round commits v, zero or tau, and h and claim are registered as rec-c<i> and rec-claim
#guard ["v", "zero", "tau", "h", "claim", "pre", "scx"].all fun p => srow ics s!"rec-sparse-{p}" 0 == none
-- rec-claim's row x is the verifier's own (`claimRow`): sparse/rep<x>'s last message round, the claim round's c_k, at the
-- verifier's RecSparse shape, so a statement that is no rec-sparse unit (R's opening) reads it too
#guard srow ics "rec-claim" 0 == ok 30 && srow ics "rec-claim" 1 == ok 50
#guard (claimRow ics sp 1).toOption.map (·.1) == some "sparse/rep1's c_10 (claim, inner round 21)"
-- without the claim round, at another shape, or for a third rep, rec-claim has no row
#guard srow { ics with streams := #[sparseStream 0 20 10, sparseStream 1 40 10] } "rec-claim" 0 == gone
#guard srow ics "rec-claim" 0 (some sp4) == gone && srow ics "rec-claim" 2 == gone
-- vt is a port only at T > 1; a port off the shape, a statement that is no rec-sparse unit, a rep with no sparse stream, or
-- a stream of other than the shape's rounds is refused
#guard srow ics "rec-sparse-vt" 0 == gone && srow ics "rec-sparse-pre6" 0 == gone && srow ics "rec-sparse-sc2" 0 == gone
#guard srow ics "rec-sparse-pre0" 0 none == gone && srow ics "rec-sparse-pre0" 2 == gone
#guard srow ics "rec-sparse-pre0" 0 (some sp4) == gone && srow ic "rec-sparse-pre0" 0 == gone
-- the other values are read as before with a shape
#guard srow ics "rec-c1" 0 == ok 2

/-! R's shape: `stackedRounds` is `rec_live.stacked_rounds` (36 rounds at m 22 to 24, the binding round first, the cap 128
nodes; the three differ past it), and an R opening unit is held to the shape R's stream was. -/

def r23 : StackedShape := ⟨23, 2⟩

#guard [22, 23, 24].all fun m => ((stackedRounds ⟨m, 2⟩).map (·.size)) == some 36
#guard ((stackedRounds r23).bind (·[0]?)) == some (7, #[128], 0, true)
#guard ((stackedRounds r23).bind (·[3]?)) == some (16, #[], 1, true)
#guard stackedRounds ⟨22, 2⟩ != stackedRounds r23 && stackedRounds ⟨24, 2⟩ != stackedRounds r23
#guard stackedRounds ⟨23, 1⟩ != stackedRounds r23 && stackedRounds ⟨21, 2⟩ == none

def held (sh : Option StackedShape) : InnerCommits := { ic with stacked := sh }

def why (x : Flock.V Unit) : Option String := match x with | .error e => some e | .ok _ => none

#guard (checkOpeningShape "rec-opening/m23-c2-p0" (held (some r23))).toOption.isSome
#guard (checkOpeningShape "rec-opening/m23-c2-p3" (held (some r23))).toOption.isSome
#guard why (checkOpeningShape "rec-opening/m24-c2-p0" (held (some r23))) == some
  "FIREWALL-REFUSED: registration: rec-opening/m24-c2-p0 opens R at (m 24, claims 2), not at the shape staging committed R at (m 23, claims 2, META's stacked)"
#guard (checkOpeningShape "rec-opening/m23-c3-p0" (held (some r23))).toOption.isNone
#guard why (checkOpeningShape "rec-opening/m23-c2-p0" (held none)) == some
  "FIREWALL-REFUSED: registration: rec-opening/m23-c2-p0 opens R at (m 23, claims 2), not at the shape staging committed R at (no shape, META's stacked)"
#guard (checkOpeningShape "rec-sparse/n18-e13-s6-t0" (held none)).toOption.isSome

/-! Candidate 3's V* list at the e2e's shape (m 25, k_log 22, the bucket's RecSparse (18, 13, 6, 0), R at (m 23, claims 2)):
RecSparse, R's three levels at `fast100(23)` under names of their own, and R's opening. A list missing any is refused by
name, as is an opening at a shape other than META's. -/

def sp18 : Flock.RecSparse.Shape := ⟨18, 13, 6, 0⟩
def c3 : Inner := { m := 25, kLog := 22, claims := 8, regions := 3, sparse := some sp18, stacked := some r23 }
def gfs0 : Flock.RecOpen.GfS := ⟨[], []⟩
def lv : List VStar.Stmt := [.recOpenStacked 0 64 4 218, .recOpenStacked 1 16 3 106, .recOpenStacked 2 16 1 71]
def op (m : Nat) : VStar.Stmt := .opening m 2 0 gfs0 [] 1
def full : List VStar.Stmt := [.sparse sp18 (hRound 25 22)] ++ lv ++ [op 23]
def err (x : Except String Unit) : Option String := match x with | .error e => some e | .ok _ => none

#guard hRound 25 22 == 38 && tableM sp18 == 23
#guard (fast100 23).map (fun s => s.levels.toList.map fun l => (2 ^ l.lanes, l.d - l.cap, l.q)) ==
  some [(64, 4, 218), (16, 3, 106), (16, 1, 71)]
#guard lv.map (·.name) == ["rec-open-stacked/l64-h4", "rec-open-stacked/l16-h3", "rec-open-stacked/l16-h1"]
#guard (VStar.Stmt.sparse sp18 38).name == "rec-sparse/n18-e13-s6-t0" && (op 23).name == "rec-opening/m23-c2-p0"
#guard openingPortName 25 0 == "m0" && openingPortName 25 24 == "m24" && openingPortName 25 25 == "acc0" &&
  openingPortName 25 26 == "acc1" && openingPortName 25 28 == "claim1"
#guard ownPorts (.sparse sp18 38) == [("h", "rec-c38", #[some 0, some 1]), ("claim", "rec-claim", #[some 0, some 1])]
#guard ownPorts (.opening 23 2 0 gfs0 [("m3", 128), ("acc0", 128), ("claim1", 1)] 1) ==
  [("m3", "rec-opening-c3", #[some 0]), ("claim1", "rec-claim", #[some 1])]

-- each claim port reads its own rep's row of rec-claim: RecSparse's `claim` rep r's at instance r, R's opening's `claim<k>`
-- row k; a reads file with a claim port at another row is refused
def port (name value : String) (reads : List Nat) : Registered.Port :=
  { name, value, root := ByteArray.empty, leaves := 2, words := none, wordBits := 16, domain := ByteArray.empty,
    reads := (reads.map some).toArray }
def sparseMsgsOwn (rd : List Nat) : Array Registered.Port :=
  ((sparseMsgs (.sparse sp18 38)).map fun q => port q.1 q.2.1 rd).toArray
def sparseOwn (claim : List Nat) : Array Registered.Port :=
  sparseMsgsOwn [0, 1] ++ #[port "h" "rec-c38" [0, 1], port "claim" "rec-claim" claim, port "v" "rec-sparse-v" [0, 1]]
def openOwn (claim1 : List Nat) : Array Registered.Port :=
  #[port "m3" "rec-opening-c3" [0], port "acc0" "rec-acc" [4], port "claim1" "rec-claim" claim1]
def openSt : VStar.Stmt := .opening 23 2 0 gfs0 [("m3", 128), ("acc0", 128), ("claim1", 1)] 1

#guard err (checkOwn (.sparse sp18 38) (sparseOwn [0, 1])) == none
#guard err (checkOwn (.sparse sp18 38) (sparseOwn [1, 0])) == some
  "rec-sparse/n18-e13-s6-t0: the registered reads' port claim under rec-claim reading [1, 0] is not the statement's: the statement's is claim under rec-claim reading [0, 1]"
#guard err (checkOwn (.sparse sp18 38) (sparseOwn [0, 1]).pop.pop) == some
  "rec-sparse/n18-e13-s6-t0: the statement's message-round port claim under rec-claim reading [0, 1] is not in the registered reads (1 message-round ports)"
-- RecSparse's other messages are read under `rec-sparse-<p>` at both reps' rows; a file missing one, or reading one at
-- other rows, is refused
#guard err (checkOwn (.sparse sp18 38) ((sparseOwn [0, 1]).extract 1 (sparseOwn [0, 1]).size)) == some
  "rec-sparse/n18-e13-s6-t0: the statement's message port pre0 under rec-sparse-pre0 reading [0, 1] is not in the registered reads"
#guard err (checkOwn (.sparse sp18 38) (sparseMsgsOwn [1, 0] ++ (sparseOwn [0, 1]).extract (sparseMsgsOwn [0, 1]).size 100))
  == some "rec-sparse/n18-e13-s6-t0: the statement's message port pre0 under rec-sparse-pre0 reading [0, 1] is not in the registered reads"
#guard claimIndex "claim12" == some 12 && claimIndex "claim01" == none && claimIndex "claim" == none
#guard err (checkOwn openSt (openOwn [1])) == none
#guard err (checkOwn openSt (openOwn [0])) == some
  "rec-opening/m23-c2-p0: the registered reads' port claim1 under rec-claim reading [0] is not the statement's: the statement's is claim1 under rec-claim reading [1]"

#guard err (checkCandidate3 c3 full.toArray) == none
#guard err (checkCandidate3 c3 (full.drop 1).toArray) ==
  some "candidate 3's V* list has no RecSparse statement (rec-sparse/n18-e13-s6-t0), and the inner shape names its sparse phase"
#guard err (checkCandidate3 c3 (full.filter (·.name != "rec-open-stacked/l16-h3")).toArray) ==
  some "candidate 3's V* list has no level 1 of R's openings (rec-open-stacked/…), and the inner shape names R (m 23, claims 2)"
#guard err (checkCandidate3 c3 full.dropLast.toArray) ==
  some "candidate 3's V* list has no part of R's opening (rec-opening/m23-c2-p0), and the inner shape names R (m 23, claims 2)"
#guard err (checkCandidate3 c3 (full.dropLast ++ [op 24]).toArray) ==
  some "candidate 3's V* list: rec-opening/m24-c2-p0 opens R at (m 24, claims 2), not at R's shape (m 23, claims 2, META's stacked)"
-- a session without candidate 3's phases needs none of them
#guard err (checkCandidate3 { c3 with sparse := none, stacked := none } #[]) == none

-- the inner shape's own refusals, before R's opening is computed
#guard err ((candidate3Statements { c3 with sparse := none }).map fun _ => ()) == some
  "candidate 3's V* list: the inner shape names R's shape (m 23, claims 2) and no RecSparse shape (rec-inner --sparse): R opens the sparse phase's claims, so RecSparse is missing"
#guard err ((candidate3Statements { c3 with stacked := some ⟨24, 2⟩ }).map fun _ => ()) ==
  some "candidate 3's V* list: R's m 24 is not rec-sparse/n18-e13-s6-t0's table_m 23"
#guard err ((candidate3Statements { c3 with stacked := some ⟨23, 3⟩ }).map fun _ => ()) ==
  some "candidate 3's V* list: R's claims 3, not the two reps' sparse claims (2)"
#guard err ((candidate3Statements { c3 with stacked := none }).map fun _ => ()) == some
  "candidate 3's V* list: the inner shape names RecSparse (rec-sparse/n18-e13-s6-t0) and no R's shape (stacked): R opens the sparse phases' claims, so R is missing"

-- candidate 3's inner shape is its inner statement's: a sparse phase, and the statement's m, k_log and regions
#guard err (checkInnerShape c3 25 22 3) == none
#guard err (checkInnerShape { c3 with sparse := none } 25 22 3) == some
  "rec-inner: candidate 3's inner shape names no RecSparse shape (rec-inner --sparse): its V* list would have no RecSparse statement, level of R's or part of R's opening"
#guard err (checkInnerShape c3 24 22 3) == some
  "rec-inner: the verifier's inner shape (m 25, k_log 22, regions 3) is not the inner statement's (m 24, k_log 22, regions 3)"
#guard err (checkInnerShape c3 25 21 3) == some
  "rec-inner: the verifier's inner shape (m 25, k_log 22, regions 3) is not the inner statement's (m 25, k_log 21, regions 3)"
#guard err (checkInnerShape c3 25 22 4) == some
  "rec-inner: the verifier's inner shape (m 25, k_log 22, regions 3) is not the inner statement's (m 25, k_log 22, regions 4)"

-- a unit under a name none of the list's is refused with why: an R level under the circuit table's name, an opening or a
-- RecSparse at another shape
#guard unlisted c3 "rec-open/l64-h4" full.toArray |>.startsWith
  "rec-open/l64-h4: R's level stacked/L0 is rec-open-stacked/l64-h4, never under the circuit table's name rec-open/…; "
#guard unlisted c3 "rec-opening/m24-c2-p0" full.toArray ==
  "rec-opening/m24-c2-p0: opens R at (m 24, claims 2), not at R's shape (m 23, claims 2, the inner circuit's META's stacked)"
#guard unlisted { c3 with stacked := none } "rec-opening/m23-c2-p0" #[] ==
  "rec-opening/m23-c2-p0: one of R's opening parts, and the verifier's inner shape names no R (rec-inner: the inner circuit's META's stacked)"
#guard unlisted c3 "rec-sparse/n18-e13-s6-t1" full.toArray ==
  "rec-sparse/n18-e13-s6-t1: a RecSparse unit, and the bucket's is rec-sparse/n18-e13-s6-t0 (rec-inner --sparse)"
#guard unlisted { c3 with sparse := none } "rec-sparse/n18-e13-s6-t0" #[] ==
  "rec-sparse/n18-e13-s6-t0: a RecSparse unit, and the verifier's inner shape names no sparse phase (rec-inner --sparse)"
#guard isVStar "rec-open-stacked/l16-h1" && isVStar "rec-sparse/n18-e13-s6-t0" && isVStar "rec-opening/m23-c2-p0"

-- the inner shape covers the transcript: candidate 3's streams need the shapes, and R's is META's; each stream's c_k
-- sit at its schedule's message rounds, and R's tops at its schedule's
def atRounds (ks : List Nat) (base : Nat) : Array (Nat × ByteArray) := ks.toArray.zipIdx.map fun (k, i) => (k, b (base + i))
def sparseAt (x : Nat) (ks : List Nat := commitRounds (sparsePhaseRounds sp18)) : InnerStream :=
  { name := s!"sparse/rep{x}", commits := atRounds ks (50 * x), tops := #[] }
def rRounds : Array SchedRound := (stackedRounds r23).getD #[]
def stackedAt (ks : List Nat := commitRounds rRounds) (ns : List Nat := topCounts rRounds) : InnerStream :=
  { name := "stacked/rep0", commits := atRounds ks 200, tops := ns.toArray.map fun n => Array.replicate n (b 9) }
def c3cs (sh : Option StackedShape) (rs : Option (Array SchedRound) := repRounds 25 22 8) : InnerCommits :=
  { reps := #[0, 1].map fun x => atRounds ((rs.map commitRounds).getD []) (100 + 50 * x),
    streams := #[sparseAt 0, sparseAt 1, stackedAt], stacked := sh }

#guard why (checkCovers c3 (c3cs (some r23))) == none
#guard why (checkCovers { c3 with sparse := none } (c3cs (some r23))) == some
  "rec-inner: the inner transcript holds candidate 3's sparse phases, and the verifier's inner shape names no RecSparse shape (rec-inner --sparse): its V* list would have no RecSparse statement"
#guard why (checkCovers c3 { reps := ic.reps }) == some
  "rec-inner: the verifier's inner shape names a RecSparse shape (--sparse), and the inner transcript holds no sparse phase"
#guard why (checkCovers { c3 with stacked := none } (c3cs (some r23))) == some
  "rec-inner: the inner transcript holds R's stream (stacked/rep0), and the verifier's inner shape names no shape of R (stacked): its V* list would have no level of R's or R's opening"
#guard why (checkCovers { c3 with stacked := some ⟨24, 2⟩ } (c3cs (some r23))) == some
  "rec-inner: the verifier's inner shape names R's shape (m 24, claims 2), and the inner circuit's META (m 23, claims 2) (stacked)"
-- private circuits' reps split z_partial's round (`holoRepRounds`): their transcript passes under the verifier's `holo`
-- alone; a rep's c_k one round off, a rep missing, and a sparse stream's c_k off its rounds or short of one are refused
def holoCs : InnerCommits := c3cs (some r23) (holoRepRounds 25 22 8)
def spRounds : List Nat := commitRounds (sparsePhaseRounds sp18)

#guard why (checkCovers { c3 with holo := true } holoCs) == none
#guard why (checkCovers c3 holoCs) == some
  "rec inner transcript: rep 0's message round 46 is at round 49, and its schedule's at round 50 (77 message rounds, the schedule's 76)"
#guard why (checkCovers { c3 with holo := true } (c3cs (some r23))) == some
  "rec inner transcript: rep 0's message round 46 is at round 50, and its schedule's at round 49 (76 message rounds, the schedule's 77)"
#guard why (checkCovers c3 { c3cs (some r23) with
    reps := (c3cs (some r23)).reps.modify 1 fun cm => cm.modify 5 fun (k, c) => (k + 1, c) }) == some
  "rec inner transcript: rep 1's message round 5 is at round 8, and its schedule's at round 7 (76 message rounds, the schedule's 76)"
#guard why (checkCovers c3 { c3cs (some r23) with reps := (c3cs (some r23)).reps.pop }) == some
  "rec inner transcript: 1 reps, not the circuit table's two"
def withS (xs : Array InnerStream) : InnerCommits := { c3cs (some r23) with streams := xs }
def msgAt (s : String) (j : Nat) (got want : String) (n m : Nat) : Option String :=
  some (s!"rec inner transcript: {s}'s message round {j} is at {got}, and its schedule's at {want} " ++
    s!"({n} message rounds, the schedule's {m})")

#guard why (checkCovers c3 (withS #[sparseAt 0, sparseAt 1 (spRounds.map (· + 1)), stackedAt])) ==
  msgAt "sparse/rep1" 0 "round 2" "round 1" 33 33
#guard why (checkCovers c3 (withS #[sparseAt 0 spRounds.dropLast, sparseAt 1, stackedAt])) ==
  msgAt "sparse/rep0" 32 "no round" "round 33" 32 33
-- R's stream at (m 23, claims 2): 25 message rounds and tops of 128, 64 and 64 nodes. Its c_k one round off or one
-- short, its tops other than its rounds commit, and R's stream under an inner shape naming no schedule of R's are
-- refused
def withR (st : InnerStream) : InnerCommits := withS #[sparseAt 0, sparseAt 1, st]
def rMsgs : List Nat := commitRounds rRounds
def badTops (ns : List Nat) : Option String :=
  some s!"rec inner transcript: stacked/rep0 holds tops of {ns} nodes, and its schedule commits tops of [128, 64, 64]"

#guard rMsgs.length == 25 && topCounts rRounds == [128, 64, 64]
#guard why (checkCovers c3 (withR (stackedAt (rMsgs.set 1 2)))) == msgAt "stacked/rep0" 1 "round 2" "round 1" 25 25
#guard why (checkCovers c3 (withR (stackedAt rMsgs.dropLast))) == msgAt "stacked/rep0" 24 "no round" "round 33" 24 25
#guard why (checkCovers c3 (withR (stackedAt (ns := [128, 64])))) == badTops [128, 64]
#guard why (checkCovers c3 (withR (stackedAt (ns := [128, 64, 32])))) == badTops [128, 64, 32]
#guard [{ c3 with stacked := none }, { c3 with stacked := some ⟨40, 2⟩ }].all fun i =>
  why (checkCommitRounds i (c3cs (some r23))) == some ("rec-inner: the inner transcript holds R's stream " ++
    "(stacked/rep0), and the verifier's inner shape names no schedule of R's")
#guard why (checkCommitRounds { c3 with stacked := none } (withS #[sparseAt 0, sparseAt 1])) == none
-- the check reads sparse/rep0 and sparse/rep1 only: any other held stream named sparse/… is refused
def otherSparse (n : String) : Option String :=
  why (checkCovers c3 (withS #[sparseAt 0, sparseAt 1, stackedAt, { sparseAt 1 with name := n }]))

#guard ["sparse/rep2", "sparse/rep10", "sparse/rep01", "sparse/x", "sparse/"].all fun n => otherSparse n ==
  some s!"rec inner transcript: it holds {n}, and a sparse phase is one of the two reps' (sparse/rep0, sparse/rep1)"
#guard otherSparse "sparse-rep2" == none && otherSparse "sparse/rep1" == none

end Flock.Test.InnerStreams
