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
#guard row ic "rec-claim" 0 == none && row ic "rec-acc" 0 == none && row ic "rec-top-L1" 0 == none &&
  row ic "rec-sparse-h" 0 == none

/-! RecSparse at NU 1, ELL 6, S_LOG 1: its message ports `pre0`..`pre5`, `v2`, `sc0`, `sc1`, `u` (ten rounds), and `vt` only
at T > 1. -/

def sp : Flock.RecSparse.Shape := ⟨1, 6, 1, 0⟩
def sp4 : Flock.RecSparse.Shape := ⟨1, 6, 1, 2⟩

#guard sparseMessages sp == ["pre0", "pre1", "pre2", "pre3", "pre4", "pre5", "v2", "sc0", "sc1", "u"]
#guard sparseMessages sp4 == sparseMessages sp ++ ["vt"]

def sparseStream (x base n : Nat) : InnerStream :=
  { name := s!"sparse/rep{x}", commits := (Array.range n).map fun i => (2 * i + 1, b (base + i)), tops := #[] }

def ics : InnerCommits := { reps := ic.reps, streams := #[sparseStream 0 20 10, sparseStream 1 40 10] }

def srow (cs : InnerCommits) (value : String) (x : Nat) (sh : Option Flock.RecSparse.Shape := some sp) :
    Option (Option (List UInt8)) :=
  (fixedRow cs value x sh).map fun r => r.toOption.map (·.2.data.toList)

-- row x of rec-sparse-<p> is sparse/rep<x>'s message round at p's place
#guard srow ics "rec-sparse-pre0" 0 == ok 20 && srow ics "rec-sparse-pre5" 1 == ok 45
#guard srow ics "rec-sparse-v2" 0 == ok 26 && srow ics "rec-sparse-sc1" 1 == ok 48 && srow ics "rec-sparse-u" 0 == ok 29
-- no round commits v, zero or tau, and h and claim are registered as rec-c<i> and rec-claim
#guard ["v", "zero", "tau", "h", "claim", "pre", "scx"].all fun p => srow ics s!"rec-sparse-{p}" 0 == none
-- vt is a port only at T > 1; a port off the shape, a statement that is no rec-sparse unit, a rep with no sparse stream, or
-- a stream of other than the shape's rounds is refused
#guard srow ics "rec-sparse-vt" 0 == gone && srow ics "rec-sparse-pre6" 0 == gone && srow ics "rec-sparse-sc2" 0 == gone
#guard srow ics "rec-sparse-pre0" 0 none == gone && srow ics "rec-sparse-pre0" 2 == gone
#guard srow ics "rec-sparse-pre0" 0 (some sp4) == gone && srow ic "rec-sparse-pre0" 0 == gone
-- the other values are read as before with a shape
#guard srow ics "rec-c1" 0 == ok 2

end Flock.Test.InnerStreams
