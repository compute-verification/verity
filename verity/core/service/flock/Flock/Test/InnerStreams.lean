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

end Flock.Test.InnerStreams
