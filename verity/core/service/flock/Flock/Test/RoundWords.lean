/-
The holders of one message round's row agree on its words (`RecVStar.checkRoundWords`, run by `checkRoundsListed`): every
port of V*'s list under one `rec-c<k>` has the words of the round's first, read off the statements' descriptions
(`ownWords`). At candidate 3's holo shape, RecSparse's `h` and the algebra part's `m<hRound>` (h's 64 elements) are both
512 words; a list whose holders of a round differ is refused, naming both. RecSparse's `claim` (`rec-claim`) and R's
opening's ports (`rec-opening-c<k>`) are no `rec-c<k>`, so they are not compared.
-/
import Flock.RecVStar

namespace Flock.Test.RoundWords

open Flock Flock.RecVStar

def sp18 : RecSparse.Shape := ⟨18, 13, 6, 0⟩

def part (j : Nat) (ports : List (String × Nat)) : VStar.Stmt :=
  .inner j s!"rec-residuals/m25-k22-c8-p{j}" { S := { gf := default, ports }, res := [], v := [] }

def err (x : Except String Unit) : Option String := match x with | .error e => some e | .ok _ => none

#guard VStar.portWordsOf 64 == 512 && VStar.portWordsOf 65 == 576 && VStar.portWordsOf 2 == 64
#guard ownWords (.sparse sp18 38) "h" == some 512 && ownWords (.sparse sp18 38) "claim" == some 64
#guard ownWords (part 0 [("m38", 64), ("acc0", 2)]) "m38" == some 512
#guard ownWords (part 0 [("m38", 64)]) "m37" == none && ownWords (.cap 0 128) "top" == none

-- RecSparse holds `h` under rec-c38 and its claim under rec-claim, which is no round's
#guard roundPorts #[.sparse sp18 38] == [(38, some 512, "rec-sparse/n18-e13-s6-t0", "h")]
#guard roundPorts #[.opening 23 2 0 ⟨[], []⟩ [("m3", 128), ("claim1", 1)] 1, .cap 0 128] == []

-- candidate 3's holo list: the part with h's round and RecSparse agree, as do parts holding distinct rounds
#guard err (checkRoundWords #[part 0 [("m37", 2), ("m38", 64)], part 1 [("m39", 2)], .sparse sp18 38]) == none
#guard err (checkRoundWords #[]) == none

-- two holders of rec-c38 at different words are refused, whichever comes first
#guard err (checkRoundWords #[part 0 [("m38", 65)], .sparse sp18 38]) == some
  "V*'s list: a rep's message round 38 (rec-c38) is held at rec-residuals/m25-k22-c8-p0's port m38 as 576 words and at rec-sparse/n18-e13-s6-t0's port h as 512"
#guard err (checkRoundWords #[.sparse sp18 38, part 0 [("m38", 65)]]) == some
  "V*'s list: a rep's message round 38 (rec-c38) is held at rec-sparse/n18-e13-s6-t0's port h as 512 words and at rec-residuals/m25-k22-c8-p0's port m38 as 576"
#guard err (checkRoundWords #[part 0 [("m5", 2)], part 1 [("m5", 9)]]) == some
  "V*'s list: a rep's message round 5 (rec-c5) is held at rec-residuals/m25-k22-c8-p0's port m5 as 64 words and at rec-residuals/m25-k22-c8-p1's port m5 as 128"
-- elements that round to one width agree
#guard err (checkRoundWords #[part 0 [("m5", 2)], part 1 [("m5", 8)]]) == none

end Flock.Test.RoundWords
