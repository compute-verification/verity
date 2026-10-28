/-
`RecCompose.check`: honest parts compose, one log per V* statement, each accepted at the one staged rows file; a part
checked against other rows, a part with no rows digest, a refused session, a missing or repeated part and a log without
a `VERDICT` are refused.
-/
import Flock.RecCompose

namespace Flock.Test.RecCompose

open Flock.RecCompose

def rows : String := String.ofList (List.replicate 128 'a')

def other : String := String.ofList (List.replicate 128 'b')

def verdict (j : Nat) (digest : String := rows) (ok : Bool := true) : String :=
  s!"VERDICT \{\"session\":\"s{j}\",\"accepted\":{ok},\"why\":\"\",\"rec_part\":{j},\"rec_rows_sha512\":\"{digest}\"}"

def log (j : Nat) (digest : String := rows) : String := s!"setup: 3 ms\n{verdict j digest}\n"

def honest : List (String × String) := [("L0", log 0), ("L1", log 1), ("alg-p0", log 2), ("cap", log 3)]

def passes (n : Nat) (parts : List (String × String)) : Bool := (check n rows parts).toBool

#guard passes 4 honest
#guard passes 2 [("L1", log 1), ("L0", log 0 ++ verdict 0)]

-- a part checked against other rows
#guard !passes 4 (honest.set 2 ("alg-p0", log 2 other))
-- a part whose second session was checked against other rows
#guard !passes 4 (honest.set 0 ("L0", log 0 ++ verdict 0 other))
-- a part with no rows digest
#guard !passes 1 [("L0", "VERDICT {\"session\":\"s\",\"accepted\":true,\"why\":\"\",\"rec_part\":0}\n")]
-- a refused session
#guard !passes 4 (honest.set 1 ("L1", verdict 1 (ok := false)))
-- a part missing, a part twice, a part outside V*'s statements
#guard !passes 4 (honest.take 3)
#guard !passes 4 (honest.set 3 ("cap", log 2))
#guard !passes 3 honest
#guard !passes 4 (honest ++ [("L0'", log 0)])
-- no VERDICT, no rec_part, one log naming two parts, a VERDICT that is not JSON, no part at all
#guard !passes 4 (honest.set 3 ("cap", "setup: refused\n"))
#guard !passes 1 [("L0", "VERDICT {\"accepted\":true,\"rec_rows_sha512\":\"" ++ rows ++ "\"}\n")]
#guard !passes 2 [("L0", log 0 ++ verdict 1), ("L1", log 1)]
#guard !passes 1 [("L0", "VERDICT {\"accepted\":true\n")]
#guard !passes 0 []

#guard match check 4 rows (honest.set 2 ("alg-p0", log 2 other)) with
  | .error e => e == s!"alg-p0: checked against the rows file {other}, not the one staged rows file {rows}"
  | .ok () => false

end Flock.Test.RecCompose
