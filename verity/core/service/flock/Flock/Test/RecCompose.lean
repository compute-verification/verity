/-
`RecCompose.check`: honest parts compose, one log per V* statement, each accepted at the one staged rows file, the one
inner shape file and one inner session and transcript; a part checked against other rows, a part under another inner
shape, two parts from different inner sessions or transcripts, a part with no rows, inner, session or transcript digest,
a refused session, a missing or repeated part and a log without a `VERDICT` are refused.
-/
import Flock.RecCompose

namespace Flock.Test.RecCompose

open Flock.RecCompose

def rows : String := String.ofList (List.replicate 128 'a')

def other : String := String.ofList (List.replicate 128 'b')

def inner : String := String.ofList (List.replicate 128 'c')

def otherInner : String := String.ofList (List.replicate 128 'd')

def session : String := String.ofList (List.replicate 128 'e')

def otherSession : String := String.ofList (List.replicate 128 'f')

def transcript : String := String.ofList (List.replicate 128 '1')

def otherTranscript : String := String.ofList (List.replicate 128 '2')

def fields (j : Nat) (digest : String := rows) (ok : Bool := true) (innerDigest : String := inner)
    (sessionDigest : String := session) (transcriptDigest : String := transcript) : List (String × Lean.Json) :=
  [("session", .str s!"s{j}"), ("accepted", .bool ok), ("why", .str ""), ("rec_part", Lean.toJson j),
   ("rec_rows_sha512", .str digest), ("rec_inner_sha512", .str innerDigest),
   ("rec_inner_session_sha512", .str sessionDigest), ("rec_inner_transcript_sha512", .str transcriptDigest)]

def verdict (j : Nat) (digest : String := rows) (ok : Bool := true) (innerDigest : String := inner)
    (sessionDigest : String := session) (transcriptDigest : String := transcript) : String :=
  s!"VERDICT {(Lean.Json.mkObj (fields j digest ok innerDigest sessionDigest transcriptDigest)).compress}"

def log (j : Nat) (digest : String := rows) (innerDigest : String := inner) (sessionDigest : String := session)
    (transcriptDigest : String := transcript) : String :=
  let v := verdict j digest (innerDigest := innerDigest) (sessionDigest := sessionDigest)
    (transcriptDigest := transcriptDigest)
  s!"setup: 3 ms\n{v}\n"

/-- Part `j`'s one `VERDICT` without `key`. -/
def without (key : String) (j : Nat := 0) : String :=
  s!"VERDICT {(Lean.Json.mkObj ((fields j).filter (·.1 != key))).compress}\n"

def honest : List (String × String) := [("L0", log 0), ("L1", log 1), ("alg-p0", log 2), ("cap", log 3)]

def passes (n : Nat) (parts : List (String × String)) : Bool := (check n rows inner parts).toBool

#guard passes 4 honest
#guard passes 2 [("L1", log 1), ("L0", log 0 ++ verdict 0)]
-- every part from one inner session and transcript, whichever they are: no file of them reaches the composition
#guard passes 4 (honest.map fun (nm, l) => (nm, (l.replace session otherSession).replace transcript otherTranscript))
#guard match check 4 rows inner honest with
  | .ok pins => pins.map (fun p => (p.key, p.digest)) == [("rec_rows_sha512", rows), ("rec_inner_sha512", inner),
      ("rec_inner_session_sha512", session), ("rec_inner_transcript_sha512", transcript)]
  | .error _ => false

-- a part checked against other rows
#guard !passes 4 (honest.set 2 ("alg-p0", log 2 other))
-- a part whose second session was checked against other rows
#guard !passes 4 (honest.set 0 ("L0", log 0 ++ verdict 0 other))
-- a part under another inner shape, every part under another one than the composition's, a second session under another
#guard !passes 4 (honest.set 1 ("L1", log 1 (innerDigest := otherInner)))
#guard !passes 4 (honest.map fun (nm, l) => (nm, l.replace inner otherInner))
#guard !passes 4 (honest.set 0 ("L0", log 0 ++ verdict 0 (innerDigest := otherInner)))
-- two parts from different inner sessions: the first part's, a later part's, a second session of one part
#guard !passes 2 [("L0", log 0 (sessionDigest := otherSession)), ("L1", log 1)]
#guard !passes 4 (honest.set 3 ("cap", log 3 (sessionDigest := otherSession)))
#guard !passes 4 (honest.set 1 ("L1", log 1 ++ verdict 1 (sessionDigest := otherSession)))
-- two parts at different transcripts of the inner session
#guard !passes 2 [("L0", log 0), ("L1", log 1 (transcriptDigest := otherTranscript))]
#guard !passes 4 (honest.set 0 ("L0", log 0 (transcriptDigest := otherTranscript)))
-- a part with no rows, inner, inner session or transcript digest
#guard (["rec_rows_sha512", "rec_inner_sha512", "rec_inner_session_sha512", "rec_inner_transcript_sha512"].map
  fun k => passes 1 [("L0", without k)]) == [false, false, false, false]
#guard passes 1 [("L0", verdict 0)]
#guard !passes 2 [("L0", log 0), ("L1", without "rec_inner_session_sha512" 1)]
-- a refused session
#guard !passes 4 (honest.set 1 ("L1", verdict 1 (ok := false)))
-- a part missing, a part twice, a part outside V*'s statements
#guard !passes 4 (honest.take 3)
#guard !passes 4 (honest.set 3 ("cap", log 2))
#guard !passes 3 honest
#guard !passes 4 (honest ++ [("L0'", log 0)])
-- no VERDICT, no rec_part, one log naming two parts, a VERDICT that is not JSON, no part at all
#guard !passes 4 (honest.set 3 ("cap", "setup: refused\n"))
#guard !passes 4 (honest.set 0 ("L0", "setup: refused\n"))
#guard !passes 1 [("L0", without "rec_part")]
#guard !passes 2 [("L0", log 0 ++ verdict 1), ("L1", log 1)]
#guard !passes 1 [("L0", "VERDICT {\"accepted\":true\n")]
#guard !passes 0 []

def refusal (parts : List (String × String)) : String :=
  match check 4 rows inner parts with | .error e => e | .ok _ => ""

#guard refusal (honest.set 2 ("alg-p0", log 2 other)) ==
  s!"alg-p0: read the rows file {other}, not the one staged rows file {rows}"
#guard refusal (honest.set 1 ("L1", log 1 (innerDigest := otherInner))) ==
  s!"L1: read the inner shape file {otherInner}, not the composition's inner shape file {inner}"
#guard refusal (honest.set 3 ("cap", log 3 (sessionDigest := otherSession))) ==
  s!"cap: read the inner session record {otherSession}, not L0's inner session record {session}"
#guard refusal (honest.set 2 ("alg-p0", log 2 (transcriptDigest := otherTranscript))) ==
  s!"alg-p0: read the inner transcript {otherTranscript}, not L0's inner transcript {transcript}"
#guard refusal (honest.set 0 ("L0", without "rec_inner_session_sha512")) ==
  "L0: its VERDICT names no rec_inner_session_sha512: it read no inner session record (--rec-inner-session)"

end Flock.Test.RecCompose
