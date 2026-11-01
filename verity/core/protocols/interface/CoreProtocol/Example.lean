/-
A toy protocol that exercises every path of the interface: the `#guard`s below pin its outcomes, and
`core-protocol-example` runs it through `Protocol.main`. It isn't a protocol of Verity's. Its parameters are
`{"k": k}` and its commitment is a 32-byte root, which is its own digest. Its evidence is canonical JSON
`{"drawn": n, "passed": m}`, recording m of n drawn units passed. It accepts when k units were drawn and all passed,
rejects when one failed, and is undecided when fewer than k were drawn.
-/
import CoreProtocol.Cli
import CoreProtocol.Profile

namespace Core.Example

open Lean (Json)
open Flock (V)

structure Evidence where
  drawn : Nat
  passed : Nat

def check (k : Nat) (root : ByteArray) (e : Evidence) : Outcome Nat Integrity :=
  if e.passed < e.drawn then .reject s!"{e.drawn - e.passed} of {e.drawn} drawn units failed"
  else if e.drawn < k then .undecided s!"{e.drawn} of the {k} units drawn"
  else .accept e.passed
    { programs := #[Flock.hexOf root], prescribed := #[], linked := true, covered := #[],
      levels := #[{ name := "unit", query := none, size := e.drawn, draw := some (.subset k) }],
      delta := ⟨1, 1000000⟩, drawn := #[("unit", e.drawn)] }

private def field (j : Json) (k : String) : V Nat :=
  match j.getObjValAs? Nat k with
  | .ok n => .ok n
  | .error _ => .error s!"{k} is missing or not a natural number"

def protocol : Protocol where
  id := "verity/example/v0"
  Params := Nat
  Commitment := ByteArray
  Evidence := Evidence
  Public := Nat
  Profile := Integrity
  params j := field j "k"
  commitment b := if b.size = 32 then .ok b else .error s!"a root is 32 bytes, not {b.size}"
  evidence b := do
    let j ← canonicalJson b
    pure ⟨← field j "drawn", ← field j "passed"⟩
  digest root := root
  check := check
  publicJson n := Json.mkObj [("passed", Json.num n)]
  profileJson := Integrity.toJson

private def root : ByteArray := ⟨(List.replicate 32 (7 : UInt8)).toArray⟩

private def run (params evidence : String) (commitment : ByteArray := root) : String :=
  match (protocol.answer params.toUTF8 commitment evidence.toUTF8).canonical with
  | .ok s => s
  | .error e => s!"error: {e}"

private def rootHex : String := Flock.hexOf root

#guard run "{\"k\":2}" "{\"drawn\":2,\"passed\":2}" ==
  "{\"commitment\":\"" ++ rootHex ++ "\",\"profile\":{\"covered\":[],\"delta\":\"1/1000000\",\"drawn\":{\"unit\":2}," ++
  "\"levels\":[{\"draw\":{\"subset\":2},\"name\":\"unit\",\"query\":null,\"size\":2}],\"linked\":true,\"prescribed\":[]," ++
  "\"programs\":[\"" ++ rootHex ++ "\"]},\"protocol\":\"verity/example/v0\",\"public\":{\"passed\":2},\"reason\":null," ++
  "\"verdict\":\"accept\"}"
#guard run "{\"k\":2}" "{\"drawn\":2,\"passed\":1}" ==
  "{\"commitment\":\"" ++ rootHex ++ "\",\"profile\":null,\"protocol\":\"verity/example/v0\",\"public\":null," ++
  "\"reason\":\"1 of 2 drawn units failed\",\"verdict\":\"reject\"}"
#guard run "{\"k\":3}" "{\"drawn\":2,\"passed\":2}" ==
  "{\"commitment\":\"" ++ rootHex ++ "\",\"profile\":null,\"protocol\":\"verity/example/v0\",\"public\":null," ++
  "\"reason\":\"2 of the 3 units drawn\",\"verdict\":\"undecided\"}"
-- non-canonical parameters, a short root and malformed evidence are rejections, never errors
#guard (protocol.answer "{ \"k\": 2 }".toUTF8 root "{\"drawn\":2,\"passed\":2}".toUTF8).verdict == .reject
#guard (protocol.answer "{\"k\":2}".toUTF8 ByteArray.empty "{\"drawn\":2,\"passed\":2}".toUTF8).digest.isNone
#guard (protocol.answer "{\"k\":2}".toUTF8 root "{\"passed\":2,\"drawn\":2}".toUTF8).reason ==
  some "evidence: not canonical JSON (compact, keys sorted, integers only)"
#guard (protocol.case "{\"commitment\":\"07\",\"evidence\":\"\",\"params\":{\"k\":1}}").reason ==
  some "commitment: a root is 32 bytes, not 1"
#guard (protocol.answer "{}".toUTF8 root "{\"drawn\":2,\"passed\":2}".toUTF8).reason ==
  some "params: k is missing or not a natural number"
#guard (protocol.case "not json").reason.map (·.startsWith "case: not JSON") == some true
-- a character outside printable ASCII is canonical escaped, in lowercase hex, and refused as a raw byte or otherwise spelt
#guard (protocol.answer "{\"k\":2}".toUTF8 root "{\"drawn\":2,\"passed\":2,\"x\":\"d\\u00e9\"}".toUTF8).verdict == .accept
#guard (protocol.answer "{\"k\":2}".toUTF8 root "{\"drawn\":2,\"d\\u007f\":1,\"passed\":2}".toUTF8).verdict == .accept
#guard (protocol.answer "{\"k\":2}".toUTF8 root "{\"drawn\":2,\"passed\":2,\"x\":\"\\ud83d\\ude00\"}".toUTF8).verdict == .accept
#guard (protocol.answer "{\"k\":2}".toUTF8 root "{\"drawn\":2,\"passed\":2,\"x\":\"dé\"}".toUTF8).reason ==
  some "evidence: not ASCII without DEL (a byte at or above 0x7f)"
#guard (protocol.answer "{\"k\":2}".toUTF8 root "{\"drawn\":2,\"passed\":2,\"x\":\"\x7f\"}".toUTF8).reason ==
  some "evidence: not ASCII without DEL (a byte at or above 0x7f)"
#guard (protocol.answer "{\"k\":2}".toUTF8 root "{\"drawn\":2,\"passed\":2,\"x\":\"d\\u00E9\"}".toUTF8).reason ==
  some "evidence: not canonical JSON (compact, keys sorted, integers only)"
#guard (protocol.answer "{\"k\":2}".toUTF8 root "{\"drawn\":2,\"passed\":2,\"x\":\"\\u007e\"}".toUTF8).reason ==
  some "evidence: not canonical JSON (compact, keys sorted, integers only)"
#guard (protocol.answer "{\"k\":2,\"x\":\"dé\"}".toUTF8 root "{\"drawn\":2,\"passed\":2}".toUTF8).reason ==
  some "params: not ASCII without DEL (a byte at or above 0x7f)"
-- an outcome's reason is free text, escaped as any string is
private def freeReason : Answer :=
  { protocol := "p", verdict := .reject, reason := some "Δ ≥ 1", digest := none, published := none, profile := none }
#guard freeReason.canonical.toOption == some ("{\"commitment\":null,\"profile\":null,\"protocol\":\"p\"," ++
  "\"public\":null,\"reason\":\"\\u0394 \\u2265 1\",\"verdict\":\"reject\"}")

private def reread (params evidence : String) : String :=
  let go : V String := do
    let b ← Answer.ofJson (protocol.answer params.toUTF8 root evidence.toUTF8).toJson
    if let some p := b.profile then
      if (← Flock.canon (← Integrity.ofJson p).toJson) != (← Flock.canon p) then throw "the profile did not read back"
    b.canonical
  match go with
  | .ok s => s
  | .error e => s!"error: {e}"

private def readsBack : Outcome Nat Integrity → Bool
  | .accept _ p => match Integrity.ofJson p.toJson with
    | .ok q => decide (q = p)
    | .error _ => false
  | _ => false

private def refusal (s : String) : String :=
  match Json.parse s with
  | .ok j => match Answer.ofJson j with
    | .ok _ => "accepted"
    | .error e => e
  | .error e => e

private def profileRefusal (s : String) : String :=
  match Json.parse s with
  | .ok j => match Integrity.ofJson j with
    | .ok _ => "accepted"
    | .error e => e
  | .error e => e

-- each outcome's answer, and the accept's profile, read back to the same canonical JSON
#guard reread "{\"k\":2}" "{\"drawn\":2,\"passed\":2}" == run "{\"k\":2}" "{\"drawn\":2,\"passed\":2}"
#guard reread "{\"k\":2}" "{\"drawn\":2,\"passed\":1}" == run "{\"k\":2}" "{\"drawn\":2,\"passed\":1}"
#guard reread "{\"k\":3}" "{\"drawn\":2,\"passed\":2}" == run "{\"k\":3}" "{\"drawn\":2,\"passed\":2}"
#guard readsBack (check 2 root ⟨2, 2⟩)
-- the shapes `answer` never returns, and the spellings `toJson` never writes, are refused
#guard refusal "{\"commitment\":null,\"profile\":null,\"protocol\":\"p\",\"public\":null,\"reason\":\"r\",\"verdict\":\"accept\"}" ==
  "answer: an accept has no reason, and both its public outputs and its profile"
#guard refusal "{\"commitment\":null,\"profile\":{},\"protocol\":\"p\",\"public\":null,\"reason\":\"r\",\"verdict\":\"reject\"}" ==
  "answer: a reject has a reason, and neither public outputs nor a profile"
#guard refusal "{\"commitment\":\"0A\",\"profile\":null,\"protocol\":\"p\",\"public\":null,\"reason\":\"r\",\"verdict\":\"reject\"}" !=
  "accepted"
#guard refusal "{\"commitment\":null,\"extra\":1,\"profile\":null,\"protocol\":\"p\",\"public\":null,\"reason\":\"r\",\"verdict\":\"reject\"}" ==
  "answer: not in the form Answer.toJson writes"
#guard refusal "{\"commitment\":null,\"profile\":null,\"protocol\":\"p\",\"public\":null,\"reason\":\"r\",\"verdict\":\"maybe\"}" ==
  "answer.verdict: not accept, reject or undecided"
#guard profileRefusal ("{\"covered\":[],\"delta\":\"2/4\",\"drawn\":{},\"levels\":[],\"linked\":true,\"prescribed\":[]," ++
  "\"programs\":[]}") == "profile.delta: \"2/4\" is not in lowest terms"
#guard profileRefusal ("{\"covered\":[],\"delta\":\"1\",\"drawn\":{},\"levels\":[],\"linked\":true,\"prescribed\":[\"b\",\"a\"]," ++
  "\"programs\":[]}") == "profile: not in the form toJson writes"
#guard profileRefusal ("{\"covered\":[],\"delta\":\"1\",\"drawn\":{\"x\":1},\"levels\":[],\"linked\":true,\"prescribed\":[]," ++
  "\"programs\":[]}") == "profile: not in the form toJson writes"

end Core.Example
