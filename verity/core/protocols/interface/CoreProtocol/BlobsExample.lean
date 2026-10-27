/-
A toy `BlobProtocol` whose `#guard`s pin `Blobs` and `BlobProtocol.answer` and `case`: `Example.protocol`'s parameters and
commitment, and evidence `{"drawn": n, "passed": m, "report": digest}`, where the blob with that SHA-512 must be the bytes
`passed` (else the check rejects). It isn't a protocol of Verity's.
-/
import CoreProtocol.Blobs
import CoreProtocol.Example

namespace Core.BlobsExample

open Lean (Json)
open Flock (V)

structure Evidence where
  base : Example.Evidence
  report : String

def protocol : BlobProtocol :=
  { Example.protocol with
    Evidence := Evidence
    evidence := fun b => do
      let j ← canonicalJson b
      let some (.str r) := (j.getObjVal? "report").toOption | throw "report is missing or not a string"
      let base ← Example.protocol.evidence
        (← Flock.canon (Json.mkObj [("drawn", j.getObjValD "drawn"), ("passed", j.getObjValD "passed")])).toUTF8
      pure ⟨base, r⟩
    check := fun k root e => Example.check k root e.base
    checkBlobs := fun k root e bs =>
      match bs.get e.report "report" with
      | .error why => .reject why
      | .ok b => if b == "passed".toUTF8 then Example.check k root e.base else .reject "the report does not say passed" }

private def root : ByteArray := ⟨(List.replicate 32 (7 : UInt8)).toArray⟩

private def digestOf (s : String) : String := Flock.hexOf (Flock.Sha512.hash s.toUTF8)

private def evidence (report : String) : ByteArray :=
  s!"\{\"drawn\":2,\"passed\":2,\"report\":\"{report}\"}".toUTF8

private def verdict (report : String) (blobs : Array String) : Verdict × Option String :=
  let a := protocol.answer "{\"k\":2}".toUTF8 root (evidence report) (blobs.map String.toUTF8)
  (a.verdict, a.reason)

-- a blob is found by the SHA-512 of its own bytes, and only by that
#guard verdict (digestOf "passed") #["passed"] == (.accept, none)
#guard verdict (digestOf "passed") #["other", "passed"] == (.accept, none)
#guard verdict (digestOf "failed") #["failed"] == (.reject, some "the report does not say passed")
#guard verdict (digestOf "passed") #["failed"] ==
  (.reject, some s!"report: no blob has the SHA-512 {digestOf "passed"}")
#guard verdict (digestOf "passed") #[] == (.reject, some s!"report: no blob has the SHA-512 {digestOf "passed"}")
#guard (verdict "ab" #["passed"]).2 == some "report: ab is not a SHA-512 digest (128 lowercase hex)"
#guard (verdict ((digestOf "passed").toUpper) #["passed"]).1 == .reject
-- `toProtocol` is the protocol without blobs, which a composition takes unchanged
#guard (protocol.toProtocol.answer "{\"k\":2}".toUTF8 root (evidence (digestOf "passed"))).verdict == .accept
#guard (protocol.answer "{\"k\":2}".toUTF8 root "{\"drawn\":2,\"passed\":2}".toUTF8 #[]).reason ==
  some "evidence: report is missing or not a string"

private def caseLine (blobs : Option String) : String :=
  "{" ++ (match blobs with | some b => s!"\"blobs\":{b}," | none => "") ++
  s!"\"commitment\":\"{Flock.hexOf root}\",\"evidence\":\"{Flock.hexOf (evidence (digestOf "passed"))}\"," ++
  "\"params\":{\"k\":2}}"

-- a serve case carries its blobs as hex under `blobs`; without them the check has none
#guard (protocol.case (caseLine (some s!"[\"{Flock.hexOf "passed".toUTF8}\"]"))).verdict == .accept
#guard (protocol.case (caseLine none)).verdict == .reject
#guard (protocol.case (caseLine (some "[1]"))).verdict == .reject
#guard (protocol.case (caseLine (some "\"x\""))).reason == some "case: the case's blobs are not a list of hex strings"

end Core.BlobsExample
