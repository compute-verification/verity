module

public import PouwCheck.Check

/-!
# PoUW's execution phase check as a `Core.Protocol`

Its parameters are `Params`, its commitment the execution phase's window commitment record (`WindowCommitmentRecord`)
with one-stage's digest, its evidence the proof system's answer, its check `PouwCheck.check`, its public outputs
`window.PUBLIC` and its profile the service's `Core.Integrity`. `pouw-check` runs it through `Core.Protocol.main`.
-/

@[expose] public section

namespace PouwCheck

def protocol : Core.Protocol where
  id := "verity/pouw/v1"
  Params := Params
  Commitment := WindowCommitmentRecord
  Evidence := Evidence
  Public := Public
  Profile := Core.Integrity
  params := paramsOf
  commitment := recordOf
  evidence := evidenceOf
  digest := WindowCommitmentRecord.digest
  check := check
  publicJson := Public.toJson
  profileJson := Core.Integrity.toJson

/-! Each separator below keeps its literal while a lock reads its definition, whose record hashes the value as
written: tied here to its entry of the table (`Identity.Domains`), it reads the entry at its next record change. -/
example : protocol.id = Domains.POUW_PROTOCOL := rfl

end PouwCheck
