import CoreProtocol.Cli
import PouwCheck.Check

/-!
# PoUW's window check as a `Core.Protocol`

Its parameters are `Params`, its commitment the window's `Registration` with one-stage's digest, its evidence the proof
system's answer, its check `PouwCheck.check`, its public outputs `window.PUBLIC` and its profile the service's
`Core.Integrity`. `pouw-check` runs it through `Core.Protocol.main`.
-/

namespace PouwCheck

def protocol : Core.Protocol where
  id := "verity/pouw/v1"
  Params := Params
  Commitment := Registration
  Evidence := Evidence
  Public := Public
  Profile := Core.Integrity
  params := paramsOf
  commitment := registrationOf
  evidence := evidenceOf
  digest := Registration.digest
  check := check
  publicJson := Public.toJson
  profileJson := Core.Integrity.toJson

end PouwCheck
