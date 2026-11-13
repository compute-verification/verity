module

public import VStarRecursive.SampledTwo

/-!
# A test's executable: the two-stage form's check with its switch on

`sampled-two-on` is `sampled-two` with the check at the switch on (`VStar.Recursive.SampledTwo.checkBlobsAtWith true`),
under an id of its own. The form's accept test drives it on one honest play, so that a change after which the form
accepts no play fails a test. No party runs it and nothing dispatches to it: the form's own check (`sampled-two`,
`VStar.Recursive.SampledTwo.protocol`) keeps its switch (`FORM`), and sampled proofs refuses the form by name.
-/

@[expose] public section

namespace VStar.Recursive.SampledTwo

/-- The form's protocol with its check at the switch on. -/
def protocolOn : Core.BlobProtocol where
  id := PROTOCOL ++ "/switch-on"
  Params := Params
  Commitment := Spot.OneStage.Commitment
  Evidence := Evidence
  Public := Spot.OneStage.Public
  Profile := Core.Integrity
  params := paramsOfJson
  commitment := Spot.OneStage.commitmentOf
  evidence := evidenceOf
  digest c := c.digest
  check p c e := checkBlobsAtWith true p c e.source e Core.Blobs.empty
  publicJson := Spot.OneStage.publicJson
  profileJson := Core.Integrity.toJson
  checkBlobs p c e bs := checkBlobsAtWith true p c e.source e bs

end VStar.Recursive.SampledTwo

def main (args : List String) : IO UInt32 := VStar.Recursive.SampledTwo.protocolOn.main args
