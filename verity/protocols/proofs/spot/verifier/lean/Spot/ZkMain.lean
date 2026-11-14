module

public import Spot.MembersParse
public import Spot.OneStageZkStream

/-!
# The one-stage v2 executable's entry point (`sampled-proofs-one-stage-zk`)

`membersMain`'s commands, and the streamed check's two, `plan PARAMS COMMITMENT EVIDENCE` and
`check-stream PARAMS COMMITMENT EVIDENCE LIST` (`Core.StreamProtocol.main` at `Spot.OneStageZk.Streamed.protocol`).
`check-stream` reads the blobs LIST names one at a time, in `plan`'s order, and answers as `check` does. Its fold is
`Core.Stream.run`'s, whose accept is `checkBlobs`' accept on the blobs fed (`Core.StreamProtocol.run_accept` at
`Spot.OneStageZk.Streamed.streamSound`); the loop that reads the files (`Core.Stream.files`) is trusted code, as
`check`'s reading of its blobs is. Only those two argument lists, by their command and their count, go to the stream's
main; every other goes to `membersMain`.
-/

@[expose] public section

def main (args : List String) : IO UInt32 :=
  match args with
  | ["plan", _, _, _] | ["check-stream", _, _, _, _] => Spot.OneStageZk.Streamed.protocol.main args
  | _ => Spot.OneStageZk.membersMain args
