import PousCheck.RunProtocol

/-! `pous-run-protocol`: the run's core protocol command line (`Core.Protocol.main`: `id`, `check`, `serve`), under S1's
LMS; after `--test`, the same under the test signers' schemes. -/

def main (args : List String) : IO UInt32 :=
  match args with
  | "--test" :: rest => PousCheck.RunProtocol.testProtocol.main rest
  | _ => PousCheck.RunProtocol.protocol.main args
