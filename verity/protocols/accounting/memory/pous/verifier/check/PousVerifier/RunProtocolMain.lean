module

public import PousVerifier.RunProtocol

/-! `pous-run-protocol`: the run's core protocol command line (`Core.Protocol.main`: `id`, `check`, `serve`), under S1's
LMS; after `--test`, the same under the test signers' schemes. -/

@[expose] public section

def main (args : List String) : IO UInt32 :=
  match args with
  | "--test" :: rest => PousVerifier.RunProtocol.testProtocol.main rest
  | _ => PousVerifier.RunProtocol.protocol.main args
