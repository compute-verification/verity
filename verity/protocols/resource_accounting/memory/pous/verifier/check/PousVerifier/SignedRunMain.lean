module

public import PousVerifier.SignedRun

/-! `pous-signed-run`: the signed run's core protocol command line (`Core.Protocol.main`: `id`, `check`, `serve`), under
S1's LMS; after `--test`, the same under the test signers' schemes. -/

@[expose] public section

def main (args : List String) : IO UInt32 :=
  match args with
  | "--test" :: rest => PousVerifier.SignedRun.testProtocol.main rest
  | _ => PousVerifier.SignedRun.protocol.main args
