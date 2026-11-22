module

public import PousVerifier.SetupProtocol

/-! `pous-setup`: the setup protocol's command line (`Core.Protocol.main`: `id`, `check`, `serve`). -/

@[expose] public section

def main (args : List String) : IO UInt32 := PousVerifier.SetupProtocol.protocol.main args
