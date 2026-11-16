module

public import PousCheck.SetupProtocol

/-! `pous-setup`: the setup protocol's command line (`Core.Protocol.main`: `id`, `check`, `serve`). -/

@[expose] public section

def main (args : List String) : IO UInt32 := PousCheck.SetupProtocol.protocol.main args
