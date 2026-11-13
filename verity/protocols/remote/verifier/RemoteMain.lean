module

public import CoreProtocol.Cli
public import CoreRemote

@[expose] public section

/-- Remote's development composite, H1's `Core.Remote.protocol` (`verity/remote/v0`): development only, no
composite a configuration can name (`Core.Composites` refuses its subprotocols) and no soundness theorem's verifier. -/
def main (args : List String) : IO UInt32 := Core.Remote.protocol.main args
