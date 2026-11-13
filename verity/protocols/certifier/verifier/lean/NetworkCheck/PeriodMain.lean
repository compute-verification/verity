module

public import NetworkCheck.Period

/-! The `network-period` executable: `Core.BlobProtocol.main` of `NetworkCheck.Period.protocol`, whose `check` takes the
scan's circuit files after the evidence; audited under `NetworkCheck`. -/

@[expose] public section

def main (args : List String) : IO UInt32 := NetworkCheck.Period.protocol.main args
