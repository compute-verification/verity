module

public import NetworkCheck.Period

/-! The `network-period` executable: `Core.Protocol.main` of `NetworkCheck.Period.protocol`, audited under `NetworkCheck`. -/

@[expose] public section

def main (args : List String) : IO UInt32 := NetworkCheck.Period.protocol.main args
