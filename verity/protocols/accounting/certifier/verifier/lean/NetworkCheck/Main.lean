module

public import NetworkCheck.Check

/-! The `network-check` executable: `Core.Protocol.main` of `NetworkCheck.Check.protocol`, audited under `NetworkCheck`. -/

@[expose] public section

def main (args : List String) : IO UInt32 := NetworkCheck.Check.protocol.main args
