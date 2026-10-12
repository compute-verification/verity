import NetworkCheck.Check

/-! The `network-check` executable: `Core.Protocol.main` of `NetworkCheck.Check.protocol`, audited under `NetworkCheck`. -/

def main (args : List String) : IO UInt32 := NetworkCheck.Check.protocol.main args
