module

public import NetworkVerifier.Period

/-! The `network-period` executable: `Core.Protocol.main` of `NetworkVerifier.Period.protocol`, audited under `NetworkVerifier`. -/

@[expose] public section

def main (args : List String) : IO UInt32 := NetworkVerifier.Period.protocol.main args
