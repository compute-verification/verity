module

public import NetworkVerifier.Check

/-! The `network-verifier` executable: `Core.Protocol.main` of `NetworkVerifier.Check.protocol`, audited under `NetworkVerifier`. -/

@[expose] public section

def main (args : List String) : IO UInt32 := NetworkVerifier.Check.protocol.main args
