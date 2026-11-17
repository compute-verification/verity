module

public import NetworkVerifier.Span

/-! The `network-span` executable: `Core.Protocol.main` of `NetworkVerifier.Span.protocol`, audited under `NetworkVerifier`. -/

@[expose] public section

def main (args : List String) : IO UInt32 := NetworkVerifier.Span.protocol.main args
