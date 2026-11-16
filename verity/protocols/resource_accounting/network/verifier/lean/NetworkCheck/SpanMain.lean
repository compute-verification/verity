module

public import NetworkCheck.Span

/-! The `network-span` executable: `Core.Protocol.main` of `NetworkCheck.Span.protocol`, audited under `NetworkCheck`. -/

@[expose] public section

def main (args : List String) : IO UInt32 := NetworkCheck.Span.protocol.main args
