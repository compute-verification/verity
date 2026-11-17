module

public import CoreProtocol.Cli
public import InferenceOnlyVerifier.Protocol

/-! The `inference-only` executable: `Core.Protocol.main` of `InferenceOnlyCheck.Protocol.protocol`
(`verity/inference-only/v0`), the core CLI that the protocol's own difftest drives. -/

@[expose] public section

def main (args : List String) : IO UInt32 := InferenceOnlyVerifier.Protocol.protocol.main args
