module

public import CoreProtocol.Cli
public import InferenceOnlyVerifier.RecordProtocol

/-! The `inference-only-record` executable: `Core.Protocol.main` of `InferenceOnly.RecordProtocol.protocol`
(`verity/inference-only/record/v1`), the core CLI that the protocol's own difftest drives. -/

@[expose] public section

def main (args : List String) : IO UInt32 := InferenceOnly.RecordProtocol.protocol.main args
