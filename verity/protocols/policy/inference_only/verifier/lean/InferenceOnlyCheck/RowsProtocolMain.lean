module

public import CoreProtocol.Cli
public import NciCheck.RowsProtocol

/-! The `inference-only-rows` executable: `Core.BlobProtocol.main` of `InferenceOnly.RowsProtocol.protocol`
(`verity/inference-only/rows/v1`), the core CLI that the protocol's own difftest drives. Its `check` takes the
circuit files after the evidence, and a `serve` case carries them as `blobs`. -/

@[expose] public section

def main (args : List String) : IO UInt32 := InferenceOnly.RowsProtocol.protocol.main args
