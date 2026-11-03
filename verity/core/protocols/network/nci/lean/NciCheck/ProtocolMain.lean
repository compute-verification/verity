import CoreProtocol.Cli
import NciCheck.Protocol

/-! The `inference-only` executable: `Core.Protocol.main` of `NciCheck.Protocol.protocol`
(`verity/inference-only/v0`), the core CLI that the protocol's own difftest drives. -/

def main (args : List String) : IO UInt32 := NciCheck.Protocol.protocol.main args
