import CoreProtocol.Cli
import NciCheck.Protocol

/-! The `nci` executable: `Core.Protocol.main` of `NciCheck.Protocol.protocol` (`verity/nci/v1`), the core CLI that
onsite's difftest and the NCI protocol's own drive. -/

def main (args : List String) : IO UInt32 := NciCheck.Protocol.protocol.main args
