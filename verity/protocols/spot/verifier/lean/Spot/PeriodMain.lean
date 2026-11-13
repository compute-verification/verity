module

public import Spot.PeriodAudit

/-! The `sampled-proofs-period` executable: `Core.BlobProtocol.main` of `Spot.Period.protocol`, audited under `Spot`. -/

@[expose] public section

def main (args : List String) : IO UInt32 := Spot.Period.protocol.main args
