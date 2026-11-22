module

public import Spot.MonitoringPeriodAudit

/-! The `spot-monitoring-period` executable: `Core.BlobProtocol.main` of `Spot.MonitoringPeriod.protocol`, audited under `Spot`. -/

@[expose] public section

def main (args : List String) : IO UInt32 := Spot.MonitoringPeriod.protocol.main args
