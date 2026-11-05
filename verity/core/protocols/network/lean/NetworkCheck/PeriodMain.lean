import NetworkCheck.PeriodFiles

/-! The `network-period` executable: `Core.Protocol.main` of `NetworkCheck.Period.protocol`, audited under `NetworkCheck`,
and `check PARAMS COMMITMENT EVIDENCE CIRCUIT…`, the period's check with its listed circuit files
(`NetworkCheck.Period.protocolWith`), read whole, the audits' trimmed, at `network-period`'s budget
(`NetworkCheck.PeriodFiles.read`, `NetworkCheck.PeriodFiles.Budget.period`). A file that can't be read rejects the
period, naming the first such file by its place in the listing. The test-only `network-period-chunked`
(`NetworkCheck.PeriodChunkedMain`) reads the audits' files in chunks instead. -/

def main (args : List String) : IO UInt32 := do
  match args with
  | "check" :: p :: c :: e :: files@(_ :: _) =>
    match ← NetworkCheck.PeriodFiles.read .period (← NetworkCheck.Period.paramsAt p) files with
    | .ok fs => (NetworkCheck.Period.protocolWith fs).main ["check", p, c, e]
    | .error why => (NetworkCheck.Period.refusing s!"circuit files: {why}").main ["check", p, c, e]
  | _ => NetworkCheck.Period.protocol.main args
