import NetworkCheck.Records
import NetworkCheck.LinkWindow
import NetworkCheck.Period

/-! The `network-records` executable: `Core.Protocol.main` of `NetworkCheck.Records.protocol`, and `network-records verdicts`,
`Difftest.serve` of every link-window's verdict on a `serve` case, with the grid compared (`verdicts`) or not
(`verdicts-no-grid`), of N2's verdict of record on one link-window (`link-window`, `NetworkCheck.LinkWindow`), of
the client's check of one window's served certificates (`certificates`, `NetworkCheck.LinkWindow.refusals`), of the
syncs the certifier signed for a link, read through its clock certificates' openings (`clocks`,
`NetworkCheck.Period.signedSyncs`), and of a compute audit's circuit record checked against its Program (`circuit`,
`NetworkCheck.Circuit.check`). Audited under `NetworkCheck`. -/

def main (args : List String) : IO UInt32 := do
  match args with
  | ["verdicts"] =>
    Difftest.serve (Difftest.table [("verdicts", NetworkCheck.Records.verdictsFn),
      ("verdicts-no-grid", NetworkCheck.Records.verdictsNoGridFn),
      ("link-window", NetworkCheck.LinkWindow.linkWindowFn),
      ("certificates", NetworkCheck.LinkWindow.certificatesFn),
      ("clocks", NetworkCheck.Period.clocksFn),
      ("circuit", NetworkCheck.Circuit.circuitFn)])
    return 0
  | _ => NetworkCheck.Records.protocol.main args
