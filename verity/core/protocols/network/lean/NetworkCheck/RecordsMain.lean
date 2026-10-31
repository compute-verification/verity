import NetworkCheck.Records
import NetworkCheck.LinkWindow

/-! The `network-records` executable: `Core.Protocol.main` of `NetworkCheck.Records.protocol`, and `network-records verdicts`,
`Difftest.serve` of every link-window's verdict on a `serve` case, with the grid compared (`verdicts`) or not
(`verdicts-no-grid`), of N2's verdict of record on one link-window (`link-window`, `NetworkCheck.LinkWindow`), and of
the client's check of one window's served certificates (`certificates`, `NetworkCheck.LinkWindow.refusals`). Audited
under `NetworkCheck`. -/

def main (args : List String) : IO UInt32 := do
  match args with
  | ["verdicts"] =>
    Difftest.serve (Difftest.table [("verdicts", NetworkCheck.Records.verdictsFn),
      ("verdicts-no-grid", NetworkCheck.Records.verdictsNoGridFn),
      ("link-window", NetworkCheck.LinkWindow.linkWindowFn),
      ("certificates", NetworkCheck.LinkWindow.certificatesFn)])
    return 0
  | _ => NetworkCheck.Records.protocol.main args
