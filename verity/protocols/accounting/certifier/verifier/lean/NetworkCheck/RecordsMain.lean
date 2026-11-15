module

public import Difftest
public import NetworkCheck.TransferBlocks
public import NetworkCheck.Records
import Flock.Sizes
import Std.Data.HashMap
import Std.Data.HashSet
import Std.Data.TreeMap.Raw.Lemmas

/-! The `network-records` executable: `Core.Protocol.main` of `NetworkCheck.Records.protocol`, and `network-records verdicts`,
`Difftest.serve` of every execution phase's verdict on a `serve` case, with the grid compared (`verdicts`) or not
(`verdicts-no-grid`), of N2's verdict of record on one execution phase (`link-window`, `NetworkCheck.TransferBlocks`), and of
the client's check of one window's served certificates (`certificates`, `NetworkCheck.TransferBlocks.refusals`). Audited
under `NetworkCheck`. -/

@[expose] public section

def main (args : List String) : IO UInt32 := do
  match args with
  | ["verdicts"] =>
    Difftest.serve (Difftest.table [("verdicts", NetworkCheck.Records.verdictsFn),
      ("verdicts-no-grid", NetworkCheck.Records.verdictsNoGridFn),
      ("link-window", NetworkCheck.TransferBlocks.transferBlocksFn),
      ("certificates", NetworkCheck.TransferBlocks.certificatesFn)])
    return 0
  | _ => NetworkCheck.Records.protocol.main args
