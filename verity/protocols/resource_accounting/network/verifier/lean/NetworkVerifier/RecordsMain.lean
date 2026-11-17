module

public import Difftest
public import NetworkVerifier.TransferBlocks
public import NetworkVerifier.Records
import Flock.Sizes
import Std.Data.HashMap
import Std.Data.HashSet
import Std.Data.TreeMap.Raw.Lemmas

/-! The `network-records` executable: `Core.Protocol.main` of `NetworkVerifier.Records.protocol`, and `network-records verdicts`,
`Difftest.serve` of every execution phase's verdict on a `serve` case, with the grid compared (`verdicts`) or not
(`verdicts-no-grid`), of N2's verdict of record on one execution phase (`link-window`, `NetworkVerifier.TransferBlocks`), and of
the client's check of one window's served certificates (`certificates`, `NetworkVerifier.TransferBlocks.refusals`). Audited
under `NetworkVerifier`. -/

@[expose] public section

def main (args : List String) : IO UInt32 := do
  match args with
  | ["verdicts"] =>
    Difftest.serve (Difftest.table [("verdicts", NetworkVerifier.Records.verdictsFn),
      ("verdicts-no-grid", NetworkVerifier.Records.verdictsNoGridFn),
      ("link-window", NetworkVerifier.TransferBlocks.transferBlocksFn),
      ("certificates", NetworkVerifier.TransferBlocks.certificatesFn)])
    return 0
  | _ => NetworkVerifier.Records.protocol.main args
