import NetworkCheck.Records

/-! The `network-records` executable: `Core.Protocol.main` of `NetworkCheck.Records.protocol`, and `network-records verdicts`,
`Difftest.serve` of its `verdicts`, every link-window's verdict on a `serve` case. Audited under `NetworkCheck`. -/

def main (args : List String) : IO UInt32 := do
  match args with
  | ["verdicts"] => Difftest.serve (Difftest.table [("verdicts", NetworkCheck.Records.verdictsFn)]); return 0
  | _ => NetworkCheck.Records.protocol.main args
