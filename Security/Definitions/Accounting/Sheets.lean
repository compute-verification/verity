import Definitions.Accounting.Sheets.B200

/-!
# The data sheets

`Accounting.TS.Sheet` at a device's vendor values where it has no card file: the HGX B200 (sm_100a, `Sheets.B200`),
generated from its `sheet.toml` in `benchmarks/pouw/machine_model/data/devices/`, which it proves its machine is the
machine model's DS census of. A device with a card file has its card table instead (`Verity.Accel.H200.card`,
`Verity.Accel.Sm120.card`), and its headline's DS encoding is `Accounting.TS.Headline`'s.
-/
