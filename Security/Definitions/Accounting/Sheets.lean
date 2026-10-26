import Definitions.Accounting.Sheets.H200
import Definitions.Accounting.Sheets.B200

/-!
# The data sheets

`Accounting.TS.Sheet` at each device's vendor values: the H200 SXM (sm_90a, `Sheets.H200`) and the HGX B200 (sm_100a,
`Sheets.B200`); sm_120's `DS-120` has no Lean sheet yet. Each module is generated from its device's `sheet.toml` in
`benchmarks/pouw/machine_model/data/devices/` and proves its machine is the machine model's DS census of that device.
-/
