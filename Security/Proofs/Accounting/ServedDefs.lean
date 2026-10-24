import Definitions.Accounting.Served

/-! The proof-only part of `Definitions.Accounting.Served`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Accounting.Served

open Accounting.Compose

/-- **The served `y` is the proved product's**, by form, where `fused` is the fused kernel's premise (the `y` it serves
is `dequant` of the `z` rows it commits): `beside` has no link (`False`); `zTree` has it exactly under `fused`, a
premise on code the prover runs, which no run-time check reaches; `yOpen` has it from `Tie.linked` and the receipt
Program's `dequant(z) = y` check over the opened `y` (a fact of that Program's Definition, which doesn't exist yet),
with no premise, given the run-time check that the unit's `served` words open under the Commit's `y` root: `Tie`
doesn't tell a `served` run input from an `act` one. `perCall` has it from `Tie.linked` and the unit's own
`dequant(z)` output, given the run-time check that the call's Commit `y` root is its units' `y` root. Under
`ValueChecked` a `zTree` unit commits its `y` too, a word it computes or its record's, which its `check` words hold to
`dq` of its product. Its link is then that run-time check, and `zTree` needs no premise on the kernel: `fused` is the
link of a `zTree` window that isn't value-checked, and `CreditTie` doesn't read `ServedLink`. It is an equality of
values: useful-work credit needs it, and it doesn't give that credit (`ValueChecked`, the module docstring). Integrity
reads none of it. -/
def Form.ServedLink (fused : Prop) : Form → Prop
  | .beside => False
  | .zTree => fused
  | .yOpen => True
  | .perCall => True

end Accounting.Served
