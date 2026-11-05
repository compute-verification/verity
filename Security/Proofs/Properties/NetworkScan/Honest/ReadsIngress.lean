import Proofs.Properties.NetworkScan.Honest.Defs

/-!
# The honest instance's ingress file reads its row

`readsIngress`: Bits' reads check (`SampledProofs.Bits.readsWhy`) finds nothing wrong with the unit's read input
columns of `fIngress` against `IngressScanFirst_v1` at `row`, by `cbv`, which builds the unit's rows from its sparse
matrices (`SampledProofs.Bits.unitReads`). A module of its own, so that it builds beside `ReadsEgress`.
-/

namespace Properties.NetworkScan.Honest

attribute [local cbv_eval] loopId Check.bits_bitsOf

-- `cbv` decodes the unit's sparse matrices byte by byte, many times the default budget
set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8192 in
set_option cbv.maxSteps 10000000 in
theorem readsIngress :
    SampledProofs.Bits.readsWhy fIngress ingress (SampledProofs.Bits.unitReads fIngress) ["row"] = none := by
  cbv

end Properties.NetworkScan.Honest
