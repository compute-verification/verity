import Proofs.Properties.NetworkScan.Honest.Loop

/-!
# The honest instance's egress file reads no padding

`readsEgress`: Bits' reads check (`SampledProofs.Bits.readsWhy`) finds nothing wrong with the unit's read input
columns of `fEgress` against `EgressBucket_v1` at `row`, which it has no parameter of, by `cbv`, which builds the
unit's rows from its sparse matrices (`SampledProofs.Bits.unitReads`). A module of its own, so that it builds beside
`ReadsIngress`.
-/

namespace Properties.NetworkScan.Honest

attribute [local cbv_eval] loopId Check.bits_bitsOf

-- `cbv` decodes the unit's sparse matrices byte by byte, many times the default budget
set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8192 in
set_option cbv.maxSteps 10000000 in
theorem readsEgress :
    SampledProofs.Bits.readsWhy fEgress egress (SampledProofs.Bits.unitReads fEgress) ["row"] = none := by
  cbv

end Properties.NetworkScan.Honest
