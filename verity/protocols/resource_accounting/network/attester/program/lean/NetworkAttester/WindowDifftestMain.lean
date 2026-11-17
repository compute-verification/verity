module

public import NetworkAttester.WindowDifftest
meta import NetworkAttester.WindowDifftest

/-! The window charge's difftest's run (`runs` in `lean-audit.json`):
`NetworkAttester.WindowCharge.WindowDifftest.main`, audited under `NetworkAttester`.

It is the one test that holds `window_charge.reference_window`, which circuit-check holds `ChargeCount_v2{N, H, O}`'s
circuit to, to network attestation's program's `WindowCharge.charges`, which `Proofs.Certifier.Code` (`CodeChargesWhole`)
proves equal to the reference definition (`security/Definitions/Certifier/WindowCharge.lean`). -/

@[expose] public section

def main (args : List String) : IO UInt32 := NetworkAttester.WindowCharge.WindowDifftest.main args
