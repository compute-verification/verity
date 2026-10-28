import NetworkCertifier.WindowDifftest

/-! The window charge's difftest's run (`runs` in `lean-audit.json`):
`NetworkCertifier.WindowCharge.WindowDifftest.main`, audited under `NetworkCertifier`.

It is the one test that holds `window_charge.reference_window`, which circuit-check holds `ChargeCount_v2{N, H, O}`'s
circuit to, to the certifier program's `WindowCharge.charges`, which `Proofs.NetworkCertifier.Code` (`CodeChargesWhole`)
proves equal to the reference definition `EncardChargedLeWhole` is proved of. -/

def main (args : List String) : IO UInt32 := NetworkCertifier.WindowCharge.WindowDifftest.main args
