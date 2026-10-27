import NetworkCertifier.ChargeDifftest

/-! The charge difftest's run (`runs` in `lean-audit.json`): `NetworkCertifier.AdviceV1.ChargeDifftest.main`, audited
under `NetworkCertifier`.

It is the one test that holds `arith_encode.reference_charge`, which circuit-check holds `ChargeCount_v1{N}`'s circuit to,
to the certifier program's `AdviceV1.charges`, which `Proofs.NetworkCertifier.Code` (`CodeChargesV1`) proves equal to the
reference definition `BitsCount` for advice v1 is proved of. -/

def main (args : List String) : IO UInt32 := NetworkCertifier.AdviceV1.ChargeDifftest.main args
