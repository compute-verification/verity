module

public import NetworkAttester.ChargeDifftest
meta import NetworkAttester.ChargeDifftest

/-! The charge difftest's run (`runs` in `lean-audit.json`): `NetworkAttester.AdviceV1.ChargeDifftest.main`, audited
under `NetworkAttester`.

It is the one test that holds `arith_encode.reference_charge`, which circuit-check holds `ChargeCount_v1{N}`'s circuit to,
to the certifier program's `AdviceV1.charges`, which `Proofs.Certifier.Code` (`CodeChargesV1`) proves equal to the
reference definition (`security/Definitions/Certifier/AdviceV1.lean`). -/

@[expose] public section

def main (args : List String) : IO UInt32 := NetworkAttester.AdviceV1.ChargeDifftest.main args
