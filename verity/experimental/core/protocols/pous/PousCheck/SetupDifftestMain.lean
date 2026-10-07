import PousCheck.SetupDifftest

/-! The setup difftest's run (`runs` in `lean-audit.json`): `PousCheck.SetupDifftest.main`, audited under `PousCheck`. It holds
the Python reference's P2 v4 maps, keys, commitment and public-encoder setup check to this package's, on the instances
`scripts/setup_difftest.py` generates and on the spec's §7 vectors. -/

def main (args : List String) : IO UInt32 := PousCheck.SetupDifftest.main args
