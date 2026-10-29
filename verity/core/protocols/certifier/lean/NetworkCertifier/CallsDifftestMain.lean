import NetworkCertifier.CallsDifftest

/-! The calls' difftest's run (`runs` in `lean-audit.json`): `NetworkCertifier.Calls.CallsDifftest.main`, audited under
`NetworkCertifier`.

It is the one test that holds `calls.Calls`, the network certifier's release of the lead's coins and its per-call
count, to the certifier program's `NetworkCertifier.Calls`. -/

def main (args : List String) : IO UInt32 := NetworkCertifier.Calls.CallsDifftest.main args
