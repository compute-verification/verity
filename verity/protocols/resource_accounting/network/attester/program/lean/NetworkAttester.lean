module

public import NetworkAttester.Grid
public import NetworkAttester.Run
public import NetworkAttester.Ingress
public import NetworkAttester.Schedule
public import NetworkAttester.Difftest
public import NetworkAttester.AdviceV1
public import NetworkAttester.ChargeDifftest
public import NetworkAttester.WindowCharge
public import NetworkAttester.WindowDifftest
public import NetworkAttester.Rows

/-!
# The network attester's program

The executable forms of the reference definitions in `security/Definitions/Certifier/` that the difftest runs
(`NetworkAttester`, advice v1's charge, `NetworkAttester.AdviceV1`, and its whole window's,
`NetworkAttester.WindowCharge`), and the difftests themselves (`NetworkAttester.Difftest`,
`NetworkAttester.AdviceV1.ChargeDifftest`, `NetworkAttester.WindowCharge.WindowDifftest`). Besides, the audit of the
active network attester's records as rows (`NetworkAttester.Rows`, the executable form of
`Definitions/Certifier/RecordRows.lean`), which `network_check`'s `network-rows` runs over a run's certified
records. Code: it imports only Lean, and never `security/`; `security_proofs` requires it and proves the agreement
(`Proofs.Certifier.Code`).
-/

@[expose] public section
