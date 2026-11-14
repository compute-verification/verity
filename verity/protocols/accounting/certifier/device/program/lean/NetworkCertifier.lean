module

public import NetworkCertifier.Grid
public import NetworkCertifier.Run
public import NetworkCertifier.Ingress
public import NetworkCertifier.Schedule
public import NetworkCertifier.Difftest
public import NetworkCertifier.AdviceV1
public import NetworkCertifier.ChargeDifftest
public import NetworkCertifier.WindowCharge
public import NetworkCertifier.WindowDifftest
public import NetworkCertifier.Rows

/-!
# The network certifier's program

The executable forms of the reference definitions in `Security/Definitions/NetworkCertifier/` that the difftest runs
(`NetworkCertifier`, advice v1's charge, `NetworkCertifier.AdviceV1`, and its whole window's,
`NetworkCertifier.WindowCharge`), and the difftests themselves (`NetworkCertifier.Difftest`,
`NetworkCertifier.AdviceV1.ChargeDifftest`, `NetworkCertifier.WindowCharge.WindowDifftest`). Besides, the audit of the
active network certifier's records as rows (`NetworkCertifier.Rows`, the executable form of
`Definitions/NetworkCertifier/RecordRows.lean`), which `network_check`'s `network-rows` runs over a run's certified
records. Code: it imports only Lean, and never `Security/`; `security_proofs` requires it and proves the agreement
(`Proofs.NetworkCertifier.Code`).
-/

@[expose] public section
