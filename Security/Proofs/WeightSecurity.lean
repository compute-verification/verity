module

public import Proofs.WeightSecurity.Deployed
public import Proofs.WeightSecurity.Faulted
public import Proofs.WeightSecurity.Pressure
public import Proofs.WeightSecurity.Served
public import Proofs.WeightSecurity.Stop
public import Proofs.WeightSecurity.Theft

/-!
# Model weight security's proofs

`Theft`: the leakage lemma, `WeightSecurity.Guarantees.Theft`, with its witnesses at the toy instance. `Served`: the
leakage lemma at an audited deployment, `WeightSecurity.Guarantees.TheftServed`, with its witnesses at the toy
deployment. `Stop`: the incorrect proof blocks a period leaves undetected, over SPOT's period machinery. `Deployed`:
`TheftServed` at the deployment audits (`TheftServedDeployed`). `Pressure`: a registered window's `pa B ≤ x^|B|`, the
pressure `TheftServedDeployed` takes. `Faulted`: the leakage lemma with the audit's incorrect proof blocks as inputs to
a faulted circuit rather than overwrites of releases, `WeightSecurity.Guarantees.TheftFaulted`, with its witnesses at the
toy deployment, the releases' faithfulness from a checked dataflow, and the range clamp.
-/
