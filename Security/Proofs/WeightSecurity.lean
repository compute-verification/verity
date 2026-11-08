module

public import Proofs.WeightSecurity.Deployed
public import Proofs.WeightSecurity.Served
public import Proofs.WeightSecurity.Stop
public import Proofs.WeightSecurity.Theft

/-!
# Model weight security's proofs

`Theft`: the leakage lemma, `WeightSecurity.Guarantees.Theft`, with its witnesses at the toy instance. `Served`: the
leakage lemma at an audited deployment, `WeightSecurity.Guarantees.TheftServed`, with its witnesses at the toy deployment.
`Stop`: a period's escape at a budget, over SPOT's period machinery. `Deployed`: `TheftServed` at the deployment audits
(`TheftServedDeployed`).
-/
