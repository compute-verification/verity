module

public import Proofs.WeightSecurity.Cut
public import Proofs.WeightSecurity.Deployed
public import Proofs.WeightSecurity.Faulted
public import Proofs.WeightSecurity.Pressure
public import Proofs.WeightSecurity.Served
public import Proofs.WeightSecurity.Stop
public import Proofs.WeightSecurity.Theft
public import Proofs.WeightSecurity.Tied
public import Proofs.WeightSecurity.Two

/-!
# Model weight security's proofs

`Theft`: the leakage lemma, `WeightSecurity.Guarantees.Theft`, with its witnesses at the toy instance. `Served`: the
leakage lemma at an audited deployment, `WeightSecurity.Guarantees.TheftServed`, with its witnesses at the toy
deployment, and per charge unit in bits (`TheftServedReq`, `TheftServedMinEntropy`). `Stop`: the incorrect proof
blocks a period leaves undetected, over SPOT's period machinery. `Deployed`: `TheftServed` at the deployment audits
(`TheftServedDeployed`). `Pressure`: a registered window's `pa B ≤ x^|B|`, the pressure `TheftServedDeployed` takes.
`Two`: `TheftServed` at a two-stage audit (`TheftServedTwo`), its budget `T·(K − 1) + 1` from the cap `WindowCap`.
`Tied`: `TheftServedReq` at a checked population whose run records each window's trace (`TheftServedTied`), its bound
ending in the probabilities that a release isn't its trace's outputs or a trace doesn't hold its window's inputs.
`Faulted`: the leakage lemma with the audit's incorrect proof blocks as inputs
to a faulted circuit rather than overwrites of releases, `WeightSecurity.Guarantees.TheftFaulted`, with its witnesses at
the toy deployment, the releases' faithfulness from a checked dataflow, and the range clamp. `Cut`: `Theft` at an honest
cut's channel (`theft_cut`), faulted use's hardness from honest use's (`faultedUseHard_of_cut`), and the cuts of
autoregressive decoding: the token gate for position-local blocks, a one-token request's token, and the release.
-/
