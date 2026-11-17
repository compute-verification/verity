module

public import Proofs.WeightSecurity.Faulted
public import Proofs.WeightSecurity.Served
public import Proofs.WeightSecurity.Theft
public import Proofs.WeightSecurity.Tied

/-!
# Model weight security's proofs

`Theft`: the leakage lemma, `WeightSecurity.Guarantees.Theft`, with its witnesses at the toy instance. `Served`: the
leakage lemma at an audited deployment, `WeightSecurity.Guarantees.TheftServed`, with its witnesses at the toy
deployment, and per charge unit in bits (`TheftServedReq`, `TheftServedMinEntropy`).
`Tied`: `TheftServedReq` at a checked population whose run records each window's trace (`TheftServedTied`), its bound
ending in the probabilities that a release isn't its trace's outputs or a trace doesn't hold its window's inputs.
`Faulted`: the leakage lemma with the audit's incorrect proof blocks as inputs
to a faulted circuit rather than overwrites of releases, `WeightSecurity.Guarantees.TheftFaulted`, with its witnesses at
the toy deployment, the releases' faithfulness from a checked dataflow, and the range clamp.
-/
