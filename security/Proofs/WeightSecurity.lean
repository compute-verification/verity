module

public import Proofs.WeightSecurity.Cut
public import Proofs.WeightSecurity.Deployed
public import Proofs.WeightSecurity.Faulted
public import Proofs.WeightSecurity.FaultedTwo
public import Proofs.WeightSecurity.Pressure
public import Proofs.WeightSecurity.Reg
public import Proofs.WeightSecurity.RegPeriod
public import Proofs.WeightSecurity.RegWitnesses
public import Proofs.WeightSecurity.Served
public import Proofs.WeightSecurity.Stop
public import Proofs.WeightSecurity.Theft
public import Proofs.WeightSecurity.Tied
public import Proofs.WeightSecurity.Two
public import Proofs.WeightSecurity.TwoProd
public import Proofs.WeightSecurity.TwoProdPeriod
public import Proofs.WeightSecurity.TwoProdWitnesses
public import Proofs.WeightSecurity.TwoReg
public import Proofs.WeightSecurity.TwoRegPeriod
public import Proofs.WeightSecurity.TwoRegWitnesses

/-!
# Model weight security's proofs

`Reg`: a period of one-stage windows at the committed traces, the statement `PeriodReg` in `period_undetected`'s form.
`RegPeriod`: `PeriodReg` from each window's committed form and miss bound (`periodReg_of`), so **`CompilerPeriodReg` is
proved** (`compilerPeriodReg`). `RegWitnesses`: its premises hold together at `twoL` (`periodReg_honest`), and
`PeriodReg` fails at `sksL` (`periodReg_nontrivial`).

`Theft`: the leakage lemma, `WeightSecurity.Guarantees.Theft`, with its witnesses at the toy instance. `Served`: the
leakage lemma at an audited deployment, `WeightSecurity.Guarantees.TheftServed`, with its witnesses at the toy
deployment, and per charge unit in bits (`TheftServedReq`, `TheftServedMinEntropy`). `Stop`: the incorrect proof
blocks a period leaves undetected, over SPOT's period machinery. `Deployed`: `TheftServed` at the deployment audits
(`TheftServedDeployed`). `Pressure`: a registered window's `pa B ≤ x^|B|`, the pressure `TheftServedDeployed` takes.
`Two`: `TheftServed` at a two-stage audit (`TheftServedTwo`), its budget `T·(K − 1) + 1` from the cap `WindowCap`.
`Tied`: `TheftServedReq` at a checked population whose run records each window's trace (`TheftServedTied`), its bound
ending in the probabilities that a release isn't its trace's outputs or a trace doesn't hold its window's inputs.
`TwoReg`: a period of two-stage windows at the committed traces, the
statement `PeriodTwoReg` in `period_undetected`'s form. `TwoRegPeriod`: `PeriodTwoReg` from each window's committed
form and miss bound (`periodTwoReg_of`), so **`CompilerPeriodTwoReg` is proved** (`compilerPeriodTwoReg`).
`TwoRegWitnesses`: its premises hold together at `twoLTwo` (`periodTwoReg_honest`), and `PeriodTwoReg` fails at
`sksLTwo` (`periodTwoReg_nontrivial`).
`TwoProd`: a period of two-stage verification phases at boundary traces its caller names, with one failure probability
`δ` for the whole period: the statements `PeriodTwoLead` and `PeriodTwoProd`, in which the phases' escapes multiply.
`TwoProdPeriod`: both from each verifier's two-stage contract (`compilerPeriodTwoProd`), and both at the committed
boundary trace under `CompilerPeriodTwoReg`'s premises (`periodTwoLead_regVal`, `periodTwoProd_regVal`).
`TwoProdWitnesses`: those premises hold together at `twoLTwo` (`periodTwoProd_honest`), and both statements fail at
`sksLTwo` (`periodTwoLead_nontrivial`, `periodTwoProd_nontrivial`).
`Faulted`: the leakage lemma with the audit's incorrect proof blocks as inputs
to a faulted circuit rather than overwrites of releases, `WeightSecurity.Guarantees.TheftFaulted`, with its witnesses at
the toy deployment, the releases' faithfulness from a checked dataflow, and the range clamp. `FaultedTwo`:
`TheftFaulted` at a two-stage audit (`TheftFaultedTwo`), its faults incorrect replay blocks within the same budget.
`Cut`: `Theft` at an honest cut's channel (`theft_cut`), faulted use's hardness from honest use's
(`faultedUseHard_of_cut`), and the cuts of autoregressive decoding: the token gate for position-local blocks, a
one-token request's token, and the release.
-/
