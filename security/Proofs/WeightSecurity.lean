module

public import Proofs.WeightSecurity.Cut
public import Proofs.WeightSecurity.Deployed
public import Proofs.WeightSecurity.Faulted
public import Proofs.WeightSecurity.FaultedTwo
public import Proofs.WeightSecurity.Pressure
public import Proofs.WeightSecurity.Reg
public import Proofs.WeightSecurity.RegMonitoringPeriod
public import Proofs.WeightSecurity.RegWitnesses
public import Proofs.WeightSecurity.Served
public import Proofs.WeightSecurity.Stop
public import Proofs.WeightSecurity.Theft
public import Proofs.WeightSecurity.Tied
public import Proofs.WeightSecurity.Two
public import Proofs.WeightSecurity.TwoProd
public import Proofs.WeightSecurity.TwoProdMonitoringPeriod
public import Proofs.WeightSecurity.TwoProdWitnesses
public import Proofs.WeightSecurity.TwoReg
public import Proofs.WeightSecurity.TwoRegMonitoringPeriod
public import Proofs.WeightSecurity.TwoRegWitnesses

/-!
# Model weight security's proofs

`Reg`: a period of one-stage windows at the committed traces, the statement `MonitoringPeriodReg` in `period_undetected`'s form.
`RegMonitoringPeriod`: `MonitoringPeriodReg` from each window's committed form and miss bound (`monitoringPeriodReg_of`), so **`CompilerMonitoringPeriodReg` is
proved** (`compilerMonitoringPeriodReg`). `RegWitnesses`: its premises hold together at `twoL` (`monitoringPeriodReg_honest`), and
`MonitoringPeriodReg` fails at `sksL` (`monitoringPeriodReg_nontrivial`).

`Theft`: the leakage lemma, `WeightSecurity.Guarantees.Theft`, with its witnesses at the toy instance. `Served`: the
leakage lemma at an audited deployment, `WeightSecurity.Guarantees.TheftServed`, with its witnesses at the toy
deployment, and per charge unit in bits (`TheftServedReq`, `TheftServedMinEntropy`). `Stop`: the incorrect proof
blocks a period leaves undetected, over SPOT's period machinery. `Deployed`: `TheftServed` at the deployment audits
(`TheftServedDeployed`). `Pressure`: a registered window's `pa B ≤ x^|B|`, the pressure `TheftServedDeployed` takes.
`Two`: `TheftServed` at a two-stage audit (`TheftServedTwo`), its budget `T·(K − 1) + 1` from the cap `ExecutionPhaseCap`.
`Tied`: `TheftServedReq` at a checked population whose run records each window's trace (`TheftServedTied`), its bound
ending in the probabilities that a release isn't its trace's outputs or a trace doesn't hold its window's inputs.
`TwoReg`: a period of two-stage windows at the committed traces, the
statement `MonitoringPeriodTwoReg` in `period_undetected`'s form. `TwoRegMonitoringPeriod`: `MonitoringPeriodTwoReg` from each window's committed
form and miss bound (`monitoringPeriodTwoReg_of`), so **`CompilerMonitoringPeriodTwoReg` is proved** (`compilerMonitoringPeriodTwoReg`).
`TwoRegWitnesses`: its premises hold together at `twoLTwo` (`monitoringPeriodTwoReg_honest`), and `MonitoringPeriodTwoReg` fails at
`sksLTwo` (`monitoringPeriodTwoReg_nontrivial`).
`TwoProd`: a period of two-stage verification phases at boundary traces its caller names, with one failure probability
`δ` for the whole period: the statements `MonitoringPeriodTwoLead` and `MonitoringPeriodTwoProd`, in which the phases' escapes multiply.
`TwoProdMonitoringPeriod`: both from each verifier's two-stage contract (`compilerMonitoringPeriodTwoProd`), and both at the committed
boundary trace under `CompilerMonitoringPeriodTwoReg`'s premises (`monitoringPeriodTwoLead_regVal`, `monitoringPeriodTwoProd_regVal`).
`TwoProdWitnesses`: those premises hold together at `twoLTwo` (`monitoringPeriodTwoProd_honest`), and both statements fail at
`sksLTwo` (`monitoringPeriodTwoLead_nontrivial`, `monitoringPeriodTwoProd_nontrivial`).
`Faulted`: the leakage lemma with the audit's incorrect proof blocks as inputs
to a faulted circuit rather than overwrites of releases, `WeightSecurity.Guarantees.TheftFaulted`, with its witnesses at
the toy deployment, the releases' faithfulness from a checked dataflow, and the range clamp. `FaultedTwo`:
`TheftFaulted` at a two-stage audit (`TheftFaultedTwo`), its faults incorrect replay blocks within the same budget.
`Cut`: `Theft` at an honest cut's channel (`theft_cut`), faulted use's hardness from honest use's
(`faultedUseHard_of_cut`), and the cuts of autoregressive decoding: the token gate for position-local blocks, a
one-token request's token, and the release.
-/
