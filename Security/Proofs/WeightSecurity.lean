module

public import Proofs.WeightSecurity.Deployed
public import Proofs.WeightSecurity.Pressure
public import Proofs.WeightSecurity.Reg
public import Proofs.WeightSecurity.RegPeriod
public import Proofs.WeightSecurity.RegWitnesses
public import Proofs.WeightSecurity.Served
public import Proofs.WeightSecurity.Stop
public import Proofs.WeightSecurity.Theft
public import Proofs.WeightSecurity.Two

/-!
# Model weight security's proofs

`Reg`: a period of one-stage windows at the committed traces, the statement `PeriodReg` in `period_undetected`'s form.
`RegPeriod`: `PeriodReg` from each window's committed form and draw law (`periodReg_of`), so **`CompilerPeriodReg` is
proved** (`compilerPeriodReg`). `RegWitnesses`: its premises hold together at `twoL` (`periodReg_honest`), and
`PeriodReg` fails at `sksL` (`periodReg_nontrivial`).

`Theft`: the leakage lemma, `WeightSecurity.Guarantees.Theft`, with its witnesses at the toy instance. `Served`: the
leakage lemma at an audited deployment, `WeightSecurity.Guarantees.TheftServed`, with its witnesses at the toy
deployment. `Stop`: the incorrect proof blocks a period leaves undetected, over SPOT's period machinery. `Deployed`:
`TheftServed` at the deployment audits (`TheftServedDeployed`). `Pressure`: a registered window's `pa B ≤ x^|B|`, the
pressure `TheftServedDeployed` takes. `Two`: `TheftServed` at a two-stage audit (`TheftServedTwo`), its budget
`T·(K − 1) + 1` from the cap `WindowCap`.
-/
