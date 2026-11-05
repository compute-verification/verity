module

public import Proofs.Flock.SessionLoop.Statement
public import Proofs.Flock.Claims.ZK.CausalExe

/-!
# The tables' side of `SessionLoopZKDeployed` at the deployed tables

At the session `CoinLayer.session Loop.loopNr Masked.dataE`, the tables' side `Flock.SecurityProofs.SessionLoopZKDeployed`
discharges (all of it but `SwCausalOn`), each at the deployed prover `Masked.PE`, its stopped round plan and its tape
plan `Masked.UE`:

* `tables_dataE`: every table is one it sets up (`Masked.tables`).
* `complete_dataE`: at the deployed relation each table's share is valid (`complete_tabC`).
* `dummyReg_dataE`: the dummy's table witnesses carry the public region data (`Masked.dummyReg`).
* `causal_dataE`: round `i` reads the coins before `i` only (`Masked.causalExe`, kept by stopping where a refusal is
  seen, `Masked.causal_encStop`).
* `tapeOK_dataE`: the tape is read with fibers of one size (`Masked.fibersEq_ofLayout` at `TapeLayout.exe`).

`SwCausalOn (Loop.SwOk Masked.dataE) vs` stays a premise: the statement's other premises read `vs` only at the coins a
strict prefix pads to, and `SwCausalOn` reads it at every coin vector with no outer coin `1` (`Loop.SwOk`). It holds at
the honest values (`swCausalOn_honestVals_dataE`, from `swCausal_honestVals_dataE`, `RefSession.swCausal_vals`), so the
statement applies to an honest worker.
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal

namespace Loop

open Masked

theorem tables_dataE : (CoinLayer.session Loop.loopNr Masked.dataE).Tables :=
  Masked.tables PE _ UE Loop.loopNr

theorem complete_dataE :
    (CoinLayer.session Loop.loopNr Masked.dataE).Complete
      (Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse) :=
  fun _ _ h j => complete_tabC PE h j

theorem dummyReg_dataE : (CoinLayer.session Loop.loopNr Masked.dataE).DummyReg :=
  Masked.dummyReg PE _ UE Loop.loopNr

theorem causal_dataE : (CoinLayer.session Loop.loopNr Masked.dataE).Causal :=
  RoundPlan.causal (causal_encStop causalExe) UE Loop.loopNr

theorem tapeOK_dataE : (CoinLayer.session Loop.loopNr Masked.dataE).TapeOK :=
  (RefSession.tapeOKOn_iff _).1 <|
    RefSession.tapeOKOn_of_tape _ (fun _ => True) fun x _ => fibersEq_ofLayout (TapeLayout.exe PE) exeTape x

theorem swCausal_honestVals_dataE :
    (CoinLayer.session Loop.loopNr Masked.dataE).SwCausal (CoinLayer.session Loop.loopNr Masked.dataE).honestVals :=
  RefSession.swCausal_vals _ causal_dataE

theorem swCausalOn_honestVals_dataE :
    (CoinLayer.session Loop.loopNr Masked.dataE).SwCausalOn (Loop.SwOk Masked.dataE)
      (CoinLayer.session Loop.loopNr Masked.dataE).honestVals :=
  swCausalOn_swOk Masked.dataE _ swCausal_honestVals_dataE

end Loop

end Claims.ZKServer
