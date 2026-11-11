module

public import Proofs.Flock.SessionLoop.ZkAny
public import Proofs.Flock.SessionLoop.Honest
public import Proofs.Flock.SessionLoop.RepSrcSat
public import Proofs.Flock.SessionLoop.Concrete.LiveNets
public import Proofs.Flock.SessionLoop.Concrete.Scope
public import Proofs.Flock.SessionLoop.Concrete.Serves

/-!
# `SessionLoopZKAnyDeployed`'s premises at the honest session loop

`Flock.Guarantees.SessionLoopZKAnyDeployed` takes, at a relation `Rl`, values `vs` and a worker `wk`, that the
session's table witnesses are valid at `Rl`'s statements (`Complete`), that the values' unswitched round encoding is
causal (`UnCausal`) and that the transport's trusts hold at `Rl`'s statements (`LeanRelease.TrustsStop`). All three hold
at one instance:

* the relation of one pair, the concrete `--zk` statement with its trace (`FlockDeploy.concreteStmt`,
  `FlockDeploy.wC`), which are in the deployed relation (`FlockDeploy.concreteStmt_rel`) and in its scope
  (`FlockDeploy.concreteStmt_scope`);
* `Loop.tapeRel`: a release whose frames sample the tape's samples at that statement (`Loop.tapeDraws`), released as
  the session loop's own inner rounds (`withInner`, at the finish's honest rounds `finHonest` and with Rust's checks
  passing, `acceptCross`), link and `Finish` (`ofLoopLink`) on the source `Loop.tapeSrcHi`;
* the honest values (`finS.honestVals`) and the honest worker (`valsWk`).

`Flock.SecurityProofs.SessionLoopZKAnyDeployed_honest` concludes the three from `finS_complete`, `finS_unCausal` and
`LeanRelease.trustsStop_tape`, since the loop serves the statement (`Loop.concreteServes`).
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer.Loop

open Claims Claims.ZKMal

/-- A release whose frames sample the tape's samples at the concrete statement (`tapeDraws`), its other fields
`LoopWitness.rel0`'s. -/
noncomputable def tapeRel : LeanRelease Flock.Wit Tp where
  rep _ _ := 0
  lay _ _ := default
  slot _ _ := none
  draws := tapeDraws FlockDeploy.concreteStmt
  prog _ _ := {}
  coinsOf _ _ := #[]
  fresh _ _ := .error ""
  device _ _ _ _ _ := []
  computed _ _ _ _ := default
  cross _ _ _ _ _ _ := false
  scheduled _ _ _ _ _ := []
  coins _ _ := 0
  sched _ _ := 0
  stop := []
  hello _ _ := []
  helloChk _ _ := .error ""
  req _ _ := ("", 0)
  link _ _ _ _ := []
  files _ _ _ _ := ("", #[])
  finStops _ _ _ _ := false
  last _ := 0

/-- **The honest session loop's inner rounds** at `tapeRel`: the finish's honest rounds with Rust's checks passing. -/
noncomputable def tapeL₀ : LeanRelease Flock.Wit Tp :=
  LeanRelease.withInner Masked.dataE (acceptCross tapeRel) (realOf tapeSrcHi)
    (finHonest (LeanRelease.ofLoopLink (acceptCross tapeRel) tapeSrcHi (realOf tapeSrcHi)))

/-- **The honest worker** at the session loop's release on `tapeL₀` and `tapeSrcHi`: the honest values' rounds. -/
noncomputable def tapeWk : (LeanRelease.ofLoopLink tapeL₀ tapeSrcHi (realOf tapeSrcHi)).toSrv.Worker :=
  valsWk finS.honestVals (LeanRelease.ofLoopLink tapeL₀ tapeSrcHi (realOf tapeSrcHi))

end Claims.ZKServer.Loop

namespace Flock.SecurityProofs

open Claims Claims.ZKMal Claims.ZKServer

/-- **`SessionLoopZKAnyDeployed`'s premises hold at the honest session loop**: at the relation of one pair, the
concrete statement and its trace, the honest values and the honest worker at the loop's own release on the tape's
samples, the session's table witnesses are valid, the values' unswitched round encoding is causal and the transport's
trusts hold. -/
theorem SessionLoopZKAnyDeployed_honest :
    (CoinLayer.session Loop.loopNr Masked.dataE).Complete
        (fun x w => x = FlockDeploy.concreteStmt ∧ w = FlockDeploy.wC) ∧
      (CoinLayer.session Loop.loopNr Masked.dataE).UnCausal Loop.finS.honestVals ∧
      LeanRelease.TrustsStop (CoinLayer.session Loop.loopNr Masked.dataE) Loop.finS.honestVals
        (LeanRelease.ofLoopLink Loop.tapeL₀ Loop.tapeSrcHi (Loop.realOf Loop.tapeSrcHi)) CoinOpen.shaH Loop.tapeWk
        (fun x w => x = FlockDeploy.concreteStmt ∧ w = FlockDeploy.wC) := by
  have hRZk : ∀ x w, x = FlockDeploy.concreteStmt ∧ w = FlockDeploy.wC → Masked.RZk x w := fun x w h => by
    obtain ⟨rfl, rfl⟩ := h
    exact FlockDeploy.concreteStmt_rel
  refine ⟨Loop.finS_complete _ hRZk, Loop.finS_unCausal, ?_⟩
  exact LeanRelease.trustsStop_tape Loop.tapeRel FlockDeploy.concreteStmt rfl
    (fun x w => x = FlockDeploy.concreteStmt ∧ w = FlockDeploy.wC)
    (fun x w h => by
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨Loop.concreteServes, FlockDeploy.concreteStmt_scope _root_.Properties.Flock.Nets.fixedNetsBuild,
        FlockDeploy.concreteStmt_rel⟩)
    (fun x w h => h.1)

end Flock.SecurityProofs
