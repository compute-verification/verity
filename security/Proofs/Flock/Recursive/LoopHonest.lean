module

public import Proofs.Flock.Recursive.LoopWitness
public import Proofs.Flock.Claims.Recursion.BaseParses

/-!
# `RecursiveZKLoop` and `RecursiveZKLoopAnySum`'s premises hold

Each guarantee's honest lemma concludes its premises at `LoopWitness`'s instance with one statement of V* (`m = 1`), the
honest fixture's (`RecDeploy.BaseParses.fixture_stmt`), whose statement and witness are in the claims' relation at the
deployed masked parser, `deployZk Readable.relation`.

* `Flock.SecurityProofs.RecursiveZKLoop_honest`.
* `Flock.SecurityProofs.RecursiveZKLoopAnySum_honest`.
-/

@[expose] public section

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Flock.ZkServer ZkServerAdaptive LoopWitness
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **`RecursiveZKLoop`'s premises hold at one instance**: the deployed tables (with `SwCausalOn` at the honest
values), the ZK server sending `sched`'s transcript on the honest worker's requests, and one statement of V*, the honest
fixture's, in the claims' relation at the deployed masked parser at every context, its simulator's coins its own. -/
theorem RecursiveZKLoop_honest (h : RecDeploy.HonestFixture) :
    ∃ (y : Hash.Msg) (w : Claims.Flock.Wit),
      Claims.ZKServer.Loop.finS.Tables ∧
      Claims.ZKServer.Loop.finS.Complete (Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse) ∧
      Claims.ZKServer.Loop.finS.DummyReg ∧ Claims.ZKServer.Loop.finS.Causal ∧
      Claims.ZKServer.Loop.finS.SwCausalOn (Claims.ZKServer.Loop.SwOk Claims.ZKMal.Masked.dataE)
        Claims.ZKServer.Loop.finS.honestVals ∧
      Claims.ZKServer.Loop.finS.TapeOK ∧
      Flock.Guarantees.ZkServerSends (Prog := Unit) (Wit := Unit) (Rr := Unit) sched
        (fun _ _ _ => fun (_ : Unit) (_ : Unit) (_ : Fin nhS → ByteArray) => Flock.ZkServer.honestReqs sched hon)
        (fun _ _ _ => ZkServerHonest.honestMsgs sched honD () () ())
        (Flock.Guarantees.zkServerT sched sched.tables) ∧
      (OuterZK.LoopSessions.atWorker 1 (fun (_ : ℕ) (_ : Unit × (Fin nhS → ByteArray)) => y)
          (fun (_ : ℕ) (_ : OuterZK.WCtx Unit Unit nhS) => w)).Honest
        (Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse)
        (fun u : (Unit × Unit) × (Fin nhS → ZkView.Salt) =>
          OuterZK.wctxAt (ZkServerHonest.honestMsgs sched honD () () ()) u.1.1 u.1.2 u.2) ∧
      ∀ (L₀ : Claims.ZKServer.LeanRelease Claims.Flock.Wit Claims.ZKServer.Loop.Tp)
        (src : Hash.Msg → Claims.Flock.Wit → Claims.ZKServer.Loop.Tp → ℕ →
          _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray)) (t : ℕ),
        (OuterZK.LoopSessions.atWorker 1 (fun (_ : ℕ) (_ : Unit × (Fin nhS → ByteArray)) => y)
          (fun (_ : ℕ) (_ : OuterZK.WCtx Unit Unit nhS) => w)).Sized (OuterZK.loopSim L₀ src t)
          (fun _ => (OuterZK.loopSim L₀ src t).C y) := by
  obtain ⟨y, w, hR⟩ := RecDeploy.BaseParses.fixture_stmt h
  exact ⟨y, w, Claims.ZKServer.Loop.tables_dataE, Claims.ZKServer.Loop.complete_dataE,
    Claims.ZKServer.Loop.dummyReg_dataE, Claims.ZKServer.Loop.causal_dataE,
    Claims.ZKServer.Loop.swCausalOn_honestVals_dataE, Claims.ZKServer.Loop.tapeOK_dataE, sends,
    fun _ _ _ => hR, fun _ _ _ _ _ _ => ⟨Equiv.refl _⟩⟩

/-- **`RecursiveZKLoopAnySum`'s premises hold at one instance**: the deployed tables (with `SwCausalOn` at the honest
values); the honest worker's requests, as a worker of the adaptive zk_server, finishing at every salt, so its contexts
are all of them; one statement of V*, the honest fixture's, in the claims' relation at the deployed masked parser at
every context, its simulator's coins its own; and the deployed release reading no stop. -/
theorem RecursiveZKLoopAnySum_honest (h : RecDeploy.HonestFixture) :
    ∃ (y : Hash.Msg) (w : Claims.Flock.Wit),
      Claims.ZKServer.Loop.finS.Tables ∧
      Claims.ZKServer.Loop.finS.Complete (Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse) ∧
      Claims.ZKServer.Loop.finS.DummyReg ∧ Claims.ZKServer.Loop.finS.Causal ∧
      Claims.ZKServer.Loop.finS.SwCausalOn (Claims.ZKServer.Loop.SwOk Claims.ZKMal.Masked.dataE)
        Claims.ZKServer.Loop.finS.honestVals ∧
      Claims.ZKServer.Loop.finS.TapeOK ∧
      (∀ u : (Unit × Unit) × (Fin (Flock.Guarantees.fwHoles sched + 0) → Fin 192 → Fin 256),
        adaptFinished sched (wAny u.1.1 u.1.2) nA (Flock.Guarantees.saltsOf u.2) = true) ∧
      (OuterZK.LoopSessions.atWorker 1
          (fun (_ : ℕ) (_ : Unit × (Fin (Flock.Guarantees.fwHoles sched) → ByteArray)) => y)
          (fun (_ : ℕ) (_ : OuterZK.WCtx Unit Unit (Flock.Guarantees.fwHoles sched)) => w)).Honest
        (Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse)
        (fun u : {u : (Unit × Unit) × (Fin (Flock.Guarantees.fwHoles sched + 0) → Fin 192 → Fin 256) //
            adaptFinished sched (wAny u.1.1 u.1.2) nA (Flock.Guarantees.saltsOf u.2) = true} =>
          Flock.Guarantees.anyCtx sched wAny nA 0 u.1.1.1 u.1.1.2 u.1.2) ∧
      (∀ (L₀ : Claims.ZKServer.LeanRelease Claims.Flock.Wit Claims.ZKServer.Loop.Tp)
        (src : Hash.Msg → Claims.Flock.Wit → Claims.ZKServer.Loop.Tp → ℕ →
          _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray)) (t : ℕ),
        (OuterZK.LoopSessions.atWorker 1
          (fun (_ : ℕ) (_ : Unit × (Fin (Flock.Guarantees.fwHoles sched) → ByteArray)) => y)
          (fun (_ : ℕ) (_ : OuterZK.WCtx Unit Unit (Flock.Guarantees.fwHoles sched)) => w)).Sized
          (OuterZK.loopSim L₀ src t) (fun _ => (OuterZK.loopSim L₀ src t).C y)) ∧
      ∀ (n : ℕ) (M : Fin n → ℕ),
        Flock.Guarantees.ReleaseReadsNoStop (Flock.Guarantees.loopRelease (Ωa := Unit) M) none := by
  obtain ⟨y, w, hR⟩ := RecDeploy.BaseParses.fixture_stmt h
  exact ⟨y, w, Claims.ZKServer.Loop.tables_dataE, Claims.ZKServer.Loop.complete_dataE,
    Claims.ZKServer.Loop.dummyReg_dataE, Claims.ZKServer.Loop.causal_dataE,
    Claims.ZKServer.Loop.swCausalOn_honestVals_dataE, Claims.ZKServer.Loop.tapeOK_dataE, fun u => wAny_finishes u.2,
    fun _ _ _ => hR, fun _ _ _ _ _ _ => ⟨Equiv.refl _⟩, fun _ M => LoopAny.releaseReadsNoStop_loopRelease M⟩

end Flock.SecurityProofs
