module

public import Proofs.Flock.Recursive.LoopAny
public import Proofs.Flock.SessionLoop.TablesSide

/-!
# `RecursiveZKLoopAnySum` at the deployed tables

`Flock.Guarantees.RecursiveZKLoopAnySumDeployed` is `RecursiveZKLoopAnySum` with the five premises it takes of the
deployed tables proved there, as `SessionLoopZKDeployed` has them: `TablesSide`'s `tables_dataE`, `complete_dataE`,
`dummyReg_dataE`, `causal_dataE` and `tapeOK_dataE`. `SwCausalOn (Loop.SwOk Masked.dataE) vs` stays a premise: the
workers' values `vs` are the statement's, and it holds at the honest ones (`swCausalOn_honestVals_dataE`).

* `Flock.SecurityProofs.RecursiveZKLoopAnySumDeployed`: `RecursiveZKLoopAnySum` at those five theorems.
-/

@[expose] public section

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **The recursive audit at the session loop is zero knowledge at any worker, released at a stop, at the deployed
tables.** `RecursiveZKLoopAnySum`, its premises on the deployed tables (`Tables`, `Complete`, `DummyReg`, `Causal`,
`TapeOK` of the loop's session) discharged. It takes, each an obligation: `SwCausalOn` at the workers' values `vs`, at
the coin vectors `Loop.SwOk` admits; the worker's values satisfying V*'s circuits wherever the inner session finished
(`OuterZK.LoopSessions.Honest` at `Claims.Flock.R`); V*'s statements fixing each one's simulator randomness
(`OuterZK.LoopSessions.Sized`); and the release reading no stop where the inner session didn't finish
(`ReleaseReadsNoStop`), which the deployed release meets (`LoopAny.releaseReadsNoStop_loopRelease`). The bound is
`RecursiveZKLoopAnySum`'s. -/
def RecursiveZKLoopAnySumDeployed : Prop :=
  ∀ {Ωa Ωp Θ St Ωk W : Type} [Fintype Ωa] [Nonempty Ωa] [Fintype Ωp] [Nonempty Ωp] [Fintype St] [Nonempty St]
    [Fintype Ωk] [Nonempty Ωk]
    (vs : Claims.ZKServer.Loop.finS.SwVals) (t : ℕ)
    (L₀ : Claims.ZKServer.LeanRelease Claims.Flock.Wit Claims.ZKServer.Loop.Tp)
    (src : Claims.Msg → Claims.Flock.Wit → Claims.ZKServer.Loop.Tp → ℕ →
      _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray))
    (s : Flock.Firewall.Schedule) (w : Ωa → Ωp → FirewallAdaptive.Worker) (N r : ℕ)
    (m : ℕ) (stmt : ℕ → Ωa × (Fin (fwHoles s) → ByteArray) → Claims.Msg)
    (wit : ℕ → OuterZK.WCtx Ωa Ωp (fwHoles s) → Claims.Flock.Wit) (CB : ℕ → Type) [∀ j, Fintype (CB j)]
    [∀ j, Nonempty (CB j)]
    (Vs : Claims.VStar Θ) (θ : Ωa → (Fin (fwHoles s) → ByteArray) → Θ),
    0 < t →
    Claims.ZKServer.Loop.finS.SwCausalOn (Claims.ZKServer.Loop.SwOk Claims.ZKMal.Masked.dataE) vs →
    (OuterZK.LoopSessions.atWorker m stmt wit).Honest
      (Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse)
      (fun u : {u : (Ωa × Ωp) × (Fin (fwHoles s + r) → Fin 192 → Fin 256) //
          FirewallAdaptive.adaptFinished s (w u.1.1 u.1.2) N (saltsOf u.2) = true} =>
        anyCtx s w N r u.1.1.1 u.1.1.2 u.1.2) →
    (OuterZK.LoopSessions.atWorker m stmt wit).Sized (OuterZK.loopSim L₀ src t) CB →
    ∀ (cut : St → Ωa × (Array Flock.Firewall.Sent × Bool) × List (Claims.Msg × Claims.Out) → W) (s₀ : St),
      ReleaseReadsNoStop cut s₀ →
    ∀ (stop : (((Ωa × Ωp) × (Fin (fwHoles s + r) → Fin 192 → Fin 256)) × Claims.Sampling.unitTape.at 0
        ((OuterZK.loopRef L₀ src).swP vs).Nonce ((OuterZK.loopRef L₀ src).swP vs).Rand m) × Ωk → St)
      (E : W → Prop),
      prCoin (fun u => E (cut (stop u) (anyReal vs L₀ src s w N r m stmt wit Vs θ u.1))) ≤
        (Fintype.card (Fin ((FirewallTemplate.tmpl s s.tables).length + 1) ⊕ St) : ℝ≥0∞) *
            prCoin (fun c : (Fin ((FirewallTemplate.tmpl s s.tables).length + 1) ⊕ St) ×
                ((Ωa × (Fin (fwHoles s) → ZkView.Bits × ZkView.Salt)) × OuterZK.LoopSessions.SimCoins CB m) =>
              E (sumCut s₀ cut c.1 (loopIdeal t L₀ src (fwFilled s) m stmt CB Vs θ c.2))) +
          (OuterZK.LoopSessions.atWorker m stmt wit).err Claims.ZKMal.CoinOpen.shaH OuterZK.loopK
            ((OuterZK.loopRef L₀ src).swP vs).prover (OuterZK.loopEps L₀ src t) (4 * t * 3 ^ t) Vs
            (fun x => θ x.1 x.2.1) +
          fwHoles s * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash)

end Flock.Guarantees

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion

/-- **`RecursiveZKLoopAnySum` at the deployed tables**: its five premises on them are `TablesSide`'s theorems. -/
theorem RecursiveZKLoopAnySumDeployed : Flock.Guarantees.RecursiveZKLoopAnySumDeployed := by
  intro Ωa Ωp Θ St Ωk W _ _ _ _ _ _ _ _ vs t L₀ src s w N r m stmt wit CB _ _ Vs θ ht hsc hH hsz
  exact RecursiveZKLoopAnySum vs t L₀ src s w N r m stmt wit CB Vs θ ht Claims.ZKServer.Loop.tables_dataE
    Claims.ZKServer.Loop.complete_dataE Claims.ZKServer.Loop.dummyReg_dataE Claims.ZKServer.Loop.causal_dataE hsc
    Claims.ZKServer.Loop.tapeOK_dataE hH hsz

end Flock.SecurityProofs
