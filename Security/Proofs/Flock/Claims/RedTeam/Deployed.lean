import Proofs.Flock.Claims.RedTeam.ZK
import Proofs.Flock.Claims.RedTeam.Complete
import Proofs.Flock.Claims.RedTeam.Efficient
import Proofs.Flock.Claims.DeployCode
import Proofs.Flock.Claims.GS.RadiusDeploy
import Proofs.Flock.Verifier.Hash

/-!
# Red team: the discharges at SHA-512 and the deployed records

At SHA-512 (`FlockDeploy.sha512`) and the deployed records (`FlockDeploy.deployPlain`, `FlockDeploy.deployZk`), at every
readable relation `rel`:

* `exists_collision_sha512`: SHA-512 has a collision; its outputs are 64 bytes.
* `flockZK_deployZk`: Flock's `ZK` holds at `ε_zk = 0` for every honest prover, at a simulator of cost `2`.
* `flockComplete_deployPlain`, `flockComplete_deployZk`: Flock's `Complete` holds at `δ_c = 0` at an empty tape.
* `efficient_deployPlain`, `efficient_deployZk`: Flock's `Efficient` holds at `p = 0` at the deployed decoders.
-/

namespace Claims.RedTeam.Deployed

theorem sha512_length (u : Msg) : (FlockDeploy.sha512 u).length = 64 := by
  have h := _root_.Flock.Sha512.hash_size (ofMsg u)
  show (_root_.Flock.Sha512.hash (ofMsg u)).data.toList.length = 64
  generalize _root_.Flock.Sha512.hash (ofMsg u) = b at h ⊢
  rw [Array.length_toList]
  exact h

theorem exists_collision_sha512 : ∃ u v, Collision FlockDeploy.sha512 u v :=
  exists_collision fun u => (sha512_length u).le

/-- **Flock's `ZK` at SHA-512 and the deployed `--zk` record holds at `ε_zk = 0`** for every honest prover, at a
simulator of cost `2`. -/
theorem flockZK_deployZk (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) {Ω : Type} [Fintype Ω]
    (P : Flock.Wit → Prover Ω) :
    ∃ S, Flock.ZK FlockDeploy.sha512 (FlockDeploy.deployZk rel) P S 0 ∧ SimCost FlockDeploy.sha512 S fun _ => 2 :=
  let ⟨_, _, h⟩ := exists_collision_sha512
  ⟨_, flockZK_collSim _ _ rfl P h, simCost_collSim _ _ _⟩

theorem flockComplete_deployPlain (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (H : Msg → Msg) : Flock.Complete H (FlockDeploy.deployPlain rel) noTape 0 :=
  flockComplete_empty H _

theorem flockComplete_deployZk (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (H : Msg → Msg) : Flock.Complete H (FlockDeploy.deployZk rel) noTape 0 :=
  flockComplete_empty H _

theorem efficient_deployPlain (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (H : Msg → Msg) :
    Flock.Efficient H (FlockDeploy.deployPlain rel) GS.plainDecodeRD (fun os => (GS.plainDecodeRD os, 0))
      (fun _ => 0) 0 :=
  efficient_zero H _ _ (GS.listWithin_deployPlain rel)

theorem efficient_deployZk (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (H : Msg → Msg) :
    Flock.Efficient H (FlockDeploy.deployZk rel) GS.zkDecodeRD (fun os => (GS.zkDecodeRD os, 0)) (fun _ => 0) 0 :=
  efficient_zero H _ _ (GS.listWithin_deployZk rel)

end Claims.RedTeam.Deployed
