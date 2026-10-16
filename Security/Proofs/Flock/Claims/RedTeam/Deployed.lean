import Proofs.Flock.Claims.RedTeam.ZK
import Proofs.Flock.Claims.RedTeam.Complete
import Proofs.Flock.Claims.RedTeam.Efficient
import Proofs.Flock.Claims.DeployCode
import Proofs.Flock.Claims.GS.RadiusDeploy
import Proofs.Flock.Claims.Assembled
import Proofs.Flock.Verifier.Hash

/-!
# Red team: the discharges at SHA-512 and the deployed records

At SHA-512 (`FlockDeploy.sha512`) and the deployed records (`FlockDeploy.deployPlain`, `FlockDeploy.deployZk`), at every
readable relation `rel`:

* `exists_collision_sha512`: SHA-512 has a collision; its outputs are 64 bytes.
* `flockZK_deployZk`: Flock's `ZK` holds at `ε_zk = 0` for every honest prover, at a simulator of cost `2`.
* `flockComplete_deployPlain`, `flockComplete_deployZk`: Flock's `Complete` holds at `δ_c = 0` at an empty tape.
* `efficient_deployPlain`, `efficient_deployZk`: Flock's `Efficient` holds at `p = 0` at the deployed decoders.
* `flock_plain_assembled_of_KS`, `flock_zk_assembled_of_KS`: the assembled claims (`FlockDeploy.Assembled`) from `KS`
  alone, their completeness hypothesis met at the empty tape.
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

/-- **The assembled `--zk` claims rest on `KS` alone**: `FlockDeploy.Assembled.flock_zk_assembled`'s completeness
hypothesis, which no branch owns, is met at the empty-tape prover at `δ_c = 0`. -/
theorem flock_zk_assembled_of_KS (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (H : Msg → Msg) (hKS : Flock.KS H (FlockDeploy.deployZk rel)) :
    Flock.ScopeMatches (FlockDeploy.deployZk rel) ∧
      Flock.KS H (FlockDeploy.deployZk rel) ∧
      Flock.Bits H (FlockDeploy.deployZk rel) ∧
      (∀ a b : Msg → ℝ, (∀ x, Flock.κ (FlockDeploy.deployZk rel) x ≤ 2 ^ (-a x)) →
        (∀ x, Flock.γ (FlockDeploy.deployZk rel) x ≤ 2 ^ b x) →
        Claims.Bits H (FlockDeploy.deployZk rel).V (Flock.R (FlockDeploy.deployZk rel).parse)
          (Flock.E (FlockDeploy.deployZk rel) GS.zkDecodeRD) fun x => min (a x) (255.5 - b x / 2) - 1) ∧
      Flock.Efficient H (FlockDeploy.deployZk rel) GS.zkDecodeRD GS.zkCountedRD GS.zkSizeRD
        (Polynomial.C 11 * (GS.zkPolyR + Polynomial.X)) ∧
      Flock.Complete H (FlockDeploy.deployZk rel) noTape 0 :=
  FlockDeploy.Assembled.flock_zk_assembled rel H hKS noTape 0 (flockComplete_deployZk rel H)

/-- **The assembled plain claims rest on `KS` alone**, as at `--zk`: `flock_plain_assembled`'s completeness hypothesis
is met at the empty-tape prover. -/
theorem flock_plain_assembled_of_KS (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (H : Msg → Msg) (hKS : Flock.KS H (FlockDeploy.deployPlain rel)) :
    Flock.ScopeMatches (FlockDeploy.deployPlain rel) ∧
      Flock.KS H (FlockDeploy.deployPlain rel) ∧
      Flock.Bits H (FlockDeploy.deployPlain rel) ∧
      (∀ a b : Msg → ℝ, (∀ x, Flock.κ (FlockDeploy.deployPlain rel) x ≤ 2 ^ (-a x)) →
        (∀ x, Flock.γ (FlockDeploy.deployPlain rel) x ≤ 2 ^ b x) →
        Claims.Bits H (FlockDeploy.deployPlain rel).V (Flock.R (FlockDeploy.deployPlain rel).parse)
          (Flock.E (FlockDeploy.deployPlain rel) GS.plainDecodeRD) fun x => min (a x) (255.5 - b x / 2) - 1) ∧
      Flock.Efficient H (FlockDeploy.deployPlain rel) GS.plainDecodeRD GS.plainCountedRD GS.plainSizeRD
        (Polynomial.C 14 * (GS.flockPolyR + Polynomial.X)) ∧
      Flock.Complete H (FlockDeploy.deployPlain rel) noTape 0 :=
  FlockDeploy.Assembled.flock_plain_assembled rel H hKS noTape 0 (flockComplete_deployPlain rel H)

end Claims.RedTeam.Deployed
