import Proofs.Flock.Claims.ZK.NonVacuous
import Definitions.Claims.Sampling
import Proofs.Flock.Claims.ZK.Indexed

/-!
# The sampling `ZK` is not vacuous

The form before kept the simulator's own SHA-512 calls in `ZKmalR`'s collision term, so a simulator that hashes a
colliding pair whatever the leaf layer met the sampling `ZK` at `ε_zk = 0`. The term is now `γ` times `Fork`'s success,
which reads only the honest prover's calls in two real runs; and the claim reads the masked deployment's own reference
prover (`ZkDeploy.prover`) and simulator (`ZkDeploy.S`), so neither is the claimant's to choose:

* `collSimLeak u v`: the simulator that hashes `u`, then `v`, whatever the leaf layer, and outputs the empty transcript.
* `not_samplingZK_collSim`: the sampling `ZK` fails at a deployment whose reference prover is quiet and whose simulator
  is `collSimLeak u v`, at every `γ`, at an honest pair of a statement of at least one round with `ε_zk < 1` there.
* `not_samplingZK_leak`: at a statement with two honest witnesses of one leaf layer, a deployment whose reference prover
  sends its witness meets the sampling `ZK` at no simulator and `ε_zk < 1/2`, at every `γ`.
* `quietZD D₀ u v`: the masked deployment of a deployment `D₀` whose reference prover is quiet, at `collSimLeak u v`;
  where `D₀` has an honest pair at a statement of at least one round, the sampling `ZK` is not met at every masked
  deployment (`not_forall_samplingZK`).
-/

namespace Claims.Sampling.ZKMal

open Claims Claims.ZKMal

/-- **The colliding simulator given the leakage**: it hashes `u`, then `v`, and outputs the empty transcript, whatever
the leaf layer. -/
def collSimLeak (u v : Msg) : SimLeak where
  C _ := Unit
  prog _ _ _ := .hash u fun _ => .hash v fun _ => .ret []

/-- **The sampling `ZK` fails at a deployment whose reference prover is quiet and whose simulator is colliding**, at
every `γ`, at an honest pair of a statement of at least one round with `ε_zk < 1` there. -/
theorem not_samplingZK_collSim {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    [DecidableEq Col] [Zero Col] (H : Msg → Msg) (D : ZkDeploy Coin Col T) (u v : Msg) (εzk γ : Msg → ℝ) (x : Msg)
    (w : Wit) (hP : ∀ w, D.prover w x = quietP (D.Nonce x) (D.Rest x) []) (hS : D.S = collSimLeak u v)
    (hR : Rhon H D.commit D.parse D.partition x w) (hk : 0 < D.V.k x) (hε : εzk x < 1) : ¬ ZK H D εzk γ := fun h => by
  have h2 := (h.2 fun x' => leafLayer H D.commit D.parse x' w).at x
  rw [funext hP, hS] at h2
  exact not_zkmalR_of_out H D.V.k _ [] _ εzk γ x w ⟨rfl, hR, rfl⟩ hk hε
    (fun _ => by simp [Simulator.out, SimLeak.given, collSimLeak, SP.eval]) h2

/-- **No simulator meets the sampling `ZK` at a deployment whose reference prover sends its witness**, at
`ε_zk < 1/2`, at every `γ`, at a statement of at least one round with two honest witnesses of one leaf layer that it
sends differently. -/
theorem not_samplingZK_leak {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    [DecidableEq Col] [Zero Col] (H : Msg → Msg) (D : ZkDeploy Coin Col T) (f : Wit → Msg) (εzk γ : Msg → ℝ) (x : Msg)
    (w₁ w₂ : Wit) (hP : ∀ w, D.prover w x = quietP (D.Nonce x) (D.Rest x) (f w))
    (h₁ : Rhon H D.commit D.parse D.partition x w₁) (h₂ : Rhon H D.commit D.parse D.partition x w₂)
    (hl : leafLayer H D.commit D.parse x w₁ = leafLayer H D.commit D.parse x w₂) (hf : f w₁ ≠ f w₂)
    (hk : 0 < D.V.k x) (hε : εzk x < 1 / 2) : ¬ ZK H D εzk γ := fun h => by
  have h2 := (h.2 fun _ => leafLayer H D.commit D.parse x w₁).at x
  rw [funext hP] at h2
  exact not_zkmalR_leak H D.V.k _ f _ εzk γ x w₁ w₂ ⟨rfl, h₁, rfl⟩ ⟨rfl, h₂, hl.symm⟩ hf hk hε h2

/-- **The masked deployment of `D₀` whose reference prover is quiet**: `D₀` with the prover that sends `[]` and calls
nothing, one nonce and one rest, and the simulator `collSimLeak u v`. -/
noncomputable def quietZD {Coin Col T : Type} (D₀ : Deploy Coin Col T) (u v : Msg) : ZkDeploy Coin Col T :=
  { D₀ with
    P := fun _ => ⟨fun _ _ _ => .ret []⟩
    Nonce := fun _ => Unit
    Rest := fun _ => Unit
    tapeOf := fun _ _ => Classical.choice D₀.tapeNonempty
    S := collSimLeak u v }

/-- **The sampling `ZK` is not met at every masked deployment**, at `ε_zk = 0` and `γ = 0`, where some masked deployment
`D₀` has an honest pair at a statement of at least one round: `quietZD D₀ [] []` fails it. -/
theorem not_forall_samplingZK {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    [DecidableEq Col] [Zero Col] (H : Msg → Msg) (D₀ : Deploy Coin Col T) (hzk : D₀.flock.zk = true) (x : Msg)
    (w : Wit) (hR : Rhon H D₀.commit D₀.parse D₀.partition x w) (hk : 0 < D₀.V.k x) :
    ¬ ∀ D : ZkDeploy Coin Col T, D.flock.zk = true → ZK H D 0 0 := fun h =>
  not_samplingZK_collSim H (quietZD D₀ [] []) [] [] 0 0 x w (fun _ => rfl) rfl hR hk one_pos (h _ hzk)

end Claims.Sampling.ZKMal
