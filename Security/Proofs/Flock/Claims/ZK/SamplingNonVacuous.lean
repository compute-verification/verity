import Proofs.Flock.Claims.ZK.NonVacuous
import Definitions.Claims.Sampling

/-!
# The sampling `ZK` is not vacuous

The form before kept the simulator's own SHA-512 calls in `ZKmalR`'s collision term, so a simulator that hashes a
colliding pair whatever the leaf layer met the sampling `ZK` at `ε_zk = 0`. The term is now `γ` times `Fork`'s success,
which reads only the honest prover's calls in two real runs:

* `collSimLeak u v`: the simulator that hashes `u`, then `v`, whatever the leaf layer, and outputs the empty transcript.
* `not_samplingZK_collSim`: the sampling `ZK` fails at it for the quiet prover, at every `γ`, at an honest pair of a
  statement of at least one round with `ε_zk < 1` there.
* `not_samplingZK_leak`: at a statement with two honest witnesses of one leaf layer, the prover that sends its witness
  has no simulator at `ε_zk < 1/2`, at every `γ`.
-/

namespace Claims.Sampling.ZKMal

open Claims Claims.ZKMal

/-- **The colliding simulator given the leakage**: it hashes `u`, then `v`, and outputs the empty transcript, whatever
the leaf layer. -/
def collSimLeak (u v : Msg) : SimLeak where
  C _ := Unit
  prog _ _ _ := .hash u fun _ => .hash v fun _ => .ret []

/-- **The sampling `ZK` fails at the colliding simulator**, at every `γ`, at an honest pair of a statement of at least
one round with `ε_zk < 1` there. -/
theorem not_samplingZK_collSim {Coin Col T N Ω : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    [DecidableEq Col] [Zero Col] [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg)
    (D : Deploy Coin Col T) (u v : Msg) (εzk γ : Msg → ℝ) (x : Msg) (w : Wit)
    (hR : Rhon H D.commit D.parse D.partition x w) (hk : 0 < D.V.k x) (hε : εzk x < 1) :
    ¬ ZK H D (fun _ => quietP N Ω []) (collSimLeak u v) εzk γ := fun h =>
  not_zkmalR_of_out H D.V.k _ [] ((collSimLeak u v).given fun x' => leafLayer H D.commit D.parse x' w) εzk γ x w
    ⟨hR, rfl⟩ hk hε (fun _ => by simp [Simulator.out, SimLeak.given, collSimLeak, SP.eval]) (h.2 _)

/-- **No simulator meets the sampling `ZK` for the witness-sending prover** at `ε_zk < 1/2`, at every `γ`, at a statement
of at least one round with two honest witnesses of one leaf layer that it sends differently. -/
theorem not_samplingZK_leak {Coin Col T N Ω : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    [DecidableEq Col] [Zero Col] [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg)
    (D : Deploy Coin Col T) (f : Wit → Msg) (S : SimLeak) (εzk γ : Msg → ℝ) (x : Msg) (w₁ w₂ : Wit)
    (h₁ : Rhon H D.commit D.parse D.partition x w₁) (h₂ : Rhon H D.commit D.parse D.partition x w₂)
    (hl : leafLayer H D.commit D.parse x w₁ = leafLayer H D.commit D.parse x w₂) (hf : f w₁ ≠ f w₂)
    (hk : 0 < D.V.k x) (hε : εzk x < 1 / 2) : ¬ ZK H D (fun w => quietP N Ω (f w)) S εzk γ := fun h =>
  not_zkmalR_leak H D.V.k _ f (S.given fun _ => leafLayer H D.commit D.parse x w₁) εzk γ x w₁ w₂ ⟨h₁, rfl⟩
    ⟨h₂, hl.symm⟩ hf hk hε (h.2 _)

end Claims.Sampling.ZKMal
