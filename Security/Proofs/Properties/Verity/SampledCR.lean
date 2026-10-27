import Proofs.Properties.Verity.SampledDeploy

/-!
# Sampled proofs' finder premise, read through C-Flock's finders

`SampledKSLivePub` takes no premise on its finders, so sampled proofs' soundness reads none (`SampledDeploy`). Where the
vocabulary's sampled finder is `LiveAt.crFinder` with the play's replay finder in its spare slot
(`SampledDeploy.finder_eq`) and meets the vocabulary's quantitative premise (`Vocab.FinderCR .sampled`: it outputs a
SHA-512 collision with probability at most the published `ε_cr` at the strategy's calls), that premise bounds what
C-Flock's analysis reads by public numbers:

* `LiveAt.fam_le_of_crFinder`: where `crFinder τ extra` succeeds at most `β`, each of C-Flock's finders (`LiveAt.fam`:
  each table slot's, averaged over the draw, each committed position's link finder, and the empty one) succeeds at most
  `K β` (`Claims.Finder.le_mix`, `K = LiveAt.K ≤ 20 J + |Pos| + 1`);
* `LiveAt.err_le_of_crFinder`: the session's error is at most `deltaPub + lossPub(J²·K·β)`;
* `SampledDeploy.crAvg_of`: both at the deployment, at `β = ε_cr`.
-/

namespace Flock.Guarantees.LiveAt

open FlockSoundness FlockSoundness.Game
open scoped ENNReal

variable {n : ℕ} {D : Flock.Desc} {hU : Built D.unroll.unit} {kd K₀ : ℕ} {desc : Claims.Msg}
  (A : LiveAt n D hU kd K₀ desc)

/-- **C-Flock's finders at `τ`** (`Bridge.pubFam`): `B^τ`'s members. -/
noncomputable abbrev fam (τ : Strategy A.G) : A.Ix → Claims.Finder :=
  Bridge.pubFam A.dj A.y₀ A.P A.vb A.k A.Rw A.M (A.qF τ) (A.qS τ) (A.t' τ) (reg (A.sim τ)) (cont (A.sim τ))

/-- **Each of C-Flock's finders succeeds at most `K β`** where `crFinder τ extra` succeeds at most `β`. -/
theorem fam_le_of_crFinder (τ : Strategy A.G) (extra : Claims.Finder) {β : ℝ}
    (hB : (A.crFinder τ extra).Pr Claims.Sampling.sha512Msg ≤ β) (i : A.Ix) :
    (A.fam τ i).Pr Claims.Sampling.sha512Msg ≤ A.K * β := by
  have hβ : 0 ≤ β := (Claims.Pr_nonneg _).trans hB
  rcases i with _ | i
  · have h0 : Claims.Finder.null.Pr Claims.Sampling.sha512Msg = 0 := by
      unfold Claims.Finder.Pr
      exact (Claims.Pr_congr (q := fun _ => False) fun _ => by simp [Claims.Finder.null]).trans Claims.Pr_false
    exact h0.le.trans (mul_nonneg (Nat.cast_nonneg _) hβ)
  · exact (Claims.Finder.le_mix Claims.Sampling.sha512Msg (A.crFam τ extra) (some i)).trans
      (mul_le_mul_of_nonneg_left hB (Nat.cast_nonneg _))

/-- **The session's error is at most `deltaPub + lossPub(J²·K·β)`** where `crFinder τ extra` succeeds at most `β`
(`pubB_le_crFinder`, `Bridge.lossPub_mono`). -/
theorem err_le_of_crFinder (τ : Strategy A.G) (extra : Claims.Finder) {β : ℝ}
    (hB : (A.crFinder τ extra).Pr Claims.Sampling.sha512Msg ≤ β) :
    A.err τ ≤ Bridge.deltaPub n kd K₀ A.k A.M A.Rw (Bridge.ioSum A.P) A.ρ +
      Bridge.lossPub A.k (Bridge.ioSum A.P) A.ρ ((A.J : ℝ) ^ 2 * A.K * β) := by
  unfold err
  refine add_le_add le_rfl (Bridge.lossPub_mono A.hρ le_rfl (A.lossArg_nonneg τ) ?_)
  exact mul_le_mul_of_nonneg_left ((A.pubB_le_crFinder _ τ extra).trans hB) (by positivity)

end Flock.Guarantees.LiveAt

namespace Verity

open Accounting Core

variable {ι α : Type} [Fintype α] {d : Device ι}

/-- **What the vocabulary's finder premise gives C-Flock at the deployment.** At registered files that parse, where
the vocabulary's sampled finder meets its premise (`Vocab.FinderCR`, at most the published `ε_cr` at the strategy's
calls), each of C-Flock's finders at the session's strategy succeeds at most `K ε_cr`, and the session's error is at
most `deltaPub + lossPub(J²·K·ε_cr)`: the vocabulary's finder is `crFinder` there (`SampledDeploy.finder_eq`). -/
theorem SampledDeploy.crAvg_of {V : Vocab d α} (Dp : SampledDeploy V) (θ F c : ByteArray)
    (σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed) (hr : V.parsed F = some r) (hreg : V.Registered θ F)
    (hcr : V.FinderCR .sampled θ F c σ) :
    (∀ i, ((Dp.A θ F c σ r hr).fam (Dp.τOf θ F c σ r hr) i).Pr Claims.Sampling.sha512Msg ≤
        (Dp.A θ F c σ r hr).K * V.crBound θ .sampled (V.cost θ F c σ)) ∧
      (Dp.A θ F c σ r hr).err (Dp.τOf θ F c σ r hr) ≤
        Flock.Bridge.deltaPub r.n (Dp.kd θ) (⌊(V.certOf θ).ε * r.n⌋₊ + 1) (Dp.A θ F c σ r hr).k
            (Dp.A θ F c σ r hr).M (Dp.A θ F c σ r hr).Rw (Flock.Bridge.ioSum (Dp.A θ F c σ r hr).P)
            (Dp.A θ F c σ r hr).ρ +
          Flock.Bridge.lossPub (Dp.A θ F c σ r hr).k (Flock.Bridge.ioSum (Dp.A θ F c σ r hr).P)
            (Dp.A θ F c σ r hr).ρ
            (((Dp.A θ F c σ r hr).J : ℝ) ^ 2 * (Dp.A θ F c σ r hr).K * V.crBound θ .sampled (V.cost θ F c σ)) := by
  have hB : ((Dp.A θ F c σ r hr).crFinder (Dp.τOf θ F c σ r hr)
      (V.replayFinder θ F c σ (Dp.col θ F c) (V.cost θ F c σ))).Pr Claims.Sampling.sha512Msg ≤
      V.crBound θ .sampled (V.cost θ F c σ) := by
    rw [sampling_sha512Msg_eq, ← Dp.finder_eq θ F c σ r hr hreg]
    exact hcr
  exact ⟨(Dp.A θ F c σ r hr).fam_le_of_crFinder _ _ hB, (Dp.A θ F c σ r hr).err_le_of_crFinder _ _ hB⟩

end Verity
