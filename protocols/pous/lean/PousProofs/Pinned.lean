import PousProofs.Thm1.Theorem1iiTimed
import PousProofs.TrackA.FibreCount
import PousProofs.TrackA.A1Compression
import PousProofs.TrackA.A3Prime
import PousProofs.TrackA.A6Rederive
import PousProofs.Pebbling.B1IndepSet
import PousProofs.Pebbling.B3Chain
import PousProofs.Dense.DenseMeets
import PousProofs.P3.InvertibleCore
import PousProofs.P3.Refutation

/-!
# The proved pinned statements

`PousProofs.X : Pous.Pinned.X` for every statement of `Pous/Pinned.lean` that has a proof. `lean-audit.json` pins
these theorems, so a change to a pinned statement, or to a definition it reads, changes their records. The open
statements (`A2LemmaAM1`, `A5RabinGcd`, `A7Params`, `M1Meets`) have no entry here.

Two proofs were written against a prover's copy of a trusted recursive definition, and a bridge lemma shows the copy
equal to the trusted one: `PousPebbling.chain` (`Pous.HashChain.chain`) and `PousThm1.seqState`
(`Pous.Sampled.seqState`).
-/

namespace PousProofs

theorem KFloor : Pous.Pinned.KFloor := by
  refine ⟨by norm_num [Pous.Params.ρ, Pous.Params.δ], fun k hk => ?_⟩
  by_contra h
  have hmono : Pous.Params.ρ ^ 85 ≤ Pous.Params.ρ ^ k :=
    pow_le_pow_of_le_one (by norm_num [Pous.Params.ρ]) (by norm_num [Pous.Params.ρ]) (by omega)
  have h85 : Pous.Params.δ < Pous.Params.ρ ^ 85 := by norm_num [Pous.Params.ρ, Pous.Params.δ]
  exact absurd (h85.trans_le (hmono.trans hk)) (lt_irrefl _)

theorem Theorem1ii : Pous.Pinned.Theorem1ii := theorem1_ii
theorem Theorem1iiTimed : Pous.Pinned.Theorem1iiTimed := theorem1_ii_timed
theorem Theorem1iTimed : Pous.Pinned.Theorem1iTimed := @theorem1_i_timed
theorem Theorem1iiSeq : Pous.Pinned.Theorem1iiSeq := @theorem1_ii_seq
theorem Theorem1iiiTimed : Pous.Pinned.Theorem1iiiTimed := @theorem1_iii_timed

theorem FibreCount : Pous.Pinned.FibreCount := @fibre_count
theorem A1CompressionCard : Pous.Pinned.A1CompressionCard := @a1_compression_card
theorem A1CompressionPr : Pous.Pinned.A1CompressionPr := @a1_compression_pr
theorem A3Prime : Pous.Pinned.A3Prime := @PousTrackA.a3_prime
theorem A3PrimeLam : Pous.Pinned.A3PrimeLam := @PousTrackA.a3_prime_lam
theorem A6RederiveUntimed : Pous.Pinned.A6RederiveUntimed := @a6_rederive_untimed
theorem A6RederiveTimed : Pous.Pinned.A6RederiveTimed := @a6_rederive_timed

theorem B1IndepSet : Pous.Pinned.B1IndepSet := @PousPebbling.b1_indep_set
theorem B2Stacking : Pous.Pinned.B2Stacking := @PousB2.b2_stacking

/-- `PousPebbling.chain` is `Pous.HashChain.chain`, the trusted chain of the pinned statement. -/
theorem chain_eq_pinned {m D : ℕ} (H : Fin (D + 1) × Fin m → Fin m) (x₀ : Fin m) :
    ∀ i, Pous.HashChain.chain H x₀ i = PousPebbling.chain H x₀ i
  | 0 => rfl
  | i + 1 => by simp only [Pous.HashChain.chain, PousPebbling.chain, chain_eq_pinned H x₀ i]

theorem B3Chain : Pous.Pinned.B3Chain := by
  intro m D q _ x₀ P hP
  simpa only [chain_eq_pinned] using PousPebbling.b3_chain m D q x₀ P hP

theorem FischClaim13 : Pous.Pinned.FischClaim13 := @PousFisch.claim13
theorem FischClaim14 : Pous.Pinned.FischClaim14 := @PousFisch.claim14
theorem B5ExPostFactoBounded : Pous.Pinned.B5ExPostFactoBounded := PousB5.b5_ex_post_facto_bounded
theorem B5ExPostFacto : Pous.Pinned.B5ExPostFacto := PousB5.b5_ex_post_facto

theorem TransposeMixing : Pous.Pinned.TransposeMixing := @PousInvertible.transpose_mixing
theorem ReverseNeedsAllChildren : Pous.Pinned.ReverseNeedsAllChildren := @PousInvertible.reverse_needs_all_children
theorem N10Refuted : Pous.Pinned.N10Refuted := PousP3Forward.not_invExPostFactoTwoLayer

theorem SampledAuditProd : Pous.Pinned.SampledAuditProd := @PousAudit.sampled_audit_prod

/-- `PousThm1.seqState` is `Pous.Sampled.seqState`, the trusted definition the pinned statement reads. -/
theorem seqState_eq_pinned {M : Pous.IdealModel} {ℓ S B : ℕ} (f : M.Qry → M.Ans)
    (step : ℕ → Pous.Bits S → Fin B → Pous.Prog M (Pous.Bits ℓ × Pous.Bits S)) :
    ∀ (j : ℕ) (h : List (Fin B)) (σ : Pous.Bits S),
      Pous.Sampled.seqState f step j h σ = PousThm1.seqState f step j h σ
  | _, [], _ => rfl
  | j, i :: is, σ => by
      simp only [Pous.Sampled.seqState, PousThm1.seqState, seqState_eq_pinned f step (j + 1) is]

theorem SampledAuditSeq : Pous.Pinned.SampledAuditSeq := by
  intro M sch S D Q t k p W A₁ A₂ h
  exact PousAudit.sampled_audit_seq sch S D Q t k p W A₁ A₂ (by simpa only [seqState_eq_pinned] using h)

theorem SampledAuditSim : Pous.Pinned.SampledAuditSim := @PousAudit.sampled_audit_sim
theorem StackedAuditSeq : Pous.Pinned.StackedAuditSeq := @PousAudit.stacked_audit_seq
theorem StackedAuditOnePercent : Pous.Pinned.StackedAuditOnePercent := @PousAudit.stacked_audit_one_percent
theorem DenseMeets : Pous.Pinned.DenseMeets := PousDense.dense_meets

end PousProofs
