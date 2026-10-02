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
import PousProofs.Chain.Flat
import PousProofs.Chain.Family
import PousProofs.P2.Margin

/-!
# The proved pinned statements

`PousProofs.X : Pous.Pinned.X` for every statement of `Pous/Pinned.lean` and `Pous/P2/Pinned.lean` that has a proof.
`lean-audit.json` pins
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

/-! ## The overwrite chain (`PousProofs/Chain/Flat.lean`, and `Family.lean` for `BandMultiMeetsFamily`)

P3 with a concrete `H`, the band, the dense chain scheme and the public encoder, each over the trusted copies
(`Pous/Model/P3Chain.lean`, `BandChain.lean`, `PubEncoder.lean`, `Pous/Accounting/PubMeets.lean`). -/

theorem P3ConcreteMeets64 : Pous.Pinned.P3ConcreteMeets64 :=
  PousP3ConcreteMulti.p3_meets_concrete_64_D10_unconditional
theorem P3ConcreteMeets14GB : Pous.Pinned.P3ConcreteMeets14GB :=
  PousP3ConcreteMulti.p3_meets_concrete_multi_64_D10_unconditional
theorem P3ConcreteMeets64D12 : Pous.Pinned.P3ConcreteMeets64D12 :=
  PousP3ConcreteMulti.p3_meets_concrete_64_D12_band16_unconditional
theorem BandChainMeets64 : Pous.Pinned.BandChainMeets64 := Pous.BandChain.band_meets_64_final
theorem BandChainMeets64D5 : Pous.Pinned.BandChainMeets64D5 := Pous.BandChain.band_meets_64_d5_final
theorem BandChainMeets64D12 : Pous.Pinned.BandChainMeets64D12 := Pous.BandChain.band_meets_64_d12_final
theorem BandMultiMeets14 : Pous.Pinned.BandMultiMeets14 := Pous.BandMulti.band_meets_14_final
theorem BandMultiMeets14K106 : Pous.Pinned.BandMultiMeets14K106 := Pous.BandMulti.band_meets_14_k106_final
theorem BandMultiMeetsFamily : Pous.Pinned.BandMultiMeetsFamily := Pous.BandMulti.band_meets_family_final
theorem ChainDenseMeets64 : Pous.Pinned.ChainDenseMeets64 := PousSpongeChain.chain_meets_64_final
theorem BandPubMeets64 : Pous.Pinned.BandPubMeets64 := PousPub.band_pub_meets_64_final
theorem BandPubMeets14 : Pous.Pinned.BandPubMeets14 := PousPub.band_pub_meets_14_final
theorem BandPubMeets14Global : Pous.Pinned.BandPubMeets14Global := PousPub.band_pub_meets_14_global_final
theorem ChainDenseMeets111 : Pous.Pinned.ChainDenseMeets111 := PousSpongeChain.chainDenseMeets111_final

theorem P2MeetsM1pB19Uncond : Pous.Pinned.P2MeetsM1pB19Uncond := PousP2p.p2_meets_m1p_b19
theorem P2MeetsM1pFamilyUncond : Pous.Pinned.P2MeetsM1pFamilyUncond := PousP2p.family
theorem P2MeetsM1pW8192Uncond : Pous.Pinned.P2MeetsM1pW8192Uncond := PousP2p.p2_meets_m1p_8192

theorem P2SlackB19Uncond : Pous.Pinned.P2SlackB19Uncond := by
  intro b hsp hb D
  obtain ⟨-, h1, h2, h3, h64⟩ := PousMargin.b19_64 b hsp hb D
  exact ⟨h1, h2, h3, h64, (PousMargin.b19_128 b hsp hb D).2.2.2.2, (PousMargin.b19_256 b hsp hb D).2.2.2.2⟩

theorem P2SlackFamilyUncond : Pous.Pinned.P2SlackFamilyUncond := by
  intro B _ h0 h1 b hsp hb D
  obtain ⟨-, c1, c2, c3, a64⟩ := PousMargin.fam_64_96 B h0 h1 b hsp hb D
  exact ⟨c1, c2, c3, a64, (PousMargin.fam_128_105 B h0 h1 b hsp hb D).2.2.2.2,
    (PousMargin.fam_256_128 B h0 h1 b hsp hb D).2.2.2.2,
    fun h9 => (PousMargin.fam_128 B h9 h1 b hsp hb D).2.2.2.2,
    fun h10 => ⟨(PousMargin.fam_64 B h10 h1 b hsp hb D).2.2.2.2, (PousMargin.fam_256 B h10 h1 b hsp hb D).2.2.2.2⟩⟩

theorem P2ErrorSplitB19Uncond : Pous.Pinned.P2ErrorSplitB19Uncond := by
  intro b hsp hb D
  obtain ⟨-, h1, h2, h3, h0⟩ := PousMargin.b19_meets_129 b hsp hb D
  exact ⟨h1, h2, h3, h0, (PousMargin.b19_64_129 b hsp hb D).2.2.2.2, (PousMargin.b19_128_129 b hsp hb D).2.2.2.2,
    (PousMargin.b19_256_129 b hsp hb D).2.2.2.2⟩

end PousProofs
