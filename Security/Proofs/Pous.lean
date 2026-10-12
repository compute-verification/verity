import Proofs.Pous.Thm1.Theorem1iiTimed
import Proofs.Pous.TrackA.FibreCount
import Proofs.Pous.TrackA.A1Compression
import Proofs.Pous.TrackA.A3Prime
import Proofs.Pous.TrackA.A6Rederive
import Proofs.Pous.Pebbling.B1IndepSet
import Proofs.Pous.Pebbling.B3Chain
import Proofs.Pous.Dense.DenseMeets
import Proofs.Pous.FreeBlocks.Zero
import Proofs.Pous.FreeBlocks.P2
import Proofs.Pous.P3.InvertibleCore
import Proofs.Pous.P3.Refutation
import Proofs.Pous.Chain.Flat
import Proofs.Pous.Chain.Family
import Proofs.Pous.Chain.FamilyRounds
import Proofs.Pous.Continuous
import Proofs.Pous.SecureErasure
import Proofs.Pous.Graded.Band
import Proofs.Pous.P2.Margin
import Proofs.Pous.P2.Bridge
import Proofs.Pous.Accounting.Numbers
import Proofs.Pous.Audit.SampledAudit
import Proofs.Pous.B5.ExPostFacto
import Proofs.Pous.B5.StackedOneWay
import Proofs.Pous.Dense.DenseReCert
import Proofs.Pous.Dense.SegTagged
import Proofs.Pous.P2.Count
import Proofs.Pous.P2.Domain
import Proofs.Pous.P2.RingBasic
import Proofs.Pous.P2.RingTight
import Proofs.Pous.P2.RingTrunc
import Proofs.Pous.P2.Rounds
import Proofs.Pous.P2.RoundTests
import Proofs.Pous.P2.Rows
import Proofs.Pous.P3.Column.BandComb
import Proofs.Pous.P3.Column.BandTwoWay
import Proofs.Pous.P3.Column.BandWeights
import Proofs.Pous.P3.Column.TwoWay
import Proofs.Pous.Pebbling.B2Stacking
import Proofs.Pous.Pebbling.FischClaims1314
import Proofs.Pous.Sanity
import Proofs.Pous.Thm1.Theorem1iTimed
import Proofs.Pous.Thm1.Theorem1ii
import Proofs.Pous.Thm1.Theorem1iiSeq
import Proofs.Pous.Thm1.Theorem1iiiTimed
import Proofs.Pous.Chain.PubFamily
import Proofs.Pous.Chain.PubRounds
import Proofs.Pous.Rebuild.Exclusion
import Proofs.Pous.EndToEnd

/-!
# The proved guarantees

`Pous.SecurityProofs.X : Pous.Guarantees.X` for every statement of `Security/Models/Pous/Guarantees.lean`,
`Security/Models/Pous/Guarantees/P2.lean`, `Security/Models/Pous/Guarantees/SecureErasure.lean` (the frozen statements)
and `Security/Proofs/Pous/Statements.lean` (the lemmas) that has a proof. `lean-audit.json` pins these theorems, so a
change to a guarantee, or to a definition it reads, changes their records. The open statements (`A2LemmaAM1`,
`A5RabinGcd`, `A7Params`, `M1Meets`) have no entry here.

Two proofs were written against a prover's copy of a trusted recursive definition, and a bridge lemma shows the copy
equal to the trusted one: `PousPebbling.chain` (`Pous.HashChain.chain`) and `PousThm1.seqState`
(`Pous.Sampled.seqState`).
-/

namespace Pous.SecurityProofs

theorem KFloor : Pous.Guarantees.KFloor := by
  refine ⟨by norm_num [Pous.Params.ρ, Pous.Params.δ], fun k hk => ?_⟩
  by_contra h
  have hmono : Pous.Params.ρ ^ 85 ≤ Pous.Params.ρ ^ k :=
    pow_le_pow_of_le_one (by norm_num [Pous.Params.ρ]) (by norm_num [Pous.Params.ρ]) (by omega)
  have h85 : Pous.Params.δ < Pous.Params.ρ ^ 85 := by norm_num [Pous.Params.ρ, Pous.Params.δ]
  exact absurd (h85.trans_le (hmono.trans hk)) (lt_irrefl _)

theorem Theorem1ii : Pous.Guarantees.Theorem1ii := PousProofs.theorem1_ii
theorem Theorem1iiTimed : Pous.Guarantees.Theorem1iiTimed := PousProofs.theorem1_ii_timed
theorem Theorem1iTimed : Pous.Guarantees.Theorem1iTimed := @PousProofs.theorem1_i_timed
theorem Theorem1iiSeq : Pous.Guarantees.Theorem1iiSeq := @PousProofs.theorem1_ii_seq
theorem Theorem1iiiTimed : Pous.Guarantees.Theorem1iiiTimed := @PousProofs.theorem1_iii_timed

theorem FibreCount : Pous.Guarantees.FibreCount := @PousProofs.fibre_count
theorem A1CompressionCard : Pous.Guarantees.A1CompressionCard := @PousProofs.a1_compression_card
theorem A1CompressionPr : Pous.Guarantees.A1CompressionPr := @PousProofs.a1_compression_pr
theorem A3Prime : Pous.Guarantees.A3Prime := @PousTrackA.a3_prime
theorem A3PrimeLam : Pous.Guarantees.A3PrimeLam := @PousTrackA.a3_prime_lam
theorem A6RederiveUntimed : Pous.Guarantees.A6RederiveUntimed := @PousProofs.a6_rederive_untimed
theorem A6RederiveTimed : Pous.Guarantees.A6RederiveTimed := @PousProofs.a6_rederive_timed

theorem B1IndepSet : Pous.Guarantees.B1IndepSet := @PousPebbling.b1_indep_set
theorem B2Stacking : Pous.Guarantees.B2Stacking := @PousB2.b2_stacking

/-- `PousPebbling.chain` is `Pous.HashChain.chain`, the trusted chain of the guarantee. -/
theorem chain_eq_pinned {m D : ℕ} (H : Fin (D + 1) × Fin m → Fin m) (x₀ : Fin m) :
    ∀ i, Pous.HashChain.chain H x₀ i = PousPebbling.chain H x₀ i
  | 0 => rfl
  | i + 1 => by simp only [Pous.HashChain.chain, PousPebbling.chain, chain_eq_pinned H x₀ i]

theorem B3Chain : Pous.Guarantees.B3Chain := by
  intro m D q _ x₀ P hP
  simpa only [chain_eq_pinned] using PousPebbling.b3_chain m D q x₀ P hP

theorem FischClaim13 : Pous.Guarantees.FischClaim13 := @PousFisch.claim13
theorem FischClaim14 : Pous.Guarantees.FischClaim14 := @PousFisch.claim14
theorem B5ExPostFactoBounded : Pous.Guarantees.B5ExPostFactoBounded := PousB5.b5_ex_post_facto_bounded
theorem B5ExPostFacto : Pous.Guarantees.B5ExPostFacto := PousB5.b5_ex_post_facto

theorem TransposeMixing : Pous.Guarantees.TransposeMixing := @PousInvertible.transpose_mixing
theorem ReverseNeedsAllChildren : Pous.Guarantees.ReverseNeedsAllChildren := @PousInvertible.reverse_needs_all_children
theorem N10Refuted : Pous.Guarantees.N10Refuted := PousP3Forward.not_invExPostFactoTwoLayer

theorem SampledAuditProd : Pous.Guarantees.SampledAuditProd := @PousAudit.sampled_audit_prod

/-- `PousThm1.seqState` is `Pous.Sampled.seqState`, the trusted definition the guarantee reads. -/
theorem seqState_eq_pinned {M : Pous.IdealModel} {ℓ S B : ℕ} (f : M.Qry → M.Ans)
    (step : ℕ → Pous.Bits S → Fin B → Pous.Prog M (Pous.Bits ℓ × Pous.Bits S)) :
    ∀ (j : ℕ) (h : List (Fin B)) (σ : Pous.Bits S),
      Pous.Sampled.seqState f step j h σ = PousThm1.seqState f step j h σ
  | _, [], _ => rfl
  | j, i :: is, σ => by
      simp only [Pous.Sampled.seqState, PousThm1.seqState, seqState_eq_pinned f step (j + 1) is]

theorem SampledAuditSeq : Pous.Guarantees.SampledAuditSeq := by
  intro M sch S D Q t k p W A₁ A₂ h
  exact PousAudit.sampled_audit_seq sch S D Q t k p W A₁ A₂ (by simpa only [seqState_eq_pinned] using h)

theorem SampledAuditSim : Pous.Guarantees.SampledAuditSim := @PousAudit.sampled_audit_sim
theorem StackedAuditSeq : Pous.Guarantees.StackedAuditSeq := @PousAudit.stacked_audit_seq
theorem StackedAuditOnePercent : Pous.Guarantees.StackedAuditOnePercent := @PousAudit.stacked_audit_one_percent
theorem DenseMeets : Pous.Guarantees.DenseMeets := PousDense.dense_meets

/-! ## The overwrite chain (`Security/Proofs/Pous/Chain/Flat.lean`, `Family.lean` for `BandMultiMeetsFamily`, and
`FamilyRounds.lean` for the band at `D = 2`, `D = 1` and `D = 0`)

P3 with a concrete `H`, the band, the dense chain scheme and the public encoder, each over the trusted copies
(`Security/Definitions/Pous/Model/P3Chain.lean`, `BandChain.lean`, `PubEncoder.lean`,
`Security/Definitions/Pous/Accounting/PubMeets.lean`). -/

theorem P3ConcreteMeets64 : Pous.Guarantees.P3ConcreteMeets64 :=
  PousP3ConcreteMulti.p3_meets_concrete_64_D10_unconditional
theorem P3ConcreteMeets14GB : Pous.Guarantees.P3ConcreteMeets14GB :=
  PousP3ConcreteMulti.p3_meets_concrete_multi_64_D10_unconditional
theorem P3ConcreteMeets64D12 : Pous.Guarantees.P3ConcreteMeets64D12 :=
  PousP3ConcreteMulti.p3_meets_concrete_64_D12_band16_unconditional
theorem BandChainMeets64 : Pous.Guarantees.BandChainMeets64 := Pous.BandChain.band_meets_64_final
theorem BandChainMeets64D5 : Pous.Guarantees.BandChainMeets64D5 := Pous.BandChain.band_meets_64_d5_final
theorem BandChainMeets64D12 : Pous.Guarantees.BandChainMeets64D12 := Pous.BandChain.band_meets_64_d12_final
theorem BandMultiMeets14 : Pous.Guarantees.BandMultiMeets14 := Pous.BandMulti.band_meets_14_final
theorem BandMultiMeets14K106 : Pous.Guarantees.BandMultiMeets14K106 := Pous.BandMulti.band_meets_14_k106_final
theorem BandMultiMeetsFamily : Pous.Guarantees.BandMultiMeetsFamily := Pous.BandMulti.band_meets_family_final
theorem ChainDenseMeets64 : Pous.Guarantees.ChainDenseMeets64 := PousSpongeChain.chain_meets_64_final
theorem BandMultiMeetsFamilyD2 : Pous.Guarantees.BandMultiMeetsFamilyD2 := Pous.BandMulti.Rounds.band_meets_family_d2
theorem BandMultiMeetsFamilyD1 : Pous.Guarantees.BandMultiMeetsFamilyD1 := Pous.BandMulti.Rounds.band_meets_family_d1
theorem BandMultiMeetsFamilyD0 : Pous.Guarantees.BandMultiMeetsFamilyD0 := Pous.BandMulti.Rounds.band_meets_family_d0
theorem BandChainMeets64Rounds2 : Pous.Guarantees.BandChainMeets64Rounds2 := Pous.BandMulti.Rounds.band_meets_64_d2
theorem BandChainMeets64Rounds1 : Pous.Guarantees.BandChainMeets64Rounds1 := Pous.BandMulti.Rounds.band_meets_64_d1
theorem BandPubMeets64 : Pous.Guarantees.BandPubMeets64 := PousPub.band_pub_meets_64_final
theorem BandPubMeets14 : Pous.Guarantees.BandPubMeets14 := PousPub.band_pub_meets_14_final
theorem BandPubMeets14Global : Pous.Guarantees.BandPubMeets14Global := PousPub.band_pub_meets_14_global_final
theorem BandPubMeetsFamily : Pous.Guarantees.BandPubMeetsFamily := PousPub.band_pub_meets_family_final
theorem BandPubMeetsFamilyD2 : Pous.Guarantees.BandPubMeetsFamilyD2 := PousPub.band_pub_meets_family_d2
theorem BandPubMeetsFamilyD1 : Pous.Guarantees.BandPubMeetsFamilyD1 := PousPub.band_pub_meets_family_d1
theorem ChainDenseMeets111 : Pous.Guarantees.ChainDenseMeets111 := PousSpongeChain.chainDenseMeets111_final
theorem BandGradedMeetsFamily : Pous.Guarantees.BandGradedMeetsFamily := PousGraded.band_graded_meets_family
theorem PerAnswer : Pous.Guarantees.PerAnswer := PousContinuous.per_answer
theorem ContinuousAudit : Pous.Guarantees.ContinuousAudit := PousContinuous.continuous_audit
theorem BackgroundRebuildExcluded : Pous.Guarantees.BackgroundRebuildExcluded :=
  PousRebuild.background_rebuild_excluded
theorem ContinuousComplete : Pous.Guarantees.ContinuousComplete := PousContinuous.continuous_complete
theorem ContinuousTimeAverage : Pous.Guarantees.ContinuousTimeAverage := PousContinuous.continuous_time_average
theorem BandFamilyContinuousTimeAverage : Pous.Guarantees.BandFamilyContinuousTimeAverage :=
  PousContinuous.band_family_continuous_time_average
theorem DigestAU : Pous.Guarantees.DigestAU := PousDigest.digestAU
theorem SecureErasureMeets : Pous.Guarantees.SecureErasureMeets := PousErase.secure_erasure_meets
theorem SecureErasureMeetsFamily : Pous.Guarantees.SecureErasureMeetsFamily := PousErase.secure_erasure_meets_family
theorem SecureErasureMeets96GiB : Pous.Guarantees.SecureErasureMeets96GiB := PousErase.secure_erasure_meets_96GiB

theorem P2MeetsM1pB19Uncond : Pous.Guarantees.P2MeetsM1pB19Uncond := PousP2p.p2_meets_m1p_b19
theorem P2MeetsM1pFamilyUncond : Pous.Guarantees.P2MeetsM1pFamilyUncond := PousP2p.family
theorem P2MeetsM1pW8192Uncond : Pous.Guarantees.P2MeetsM1pW8192Uncond := PousP2p.p2_meets_m1p_8192

theorem P2SlackB19Uncond : Pous.Guarantees.P2SlackB19Uncond := by
  intro b hsp hb D
  obtain ⟨-, h1, h2, h3, h64⟩ := PousMargin.b19_64 b hsp hb D
  exact ⟨h1, h2, h3, h64, (PousMargin.b19_128 b hsp hb D).2.2.2.2, (PousMargin.b19_256 b hsp hb D).2.2.2.2⟩

theorem P2SlackFamilyUncond : Pous.Guarantees.P2SlackFamilyUncond := by
  intro B _ h0 h1 b hsp hb D
  obtain ⟨-, c1, c2, c3, a64⟩ := PousMargin.fam_64_96 B h0 h1 b hsp hb D
  exact ⟨c1, c2, c3, a64, (PousMargin.fam_128_105 B h0 h1 b hsp hb D).2.2.2.2,
    (PousMargin.fam_256_128 B h0 h1 b hsp hb D).2.2.2.2,
    fun h9 => (PousMargin.fam_128 B h9 h1 b hsp hb D).2.2.2.2,
    fun h10 => ⟨(PousMargin.fam_64 B h10 h1 b hsp hb D).2.2.2.2, (PousMargin.fam_256 B h10 h1 b hsp hb D).2.2.2.2⟩⟩

theorem P2ErrorSplitB19Uncond : Pous.Guarantees.P2ErrorSplitB19Uncond := by
  intro b hsp hb D
  obtain ⟨-, h1, h2, h3, h0⟩ := PousMargin.b19_meets_129 b hsp hb D
  exact ⟨h1, h2, h3, h0, (PousMargin.b19_64_129 b hsp hb D).2.2.2.2, (PousMargin.b19_128_129 b hsp hb D).2.2.2.2,
    (PousMargin.b19_256_129 b hsp hb D).2.2.2.2⟩

theorem P2SlackFamilyFreeBlocks : Pous.Guarantees.P2SlackFamilyFreeBlocks := Pous.FreeBlocks.p2_slack_family_free_blocks

theorem P2RingAuditSecureOfReduces : Pous.Guarantees.P2RingAuditSecureOfReduces := by
  intro D Q L h W A₁ A₂
  obtain ⟨W', D', A₁', A₂', hle⟩ := h W A₁ A₂
  have h104 := (P2ErrorSplitB19Uncond 16432 (by norm_num) (by norm_num) D').2.2.2.2.2.1 W' A₁' A₂'
  calc pr (seqAcceptsOn _ 105 W A₁ A₂)
      ≤ pr (seqAccepts _ 105 W' A₁' A₂') + (2⁻¹ : ENNReal) ^ 130 := hle
    _ ≤ pr (seqAccepts _ 104 W' A₁' A₂') + (2⁻¹ : ENNReal) ^ 130 :=
        add_le_add (PousP2Bridge.pr_seqAccepts_le _ (by norm_num) W' A₁' A₂') le_rfl
    _ ≤ ENNReal.ofReal Params.δ + (2⁻¹ : ENNReal) ^ 129 + (2⁻¹ : ENNReal) ^ 130 := add_le_add h104 le_rfl
    _ = _ := add_assoc _ _ _

theorem P2RingAuditSecureOfRoundSecure : Pous.Guarantees.P2RingAuditSecureOfRoundSecure :=
  PousP2Rounds.auditSecure_of_roundSecure
theorem P2RingGradedOfRoundSecure : Pous.Guarantees.P2RingGradedOfRoundSecure := PousP2Rounds.graded

/-! ## Proved results cited outside the package (`Security/Models/Pous/Guarantees.lean`, `Statements.lean`) -/

theorem B5Stacked : Pous.Guarantees.B5Stacked := @PousBridge.b5_stacked
theorem DenseMeets64 : Pous.Guarantees.DenseMeets64 := PousDenseReCert.dense_meets_64
theorem SegMeets14 : Pous.Guarantees.SegMeets14 := PousDenseReCert.seg_meets_14
theorem SegTagMeets14 : Pous.Guarantees.SegTagMeets14 := @PousDenseReCert.segTag_meets_14
theorem SegTagAuditSecure : Pous.Guarantees.SegTagAuditSecure := @PousDenseReCert.segTag_auditSecure
theorem DomainSepEquiv : Pous.Guarantees.DomainSepEquiv := @PousDenseReCert.domainSep_equiv
theorem TagInjective14 : Pous.Guarantees.TagInjective14 := @PousDenseReCert.tag_injective_14
theorem BandTwoWay220_12 : Pous.Guarantees.BandTwoWay220_12 := PousP3Column.band_twoWay_220_12
theorem BandTwoWay220_12D11 : Pous.Guarantees.BandTwoWay220_12D11 := PousP3Column.band_twoWay_220_12_d11
theorem BandTwoWay220_16 : Pous.Guarantees.BandTwoWay220_16 := PousP3Column.band_twoWay_220_16

end Pous.SecurityProofs
