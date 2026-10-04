import Pous.SecurityProofs.SecureErasure.Reduction

/-!
# Secure erasure from the sequential audit, over any read draw

`Pous.Guarantees.SecureErasureMeetsDraw`: `secure_erasure_meets` with the read indices `rd r` for a uniform `r : R`.
The proof is that one, line for line: it never looks at how the indices are drawn. `auditSecure_iff_draw` shows the
draw `id` on `Fin s → Fin B` is the with-replacement audit.
-/

namespace PousErase

open Pous PousThm1
open scoped ENNReal

variable {M : IdealModel}

/-- The with-replacement sequential audit is the audit over the draw `id`. -/
theorem auditSecure_iff_draw (sch : Scheme M) (S D Q s : ℕ) (δ ε : ℝ≥0∞) :
    AuditSecure sch .sequential S D Q s δ ε ↔ AuditSecureDraw sch (id : (Fin s → Fin sch.B) → _) S D Q δ ε :=
  Iff.rfl

theorem secure_erasure_meets_draw : Pous.Guarantees.SecureErasureMeetsDraw := by
  intro M sch K V _ _ dig R _ _ s rd S S' S_φ D Q δ ε hsec W P
  classical
  set bad : Bits S_φ → Prop := fun z =>
    ∃ j ∈ Finset.range s, 2 ^ S' < (Finset.univ.filter fun σ => P.keep j σ = z).card
  have hbad : pr bad ≤ (s : ℝ≥0∞) * 2 ^ S / (2 ^ S' * 2 ^ S_φ) := by
    refine (pr_exists_le _ _).trans ?_
    calc ∑ j ∈ Finset.range s, pr (fun z => 2 ^ S' < (Finset.univ.filter fun σ => P.keep j σ = z).card)
        ≤ ∑ _j ∈ Finset.range s, (2 : ℝ≥0∞) ^ S / (2 ^ S' * 2 ^ S_φ) := by
          refine Finset.sum_le_sum fun j _ => ?_
          have := pr_big_fibre_le (P.keep j) (2 ^ S') (by positivity)
          simpa only [card_bits, Nat.cast_pow, Nat.cast_ofNat] using this
      _ = (s : ℝ≥0∞) * 2 ^ S / (2 ^ S' * 2 ^ S_φ) := by
          rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_div_assoc]
  have hgood : ∀ z, ¬ bad z → pr (fun y : K × OutcomeDraw sch R => eraseWinsDraw sch rd dig W P (z, y)) ≤ δ + ε := by
    intro z hz
    have hcode : ∀ j, ∃ (e : Bits S → Bits S') (d : Bits S' → Bits S),
        j < s → ∀ σ, P.keep j σ = z → d (e σ) = σ := by
      intro j
      by_cases hj : j < s
      · have hle : (Finset.univ.filter fun σ => P.keep j σ = z).card ≤ 2 ^ S' :=
          not_lt.1 fun h => hz ⟨j, Finset.mem_range.2 hj, h⟩
        obtain ⟨e, d, hed⟩ := exists_code _ S' hle
        exact ⟨e, d, fun _ σ hσ => hed σ (Finset.mem_filter.2 ⟨Finset.mem_univ _, hσ⟩)⟩
      · exact ⟨fun _ => fun _ => false, fun _ => fun _ => false, fun h => absurd h hj⟩
    choose e d hed using hcode
    have hκ : ∀ κ : K, pr (fun o : OutcomeDraw sch R => eraseWinsDraw sch rd dig W P (z, κ, o)) ≤ δ + ε := by
      intro κ
      let A₁ : Preprocessor sch S' := fun W pp C ω => e 0 (P.fill z W pp C ω κ).2
      let A₂ : SeqResponder sch S' D Q :=
        { step := fun W pp j τ i => progMap (fun out => (out.1, e (j + 1) out.2)) (P.read.step W pp j (d j τ) i)
          bounded := fun W pp j τ i => bounded_progMap _ (P.read.bounded W pp j (d j τ) i) }
      refine (pr_mono fun o hwin => ?_).trans (hsec W A₁ A₂)
      obtain ⟨_, hans, hkept⟩ := hwin
      have h := seqAnswers_sim (M.answer o.1) (P.read.step W (sch.enc W o.1 o.2.1).1) P.keep z s e d hed
        (List.ofFn (rd o.2.2)) 0 (P.fill z W (sch.enc W o.1 o.2.1).1 (sch.enc W o.1 o.2.1).2 o.1 κ).2 (by simp)
        hkept
      exact h.trans hans
    have := pr_prod_le (fun (κ : K) (o : OutcomeDraw sch R) => eraseWinsDraw sch rd dig W P (z, κ, o))
      (fun _ => False) (δ + ε) fun κ _ => hκ κ
    rwa [pr_false', zero_add] at this
  have h : pr (eraseWinsDraw sch rd dig W P) ≤ pr bad + (δ + ε) :=
    pr_prod_le (fun (z : Bits S_φ) (y : K × OutcomeDraw sch R) => eraseWinsDraw sch rd dig W P (z, y)) bad (δ + ε)
      hgood
  rw [add_comm] at h
  exact h.trans (add_le_add_right hbad _)

end PousErase
