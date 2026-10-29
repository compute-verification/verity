import Proofs.Pous.Thm1.Theorem1iiSeq
import Proofs.Pous.Thm1.Theorem1iiiTimed

/-!
# Secure erasure from the timed audit

`Pous.Guarantees.SecureErasureMeets`, by counting. For a round `j`, the memories from which `keep j` reads the foreign
content `z` are disjoint as `z` varies, so few `z` have more than `2^S'` of them (`pr_big_fibre_le`). For every other
`z` those memories have an `S'`-bit code (`PousThm1.exists_code`), and the prover run on the codes is an `S'`-bit
sequential responder with the same programs (`seqAnswers_sim`), which `AuditSecure` bounds.
-/

namespace PousErase

open Pous PousThm1
open scoped ENNReal

variable {M : IdealModel}

theorem card_bits (n : ℕ) : Fintype.card (Bits n) = 2 ^ n := by simp [Bits]

/-- Fewer than `|X| / c` points have a fibre of more than `c` points. -/
theorem card_big_fibres_le {X Z : Type} [Fintype X] [Fintype Z] [DecidableEq Z] (keep : X → Z) (c : ℕ) :
    (Finset.univ.filter fun z => c < (Finset.univ.filter fun x => keep x = z).card).card * c ≤ Fintype.card X := by
  set G := Finset.univ.filter fun z => c < (Finset.univ.filter fun x => keep x = z).card
  calc G.card * c = ∑ _z ∈ G, c := by rw [Finset.sum_const, smul_eq_mul]
    _ ≤ ∑ z ∈ G, (Finset.univ.filter fun x => keep x = z).card :=
        Finset.sum_le_sum fun z hz => (Finset.mem_filter.1 hz).2.le
    _ ≤ ∑ z, (Finset.univ.filter fun x => keep x = z).card :=
        Finset.sum_le_sum_of_subset (Finset.subset_univ _)
    _ = Fintype.card X :=
        (Finset.card_eq_sum_card_fiberwise fun x _ => Finset.mem_univ (keep x)).symm

/-- A uniform `z` has a fibre of more than `c ≠ 0` points with probability at most `|X| / (c · |Z|)`. -/
theorem pr_big_fibre_le {X Z : Type} [Fintype X] [Fintype Z] [DecidableEq Z] (keep : X → Z) (c : ℕ) (hc : c ≠ 0) :
    pr (fun z => c < (Finset.univ.filter fun x => keep x = z).card) ≤
      (Fintype.card X : ℝ≥0∞) / (c * Fintype.card Z) := by
  classical
  rw [pr_def]
  have hc0 : (c : ℝ≥0∞) ≠ 0 := by exact_mod_cast hc
  rw [← ENNReal.mul_div_mul_right _ _ hc0 (ENNReal.natCast_ne_top c), mul_comm (Fintype.card Z : ℝ≥0∞)]
  gcongr
  exact_mod_cast card_big_fibres_le keep c

/-- **The prover on codes.** If `d j (e j σ) = σ` for every memory `σ` that gives `z` back at a round `j < s`, then on
any rounds where the prover keeps `z`, the responder that decodes its memory, runs the prover's step and encodes the
next memory returns the prover's blocks. -/
theorem seqAnswers_sim {ℓ S S' S_φ B : ℕ} (f : M.Qry → M.Ans) (step : ℕ → Bits S → Fin B → Prog M (Bits ℓ × Bits S))
    (keep : ℕ → Bits S → Bits S_φ) (z : Bits S_φ) (s : ℕ) (e : ℕ → Bits S → Bits S') (d : ℕ → Bits S' → Bits S)
    (hed : ∀ j < s, ∀ σ, keep j σ = z → d j (e j σ) = σ) (is : List (Fin B)) :
    ∀ (j : ℕ) (σ : Bits S), j + is.length ≤ s → (∀ a ∈ eraseAnswers f step keep j is σ, a.2 = z) →
      seqAnswers f (fun j τ i => progMap (fun out => (out.1, e (j + 1) out.2)) (step j (d j τ) i)) j is (e j σ) =
        (eraseAnswers f step keep j is σ).map Prod.fst := by
  induction is with
  | nil => intro _ _ _ _; rfl
  | cons i is ih =>
    intro j σ hlen hz
    have hkeep : keep j σ = z := hz _ List.mem_cons_self
    have hdj : d j (e j σ) = σ := hed j (by simp only [List.length_cons] at hlen; omega) σ hkeep
    simp only [seqAnswers, eraseAnswers, List.map_cons, run_progMap, hdj]
    exact congrArg _ (ih (j + 1) _ (by simp only [List.length_cons] at hlen; omega)
      fun a ha => hz a (List.mem_cons_of_mem _ ha))

theorem secure_erasure_meets : Pous.Guarantees.SecureErasureMeets := by
  intro M sch K V _ _ dig S S' S_φ D Q s δ ε hsec W P
  classical
  have : Nonempty (Fin sch.B) := ⟨⟨0, sch.B_pos⟩⟩
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
  have hgood : ∀ z, ¬ bad z → pr (fun y : K × Outcome sch s => eraseWins sch dig s W P (z, y)) ≤ δ + ε := by
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
    have hκ : ∀ κ : K, pr (fun o : Outcome sch s => eraseWins sch dig s W P (z, κ, o)) ≤ δ + ε := by
      intro κ
      let A₁ : Preprocessor sch S' := fun W pp C ω => e 0 (P.fill z W pp C ω κ).2
      let A₂ : SeqResponder sch S' D Q :=
        { step := fun W pp j τ i => progMap (fun out => (out.1, e (j + 1) out.2)) (P.read.step W pp j (d j τ) i)
          bounded := fun W pp j τ i => bounded_progMap _ (P.read.bounded W pp j (d j τ) i) }
      refine (pr_mono fun o hwin => ?_).trans (hsec W A₁ A₂)
      obtain ⟨_, hans, hkept⟩ := hwin
      have h := seqAnswers_sim (M.answer o.1) (P.read.step W (sch.enc W o.1 o.2.1).1) P.keep z s e d hed
        (List.ofFn o.2.2) 0 (P.fill z W (sch.enc W o.1 o.2.1).1 (sch.enc W o.1 o.2.1).2 o.1 κ).2 (by simp) hkept
      exact h.trans hans
    have := pr_prod_le (fun (κ : K) (o : Outcome sch s) => eraseWins sch dig s W P (z, κ, o))
      (fun _ => False) (δ + ε) fun κ _ => hκ κ
    rwa [pr_false', zero_add] at this
  have h : pr (eraseWins sch dig s W P) ≤ pr bad + (δ + ε) :=
    pr_prod_le (fun (z : Bits S_φ) (y : K × Outcome sch s) => eraseWins sch dig s W P (z, y)) bad (δ + ε) hgood
  rw [add_comm] at h
  exact h.trans (add_le_add_right hbad _)

end PousErase
