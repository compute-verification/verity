import Definitions.Claims.Sampling

/-! **The confidence set is closed downward** at a one-class uniform `kd`-subset design (`d.rules = [⟨.subset kd,
none⟩]`, as `designOf kd` is): a subset of a set in `M_δ` is in `M_δ`, since dropping members of `B` only raises
`P_a(B) = C(n − |B|, kd)/C(n, kd)`. `RelS` is `faulty ∈ M_δ`, so where one commitment's faulty set holds another's,
`RelS` failing at the smaller fails it at the larger. -/

namespace Claims.Sampling

theorem window_mono {o n : ℕ} {B B' : Finset ℕ} (h : B' ⊆ B) : window o n B' ⊆ window o n B :=
  Finset.image_subset_image (Finset.filter_subset_filter _ h)

theorem classesPa_nil_right (rs : List Rule) (o : ℕ) (B : Finset ℕ) : classesPa rs [] o B = 1 := by
  cases rs <;> rfl

theorem Rule.Pa_subset_none (kd : ℕ) (ns sz : List ℕ) (B : Finset ℕ) :
    (⟨.subset kd, none⟩ : Rule).Pa ns sz B = ((ns.sum - B.card).choose kd : ℝ) / (ns.sum.choose kd : ℝ) := rfl

/-- **`P_a` is antitone** at a one-class uniform `kd`-subset design. -/
theorem Design.Pa_anti_subset {d : Design} {kd : ℕ} (hd : d.rules = [⟨.subset kd, none⟩]) (P : Partition)
    {B B' : Finset ℕ} (h : B' ⊆ B) : d.Pa P B ≤ d.Pa P B' := by
  obtain ⟨_ | ⟨c, cs⟩⟩ := P
  · simp only [Design.Pa, classesPa_nil_right, le_refl]
  · simp only [Design.Pa, hd, classesPa, Rule.Pa_subset_none, mul_one]
    refine div_le_div_of_nonneg_right ?_ (Nat.cast_nonneg _)
    exact_mod_cast Nat.choose_le_choose kd (Nat.sub_le_sub_left (Finset.card_le_card (window_mono h)) _)

/-- **`M_δ` is closed downward** at a one-class uniform `kd`-subset design. -/
theorem Design.M_down_subset {d : Design} {kd : ℕ} (hd : d.rules = [⟨.subset kd, none⟩]) (P : Partition) (δ : ℝ)
    {B B' : Finset ℕ} (h : B' ⊆ B) (hB : B ∈ d.M P δ) : B' ∈ d.M P δ :=
  lt_of_lt_of_le hB (Design.Pa_anti_subset hd P h)

end Claims.Sampling
