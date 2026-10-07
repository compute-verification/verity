import Models.Pouw.Assumptions.Dimension.H100Frag
import Proofs.Pouw.Dimension.H100DenseProofs

/-!
# Computed fragments in the k8 lemma: proofs

* **The reduction** (`accInOfFrag`): a cheap register's increment is determined by at most `K` multiplicands, each
  determined by at most `q` lanes, so it is determined by the union of their lanes, at most `K·q`.
* **The bound** (`denseWritesFrag`): `denseWritesLegacy` at `D = LaneFns (K·q) β S p`.
* **The counterexample** (`fragShareGain`): `fragProg m n` writes the sum of the coordinates once, as an `other`
  register, and then `(i + 1)` times it with one cheap instruction each.
-/

namespace Pouw.Dimension.H100

open Finset Pouw.Dimension Pouw.Dimension.Embed

theorem accInOfFrag : AccInOfFrag := by
  intro G Λ S p F W a K q β fpSem P L hW
  classical
  choose J hJc hJd using hW
  let Jt : (G → ℚ) → Finset Λ := fun w => if h : w ∈ W then J w h else ∅
  have hJt_card : ∀ w, (Jt w).card ≤ q := by
    intro w
    by_cases h : w ∈ W
    · simp only [Jt, dite_eq_left h]
      exact hJc w h
    · simp [Jt, dite_eq_right h]
  have hJt_dep : ∀ w ∈ W, DependsOn S p (Jt w) w := by
    intro w h
    simp only [Jt, dite_eq_left h]
    exact hJd w h
  induction P generalizing L with
  | nil => intro _; trivial
  | cons o P ih =>
    intro h
    obtain ⟨hok, hrest⟩ := h
    refine And.intro ?_ (ih _ hrest)
    cases o
    case mma outs pp =>
      intro hp f hf
      obtain ⟨C, hC, ws, hlen, hws, d₀, hdet, hβ⟩ := hok hp f hf
      refine ⟨C, hC, fun g => f g - C g, ?_, fun g _ => by simp⟩
      simp only [LaneFns, Set.mem_ofPred_eq]
      refine ⟨ws.toFinset.biUnion Jt, ?_, d₀, ?_, fun g hg => ?_⟩
      · calc (ws.toFinset.biUnion Jt).card ≤ ∑ w ∈ ws.toFinset, (Jt w).card := Finset.card_biUnion_le
          _ ≤ ws.toFinset.card * q := by
              simpa using Finset.sum_le_card_nsmul ws.toFinset (fun w => (Jt w).card) q
                (fun w _ => hJt_card w)
          _ ≤ K * q := Nat.mul_le_mul_right q ((List.toFinset_card_le ws).trans hlen)
      · intro g hg g' hg' hagree
        apply hdet g hg g' hg'
        intro w hw
        exact hJt_dep w (hws w hw).2 g hg g' hg' fun j hj =>
          hagree j (Finset.mem_biUnion.2 ⟨w, List.mem_toFinset.2 hw, hj⟩)
      · have e : f g - C g - d₀ g = f g - (C g + d₀ g) := by ring
        rw [e]
        exact hβ g hg
    all_goals trivial

theorem denseWritesFrag : DenseWritesFrag := by
  intro a c pr hpr hca hac G Λ S p F W K q β hconst hβ hW fpSem L P T₁ T₂ hL hwf hacc hdense hsep hd₂ hapart
    ho₁ ho₂
  refine denseWritesLegacy a c pr hpr hca hac G F (LaneFns (K * q) β S p) hconst ?_ S fpSem L P T₁ T₂ hL hwf
    (accInOfFrag G Λ S p F W a K q β fpSem P L hW hacc) hdense hsep hd₂ hapart ho₁ ho₂
  simp only [LaneFns, Set.mem_ofPred_eq]
  exact ⟨∅, by simp, 0, fun _ _ _ _ _ => rfl, fun g _ => by simpa using hβ⟩

/-! ## The counterexample -/

/-- The `i`-th word of the counterexample: `i + 1` times the sum of the coordinates. -/
def fragWord (m i : ℕ) : (Fin (2 * m + 1) → ℚ) → ℚ := fun g => ((i : ℚ) + 1) * sumCoords (2 * m) g

/-- The counterexample's cheap instructions, one per index. -/
def fragMmas (m : ℕ) (is : List ℕ) : List (Op100 (Fin (2 * m + 1) → ℚ)) :=
  is.map fun i => Op100.mma [fragWord m i] 31

/-- The counterexample's program: the sum of the coordinates as an `other` register, then `n` cheap instructions. -/
def fragProg (m n : ℕ) : List (Op100 (Fin (2 * m + 1) → ℚ)) :=
  Op100.other (sumCoords (2 * m)) :: fragMmas m (List.range n)

theorem sumCoords_single (m : ℕ) (j : Fin (m + 1)) : sumCoords m (Pi.single j 1) = 1 := by
  simp [sumCoords, Finset.sum_pi_single']

theorem sumCoords_zero (m : ℕ) : sumCoords m 0 = 0 := by
  simp [sumCoords]

theorem accFragIn_mmas (m : ℕ) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) :
    ∀ (is : List ℕ) (Rs : List ((Fin (2 * m + 1) → ℚ) → ℚ)), sumCoords (2 * m) ∈ Rs →
      AccFragIn 32 1 0 fpSem Set.univ {f | ∃ x : ℚ, f = fun _ => x} Set.univ (fragMmas m is) Rs := by
  intro is
  induction is with
  | nil => intro _ _; trivial
  | cons i is ih =>
    intro Rs hu
    refine And.intro ?_ (ih _ (List.mem_append_left _ hu))
    intro _ f hf
    rw [List.mem_singleton] at hf
    subst hf
    refine ⟨fun _ => 0, Or.inr ⟨0, rfl⟩, [sumCoords (2 * m)], by simp, ?_, fragWord m i, ?_, ?_⟩
    · intro w hw
      rw [List.mem_singleton] at hw
      subst hw
      exact ⟨Or.inl hu, Set.mem_univ _⟩
    · intro g _ g' _ h
      have := h (sumCoords (2 * m)) (List.mem_singleton_self _)
      simp only [fragWord]
      rw [this]
    · intro g _
      simp

theorem mem_regs_mmas (m : ℕ) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) :
    ∀ (is : List ℕ) (Rs : List ((Fin (2 * m + 1) → ℚ) → ℚ)), ∀ i ∈ is,
      fragWord m i ∈ regs100 fpSem (fragMmas m is) Rs := by
  intro is
  induction is with
  | nil => intro _ i hi; simp at hi
  | cons j is ih =>
    intro Rs i hi
    have hr : regs100 fpSem (fragMmas m (j :: is)) Rs =
        regs100 fpSem (fragMmas m is) (Rs ++ [fragWord m j]) := rfl
    rw [hr]
    rcases List.mem_cons.1 hi with rfl | hi
    · exact mem_regs100_of_mem fpSem _ _ _ (List.mem_append_right _ (List.mem_singleton_self _))
    · exact ih _ i hi

theorem fragWord_injective (m : ℕ) : Function.Injective (fragWord m) := by
  intro i j h
  have := congrFun h (Pi.single 0 1)
  simp only [fragWord, sumCoords_single, mul_one, add_left_inj, Nat.cast_inj] at this
  exact this

theorem fragWord_contributes (m i : ℕ) (j : Fin (2 * m + 1)) :
    Contributes Set.univ (fun j (g : Fin (2 * m + 1) → ℚ) => g j) (2 * 0) (fragWord m i) j := by
  refine ⟨0, Set.mem_univ _, Pi.single j 1, Set.mem_univ _, fun k hk => ?_, ?_⟩
  · simp [Pi.single_eq_of_ne hk]
  · simp only [fragWord, sumCoords_zero, sumCoords_single]
    have e : ((i : ℚ) + 1) * 0 - ((i : ℚ) + 1) * 1 = -((i : ℚ) + 1) := by ring
    have hi : (0 : ℚ) < (i : ℚ) + 1 := by positivity
    rw [e, abs_neg, abs_of_pos hi]
    linarith

theorem fragWord_sub_contributes (m i i' : ℕ) (hne : i ≠ i') (j : Fin (2 * m + 1)) :
    Contributes Set.univ (fun j (g : Fin (2 * m + 1) → ℚ) => g j) (4 * 0) (fragWord m i - fragWord m i') j := by
  refine ⟨0, Set.mem_univ _, Pi.single j 1, Set.mem_univ _, fun k hk => ?_, ?_⟩
  · simp [Pi.single_eq_of_ne hk]
  · simp only [Pi.sub_apply, fragWord, sumCoords_zero, sumCoords_single]
    have hd : ((i : ℚ) - i') ≠ 0 := sub_ne_zero.2 (by exact_mod_cast hne)
    have e : ((i : ℚ) + 1) * 0 - ((i' : ℚ) + 1) * 0 - (((i : ℚ) + 1) * 1 - ((i' : ℚ) + 1) * 1) =
        -((i : ℚ) - i') := by ring
    rw [e, abs_neg]
    have := abs_pos.2 hd
    linarith

theorem fragShareGain : FragShareGain := by
  intro m n fpSem
  classical
  refine ⟨fragProg m n, (Finset.range n).image (fragWord m), ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [Finset.card_image_of_injective _ (fragWord_injective m), Finset.card_range]
  · intro o ho
    simp only [fragProg, fragMmas, List.mem_cons, List.mem_map] at ho
    rcases ho with rfl | ⟨i, _, rfl⟩
    · trivial
    · show h100Legacy.tc * 1 ≤ 31
      decide
  · exact And.intro trivial (accFragIn_mmas m fpSem _ _ (List.mem_singleton_self _))
  · refine denseCert _ _ Set.univ (fun j (g : Fin (2 * m + 1) → ℚ) => g j) _ m 0 0 _ ?_ ?_
    · rintro f ⟨x, rfl⟩
      exact ⟨∅, by simp, fun _ _ _ _ _ => rfl⟩
    · intro t ht
      obtain ⟨i, _, rfl⟩ := Finset.mem_image.1 ht
      exact ⟨Finset.univ, by simp; omega, fun j _ => fragWord_contributes m i j⟩
  · intro t ht t' ht' hne
    obtain ⟨i, _, rfl⟩ := Finset.mem_image.1 ht
    obtain ⟨i', _, rfl⟩ := Finset.mem_image.1 ht'
    have hii : i ≠ i' := fun h => hne (h ▸ rfl)
    exact apartCert _ _ Set.univ (fun j (g : Fin (2 * m + 1) → ℚ) => g j) m 0 _ _
      ⟨Finset.univ, by simp, fun j _ => fragWord_sub_contributes m i i' hii j⟩
  · intro t ht
    obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 ht
    have hr : regs100 fpSem (fragProg m n) [] = regs100 fpSem (fragMmas m (List.range n)) [sumCoords (2 * m)] :=
      rfl
    refine ⟨fragWord m i, ?_, fun _ _ => rfl⟩
    rw [hr]
    exact mem_regs_mmas m fpSem _ _ i (List.mem_range.2 (Finset.mem_range.1 hi))
  · simp [fragProg, fragMmas, cost100, Op100.price, h100Legacy, h100Nominal, Function.comp_def]
    ring

end Pouw.Dimension.H100
