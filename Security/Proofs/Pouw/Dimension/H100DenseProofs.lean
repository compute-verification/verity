import Models.Pouw.Assumptions.Dimension.H100Dense
import Proofs.Pouw.Dimension.H100WProofs

/-!
# Dense words and the legacy k8 class in `MH100w`: proofs

* **The cover** (`acc_cover`): every register is equal on `S` to a free value, to a register written at `a` or more
  per register (`H`), or to a cheap register (`Lo`) that is not equal on `S` to its own accumulator. A cheap register
  equal to its accumulator is a copy of an earlier register, and is covered by it.
* **The count** (`denseWritesLegacy`): each target matches an element of `H` or `Lo`, preferring `H`. A dense target
  matching only `Lo` has an accumulator that is not free, so it matches a further element of `H` or `Lo`. Apart targets
  give distinct elements throughout, so the program pays for `|T₁| + |T₂|` registers plus one per such target.
-/

namespace Pouw.Dimension.H100

open Finset Pouw.Dimension Pouw.Dimension.Embed

variable {G : Type}

/-- A register covered on `S` by the free values `F`, the registers `H` written at `a` or more, and the cheap
registers `Lo`. -/
def CovS (S : Set G) (F : Set (G → ℚ)) (H Lo : List (G → ℚ)) (r : G → ℚ) : Prop :=
  (∃ f ∈ F, Set.EqOn r f S) ∨ (∃ h ∈ H, Set.EqOn r h S) ∨ (∃ u ∈ Lo, Set.EqOn r u S)

/-- A cheap register with an accumulator among the registers `Rs` or the free values, not equal to it on `S`. -/
def AccGood (S : Set G) (F D : Set (G → ℚ)) (Rs : List (G → ℚ)) (u : G → ℚ) : Prop :=
  ∃ C : G → ℚ, (C ∈ Rs ∨ C ∈ F) ∧ ¬ Set.EqOn C u S ∧ ∃ d ∈ D, Set.EqOn u (C + d) S

theorem covS_mono {S : Set G} {F : Set (G → ℚ)} {H Lo : List (G → ℚ)} (H₁ Lo₁ : List (G → ℚ)) {r : G → ℚ}
    (h : CovS S F H Lo r) : CovS S F (H ++ H₁) (Lo ++ Lo₁) r := by
  rcases h with ⟨f, hf, e⟩ | ⟨x, hx, e⟩ | ⟨x, hx, e⟩
  · exact Or.inl ⟨f, hf, e⟩
  · exact Or.inr (Or.inl ⟨x, List.mem_append_left _ hx, e⟩)
  · exact Or.inr (Or.inr ⟨x, List.mem_append_left _ hx, e⟩)

theorem covS_of_eqOn {S : Set G} {F : Set (G → ℚ)} {H Lo : List (G → ℚ)} {r r' : G → ℚ}
    (e : Set.EqOn r r' S) (h : CovS S F H Lo r') : CovS S F H Lo r := by
  rcases h with ⟨f, hf, e'⟩ | ⟨x, hx, e'⟩ | ⟨x, hx, e'⟩
  · exact Or.inl ⟨f, hf, e.trans e'⟩
  · exact Or.inr (Or.inl ⟨x, hx, e.trans e'⟩)
  · exact Or.inr (Or.inr ⟨x, hx, e.trans e'⟩)

theorem accGood_mono {S : Set G} {F D : Set (G → ℚ)} {Rs : List (G → ℚ)} (X : List (G → ℚ)) {u : G → ℚ}
    (h : AccGood S F D Rs u) : AccGood S F D (Rs ++ X) u := by
  obtain ⟨C, hC, hne, d, hd, e⟩ := h
  exact ⟨C, hC.imp_left (List.mem_append_left _), hne, d, hd, e⟩

theorem acc_step (a c : ℕ) (pr : Prices100) (hpr : FloorTc a c pr) (F D : Set (G → ℚ)) (h0 : (0 : G → ℚ) ∈ F)
    (S : Set G) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (o : Op100 G) (Rs H Lo : List (G → ℚ))
    (hcov : ∀ r ∈ Rs, CovS S F H Lo r) (hag : ∀ u ∈ Lo, AccGood S F D Rs u) (hwo : o.WFw pr F)
    (hacc : o.AccOK a S F D Rs) :
    ∃ H₁ Lo₁ : List (G → ℚ), (∀ r ∈ Rs ++ o.outs fpSem Rs, CovS S F (H ++ H₁) (Lo ++ Lo₁) r) ∧
      (∀ u ∈ Lo ++ Lo₁, AccGood S F D (Rs ++ o.outs fpSem Rs) u) ∧
      a * H₁.length + c * Lo₁.length ≤ o.price pr := by
  classical
  obtain ⟨hs, htc, hffma, hint, hfp64, hother⟩ := hpr
  have hiCase : a * (o.outs fpSem Rs).length ≤ o.price pr →
      ∃ H₁ Lo₁ : List (G → ℚ), (∀ r ∈ Rs ++ o.outs fpSem Rs, CovS S F (H ++ H₁) (Lo ++ Lo₁) r) ∧
        (∀ u ∈ Lo ++ Lo₁, AccGood S F D (Rs ++ o.outs fpSem Rs) u) ∧
        a * H₁.length + c * Lo₁.length ≤ o.price pr := by
    intro hp
    refine ⟨o.outs fpSem Rs, [], fun r hr => ?_, fun u hu => ?_, by simpa using hp⟩
    · rcases List.mem_append.1 hr with hr | hr
      · exact covS_mono _ _ (hcov r hr)
      · exact Or.inr (Or.inl ⟨r, List.mem_append_right _ hr, Set.eqOn_refl _ _⟩)
    · rw [List.append_nil] at hu
      exact accGood_mono _ (hag u hu)
  have freeCase : (∀ r ∈ o.outs fpSem Rs, CovS S F H Lo r) →
      ∃ H₁ Lo₁ : List (G → ℚ), (∀ r ∈ Rs ++ o.outs fpSem Rs, CovS S F (H ++ H₁) (Lo ++ Lo₁) r) ∧
        (∀ u ∈ Lo ++ Lo₁, AccGood S F D (Rs ++ o.outs fpSem Rs) u) ∧
        a * H₁.length + c * Lo₁.length ≤ o.price pr := by
    intro hnew
    refine ⟨[], [], fun r hr => ?_, fun u hu => ?_, by simp⟩
    · rcases List.mem_append.1 hr with hr | hr
      · exact covS_mono _ _ (hcov r hr)
      · exact covS_mono _ _ (hnew r hr)
    · rw [List.append_nil] at hu
      exact accGood_mono _ (hag u hu)
  cases o with
  | wgmma s outs =>
    exact hiCase (by simpa [Op100.price, Op100.outs] using Nat.mul_le_mul_right outs.length (hs s))
  | mma outs p =>
    by_cases hp : a * outs.length ≤ p
    · exact hiCase (by simpa [Op100.price, Op100.outs] using hp)
    · have hA : ∀ f ∈ outs, ∃ C : G → ℚ, (C ∈ Rs ∨ C ∈ F) ∧ ∃ d ∈ D, ∀ g ∈ S, f g = C g + d g :=
        hacc (by omega)
      refine ⟨[], outs.filter fun f => decide (AccGood S F D Rs f), fun r hr => ?_, fun u hu => ?_, ?_⟩
      · rcases List.mem_append.1 hr with hr | hr
        · exact covS_mono _ _ (hcov r hr)
        · simp only [Op100.outs] at hr
          by_cases hg : AccGood S F D Rs r
          · exact Or.inr (Or.inr ⟨r, List.mem_append_right _ (List.mem_filter.2 ⟨hr, by simpa using hg⟩),
              Set.eqOn_refl _ _⟩)
          · obtain ⟨C, hC, d, hd, e⟩ := hA r hr
            have hCr : Set.EqOn C r S := by
              by_contra hne
              exact hg ⟨C, hC, hne, d, hd, fun g hg' => e g hg'⟩
            rcases hC with hC | hC
            · exact covS_mono _ _ (covS_of_eqOn hCr.symm (hcov C hC))
            · exact Or.inl ⟨C, hC, hCr.symm⟩
      · rcases List.mem_append.1 hu with hu | hu
        · exact accGood_mono _ (hag u hu)
        · exact accGood_mono _ (by simpa using (List.mem_filter.1 hu).2)
      · simp only [Op100.WFw] at hwo
        simp only [Op100.price, List.length_nil, Nat.mul_zero, Nat.zero_add]
        calc c * (outs.filter fun f => decide (AccGood S F D Rs f)).length ≤ c * outs.length :=
              Nat.mul_le_mul_left _ (List.length_filter_le _ _)
          _ ≤ pr.tc * outs.length := Nat.mul_le_mul_right _ htc
          _ ≤ p := hwo
  | fp32 κ i j l => exact hiCase (by simpa [Op100.price, Op100.outs] using hffma)
  | int f => exact hiCase (by simpa [Op100.price, Op100.outs] using hint)
  | fp64 f => exact hiCase (by simpa [Op100.price, Op100.outs] using hfp64)
  | other f => exact hiCase (by simpa [Op100.price, Op100.outs] using hother)
  | move i =>
    apply freeCase
    intro r hr
    simp only [Op100.outs, List.mem_singleton] at hr
    subst hr
    rcases getD_mem_or Rs i with h | h
    · exact hcov _ h
    · rw [h]
      exact Or.inl ⟨0, h0, Set.eqOn_refl _ _⟩
  | operand f =>
    apply freeCase
    intro r hr
    simp only [Op100.outs, List.mem_singleton] at hr
    subst hr
    exact Or.inl ⟨r, hwo, Set.eqOn_refl _ _⟩

theorem acc_cover (a c : ℕ) (pr : Prices100) (hpr : FloorTc a c pr) (F D : Set (G → ℚ)) (h0 : (0 : G → ℚ) ∈ F)
    (S : Set G) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) :
    ∀ (P : List (Op100 G)) (Rs H Lo : List (G → ℚ)),
      (∀ r ∈ Rs, CovS S F H Lo r) → (∀ u ∈ Lo, AccGood S F D Rs u) →
      (∀ o ∈ P, o.WFw pr F) → AccIn a fpSem S F D P Rs →
      ∃ H' Lo' : List (G → ℚ), (∀ r ∈ regs100 fpSem P Rs, CovS S F (H ++ H') (Lo ++ Lo') r) ∧
        (∀ u ∈ Lo ++ Lo', AccGood S F D (regs100 fpSem P Rs) u) ∧
        a * H'.length + c * Lo'.length ≤ cost100 pr P := by
  intro P
  induction P with
  | nil =>
    intro Rs H Lo hcov hag _ _
    exact ⟨[], [], by simpa [regs100] using hcov, by simpa [regs100] using hag, by simp [cost100]⟩
  | cons o P ih =>
    intro Rs H Lo hcov hag hwf hacc
    obtain ⟨hacc₁, hacc₂⟩ := hacc
    obtain ⟨H₁, Lo₁, hc₁, ha₁, hp₁⟩ :=
      acc_step a c pr hpr F D h0 S fpSem o Rs H Lo hcov hag (hwf o List.mem_cons_self) hacc₁
    obtain ⟨H₂, Lo₂, hc₂, ha₂, hp₂⟩ := ih (Rs ++ o.outs fpSem Rs) (H ++ H₁) (Lo ++ Lo₁) hc₁ ha₁
      (fun o' ho' => hwf o' (List.mem_cons_of_mem _ ho')) hacc₂
    have hr : regs100 fpSem (o :: P) Rs = regs100 fpSem P (Rs ++ o.outs fpSem Rs) := rfl
    have e : cost100 pr (o :: P) = o.price pr + cost100 pr P := by simp [cost100]
    rw [hr, e]
    refine ⟨H₁ ++ H₂, Lo₁ ++ Lo₂, ?_, ?_, ?_⟩
    · rw [← List.append_assoc, ← List.append_assoc]
      exact hc₂
    · rw [← List.append_assoc]
      exact ha₂
    · simp only [List.length_append, Nat.mul_add]
      omega

/-- **Dense words cost `a` against the legacy class.** -/
theorem denseWritesLegacy : DenseWritesLegacy := by
  intro a c pr hpr hca hac G F D hconst hD0 S fpSem L P T₁ T₂ hL hwf hacc hdense hsep hd₂ hapart ho₁ ho₂
  classical
  obtain ⟨H', Lo', hcov, hag, hcost⟩ := acc_cover a c pr hpr F D (const_zero hconst) S fpSem P L [] []
    (fun r hr => Or.inl ⟨r, hL r hr, Set.eqOn_refl _ _⟩) (by simp) hwf hacc
  simp only [List.nil_append] at hcov hag
  have hz : ∀ t t' : G → ℚ, Set.EqOn t t' S → ∀ g ∈ S, t g - (0 : G → ℚ) g = t' g - (0 : G → ℚ) g := by
    intro t t' e g hg
    simp [e hg]
  have hnf₁ : ∀ t ∈ T₁, ∀ f ∈ F, ∀ d ∈ D, ¬ Set.EqOn t (f + d) S :=
    fun t ht f hf d hd e => hdense t ht f hf d hd fun g hg => e hg
  have hdisj : Disjoint T₁ T₂ := by
    rw [Finset.disjoint_left]
    intro t h1 h2
    exact hapart t h1 t h2 0 hD0 0 hD0 (hz t t (Set.eqOn_refl _ _))
  have heq : ∀ t ∈ T₁ ∪ T₂, ∀ t' ∈ T₁ ∪ T₂, Set.EqOn t t' S → t = t' := by
    intro t ht t' ht' e
    rcases Finset.mem_union.1 ht with h | h <;> rcases Finset.mem_union.1 ht' with h' | h'
    · by_contra hne
      exact hsep t h t' h' hne 0 hD0 0 hD0 (hz t t' e)
    · exact absurd (hz t t' e) (hapart t h t' h' 0 hD0 0 hD0)
    · exact absurd (hz t' t e.symm) (hapart t' h' t h 0 hD0 0 hD0)
    · exact hd₂.1 t h t' h' fun g hg => e hg
  have hex : ∀ t ∈ T₁ ∪ T₂, ∃ x, x ∈ H' ++ Lo' ∧ Set.EqOn t x S ∧ ((∃ h ∈ H', Set.EqOn t h S) → x ∈ H') := by
    intro t ht
    have hnf : ∀ f ∈ F, ¬ Set.EqOn t f S := by
      rcases Finset.mem_union.1 ht with h | h
      · intro f hf e
        exact hdense t h f hf 0 hD0 fun g hg => by simpa using e hg
      · intro f hf e
        exact hd₂.2 t h f hf fun g hg => e hg
    obtain ⟨r, hr, e⟩ : ∃ r ∈ regs100 fpSem P L, Set.EqOn r t S := by
      rcases Finset.mem_union.1 ht with h | h
      · obtain ⟨r, hr, e⟩ := ho₁ t h
        exact ⟨r, hr, fun g hg => e g hg⟩
      · obtain ⟨r, hr, e⟩ := ho₂ t h
        exact ⟨r, hr, fun g hg => e g hg⟩
    by_cases hH : ∃ h ∈ H', Set.EqOn t h S
    · obtain ⟨h, hh, e'⟩ := hH
      exact ⟨h, List.mem_append_left _ hh, e', fun _ => hh⟩
    · rcases hcov r hr with ⟨f, hf, e'⟩ | ⟨x, hx, e'⟩ | ⟨x, hx, e'⟩
      · exact absurd (e.symm.trans e') (hnf f hf)
      · exact absurd ⟨x, hx, e.symm.trans e'⟩ hH
      · exact ⟨x, List.mem_append_right _ hx, e.symm.trans e', fun h => absurd h hH⟩
  choose! ρ hρm hρe hρH using hex
  have hacc' : ∀ t ∈ T₁, ρ t ∉ H' → ∃ y, y ∈ H' ++ Lo' ∧ ∃ C : G → ℚ, Set.EqOn C y S ∧
      ¬ Set.EqOn C (ρ t) S ∧ ∃ d, d ∈ D ∧ Set.EqOn t (C + d) S := by
    intro t ht hnH
    have htu : t ∈ T₁ ∪ T₂ := Finset.mem_union_left _ ht
    have hLo : ρ t ∈ Lo' := by
      rcases List.mem_append.1 (hρm t htu) with h | h
      · exact absurd h hnH
      · exact h
    obtain ⟨C, hC, hne, d, hd, e⟩ := hag _ hLo
    have htC : Set.EqOn t (C + d) S := (hρe t htu).trans e
    rcases hC with hC | hC
    · rcases hcov C hC with ⟨f, hf, e'⟩ | ⟨x, hx, e'⟩ | ⟨x, hx, e'⟩
      · refine absurd (fun g hg => ?_) (hnf₁ t ht f hf d hd)
        have h1 := htC hg
        have h2 := e' hg
        simp only [Pi.add_apply] at h1 ⊢
        rw [h1, h2]
      · exact ⟨x, List.mem_append_left _ hx, C, e', hne, d, hd, htC⟩
      · exact ⟨x, List.mem_append_right _ hx, C, e', hne, d, hd, htC⟩
    · exact absurd htC (hnf₁ t ht C hC d hd)
  choose! σ hσm C hCσ hCne dd hdd htC using hacc'
  set T₁lo := T₁.filter fun t => ρ t ∉ H'
  set T₁hi := T₁.filter fun t => ρ t ∈ H'
  have hmem_lo : ∀ t ∈ T₁lo, t ∈ T₁ ∧ ρ t ∉ H' := fun t ht => Finset.mem_filter.1 ht
  have hinjρ : Set.InjOn ρ ↑(T₁ ∪ T₂) := by
    intro t ht t' ht' h
    refine heq t ht t' ht' fun g hg => ?_
    rw [hρe t ht hg, hρe t' ht' hg, h]
  have hsub : ∀ t ∈ T₁lo, ∀ g ∈ S, t g - dd t g = C t g := by
    intro t ht g hg
    obtain ⟨h1, h2⟩ := hmem_lo t ht
    have := htC t h1 h2 hg
    simp only [Pi.add_apply] at this
    linarith
  have hinjσ : Set.InjOn σ ↑T₁lo := by
    intro t ht t' ht' h
    obtain ⟨h1, h2⟩ := hmem_lo t ht
    obtain ⟨h1', h2'⟩ := hmem_lo t' ht'
    by_contra hne
    refine hsep t h1 t' h1' hne (dd t) (hdd t h1 h2) (dd t') (hdd t' h1' h2') fun g hg => ?_
    rw [hsub t ht g hg, hsub t' ht' g hg, hCσ t h1 h2 hg, hCσ t' h1' h2' hg, h]
  have hdisjAB : Disjoint ((T₁ ∪ T₂).image ρ) (T₁lo.image σ) := by
    rw [Finset.disjoint_left]
    intro x hxA hxB
    obtain ⟨t', ht', rfl⟩ := Finset.mem_image.1 hxA
    obtain ⟨t, ht, hσt⟩ := Finset.mem_image.1 hxB
    obtain ⟨h1, h2⟩ := hmem_lo t ht
    have hCt : ∀ g ∈ S, t g - dd t g = t' g - (0 : G → ℚ) g := by
      intro g hg
      rw [hsub t ht g hg, hCσ t h1 h2 hg, hσt, ← hρe t' ht' hg]
      simp
    by_cases htt : t = t'
    · subst htt
      refine hCne t h1 h2 fun g hg => ?_
      have := hCt g hg
      rw [hsub t ht g hg] at this
      rw [this, ← hρe t ht' hg]
      simp
    · rcases Finset.mem_union.1 ht' with h' | h'
      · exact hsep t h1 t' h' htt (dd t) (hdd t h1 h2) 0 hD0 hCt
      · exact hapart t h1 t' h' (dd t) (hdd t h1 h2) 0 hD0 hCt
  have hU : (T₁ ∪ T₂).image ρ ∪ T₁lo.image σ ⊆ (H' ++ Lo').toFinset := by
    intro x hx
    rcases Finset.mem_union.1 hx with hx | hx
    · obtain ⟨t, ht, rfl⟩ := Finset.mem_image.1 hx
      exact List.mem_toFinset.2 (hρm t ht)
    · obtain ⟨t, ht, rfl⟩ := Finset.mem_image.1 hx
      obtain ⟨h1, h2⟩ := hmem_lo t ht
      exact List.mem_toFinset.2 (hσm t h1 h2)
  have c1 : (T₁ ∪ T₂).card + T₁lo.card ≤ H'.length + Lo'.length := by
    rw [← Finset.card_image_of_injOn hinjρ, ← Finset.card_image_of_injOn hinjσ,
      ← Finset.card_union_of_disjoint hdisjAB]
    refine le_trans (Finset.card_le_card hU) (le_trans (List.toFinset_card_le _) ?_)
    simp
  have c2 : T₁hi.card ≤ H'.length := by
    refine le_trans (Finset.card_le_card_of_injOn ρ (fun t ht => ?_) (hinjρ.mono ?_))
      (List.toFinset_card_le _)
    · exact List.mem_toFinset.2 (Finset.mem_filter.1 ht).2
    · intro t ht
      exact Finset.mem_union_left _ (Finset.mem_filter.1 ht).1
  have c3 : T₁hi.card + T₁lo.card = T₁.card := Finset.card_filter_add_card_filter_not _
  rw [Finset.card_union_of_disjoint hdisj] at c1
  obtain ⟨e, rfl⟩ := Nat.exists_eq_add_of_le hca
  have he : e ≤ c := by omega
  nlinarith [Nat.mul_le_mul_left e c2, Nat.mul_le_mul_left c c1, Nat.mul_le_mul_right T₁lo.card he]

/-- **The dense certificate.** -/
theorem denseCert : DenseCert := by
  intro G Λ S p F m r β T hF hT t ht f hf d hd heq
  classical
  obtain ⟨J, hJ, d₀, hd₀, hβ⟩ := hd
  obtain ⟨Jf, hJf, hfdep⟩ := hF f hf
  obtain ⟨W, hW, hc⟩ := hT t ht
  obtain ⟨j, hjW, hjn⟩ : ∃ j, j ∈ W ∧ j ∉ J ∪ Jf :=
    Finset.exists_mem_notMem_of_card_lt_card (lt_of_le_of_lt ((Finset.card_union_le _ _).trans (by omega)) hW)
  obtain ⟨g, hg, g', hg', hagree, hθ⟩ := hc j hjW
  have hsame : ∀ i ∈ J ∪ Jf, p i g = p i g' := fun i hi => hagree i fun h => hjn (h ▸ hi)
  have e1 : f g = f g' := hfdep g hg g' hg' fun i hi => hsame i (Finset.mem_union_right _ hi)
  have e2 : d₀ g = d₀ g' := hd₀ g hg g' hg' fun i hi => hsame i (Finset.mem_union_left _ hi)
  have b1 := abs_le.1 (hβ g hg)
  have b2 := abs_le.1 (hβ g' hg')
  have t1 := heq g hg
  have t2 := heq g' hg'
  have : |t g - t g'| ≤ 2 * β := abs_le.2 ⟨by linarith, by linarith⟩
  linarith

/-- **The apart certificate.** -/
theorem apartCert : ApartCert := by
  intro G Λ S p m β t t' hW' d hd d' hd' heq
  classical
  obtain ⟨W, hW, hc⟩ := hW'
  obtain ⟨J, hJ, d₀, hd₀, hβ⟩ := hd
  obtain ⟨J', hJ', d₀', hd₀', hβ'⟩ := hd'
  obtain ⟨j, hjW, hjn⟩ : ∃ j, j ∈ W ∧ j ∉ J ∪ J' :=
    Finset.exists_mem_notMem_of_card_lt_card (lt_of_le_of_lt ((Finset.card_union_le _ _).trans (by omega)) hW)
  obtain ⟨g, hg, g', hg', hagree, hθ⟩ := hc j hjW
  have hsame : ∀ i ∈ J ∪ J', p i g = p i g' := fun i hi => hagree i fun h => hjn (h ▸ hi)
  have e1 : d₀ g = d₀ g' := hd₀ g hg g' hg' fun i hi => hsame i (Finset.mem_union_left _ hi)
  have e2 : d₀' g = d₀' g' := hd₀' g hg g' hg' fun i hi => hsame i (Finset.mem_union_right _ hi)
  have b1 := abs_le.1 (hβ g hg)
  have b2 := abs_le.1 (hβ g' hg')
  have b3 := abs_le.1 (hβ' g hg)
  have b4 := abs_le.1 (hβ' g' hg')
  have t1 := heq g hg
  have t2 := heq g' hg'
  have : |(t - t') g - (t - t') g'| ≤ 4 * β := by
    simp only [Pi.sub_apply]
    exact abs_le.2 ⟨by linarith, by linarith⟩
  linarith

/-- **The γ bound with a legacy class.** -/
theorem legacyGamma : LegacyGamma := by
  intro a c pr hpr hca hac ha G F D hconst hD0 S fpSem L P ts T₁ T₂ ε μ hn _ _ hL hwf hacc hdense hsep hd₂ hapart
    ho₁ ho₂ hlive hμ
  have h := denseWritesLegacy a c pr hpr hca hac G F D hconst hD0 S fpSem L P T₁ T₂ hL hwf hacc hdense hsep hd₂
    hapart ho₁ ho₂
  have hq : ((a * T₁.card + c * T₂.card : ℕ) : ℚ) ≤ cost100 pr P := by exact_mod_cast h
  push_cast at hq
  have hn' : (0 : ℚ) < ts.length := by exact_mod_cast hn
  have ha' : (0 : ℚ) < a := by exact_mod_cast ha
  have hca' : (c : ℚ) ≤ a := by exact_mod_cast hca
  have key : (1 - ε - μ * (a - c) / a) * (a * ts.length) ≤ cost100 pr P := by
    have e : (1 - ε - μ * (a - c) / a) * (a * ts.length) =
        a * ((1 - ε) * ts.length) - (a - c) * (μ * ts.length) := by
      field_simp
    rw [e]
    nlinarith [mul_le_mul_of_nonneg_left hlive ha'.le, mul_le_mul_of_nonneg_left hμ (sub_nonneg.2 hca')]
  have : 1 - ε - μ * (a - c) / a ≤ (cost100 pr P : ℚ) / (a * ts.length) := by
    rw [le_div_iff₀ (by positivity)]
    exact key
  linarith

/-- **Dense words exist for every lane count.** -/
theorem denseLaneWitness : DenseLaneWitness := by
  intro m
  classical
  refine ⟨?_, ?_⟩
  · refine denseCert _ _ Set.univ (fun j (g : Fin (m + 1) → ℚ) => g j) _ m 0 0 {sumCoords m} ?_ ?_
    · rintro f ⟨q, rfl⟩
      exact ⟨∅, by simp, fun _ _ _ _ _ => rfl⟩
    · intro t ht
      rw [Finset.mem_singleton] at ht
      subst ht
      refine ⟨Finset.univ, by simp, fun j _ => ⟨0, Set.mem_univ _, Pi.single j 1, Set.mem_univ _, ?_, ?_⟩⟩
      · intro i hi
        simp [Pi.single_eq_of_ne hi]
      · simp [sumCoords, Finset.sum_pi_single']
  · intro t ht t' ht' hne
    rw [Finset.mem_singleton] at ht ht'
    exact absurd (ht.trans ht'.symm) hne

end Pouw.Dimension.H100
