import Models.Accounting.Guarantees.Bridges
import Proofs.Accounting.Actual.Physical
import Proofs.Accounting.Actual.Localization

/-!
# B4, proved

What the outside `O = (X ∪ cert)ᶜ` does during `[a, b)` is part of what it does before `b` (`acts_restrict`), which
causality's global form makes a function of what it reads from outside before `b`, its inputs being fixed. That is
what it read before `a` (fixed), what `X` sent it during `[a, b)` and what the certifiers sent it then
(`crossing_split`). `X`'s wires to `O` run on the uncertified links across `X`'s boundary (`Links`), so they are the
signals those links deliver at their ends outside `X` (`crossing_signal`). The certifiers' wires to `O` read during
`[a, b)` left them before `b` (`crossing_sent`, by causality): before `a`, a function of the certifiers' inputs and of
what they read before `a`, both fixed, or during `[a, b)`, a function of their record and of their jitter word on
`[a, b)`'s buckets (`TrustedUnits`). So what `O` does is a function of the record, of the jitter word on those buckets
and of the uncertified links' signals during `[a, b)`, which take at most (the number of records) × `χ^{buckets}` ×
`2^{β°_X (b − a)}` values (`Alphabet`, `encard_range_le`).
-/

namespace Accounting.SecurityProofs

open Finset

namespace Actual

/-- A map that factors through another takes at most as many values. -/
theorem encard_range_le {Ω α β : Type*} {f : Ω → α} {g : Ω → β} (h : ∀ r r', g r = g r' → f r = f r') :
    (Set.range f).encard ≤ (Set.range g).encard := by
  rcases isEmpty_or_nonempty Ω with hΩ | ⟨⟨r₀⟩⟩
  · simp [Set.range_eq_empty]
  · have hf : Function.FactorsThrough f g := h
    refine le_trans (Set.encard_le_encard ?_) (Set.encard_image_le (Function.extend g f fun _ => f r₀) _)
    rintro _ ⟨r, rfl⟩
    exact ⟨g r, ⟨r, rfl⟩, hf.extend_apply _ r⟩

/-- What a location does during `J ⊆ I` is what it does during `I`, cut to `J`. -/
theorem acts_restrict {L : Type*} (R : Actual.Run L) {X : Set L} {I J : Set ℕ} (hJ : J ⊆ I) :
    R.acts X J = ((R.acts X I).1 ∩ {p | p.1 ∈ J}, (R.acts X I).2.1 ∩ {p | p.2.2.1 ∈ J},
      (R.acts X I).2.2 ∩ {p | p.1 ∈ J}) := by
  simp only [Actual.Run.acts, Actual.Run.sent, Prod.mk.injEq]
  refine ⟨?_, ?_, ?_⟩ <;> ext p <;> simp only [Set.mem_inter_iff, Set.mem_ofPred_eq] <;> constructor
  · rintro ⟨g, hx, ht, rfl⟩
    exact ⟨⟨g, hx, hJ ht, rfl⟩, ht⟩
  · rintro ⟨⟨g, hx, -, rfl⟩, ht⟩
    exact ⟨g, hx, ht, rfl⟩
  · rintro ⟨g, a, hr, ha, hg, ht, rfl⟩
    exact ⟨⟨g, a, hr, ha, hg, hJ ht, rfl⟩, ht⟩
  · rintro ⟨⟨g, a, hr, ha, hg, -, rfl⟩, ht⟩
    exact ⟨g, a, hr, ha, hg, ht, rfl⟩
  · rintro ⟨y, v, a, hy, hv, ha, rfl⟩
    exact ⟨⟨y, v, a, hy, hJ hv, ha, rfl⟩, hv⟩
  · rintro ⟨⟨y, v, a, hy, -, ha, rfl⟩, hv⟩
    exact ⟨y, v, a, hy, hv, ha, rfl⟩

/-- Under causality, what crosses from `X` to `Y` during `[a, b)` is what `X` sent before `b` that arrives in `Y`
then. -/
theorem crossing_sent {L : Type*} (R : Actual.Run L) {ℓ : L → L → ℕ} (hc : R.A.Causal ℓ) {X Y : Set L}
    (hXY : Disjoint X Y) (a b : ℕ) : R.crossing X Y (Set.Ico a b) =
      (R.sent X (Set.Iio a) ∪ R.sent X (Set.Ico a b)) ∩ {p | p.1 ∈ Set.Ico a b ∧ p.2.1 ∈ Y} := by
  ext p
  simp only [Actual.Run.crossing, Actual.Run.sent, Set.mem_inter_iff, Set.mem_union, Set.mem_ofPred_eq]
  constructor
  · rintro ⟨g, s, hs, hx, hy, ht, rfl⟩
    have hgX : R.A.x g ∉ X := fun h => Set.disjoint_left.1 hXY h hy
    have hst : R.A.t s ≤ R.A.t g := (Nat.le_add_right _ _).trans (hc g s hs)
    refine ⟨?_, ht, hy⟩
    by_cases h : R.A.t s < a
    · exact Or.inl ⟨g, s, hs, hx, hgX, h, rfl⟩
    · exact Or.inr ⟨g, s, hs, hx, hgX, ⟨not_lt.1 h, hst.trans_lt ht.2⟩, rfl⟩
  · rintro ⟨⟨g, s, hs, hx, -, -, rfl⟩ | ⟨g, s, hs, hx, -, -, rfl⟩, ht, hy⟩ <;> exact ⟨g, s, hs, hx, hy, ht, rfl⟩

open Classical in
/-- The uncertified links across `X`'s boundary, the ones `β°_X` counts. -/
noncomputable def uncertified {L E : Type*} [Fintype E] (ends : E → L × L) (cert X : Finset L) : Finset E :=
  (boundary ends X).filter fun e => (ends e).1 ∉ cert ∧ (ends e).2 ∉ cert

open Classical in
theorem mem_uncertified {L E : Type*} [Fintype E] {ends : E → L × L} {cert X : Finset L} {e : E} :
    e ∈ uncertified ends cert X ↔ e ∈ boundary ends X ∧ (ends e).1 ∉ cert ∧ (ends e).2 ∉ cert :=
  mem_filter

theorem uncertifiedCapacity_eq {L E : Type*} [Fintype E] (ends : E → L × L) (β : E → ℕ) (cert X : Finset L) :
    uncertifiedCapacity ends β cert X = ∑ e ∈ uncertified ends cert X, β e := rfl

/-- Under `Links`, what crosses from `X` to the leaves in neither `X` nor `cert` is the signals the uncertified links
across `X`'s boundary deliver at their ends outside `X`: runs alike in those signals are alike in it. -/
theorem crossing_signal {L E : Type*} [Fintype E] {R R' : Actual.Run L} {ends : E → L × L} {β β' : E → ℕ}
    {link : Fin R.A.N → Fin R.A.N → E} {link' : Fin R'.A.N → Fin R'.A.N → E} (hL : R.A.Links ends β link)
    (hL' : R'.A.Links ends β' link') {cert X : Finset L} (hXc : Disjoint X cert) {I : Set ℕ}
    (h : ∀ e ∈ uncertified ends cert X, ∀ u ∈ I,
      R.signal link e (outer ends X e) u = R'.signal link' e (outer ends X e) u) :
    R.crossing ↑X (↑X ∪ ↑cert)ᶜ I ⊆ R'.crossing ↑X (↑X ∪ ↑cert)ᶜ I := by
  rintro _ ⟨g, s, hs, hx, hy, ht, rfl⟩
  have hgX : R.A.x g ∉ X := fun h' => hy (Or.inl h')
  have hgc : R.A.x g ∉ cert := fun h' => hy (Or.inr h')
  have hne : R.A.x s ≠ R.A.x g := fun h' => hgX (h' ▸ hx)
  have he := (hL.1 g s hs hne).2
  have ho : outer ends X (link g s) = R.A.x g := (inner_outer hgX hx he.symm).2
  have hu : link g s ∈ uncertified ends cert X := by
    refine mem_uncertified.2 ⟨mem_boundary hgX hx he.symm, ?_⟩
    have hsc : R.A.x s ∉ cert := Finset.disjoint_left.1 hXc hx
    rcases he with he | he <;> rw [he] <;> exact ⟨‹_›, ‹_›⟩
  have hsig : R.seen s ∈ R.signal link (link g s) (outer ends X (link g s)) (R.A.t g) :=
    ⟨g, s, hs, hne, rfl, ho.symm, rfl, rfl⟩
  rw [h _ hu _ ht] at hsig
  obtain ⟨g', s', hs', hne', hl', hy', ht', hp⟩ := hsig
  have key : R'.A.x s' = R.A.x s := by
    have he' := (hL'.1 g' s' hs' hne').2
    rw [hl'] at he'
    rcases he with he | he <;> rcases he' with he' | he' <;> rw [he] at he' <;>
      simp only [Prod.mk.injEq] at he'
    · exact he'.1.symm
    · exact absurd (he'.1.trans (hy'.trans ho)) hne
    · exact absurd (he'.2.trans (hy'.trans ho)) hne
    · exact he'.2.symm
  refine ⟨g', s', hs', ?_, ?_, ?_, ?_⟩
  · rw [key]
    exact hx
  · rw [hy', ho]
    exact hy
  · rw [ht']
    exact ht
  · rw [ht', hy', ho, hp]

end Actual

theorem B4 : Guarantees.B4 := by
  intro L E Ω _ _ _ S ℓ ends β link 𝒱 cert X spec record bucket jitter a b hc hα hT hXc hh hr
  obtain ⟨-, -, J, hJ⟩ := hT
  set O : Finset L := (X ∪ cert)ᶜ with hO
  have hOc : (↑O : Set L) = (↑X ∪ ↑cert)ᶜ := by rw [hO, coe_compl, coe_union]
  have hXO : Disjoint X O := Finset.disjoint_left.2 fun y hy hy' => (mem_compl.1 hy') (mem_union_left _ hy)
  have hcO : Disjoint (↑cert : Set L) ↑O := by
    rw [hOc]
    exact disjoint_compl_right.mono_left Set.subset_union_right
  have hrest : ((↑X : Set L) ∪ ↑O)ᶜ = ↑cert := by
    ext y
    by_cases hy : y ∈ X
    · simp [hO, hy, Finset.disjoint_left.1 hXc hy]
    · simp [hO, hy]
  let K := (Finset.Ico a b).image bucket
  let B := Actual.uncertified ends cert X
  let sig : ↥B × ↥(Finset.Ico a b) → Ω → Set (ℕ × L × ℕ × ℕ) := fun i r =>
    (S r).signal (link r) i.1 (Actual.outer ends X i.1) i.2
  have hfac : ∀ r r', ((S r).seen '' ↑(record r), (fun k : ↥K => J r k), fun i => sig i r) =
      ((S r').seen '' ↑(record r'), (fun k : ↥K => J r' k), fun i => sig i r') →
      (S r).acts ↑O (Set.Ico a b) = (S r').acts ↑O (Set.Ico a b) := by
    intro r r' hΦ
    simp only [Prod.mk.injEq] at hΦ
    obtain ⟨hrec, hj, hsig⟩ := hΦ
    have hcin : ∀ r r', (S r).inputs ↑cert = (S r').inputs ↑cert := fun r r' => congrArg Prod.fst (hh r r').1
    have hcpre : (S r).crossing (↑cert)ᶜ ↑cert (Set.Iio a) = (S r').crossing (↑cert)ᶜ ↑cert (Set.Iio a) :=
      congrArg (fun p => p.2.2) (hh r r').1
    have hs0 : (S r).sent ↑cert (Set.Iio a) = (S r').sent ↑cert (Set.Iio a) :=
      congrArg (fun p => p.2.1) (Actual.global_of_causal hc cert hcin a r r' hcpre)
    have hs1 : (S r).sent ↑cert (Set.Ico a b) = (S r').sent ↑cert (Set.Ico a b) :=
      hJ a b r r' hrec fun t ht => congrFun hj ⟨bucket t, mem_image_of_mem _ (Finset.mem_Ico.2 ht)⟩
    have hcert : (S r).crossing ↑cert ↑O (Set.Ico a b) = (S r').crossing ↑cert ↑O (Set.Ico a b) := by
      rw [Actual.crossing_sent (S r) (hr r).1 hcO, Actual.crossing_sent (S r') (hr r').1 hcO, hs0, hs1]
    have hX : (S r).crossing ↑X ↑O (Set.Ico a b) = (S r').crossing ↑X ↑O (Set.Ico a b) := by
      rw [hOc]
      exact (Actual.crossing_signal (hr r).2 (hr r').2 hXc fun e he u hu =>
          congrFun hsig ⟨⟨e, he⟩, ⟨u, Finset.mem_Ico.2 hu⟩⟩).antisymm
        (Actual.crossing_signal (hr r').2 (hr r).2 hXc fun e he u hu =>
          (congrFun hsig ⟨⟨e, he⟩, ⟨u, Finset.mem_Ico.2 hu⟩⟩).symm)
    have hOin : ∀ r r', (S r).inputs ↑O = (S r').inputs ↑O := fun r r' => congrArg Prod.fst (hh r r').2
    have hOpre : (S r).crossing (↑O)ᶜ ↑O (Set.Iio a) = (S r').crossing (↑O)ᶜ ↑O (Set.Iio a) :=
      congrArg (fun p => p.2.2) (hh r r').2
    have hall : (S r).acts ↑O (Set.Iio b) = (S r').acts ↑O (Set.Iio b) := by
      refine Actual.global_of_causal hc O hOin b r r' ?_
      rw [Actual.crossing_split (S r) hXO (t₀ := a) (le_refl b),
        Actual.crossing_split (S r') hXO (t₀ := a) (le_refl b), hrest, hOpre, hX, hcert]
    rw [Actual.acts_restrict (S r) (I := Set.Iio b) Set.Ico_subset_Iio_self,
      Actual.acts_restrict (S r') (I := Set.Iio b) Set.Ico_subset_Iio_self, hall]
  refine (Actual.encard_range_le hfac).trans ?_
  have hsub : (Set.range fun r => ((S r).seen '' ↑(record r), (fun k : ↥K => J r k), fun i => sig i r)) ⊆
      (Set.range fun r => (S r).seen '' ↑(record r)) ×ˢ
        ((Set.univ : Set (↥K → Fin (2 ^ jitter))) ×ˢ Set.univ.pi fun i => Set.range (sig i)) := by
    rintro _ ⟨r, rfl⟩
    exact ⟨⟨r, rfl⟩, Set.mem_univ _, fun i _ => ⟨r, rfl⟩⟩
  refine (Set.encard_le_encard hsub).trans ?_
  rw [Set.encard_prod, Set.encard_prod, Set.encard_pi_eq_prod_encard, Set.encard_univ,
    ENat.card_eq_coe_fintype_card, Fintype.card_fun, Fintype.card_fin, Fintype.card_coe]
  have hsig : ∏ i, (Set.range (sig i)).encard ≤
      ((2 ^ (uncertifiedCapacity ends β cert X * (b - a)) : ℕ) : ℕ∞) := by
    refine (Finset.prod_le_prod fun i _ => hα i.1.1 (Actual.outer ends X i.1.1) i.2.1).trans (le_of_eq ?_)
    rw [← Nat.cast_prod, Finset.prod_pow_eq_pow_sum, Fintype.sum_prod_type, Actual.uncertifiedCapacity_eq,
      ← Finset.sum_coe_sort B, Finset.sum_mul]
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_coe, Nat.card_Ico, smul_eq_mul, mul_comm]
  calc _ ≤ (Set.range fun r => (S r).seen '' ↑(record r)).encard *
        ((((2 ^ jitter) ^ K.card : ℕ) : ℕ∞) * ((2 ^ (uncertifiedCapacity ends β cert X * (b - a)) : ℕ) : ℕ∞)) := by
        gcongr
    _ = _ := by rw [← Nat.cast_mul, ← pow_mul, ← pow_add]

end Accounting.SecurityProofs
