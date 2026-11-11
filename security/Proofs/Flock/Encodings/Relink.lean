module

public import Proofs.Flock.Encodings.Binding
public import Proofs.Flock.Claims.Sampling.Down
public import Proofs.Flock.Claims.Sampling.ProgReads

/-!
# The relation carries over the link (`relinkR`)

A trace in the sampling relation under the sum, relinked, is in it under the record, or the checks collide. Where the
roots don't encode `g`, the relinked trace's value there is the trace's: its openings at `g` are the trace's, in order,
each checked by the record as the sum checks it (`relink_val_of_not_enc`). Where they do, it is the first record opening
of the pairs at `g` that passes (`relink_val_of_enc`), which exists since the pairs link `g`. If the trace's value there
differs from the pair's encoding opening, the encoding's binding collides on the two; if that record opening's differs
from the pair's record opening, the record's does. Otherwise every value the trace opens, the relinked trace opens alike
(`hext`), so a proof block correct on the trace's values is correct on the relinked ones (`Prog.correct_mono`,
it reads only opened values): the relinked faulty set is a subset of the trace's (`faulty_mono`), and `M_δ` is closed
downward at every design (`design_M_down`: `P_a` is antitone in each one-stage design, and so in their product).
-/

@[expose] public section

namespace Claims.Encodings

open Claims
open Claims.Sampling

/-! ## The confidence set is closed downward, at every design -/

theorem strataPa_nonneg : ∀ (ss : List (ℕ × ℕ)) (o : ℕ) (B : Finset ℕ), 0 ≤ strataPa ss o B
  | [], _, _ => by simp [strataPa]
  | (n, k) :: ss, o, B => by
    simp only [strataPa]
    exact mul_nonneg (div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)) (strataPa_nonneg ss (o + n) B)

theorem strataPa_anti {B B' : Finset ℕ} (h : B' ⊆ B) :
    ∀ (ss : List (ℕ × ℕ)) (o : ℕ), strataPa ss o B ≤ strataPa ss o B'
  | [], _ => by simp [strataPa]
  | (n, k) :: ss, o => by
    simp only [strataPa]
    refine mul_le_mul ?_ (strataPa_anti h ss (o + n)) (strataPa_nonneg _ _ _)
      (div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _))
    refine div_le_div_of_nonneg_right ?_ (Nat.cast_nonneg _)
    exact_mod_cast Nat.choose_le_choose k (Nat.sub_le_sub_left (Finset.card_le_card (window_mono h)) _)

theorem oneStagePa_nonneg (ns : List ℕ) : ∀ (r : OneStage) (B : Finset ℕ), 0 ≤ r.Pa ns B
  | .all, B => by
    simp only [OneStage.Pa]
    split_ifs <;> simp
  | .subset K, B => by
    simp only [OneStage.Pa]
    exact div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)
  | .bernoulli num den, B => by
    simp only [OneStage.Pa]
    exact pow_nonneg (div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)) _
  | .stratified K, B => by
    simp only [OneStage.Pa]
    exact strataPa_nonneg _ _ _
  | .work K w f, B => by
    simp only [OneStage.Pa]
    exact strataPa_nonneg _ _ _

/-- **`P_a` is antitone** in each one-stage design. -/
theorem oneStagePa_anti (ns : List ℕ) {B B' : Finset ℕ} (h : B' ⊆ B) : ∀ r : OneStage, r.Pa ns B ≤ r.Pa ns B'
  | .all => by
    simp only [OneStage.Pa]
    by_cases h1 : B = ∅
    · have h2 : B' = ∅ := Finset.subset_empty.1 (by rwa [h1] at h)
      simp [h1, h2]
    · simp only [h1, ↓reduceIte]
      split_ifs <;> simp
  | .subset K => by
    simp only [OneStage.Pa]
    refine div_le_div_of_nonneg_right ?_ (Nat.cast_nonneg _)
    exact_mod_cast Nat.choose_le_choose K (Nat.sub_le_sub_left (Finset.card_le_card h) _)
  | .bernoulli num den => by
    simp only [OneStage.Pa]
    exact pow_le_pow_of_le_one (div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _))
      (div_le_one_of_le₀ (by exact_mod_cast Nat.sub_le den num) (Nat.cast_nonneg _)) (Finset.card_le_card h)
  | .stratified K => by
    simp only [OneStage.Pa]
    exact strataPa_anti h _ 0
  | .work K w f => by
    simp only [OneStage.Pa]
    exact strataPa_anti h _ 0

theorem classesPa_nonneg : ∀ (rs : List OneStage) (cs : List (List (List (List ℕ)))) (o : ℕ) (B : Finset ℕ),
    0 ≤ classesPa rs cs o B
  | r :: rs, c :: cs, o, B => by
    simp only [classesPa]
    exact mul_nonneg (oneStagePa_nonneg _ _ _) (classesPa_nonneg rs cs _ B)
  | [], _, _, _ => by simp [classesPa]
  | _ :: _, [], _, _ => by simp [classesPa]

theorem classesPa_anti {B B' : Finset ℕ} (h : B' ⊆ B) :
    ∀ (rs : List OneStage) (cs : List (List (List (List ℕ)))) (o : ℕ), classesPa rs cs o B ≤ classesPa rs cs o B'
  | r :: rs, c :: cs, o => by
    simp only [classesPa]
    exact mul_le_mul (oneStagePa_anti _ (window_mono h) r) (classesPa_anti h rs cs _) (classesPa_nonneg _ _ _ _)
      (oneStagePa_nonneg _ _ _)
  | [], _, _ => by simp [classesPa]
  | _ :: _, [], _ => by simp [classesPa]

/-- **`P_a` is antitone**, at every design. -/
theorem design_Pa_anti (d : Design) (P : Partition) {B B' : Finset ℕ} (h : B' ⊆ B) : d.Pa P B ≤ d.Pa P B' :=
  classesPa_anti h _ _ _

/-- **`M_δ` is closed downward**, at every design. -/
theorem design_M_down {d : Design} {P : Partition} {δ : ℝ} {B B' : Finset ℕ} (h : B' ⊆ B) (hB : B ∈ d.M P δ) :
    B' ∈ d.M P δ := by
  have hB' : δ < d.Pa P B := hB
  show δ < d.Pa P B'
  exact lt_of_lt_of_le hB' (design_Pa_anti d P h)

/-! ## A trace's values -/

theorem find?_congr {α : Type} {p q : α → Bool} : ∀ {l : List α}, (∀ x ∈ l, p x = q x) → l.find? p = l.find? q
  | [], _ => rfl
  | x :: l, h => by
    simp only [List.find?_cons, h x List.mem_cons_self, find?_congr fun y hy => h y (List.mem_cons_of_mem _ hy)]

theorem find?_append_of_none {α : Type} {p : α → Bool} {l₁ l₂ : List α} (h : ∀ x ∈ l₂, p x = false) :
    (l₁ ++ l₂).find? p = l₁.find? p := by
  have h2 : l₂.find? p = none := List.find?_eq_none.2 fun x hx => by simp [h x hx]
  rw [List.find?_append, h2, Option.or_none]

theorem find?_append_of_left_none {α : Type} {p : α → Bool} {l₁ l₂ : List α} (h : ∀ x ∈ l₁, p x = false) :
    (l₁ ++ l₂).find? p = l₂.find? p := by
  have h1 : l₁.find? p = none := List.find?_eq_none.2 fun x hx => by simp [h x hx]
  rw [List.find?_append, h1, Option.none_or]

/-- A value the trace opens is an opening of it at the position that passes. -/
theorem val_eq_some {H : Msg → Msg} {Cm : CommitmentScheme Bool} {s : AuditStatement} {w : Wit} {g : ℕ} {b : Bool}
    (h : w.val H Cm s g = some b) : ∃ o ∈ w.vals, o.pos = g ∧ Cm.Opens H s.roots o = true ∧ o.val = b := by
  unfold Wit.val at h
  obtain ⟨o, ho, rfl⟩ := Option.map_eq_some_iff.1 h
  have hp := List.find?_some ho
  simp only [Bool.and_eq_true, beq_iff_eq] at hp
  exact ⟨o, List.mem_of_find?_eq_some ho, hp.1, hp.2, rfl⟩

/-- Where the roots don't encode `g`, the relinked trace's value is the trace's under the sum. -/
theorem relink_val_of_not_enc (H : Msg → Msg) (enc : List Msg → ℕ → Bool) (Ce Cm : CommitmentScheme Bool)
    (s : AuditStatement) (w : Wit) (ℓ : List (Opening Bool × Opening Bool)) {g : ℕ} (hg : enc s.roots g = false) :
    (relink enc s.roots w ℓ).val H Cm s g = w.val H (byPos enc Ce Cm) s g := by
  have h2 : ∀ x ∈ (ℓ.map Prod.fst).filter (fun o => enc s.roots o.pos),
      (x.pos == g && Cm.Opens H s.roots x) = false := by
    intro x hx
    have hx' := (List.mem_filter.1 hx).2
    have hne : x.pos ≠ g := fun h => by simp [h, hg] at hx'
    simp [hne]
  simp only [Wit.val, relink]
  rw [find?_append_of_none h2, List.find?_filter]
  congr 1
  exact find?_congr fun o _ => by
    by_cases ho : o.pos = g
    · have he : enc s.roots o.pos = false := by rw [ho]; exact hg
      simp [ho, hg, byPos_Opens_false (Ce := Ce) (Cm := Cm) (H := H) he]
    · simp [ho]

/-- Where the roots encode `g`, the relinked trace's value is the first record opening of the pairs there that passes. -/
theorem relink_val_of_enc (H : Msg → Msg) (enc : List Msg → ℕ → Bool) (Cm : CommitmentScheme Bool)
    (s : AuditStatement) (w : Wit) (ℓ : List (Opening Bool × Opening Bool)) {g : ℕ} (hg : enc s.roots g = true) :
    (relink enc s.roots w ℓ).val H Cm s g =
      ((ℓ.map Prod.fst).find? fun o => o.pos == g && Cm.Opens H s.roots o).map (·.val) := by
  have h1 : ∀ x ∈ w.vals.filter (fun o => !enc s.roots o.pos), (x.pos == g && Cm.Opens H s.roots x) = false := by
    intro x hx
    have hx' := (List.mem_filter.1 hx).2
    have hne : x.pos ≠ g := fun h => by simp [h, hg] at hx'
    simp [hne]
  simp only [Wit.val, relink]
  rw [find?_append_of_left_none h1, List.find?_filter]
  congr 1
  exact find?_congr fun o _ => by
    by_cases ho : o.pos = g
    · simp [ho, hg]
    · simp [ho]

/-- A trace that opens every value another opens, alike, has no more faulty proof blocks. -/
theorem faulty_mono {H : Msg → Msg} {Cm Cm' : CommitmentScheme Bool} {s : AuditStatement} {P : Partition}
    {w w' : Wit} (hext : ∀ g b, w.val H Cm s g = some b → w'.val H Cm' s g = some b) :
    faulty H Cm' s P w' ⊆ faulty H Cm s P w := by
  intro v hv
  simp only [faulty, Finset.mem_filter, Faulty] at hv ⊢
  exact ⟨hv.1, fun ⟨u, hu, ι, hc⟩ => hv.2 ⟨u, hu, ι,
    Prog.correct_mono (Prog.refIn_mono hext) hc⟩⟩

/-- **The relation carries over the link.** -/
theorem relinkR : RelinkR := by
  intro H enc Ce Cm δ s P w ℓ hCe hCm hlk hR
  by_cases hcoll : LogColl ((w.vals.flatMap fun o => (((byPos enc Ce Cm).opens s.roots o).eval H).2) ++
      linkLog H Cm Ce s.roots ℓ)
  · exact Or.inr hcoll
  have hR' : faulty H (byPos enc Ce Cm) s P w ∈ s.design.M P δ := hR
  refine Or.inl (design_M_down (faulty_mono fun g b hb => ?_) hR')
  cases hg : enc s.roots g with
  | false =>
    rw [relink_val_of_not_enc H enc Ce Cm s w ℓ hg]
    exact hb
  | true =>
    rw [relink_val_of_enc H enc Cm s w ℓ hg]
    obtain ⟨o, ho, hop, hoo, rfl⟩ := val_eq_some hb
    obtain ⟨p, hp, hp1, hp2, hpv, hpm, hpe⟩ := linkedAt_some (hlk g hg)
    have hsome : ((ℓ.map Prod.fst).find? fun o => o.pos == g && Cm.Opens H s.roots o).isSome :=
      List.find?_isSome.2 ⟨p.1, List.mem_map_of_mem hp, by simp [hp1, hpm]⟩
    obtain ⟨q, hq⟩ := Option.isSome_iff_exists.1 hsome
    have hqp := List.find?_some hq
    simp only [Bool.and_eq_true, beq_iff_eq] at hqp
    obtain ⟨p', hp', rfl⟩ := List.mem_map.1 (List.mem_of_find?_eq_some hq)
    rw [hq]
    show some p'.1.val = some o.val
    have hoe : enc s.roots o.pos = true := by rw [hop]; exact hg
    have hoo' : Ce.Opens H s.roots o = true := by rw [← byPos_Opens_true (Cm := Cm) hoe]; exact hoo
    by_cases h1 : o.val = p.2.val
    · by_cases h2 : p'.1.val = p.1.val
      · rw [h2, hpv, h1]
      · refine absurd ((hCm s.roots p'.1 p.1 (by rw [hqp.1, hp1]) h2 hqp.2 hpm).mono fun x hx => ?_) hcoll
        refine List.mem_append_right _ ?_
        rcases List.mem_append.1 hx with hx | hx
        · exact linkLog_mem hp' (Or.inl hx)
        · exact linkLog_mem hp (Or.inl hx)
    · refine absurd ((hCe s.roots o p.2 (by rw [hop, hp2]) h1 hoo' hpe).mono fun x hx => ?_) hcoll
      rcases List.mem_append.1 hx with hx | hx
      · exact List.mem_append_left _ (List.mem_flatMap.2 ⟨o, ho, by rw [byPos_opens_true hoe]; exact hx⟩)
      · exact List.mem_append_right _ (linkLog_mem hp (Or.inr hx))

end Claims.Encodings
