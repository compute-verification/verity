module

public import Definitions.Claims.Tolerance
public import Proofs.Flock.Claims.Sampling.Escape

/-!
# The tolerance of a counts design, proved

* `countsPaLeProd`: a counts design misses `B` at most as `∏_s (1 − k_s/n_s)^{|B ∩ s|}` (`CountsPaLeProd`), stratum by
  stratum by `choose_div_le_pow_of_rate`, at counts `k_s ≤ n_s`.
* `toleranceOfCounts`: every set the counts design misses with probability at least δ holds declared use at most
  `tolerance` (`ToleranceOfCounts`).
* `toleranceCappedOfCounts`: and at most `toleranceCapped` (`ToleranceCappedOfCounts`); both from `countsBounds`.
-/

@[expose] public section

namespace Claims.Sampling

open Finset

/-- The miss bound's factors from slot `o`: `(1 − k/n)^{|B ∩ [o, o + n)|}` over the strata in order. -/
noncomputable def strataPow : List (ℕ × ℕ) → ℕ → Finset ℕ → ℝ
  | [], _, _ => 1
  | (n, k) :: ss, o, B => (1 - (k : ℝ) / n) ^ (window o n B).card * strataPow ss (o + n) B

/-- The slots of `B` in stratum `s` of the strata `ns` laid from slot `o`. -/
def segOf (ns : List ℕ) (o s : ℕ) (B : Finset ℕ) : Finset ℕ :=
  B.filter fun v => o + (ns.take s).sum ≤ v ∧ v < o + (ns.take s).sum + ns[s]!

theorem share_le_one {n k : ℕ} (h : k ≤ n) : (k : ℝ) / n ≤ 1 := by
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · simp
  · exact (div_le_one (by exact_mod_cast hn)).2 (by exact_mod_cast h)

theorem strataPa_le_strataPow : ∀ (ss : List (ℕ × ℕ)) (o : ℕ) (B : Finset ℕ), (∀ p ∈ ss, p.2 ≤ p.1) →
    strataPa ss o B ≤ strataPow ss o B
  | [], _, _, _ => le_refl _
  | (n, k) :: ss, o, B, hk => by
    have h := hk (n, k) (by simp)
    have hw : (window o n B).card ≤ n := by simpa using Finset.card_le_card (window_subset o n B)
    have hr : (k : ℝ) / n * n ≤ k := by
      rcases Nat.eq_zero_or_pos n with rfl | hn
      · simp
      · rw [div_mul_cancel₀ _ (by exact_mod_cast hn.ne')]
    have h1 := share_le_one h
    rw [strataPa, strataPow]
    exact mul_le_mul (choose_div_le_pow_of_rate hw h hr)
      (strataPa_le_strataPow ss (o + n) B fun p hp => hk p (by simp [hp])) (strataPa_nonneg ss (o + n) B)
      (pow_nonneg (by linarith) _)

theorem card_window (o n : ℕ) (B : Finset ℕ) :
    (window o n B).card = (B.filter fun v => o ≤ v ∧ v < o + n).card := by
  rw [window, Finset.card_image_of_injOn fun a ha b hb hab => by
    have := (Finset.mem_filter.1 (Finset.mem_coe.1 ha)).2
    have := (Finset.mem_filter.1 (Finset.mem_coe.1 hb)).2
    have h' : a - o = b - o := hab
    omega]

theorem segOf_zero (n : ℕ) (ns : List ℕ) (o : ℕ) (B : Finset ℕ) :
    segOf (n :: ns) o 0 B = B.filter fun v => o ≤ v ∧ v < o + n := by
  simp [segOf]

theorem segOf_succ (n : ℕ) (ns : List ℕ) (o s : ℕ) (B : Finset ℕ) :
    segOf (n :: ns) o (s + 1) B = segOf ns (o + n) s B := by
  have : (n :: ns)[s + 1]! = ns[s]! := by simp
  unfold segOf
  simp only [List.take_succ_cons, List.sum_cons, this]
  ext v
  simp only [Finset.mem_filter]
  constructor <;> rintro ⟨h1, h2, h3⟩ <;> exact ⟨h1, by omega, by omega⟩

theorem strataPow_zip : ∀ (ns ks : List ℕ) (o : ℕ) (B : Finset ℕ), ks.length = ns.length →
    strataPow (ns.zip ks) o B = ∏ s ∈ range ns.length, (1 - (ks[s]! : ℝ) / ns[s]!) ^ (segOf ns o s B).card
  | [], ks, o, B, _ => by simp [strataPow]
  | n :: ns, [], o, B, h => by simp at h
  | n :: ns, k :: ks, o, B, h => by
    rw [List.zip_cons_cons, strataPow, strataPow_zip ns ks (o + n) B (by simpa using h), List.length_cons,
      Finset.prod_range_succ', segOf_zero, card_window, mul_comm]
    congr 1
    · refine Finset.prod_congr rfl fun s _ => ?_
      rw [segOf_succ]
      simp

/-- **`CountsPaLeProd`, proved**: `choose_div_le_pow_of_rate` at each stratum's own share `k_s/n_s`. It reads
`k_s ≤ n_s` only; a stratum with `n_s = 0` or `k_s = 0` is a factor 1. -/
theorem countsPaLeProd : CountsPaLeProd := by
  intro ns ks B hlen hk
  rw [OneStage.Pa]
  refine (strataPa_le_strataPow (ns.zip ks) 0 B hk).trans (le_of_eq ?_)
  rw [strataPow_zip ns ks 0 B hlen]
  refine Finset.prod_congr rfl fun s _ => ?_
  simp [segOf, inStratum, stratumStart]

theorem sum_segOf : ∀ (ns : List ℕ) (o : ℕ) (B : Finset ℕ) (f : ℕ → ℕ), B ⊆ Ico o (o + ns.sum) →
    ∑ v ∈ B, f v = ∑ s ∈ range ns.length, ∑ v ∈ segOf ns o s B, f v
  | [], o, B, f, hB => by
    have : B = ∅ := Finset.subset_empty.1 (by simpa using hB)
    simp [this]
  | n :: ns, o, B, f, hB => by
    rw [← Finset.sum_filter_add_sum_filter_not B (fun v => v < o + n), List.length_cons, Finset.sum_range_succ',
      add_comm]
    congr 1
    · rw [sum_segOf ns (o + n) _ f fun v hv => by
        obtain ⟨h1, h1'⟩ := Finset.mem_filter.1 hv
        have h2 := Finset.mem_Ico.1 (hB h1)
        simp only [List.sum_cons] at h2
        exact Finset.mem_Ico.2 ⟨by omega, by omega⟩]
      refine Finset.sum_congr rfl fun s _ => Finset.sum_congr ?_ fun _ _ => rfl
      rw [segOf_succ]
      ext v
      simp only [segOf, Finset.mem_filter]
      constructor
      · rintro ⟨⟨h1, _⟩, h2, h3⟩; exact ⟨h1, h2, h3⟩
      · rintro ⟨h1, h2, h3⟩; exact ⟨⟨h1, by omega⟩, h2, h3⟩
    · rw [segOf_zero]
      refine Finset.sum_congr (Finset.filter_congr fun v hv => ?_) fun _ _ => rfl
      have := Finset.mem_Ico.1 (hB hv)
      constructor
      · intro h; exact ⟨this.1, h⟩
      · intro h; exact h.2

theorem le_foldr_max : ∀ (l : List ℕ) {x : ℕ}, x ∈ l → x ≤ l.foldr max 0
  | [], _, h => by simp at h
  | a :: l, x, h => by
    rw [List.foldr_cons]
    rcases List.mem_cons.1 h with rfl | h
    · exact le_max_left _ _
    · exact (le_foldr_max l h).trans (le_max_right _ _)

theorem sum_range_list (n : ℕ) (g : ℕ → ℕ) : ((List.range n).map g).sum = ∑ j ∈ range n, g j := by
  rw [← List.sum_toFinset g (List.nodup_range), List.toFinset_range]

theorem sum_filter_list (n : ℕ) (p : ℕ → Prop) [DecidablePred p] (g : ℕ → ℕ) :
    (((List.range n).filter fun s => decide (p s)).map g).sum = ∑ s ∈ (range n).filter p, g s := by
  rw [← List.sum_toFinset g ((List.nodup_range).filter _)]
  congr 1
  ext s
  simp

theorem ceil_div_ge (a d : ℕ) (hd : 0 < d) : a ≤ (a + d - 1) / d * d := by
  have h1 := Nat.div_add_mod (a + d - 1) d
  have h2 := Nat.mod_lt (a + d - 1) hd
  rw [mul_comm]
  generalize (a + d - 1) % d = r at h1 h2
  generalize d * ((a + d - 1) / d) = m at h1 ⊢
  omega

/-- **What a missed set holds, stratum by stratum**: a set the counts design misses with probability at least δ has, in
each stratum it draws, `t_s` blocks with `∑ t_s k_s/n_s ≤ ln(1/δ)` (`countsPaLeProd`, `1 − x ≤ e^{−x}`). So the strata it
draws hold at most the largest `stratumTerm`, each of them at most its own term and its whole declared use, and a
stratum it never draws holds at most its whole declared use. -/
theorem countsBounds {ns ks : List ℕ} {w : ℕ → ℕ} {δ : ℝ} {L b : ℕ} (hδ : 0 < δ) (hlen : ks.length = ns.length)
    (hk : ∀ p ∈ ns.zip ks, p.2 ≤ p.1) (hL : Real.log (1 / δ) ≤ (L : ℝ) / 2 ^ b) {B : Finset ℕ}
    (hB : B ⊆ range ns.sum) (hPa : δ ≤ (OneStage.counts ks).Pa ns B) :
    ∑ v ∈ B, w v ≤ (((List.range ns.length).filter fun s => ks[s]! = 0).map (stratumUse ns w)).sum +
        ∑ s ∈ (range ns.length).filter (fun s => ks[s]! ≠ 0), ∑ v ∈ inStratum ns s B, w v ∧
      ∑ s ∈ (range ns.length).filter (fun s => ks[s]! ≠ 0), ∑ v ∈ inStratum ns s B, w v ≤
        (((List.range ns.length).filter fun s => ks[s]! ≠ 0).map (stratumTerm ns ks w L b)).foldr max 0 ∧
      ∀ s ∈ (range ns.length).filter (fun s => ks[s]! ≠ 0),
        ∑ v ∈ inStratum ns s B, w v ≤ min (stratumTerm ns ks w L b s) (stratumUse ns w s) := by
  have hget : ∀ s, s < ns.length → ks[s]! ≤ ns[s]! := fun s hs => by
    have hs' : s < ks.length := hlen ▸ hs
    have hz : s < (ns.zip ks).length := by simp [hlen, hs]
    have := hk _ (List.getElem_mem hz)
    rw [List.getElem_zip] at this
    simpa [hs, hs'] using this
  -- the strata's split of B
  have hsplit : ∑ v ∈ B, w v = ∑ s ∈ range ns.length, ∑ v ∈ inStratum ns s B, w v := by
    rw [sum_segOf ns 0 B w (by simpa using hB)]
    refine Finset.sum_congr rfl fun s _ => Finset.sum_congr ?_ fun _ _ => rfl
    simp [segOf, inStratum, stratumStart]
  have hsub : ∀ s, inStratum ns s B ⊆ Ico (stratumStart ns s) (stratumStart ns s + ns[s]!) := fun s v hv => by
    have := (Finset.mem_filter.1 hv).2
    exact Finset.mem_Ico.2 this
  have hwhole : ∀ s, ∑ v ∈ inStratum ns s B, w v ≤ stratumUse ns w s := fun s => by
    calc ∑ v ∈ inStratum ns s B, w v ≤ ∑ v ∈ Ico (stratumStart ns s) (stratumStart ns s + ns[s]!), w v :=
          Finset.sum_le_sum_of_subset (hsub s)
      _ = stratumUse ns w s := by
          rw [Finset.sum_Ico_eq_sum_range, stratumUse, sum_range_list]
          simp
  have hmax : ∀ s, ∀ v ∈ inStratum ns s B, w v ≤ stratumUseMax ns w s := fun s v hv => by
    have h := Finset.mem_Ico.1 (hsub s hv)
    apply le_foldr_max
    exact List.mem_map.2 ⟨v - stratumStart ns s, List.mem_range.2 (by omega), by congr 1; omega⟩
  have hcnt : ∀ s, ∑ v ∈ inStratum ns s B, w v ≤ (inStratum ns s B).card * stratumUseMax ns w s := fun s => by
    simpa using Finset.sum_le_card_nsmul _ _ _ (hmax s)
  -- the miss bound gives `∑ t_s k_s/n_s ≤ ln(1/δ)`
  set r : ℕ → ℝ := fun s => (ks[s]! : ℝ) / ns[s]! with hr
  set t : ℕ → ℕ := fun s => (inStratum ns s B).card with ht
  have hr01 : ∀ s ∈ range ns.length, 0 ≤ r s ∧ r s ≤ 1 := fun s hs =>
    ⟨by positivity, share_le_one (hget s (Finset.mem_range.1 hs))⟩
  have hexp : ∏ s ∈ range ns.length, (1 - r s) ^ t s ≤ Real.exp (-(∑ s ∈ range ns.length, r s * t s)) := by
    rw [← Finset.sum_neg_distrib, Real.exp_sum]
    refine Finset.prod_le_prod₀ (fun s hs => pow_nonneg (by linarith [(hr01 s hs).2]) _) fun s hs => ?_
    calc (1 - r s) ^ t s ≤ Real.exp (-r s) ^ t s :=
          pow_le_pow_left₀ (by linarith [(hr01 s hs).2]) (by linarith [Real.add_one_le_exp (-r s)]) _
      _ = Real.exp (-(r s * t s)) := by rw [← Real.exp_nat_mul]; ring_nf
  have hlog : ∑ s ∈ range ns.length, r s * t s ≤ Real.log (1 / δ) := by
    have h1 : δ ≤ Real.exp (-(∑ s ∈ range ns.length, r s * t s)) :=
      hPa.trans ((countsPaLeProd ns ks B hlen hk).trans hexp)
    have h2 := Real.log_le_log hδ h1
    rw [Real.log_exp] at h2
    rw [one_div, Real.log_inv]
    linarith
  -- the tolerance's parts
  set D := (range ns.length).filter fun s => ks[s]! ≠ 0 with hD
  set lam : ℝ := (L : ℝ) / 2 ^ b with hlam
  set ratio := (((List.range ns.length).filter fun s => ks[s]! ≠ 0).map (stratumTerm ns ks w L b)).foldr max 0
    with hratio
  have hterm' : ∀ s ∈ D, (stratumUseMax ns w s : ℝ) * ns[s]! / ks[s]! * lam ≤ stratumTerm ns ks w L b s :=
      fun s hs => by
    obtain ⟨-, hs2⟩ := Finset.mem_filter.1 hs
    unfold stratumTerm
    have hkpos : 0 < ks[s]! * 2 ^ b := Nat.mul_pos (Nat.pos_of_ne_zero hs2) (by positivity)
    have hle := ceil_div_ge (L * stratumUseMax ns w s * ns[s]!) (ks[s]! * 2 ^ b) hkpos
    have hle' : ((L * stratumUseMax ns w s * ns[s]! : ℕ) : ℝ) ≤
        (((L * stratumUseMax ns w s * ns[s]! + ks[s]! * 2 ^ b - 1) / (ks[s]! * 2 ^ b) : ℕ) : ℝ) *
          ((ks[s]! * 2 ^ b : ℕ) : ℝ) := by exact_mod_cast hle
    have hkR : (0 : ℝ) < ((ks[s]! * 2 ^ b : ℕ) : ℝ) := by exact_mod_cast hkpos
    have hk0 : (0 : ℝ) < ks[s]! := by exact_mod_cast Nat.pos_of_ne_zero hs2
    calc (stratumUseMax ns w s : ℝ) * ns[s]! / ks[s]! * lam
        = ((L * stratumUseMax ns w s * ns[s]! : ℕ) : ℝ) / ((ks[s]! * 2 ^ b : ℕ) : ℝ) := by
          rw [hlam]; push_cast; field_simp
      _ ≤ _ := (div_le_iff₀ hkR).2 hle'
  have hcs : ∀ s ∈ D, (stratumUseMax ns w s : ℝ) * ns[s]! / ks[s]! * lam ≤ ratio := fun s hs => by
    obtain ⟨hs1, hs2⟩ := Finset.mem_filter.1 hs
    have hmem : stratumTerm ns ks w L b s ≤ ratio :=
      le_foldr_max _ (List.mem_map.2 ⟨s, List.mem_filter.2 ⟨List.mem_range.2 (Finset.mem_range.1 hs1),
        by simpa using hs2⟩, rfl⟩)
    exact (hterm' s hs).trans (by exact_mod_cast hmem)
  have hterm : ∀ s ∈ D, (∑ v ∈ inStratum ns s B, w v : ℝ) ≤
      r s * t s * ((stratumUseMax ns w s : ℝ) * ns[s]! / ks[s]!) := fun s hs => by
    obtain ⟨hs1, hs2⟩ := Finset.mem_filter.1 hs
    have hk0 : (ks[s]! : ℝ) ≠ 0 := by exact_mod_cast hs2
    have hn0 : (ns[s]! : ℝ) ≠ 0 := by
      have := hget s (Finset.mem_range.1 hs1)
      exact_mod_cast (show ns[s]! ≠ 0 by omega)
    have h := hcnt s
    have h' : (∑ v ∈ inStratum ns s B, w v : ℝ) ≤ (t s : ℝ) * stratumUseMax ns w s := by exact_mod_cast h
    calc _ ≤ (t s : ℝ) * stratumUseMax ns w s := h'
      _ = r s * t s * ((stratumUseMax ns w s : ℝ) * ns[s]! / ks[s]!) := by
        rw [hr]; field_simp
  have hnn : ∀ s ∈ D, 0 ≤ r s * t s := fun s hs =>
    mul_nonneg (hr01 s (Finset.mem_filter.1 hs).1).1 (Nat.cast_nonneg _)
  have hDsum : ∑ s ∈ D, r s * t s ≤ lam :=
    (Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _) fun s hs _ =>
      mul_nonneg (hr01 s hs).1 (Nat.cast_nonneg _)).trans (hlog.trans hL)
  have hdrawn : ∑ s ∈ D, (∑ v ∈ inStratum ns s B, w v : ℝ) ≤ ratio := by
    refine (Finset.sum_le_sum hterm).trans ?_
    rcases (show (0 : ℝ) ≤ lam by positivity).lt_or_eq with hpos | hzero
    · calc ∑ s ∈ D, r s * t s * ((stratumUseMax ns w s : ℝ) * ns[s]! / ks[s]!)
          ≤ ∑ s ∈ D, r s * t s * (ratio / lam) := Finset.sum_le_sum fun s hs =>
            mul_le_mul_of_nonneg_left ((le_div_iff₀ hpos).2 (hcs s hs)) (hnn s hs)
        _ = (∑ s ∈ D, r s * t s) * (ratio / lam) := by rw [Finset.sum_mul]
        _ ≤ lam * (ratio / lam) := mul_le_mul_of_nonneg_right hDsum (by positivity)
        _ = ratio := by field_simp
    · have h0 : ∀ s ∈ D, r s * t s = 0 := fun s hs =>
        le_antisymm ((Finset.single_le_sum hnn hs).trans (hDsum.trans hzero.symm.le)) (hnn s hs)
      calc ∑ s ∈ D, r s * t s * ((stratumUseMax ns w s : ℝ) * ns[s]! / ks[s]!)
          = 0 := Finset.sum_eq_zero fun s hs => by rw [h0 s hs, zero_mul]
        _ ≤ ratio := Nat.cast_nonneg _
  refine ⟨?_, by exact_mod_cast hdrawn, fun s hs => le_min ?_ (hwhole s)⟩
  · rw [hsplit, ← Finset.sum_filter_add_sum_filter_not (range ns.length) (fun s => ks[s]! = 0)]
    refine Nat.add_le_add_right ?_ _
    rw [sum_filter_list]
    exact Finset.sum_le_sum fun s _ => hwhole s
  · have h1 := hterm s hs
    have h2 : r s * t s ≤ lam := (Finset.single_le_sum hnn hs).trans hDsum
    have h3 : (0 : ℝ) ≤ (stratumUseMax ns w s : ℝ) * ns[s]! / ks[s]! := by positivity
    have h4 : (∑ v ∈ inStratum ns s B, w v : ℝ) ≤ stratumTerm ns ks w L b s :=
      h1.trans ((mul_le_mul_of_nonneg_right h2 h3).trans (by rw [mul_comm]; exact hterm' s hs))
    exact_mod_cast h4

/-- **`ToleranceOfCounts`, proved**, from `countsBounds`. -/
theorem toleranceOfCounts : ToleranceOfCounts := by
  intro ns ks w δ L b hδ hlen hk hL B hB hPa
  obtain ⟨h1, h2, -⟩ := countsBounds hδ hlen hk hL hB hPa
  unfold tolerance
  dsimp only
  exact h1.trans (Nat.add_le_add_left h2 _)

/-- **`ToleranceCappedOfCounts`, proved**, from `countsBounds`: the drawn strata hold at most the largest term and at
most the sum of their capped terms. -/
theorem toleranceCappedOfCounts : ToleranceCappedOfCounts := by
  intro ns ks w δ L b hδ hlen hk hL B hB hPa
  obtain ⟨h1, h2, h3⟩ := countsBounds hδ hlen hk hL hB hPa
  unfold toleranceCapped
  dsimp only
  refine h1.trans (Nat.add_le_add_left (le_min h2 ?_) _)
  rw [sum_filter_list]
  exact Finset.sum_le_sum h3

/-- **The whole-population cap**: the capped tolerance is at most the tolerance and at most the population's whole
declared use. -/
theorem toleranceCapped_le (ns ks : List ℕ) (w : ℕ → ℕ) (L b : ℕ) :
    toleranceCapped ns ks w L b ≤ tolerance ns ks w L b ∧
      toleranceCapped ns ks w L b ≤ ((List.range ns.length).map (stratumUse ns w)).sum := by
  unfold toleranceCapped tolerance
  dsimp only
  refine ⟨Nat.add_le_add_left (min_le_left _ _) _, ?_⟩
  rw [sum_range_list, sum_filter_list, sum_filter_list]
  calc _ ≤ ∑ s ∈ (range ns.length).filter (fun s => ks[s]! = 0), stratumUse ns w s +
        ∑ s ∈ (range ns.length).filter (fun s => ¬ ks[s]! = 0), stratumUse ns w s :=
        Nat.add_le_add_left ((min_le_right _ _).trans (Finset.sum_le_sum fun s _ => min_le_right _ _)) _
    _ = _ := Finset.sum_filter_add_sum_filter_not _ _ _

end Claims.Sampling
