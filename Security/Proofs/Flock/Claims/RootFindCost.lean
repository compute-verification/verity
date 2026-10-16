import Proofs.Flock.Claims.RootFindCorrect

/-!
# The root finder's operation count

Every bound holds for every input; only `split`'s uses the shape of the pieces a split returns (`trySplit_some`).

* Modulo `g` with at most `L` coefficients, a squaring takes at most `6L² + L` operations and a trace step `6L² + 2L`
  (`sqLoop_bound`, `traceLoop_bound`), so `trySplit` takes at most `(8n + 14)(L + 1)³` (`ops_trySplit_le`) and
  `firstSplit` `|basis|` times that.
* `split` on `g` with `m ≥ 2` coefficients takes at most `(2m − 3)` times `splitStep`, `firstSplit`'s bound plus one:
  each split node costs one `firstSplit`, a linear piece one negation, and pieces of `m₁ + m₂ = m + 1` coefficients
  cost `(2m₁ − 3) + (2m₂ − 3) = 2m − 4` steps (`ops_split_le`).
* `rootProduct` on `f` with `d` coefficients takes at most `(7n + 20)(d + 3)³` and returns at most `d + 2`
  coefficients (`rootProduct_bound`), so `rootsC` takes at most `costBound n |basis| d` (`rootsC_ops_le`).
-/

namespace Claims.RootFind

open Polynomial

section Cost

variable {F : Type} [Field F]

theorem ops_divmod_le [DecidableEq F] (p g : List F) : (divmod p g).ops ≤ 2 * p.length * g.length + g.length := by
  by_cases hne : g = []
  · subst hne; simp [divmod_nil]
  · exact (divmod_spec p g hne).2.2.2.2

theorem length_rem_lt [DecidableEq F] {p g : List F} (hne : g ≠ []) : (divmod p g).val.2.length < g.length :=
  (divmod_spec p g hne).2.2.2.1

theorem sqLoop_bound [DecidableEq F] {g : List F} (hne : g ≠ []) {L : ℕ} (hg : g.length ≤ L) :
    ∀ (j : ℕ) (u : List F), u.length ≤ L →
      (sqLoop g j u).val.length ≤ L ∧ (sqLoop g j u).ops ≤ j * (6 * L * L + L)
  | 0, u, hu => by simp [sqLoop, hu]
  | j + 1, u, hu => by
    have hm := length_mul_le u u
    have hm' := ops_mul_le u u
    have hd := ops_divmod_le (mul u u).val g
    have hr := length_rem_lt (p := (mul u u).val) hne
    obtain ⟨ih1, ih2⟩ := sqLoop_bound hne hg j _ (by omega : (divmod (mul u u).val g).val.2.length ≤ L)
    simp only [sqLoop, Counted.val_bind, Counted.ops_bind]
    refine ⟨ih1, ?_⟩
    nlinarith [Nat.mul_le_mul hu hu, Nat.mul_le_mul (by omega : (mul u u).val.length ≤ 2 * L) hg]

theorem traceLoop_bound [DecidableEq F] {g : List F} (hne : g ≠ []) {L : ℕ} (hg : g.length ≤ L) :
    ∀ (j : ℕ) (acc u : List F), acc.length ≤ L → u.length ≤ L →
      (traceLoop g j acc u).val.length ≤ L ∧ (traceLoop g j acc u).ops ≤ j * (6 * L * L + 2 * L)
  | 0, acc, u, ha, _ => by simp [traceLoop, ha]
  | j + 1, acc, u, ha, hu => by
    have hal := length_add acc u
    have hao := ops_add acc u
    have hm := length_mul_le u u
    have hm' := ops_mul_le u u
    have hd := ops_divmod_le (mul u u).val g
    have hr := length_rem_lt (p := (mul u u).val) hne
    obtain ⟨ih1, ih2⟩ := traceLoop_bound hne hg j (add acc u).val _ (by omega)
      (by omega : (divmod (mul u u).val g).val.2.length ≤ L)
    simp only [traceLoop, Counted.val_bind, Counted.ops_bind]
    refine ⟨ih1, ?_⟩
    have : min acc.length u.length ≤ L := by omega
    nlinarith [Nat.mul_le_mul hu hu, Nat.mul_le_mul (by omega : (mul u u).val.length ≤ 2 * L) hg]

/-- One `trySplit` on `g` with at most `L` coefficients. -/
theorem ops_trySplit_le [DecidableEq F] (n : ℕ) {g : List F} (hne : g ≠ []) {L : ℕ} (hg : g.length ≤ L) (β : F) :
    (trySplit n g β).ops ≤ (8 * n + 14) * (L + 1) ^ 3 := by
  simp only [trySplit, Counted.ops_bind]
  have d1 : (divmod [0, β] g).ops ≤ 5 * L := by
    have := ops_divmod_le ([0, β] : List F) g
    simp only [List.length_cons, List.length_nil] at this
    omega
  have r1 := length_rem_lt (p := [0, β]) hne
  obtain ⟨t1, t2⟩ := traceLoop_bound hne hg n [] (divmod [0, β] g).val.2 (by simp) (by omega)
  have c1 := ops_gcd_le hg t1
  have l1 := length_gcd_le g (traceLoop g n [] (divmod [0, β] g).val.2).val
  split_ifs with hc
  · have m1 := ops_monic_le (gcd g (traceLoop g n [] (divmod [0, β] g).val.2).val).val
    have m2 := length_monic (gcd g (traceLoop g n [] (divmod [0, β] g).val.2).val).val
    have d2 := ops_divmod_le g (monic (gcd g (traceLoop g n [] (divmod [0, β] g).val.2).val).val).val
    simp only [Counted.ops_bind, Counted.ops_pure]
    have hh : (monic (gcd g (traceLoop g n [] (divmod [0, β] g).val.2).val).val).val.length ≤ L := by omega
    nlinarith [Nat.mul_le_mul hg hh, Nat.zero_le (n * L * L * L), Nat.zero_le (n * L * L), Nat.zero_le (n * L),
      Nat.zero_le (L * L * L), Nat.zero_le (L * L)]
  · simp only [Counted.ops_pure]
    nlinarith [Nat.zero_le (n * L * L * L), Nat.zero_le (n * L * L), Nat.zero_le (n * L), Nat.zero_le (L * L * L),
      Nat.zero_le (L * L)]

theorem ops_firstSplit_le [DecidableEq F] (n : ℕ) (g : List F) {T : ℕ} (hT : ∀ β, (trySplit n g β).ops ≤ T) :
    ∀ basis : List F, (firstSplit n g basis).ops ≤ basis.length * T
  | [] => by simp [firstSplit]
  | β :: basis => by
    have ih := ops_firstSplit_le n g hT basis
    have h1 := hT β
    simp only [firstSplit, Counted.ops_bind, List.length_cons, Nat.succ_mul]
    cases (trySplit n g β).val with
    | some hk => simp only [Counted.ops_pure]; omega
    | none => simp only; omega

/-- A split node's bound: one `firstSplit` on `m` coefficients, plus one. -/
def splitStep (n b m : ℕ) : ℕ := b * ((8 * n + 14) * (m + 1) ^ 3) + 1

theorem splitStep_mono (n b : ℕ) {m m' : ℕ} (h : m ≤ m') : splitStep n b m ≤ splitStep n b m' := by
  unfold splitStep
  gcongr

theorem ops_split_le [DecidableEq F] (n : ℕ) (basis : List F) :
    ∀ (fuel : ℕ) (g : List F), Norm g → (toPoly g).Monic →
      (split n basis fuel g).ops ≤ (2 * g.length - 3) * splitStep n basis.length g.length
  | 0, g, _, _ => by simp [split]
  | fuel + 1, g, hg, hgm => by
    by_cases h1 : g.length ≤ 1
    · simp only [split, ite_eq_left h1, Counted.ops_pure]
      exact Nat.zero_le _
    · by_cases h2 : g.length = 2
      · simp only [split, ite_eq_right h1, ite_eq_left h2, Counted.ops_bind, ops_op, Counted.ops_pure]
        rw [h2, splitStep]
        omega
      · have hne : g ≠ [] := by rintro rfl; simp at h1
        have hfs := ops_firstSplit_le n g (fun β => ops_trySplit_le n hne le_rfl β) basis
        have hS : splitStep n basis.length g.length =
            basis.length * ((8 * n + 14) * (g.length + 1) ^ 3) + 1 := rfl
        simp only [split, ite_eq_right h1, ite_eq_right h2, Counted.ops_bind]
        cases ho : (firstSplit n g basis).val with
        | none =>
          simp only [Counted.ops_pure]
          have := Nat.le_mul_of_pos_left (splitStep n basis.length g.length) (by omega : 0 < 2 * g.length - 3)
          omega
        | some hk =>
          obtain ⟨β', -, hts⟩ := firstSplit_some ho
          obtain ⟨Hn, Kn, Hm, Km, -, l1, l2, l3, l4⟩ := trySplit_some hg hgm hts
          have A := (ops_split_le n basis fuel hk.1 Hn Hm).trans
            (Nat.mul_le_mul_left _ (splitStep_mono n basis.length l2.le))
          have B := (ops_split_le n basis fuel hk.2 Kn Km).trans
            (Nat.mul_le_mul_left _ (splitStep_mono n basis.length l3.le))
          simp only [Counted.ops_bind, Counted.ops_pure]
          have e : 2 * g.length - 3 = (2 * hk.1.length - 3) + (2 * hk.2.length - 3) + 1 := by omega
          rw [e, add_mul, add_mul, one_mul]
          omega

theorem rootProduct_norm_monic [DecidableEq F] {n : ℕ} {f : List F} (h : (rootProduct n f).val ≠ []) :
    Norm (rootProduct n f).val ∧ (toPoly (rootProduct n f).val).Monic := by
  have hn := norm_trim f
  by_cases hne : (trim f).val = []
  · simp [rootProduct, hne] at h
  · simp only [rootProduct, Counted.val_bind, ite_eq_right hne] at h ⊢
    obtain ⟨gn, gd⟩ := gcd_spec hn
      (add (sqLoop (trim f).val n (divmod [0, 1] (trim f).val).val.2).val [0, -1]).val
    have h0 := ne_zero_of_dvd_iff gd (hn.toPoly_ne_zero hne)
    obtain ⟨Gn, Gm, -⟩ := monic_spec gn (ne_nil_of_toPoly_ne_zero h0)
    exact ⟨Gn, Gm⟩

theorem rootProduct_arith (n D a b c d e g : ℕ) (ha : a ≤ D) (hb : b ≤ 5 * D) (hc : c ≤ n * (6 * D * D + D))
    (hd : d ≤ 2) (he : e ≤ D + 2 + (D + 2 + 1) * (2 * (D + 2) * (D + 2) + (D + 2))) (hg : g ≤ D + 3) :
    a + (b + (c + (d + (e + g)))) ≤ (7 * n + 20) * (D + 3) ^ 3 := by
  have h : (7 * n + 20) * (D + 3) ^ 3 = (D + 5 * D + n * (6 * D * D + D) + 2 +
      (D + 2 + (D + 2 + 1) * (2 * (D + 2) * (D + 2) + (D + 2))) + (D + 3)) +
      (7 * n * D ^ 3 + 18 * D ^ 3 + 57 * n * D ^ 2 + 165 * D ^ 2 + 188 * n * D + 495 * D + 189 * n + 503) := by
    ring
  rw [h]
  omega

theorem rootProduct_bound [DecidableEq F] (n : ℕ) (f : List F) :
    (rootProduct n f).val.length ≤ f.length + 2 ∧ (rootProduct n f).ops ≤ (7 * n + 20) * (f.length + 3) ^ 3 := by
  have t1 := ops_trim_le f
  have t2 := length_trim_le f
  simp only [rootProduct, Counted.ops_bind, Counted.val_bind]
  split_ifs with hne
  · simp only [Counted.val_pure, Counted.ops_pure, List.length_nil]
    refine ⟨by omega, ?_⟩
    have : f.length ≤ (f.length + 3) ^ 3 := (Nat.le_add_right _ 3).trans (Nat.le_self_pow (by norm_num) _)
    nlinarith
  · simp only [Counted.val_bind, Counted.ops_bind]
    have d1 : (divmod [0, 1] (trim f).val).ops ≤ 5 * f.length := by
      have := ops_divmod_le ([0, 1] : List F) (trim f).val
      simp only [List.length_cons, List.length_nil] at this
      omega
    have r1 := length_rem_lt (p := [0, 1]) hne
    obtain ⟨s1, s2⟩ := sqLoop_bound hne t2 n (divmod [0, 1] (trim f).val).val.2 (by omega)
    have a1 := length_add (sqLoop (trim f).val n (divmod [0, 1] (trim f).val).val.2).val [0, -1]
    have a2 := ops_add (sqLoop (trim f).val n (divmod [0, 1] (trim f).val).val.2).val [0, -1]
    simp only [List.length_cons, List.length_nil, zero_add, Nat.reduceAdd] at a1 a2
    have g1 := ops_gcd_le (L := f.length + 2) (p := (trim f).val)
      (g := (add (sqLoop (trim f).val n (divmod [0, 1] (trim f).val).val.2).val [0, -1]).val)
      (by omega) (by omega)
    have g2 := length_gcd_le (trim f).val
      (add (sqLoop (trim f).val n (divmod [0, 1] (trim f).val).val.2).val [0, -1]).val
    have m1 := ops_monic_le (gcd (trim f).val
      (add (sqLoop (trim f).val n (divmod [0, 1] (trim f).val).val.2).val [0, -1]).val).val
    have m2 := length_monic (gcd (trim f).val
      (add (sqLoop (trim f).val n (divmod [0, 1] (trim f).val).val.2).val [0, -1]).val).val
    have a3 : (add (sqLoop (trim f).val n (divmod [0, 1] (trim f).val).val.2).val [0, -1]).ops ≤ 2 := by omega
    have m3 : (monic (gcd (trim f).val
        (add (sqLoop (trim f).val n (divmod [0, 1] (trim f).val).val.2).val [0, -1]).val).val).ops ≤
        f.length + 3 := by omega
    refine ⟨by omega, ?_⟩
    have := rootProduct_arith n f.length _ _ _ _ _ _ t1 d1 s2 a3 g1 m3
    omega

theorem rootsC_arith (n b D o s : ℕ) (ho : o ≤ (7 * n + 20) * (D + 3) ^ 3)
    (hs : s ≤ (2 * D + 1) * (b * ((8 * n + 14) * (D + 2 + 1) ^ 3) + 1)) :
    o + s ≤ 64 * (n + 1) * (b + 1) * (D + 3) ^ 4 := by
  have h : 64 * (n + 1) * (b + 1) * (D + 3) ^ 4 = (7 * n + 20) * (D + 3) ^ 3 +
      (2 * D + 1) * (b * ((8 * n + 14) * (D + 2 + 1) ^ 3) + 1) +
      (48 * D ^ 4 * b * n + 36 * D ^ 4 * b + 64 * D ^ 4 * n + 64 * D ^ 4 + 616 * D ^ 3 * b * n + 502 * D ^ 3 * b +
        761 * D ^ 3 * n + 748 * D ^ 3 + 2952 * D ^ 2 * b * n + 2574 * D ^ 2 * b + 3393 * D ^ 2 * n + 3276 * D ^ 2 +
        6264 * D * b * n + 5778 * D * b + 6723 * D * n + 6370 * D + 4968 * b * n + 4806 * b + 4995 * n + 4643) := by
    ring
  rw [h]
  omega

/-- **The root finder's operation count** is polynomial in `f`'s length, `n` and `basis`'s length. -/
theorem rootsC_ops_le [DecidableEq F] (n : ℕ) (basis f : List F) :
    (rootsC n basis f).ops ≤ costBound n basis.length f.length := by
  obtain ⟨l1, o1⟩ := rootProduct_bound n f
  simp only [rootsC, Counted.ops_bind]
  have hs : (split n basis (rootProduct n f).val.length (rootProduct n f).val).ops ≤
      (2 * f.length + 1) * splitStep n basis.length (f.length + 2) := by
    by_cases hg : (rootProduct n f).val = []
    · rw [hg]; simp [split]
    · obtain ⟨gn, gm⟩ := rootProduct_norm_monic hg
      exact (ops_split_le n basis _ _ gn gm).trans
        (Nat.mul_le_mul (by omega) (splitStep_mono n basis.length l1))
  unfold splitStep at hs
  exact rootsC_arith n basis.length f.length _ _ o1 hs

end Cost

end Claims.RootFind
