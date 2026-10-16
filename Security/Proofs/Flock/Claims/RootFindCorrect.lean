import Proofs.Flock.Claims.RootFindAlg

/-!
# The root finder is correct

* The loops compute modulo `g`: `sqLoop` is `u^(2^j)` (`sqLoop_dvd`) and `traceLoop` is `acc + Σ_{i<j} u^(2^i)`
  (`traceLoop_dvd`), each up to a multiple of `g`.
* `X^q − X` for `q = |F|` has every element of `F` as a root, once, so a monic divisor `g` has distinct roots, as many as
  its degree (`roots_nodup_of_dvd`, `card_roots_of_dvd`). `rootProduct` is such a divisor with `f`'s roots
  (`rootProduct_spec`).
* At a root `c` of `g`, `Tr(β·X) mod g` is `Tr(β·c)`, which is 0 or 1 (`trace_sq`), so `gcd(g, Tr(β·X) mod g)` has the
  roots `c` of `g` with `Tr(β·c) = 0`. A split `trySplit` returns is a factorization `g = h·k` into monic pieces of
  smaller degree (`trySplit_some`), and `β` splits `g` when `g` has roots on both sides (`trySplit_isSome`).
* `split` lists `g`'s roots, each once (`split_spec`), and so `roots` lists `f`'s (`roots_spec`).
-/

namespace Claims.RootFind

open Polynomial

section Algebra

variable {F : Type} [Field F]

theorem eval_eq_of_dvd_sub {g A B : F[X]} (h : g ∣ A - B) {a : F} (ha : g.IsRoot a) : A.eval a = B.eval a := by
  obtain ⟨k, hk⟩ := h
  have := congrArg (eval a) hk
  rw [eval_sub, eval_mul, ha.eq_zero, zero_mul, sub_eq_zero] at this
  exact this

theorem dvd_rem_sub [DecidableEq F] {p g : List F} (hg : Norm g) (hne : g ≠ []) :
    toPoly g ∣ toPoly (divmod p g).val.2 - toPoly p :=
  ⟨-toPoly (divmod p g).val.1, by linear_combination -(divmod_eq p g hg hne)⟩

theorem sq_pow_two_pow (u : F[X]) (i : ℕ) : (u * u) ^ 2 ^ i = u ^ 2 ^ (i + 1) := by
  rw [← sq, ← pow_mul, ← pow_succ']

/-- `sqLoop` squares modulo `g`. -/
theorem sqLoop_dvd [DecidableEq F] {g : List F} (hg : Norm g) (hne : g ≠ []) :
    ∀ (j : ℕ) (u : List F), toPoly g ∣ toPoly (sqLoop g j u).val - toPoly u ^ 2 ^ j
  | 0, u => by simp [sqLoop]
  | j + 1, u => by
    have h1 := dvd_rem_sub (p := (mul u u).val) hg hne
    rw [toPoly_mul] at h1
    have ih := sqLoop_dvd hg hne j (divmod (mul u u).val g).val.2
    have h2 := h1.trans (sub_dvd_pow_sub_pow _ _ (2 ^ j))
    rw [sq_pow_two_pow] at h2
    simp only [sqLoop, Counted.val_bind]
    have := dvd_add ih h2
    rwa [sub_add_sub_cancel] at this

/-- `traceLoop` sums the squares modulo `g`. -/
theorem traceLoop_dvd [DecidableEq F] {g : List F} (hg : Norm g) (hne : g ≠ []) :
    ∀ (j : ℕ) (acc u : List F), toPoly g ∣
      toPoly (traceLoop g j acc u).val - (toPoly acc + ∑ i ∈ Finset.range j, toPoly u ^ 2 ^ i)
  | 0, acc, u => by simp [traceLoop]
  | j + 1, acc, u => by
    have h1 := dvd_rem_sub (p := (mul u u).val) hg hne
    rw [toPoly_mul] at h1
    have ih := traceLoop_dvd hg hne j (add acc u).val (divmod (mul u u).val g).val.2
    rw [toPoly_add] at ih
    have h2 : toPoly g ∣ ∑ i ∈ Finset.range j, toPoly (divmod (mul u u).val g).val.2 ^ 2 ^ i -
        ∑ i ∈ Finset.range j, toPoly u ^ 2 ^ (i + 1) := by
      rw [← Finset.sum_sub_distrib]
      refine Finset.dvd_sum fun i _ => h1.trans ?_
      rw [← sq_pow_two_pow]
      exact sub_dvd_pow_sub_pow _ _ _
    simp only [traceLoop, Counted.val_bind]
    rw [Finset.sum_range_succ', pow_zero, pow_one]
    have := dvd_add ih h2
    convert this using 1
    ring

theorem isRoot_iff_of_dvd_iff {h p q : F[X]} (hd : ∀ d, d ∣ h ↔ d ∣ p ∧ d ∣ q) (a : F) :
    h.IsRoot a ↔ p.IsRoot a ∧ q.IsRoot a := by
  rw [← dvd_iff_isRoot, ← dvd_iff_isRoot, ← dvd_iff_isRoot]
  exact hd _

theorem ne_zero_of_dvd_iff {h p q : F[X]} (hd : ∀ d, d ∣ h ↔ d ∣ p ∧ d ∣ q) (hp : p ≠ 0) : h ≠ 0 := by
  rintro rfl
  exact hp (zero_dvd_iff.1 ((hd 0).1 dvd_rfl).1)

theorem ne_nil_of_toPoly_ne_zero {l : List F} (h : toPoly l ≠ 0) : l ≠ [] := by
  rintro rfl; exact h rfl

/-- A split `trySplit` returns: `g = h·k`, both monic and normalized, `h` of degree at least 1, both shorter than
`g`. -/
theorem trySplit_some [DecidableEq F] {n : ℕ} {g : List F} (hg : Norm g) (hgm : (toPoly g).Monic) {β : F}
    {hk : List F × List F} (h : (trySplit n g β).val = some hk) :
    Norm hk.1 ∧ Norm hk.2 ∧ (toPoly hk.1).Monic ∧ (toPoly hk.2).Monic ∧ toPoly g = toPoly hk.1 * toPoly hk.2 ∧
      1 < hk.1.length ∧ hk.1.length < g.length ∧ hk.2.length < g.length ∧
      hk.1.length + hk.2.length = g.length + 1 := by
  simp only [trySplit, Counted.val_bind] at h
  split_ifs at h with hc
  · simp only [Counted.val_bind, Counted.val_pure, Option.some.injEq] at h
    subst h
    obtain ⟨hc1, hc2⟩ := hc
    dsimp only
    obtain ⟨gn, gd⟩ := gcd_spec hg (traceLoop g n [] (divmod [0, β] g).val.2).val
    set h' := (gcd g (traceLoop g n [] (divmod [0, β] g).val.2).val).val
    have h'ne : h' ≠ [] := by rintro h0; rw [h0] at hc1; simp at hc1
    obtain ⟨Hn, Hm, Hd⟩ := monic_spec gn h'ne
    set H := (monic h').val
    have HL : H.length = h'.length := length_monic h'
    have Hne : H ≠ [] := by rintro h0; rw [h0] at HL; simp at HL; omega
    have Hg : toPoly H ∣ toPoly g := ((gd _).1 ((Hd _).1 dvd_rfl)).1
    have e := divmod_eq g H Hn Hne
    obtain ⟨Kn, Rn, -, Rl, -⟩ := divmod_spec g H Hne
    set K := (divmod g H).val.1
    set R := (divmod g H).val.2
    have R0 : toPoly R = 0 := by
      by_contra hR
      have hRd : toPoly H ∣ toPoly R := by
        have : toPoly R = toPoly g - toPoly K * toPoly H := by rw [e]; ring
        rw [this]; exact dvd_sub Hg (dvd_mul_left _ _)
      have := natDegree_le_of_dvd hRd hR
      rw [Hn.natDegree Hne, Rn.natDegree (ne_nil_of_toPoly_ne_zero hR)] at this
      have : 0 < R.length := List.length_pos_iff.2 (ne_nil_of_toPoly_ne_zero hR)
      omega
    have egk : toPoly g = toPoly H * toPoly K := by rw [e, R0, add_zero, mul_comm]
    have Km : (toPoly K).Monic := Monic.of_mul_monic_left Hm (egk ▸ hgm)
    have hdeg := Hm.natDegree_mul Km
    rw [← egk] at hdeg
    have lg := hg.length_eq hgm.ne_zero
    have lH := Hn.length_eq Hm.ne_zero
    have lK := Kn.length_eq Km.ne_zero
    exact ⟨Hn, Kn, Hm, Km, egk, by omega, by omega, by omega, by omega⟩

/-- `β` splits `g` when `g` has a root `a` with `Tr(β·a) = 0` and a root `b` with `Tr(β·b) ≠ 0`. -/
theorem trySplit_isSome [DecidableEq F] {n : ℕ} {g : List F} (hg : Norm g) (hgm : (toPoly g).Monic) {β a b : F}
    (ha : (toPoly g).IsRoot a) (hb : (toPoly g).IsRoot b) (hta : trace n (β * a) = 0)
    (htb : trace n (β * b) ≠ 0) : (trySplit n g β).val.isSome := by
  have hne : g ≠ [] := ne_nil_of_toPoly_ne_zero hgm.ne_zero
  set u := (divmod [0, β] g).val.2
  have hu : toPoly g ∣ toPoly u - C β * X := by
    have e : toPoly ([0, β] : List F) = C β * X := by
      simp only [toPoly_cons, toPoly_nil, map_zero, zero_add, mul_zero, add_zero]; ring
    have := dvd_rem_sub (p := [0, β]) hg hne
    rwa [e] at this
  set s := (traceLoop g n [] u).val
  have hs := traceLoop_dvd hg hne n [] u
  have hsc : ∀ c, (toPoly g).IsRoot c → (toPoly s).eval c = trace n (β * c) := fun c hc => by
    rw [eval_eq_of_dvd_sub hs hc, toPoly_nil, zero_add, eval_finsetSum, trace]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [eval_pow, eval_eq_of_dvd_sub hu hc]
    simp
  obtain ⟨gn, gd⟩ := gcd_spec hg s
  set h' := (gcd g s).val
  have hr := isRoot_iff_of_dvd_iff gd
  have h'0 : toPoly h' ≠ 0 := ne_zero_of_dvd_iff gd hgm.ne_zero
  have h'ne : h' ≠ [] := ne_nil_of_toPoly_ne_zero h'0
  have hra : (toPoly h').IsRoot a := (hr a).2 ⟨ha, by rw [IsRoot.def, hsc a ha, hta]⟩
  have hrb : ¬ (toPoly h').IsRoot b := fun h => htb (by rw [← hsc b hb]; exact ((hr b).1 h).2)
  have hd1 : 0 < (toPoly h').natDegree := by
    by_contra h0
    rw [eq_C_of_natDegree_le_zero (p := toPoly h') (by omega)] at hra h'0
    exact h'0 (by rw [IsRoot.def, eval_C] at hra; rw [hra, map_zero])
  have hdvd : toPoly h' ∣ toPoly g := ((gd _).1 dvd_rfl).1
  have hd2 : (toPoly h').natDegree < (toPoly g).natDegree := by
    obtain ⟨m, hm⟩ := hdvd
    have hm0 : m ≠ 0 := by rintro rfl; rw [mul_zero] at hm; exact hgm.ne_zero hm
    have hdeg := congrArg natDegree hm
    rw [natDegree_mul h'0 hm0] at hdeg
    by_contra hle
    have hmd : m.natDegree = 0 := by omega
    rw [eq_C_of_natDegree_eq_zero hmd] at hm hm0
    have hb' := hb.eq_zero
    rw [hm, eval_mul, eval_C, mul_eq_zero] at hb'
    exact hrb (hb'.resolve_right fun h => hm0 (by rw [h, map_zero]))
  have l1 := gn.length_eq h'0
  have l2 := hg.length_eq hgm.ne_zero
  have hcond : 1 < h'.length ∧ h'.length < g.length := ⟨by omega, by omega⟩
  simp only [trySplit, Counted.val_bind]
  rw [ite_eq_left hcond]
  simp

theorem firstSplit_some [DecidableEq F] {n : ℕ} {g : List F} :
    ∀ {basis : List F} {hk : List F × List F}, (firstSplit n g basis).val = some hk →
      ∃ β ∈ basis, (trySplit n g β).val = some hk
  | [], _, h => by simp [firstSplit] at h
  | β :: basis, hk, h => by
    simp only [firstSplit, Counted.val_bind] at h
    cases ht : (trySplit n g β).val with
    | some hk' =>
      rw [ht] at h
      simp only [Counted.val_pure, Option.some.injEq] at h
      exact ⟨β, List.mem_cons_self .., h ▸ ht⟩
    | none =>
      rw [ht] at h
      obtain ⟨β', hβ', h'⟩ := firstSplit_some h
      exact ⟨β', List.mem_cons_of_mem _ hβ', h'⟩

theorem firstSplit_isSome [DecidableEq F] {n : ℕ} {g : List F} {β : F} (hβ : (trySplit n g β).val.isSome) :
    ∀ {basis : List F}, β ∈ basis → (firstSplit n g basis).val.isSome
  | [], h => by simp at h
  | β' :: basis, h => by
    simp only [firstSplit, Counted.val_bind]
    cases ht : (trySplit n g β').val with
    | some hk' => simp
    | none =>
      simp only
      rcases List.mem_cons.1 h with rfl | h
      · rw [ht] at hβ; simp at hβ
      · exact firstSplit_isSome hβ h

end Algebra

section Finite

variable {F : Type} [Field F]

theorem frob_ne_zero [Fintype F] : (X ^ Fintype.card F - X : F[X]) ≠ 0 :=
  FiniteField.X_pow_card_sub_X_ne_zero F Fintype.one_lt_card

/-- A divisor of `X^q − X` has distinct roots. -/
theorem roots_nodup_of_dvd [Fintype F] {g : F[X]} (h : g ∣ X ^ Fintype.card F - X) : g.roots.Nodup := by
  refine Multiset.nodup_of_le (roots.le_of_dvd frob_ne_zero h) ?_
  rw [FiniteField.roots_X_pow_card_sub_X]
  exact Finset.univ.nodup

/-- A monic divisor of `X^q − X` has as many roots as its degree. -/
theorem card_roots_of_dvd [Fintype F] {g : F[X]} (hg : g.Monic) (h : g ∣ X ^ Fintype.card F - X) :
    g.roots.card = g.natDegree := by
  obtain ⟨k, hk⟩ := h
  have hk0 : k ≠ 0 := by rintro rfl; rw [mul_zero] at hk; exact frob_ne_zero hk
  have h1 := congrArg (fun p => Multiset.card (Polynomial.roots p)) hk
  rw [FiniteField.roots_X_pow_card_sub_X, roots_mul (hk ▸ frob_ne_zero), Multiset.card_add] at h1
  have h2 := congrArg natDegree hk
  rw [FiniteField.X_pow_card_sub_X_natDegree_eq F Fintype.one_lt_card, natDegree_mul hg.ne_zero hk0] at h2
  have := card_roots' g
  have := card_roots' k
  simp only [Finset.card_val, Finset.card_univ] at h1
  omega

theorem trace_add [CharP F 2] (n : ℕ) (a b : F) : trace n (a + b) = trace n a + trace n b := by
  simp only [trace, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun i _ => add_pow_char_pow a b 2 i

/-- `Tr(a)² = Tr(a)`. -/
theorem trace_sq [Fintype F] [CharP F 2] {n : ℕ} (hcard : Fintype.card F = 2 ^ n) (a : F) : trace n a ^ 2 = trace n a := by
  have hfrob : a ^ 2 ^ n = a := hcard ▸ FiniteField.pow_card a
  rw [trace, sum_pow_char 2]
  have e : ∀ i, (a ^ 2 ^ i) ^ 2 = a ^ 2 ^ (i + 1) := fun i => by rw [← pow_mul, ← pow_succ]
  simp only [e]
  have h1 := Finset.sum_range_succ' (fun i => a ^ 2 ^ i) n
  rw [Finset.sum_range_succ, hfrob, pow_zero, pow_one] at h1
  exact (add_right_cancel h1).symm

theorem trace_eq_zero_or_one [Fintype F] [CharP F 2] {n : ℕ} (hcard : Fintype.card F = 2 ^ n) (a : F) :
    trace n a = 0 ∨ trace n a = 1 := by
  have h := trace_sq hcard a
  have : trace n a * (trace n a - 1) = 0 := by linear_combination h
  rcases mul_eq_zero.1 this with h0 | h1
  · exact Or.inl h0
  · exact Or.inr (sub_eq_zero.1 h1)

/-- **The root product**: normalized, monic, a divisor of `X^q − X`, with `f`'s roots. -/
theorem rootProduct_spec [Fintype F] [DecidableEq F] {n : ℕ} (hcard : Fintype.card F = 2 ^ n) {f : List F} (hf : toPoly f ≠ 0) :
    Norm (rootProduct n f).val ∧ (toPoly (rootProduct n f).val).Monic ∧
      toPoly (rootProduct n f).val ∣ X ^ Fintype.card F - X ∧
      ∀ a, (toPoly (rootProduct n f).val).IsRoot a ↔ (toPoly f).IsRoot a := by
  have hn : Norm (trim f).val := norm_trim f
  have he : toPoly (trim f).val = toPoly f := toPoly_trim f
  have hne : (trim f).val ≠ [] := ne_nil_of_toPoly_ne_zero (he ▸ hf)
  simp only [rootProduct, Counted.val_bind, ite_eq_right hne]
  set f' := (trim f).val
  set x := (divmod [0, 1] f').val.2
  have hx : toPoly f' ∣ toPoly x - X := by simpa using dvd_rem_sub (p := [0, 1]) hn hne
  set u := (sqLoop f' n x).val
  have hu : toPoly f' ∣ toPoly u - X ^ 2 ^ n :=
    (dvd_add (sqLoop_dvd hn hne n x) (hx.trans (sub_dvd_pow_sub_pow _ _ _))).trans (by rw [sub_add_sub_cancel])
  set w := (add u [0, -1]).val
  have hw : toPoly f' ∣ toPoly w - (X ^ Fintype.card F - X) := by
    rw [toPoly_add, hcard]
    convert hu using 1
    simp; ring
  obtain ⟨gn, gd⟩ := gcd_spec hn w
  set h := (gcd f' w).val
  have h0 : toPoly h ≠ 0 := ne_zero_of_dvd_iff gd (he ▸ hf)
  obtain ⟨Gn, Gm, Gd⟩ := monic_spec gn (ne_nil_of_toPoly_ne_zero h0)
  have Gh := (Gd _).1 dvd_rfl
  have Gf := ((gd _).1 Gh).1
  have Gw := ((gd _).1 Gh).2
  refine ⟨Gn, Gm, ?_, fun a => ?_⟩
  · have := dvd_sub Gw (Gf.trans hw)
    rwa [sub_sub_cancel] at this
  · rw [← dvd_iff_isRoot, Gd, dvd_iff_isRoot, isRoot_iff_of_dvd_iff gd, he]
    refine ⟨And.left, fun ha => ⟨ha, ?_⟩⟩
    rw [IsRoot.def, eval_eq_of_dvd_sub hw (he ▸ ha)]
    simp [FiniteField.pow_card]

/-- **`split` lists the roots of a monic divisor of `X^q − X`, each once**, given `fuel` at least its length. -/
theorem split_spec [Fintype F] [DecidableEq F] [CharP F 2] {n : ℕ} (hcard : Fintype.card F = 2 ^ n) {basis : List F}
    (hsep : Separates n basis) : ∀ (fuel : ℕ) (g : List F), Norm g → (toPoly g).Monic →
      toPoly g ∣ X ^ Fintype.card F - X → g.length ≤ fuel →
      (split n basis fuel g).val.Nodup ∧ ∀ a, a ∈ (split n basis fuel g).val ↔ (toPoly g).IsRoot a
  | 0, g, _, hgm, _, hl => by
    have : g = [] := List.eq_nil_of_length_eq_zero (by omega)
    subst this
    exact absurd hgm (by simp [Monic])
  | fuel + 1, g, hg, hgm, hdvd, hl => by
    have lg := hg.length_eq hgm.ne_zero
    by_cases h1 : g.length ≤ 1
    · have := eq_one_of_monic_natDegree_zero hgm (by omega)
      simp only [split, ite_eq_left h1, Counted.val_pure]
      exact ⟨List.nodup_nil, fun a => by simp [this]⟩
    · by_cases h2 : g.length = 2
      · simp only [split, ite_eq_right h1, ite_eq_left h2, Counted.val_bind, val_op, Counted.val_pure]
        obtain ⟨a0, a1, rfl⟩ : ∃ a0 a1, g = [a0, a1] := match g, h2 with | [a0, a1], _ => ⟨a0, a1, rfl⟩
        have ha1 : a1 = 1 := by
          have := hg.leadingCoeff (List.cons_ne_nil _ _)
          rw [hgm.leadingCoeff] at this
          simpa using this.symm
        subst ha1
        refine ⟨List.nodup_singleton _, fun a => ?_⟩
        simp only [List.mem_singleton, List.headD_cons, IsRoot.def, toPoly_cons, toPoly_nil, mul_zero, add_zero,
          map_one, mul_one, eval_add, eval_C, eval_X]
        constructor
        · rintro rfl; ring
        · intro h; linear_combination h
      · have hd2 : 2 ≤ (toPoly g).natDegree := by omega
        have hcardr := card_roots_of_dvd hgm hdvd
        have hnd := roots_nodup_of_dvd hdvd
        obtain ⟨a, ha, b, hb, hab⟩ : ∃ a ∈ (toPoly g).roots.toFinset, ∃ b ∈ (toPoly g).roots.toFinset, a ≠ b := by
          refine Finset.one_lt_card.1 ?_
          rw [Multiset.toFinset_card_of_nodup hnd]
          omega
        rw [Multiset.mem_toFinset, mem_roots hgm.ne_zero] at ha hb
        obtain ⟨β, hβ, hβab⟩ := hsep (a - b) (sub_ne_zero.2 hab)
        have htr : trace n (β * a) = trace n (β * (a - b)) + trace n (β * b) := by
          rw [← trace_add]; congr 1; ring
        have hsome : (firstSplit n g basis).val.isSome := by
          refine firstSplit_isSome ?_ hβ
          rcases trace_eq_zero_or_one hcard (β * a) with h0 | h0
          · exact trySplit_isSome hg hgm ha hb h0 fun h => hβab (by rw [h0, h, add_zero] at htr; exact htr.symm)
          · refine trySplit_isSome hg hgm hb ha ?_ (by rw [h0]; exact one_ne_zero)
            rcases trace_eq_zero_or_one hcard (β * b) with h1 | h1
            · exact h1
            · rw [h0, h1] at htr; exact absurd (by linear_combination -htr) hβab
        obtain ⟨hk, hfs⟩ := Option.isSome_iff_exists.1 hsome
        obtain ⟨β', -, hts⟩ := firstSplit_some hfs
        obtain ⟨Hn, Kn, Hm, Km, egk, -, lH, lK, -⟩ := trySplit_some hg hgm hts
        have hHd : toPoly hk.1 ∣ X ^ Fintype.card F - X := (Dvd.intro _ egk.symm).trans hdvd
        have hKd : toPoly hk.2 ∣ X ^ Fintype.card F - X := (Dvd.intro_left _ egk.symm).trans hdvd
        obtain ⟨A1, A2⟩ := split_spec hcard hsep fuel hk.1 Hn Hm hHd (by omega)
        obtain ⟨B1, B2⟩ := split_spec hcard hsep fuel hk.2 Kn Km hKd (by omega)
        simp only [split, ite_eq_right h1, ite_eq_right h2, Counted.val_bind, hfs, Counted.val_pure]
        have hroots := roots_mul (egk ▸ hgm.ne_zero)
        rw [← egk] at hroots
        rw [hroots] at hnd
        obtain ⟨-, -, hdis⟩ := Multiset.nodup_add.1 hnd
        refine ⟨List.nodup_append.2 ⟨A1, B1, fun x hx y hy hxy => ?_⟩, fun x => ?_⟩
        · subst hxy
          exact Multiset.disjoint_left.1 hdis ((mem_roots Hm.ne_zero).2 ((A2 x).1 hx))
            ((mem_roots Km.ne_zero).2 ((B2 x).1 hy))
        · rw [List.mem_append, A2, B2, egk, IsRoot.def, IsRoot.def, IsRoot.def, eval_mul, mul_eq_zero]

/-- **The root finder is correct**: over `F` of characteristic 2 with `2^n` elements and a separating `basis`, for
`f ≠ 0` it lists `f`'s roots, each once. -/
theorem roots_spec [Fintype F] [DecidableEq F] [CharP F 2] {n : ℕ} (hcard : Fintype.card F = 2 ^ n) {basis : List F}
    (hsep : Separates n basis) {f : List F} (hf : toPoly f ≠ 0) :
    (roots n basis f).Nodup ∧ ∀ a, a ∈ roots n basis f ↔ (toPoly f).IsRoot a := by
  obtain ⟨h1, h2, h3, h4⟩ := rootProduct_spec hcard hf
  have := split_spec hcard hsep (rootProduct n f).val.length (rootProduct n f).val h1 h2 h3 le_rfl
  simp only [roots, rootsC, Counted.val_bind]
  exact ⟨this.1, fun a => (this.2 a).trans (h4 a)⟩

end Finite

end Claims.RootFind
