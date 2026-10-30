import Properties.PoubCheck.Pooled
import Mathlib.Algebra.Order.BigOperators.Group.List
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity

/-!
# The verifier's pooled bound covers the draw's (`PooledCovers`)

The budget: `powUp` is at least the power it rounds (`powUp_ge`, each product rounded up), so a numerator `below`
accepts has `(m/2^64)^S ≤ δ` (`below_le`), and since `x ↦ x^S` is monotone on `[0, ∞)`, the root is at most `δ^(1/S)`
(`root_le`, the fallback `0` trivially) and the code's budget at least `S·(1 − δ^(1/S))` (`budget_le`).

W: `G` is `fill` over the reals (`fill_cast`). Over windows by `U/s` falling, `G` gains at most `r` a unit of budget when
every ratio is at most `r` (`G_lip`), so a share vector that spends `b` on windows in that order wins at most what the
greedy wins with `b` (`G_ge`): the head's share is at most the greedy's, and the budget the greedy spends beyond it on
the head buys, on the rest, no more than the head's ratio. `order` permutes the windows (`List.mergeSort_perm`) and
sorts them (`List.pairwise_mergeSort`), so every share vector `Draw.pooled` ranges over is one `G_ge` bounds.
-/

open PoubCheck Pooled

namespace Properties.PoubCheck

namespace Greedy

theorem mulUp_ge (p a b : ℕ) : a * b ≤ mulUp p a b * 2 ^ p := by
  unfold mulUp
  have hk : 0 < 2 ^ p := by positivity
  have h1 := Nat.div_add_mod (a * b + 2 ^ p - 1) (2 ^ p)
  have h2 := Nat.mod_lt (a * b + 2 ^ p - 1) hk
  have h3 : (a * b + 2 ^ p - 1) / 2 ^ p * 2 ^ p = 2 ^ p * ((a * b + 2 ^ p - 1) / 2 ^ p) := Nat.mul_comm _ _
  omega

theorem mulUp_ge_real (p a b : ℕ) : (a : ℝ) * b / 2 ^ p ≤ mulUp p a b := by
  have h := mulUp_ge p a b
  have hk : (0 : ℝ) < 2 ^ p := by positivity
  rw [div_le_iff₀ hk]
  exact_mod_cast h

theorem powUp_ge (p x : ℕ) : ∀ n, (2 : ℝ) ^ p * ((x : ℝ) / 2 ^ p) ^ n ≤ powUp p x n := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    have hk : (0 : ℝ) < 2 ^ p := by positivity
    have ha : (0 : ℝ) ≤ (x : ℝ) / 2 ^ p := by positivity
    rw [powUp]
    by_cases h0 : n = 0
    · simp [h0]
    · simp only [h0, dite_false]
      have hr := ih (n / 2) (Nat.div_lt_self (Nat.pos_of_ne_zero h0) (by norm_num))
      set r := powUp p x (n / 2)
      set a := (x : ℝ) / 2 ^ p
      have hsq : (2 : ℝ) ^ p * a ^ (2 * (n / 2)) ≤ mulUp p r r := by
        refine le_trans ?_ (mulUp_ge_real p r r)
        have h0' : (0 : ℝ) ≤ 2 ^ p * a ^ (n / 2) := by positivity
        have := mul_le_mul hr hr h0' (by positivity)
        rw [le_div_iff₀ hk]
        calc (2 : ℝ) ^ p * a ^ (2 * (n / 2)) * 2 ^ p = (2 ^ p * a ^ (n / 2)) * (2 ^ p * a ^ (n / 2)) := by ring
          _ ≤ _ := this
      split_ifs with hodd
      · refine le_trans ?_ (mulUp_ge_real p _ x)
        have hn : n = 2 * (n / 2) + 1 := by omega
        have hax : a * 2 ^ p = x := by simp only [a]; field_simp
        rw [le_div_iff₀ hk]
        calc (2 : ℝ) ^ p * a ^ n * 2 ^ p = (2 ^ p * a ^ (2 * (n / 2))) * (a * 2 ^ p) := by
              conv_lhs => rw [hn]
              ring
          _ ≤ (mulUp p r r : ℝ) * x := by rw [hax]; exact mul_le_mul_of_nonneg_right hsq (by positivity)
      · have hn : 2 * (n / 2) = n := by omega
        rw [hn] at hsq
        exact hsq


theorem below_le {S : ℕ} {δ : ℚ} {m : ℕ} (h : below S δ m = true) :
    ((m : ℝ) / 2 ^ ROOT_BITS) ^ S ≤ (δ : ℝ) := by
  unfold below at h
  have h' := of_decide_eq_true h
  have hp := powUp_ge POW_BITS (m * 2 ^ (POW_BITS - ROOT_BITS)) S
  have hx : ((m * 2 ^ (POW_BITS - ROOT_BITS) : ℕ) : ℝ) / 2 ^ POW_BITS = (m : ℝ) / 2 ^ ROOT_BITS := by
    simp only [POW_BITS, ROOT_BITS]; push_cast; field_simp; norm_num
  rw [hx] at hp
  have hden : (0 : ℝ) < δ.den := by exact_mod_cast δ.den_pos
  have hcast : ((powUp POW_BITS (m * 2 ^ (POW_BITS - ROOT_BITS)) S : ℕ) : ℝ) * δ.den ≤
      (δ.num : ℝ) * 2 ^ POW_BITS := by exact_mod_cast h'
  have hk : (0 : ℝ) < 2 ^ POW_BITS := by positivity
  rw [Rat.cast_def]
  rw [le_div_iff₀ hden]
  have := mul_le_mul_of_nonneg_right hp hden.le
  nlinarith

theorem root_le {S : ℕ} {δ : ℚ} (hS : S ≠ 0) (hδ : 0 < δ) :
    (root S δ : ℝ) / 2 ^ ROOT_BITS ≤ (δ : ℝ) ^ ((S : ℝ)⁻¹) := by
  have hδ' : (0 : ℝ) ≤ δ := by exact_mod_cast hδ.le
  unfold root
  simp only
  split_ifs with h
  · have hq := below_le h
    have h0 : (0 : ℝ) ≤ ((bisect S δ (ROOT_BITS + 1) 0 (2 ^ ROOT_BITS) : ℕ) : ℝ) / 2 ^ ROOT_BITS := by positivity
    calc _ = ((((bisect S δ (ROOT_BITS + 1) 0 (2 ^ ROOT_BITS) : ℕ) : ℝ) / 2 ^ ROOT_BITS) ^ S) ^ ((S : ℝ)⁻¹) :=
          (Real.pow_rpow_inv_natCast h0 hS).symm
      _ ≤ _ := Real.rpow_le_rpow (by positivity) hq (by positivity)
  · simp only [Nat.cast_zero, zero_div]
    exact Real.rpow_nonneg hδ' _

theorem budget_le {S : ℕ} {δ : ℚ} (hS : S ≠ 0) (hδ : 0 < δ) :
    Accounting.Poub.Draw.budget S δ ≤ (budget S δ : ℝ) := by
  unfold Accounting.Poub.Draw.budget budget
  push_cast
  exact mul_le_mul_of_nonneg_left (sub_le_sub_left (root_le hS hδ) 1) (by positivity)

/-- The greedy over ℝ. -/
noncomputable def G (u s : ℕ → ℝ) : List ℕ → ℝ → ℝ
  | [], _ => 0
  | i :: rest, b =>
    let f := min (1 - s i / u i) (b / s i)
    f * u i + G u s rest (max 0 (b - f * s i))

theorem G_lip (u s : ℕ → ℝ) (r : ℝ) (hr : 0 ≤ r) : ∀ (l : List ℕ), (∀ i ∈ l, 0 < s i ∧ u i ≤ s i * r) →
    ∀ (b Δ : ℝ), 0 ≤ Δ → G u s l (b + Δ) ≤ G u s l b + Δ * r := by
  intro l
  induction l with
  | nil => intro _ b Δ hΔ; simp only [G]; nlinarith
  | cons i rest ih =>
    intro hl b Δ hΔ
    obtain ⟨hs, hu⟩ := hl i (by simp)
    have hrest : ∀ j ∈ rest, 0 < s j ∧ u j ≤ s j * r := fun j hj => hl j (by simp [hj])
    simp only [G]
    set c := 1 - s i / u i
    set f1 := min c ((b + Δ) / s i) with hf1d
    set f0 := min c (b / s i) with hf0d
    have hf1 : f1 * s i ≤ b + Δ := (le_div_iff₀ hs).mp (min_le_right _ _)
    have hf0 : f0 * s i ≤ b := (le_div_iff₀ hs).mp (min_le_right _ _)
    have hbb : b / s i ≤ (b + Δ) / s i := div_le_div_of_nonneg_right (by linarith) hs.le
    have h01 : f0 ≤ f1 := min_le_min_left c hbb
    have hd : (f1 - f0) * s i ≤ Δ := by
      rcases le_total c (b / s i) with h | h
      · have e0 : f0 = c := min_eq_left h
        have e1 : f1 = c := min_eq_left (h.trans hbb)
        rw [e0, e1]; simp [hΔ]
      · have e0 : f0 = b / s i := min_eq_right h
        have : f1 - f0 ≤ Δ / s i := by
          have := min_le_right c ((b + Δ) / s i)
          rw [e0]; rw [← hf1d] at this
          calc f1 - b / s i ≤ (b + Δ) / s i - b / s i := by linarith
            _ = Δ / s i := by ring
        exact (le_div_iff₀ hs).mp this
    rw [max_eq_right (by linarith), max_eq_right (by linarith)]
    have h := ih hrest (b - f0 * s i) (Δ - (f1 - f0) * s i) (by linarith)
    have e : b - f0 * s i + (Δ - (f1 - f0) * s i) = b + Δ - f1 * s i := by ring
    rw [e] at h
    have : (f1 - f0) * u i ≤ (f1 - f0) * (s i * r) := mul_le_mul_of_nonneg_left hu (by linarith)
    nlinarith

theorem G_ge (u s φ : ℕ → ℝ) : ∀ (l : List ℕ), l.Pairwise (fun i j => u j / s j ≤ u i / s i) →
    (∀ i ∈ l, 0 < s i ∧ 0 ≤ u i) → (∀ i ∈ l, 0 ≤ φ i ∧ φ i ≤ 1 - s i / u i) →
    ∀ b, (l.map fun i => s i * φ i).sum ≤ b → (l.map fun i => φ i * u i).sum ≤ G u s l b := by
  intro l
  induction l with
  | nil => intros; simp [G]
  | cons i rest ih =>
    intro hp hpos hφ b hb
    rw [List.pairwise_cons] at hp
    obtain ⟨hs, hu⟩ := hpos i (by simp)
    obtain ⟨hφ0, hφc⟩ := hφ i (by simp)
    simp only [List.map_cons, List.sum_cons] at hb ⊢
    have hrest0 : 0 ≤ (rest.map fun j => s j * φ j).sum := by
      apply List.sum_nonneg
      intro x hx
      obtain ⟨j, hj, rfl⟩ := List.mem_map.mp hx
      have := (hpos j (by simp [hj])).1
      have := (hφ j (by simp [hj])).1
      positivity
    simp only [G]
    set c := 1 - s i / u i
    set f := min c (b / s i)
    have hφb : φ i ≤ b / s i := by rw [le_div_iff₀ hs]; nlinarith
    have hφf : φ i ≤ f := le_min hφc hφb
    have hfs : f * s i ≤ b := (le_div_iff₀ hs).mp (min_le_right _ _)
    rw [max_eq_right (by linarith)]
    have ih' := ih hp.2 (fun j hj => hpos j (by simp [hj])) (fun j hj => hφ j (by simp [hj]))
      (b - s i * φ i) (by linarith)
    have hlip := G_lip u s (u i / s i) (by positivity) rest (fun j hj => by
        have hsj := (hpos j (by simp [hj])).1
        refine ⟨hsj, ?_⟩
        have := hp.1 j hj
        rw [div_le_iff₀ hsj] at this
        linarith [mul_comm (u i / s i) (s j)]) (b - f * s i) ((f - φ i) * s i) (by nlinarith)
    have e : b - f * s i + (f - φ i) * s i = b - s i * φ i := by ring
    rw [e] at hlip
    have e2 : (f - φ i) * s i * (u i / s i) = (f - φ i) * u i := by field_simp
    rw [e2] at hlip
    linarith

theorem fill_cast (ws : Array (ℕ × ℕ)) : ∀ (l : List ℕ) (b : ℚ),
    (fill ws l b : ℝ) = G (fun i => (ws[i]!.1 : ℝ)) (fun i => (ws[i]!.2 : ℝ)) l b := by
  intro l
  induction l with
  | nil => intro b; simp [fill, G]
  | cons i rest ih =>
    intro b
    simp only [fill, G, share]
    rw [Rat.cast_add, ih]
    push_cast
    rfl

theorem ok_facts {ws : Array (ℕ × ℕ)} {δ : ℚ} {b : Bound} (h : pooled ws δ = .ok b) :
    ws.size ≠ 0 ∧ 0 < δ ∧ δ < 1 ∧ refusal ws = none ∧
      b.wrong = fill ws (order ws) (budget (ws.toList.map (·.2)).sum δ) := by
  unfold pooled at h
  by_cases h1 : ws.size = 0
  · simp [h1] at h
    cases h
  by_cases h2 : δ ≤ 0 ∨ 1 ≤ δ
  · simp [h1, h2] at h
    cases h
  cases h3 : refusal ws with
  | some why =>
    simp [h1, h2, h3] at h
    cases h
  | none =>
    simp [h1, h2, h3] at h
    change Except.ok _ = Except.ok b at h
    cases h
    push Not at h2
    exact ⟨h1, h2.1, h2.2, rfl, rfl⟩

theorem refusal_none {ws : Array (ℕ × ℕ)} (h : refusal ws = none) :
    ∀ t < ws.size, ws[t]!.2 ≠ 0 ∧ ws[t]!.2 ≤ ws[t]!.1 := by
  intro t ht
  have := List.findSome?_eq_none_iff.mp h t (List.mem_range.mpr ht)
  split_ifs at this with h1 h2
  exact ⟨h1, Nat.le_of_not_lt h2⟩

theorem sum_range_list (n : ℕ) (h : ℕ → ℝ) : ∑ i ∈ Finset.range n, h i = ((List.range n).map h).sum := by
  induction n with
  | zero => simp
  | succ n ih => rw [Finset.sum_range_succ, List.sum_range_succ, ih]

theorem sum_order (ws : Array (ℕ × ℕ)) (h : ℕ → ℝ) :
    ((order ws).map h).sum = ∑ i ∈ Finset.range ws.size, h i := by
  rw [sum_range_list]
  exact ((List.mergeSort_perm _ _).map h).sum_eq

theorem mem_order {ws : Array (ℕ × ℕ)} {i : ℕ} (h : i ∈ order ws) : i < ws.size :=
  List.mem_range.mp ((List.mergeSort_perm (List.range ws.size) _).mem_iff.mp h)

theorem order_sorted (ws : Array (ℕ × ℕ)) : (order ws).Pairwise (fun i j => ratio ws[j]! ≤ ratio ws[i]!) := by
  have := List.pairwise_mergeSort (le := fun i j => decide (ratio ws[j]! ≤ ratio ws[i]!))
    (by intro a b c h1 h2; simp only [decide_eq_true_eq] at *; exact le_trans h2 h1)
    (by intro a b; simp only [Bool.or_eq_true, decide_eq_true_eq]; exact le_total _ _) (List.range ws.size)
  unfold order
  simpa using this

end Greedy

open Greedy

theorem pooledCovers : PooledCovers := by
  intro n U d δ b h
  set ws := Array.ofFn fun t => (U t, d t) with hws
  obtain ⟨hn, hδ0, _, hr, hw⟩ := ok_facts h
  have hsize : ws.size = n := Array.size_ofFn
  have hget : ∀ (t : Fin n), ws[(t : ℕ)]! = (U t, d t) := by
    intro t
    rw [getElem!_pos ws (t : ℕ) (by rw [hsize]; exact t.2)]
    simp [ws, Array.getElem_ofFn]
  have hwin : ∀ t : Fin n, 1 ≤ d t ∧ d t ≤ U t := by
    intro t
    have := refusal_none hr t (by rw [hsize]; exact t.2)
    rw [hget t] at this
    exact ⟨Nat.one_le_iff_ne_zero.mpr this.1, this.2⟩
  have hS : (ws.toList.map (·.2)).sum = ∑ t, d t := by
    simp [ws, Array.toList_ofFn, List.map_ofFn, List.sum_ofFn]
  have hn' : 0 < n := Nat.pos_of_ne_zero (by rwa [hsize] at hn)
  have hSpos : ∑ t, d t ≠ 0 := by
    have : d ⟨0, hn'⟩ ≤ ∑ t, d t := Finset.single_le_sum (fun _ _ => Nat.zero_le _) (Finset.mem_univ _)
    have := (hwin ⟨0, hn'⟩).1
    omega
  set B := budget (ws.toList.map (·.2)).sum δ with hB
  have hBle : Accounting.Poub.Draw.budget (∑ t, d t) δ ≤ (B : ℝ) := by
    rw [hB, hS]; exact budget_le hSpos hδ0
  have hδ1 : (δ : ℝ) ≤ 1 := by exact_mod_cast (le_of_lt ‹δ < 1›)
  have hB0 : (0 : ℝ) ≤ B := by
    refine le_trans ?_ hBle
    unfold Accounting.Poub.Draw.budget
    have := Real.rpow_le_one (by exact_mod_cast hδ0.le : (0 : ℝ) ≤ δ) hδ1 (by positivity : (0 : ℝ) ≤ ((∑ t, d t : ℕ) : ℝ)⁻¹)
    have : (0 : ℝ) ≤ ((∑ t, d t : ℕ) : ℝ) := by positivity
    nlinarith
  set u : ℕ → ℝ := fun i => (ws[i]!.1 : ℝ)
  set s : ℕ → ℝ := fun i => (ws[i]!.2 : ℝ)
  have hus : ∀ i ∈ order ws, 0 < s i ∧ 0 ≤ u i := by
    intro i hi
    have hi' := mem_order hi
    have := hwin ⟨i, by rwa [hsize] at hi'⟩
    have hg := hget ⟨i, by rwa [hsize] at hi'⟩
    simp only [s, u]; simp only at hg; rw [hg]
    refine ⟨by exact_mod_cast this.1, by positivity⟩
  have hsorted : (order ws).Pairwise (fun i j => u j / s j ≤ u i / s i) := by
    refine (order_sorted ws).imp ?_
    intro i j hij
    have : ((ratio ws[j]! : ℚ) : ℝ) ≤ ((ratio ws[i]! : ℚ) : ℝ) := by exact_mod_cast hij
    simpa [ratio, u, s] using this
  have hfeas : ∀ φ : ℕ → ℝ, (∀ i ∈ order ws, 0 ≤ φ i ∧ φ i ≤ 1 - s i / u i) →
      ((order ws).map fun i => s i * φ i).sum ≤ B →
      ((order ws).map fun i => φ i * u i).sum ≤ (fill ws (order ws) B : ℝ) := by
    intro φ h1 h2
    rw [fill_cast]
    exact G_ge u s φ (order ws) hsorted hus h1 B h2
  rw [hw]
  apply Real.sSup_le
  · rintro _ ⟨f, ⟨hf, hfb⟩, rfl⟩
    let φ : ℕ → ℝ := fun i => if hi : i < n then f ⟨i, hi⟩ else 0
    have hφ : ∀ t : Fin n, φ t = f t := fun t => by simp [φ, t.2]
    have conv : ∀ g : ℕ → ℝ, ((order ws).map g).sum = ∑ t : Fin n, g t := by
      intro g; rw [sum_order, hsize, ← Fin.sum_univ_eq_sum_range]
    have hU : ∀ t : Fin n, u t = U t := fun t => by simp only [u]; rw [hget t]
    have hd : ∀ t : Fin n, s t = d t := fun t => by simp only [s]; rw [hget t]
    have e1 : ((order ws).map fun i => φ i * u i).sum = ∑ t, f t * U t := by
      rw [conv]; exact Finset.sum_congr rfl fun t _ => by rw [hφ, hU]
    have e2 : ((order ws).map fun i => s i * φ i).sum = ∑ t, (d t : ℝ) * f t := by
      rw [conv]; exact Finset.sum_congr rfl fun t _ => by rw [hφ, hd]
    show ∑ t, f t * (U t : ℝ) ≤ _
    rw [← e1]
    refine hfeas φ ?_ (by rw [e2]; linarith)
    intro i hi
    have hi' : i < n := by have := mem_order hi; rwa [hsize] at this
    have := hf ⟨i, hi'⟩
    have h1 := hφ ⟨i, hi'⟩
    have h2 := hU ⟨i, hi'⟩
    have h3 := hd ⟨i, hi'⟩
    simp only at h1 h2 h3
    rw [h1, h2, h3]
    exact this
  · have := hfeas (fun _ => 0) (fun i hi => ⟨le_refl _, by
      obtain ⟨hs, _⟩ := hus i hi
      have hi' := mem_order hi
      have hw' := hwin ⟨i, by rwa [hsize] at hi'⟩
      have hg := hget ⟨i, by rwa [hsize] at hi'⟩
      simp only at hg
      simp only [s, u, hg]
      rw [sub_nonneg, div_le_one (by exact_mod_cast (lt_of_lt_of_le hw'.1 hw'.2))]
      exact_mod_cast hw'.2⟩) (by simpa using hB0)
    simpa using this

theorem pooledCoversSession : PooledCoversSession :=
  fun n _ _ _ _ D ω δ b h => pooledCovers n (fun t => D.size t ω) (fun t => D.draws t ω) δ b h

theorem line_wrong {gs : Array Line.Group} {per blocks bb cb nb : ℕ} {ws : Array (ℕ × ℕ)} {δ : ℚ}
    {rates : Line.Rates} {l : Line.Line} (h : Line.line gs per blocks bb cb nb ws δ rates = .ok l) :
    ∃ b, Pooled.pooled ws δ = .ok b ∧ l.wrong = b.wrong := by
  unfold Line.line at h
  cases hp : Pooled.pooled ws δ with
  | error e => simp [hp, bind, Except.bind] at h
  | ok b =>
    refine ⟨b, rfl, ?_⟩
    simp only [hp, bind, Except.bind] at h
    split at h
    · cases h
    · split at h
      · cases h
      · split at h
        · cases h
        · cases h; rfl

theorem lineCoversSession : LineCoversSession := by
  intro n _ _ _ _ D ω δ _ _ _ _ _ _ _ l h
  obtain ⟨b, hb, hw⟩ := line_wrong h
  rw [hw]
  exact pooledCoversSession n _ _ _ _ D ω δ b hb

end Properties.PoubCheck
