import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise
import Mathlib.Algebra.BigOperators.Group.Finset.Sigma
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.Ring.GeomSum
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Int.Interval
import Mathlib.Data.List.GetD
import Mathlib.NumberTheory.LegendreSymbol.Basic
import Mathlib.Tactic.NormNum.Basic
import Mathlib.Tactic.Ring
import Models.Accounting.Guarantees.Gates

/-!
# The circuit model's window and lower bounds, proved

`ProvenWindow` by induction over the gates (each value lies in its derived interval: `Iv.add`, `Iv.scale` and
`Iv.mul` are sound), and `ClampOrWrap` from it (in the window a clamp and a wrap are both the identity); `Fooling` by
induction over the gates (every free gate is 2-adically triangular in its window) and a count of the
charged gates' joint values; `SquaringCarry` from it at `j = 1` on the sparse family, whose squares have no carries
between bytes; `EncodeBudget` and `EncodeCost` from `EncodeChainHardness` by summing the census's units over the
windows and classes; `RhoPairs` from Fermat's little theorem (`s² = x · x^((p−1)/2) = ±x`).
-/

namespace Accounting.SecurityProofs.Gates

open Accounting.Gates Accounting.Assumptions

variable {ι : Type}

/-! ## Values -/

lemma foldl_length (C : Dag ι) (x : ι → ℤ) (vs : List ℤ) :
    (C.foldl (fun vs g => vs ++ [g.value x vs]) vs).length = vs.length + C.length := by
  induction C generalizing vs with
  | nil => simp
  | cons g C ih => simp [List.foldl_cons, ih]; omega

lemma values_length (C : Dag ι) (x : ι → ℤ) : (values C x).length = C.length := by
  simp [values, foldl_length]

lemma values_take_succ (C : Dag ι) (x : ι → ℤ) (i : ℕ) (h : i < C.length) :
    values (C.take (i + 1)) x = values (C.take i) x ++ [C[i].value x (values (C.take i) x)] := by
  rw [List.take_add_one, List.getElem?_eq_getElem h, Option.toList_some]
  simp only [values, List.foldl_append, List.foldl_cons, List.foldl_nil]

lemma values_take_length (C : Dag ι) (x : ι → ℤ) (n : ℕ) (h : n ≤ C.length) : (values (C.take n) x).length = n := by
  rw [values_length, List.length_take]; omega

lemma getD_values_take (C : Dag ι) (x : ι → ℤ) (i : ℕ) (h : i < C.length) :
    ∀ n, i < n → n ≤ C.length → (values (C.take n) x).getD i 0 = C[i].value x (values (C.take i) x) := by
  intro n hin hn
  induction n with
  | zero => omega
  | succ n ih =>
    rw [values_take_succ C x n (by omega)]
    rcases Nat.lt_or_ge i n with hi | hi
    · rw [List.getD_append _ _ _ _ (by rw [values_take_length C x n (by omega)]; exact hi)]
      exact ih hi (by omega)
    · have : i = n := by omega
      subst this
      rw [List.getD_append_right _ _ _ _ (by rw [values_take_length C x i (by omega)]),
        values_take_length C x i (by omega)]
      simp

lemma val_eq (C : Dag ι) (x : ι → ℤ) (i : ℕ) (h : i < C.length) :
    val C x i = C[i].value x (values (C.take i) x) := by
  have := getD_values_take C x i h C.length h le_rfl
  rwa [List.take_length] at this

/-! ## Free gates are 2-adically triangular -/

lemma sum_zipWith_const_congr {m : ℤ} (F F' : ℕ → ℤ) (hF : ∀ j, F j ≡ F' j [ZMOD m]) :
    ∀ (cs : List ℤ) (u : List ℕ),
      (List.zipWith (· * ·) cs (u.map F)).sum ≡ (List.zipWith (· * ·) cs (u.map F')).sum [ZMOD m]
  | [], _ => by simp [Int.ModEq.refl]
  | _ :: _, [] => by simp [Int.ModEq.refl]
  | c :: cs, j :: u => by
    simp only [List.map_cons, List.zipWith_cons_cons, List.sum_cons]
    exact ((hF j).mul_left c).add (sum_zipWith_const_congr F F' hF cs u)

lemma sum_zipWith_congr {m : ℤ} (F F' : ℕ → ℤ) (hF : ∀ j, F j ≡ F' j [ZMOD m]) :
    ∀ (u v : List ℕ),
      (List.zipWith (· * ·) (u.map F) (v.map F)).sum ≡ (List.zipWith (· * ·) (u.map F') (v.map F')).sum [ZMOD m]
  | [], _ => by simp [Int.ModEq.refl]
  | _ :: _, [] => by simp [Int.ModEq.refl]
  | i :: u, j :: v => by
    simp only [List.map_cons, List.zipWith_cons_cons, List.sum_cons]
    exact ((hF i).mul (hF j)).add (sum_zipWith_congr F F' hF u v)

lemma clampS32_of {z : ℤ} (h1 : s32Min ≤ z) (h2 : z ≤ s32Max) : clampS32 z = z := by
  unfold clampS32; rw [min_eq_right h2, max_eq_right h1]

lemma wrapS32_of {z : ℤ} (h1 : s32Min ≤ z) (h2 : z ≤ s32Max) : wrapS32 z = z := by
  unfold wrapS32
  have hmin : s32Min = -2147483648 := by norm_num [s32Min]
  have hmax : s32Max = 2147483647 := by norm_num [s32Max]
  rw [Int.emod_eq_of_lt (by omega) (by rw [hmin] at h1 ⊢; rw [hmax] at h2; norm_num; omega)]
  omega

/-- A gate's value is its exact value in its window. -/
lemma value_eq_raw (g : Gate ι) (x : ι → ℤ) (vs : List ℤ) (hg : g.Ok x vs) : g.value x vs = g.raw x vs := by
  obtain ⟨h1, h2, -, h4⟩ := hg
  unfold Gate.value
  cases hop : g.op with
  | mma a b k sat =>
    have hb := h4 (by rw [hop]; trivial)
    cases sat
    · simp only [Op.out]; exact wrapS32_of (by omega) (by omega)
    · simp only [Op.out]; exact clampS32_of (by omega) (by omega)
  | _ => simp [Op.out]

/-- **The triangular step**: an uncharged gate in its window, on operands that agree mod `m`, takes values that agree
mod `m`, given that a registered or coin input it reads agrees mod `m`. -/
lemma value_congr {m : ℤ} (g : Gate ι) (x x' : ι → ℤ) (vs vs' : List ℤ)
    (hv : ∀ j, vs.getD j 0 ≡ vs'.getD j 0 [ZMOD m]) (hc : g.op.charged = false)
    (hx : ∀ b i, b ≠ .run → g.op = .input b i → x i ≡ x' i [ZMOD m]) (hg : g.Ok x vs) (hg' : g.Ok x' vs') :
    g.value x vs ≡ g.value x' vs' [ZMOD m] := by
  rw [value_eq_raw g x vs hg, value_eq_raw g x' vs' hg']
  unfold Gate.raw Gate.operands
  cases hop : g.op with
  | input b i =>
    cases b
    · exact hx _ i (by decide) hop
    · exact hx _ i (by decide) hop
    · rw [hop] at hc; simp [Op.charged] at hc
  | const c => exact Int.ModEq.refl _
  | affine c₀ cs =>
    simp only [Op.raw]
    exact (Int.ModEq.refl c₀).add (sum_zipWith_const_congr _ _ hv cs g.args)
  | mma a b k sat =>
    simp only [Op.raw, dot]
    generalize g.args = u
    cases u with
    | nil => simp [Int.ModEq.refl]
    | cons j u =>
      simp only [List.map_cons, List.headD_cons, List.tail_cons, ← List.map_take, ← List.map_drop]
      exact (hv j).add (sum_zipWith_congr _ _ hv _ _)
  | gather => rw [hop] at hc; simp [Op.charged] at hc
  | budgeted cls f => rw [hop] at hc; simp [Op.charged] at hc

/-- Two inputs on which `C` is in its window, that agree mod `m` in every registered and coin input, and on which
every charged gate takes one value: every gate's values agree mod `m`. -/
lemma values_congr {m : ℤ} (C : Dag ι) (x x' : ι → ℤ) (hw : InWindow C x) (hw' : InWindow C x')
    (hreg : ∀ i, FreeInput C i → x i ≡ x' i [ZMOD m])
    (hch : ∀ i (h : i < C.length), C[i].op.charged = true → val C x i = val C x' i) :
    ∀ n, n ≤ C.length → ∀ o, (values (C.take n) x).getD o 0 ≡ (values (C.take n) x').getD o 0 [ZMOD m] := by
  intro n
  induction n with
  | zero => intro _ o; simp [values, Int.ModEq.refl]
  | succ n ih =>
    intro hn o
    have hn' : n < C.length := hn
    have hlen := values_take_length C x n (by omega)
    have hlen' := values_take_length C x' n (by omega)
    rw [values_take_succ C x n hn', values_take_succ C x' n hn']
    rcases Nat.lt_trichotomy o n with ho | ho | ho
    · rw [List.getD_append _ _ _ _ (by rw [hlen]; exact ho), List.getD_append _ _ _ _ (by rw [hlen']; exact ho)]
      exact ih (by omega) o
    · subst ho
      rw [List.getD_append_right _ _ _ _ (by rw [hlen]), List.getD_append_right _ _ _ _ (by rw [hlen']), hlen,
        hlen', Nat.sub_self, List.getD_cons_zero, List.getD_cons_zero]
      cases hc : C[o].op.charged
      · exact value_congr _ x x' _ _ (ih (by omega)) hc
          (fun b i hb hi => hreg i ⟨C[o], List.getElem_mem hn', b, hb, hi⟩) (hw o hn') (hw' o hn')
      · rw [← val_eq C x o hn', ← val_eq C x' o hn', hch o hn' hc]
    · rw [List.getD_append_right _ _ _ _ (by rw [hlen]; omega), List.getD_append_right _ _ _ _ (by rw [hlen']; omega),
        hlen, hlen']
      obtain ⟨k, hk⟩ : ∃ k, o - n = k + 1 := ⟨o - n - 1, by omega⟩
      rw [hk]
      simp [Int.ModEq.refl]

/-! ## The window, proven -/

/-- `z` lies in interval `a`. -/
abbrev Holds (a : Iv) (z : ℤ) : Prop := a.1 ≤ z ∧ z ≤ a.2

lemma Holds.add {a b : Iv} {u v : ℤ} (hu : Holds a u) (hv : Holds b v) : Holds (Iv.add a b) (u + v) :=
  ⟨Int.add_le_add hu.1 hv.1, Int.add_le_add hu.2 hv.2⟩

lemma Holds.scale {a : Iv} {u : ℤ} (c : ℤ) (hu : Holds a u) : Holds (Iv.scale c a) (c * u) := by
  unfold Iv.scale
  split_ifs with hc
  · exact ⟨Int.mul_le_mul_of_nonneg_left hu.1 hc, Int.mul_le_mul_of_nonneg_left hu.2 hc⟩
  · exact ⟨Int.mul_le_mul_of_nonpos_left (by omega) hu.2, Int.mul_le_mul_of_nonpos_left (by omega) hu.1⟩

lemma mul_between {p q t : ℤ} (v : ℤ) (h1 : p ≤ t) (h2 : t ≤ q) :
    min (p * v) (q * v) ≤ t * v ∧ t * v ≤ max (p * v) (q * v) := by
  rcases le_total 0 v with hv | hv
  · exact ⟨min_le_of_left_le (Int.mul_le_mul_of_nonneg_right h1 hv),
      le_max_of_le_right (Int.mul_le_mul_of_nonneg_right h2 hv)⟩
  · exact ⟨min_le_of_right_le (Int.mul_le_mul_of_nonpos_right h2 hv),
      le_max_of_le_left (Int.mul_le_mul_of_nonpos_right h1 hv)⟩

lemma Holds.mul {a b : Iv} {u v : ℤ} (hu : Holds a u) (hv : Holds b v) : Holds (Iv.mul a b) (u * v) := by
  have h := mul_between v hu.1 hu.2
  have g1 := mul_between a.1 hv.1 hv.2
  have g2 := mul_between a.2 hv.1 hv.2
  rw [mul_comm b.1 a.1, mul_comm b.2 a.1, mul_comm v a.1] at g1
  rw [mul_comm b.1 a.2, mul_comm b.2 a.2, mul_comm v a.2] at g2
  exact ⟨le_trans (min_le_min g1.1 g2.1) h.1, le_trans h.2 (max_le_max g1.2 g2.2)⟩

/-- An affine map's derived interval holds its exact value, given each operand in its interval. -/
lemma holds_scale_sum (I : ℕ → Iv) (V : ℕ → ℤ) (hIV : ∀ j, Holds (I j) (V j)) :
    ∀ (cs : List ℤ) (u : List ℕ) (s : Iv) (z : ℤ), Holds s z →
      Holds ((List.zipWith Iv.scale cs (u.map I)).foldl Iv.add s) (z + (List.zipWith (· * ·) cs (u.map V)).sum)
  | [], _, s, z, h => by simpa using h
  | _ :: _, [], s, z, h => by simpa using h
  | c :: cs, j :: u, s, z, h => by
    simp only [List.map_cons, List.zipWith_cons_cons, List.foldl_cons, List.sum_cons]
    have := holds_scale_sum I V hIV cs u _ _ (h.add ((hIV j).scale c))
    rwa [add_assoc] at this

/-- An s8-family step's derived interval holds its exact sum, given each operand in its interval. -/
lemma holds_mul_sum (I : ℕ → Iv) (V : ℕ → ℤ) (hIV : ∀ j, Holds (I j) (V j)) :
    ∀ (u w : List ℕ) (s : Iv) (z : ℤ), Holds s z →
      Holds ((List.zipWith Iv.mul (u.map I) (w.map I)).foldl Iv.add s)
        (z + (List.zipWith (· * ·) (u.map V) (w.map V)).sum)
  | [], _, s, z, h => by simpa using h
  | _ :: _, [], s, z, h => by simpa using h
  | i :: u, j :: w, s, z, h => by
    simp only [List.map_cons, List.zipWith_cons_cons, List.foldl_cons, List.sum_cons]
    have := holds_mul_sum I V hIV u w _ _ (h.add ((hIV i).mul (hIV j)))
    rwa [add_assoc] at this

/-- A derived gate's derived interval holds its exact value, given each operand in its derived interval. -/
lemma derive_holds (g : Gate ι) (x : ι → ℤ) (ivs : List Iv) (vs : List ℤ)
    (hinv : ∀ j, Holds (ivs.getD j (0, 0)) (vs.getD j 0)) (hd : g.op.IsDerived) :
    Holds (g.derive ivs) (g.raw x vs) := by
  unfold Gate.derive Gate.raw Gate.operands
  cases hop : g.op with
  | const c => exact ⟨le_rfl, le_rfl⟩
  | affine c₀ cs =>
    simp only [Op.raw]
    exact holds_scale_sum _ _ hinv cs g.args (c₀, c₀) c₀ ⟨le_rfl, le_rfl⟩
  | mma a b k sat =>
    simp only [Op.raw, dot]
    generalize g.args = u
    cases u with
    | nil => simp
    | cons j u =>
      simp only [List.map_cons, List.headD_cons, List.tail_cons, ← List.map_take, ← List.map_drop]
      exact holds_mul_sum _ _ hinv _ _ _ _ (hinv j)
  | input b i => rw [hop] at hd; exact hd.elim
  | gather => rw [hop] at hd; exact hd.elim
  | budgeted cls f => rw [hop] at hd; exact hd.elim

/-- **The window's step**: a gate whose window is proven (`Proven`'s clauses for it) and whose declared interval
holds if it isn't derived, on operands each in its derived interval, is in its window, its value in its derived
interval. -/
lemma gate_window (g : Gate ι) (x : ι → ℤ) (ivs : List Iv) (vs : List ℤ)
    (hinv : ∀ j, Holds (ivs.getD j (0, 0)) (vs.getD j 0))
    (hP1 : g.op.IsDerived → (g.derive ivs).Within g.lo g.hi)
    (hP2 : ∀ fa fb k sat, g.op = .mma fa fb k sat →
      g.args.length = 2 * k + 1 ∧ s32Min ≤ g.lo ∧ g.hi ≤ s32Max ∧
        (∀ j ∈ g.args.tail.take k, (ivs.getD j (0, 0)).Within fa.lo fa.hi) ∧
        ∀ j ∈ g.args.tail.drop k, (ivs.getD j (0, 0)).Within fb.lo fb.hi)
    (hD : ¬ g.op.IsDerived → g.lo ≤ g.raw x vs ∧ g.raw x vs ≤ g.hi) :
    g.Ok x vs ∧ Holds (g.derive ivs) (g.value x vs) := by
  have hraw : g.lo ≤ g.raw x vs ∧ g.raw x vs ≤ g.hi := by
    by_cases hd : g.op.IsDerived
    · have h1 := derive_holds g x ivs vs hinv hd
      have h2 := hP1 hd
      exact ⟨le_trans h2.1 h1.1, le_trans h1.2 h2.2⟩
    · exact hD hd
  have hmma : g.op.IsMma → s32Min ≤ g.lo ∧ g.hi ≤ s32Max := by
    intro hm
    cases hop : g.op with
    | mma a b k sat => exact ⟨(hP2 a b k sat hop).2.1, (hP2 a b k sat hop).2.2.1⟩
    | _ => rw [hop] at hm; exact hm.elim
  have hfmt : ∀ (us : List ℕ) (lo hi : ℤ), (∀ j ∈ us, (ivs.getD j (0, 0)).Within lo hi) →
      ∀ z ∈ us.map (fun j => vs.getD j 0), lo ≤ z ∧ z ≤ hi := by
    intro us lo hi h z hz
    obtain ⟨j, hj, rfl⟩ := List.mem_map.mp hz
    exact ⟨le_trans (h j hj).1 (hinv j).1, le_trans (hinv j).2 (h j hj).2⟩
  have hok : g.Ok x vs := by
    refine ⟨hraw.1, hraw.2, ?_, hmma⟩
    unfold Gate.operands
    cases hop : g.op with
    | mma a b k sat =>
      obtain ⟨hlen, -, -, hA, hB⟩ := hP2 a b k sat hop
      refine ⟨by rw [List.length_map, hlen], ?_, ?_⟩
      · rw [← List.map_tail, ← List.map_take]; exact hfmt _ _ _ hA
      · rw [← List.map_tail, ← List.map_drop]; exact hfmt _ _ _ hB
    | _ => trivial
  refine ⟨hok, ?_⟩
  rw [value_eq_raw g x vs hok]
  by_cases hd : g.op.IsDerived
  · exact derive_holds g x ivs vs hinv hd
  · have : g.derive ivs = (g.lo, g.hi) := by
      unfold Gate.derive
      cases hop : g.op with
      | input b i => rfl
      | gather => rfl
      | budgeted cls f => rfl
      | _ => rw [hop] at hd; exact absurd trivial hd
    rw [this]
    exact hraw

lemma foldl_derive_length (C : Dag ι) (ivs : List Iv) :
    (C.foldl (fun ivs g => ivs ++ [g.derive ivs]) ivs).length = ivs.length + C.length := by
  induction C generalizing ivs with
  | nil => simp
  | cons g C ih => simp [List.foldl_cons, ih]; omega

lemma intervals_take_length (C : Dag ι) (n : ℕ) (h : n ≤ C.length) : (intervals (C.take n)).length = n := by
  simp only [intervals, foldl_derive_length, List.length_nil, List.length_take]; omega

lemma intervals_take_succ (C : Dag ι) (i : ℕ) (h : i < C.length) :
    intervals (C.take (i + 1)) = intervals (C.take i) ++ [C[i].derive (intervals (C.take i))] := by
  rw [List.take_add_one, List.getElem?_eq_getElem h, Option.toList_some]
  simp only [intervals, List.foldl_append, List.foldl_cons, List.foldl_nil]

/-- Every value of a proven circuit, on an input on which its declared intervals hold, lies in its derived interval
(and past the gates, `0` in `(0, 0)`). -/
lemma window_inv (C : Dag ι) (x : ι → ℤ) (hP : Proven C) (hD : DeclaredHold C x) :
    ∀ n, n ≤ C.length → ∀ j, Holds ((intervals (C.take n)).getD j (0, 0)) ((values (C.take n) x).getD j 0) := by
  intro n
  induction n with
  | zero => intro _ j; simp [intervals, values]
  | succ n ih =>
    intro hn j
    have hn' : n < C.length := hn
    have hlen := values_take_length C x n (by omega)
    have hlenI := intervals_take_length C n (by omega)
    rw [values_take_succ C x n hn', intervals_take_succ C n hn']
    rcases Nat.lt_trichotomy j n with hj | hj | hj
    · rw [List.getD_append _ _ _ _ (by rw [hlenI]; exact hj), List.getD_append _ _ _ _ (by rw [hlen]; exact hj)]
      exact ih (by omega) j
    · subst hj
      rw [List.getD_append_right _ _ _ _ (by rw [hlenI]), List.getD_append_right _ _ _ _ (by rw [hlen]), hlenI, hlen,
        Nat.sub_self, List.getD_cons_zero, List.getD_cons_zero]
      obtain ⟨h1, -, h2⟩ := hP j hn'
      exact (gate_window C[j] x _ _ (ih (by omega)) h1 h2 (hD j hn')).2
    · rw [List.getD_append_right _ _ _ _ (by rw [hlenI]; omega), List.getD_append_right _ _ _ _ (by rw [hlen]; omega),
        hlenI, hlen]
      obtain ⟨k, hk⟩ : ∃ k, j - n = k + 1 := ⟨j - n - 1, by omega⟩
      rw [hk]
      simp

theorem ProvenWindow : Accounting.Guarantees.Gates.ProvenWindow := by
  intro ι C x hP hD i hi
  obtain ⟨h1, -, h2⟩ := hP i hi
  exact (gate_window C[i] x _ _ (window_inv C x hP hD i hi.le) h1 h2 (hD i hi)).1

theorem ClampOrWrap : Accounting.Guarantees.Gates.ClampOrWrap := by
  intro ι C C' x hP hD hl hR
  have hw := ProvenWindow C x hP hD
  suffices h : ∀ n, n ≤ C.length → values (C'.take n) x = values (C.take n) x by
    have := h C.length le_rfl
    rwa [List.take_of_length_le (le_of_eq hl), List.take_length] at this
  intro n
  induction n with
  | zero => intro _; simp [values]
  | succ n ih =>
    intro hn
    have hn' : n < C.length := hn
    have hn'' : n < C'.length := by omega
    rw [values_take_succ C x n hn', values_take_succ C' x n hn'', ih (by omega)]
    obtain ⟨hargs, hop⟩ := hR n hn' hn''
    have hraw : C'[n].raw x (values (C.take n) x) = C[n].raw x (values (C.take n) x) := by
      unfold Gate.raw Gate.operands
      rw [hargs]
      rcases hop with hop | ⟨a, b, k, s, s', h1, h2⟩
      · rw [hop]
      · rw [h1, h2]; rfl
    have hval : C'[n].value x (values (C.take n) x) = C[n].value x (values (C.take n) x) := by
      unfold Gate.value
      rw [hraw]
      rcases hop with hop | ⟨a, b, k, s, s', h1, h2⟩
      · rw [hop]
      · have hok := hw n hn'
        have hb := hok.2.2.2 (by rw [h1]; trivial)
        have hz1 : s32Min ≤ C[n].raw x (values (C.take n) x) := le_trans hb.1 hok.1
        have hz2 : C[n].raw x (values (C.take n) x) ≤ s32Max := le_trans hok.2.1 hb.2
        have hout : ∀ t, (Op.mma a b k t : Op ι).out (C[n].raw x (values (C.take n) x)) =
            C[n].raw x (values (C.take n) x) := by
          intro t
          cases t
          · exact wrapS32_of hz1 hz2
          · exact clampS32_of hz1 hz2
        rw [h1, h2, hout s', hout s]
    rw [hval]

/-! ## The fooling bound -/

theorem Fooling : Accounting.Guarantees.Gates.Fooling := by
  intro ι κ C F x j out hw hreg hdist
  classical
  let T : κ → Fin C.length → ℤ := fun s i => if C[i.1].op.charged then val C (x s) i.1 else 0
  let box : Finset (Fin C.length → ℤ) :=
    Fintype.piFinset fun i => if C[i.1].op.charged then Finset.Icc C[i.1].lo C[i.1].hi else {0}
  have hmaps : Set.MapsTo T F box := by
    intro s hs
    simp only [Finset.mem_coe] at hs
    simp only [box, Finset.mem_coe, Fintype.mem_piFinset]
    intro i
    by_cases hc : C[i.1].op.charged
    · simp only [T, hc, ite_true]
      have hok := hw s hs i.1 i.2
      rw [Finset.mem_Icc, val_eq C (x s) i.1 i.2, value_eq_raw _ _ _ hok]
      exact ⟨hok.1, hok.2.1⟩
    · simp [T, hc]
  have hinj : Set.InjOn T F := by
    intro s hs s' hs' hT
    simp only [Finset.mem_coe] at hs hs'
    by_contra hne
    obtain ⟨o, -, ho⟩ := hdist s hs s' hs' hne
    apply ho
    have h := values_congr C (x s) (x s') (hw s hs) (hw s' hs') (hreg s hs s' hs')
      (fun i h hc => by simpa [T, hc] using congrFun hT ⟨i, h⟩) C.length le_rfl o
    rwa [List.take_length] at h
  calc F.card ≤ box.card := Finset.card_le_card_of_injOn T hmaps hinj
    _ = volume C := by
      rw [Fintype.card_piFinset, volume, ← Fin.prod_univ_fun_getElem C Gate.charge]
      apply Finset.prod_congr rfl
      intro i _
      unfold Gate.charge Gate.size
      split_ifs <;> simp [Int.card_Icc]

/-! ## The carry bound -/

/-- Byte `l` of a base-256 numeral whose digits are `f`. -/
lemma byte_sum : ∀ (M : ℕ) (f : ℕ → ℕ), (∀ m, f m < 256) → ∀ l,
    byte (∑ m ∈ Finset.range M, f m * 256 ^ m) l = if l < M then f l else 0
  | 0, f, _, l => by simp [byte]
  | M + 1, f, hf, l => by
    have hR : ∑ m ∈ Finset.range (M + 1), f m * 256 ^ m =
        f 0 + 256 * ∑ m ∈ Finset.range M, f (m + 1) * 256 ^ m := by
      rw [Finset.sum_range_succ', Finset.mul_sum, add_comm]
      congr 1
      · simp
      · apply Finset.sum_congr rfl; intro m _; ring
    rw [hR]
    have h0 := hf 0
    cases l with
    | zero =>
      simp only [byte, pow_zero, Nat.div_one]
      rw [ite_eq_left (Nat.succ_pos M)]
      omega
    | succ l =>
      have ih := byte_sum M (fun m => f (m + 1)) (fun m => hf _) l
      unfold byte at ih ⊢
      rw [pow_succ', ← Nat.div_div_eq_div_mul,
        show (f 0 + 256 * ∑ m ∈ Finset.range M, f (m + 1) * 256 ^ m) / 256 =
          ∑ m ∈ Finset.range M, f (m + 1) * 256 ^ m by omega, ih]
      simp

lemma byte_mul_256 (N l : ℕ) : byte (256 * N) (l + 1) = byte N l := by
  unfold byte
  rw [pow_succ', Nat.mul_div_mul_left _ _ (by norm_num : 0 < 256)]

/-- `ν_m`: the pairs of `S` that sum to `m`, the coefficient of `256^m` in `(Σ_{i ∈ S} 256^i)²`. -/
def nu (S : Finset ℕ) (m : ℕ) : ℕ := ((S ×ˢ S).filter fun q => q.1 + q.2 = m).card

lemma nu_le (S : Finset ℕ) (m : ℕ) : nu S m ≤ S.card := by
  unfold nu
  apply Finset.card_le_card_of_injOn Prod.fst
  · intro q hq
    simp only [Finset.coe_filter, Finset.mem_product, Set.mem_ofPred_eq] at hq
    exact hq.1.1
  · intro q hq q' hq' h
    simp only [Finset.coe_filter, Finset.mem_product, Set.mem_ofPred_eq] at hq hq'
    exact Prod.ext h (by omega)

lemma nu_two_mul_mod (S : Finset ℕ) (i : ℕ) : nu S (2 * i) % 2 = if i ∈ S then 1 else 0 := by
  have hsplit : nu S (2 * i) =
      ((S ×ˢ S).filter fun q => q.1 + q.2 = 2 * i ∧ q.1 < q.2).card +
        ((S ×ˢ S).filter fun q => q.1 + q.2 = 2 * i ∧ q.2 < q.1).card +
        ((S ×ˢ S).filter fun q => q.1 + q.2 = 2 * i ∧ q.1 = q.2).card := by
    unfold nu
    simp only [Finset.card_filter, ← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro q _
    split_ifs <;> omega
  have hswap : ((S ×ˢ S).filter fun q => q.1 + q.2 = 2 * i ∧ q.1 < q.2).card =
      ((S ×ˢ S).filter fun q => q.1 + q.2 = 2 * i ∧ q.2 < q.1).card := by
    apply Finset.card_nbij' Prod.swap Prod.swap
    · intro q hq
      simp only [Finset.coe_filter, Finset.mem_product, Set.mem_ofPred_eq, Prod.fst_swap, Prod.snd_swap] at hq ⊢
      exact ⟨⟨hq.1.2, hq.1.1⟩, by omega, hq.2.2⟩
    · intro q hq
      simp only [Finset.coe_filter, Finset.mem_product, Set.mem_ofPred_eq, Prod.fst_swap, Prod.snd_swap] at hq ⊢
      exact ⟨⟨hq.1.2, hq.1.1⟩, by omega, hq.2.2⟩
    · intro q _; exact Prod.swap_swap q
    · intro q _; exact Prod.swap_swap q
  have hdiag : ((S ×ˢ S).filter fun q => q.1 + q.2 = 2 * i ∧ q.1 = q.2).card = if i ∈ S then 1 else 0 := by
    by_cases hi : i ∈ S
    · rw [ite_eq_left hi, Finset.card_eq_one]
      refine ⟨(i, i), ?_⟩
      ext ⟨a, b⟩
      simp only [Finset.mem_filter, Finset.mem_product, Finset.mem_singleton, Prod.mk.injEq]
      constructor
      · rintro ⟨-, h1, h2⟩; omega
      · rintro ⟨rfl, rfl⟩; exact ⟨⟨hi, hi⟩, by omega, rfl⟩
    · rw [ite_eq_right hi, Finset.card_eq_zero, Finset.filter_eq_empty_iff]
      rintro ⟨a, b⟩ hq ⟨h1, h2⟩
      rw [Finset.mem_product] at hq
      have : a = i := by simp only at h1 h2; omega
      exact hi (this ▸ hq.1)
  rw [hsplit, hswap, hdiag]
  split_ifs <;> omega

lemma sq_sum (S : Finset ℕ) (s : ℕ) (hS : S ⊆ Finset.range s) :
    (∑ i ∈ S, 256 ^ i) ^ 2 = ∑ m ∈ Finset.range (2 * s), nu S m * 256 ^ m := by
  have hmaps : ∀ q ∈ S ×ˢ S, (fun q : ℕ × ℕ => q.1 + q.2) q ∈ Finset.range (2 * s) := by
    intro q hq
    rw [Finset.mem_product] at hq
    have h1 := Finset.mem_range.mp (hS hq.1)
    have h2 := Finset.mem_range.mp (hS hq.2)
    exact Finset.mem_range.mpr (by dsimp only; omega)
  rw [sq, Finset.sum_mul_sum, ← Finset.sum_product' (f := fun a b => 256 ^ a * 256 ^ b),
    ← Finset.sum_fiberwise_of_maps_to hmaps]
  apply Finset.sum_congr rfl
  intro m _
  rw [nu, Finset.card_eq_sum_ones, Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro q hq
  rw [Finset.mem_filter] at hq
  rw [← pow_add, hq.2, one_mul]

lemma sum_pow_lt (S : Finset ℕ) (s : ℕ) (hS : S ⊆ Finset.range s) : ∑ i ∈ S, 256 ^ i < 256 ^ s :=
  Nat.geomSum_lt (by norm_num) fun k hk => Finset.mem_range.mp (hS hk)

lemma sparse_sq_lt (S : Finset ℕ) (s : ℕ) (hS : S ⊆ Finset.range s) : sparse S ^ 2 < 256 ^ (2 * s + 1) := by
  have h := sum_pow_lt S s hS
  have : sparse S < 16 * 256 ^ s := by unfold sparse; omega
  calc sparse S ^ 2 < (16 * 256 ^ s) ^ 2 := Nat.pow_lt_pow_left this (by norm_num)
    _ = 256 ^ (2 * s + 1) := by ring

lemma sparse_sq (S : Finset ℕ) : sparse S ^ 2 = 256 * (∑ i ∈ S, 256 ^ i) ^ 2 := by unfold sparse; ring

/-- Byte `2i + 1` of `a_S²` is `ν_{2i}`, odd exactly when `i ∈ S`. -/
lemma byte_sparse_sq (S : Finset ℕ) (s i : ℕ) (hs : s ≤ 255) (hS : S ⊆ Finset.range s) (hi : i < s) :
    byte (sparse S ^ 2) (2 * i + 1) % 2 = if i ∈ S then 1 else 0 := by
  have hcard : S.card ≤ s := by simpa using Finset.card_le_card hS
  rw [sparse_sq, byte_mul_256, sq_sum S s hS,
    byte_sum (2 * s) (nu S) (fun m => by have := nu_le S m; omega), ite_eq_left (by omega)]
  exact nu_two_mul_mod S i

lemma sparse_eq (S : Finset ℕ) (s : ℕ) (hS : S ⊆ Finset.range s) :
    sparse S = ∑ m ∈ Finset.range s, (if m ∈ S then 16 else 0) * 256 ^ m := by
  unfold sparse
  rw [Finset.mul_sum]
  simp only [ite_mul, zero_mul]
  rw [Finset.sum_ite_mem, Finset.inter_eq_right.mpr hS]

/-- Every byte of `a_S` is even. -/
lemma byte_sparse_even (S : Finset ℕ) (s l : ℕ) (hS : S ⊆ Finset.range s) : byte (sparse S) l % 2 = 0 := by
  rw [sparse_eq S s hS, byte_sum s (fun m => if m ∈ S then 16 else 0) (fun m => by split_ifs <;> norm_num)]
  split_ifs <;> norm_num

theorem SquaringCarry : Accounting.Guarantees.Gates.SquaringCarry := by
  intro ι C p n s G x slot out hs hp hn hG hreg hout
  have hsub : ∀ S ∈ G, S ⊆ Finset.range s := fun S hS => Finset.mem_powerset.mp (hG hS)
  have hmod : ∀ S ∈ G, sparse S ^ 2 % p = sparse S ^ 2 := fun S hS =>
    Nat.mod_eq_of_lt (lt_of_lt_of_le (sparse_sq_lt S s (hsub S hS)) hp)
  refine Fooling C G x 1 ((List.range n).map out) (fun S hS => (hout S hS).1) ?_ ?_
  · intro S hS S' hS' i hi
    rw [hreg S hS i hi, hreg S' hS' i hi, pow_one]
    have h1 := byte_sparse_even S s (slot i) (hsub S hS)
    have h2 := byte_sparse_even S' s (slot i) (hsub S' hS')
    unfold Int.ModEq
    omega
  · intro S hS S' hS' hne
    obtain ⟨i, hi⟩ : ∃ i, ¬ (i ∈ S ↔ i ∈ S') := by
      by_contra h
      exact hne (Finset.ext fun a => by by_contra ha; exact h ⟨a, ha⟩)
    have his : i < s := by
      by_cases h : i ∈ S
      · exact Finset.mem_range.mp (hsub S hS h)
      · have h' : i ∈ S' := by
          by_contra h'
          exact hi ⟨fun h1 => absurd h1 h, fun h1 => absurd h1 h'⟩
        exact Finset.mem_range.mp (hsub S' hS' h')
    refine ⟨out (2 * i + 1), List.mem_map.mpr ⟨2 * i + 1, List.mem_range.mpr (by omega), rfl⟩, ?_⟩
    rw [(hout S hS).2 _ (by omega), (hout S' hS').2 _ (by omega), hmod S hS, hmod S' hS', pow_one]
    have h1 := byte_sparse_sq S s i hs (hsub S hS) his
    have h2 := byte_sparse_sq S' s i hs (hsub S' hS') his
    unfold Int.ModEq
    by_cases h : i ∈ S
    · have h' : i ∉ S' := fun h' => hi ⟨fun _ => h', fun _ => h⟩
      rw [ite_eq_left h] at h1
      rw [ite_eq_right h'] at h2
      omega
    · have h' : i ∈ S' := by
        by_contra h'
        exact hi ⟨fun h1 => absurd h1 h, fun h1 => absurd h1 h'⟩
      rw [ite_eq_right h] at h1
      rw [ite_eq_left h'] at h2
      omega

/-! ## The encode budget -/

lemma countAt_cons (g : Gate ι) (C : Dag ι) (s : ℕ → Prop) [DecidablePred s] (p : Gate ι → ℕ) :
    countAt (g :: C) s p = (if s g.step then p g else 0) + countAt C s p := by
  unfold countAt
  by_cases hs : s g.step <;> simp [hs]

/-- Summing a gate count over the windows: every gate is in one of them. -/
lemma sum_countAt (C : Dag ι) (win : ℕ → ℕ) (N : ℕ) (p : Gate ι → ℕ) (h : ∀ g ∈ C, win g.step < N) :
    ∑ v ∈ Finset.range N, countAt C (fun t => win t = v) p = (C.map p).sum := by
  induction C with
  | nil => simp [countAt]
  | cons g C ih =>
    have hg := h g (List.mem_cons_self ..)
    have ih := ih fun g' hg' => h g' (List.mem_cons_of_mem _ hg')
    simp only [countAt_cons, Finset.sum_add_distrib, Finset.sum_ite_eq, Finset.mem_range, hg, ite_true, ih,
      List.map_cons, List.sum_cons]

/-- A gate of no budgeted class has no units. -/
lemma units_of_budget_none (C : Dag ι) (i : ℕ) (g : Gate ι) (hg : C[i]? = some g) (hb : g.op.budget = none) :
    units C i = 0 := by
  simp only [units, hg]
  split <;> simp_all [Op.budget]

/-- Summing the census's units over the windows and classes: each gate's units are in its window's census, under its
class. -/
lemma budgeted_eq (C : Dag ι) (A : Audit) (h : ∀ g ∈ C, A.win g.step < A.windows) :
    budgeted C = ∑ v ∈ Finset.range A.windows, (census C fun t => A.win t = v).budgeted := by
  simp only [census, Census.budgeted, unitsAt, ← Finset.sum_add_distrib]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun i hi => ?_
  have hi' : i < C.length := Finset.mem_range.mp hi
  have hg : C[i]? = some C[i] := List.getElem?_eq_getElem hi'
  have hw := h C[i] (List.getElem_mem hi')
  simp only [hg, Option.any_some]
  rcases hb : C[i].op.budget with _ | b
  · simp [units_of_budget_none C i C[i] hg hb]
  · cases b <;> simp [Finset.sum_ite_eq, hw]

theorem EncodeBudget : Accounting.Guarantees.Gates.EncodeBudget := by
  intro F k B E εnum εden ι W caps A m x words out H hP hR hci hnd hlt hcw
  have h := H W m x words out hR hci hnd hlt hcw
  obtain ⟨hwin, hcap, -⟩ := hP
  have hb : budgeted W ≤ caps.total * ∑ v ∈ Finset.range A.windows, A.reads v := by
    rw [budgeted_eq W A hwin, Finset.mul_sum]
    apply Finset.sum_le_sum
    intro v hv
    obtain ⟨h1, h2, h3, h4, h5, -⟩ := hcap v (Finset.mem_range.mp hv)
    simp only [Census.budgeted, Caps.total, add_mul]
    omega
  have hr : runBits W ≤ 8 * caps.runBytes * A.windows := by
    rw [runBits, ← sum_countAt W A.win A.windows _ hwin]
    calc _ ≤ ∑ v ∈ Finset.range A.windows, 8 * caps.runBytes :=
          Finset.sum_le_sum fun v hv => (hcap v (Finset.mem_range.mp hv)).2.2.2.2.2
      _ = 8 * caps.runBytes * A.windows := by simp [mul_comm]
  calc words.length * E * (F.n - 1) ≤ budgeted W * (F.n - 1) + runBits W * E := h
    _ ≤ _ := add_le_add (Nat.mul_le_mul_right _ hb) (Nat.mul_le_mul_right _ hr)

theorem EncodeCost : Accounting.Guarantees.Gates.EncodeCost := by
  intro F k B E εnum εden q H hn
  unfold encodeCost
  refine le_iInf₂ fun c hc => ?_
  obtain ⟨ι, C, m, x, words, out, hlen, hnd, hlt, hR, hrb, hci, hcw, rfl⟩ := hc
  have h := H C m x words out hR hci hnd hlt hcw
  rw [hrb, zero_mul, add_zero, hlen] at h
  exact_mod_cast Nat.le_of_mul_le_mul_right h (by omega)

/-! ## `ρ` on pairs -/

/-- A natural below `p` is its residue: two that agree mod `p` are equal. -/
lemma eq_of_cast_eq {p a b : ℕ} (ha : a < p) (hb : b < p) (h : (a : ZMod p) = b) : a = b := by
  rw [ZMod.natCast_eq_natCast_iff'] at h
  rwa [Nat.mod_eq_of_lt ha, Nat.mod_eq_of_lt hb] at h

/-- `ρ(x)` is `s = x^((p+1)/4) mod p` or `p − s`, of parity `[s² ≢ x]`, when `0 < s < p` and `p` is odd. -/
lemma rho_cases (p x : ℕ) (hp : p % 2 = 1) (h0 : 0 < x ^ ((p + 1) / 4) % p) (hsp : x ^ ((p + 1) / 4) % p < p) :
    (rho p x = x ^ ((p + 1) / 4) % p ∨ rho p x = p - x ^ ((p + 1) / 4) % p) ∧
      rho p x % 2 = if x ^ ((p + 1) / 4) % p * (x ^ ((p + 1) / 4) % p) % p = x then 0 else 1 := by
  simp only [rho]
  generalize x ^ ((p + 1) / 4) % p = s at h0 hsp ⊢
  split_ifs <;> omega

/-- At `p ≡ 3 mod 4`, the square of `a^((p+1)/4)` is `a · a^(p/2)`, so `±a`. -/
lemma sq_pow_quarter {p : ℕ} [Fact p.Prime] (hp : p % 4 = 3) (a : ZMod p) :
    (a ^ ((p + 1) / 4)) ^ 2 = a * a ^ (p / 2) := by
  rw [← pow_mul, ← pow_succ']
  congr 1
  omega

/-- `(p − a) mod p` is `−a`. -/
lemma cast_sub_self {p a : ℕ} (h : a ≤ p) : ((p - a : ℕ) : ZMod p) = -(a : ZMod p) := by
  rw [Nat.cast_sub h, ZMod.natCast_self, zero_sub]

/-- A residue `b < p` whose cast is `±a`'s, for `0 < a < p`, is `a` or `p − a`. -/
lemma eq_or_sub_of_cast {p a b : ℕ} (ha : 0 < a) (hap : a < p) (hb : b < p)
    (h : (b : ZMod p) = a ∨ (b : ZMod p) = -(a : ZMod p)) : b = a ∨ b = p - a := by
  rcases h with h | h
  · exact Or.inl (eq_of_cast_eq hb hap h)
  · rw [← cast_sub_self hap.le] at h
    exact Or.inr (eq_of_cast_eq hb (by omega) h)

theorem RhoPairs : Accounting.Guarantees.Gates.RhoPairs := by
  intro p x hpr hp hx hxp
  have := Fact.mk hpr
  have hpos : 0 < p := hpr.pos
  have hxz : (x : ZMod p) ≠ 0 := by
    rw [Ne, ZMod.natCast_eq_zero_iff]
    exact Nat.not_dvd_of_pos_of_lt hx hxp
  have hrt : ∀ a : ℕ, (a : ZMod p) ≠ 0 → 0 < a ^ ((p + 1) / 4) % p := fun a ha => by
    refine Nat.pos_of_ne_zero fun h => pow_ne_zero ((p + 1) / 4) ha ?_
    rw [← Nat.cast_pow, ← ZMod.natCast_mod, h, Nat.cast_zero]
  have hmx := cast_sub_self (p := p) hxp.le
  have h1 := rho_cases p x (by omega) (hrt x hxz) (Nat.mod_lt _ hpos)
  have h2 := rho_cases p (p - x) (by omega) (hrt (p - x) (by rw [hmx]; exact neg_ne_zero.mpr hxz))
    (Nat.mod_lt _ hpos)
  have hcast : ∀ a : ℕ, ((a ^ ((p + 1) / 4) % p : ℕ) : ZMod p) = (a : ZMod p) ^ ((p + 1) / 4) := fun a => by
    rw [ZMod.natCast_mod, Nat.cast_pow]
  generalize hs : x ^ ((p + 1) / 4) % p = s at h1
  generalize hs' : (p - x) ^ ((p + 1) / 4) % p = s' at h2
  have hsc : (s : ZMod p) = (x : ZMod p) ^ ((p + 1) / 4) := by rw [← hs, hcast]
  have hsc' : (s' : ZMod p) = (-(x : ZMod p)) ^ ((p + 1) / 4) := by rw [← hs', hcast, hmx]
  have hs0 : 0 < s := hs ▸ hrt x hxz
  have hsp : s < p := hs ▸ Nat.mod_lt _ hpos
  have hsp' : s' < p := hs' ▸ Nat.mod_lt _ hpos
  have hsq : (s : ZMod p) ^ 2 = (x : ZMod p) * (x : ZMod p) ^ (p / 2) := by rw [hsc, sq_pow_quarter hp]
  have hss : s * s % p = x ∨ s * s % p = p - x := by
    refine eq_or_sub_of_cast hx hxp (Nat.mod_lt _ hpos) ?_
    rw [ZMod.natCast_mod, Nat.cast_mul, ← sq, hsq]
    rcases ZMod.pow_div_two_eq_neg_one_or_one p hxz with h | h <;> rw [h] <;> simp
  have hss' : s' * s' % p = s * s % p := by
    rw [← ZMod.natCast_eq_natCast_iff', Nat.cast_mul, Nat.cast_mul, ← sq, ← sq, hsc, hsc', ← pow_mul, ← pow_mul,
      mul_comm _ 2, pow_mul, pow_mul, neg_sq]
  have hs's : s' = s ∨ s' = p - s := by
    refine eq_or_sub_of_cast hs0 hsp hsp' ?_
    rw [hsc', neg_pow, ← hsc]
    rcases neg_one_pow_eq_or (ZMod p) ((p + 1) / 4) with h | h <;> rw [h] <;> simp
  refine ⟨?_, ?_⟩
  · rw [hss'] at h2
    generalize s * s % p = r at hss h1 h2
    obtain ⟨h1a, h1b⟩ := h1
    obtain ⟨h2a, h2b⟩ := h2
    split_ifs at h1b h2b <;> omega
  · rw [h1.2]
    have hsq_iff : s * s % p = x ↔ ∃ y, y ^ 2 % p = x := by
      refine ⟨fun h => ⟨s, by rw [sq]; exact h⟩, fun ⟨y, hy⟩ => ?_⟩
      have hyx : (y : ZMod p) ^ 2 = x := by rw [← hy, ZMod.natCast_mod, Nat.cast_pow]
      have hy0 : (y : ZMod p) ≠ 0 := fun h => hxz (by rw [← hyx, h, zero_pow two_ne_zero])
      have he : (x : ZMod p) ^ (p / 2) = 1 := by
        rw [← hyx, ← pow_mul, show 2 * (p / 2) = p - 1 by omega]
        exact ZMod.pow_card_sub_one_eq_one hy0
      refine eq_of_cast_eq (Nat.mod_lt _ hpos) hxp ?_
      rw [ZMod.natCast_mod, Nat.cast_mul, ← sq, hsq, he, mul_one]
    rw [← hsq_iff]
    split_ifs <;> simp_all

/-! ## The codec at toy fields, against `codec.py` -/

/-- Rows `(n, c, c_r, K, τ, i, m, y)`: `codec.encode_word(m, i)` is `y` at `p = 2^n − c` (251 and 65519) under round
constants `c_r`, mask `K` and tweak multiplier `τ`. Between them they take both arms of `σ_K` and every arm of `ρ`
(a square, a non-square, zero). `test_encode_chain.py` reads them from this file and recomputes each with
`codec.encode_word`. -/
def codecRows : List (ℕ × ℕ × List ℕ × ℕ × ℕ × ℕ × ℕ × ℕ) := [
  (8, 5, [17, 200, 99], 248, 3, 5, 219, 234),
  (8, 5, [92, 7], 253, 18, 54, 195, 147),
  (8, 5, [44, 209], 252, 132, 41, 181, 166),
  (8, 5, [139, 0], 254, 232, 36, 68, 225),
  (8, 5, [56, 35], 12, 69, 52, 101, 18),
  (8, 5, [88, 65], 93, 168, 12, 225, 4),
  (8, 5, [103, 10, 98], 252, 50, 51, 47, 10),
  (8, 5, [156, 202], 253, 47, 41, 11, 126),
  (16, 17, [26622, 53477], 34513, 22959, 218370, 28017, 16283),
  (16, 17, [27420, 1802, 17095], 22236, 6729, 817333, 53283, 48655),
  (16, 17, [60688, 23696, 15300], 349, 10049, 806969, 33473, 64012)]

/-- `Encodes`'s word, `encodeWord F.p d.key (d.tweak F i) m`, is `codec.encode_word`'s on every row. -/
theorem encodeWord_codecRows :
    codecRows.all (fun (n, c, cs, K, τ, i, m, y) =>
      let F : Field := ⟨n, c⟩
      let d : Draw cs.length := (fun r => cs.getD r 0, K, τ)
      encodeWord F.p d.key (d.tweak F i) m == y) := by
  decide +kernel

end Accounting.SecurityProofs.Gates
