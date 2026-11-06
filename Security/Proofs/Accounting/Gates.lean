import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise
import Mathlib.Algebra.BigOperators.Group.Finset.Sigma
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.Ring.GeomSum
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Int.Interval
import Mathlib.Data.List.GetD
import Mathlib.NumberTheory.LegendreSymbol.Basic
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.NormNum.Basic
import Mathlib.Tactic.Ring
import Definitions.Accounting.Guarantees.Gates

/-!
# The circuit model's window and lower bounds, proved

`ProvenWindow` by induction over the gates (each value lies in its derived interval: `Iv.add`, `Iv.scale` and
`Iv.mul` are sound, and a gather's value is a column word or its fill, in the hull `Iv.join` gives), and `ClampOrWrap`
from it (in the window a clamp and a wrap are both the identity); `Fooling` by induction over the gates (every free
gate is 2-adically triangular in its window) and a count of the charged gates' joint values; `SquaringCarry` from it
at `j = 1` on the sparse family, whose squares have no carries between bytes; `SquaringCarryNibbles` from it at
`j = 4` on the nibble family, whose squares first differ two bytes above their first differing nibble
(`a² − a'² = (a − a')(a + a')`); `UnitsFooling` by the same induction, the trace being each dynamic run input's and
budgeted gate's value and each dynamic index's two classes, `idxCls b v` (`min(v, 2^b − 1)`, and `2^b − 1` for a
negative `v`, which reads the fill as an index past the column does) at the most index bits `b` its gathers read and
`v mod 2^b'` at the most `b'` its table reads read, which the fresh index bits of each kind pay for; `SquaringUnits`
from it on the nibble family; `FoolingCap` by injecting the family into its outputs' residues mod `2^j`;
`EncodeBudget` and `EncodeCost` from `SquaringFloor` and `ChainSum` (the encode bound at `k · S · e`, so at any
`E` below it) by summing the census's units and data products, and each window's run-input bound, over the windows and
classes; `RhoPairs` from Fermat's little theorem
(`s² = x · x^((p−1)/2) = ±x`). `DeclaredWindow` by `ProvenWindow`'s induction, each gate's declared bound earned from
the earlier gates' intervals; `Int8LinearWindow` and `LutPvWindow` from `dotChain_ok`, which proves any s8 dot chain
over declared operands whose `32·S` products fit s32. `Bf16ToF32Affine` by the registry's entry and, on a 16-bit
word, the mask's identity and `(u + 2^15) mod 2^16 · 2^16 = (u · 2^16 + 2^31) mod 2^32`.
-/

namespace Accounting.SecurityProofs.Gates

open Accounting.Gates Accounting.Assumptions

variable {ι : Type}

/-! ## Values -/

lemma foldl_length (C : IntDag ι) (x : ι → ℤ) (vs : List ℤ) :
    (C.foldl (fun vs g => vs ++ [g.value x vs]) vs).length = vs.length + C.length := by
  induction C generalizing vs with
  | nil => simp
  | cons g C ih => simp [List.foldl_cons, ih]; omega

lemma values_length (C : IntDag ι) (x : ι → ℤ) : (values C x).length = C.length := by
  simp [values, foldl_length]

lemma values_take_succ (C : IntDag ι) (x : ι → ℤ) (i : ℕ) (h : i < C.length) :
    values (C.take (i + 1)) x = values (C.take i) x ++ [C[i].value x (values (C.take i) x)] := by
  rw [List.take_add_one, List.getElem?_eq_getElem h, Option.toList_some]
  simp only [values, List.foldl_append, List.foldl_cons, List.foldl_nil]

lemma values_take_length (C : IntDag ι) (x : ι → ℤ) (n : ℕ) (h : n ≤ C.length) : (values (C.take n) x).length = n := by
  rw [values_length, List.length_take]; omega

lemma getD_values_take (C : IntDag ι) (x : ι → ℤ) (i : ℕ) (h : i < C.length) :
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

lemma val_eq (C : IntDag ι) (x : ι → ℤ) (i : ℕ) (h : i < C.length) :
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
lemma value_eq_raw (g : IntGate ι) (x : ι → ℤ) (vs : List ℤ) (hg : g.Ok x vs) : g.value x vs = g.raw x vs := by
  obtain ⟨h1, h2, -, h4⟩ := hg
  unfold IntGate.value
  cases hop : g.op with
  | mma a b k sat =>
    have hb := h4 (by rw [hop]; trivial)
    cases sat
    · simp only [IntOp.out]; exact wrapS32_of (by omega) (by omega)
    · simp only [IntOp.out]; exact clampS32_of (by omega) (by omega)
  | _ => simp [IntOp.out]

/-- **The triangular step**: an uncharged gate in its window, on operands that agree mod `m`, takes values that agree
mod `m`, given that a registered or coin input it reads agrees mod `m`. -/
lemma value_congr {m : ℤ} (g : IntGate ι) (x x' : ι → ℤ) (vs vs' : List ℤ)
    (hv : ∀ j, vs.getD j 0 ≡ vs'.getD j 0 [ZMOD m]) (hc : g.op.charged = false)
    (hx : ∀ b i, b ≠ .run → g.op = .input b i → x i ≡ x' i [ZMOD m]) (hg : g.Ok x vs) (hg' : g.Ok x' vs') :
    g.value x vs ≡ g.value x' vs' [ZMOD m] := by
  rw [value_eq_raw g x vs hg, value_eq_raw g x' vs' hg']
  unfold IntGate.raw IntGate.operands
  cases hop : g.op with
  | input b i =>
    cases b
    · exact hx _ i (by decide) hop
    · exact hx _ i (by decide) hop
    · rw [hop] at hc; simp [IntOp.charged] at hc
  | const c => exact Int.ModEq.refl _
  | affine c₀ cs =>
    simp only [IntOp.raw]
    exact (Int.ModEq.refl c₀).add (sum_zipWith_const_congr _ _ hv cs g.args)
  | mma a b k sat =>
    simp only [IntOp.raw, dot]
    generalize g.args = u
    cases u with
    | nil => simp [Int.ModEq.refl]
    | cons j u =>
      simp only [List.map_cons, List.headD_cons, List.tail_cons, ← List.map_take, ← List.map_drop]
      exact (hv j).add (sum_zipWith_congr _ _ hv _ _)
  | gather _ _ => rw [hop] at hc; simp [IntOp.charged] at hc
  | budgeted cls f => rw [hop] at hc; simp [IntOp.charged] at hc

/-- Two inputs on which `C` is in its window, that agree mod `m` in every registered and coin input, and on which
every charged gate takes one value: every gate's values agree mod `m`. -/
lemma values_congr {m : ℤ} (C : IntDag ι) (x x' : ι → ℤ) (hw : InWindow C x) (hw' : InWindow C x')
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

/-- A hull holds what its start or any of its intervals holds. -/
lemma holds_foldl_join : ∀ (l : List Iv) (s : Iv) (z : ℤ), (Holds s z ∨ ∃ b ∈ l, Holds b z) →
    Holds (l.foldl Iv.join s) z
  | [], s, z, h => by simpa using h
  | b :: l, s, z, h => by
    simp only [List.foldl_cons]
    refine holds_foldl_join l _ z ?_
    rcases h with h | ⟨b', hb', h⟩
    · exact .inl ⟨min_le_of_left_le h.1, le_max_of_le_left h.2⟩
    · rcases List.mem_cons.mp hb' with rfl | hb'
      · exact .inl ⟨min_le_of_right_le h.1, le_max_of_le_right h.2⟩
      · exact .inr ⟨b', hb', h⟩

/-- A derived gate's derived interval holds its exact value, given each operand in its derived interval. -/
lemma derive_holds (g : IntGate ι) (x : ι → ℤ) (ivs : List Iv) (vs : List ℤ)
    (hinv : ∀ j, Holds (ivs.getD j (0, 0)) (vs.getD j 0)) (hd : g.op.IsDerived) :
    Holds (g.derive ivs) (g.raw x vs) := by
  unfold IntGate.derive IntGate.raw IntGate.operands
  cases hop : g.op with
  | const c => exact ⟨le_rfl, le_rfl⟩
  | affine c₀ cs =>
    simp only [IntOp.raw]
    exact holds_scale_sum _ _ hinv cs g.args (c₀, c₀) c₀ ⟨le_rfl, le_rfl⟩
  | mma a b k sat =>
    simp only [IntOp.raw, dot]
    generalize g.args = u
    cases u with
    | nil => simp
    | cons j u =>
      simp only [List.map_cons, List.headD_cons, List.tail_cons, ← List.map_take, ← List.map_drop]
      exact holds_mul_sum _ _ hinv _ _ _ _ (hinv j)
  | gather low d =>
    simp only [IntOp.raw]
    generalize g.args = u
    cases u with
    | nil => simp
    | cons j u =>
      simp only [List.map_cons, List.headD_cons, List.tail_cons]
      refine holds_foldl_join _ _ _ ?_
      generalize IntOp.gatherAt low _ (vs.getD j 0) = k
      rcases Nat.lt_or_ge k u.length with hk | hk
      · refine .inr ⟨_, List.mem_map.mpr ⟨_, List.getElem_mem hk, rfl⟩, ?_⟩
        rw [List.getD_eq_getElem (u.map fun c => vs.getD c 0) d (by simpa using hk), List.getElem_map]
        exact hinv _
      · exact .inl (by rw [List.getD_eq_default _ _ (by simpa using hk)]; exact ⟨le_rfl, le_rfl⟩)
  | input b i => rw [hop] at hd; exact hd.elim
  | budgeted cls f => rw [hop] at hd; exact hd.elim

/-- **The window's step**: a gate whose window is proven (`Proven`'s clauses for it) and whose declared interval
holds if it isn't derived, on operands each in its derived interval, is in its window, its value in its derived
interval. -/
lemma gate_window (g : IntGate ι) (x : ι → ℤ) (ivs : List Iv) (vs : List ℤ)
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
    unfold IntGate.operands
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
      unfold IntGate.derive
      cases hop : g.op with
      | input b i => rfl
      | budgeted cls f => rfl
      | _ => rw [hop] at hd; exact absurd trivial hd
    rw [this]
    exact hraw

lemma foldl_derive_length (C : IntDag ι) (ivs : List Iv) :
    (C.foldl (fun ivs g => ivs ++ [g.derive ivs]) ivs).length = ivs.length + C.length := by
  induction C generalizing ivs with
  | nil => simp
  | cons g C ih => simp [List.foldl_cons, ih]; omega

lemma intervals_take_length (C : IntDag ι) (n : ℕ) (h : n ≤ C.length) : (intervals (C.take n)).length = n := by
  simp only [intervals, foldl_derive_length, List.length_nil, List.length_take]; omega

lemma intervals_take_succ (C : IntDag ι) (i : ℕ) (h : i < C.length) :
    intervals (C.take (i + 1)) = intervals (C.take i) ++ [C[i].derive (intervals (C.take i))] := by
  rw [List.take_add_one, List.getElem?_eq_getElem h, Option.toList_some]
  simp only [intervals, List.foldl_append, List.foldl_cons, List.foldl_nil]

/-- Every value of a proven circuit lies in its derived interval (and past the gates, `0` in `(0, 0)`), on an input on
which each gate that isn't derived holds its declared interval whenever the gates before it hold their derived ones. -/
lemma window_inv' (C : IntDag ι) (x : ι → ℤ) (hP : Proven C)
    (hD : ∀ i (h : i < C.length),
      (∀ j, Holds ((intervals (C.take i)).getD j (0, 0)) ((values (C.take i) x).getD j 0)) → ¬ C[i].op.IsDerived →
        C[i].lo ≤ C[i].raw x (values (C.take i) x) ∧ C[i].raw x (values (C.take i) x) ≤ C[i].hi) :
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
      exact (gate_window C[j] x _ _ (ih (by omega)) h1 h2 (hD j hn' (ih (by omega)))).2
    · rw [List.getD_append_right _ _ _ _ (by rw [hlenI]; omega), List.getD_append_right _ _ _ _ (by rw [hlen]; omega),
        hlenI, hlen]
      obtain ⟨k, hk⟩ : ∃ k, j - n = k + 1 := ⟨j - n - 1, by omega⟩
      rw [hk]
      simp

/-- Every value of a proven circuit, on an input on which its declared intervals hold, lies in its derived interval
(and past the gates, `0` in `(0, 0)`). -/
lemma window_inv (C : IntDag ι) (x : ι → ℤ) (hP : Proven C) (hD : DeclaredHold C x) :
    ∀ n, n ≤ C.length → ∀ j, Holds ((intervals (C.take n)).getD j (0, 0)) ((values (C.take n) x).getD j 0) :=
  window_inv' C x hP fun i h _ => hD i h

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
      unfold IntGate.raw IntGate.operands
      rw [hargs]
      rcases hop with hop | ⟨a, b, k, s, s', h1, h2⟩
      · rw [hop]
      · rw [h1, h2]; rfl
    have hval : C'[n].value x (values (C.take n) x) = C[n].value x (values (C.take n) x) := by
      unfold IntGate.value
      rw [hraw]
      rcases hop with hop | ⟨a, b, k, s, s', h1, h2⟩
      · rw [hop]
      · have hok := hw n hn'
        have hb := hok.2.2.2 (by rw [h1]; trivial)
        have hz1 : s32Min ≤ C[n].raw x (values (C.take n) x) := le_trans hb.1 hok.1
        have hz2 : C[n].raw x (values (C.take n) x) ≤ s32Max := le_trans hok.2.1 hb.2
        have hout : ∀ t, (IntOp.mma a b k t : IntOp ι).out (C[n].raw x (values (C.take n) x)) =
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
      rw [Fintype.card_piFinset, volume, ← Fin.prod_univ_fun_getElem C IntGate.charge]
      apply Finset.prod_congr rfl
      intro i _
      unfold IntGate.charge IntGate.size
      split_ifs <;> simp [Int.card_Icc]

/-! ## The fooling bound's information cap -/

theorem FoolingCap : Accounting.Guarantees.Gates.FoolingCap := by
  intro ι κ C F x j out hsep
  classical
  let f : κ → (Fin out.length → ZMod (2 ^ j)) := fun s i => (val C (x s) out[i] : ZMod (2 ^ j))
  have hinj : Set.InjOn f F := by
    intro s hs s' hs' hf
    by_contra hne
    obtain ⟨o, ho, hno⟩ := hsep s hs s' hs' hne
    obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.mp ho
    apply hno
    have h := congrFun hf ⟨i, hi⟩
    have h' := (ZMod.intCast_eq_intCast_iff (a := val C (x s) out[i]) (b := val C (x s') out[i]) (c := 2 ^ j)).mp h
    exact_mod_cast h'
  calc F.card ≤ (Finset.univ : Finset (Fin out.length → ZMod (2 ^ j))).card :=
        Finset.card_le_card_of_injOn f (fun _ _ => Finset.mem_coe.mpr (Finset.mem_univ _)) hinj
    _ = 2 ^ (j * out.length) := by simp [Finset.card_univ, ZMod.card, ← pow_mul]

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

/-! ## The nibble family -/

/-- `a_t`'s base-256 digits: byte 0 is `8`, byte `i + 1` is `16·t_i + 8`, and none above byte `s`. -/
def nibDigit {s : ℕ} (t : Fin s → Fin 16) (m : ℕ) : ℕ :=
  if m = 0 then 8 else if h : m - 1 < s then 16 * (t ⟨m - 1, h⟩ : ℕ) + 8 else 0

lemma nibDigit_le {s : ℕ} (t : Fin s → Fin 16) (m : ℕ) : nibDigit t m ≤ 248 := by
  unfold nibDigit
  split_ifs with h0 h1
  · norm_num
  · have := (t ⟨m - 1, h1⟩).isLt; omega
  · norm_num

lemma nibbles_eq (s : ℕ) (t : Fin s → Fin 16) :
    nibbles s t = ∑ m ∈ Finset.range (s + 1), nibDigit t m * 256 ^ m := by
  rw [Finset.sum_range_succ', Finset.sum_range, nibbles, add_comm]
  congr 1
  apply Finset.sum_congr rfl
  intro i _
  simp [nibDigit, i.isLt]

/-- Every byte of `a_t` is `8 mod 16` up to byte `s`, and `0` above it. -/
lemma byte_nibbles (s : ℕ) (t : Fin s → Fin 16) (l : ℕ) :
    byte (nibbles s t) l % 16 = if l ≤ s then 8 else 0 := by
  rw [nibbles_eq, byte_sum (s + 1) (nibDigit t) (fun m => by have := nibDigit_le t m; omega)]
  unfold nibDigit
  split_ifs <;> omega

lemma nibbles_lt (s : ℕ) (t : Fin s → Fin 16) : nibbles s t < 249 * 256 ^ s := by
  have hg := geom_sum_mul_add 255 (s + 1)
  norm_num at hg
  have hle : nibbles s t ≤ 248 * ∑ m ∈ Finset.range (s + 1), 256 ^ m := by
    rw [nibbles_eq, Finset.mul_sum]
    exact Finset.sum_le_sum fun m _ => Nat.mul_le_mul_right _ (nibDigit_le t m)
  rw [pow_succ] at hg
  omega

lemma nibbles_mod (p s : ℕ) (hp : (249 * 256 ^ s) ^ 2 ≤ p) (t : Fin s → Fin 16) :
    nibbles s t ^ 2 % p = nibbles s t ^ 2 :=
  Nat.mod_eq_of_lt (lt_of_lt_of_le (Nat.pow_lt_pow_left (nibbles_lt s t) (by norm_num)) hp)

/-- Two numbers `256^k·D` apart, `D ≢ 0 mod 16`, differ mod 16 in byte `k`. -/
lemma byte_mod16_ne (N N' k : ℕ) (D : ℤ) (h : (N : ℤ) - N' = 256 ^ k * D) (hD : ¬ (16 : ℤ) ∣ D) :
    byte N k % 16 ≠ byte N' k % 16 := by
  have hP : 0 < 256 ^ k := by positivity
  have hbyte : ∀ q r, r < 256 ^ k → byte (256 ^ k * q + r) k % 16 = q % 16 := by
    intro q r hr
    unfold byte
    rw [show (256 ^ k * q + r) / 256 ^ k = q by
      rw [add_comm, Nat.add_mul_div_left _ _ hP, Nat.div_eq_of_lt hr, zero_add],
      Nat.mod_mod_of_dvd _ (by norm_num : 16 ∣ 256)]
  obtain ⟨q, r, hr, rfl⟩ : ∃ q r, r < 256 ^ k ∧ N = 256 ^ k * q + r :=
    ⟨N / 256 ^ k, N % 256 ^ k, Nat.mod_lt _ hP, (Nat.div_add_mod N _).symm⟩
  obtain ⟨q', r', hr', rfl⟩ : ∃ q r, r < 256 ^ k ∧ N' = 256 ^ k * q + r :=
    ⟨N' / 256 ^ k, N' % 256 ^ k, Nat.mod_lt _ hP, (Nat.div_add_mod N' _).symm⟩
  rw [hbyte q r hr, hbyte q' r' hr']
  push_cast at h
  have hrz : (r : ℤ) < 256 ^ k := by exact_mod_cast hr
  have hrz' : (r' : ℤ) < 256 ^ k := by exact_mod_cast hr'
  have hdvd : (256 : ℤ) ^ k ∣ (r : ℤ) - r' := ⟨D - q + q', by linear_combination h⟩
  have hrr : (r : ℤ) - r' = 0 := by
    apply Int.eq_zero_of_abs_lt_dvd hdvd
    rw [abs_lt]
    constructor <;> linarith [(Nat.cast_nonneg r : (0 : ℤ) ≤ r), (Nat.cast_nonneg r' : (0 : ℤ) ≤ r')]
  have hq : (256 : ℤ) ^ k * ((q : ℤ) - q' - D) = 0 := by linear_combination h - hrr
  rcases mul_eq_zero.mp hq with h0 | h0
  · exact absurd h0 (by positivity)
  · intro heq
    apply hD
    omega

/-- `Σ_{i<s} e_i·256^(i+1)` with `e` zero below `m < s` is `256^(m+1)·(e_m + 256·γ)`. -/
lemma sum_split (e : ℕ → ℤ) (m : ℕ) (h0 : ∀ i < m, e i = 0) (s : ℕ) (hs : m + 1 ≤ s) :
    ∃ γ : ℤ, ∑ i ∈ Finset.range s, e i * 256 ^ (i + 1) = 256 ^ (m + 1) * (e m + 256 * γ) := by
  induction s, hs using Nat.le_induction with
  | base =>
    refine ⟨0, ?_⟩
    rw [Finset.sum_range_succ, Finset.sum_eq_zero fun i hi => by rw [h0 i (Finset.mem_range.mp hi), zero_mul]]
    ring
  | succ s hs ih =>
    obtain ⟨γ, hγ⟩ := ih
    obtain ⟨d, rfl⟩ : ∃ d, s = m + 1 + d := ⟨s - m - 1, by omega⟩
    refine ⟨γ + e (m + 1 + d) * 256 ^ d, ?_⟩
    rw [Finset.sum_range_succ, hγ]
    ring

/-- The squares of two members of the nibble family first apart at nibble `m` differ mod 16 in byte `m + 2`:
`a_t² − a_t'² = (a_t − a_t')(a_t + a_t') = 16·256^(m+1)·(d + 256γ) · 16·(1 + 256β)`, with `0 < |d| < 16`. -/
lemma nibbles_sq_ne (s : ℕ) (t t' : Fin s → Fin 16) (hne : t ≠ t') :
    ∃ m < s, byte (nibbles s t ^ 2) (m + 2) % 16 ≠ byte (nibbles s t' ^ 2) (m + 2) % 16 := by
  classical
  have hex : ∃ m, ∃ h : m < s, t ⟨m, h⟩ ≠ t' ⟨m, h⟩ := by
    by_contra hc
    push Not at hc
    exact hne (funext fun i => hc i.1 i.2)
  obtain ⟨hm, hmne⟩ := Nat.find_spec hex
  set m := Nat.find hex
  have hlt : ∀ i (h : i < s), i < m → t ⟨i, h⟩ = t' ⟨i, h⟩ := fun i h hi => by
    have := Nat.find_min hex hi
    push Not at this
    exact this h
  let e : ℕ → ℤ := fun i => if h : i < s then ((t ⟨i, h⟩ : ℕ) : ℤ) - (t' ⟨i, h⟩ : ℕ) else 0
  have h0 : ∀ i < m, e i = 0 := fun i hi => by
    simp only [e]
    split_ifs with h
    · rw [hlt i h hi, sub_self]
    · rfl
  obtain ⟨γ, hγ⟩ := sum_split e m h0 s hm
  have hdiff : (nibbles s t : ℤ) - nibbles s t' = 16 * (256 ^ (m + 1) * (e m + 256 * γ)) := by
    rw [← hγ, Finset.sum_range, Finset.mul_sum]
    simp only [nibbles]
    push_cast
    rw [add_sub_add_left_eq_sub, ← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro i _
    simp only [e, i.isLt, ↓reduceDIte, Fin.eta]
    ring
  have hsum : (nibbles s t : ℤ) + nibbles s t' =
      16 * (1 + 256 * ∑ i : Fin s, (((t i : ℕ) : ℤ) + (t' i : ℕ) + 1) * 256 ^ i.1) := by
    simp only [nibbles]
    push_cast
    rw [mul_add, mul_one, Finset.mul_sum, Finset.mul_sum, add_add_add_comm, ← Finset.sum_add_distrib]
    norm_num only
    congr 1
    apply Finset.sum_congr rfl
    intro i _
    ring
  set β := ∑ i : Fin s, (((t i : ℕ) : ℤ) + (t' i : ℕ) + 1) * 256 ^ i.1
  refine ⟨m, hm, byte_mod16_ne _ _ _ ((e m + 256 * γ) * (1 + 256 * β)) ?_ ?_⟩
  · push_cast
    rw [show (nibbles s t : ℤ) ^ 2 - (nibbles s t' : ℤ) ^ 2 =
      ((nibbles s t : ℤ) - nibbles s t') * ((nibbles s t : ℤ) + nibbles s t') by ring, hdiff, hsum]
    ring
  · have hem : e m = ((t ⟨m, hm⟩ : ℕ) : ℤ) - (t' ⟨m, hm⟩ : ℕ) := by simp [e, hm]
    have h1 := (t ⟨m, hm⟩).isLt
    have h2 := (t' ⟨m, hm⟩).isLt
    have h3 : (t ⟨m, hm⟩ : ℕ) ≠ t' ⟨m, hm⟩ := fun h => hmne (Fin.ext h)
    rw [show (e m + 256 * γ) * (1 + 256 * β) = e m + 256 * (γ + e m * β + 256 * γ * β) by ring]
    generalize γ + e m * β + 256 * γ * β = K
    rw [hem]
    omega

/-- On the nibble family, the squares' bytes tell members apart mod 16. -/
lemma nibbles_out (C : IntDag ι) (p n s : ℕ) (G : Finset (Fin s → Fin 16)) (x : (Fin s → Fin 16) → ι → ℤ)
    (out : ℕ → ℕ) (hp : (249 * 256 ^ s) ^ 2 ≤ p) (hn : s + 2 ≤ n)
    (hout : ∀ t ∈ G, ∀ l < n, val C (x t) (out l) = byte (nibbles s t ^ 2 % p) l) :
    ∀ t ∈ G, ∀ t' ∈ G, t ≠ t' → ∃ o ∈ (List.range n).map out, ¬ val C (x t) o ≡ val C (x t') o [ZMOD 2 ^ 4] := by
  intro t ht t' ht' hne
  obtain ⟨m, hm, hb⟩ := nibbles_sq_ne s t t' hne
  refine ⟨out (m + 2), List.mem_map.mpr ⟨m + 2, List.mem_range.mpr (by omega), rfl⟩, ?_⟩
  rw [hout t ht _ (by omega), hout t' ht' _ (by omega), nibbles_mod p s hp, nibbles_mod p s hp]
  unfold Int.ModEq
  norm_num
  omega

/-- A byte of a member of the nibble family, mod 16, is the same on every member. -/
lemma nibbles_byte_congr (s : ℕ) (t t' : Fin s → Fin 16) (l : ℕ) :
    ((byte (nibbles s t) l : ℕ) : ℤ) ≡ (byte (nibbles s t') l : ℕ) [ZMOD 2 ^ 4] := by
  have h1 := byte_nibbles s t l
  have h2 := byte_nibbles s t' l
  unfold Int.ModEq
  norm_num
  split_ifs at h1 h2 <;> omega

theorem SquaringCarryNibbles : Accounting.Guarantees.Gates.SquaringCarryNibbles := by
  intro ι C p n s G x slot out hp hn hreg hout
  refine Fooling C G x 4 ((List.range n).map out) (fun t ht => (hout t ht).1) ?_
    (nibbles_out C p n s G x out hp hn fun t ht => (hout t ht).2)
  intro t ht t' ht' i hi
  rw [hreg t ht i hi, hreg t' ht' i hi]
  exact nibbles_byte_congr s t t' (slot i)

/-! ## The encode budget -/

lemma countAt_cons (g : IntGate ι) (C : IntDag ι) (s : ℕ → Prop) [DecidablePred s] (p : IntGate ι → ℕ) :
    countAt (g :: C) s p = (if s g.step then p g else 0) + countAt C s p := by
  unfold countAt
  by_cases hs : s g.step <;> simp [hs]

/-- Summing a gate count over the windows: every gate is in one of them. -/
lemma sum_countAt (C : IntDag ι) (win : ℕ → ℕ) (N : ℕ) (p : IntGate ι → ℕ) (h : ∀ g ∈ C, win g.step < N) :
    ∑ v ∈ Finset.range N, countAt C (fun t => win t = v) p = (C.map p).sum := by
  induction C with
  | nil => simp [countAt]
  | cons g C ih =>
    have hg := h g (List.mem_cons_self ..)
    have ih := ih fun g' hg' => h g' (List.mem_cons_of_mem _ hg')
    simp only [countAt_cons, Finset.sum_add_distrib, Finset.sum_ite_eq, Finset.mem_range, hg, ite_true, ih,
      List.map_cons, List.sum_cons]

/-- A gate of no budgeted class has no units. -/
lemma units_of_budget_none (C : IntDag ι) (i : ℕ) (g : IntGate ι) (hg : C[i]? = some g) (hb : g.op.budget = none) :
    units C i = 0 := by
  simp only [units, hg]
  split <;> simp_all [IntOp.budget]

/-- A gather's outcomes at its index, its column's `V` words and the fill outside it, fit its index bits. -/
lemma indexBits_outcomes (g : IntGate ι) (d : ℤ) (hg : g.op = .gather none d) : g.args.length ≤ 2 ^ g.indexBits := by
  unfold IntGate.indexBits
  rw [hg]
  exact Nat.le_pow_clog (by norm_num) _

/-- Summing the census's units over the windows and classes: each gate's units are in its window's census, under its
class. -/
lemma budgeted_eq (C : IntDag ι) (A : Audit) (h : ∀ g ∈ C, A.win g.step < A.windows) :
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

/-- Summing the data products over the windows: each gate's products are in its window's count. -/
lemma products_eq (C : IntDag ι) (A : Audit) (h : ∀ g ∈ C, A.win g.step < A.windows) :
    products C = ∑ v ∈ Finset.range A.windows, productsAt C fun t => A.win t = v := by
  simp only [products, productsAt]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun i hi => ?_
  have hi' : i < C.length := Finset.mem_range.mp hi
  have hg : C[i]? = some C[i] := List.getElem?_eq_getElem hi'
  have hw := h C[i] (List.getElem_mem hi')
  simp [hg, Finset.sum_ite_eq, hw]

/-- The encode bound at a price holds at any lower one. -/
lemma EncodeBound.mono {F : Field} {k B E E' εnum εden : ℕ} (h : EncodeBound F k B E' εnum εden) (hE : E ≤ E') :
    EncodeBound F k B E εnum εden := by
  intro ι C m x words out hR hci hnd hlt hcw
  have h' := h C m x words out hR hci hnd hlt hcw
  generalize words.length = q at h' ⊢
  generalize nonlinear C = c at h' ⊢
  generalize runBits C = r at h' ⊢
  generalize F.n - 1 = N at h' ⊢
  rcases Nat.lt_or_ge r (q * N) with hqr | hqr
  swap
  · calc q * E * N = q * N * E := by ring
      _ ≤ r * E := Nat.mul_le_mul_right E hqr
      _ ≤ c * N + r * E := Nat.le_add_left _ _
  · obtain ⟨t, ht⟩ : ∃ t, q * N = r + t := ⟨q * N - r, by omega⟩
    have h1 : q * E' * N = r * E' + t * E' := by rw [show q * E' * N = q * N * E' by ring, ht]; ring
    have h3 : t * E ≤ t * E' := Nat.mul_le_mul_left t hE
    calc q * E * N = r * E + t * E := by rw [show q * E * N = q * N * E by ring, ht]; ring
      _ ≤ c * N + r * E := by omega

/-- The encode bound at `E ≤ k · S · e`, from `SquaringFloor` at `e` and `ChainSum`, inside their floor. -/
theorem encodeBound_of {F : Field} {k B E e εnum εden : ℕ} (hS : SquaringFloor F e εnum εden)
    (hC : ChainSum F k B e εnum εden) (hE : E ≤ k * F.squarings * e) (hε : εden ≤ 2 ^ 64 * εnum) (hk : 0 < k)
    (hp : 2 ^ 66 ≤ F.p) (hp4 : F.p % 4 = 3) : EncodeBound F k B E εnum εden :=
  EncodeBound.mono (hC hε hk hp hp4 hS) hE

theorem EncodeBudget : Accounting.Guarantees.Gates.EncodeBudget := by
  intro F k B E e εnum εden ι W caps A m x words out hS hC hE hε hk hp hp4 hP hR hci hnd hlt hcw
  have h := encodeBound_of hS hC hE hε hk hp hp4 W m x words out hR hci hnd hlt hcw
  obtain ⟨hwin, hcap, -⟩ := hP
  have hb : (nonlinear W : ℚ) ≤ caps.total * (∑ v ∈ Finset.range A.windows, A.reads v : ℕ) := by
    rw [nonlinear, budgeted_eq W A hwin, products_eq W A hwin, ← Finset.sum_add_distrib]
    push_cast
    rw [Finset.mul_sum]
    apply Finset.sum_le_sum
    intro v hv
    obtain ⟨h1, h2, h3, h4, h5, -, h7⟩ := hcap v (Finset.mem_range.mp hv)
    simp only [Census.budgeted, Caps.total, h5]
    push_cast
    linarith
  have hr : runBits W ≤ 8 * ∑ v ∈ Finset.range A.windows, caps.runBytes v := by
    rw [runBits, ← sum_countAt W A.win A.windows _ hwin, Finset.mul_sum]
    exact Finset.sum_le_sum fun v hv => (hcap v (Finset.mem_range.mp hv)).2.2.2.2.2.1
  have h' : ((words.length * E * (F.n - 1) : ℕ) : ℚ) ≤ (nonlinear W : ℚ) * (F.n - 1 : ℕ) + (runBits W * E : ℕ) := by
    exact_mod_cast h
  have hr' : ((runBits W * E : ℕ) : ℚ) ≤ (8 * (∑ v ∈ Finset.range A.windows, caps.runBytes v) * E : ℕ) := by
    exact_mod_cast Nat.mul_le_mul_right E hr
  exact h'.trans (add_le_add (mul_le_mul_of_nonneg_right hb (Nat.cast_nonneg _)) hr')

/-- The encode bound at `E` prices `q` distinct words at `q · E` nonlinear units, for `n ≥ 2`. -/
theorem encodeCost_ge {F : Field} {k B E εnum εden : ℕ} (H : EncodeBound F k B E εnum εden) (hn : 2 ≤ F.n)
    (q : ℕ) : ((q * E : ℕ) : ℕ∞) ≤ encodeCost F k B εnum εden q := by
  unfold encodeCost
  refine le_iInf₂ fun c hc => ?_
  obtain ⟨ι, C, m, x, words, out, hlen, hnd, hlt, hR, hrb, hci, hcw, rfl⟩ := hc
  have h := H C m x words out hR hci hnd hlt hcw
  rw [hrb, zero_mul, add_zero, hlen] at h
  exact_mod_cast Nat.le_of_mul_le_mul_right h (by omega)

theorem EncodeCost : Accounting.Guarantees.Gates.EncodeCost := by
  intro F k B E e εnum εden q hS hC hE hε hk hp hp4 hn
  exact encodeCost_ge (encodeBound_of hS hC hE hε hk hp hp4) hn q

set_option exponentiation.threshold 1100 in
/-- A root's squarings at p1024: `⌊log₂((p + 1)/4)⌋ = 1021`. -/
theorem p1024_squarings : p1024.squarings = 1021 := by
  have h8 : (2 : ℕ) ^ 1024 = 2 ^ 1021 * 8 := by rw [show (8 : ℕ) = 2 ^ 3 by norm_num, ← pow_add]
  have hN : 1021 < (2 : ℕ) ^ 1021 := Nat.lt_two_pow_self
  unfold Field.squarings Field.p
  rw [Nat.log_eq_iff (by norm_num), show (2 : ℕ) ^ (1021 + 1) = 2 ^ 1021 * 2 from pow_succ 2 1021]
  simp only [p1024]
  rw [h8]
  generalize (2 : ℕ) ^ 1021 = N at hN ⊢
  omega

/-! ## The premises' floor

`SquaringFloor` and `ChainSum` keep the falsified `EncodeChainHardness`'s floor (`ε ≥ 2^−64`, `k ≥ 1`, `p ≥ 2^66`,
`p ≡ 3 mod 4`). `2^n − c` with `n ≥ 68` and `c ≡ 1 mod 4`, `c ≤ 2^66`, is inside it, p1024 among them. Outside it the
premises claim nothing, so the counterexamples outside it don't refute them: at `ε = 0`, at `k = 0` and at a field
narrower than a byte `ChainSum` holds, and the one-draw circuit behind the `ε = 0` counterexample is right on more than
`ε` of the draws at no `ε` and field the floor admits. -/

/-- A field `2^n − c` with `n ≥ 68`, `c ≡ 1 mod 4` and `c ≤ 2^66` is inside the premises' floor on `p`. -/
theorem floor_of_pow_sub (n c : ℕ) (hn : 68 ≤ n) (hc : c % 4 = 1) (hc' : c ≤ 2 ^ 66) :
    2 ^ 66 ≤ 2 ^ n - c ∧ (2 ^ n - c) % 4 = 3 := by
  obtain ⟨m, rfl⟩ : ∃ m, n = m + 2 := ⟨n - 2, by omega⟩
  have hM : (2 : ℕ) ^ 66 ≤ 2 ^ m := Nat.pow_le_pow_right (by norm_num) (by omega)
  have h4 : (2 : ℕ) ^ 2 = 4 := by norm_num
  rw [pow_add 2 m 2, h4]
  obtain ⟨N, hNe⟩ : ∃ N, (2 : ℕ) ^ m = N := ⟨_, rfl⟩
  obtain ⟨M, hMe⟩ : ∃ M, (2 : ℕ) ^ 66 = M := ⟨_, rfl⟩
  rw [hNe, hMe] at hM ⊢
  rw [hMe] at hc'
  omega

/-- p1024 is inside the premises' floor on `p`: `2^66 ≤ p` and `p ≡ 3 mod 4`. -/
theorem p1024_floor : 2 ^ 66 ≤ p1024.p ∧ p1024.p % 4 = 3 := by
  unfold Field.p
  exact floor_of_pow_sub p1024.n p1024.c (by show 68 ≤ 1024; norm_num) (by show 105 % 4 = 1; norm_num)
    (by show 105 ≤ 2 ^ 66; norm_num)

/-- At `ε = 0` it holds: the floor excludes `encodeChainHardness_false_eps0`. -/
example (F : Field) (k B e εden : ℕ) (hden : 0 < εden) : ChainSum F k B e 0 εden := by
  intro hε
  rw [Nat.mul_zero] at hε
  exact absurd hden (by omega)

/-- At a field narrower than a byte it holds: the floor excludes `encodeChainHardness_false_toy`. -/
example (k B e εnum εden : ℕ) : ChainSum ⟨2, 0⟩ k B e εnum εden := by
  intro _ _ hp
  exact absurd hp (by norm_num [Field.p])

/-- At `k = 0`, where the encoding is the registered payload word, it holds. -/
example (F : Field) (B e εnum εden : ℕ) : ChainSum F 0 B e εnum εden := by
  intro _ hk
  exact absurd hk (lt_irrefl 0)

/-- Inside the floor one draw is never more than `ε` of the draws, so the circuit behind
`encodeChainHardness_false_eps0`, right on one draw, computes no word at any `ε` and field the floor admits. -/
example (F : Field) (k εnum εden : ℕ) (d : Draw k) (hε : εden ≤ 2 ^ 64 * εnum) (hk : 0 < k)
    (hp : 2 ^ 66 ≤ F.p) : ¬ MoreThan (draws F k) {d} εnum εden := by
  intro h
  unfold MoreThan at h
  rw [Finset.card_singleton, Nat.mul_one] at h
  have hp2 : 2 ≤ F.p := le_trans (by norm_num) hp
  have hc : F.p ≤ (draws F k).card := by
    simp only [draws, Finset.card_product, Fintype.card_piFinset, Finset.card_range, Finset.prod_const,
      Finset.card_univ, Fintype.card_fin, Nat.card_Ico]
    have h1 : 1 ≤ 2 ^ F.n * (F.p - 1) := Nat.mul_pos (Nat.two_pow_pos _) (by omega)
    calc F.p ≤ F.p ^ k := Nat.le_self_pow hk.ne' _
      _ = F.p ^ k * 1 := (Nat.mul_one _).symm
      _ ≤ _ := Nat.mul_le_mul_left _ h1
  have h3 : 2 ^ 64 * εnum ≤ εnum * (draws F k).card :=
    calc 2 ^ 64 * εnum ≤ 2 ^ 66 * εnum := Nat.mul_le_mul_right _ (Nat.pow_le_pow_right (by norm_num) (by norm_num))
      _ ≤ (draws F k).card * εnum := Nat.mul_le_mul_right _ (hp.trans hc)
      _ = εnum * (draws F k).card := Nat.mul_comm _ _
  exact lt_irrefl _ (h.trans_le (hε.trans h3))

/-! ## The fooling bound in units -/

/-- Gate `i` depends on a run or coin input. -/
abbrev dynAt (C : IntDag ι) (i : ℕ) : Bool := (dynamic C).getD i false

/-- The most index bits a gather of index `j` and kind `k` in `C` reads. -/
def maxIdx (C : IntDag ι) (k : Bool) (j : ℕ) : ℕ :=
  ((C.filter fun h => h.op.gatherKind == some k && h.args.head? == some j).map IntGate.indexBits).foldl max 0

/-- `dynamic`'s step at a gate, from the flags of the gates before it. -/
def dynStep (ds : List Bool) (g : IntGate ι) : Bool :=
  match g.op with
  | .input b _ => b != .registered
  | _ => g.args.any fun j => ds.getD j false

/-- A run input or a budgeted gate but a gather: the gates whose values the units bound reads off. -/
def traced (g : IntGate ι) : Bool :=
  match g.op with
  | .input .run _ => true
  | .budgeted _ _ => true
  | _ => false

/-- A budgeted gate but a gather. -/
def isBudg (g : IntGate ι) : Bool :=
  match g.op with
  | .budgeted _ _ => true
  | _ => false

/-- The fresh index bits a gather of kind `k` at `i` pays when its index depends on a run or coin input. -/
def freshAt (C : IntDag ι) (k : Bool) (i : ℕ) (g : IntGate ι) : ℕ :=
  match g.op, g.args with
  | .gather low _, j :: _ => if low.isSome = k ∧ dynAt C j = true then g.indexBits - maxIdx (C.take i) k j else 0
  | _, _ => 0

lemma dynamic_eq (C : IntDag ι) : dynamic C = C.foldl (fun ds g => ds ++ [dynStep ds g]) [] := rfl

lemma scan_append {α : Type} (f : List α → IntGate ι → α) (L : IntDag ι) (vs : List α) :
    ∃ r, L.foldl (fun vs g => vs ++ [f vs g]) vs = vs ++ r := by
  induction L generalizing vs with
  | nil => exact ⟨[], by simp⟩
  | cons g L ih =>
    obtain ⟨r, hr⟩ := ih (vs ++ [f vs g])
    exact ⟨f vs g :: r, by simp [List.foldl_cons, hr]⟩

lemma scan_length {α : Type} (f : List α → IntGate ι → α) (L : IntDag ι) (vs : List α) :
    (L.foldl (fun vs g => vs ++ [f vs g]) vs).length = vs.length + L.length := by
  induction L generalizing vs with
  | nil => simp
  | cons g L ih => simp [List.foldl_cons, ih]; omega

lemma dynamic_length (C : IntDag ι) : (dynamic C).length = C.length := by
  simp [dynamic_eq, scan_length]

lemma getD_dynamic_take (C : IntDag ι) (n i : ℕ) (hn : n ≤ C.length) :
    (dynamic (C.take n)).getD i false = if i < n then dynAt C i else false := by
  have hlen : (dynamic (C.take n)).length = n := by rw [dynamic_length, List.length_take]; omega
  split_ifs with hi
  · obtain ⟨r, hr⟩ := scan_append dynStep (C.drop n) (dynamic (C.take n))
    have hC : dynamic C = dynamic (C.take n) ++ r := by
      conv_lhs => rw [← List.take_append_drop n C]
      rw [dynamic_eq, List.foldl_append, ← dynamic_eq, hr]
    rw [dynAt, hC, List.getD_append _ _ _ _ (by omega)]
  · exact List.getD_eq_default _ _ (by omega)

lemma lt_of_dynAt (C : IntDag ι) (j : ℕ) (h : dynAt C j = true) : j < C.length := by
  by_contra hj
  rw [dynAt, List.getD_eq_default _ _ (by rw [dynamic_length]; omega)] at h
  exact Bool.false_ne_true h

lemma dynAt_eq (C : IntDag ι) (n : ℕ) (h : n < C.length) : dynAt C n = dynStep (dynamic (C.take n)) C[n] := by
  have := getD_dynamic_take C (n + 1) n h
  rw [ite_eq_left (by omega)] at this
  rw [← this, List.take_add_one, List.getElem?_eq_getElem h, Option.toList_some, dynamic_eq, List.foldl_append,
    ← dynamic_eq]
  simp only [List.foldl_cons, List.foldl_nil]
  rw [List.getD_append_right _ _ _ _ (by rw [dynamic_length, List.length_take]; omega), dynamic_length,
    List.length_take, min_eq_left h.le, Nat.sub_self, List.getD_cons_zero]

/-- A gate but an input depends on a run or coin input when one of its operands does. -/
lemma dynAt_of_op (C : IntDag ι) (n : ℕ) (h : n < C.length) (hop : ∀ b i, C[n].op ≠ .input b i) :
    dynAt C n = C[n].args.any fun j => (dynamic (C.take n)).getD j false := by
  rw [dynAt_eq C n h]
  unfold dynStep
  cases hopn : C[n].op with
  | input b i => exact absurd hopn (hop b i)
  | _ => rfl

/-- A gate but an input that depends on no run or coin input reads only gates that don't either. -/
lemma args_static (C : IntDag ι) (n : ℕ) (h : n < C.length) (hop : ∀ b i, C[n].op ≠ .input b i)
    (hd : dynAt C n = false) : ∀ j ∈ C[n].args, j < n → dynAt C j = false := by
  intro j hj hjn
  rw [dynAt_of_op C n h hop, List.any_eq_false] at hd
  have := hd j hj
  rw [getD_dynamic_take C n j h.le, ite_eq_left hjn] at this
  simpa using this

/-- A gate but an input that depends on a run or coin input reads one that does. -/
lemma any_dyn (C : IntDag ι) (n : ℕ) (h : n < C.length) (hop : ∀ b i, C[n].op ≠ .input b i)
    (hd : dynAt C n = true) : (C[n].args.any fun j => (dynamic C).getD j false) = true := by
  rw [dynAt_of_op C n h hop, List.any_eq_true] at hd
  rw [List.any_eq_true]
  obtain ⟨j, hj, hjd⟩ := hd
  rw [getD_dynamic_take C n j h.le] at hjd
  split_ifs at hjd
  · exact ⟨j, hj, hjd⟩

/-- A gate but an input takes one value on operands that agree. -/
lemma value_eq_of_operands (g : IntGate ι) (x x' : ι → ℤ) (vs vs' : List ℤ) (hop : ∀ b i, g.op ≠ .input b i)
    (h : ∀ j ∈ g.args, vs.getD j 0 = vs'.getD j 0) : g.value x vs = g.value x' vs' := by
  have ho : g.operands vs = g.operands vs' := List.map_congr_left h
  unfold IntGate.value IntGate.raw
  rw [ho]
  cases hg : g.op with
  | input b i => exact absurd hg (hop b i)
  | _ => rfl

/-- A gather's word, from columns that agree mod `m`, at positions of one class among the column's words and the fill
`d` past it. -/
lemma getD_map_congr {m : ℤ} (col : List ℕ) (vs vs' : List ℤ) (hv : ∀ j, vs.getD j 0 ≡ vs'.getD j 0 [ZMOD m])
    (d : ℤ) (k k' : ℕ) (hk : min k col.length = min k' col.length) :
    (col.map fun c => vs.getD c 0).getD k d ≡ (col.map fun c => vs'.getD c 0).getD k' d [ZMOD m] := by
  rcases Nat.lt_or_ge k col.length with h | h
  · have hkk : k' = k := by omega
    subst hkk
    rw [List.getD_eq_getElem _ _ (by simpa using h), List.getD_eq_getElem _ _ (by simpa using h)]
    simp only [List.getElem_map]
    exact hv _
  · have hk' : col.length ≤ k' := by omega
    rw [List.getD_eq_default _ _ (by simpa using h), List.getD_eq_default _ _ (by simpa using hk')]

/-- A gather's index class: `min(v, 2^b − 1)`, and `2^b − 1` for a negative `v`, which reads the fill as an index
past the column does. -/
def idxCls (b : ℕ) (v : ℤ) : ℕ := if v < 0 then 2 ^ b - 1 else min v.toNat (2 ^ b - 1)

/-- In a column of `L ≤ 2^b − 1` words, a gather's position is fixed, up to every position past the column, by its
index's class. -/
lemma min_gatherAt (b L : ℕ) (hL : L ≤ 2 ^ b - 1) (z : ℤ) :
    min (IntOp.gatherAt none L z) L = min (idxCls b z) L := by
  simp only [IntOp.gatherAt, idxCls]
  split_ifs <;> omega

lemma le_foldl_max (l : List ℕ) : ∀ b : ℕ, b ≤ l.foldl max b ∧ ∀ a ∈ l, a ≤ l.foldl max b := by
  induction l with
  | nil => intro b; simp
  | cons y l ih =>
    intro b
    simp only [List.foldl_cons, List.mem_cons, forall_eq_or_imp]
    obtain ⟨h1, h2⟩ := ih (max b y)
    exact ⟨le_trans (le_max_left _ _) h1, le_trans (le_max_right _ _) h1, h2⟩

lemma indexBits_le_maxIdx (C : IntDag ι) (n : ℕ) (h : n < C.length) (low : Option ℕ) (d : ℤ) (j : ℕ) (col : List ℕ)
    (hop : C[n].op = .gather low d) (hargs : C[n].args = j :: col) : C[n].indexBits ≤ maxIdx C low.isSome j :=
  (le_foldl_max _ 0).2 _ (List.mem_map.mpr ⟨C[n], List.mem_filter.mpr ⟨List.getElem_mem h,
    by simp [hop, hargs, IntOp.gatherKind]⟩, rfl⟩)

/-- Two inputs on which `C` is in its window, that agree in every registered input and mod `m` in every coin input,
on which each run input and each budgeted gate but a gather that depends on a run or coin input takes one value, and
each gate that does takes one class `idxCls b v` at the most index bits `b` a gather of it reads at its index, and
one class `v mod 2^b'` at the most `b'` a table read of it reads: every gate's values agree mod `m`, and those of the
gates that depend on no run or coin input agree. -/
lemma values_congr_units {m : ℤ} (C : IntDag ι) (x x' : ι → ℤ) (hw : InWindow C x) (hw' : InWindow C x')
    (hreg : ∀ i, (∃ g ∈ C, g.op = .input .registered i) → x i = x' i)
    (hcoin : ∀ i, (∃ g ∈ C, g.op = .input .coin i) → x i ≡ x' i [ZMOD m])
    (htr : ∀ i (h : i < C.length), traced C[i] = true → dynAt C i = true → val C x i = val C x' i)
    (hidx : ∀ i, i < C.length → dynAt C i = true →
      idxCls (maxIdx C false i) (val C x i) = idxCls (maxIdx C false i) (val C x' i) ∧
        val C x i % 2 ^ maxIdx C true i = val C x' i % 2 ^ maxIdx C true i) :
    ∀ n, n ≤ C.length → ∀ o, (values (C.take n) x).getD o 0 ≡ (values (C.take n) x').getD o 0 [ZMOD m] ∧
      ((o < n → dynAt C o = false) → (values (C.take n) x).getD o 0 = (values (C.take n) x').getD o 0) := by
  intro n
  induction n with
  | zero => intro _ o; simp [values, Int.ModEq.refl]
  | succ n ih =>
    intro hn o
    have hn' : n < C.length := hn
    have hlen := values_take_length C x n (by omega)
    have hlen' := values_take_length C x' n (by omega)
    have hv : ∀ j, (values (C.take n) x).getD j 0 ≡ (values (C.take n) x').getD j 0 [ZMOD m] :=
      fun j => (ih (by omega) j).1
    have he : ∀ j, (j < n → dynAt C j = false) →
        (values (C.take n) x).getD j 0 = (values (C.take n) x').getD j 0 :=
      fun j => (ih (by omega) j).2
    rw [values_take_succ C x n hn', values_take_succ C x' n hn']
    rcases Nat.lt_trichotomy o n with ho | ho | ho
    · rw [List.getD_append _ _ _ _ (by rw [hlen]; exact ho), List.getD_append _ _ _ _ (by rw [hlen']; exact ho)]
      exact ⟨hv o, fun h => he o fun _ => h (by omega)⟩
    · subst ho
      rw [List.getD_append_right _ _ _ _ (by rw [hlen]), List.getD_append_right _ _ _ _ (by rw [hlen']), hlen,
        hlen', Nat.sub_self, List.getD_cons_zero, List.getD_cons_zero]
      have hargs : (∀ b i, C[o].op ≠ .input b i) → dynAt C o = false →
          ∀ j ∈ C[o].args, (values (C.take o) x).getD j 0 = (values (C.take o) x').getD j 0 :=
        fun hop hd j hj => he j fun hjo => args_static C o hn' hop hd j hj hjo
      have hval := val_eq C x o hn'
      have hval' := val_eq C x' o hn'
      have key : C[o].value x (values (C.take o) x) = C[o].value x' (values (C.take o) x') ∨
          (dynAt C o = true ∧
            C[o].value x (values (C.take o) x) ≡ C[o].value x' (values (C.take o) x') [ZMOD m]) := by
        have hdyn : ∀ b i, C[o].op = .input b i → dynAt C o = (b != .registered) := fun b i hop => by
          rw [dynAt_eq C o hn']; unfold dynStep; rw [hop]
        cases hop : C[o].op with
        | input b i =>
          cases b with
          | registered =>
            left
            simp only [IntGate.value, IntGate.raw, hop, IntOp.raw, IntOp.out]
            exact hreg i ⟨C[o], List.getElem_mem hn', hop⟩
          | coin =>
            right
            refine ⟨by rw [hdyn _ i hop]; rfl, ?_⟩
            simp only [IntGate.value, IntGate.raw, hop, IntOp.raw, IntOp.out]
            exact hcoin i ⟨C[o], List.getElem_mem hn', hop⟩
          | run =>
            left
            rw [← hval, ← hval']
            exact htr o hn' (by simp [traced, hop]) (by rw [hdyn _ i hop]; rfl)
        | const c => left; simp [IntGate.value, IntGate.raw, hop, IntOp.raw, IntOp.out]
        | affine c₀ cs =>
          have hni : ∀ b i, C[o].op ≠ .input b i := by intro b i; rw [hop]; simp
          by_cases hd : dynAt C o = true
          · exact .inr ⟨hd, value_congr _ x x' _ _ hv (by simp [hop, IntOp.charged])
              (fun b i _ h => absurd h (hni b i)) (hw o hn') (hw' o hn')⟩
          · exact .inl (value_eq_of_operands _ x x' _ _ hni (hargs hni (by simpa using hd)))
        | mma a b k sat =>
          have hni : ∀ b i, C[o].op ≠ .input b i := by intro b i; rw [hop]; simp
          by_cases hd : dynAt C o = true
          · exact .inr ⟨hd, value_congr _ x x' _ _ hv (by simp [hop, IntOp.charged])
              (fun b i _ h => absurd h (hni b i)) (hw o hn') (hw' o hn')⟩
          · exact .inl (value_eq_of_operands _ x x' _ _ hni (hargs hni (by simpa using hd)))
        | budgeted cls f =>
          have hni : ∀ b i, C[o].op ≠ .input b i := by intro b i; rw [hop]; simp
          by_cases hd : dynAt C o = true
          · left
            rw [← hval, ← hval']
            exact htr o hn' (by simp [traced, hop]) hd
          · exact .inl (value_eq_of_operands _ x x' _ _ hni (hargs hni (by simpa using hd)))
        | gather low d =>
          have hni : ∀ b i, C[o].op ≠ .input b i := by intro b i; rw [hop]; simp
          by_cases hd : dynAt C o = true
          · right
            refine ⟨hd, ?_⟩
            rw [value_eq_raw _ x _ (hw o hn'), value_eq_raw _ x' _ (hw' o hn')]
            unfold IntGate.raw IntGate.operands
            rw [hop]
            cases hga : C[o].args with
            | nil => simp [IntOp.raw, Int.ModEq.refl]
            | cons j col =>
              simp only [IntOp.raw, List.map_cons, List.headD_cons, List.tail_cons, List.length_map]
              have hbits := indexBits_le_maxIdx C o hn' low d j col hop hga
              have hjv : (j < C.length ∧ dynAt C j = true ∧ (values (C.take o) x).getD j 0 = val C x j ∧
                  (values (C.take o) x').getD j 0 = val C x' j) ∨
                  (values (C.take o) x).getD j 0 = (values (C.take o) x').getD j 0 := by
                by_cases hj : j < o ∧ dynAt C j = true
                · have hjL : j < C.length := by omega
                  exact .inl ⟨hjL, hj.2, by rw [getD_values_take C x j hjL o hj.1 hn'.le, val_eq C x j hjL],
                    by rw [getD_values_take C x' j hjL o hj.1 hn'.le, val_eq C x' j hjL]⟩
                · exact .inr (he j fun hjo => by simpa [hjo] using hj)
              cases low with
              | none =>
                have hK : col.length ≤ 2 ^ maxIdx C false j - 1 := by
                  have h1 := indexBits_outcomes C[o] d hop
                  have h2 := Nat.pow_le_pow_right (by norm_num : 0 < 2) hbits
                  rw [hga, List.length_cons] at h1
                  simp only [Option.isSome_none] at h2
                  omega
                refine getD_map_congr col _ _ hv d _ _ ?_
                rw [min_gatherAt _ _ hK, min_gatherAt _ _ hK]
                rcases hjv with ⟨hjL, hjd, h1, h2⟩ | h
                · rw [h1, h2, (hidx j hjL hjd).1]
                · rw [h]
              | some nb =>
                refine getD_map_congr col _ _ hv d _ _ ?_
                congr 1
                simp only [IntOp.gatherAt]
                rcases hjv with ⟨hjL, hjd, h1, h2⟩ | h
                · have hnb : nb ≤ maxIdx C true j := by
                    have : C[o].indexBits = nb := by unfold IntGate.indexBits; rw [hop]
                    simpa [this] using hbits
                  have hdvd : (2 : ℤ) ^ nb ∣ 2 ^ maxIdx C true j := pow_dvd_pow 2 hnb
                  rw [h1, h2, ← Int.emod_emod_of_dvd (val C x j) hdvd, ← Int.emod_emod_of_dvd (val C x' j) hdvd,
                    (hidx j hjL hjd).2]
                · rw [h]
          · exact .inl (value_eq_of_operands _ x x' _ _ hni (hargs hni (by simpa using hd)))
      rcases key with hk | ⟨hd, hk⟩
      · exact ⟨by unfold Int.ModEq; rw [hk], fun _ => hk⟩
      · exact ⟨hk, fun h => by rw [h (by omega)] at hd; exact absurd hd Bool.false_ne_true⟩
    · rw [List.getD_append_right _ _ _ _ (by rw [hlen]; omega), List.getD_append_right _ _ _ _ (by rw [hlen']; omega),
        hlen, hlen']
      obtain ⟨k, hk⟩ : ∃ k, o - n = k + 1 := ⟨o - n - 1, by omega⟩
      rw [hk]
      simp [Int.ModEq.refl]

/-- Each gate's units are at least its unit as a budgeted gate but a gather and its fresh index bits as a gather of
either kind. -/
lemma units_ge (C : IntDag ι) (i : ℕ) (h : i < C.length) :
    (if (isBudg C[i] && dynAt C i) = true then 1 else 0) + (freshAt C false i C[i] + freshAt C true i C[i]) ≤
      units C i := by
  have hg : C[i]? = some C[i] := List.getElem?_eq_getElem h
  have hd := any_dyn C i h
  simp only [units, hg]
  cases hop : C[i].op with
  | budgeted cls f =>
    simp only [isBudg, freshAt, hop, Bool.true_and, add_zero]
    have := hd (by intro b k; rw [hop]; simp)
    split_ifs <;> simp_all
  | gather low _ =>
    cases hargs : C[i].args with
    | nil => simp [isBudg, freshAt, hop, hargs]
    | cons j col =>
      simp only [isBudg, freshAt, hop, hargs, maxIdx, dynAt, Bool.false_and, Bool.false_eq_true, ↓reduceIte, zero_add]
      cases low <;> simp only [Option.isSome_none, Option.isSome_some, true_and, false_and, Bool.false_eq_true,
        Bool.true_eq_false, ↓reduceIte, zero_add, add_zero] <;> split_ifs <;> omega
  | _ => simp [isBudg, freshAt, hop]

lemma maxIdx_take_succ (C : IntDag ι) (n : ℕ) (h : n < C.length) (k : Bool) (j : ℕ) :
    maxIdx (C.take (n + 1)) k j = max (maxIdx (C.take n) k j)
      (if (C[n].op.gatherKind == some k && C[n].args.head? == some j) = true then C[n].indexBits else 0) := by
  unfold maxIdx
  rw [List.take_add_one, List.getElem?_eq_getElem h, Option.toList_some, List.filter_append, List.map_append,
    List.foldl_append]
  split_ifs with hp <;> simp [hp]

/-- The fresh index bits telescope: over the gathers of each index and kind, they sum to the most that index's
gathers of that kind read. -/
lemma sum_freshAt (C : IntDag ι) (k : Bool) : ∀ n, n ≤ C.length →
    ∑ i ∈ Finset.range n, (if h : i < C.length then freshAt C k i C[i] else 0) =
      ∑ j ∈ Finset.range C.length, if dynAt C j = true then maxIdx (C.take n) k j else 0 := by
  intro n
  induction n with
  | zero => intro _; simp [maxIdx]
  | succ n ih =>
    intro hn
    have hnL : n < C.length := hn
    rw [Finset.sum_range_succ, ih (by omega), dite_eq_left hnL]
    simp only [maxIdx_take_succ C n hnL]
    have hsplit : ∀ j, (if dynAt C j = true then max (maxIdx (C.take n) k j)
        (if (C[n].op.gatherKind == some k && C[n].args.head? == some j) = true then C[n].indexBits else 0) else 0) =
        (if dynAt C j = true then maxIdx (C.take n) k j else 0) +
        (if dynAt C j = true then
          (if (C[n].op.gatherKind == some k && C[n].args.head? == some j) = true then
            C[n].indexBits - maxIdx (C.take n) k j else 0) else 0) := by
      intro j
      split_ifs <;> omega
    simp only [hsplit, Finset.sum_add_distrib]
    congr 1
    unfold freshAt
    cases hop : C[n].op with
    | gather low _ =>
      cases hargs : C[n].args with
      | nil => simp [IntOp.gatherKind]
      | cons j0 col =>
        simp only [IntOp.gatherKind, List.head?_cons, Option.some_beq_some, beq_iff_eq, Bool.and_eq_true]
        by_cases hk : low.isSome = k
        · simp only [hk, true_and]
          rw [Finset.sum_congr rfl fun j _ => (show (if dynAt C j = true then
            (if j0 = j then C[n].indexBits - maxIdx (C.take n) k j else 0) else 0) =
            if j0 = j then (if dynAt C j = true then C[n].indexBits - maxIdx (C.take n) k j else 0) else 0 by
              split_ifs <;> rfl), Finset.sum_ite_eq]
          by_cases hd : dynAt C j0 = true
          · rw [ite_eq_left (Finset.mem_range.mpr (lt_of_dynAt C j0 hd))]
          · simp only [hd, Bool.false_eq_true, ↓reduceIte, ite_self]
        · simp [hk]
    | _ => simp [IntOp.gatherKind]

theorem UnitsFooling : Accounting.Guarantees.Gates.UnitsFooling := by
  intro ι κ C F x j w out hw1 hsize hwin hreg hcoin hdist
  classical
  let T : κ → Fin C.length → ℤ × ℕ × ℤ := fun s i =>
    (if (traced C[i.1] && dynAt C i.1) = true then val C (x s) i.1 else 0,
      if dynAt C i.1 = true then idxCls (maxIdx C false i.1) (val C (x s) i.1) else 0,
      if dynAt C i.1 = true then val C (x s) i.1 % 2 ^ maxIdx C true i.1 else 0)
  let A : Fin C.length → Finset ℤ := fun i =>
    if (traced C[i.1] && dynAt C i.1) = true then Finset.Icc C[i.1].lo C[i.1].hi else {0}
  let B : Fin C.length → Finset ℕ := fun i =>
    if dynAt C i.1 = true then Finset.range (2 ^ maxIdx C false i.1) else {0}
  let D : Fin C.length → Finset ℤ := fun i =>
    if dynAt C i.1 = true then Finset.Ico 0 (2 ^ maxIdx C true i.1) else {0}
  have hmaps : Set.MapsTo T F (Fintype.piFinset fun i => A i ×ˢ (B i ×ˢ D i)) := by
    intro s hs
    simp only [Finset.mem_coe] at hs
    simp only [Finset.mem_coe, Fintype.mem_piFinset, Finset.mem_product]
    intro i
    refine ⟨?_, ?_, ?_⟩
    · simp only [T, A]
      split_ifs
      · have hok := hwin s hs i.1 i.2
        rw [Finset.mem_Icc, val_eq C (x s) i.1 i.2, value_eq_raw _ _ _ hok]
        exact ⟨hok.1, hok.2.1⟩
      · simp
    · simp only [T, B]
      split_ifs
      · rw [Finset.mem_range]
        have : 0 < 2 ^ maxIdx C false i.1 := by positivity
        unfold idxCls
        generalize 2 ^ maxIdx C false i.1 = P at *
        split_ifs <;> omega
      · simp
    · simp only [T, D]
      split_ifs
      · rw [Finset.mem_Ico]
        exact ⟨Int.emod_nonneg _ (by positivity), Int.emod_lt_of_pos _ (by positivity)⟩
      · simp
  have hinj : Set.InjOn T F := by
    intro s hs s' hs' hT
    simp only [Finset.mem_coe] at hs hs'
    by_contra hne
    obtain ⟨o, -, ho⟩ := hdist s hs s' hs' hne
    apply ho
    have hcomp : ∀ i (h : i < C.length), T s ⟨i, h⟩ = T s' ⟨i, h⟩ := fun i h => congrFun hT ⟨i, h⟩
    have h := values_congr_units C (x s) (x s') (hwin s hs) (hwin s' hs')
      (fun i hi => hreg s hs s' hs' i hi) (fun i hi => hcoin s hs s' hs' i hi)
      (fun i h ht hd => by
        have hc : (traced C[i] && dynAt C i) = true := by rw [ht, hd]; rfl
        have := congrArg Prod.fst (hcomp i h)
        simp only [T] at this
        rwa [ite_eq_left hc, ite_eq_left hc] at this)
      (fun i h hd => by
        have h2 := congrArg (fun t => t.2.1) (hcomp i h)
        have h3 := congrArg (fun t => t.2.2) (hcomp i h)
        simp only [T] at h2 h3
        rw [ite_eq_left hd, ite_eq_left hd] at h2 h3
        exact ⟨h2, h3⟩)
      C.length le_rfl o
    rw [List.take_length] at h
    exact h.1
  have hA : ∀ i : Fin C.length,
      (A i).card ≤ 2 ^ (C[i.1].inputBits .run + if (isBudg C[i.1] && dynAt C i.1) = true then w else 0) := by
    intro i
    simp only [A]
    split_ifs with ht hb hb
    · rw [Int.card_Icc]
      change C[i.1].size ≤ _
      cases hop : C[i.1].op with
      | budgeted cls f =>
        simp only [IntGate.inputBits, hop, zero_add]
        exact hsize _ (List.getElem_mem i.2) cls f hop
      | _ => simp [isBudg, hop] at hb
    · rw [Int.card_Icc]
      change C[i.1].size ≤ _
      cases hop : C[i.1].op with
      | input b k =>
        cases b with
        | run => simp only [IntGate.inputBits, hop, ite_true, add_zero]; exact Nat.le_pow_clog (by norm_num) _
        | _ => simp [traced, hop] at ht
      | budgeted cls f => simp [isBudg, hop] at hb; simp_all
      | _ => simp [traced, hop] at ht
    · simp only [Finset.card_singleton]; exact Nat.one_le_two_pow
    · simp only [Finset.card_singleton]; exact Nat.one_le_two_pow
  have hB : ∀ i : Fin C.length, (B i).card = 2 ^ (if dynAt C i.1 = true then maxIdx C false i.1 else 0) := by
    intro i
    simp only [B]
    split_ifs <;> simp
  have hD : ∀ i : Fin C.length, (D i).card = 2 ^ (if dynAt C i.1 = true then maxIdx C true i.1 else 0) := by
    intro i
    simp only [D]
    split_ifs
    · rw [Int.card_Ico, sub_zero, ← Nat.cast_ofNat, ← Nat.cast_pow, Int.toNat_natCast]
    · simp
  have hunits : ∑ i : Fin C.length, (if (isBudg C[i.1] && dynAt C i.1) = true then 1 else 0) +
      (∑ i : Fin C.length, (if dynAt C i.1 = true then maxIdx C false i.1 else 0) +
        ∑ i : Fin C.length, (if dynAt C i.1 = true then maxIdx C true i.1 else 0)) ≤ budgeted C := by
    have hf0 := sum_freshAt C false C.length le_rfl
    have hf1 := sum_freshAt C true C.length le_rfl
    rw [List.take_length, ← Fin.sum_univ_eq_sum_range, ← Fin.sum_univ_eq_sum_range] at hf0 hf1
    simp only [Fin.is_lt, dite_true] at hf0 hf1
    rw [← hf0, ← hf1, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib, budgeted, ← Fin.sum_univ_eq_sum_range]
    exact Finset.sum_le_sum fun i _ => units_ge C i.1 i.2
  have hrun : ∑ i : Fin C.length, C[i.1].inputBits .run = runBits C := by
    rw [runBits, ← Fin.sum_univ_fun_getElem C (IntGate.inputBits .run)]
  calc F.card ≤ (Fintype.piFinset fun i => A i ×ˢ (B i ×ˢ D i)).card := Finset.card_le_card_of_injOn T hmaps hinj
    _ = (∏ i, (A i).card) * ((∏ i, (B i).card) * ∏ i, (D i).card) := by
      rw [Fintype.card_piFinset, ← Finset.prod_mul_distrib, ← Finset.prod_mul_distrib]
      simp [Finset.card_product]
    _ ≤ (∏ i : Fin C.length,
          2 ^ (C[i.1].inputBits .run + if (isBudg C[i.1] && dynAt C i.1) = true then w else 0)) *
        ((∏ i : Fin C.length, 2 ^ (if dynAt C i.1 = true then maxIdx C false i.1 else 0)) *
          ∏ i : Fin C.length, 2 ^ (if dynAt C i.1 = true then maxIdx C true i.1 else 0)) := by
      apply Nat.mul_le_mul (Finset.prod_le_prod fun i _ => hA i)
      rw [Finset.prod_congr rfl fun i _ => hB i, Finset.prod_congr rfl fun i _ => hD i]
    _ ≤ 2 ^ (runBits C + w * budgeted C) := by
      rw [Finset.prod_pow_eq_pow_sum, Finset.prod_pow_eq_pow_sum, Finset.prod_pow_eq_pow_sum, ← pow_add, ← pow_add]
      apply Nat.pow_le_pow_right (by norm_num)
      rw [Finset.sum_add_distrib, hrun]
      have hw : ∑ i : Fin C.length, (if (isBudg C[i.1] && dynAt C i.1) = true then w else 0) =
          w * ∑ i : Fin C.length, (if (isBudg C[i.1] && dynAt C i.1) = true then 1 else 0) := by
        rw [Finset.mul_sum]
        exact Finset.sum_congr rfl fun i _ => by split_ifs <;> simp
      rw [hw]
      generalize ∑ i : Fin C.length, (if (isBudg C[i.1] && dynAt C i.1) = true then 1 else 0) = NB at hunits ⊢
      generalize ∑ i : Fin C.length, (if dynAt C i.1 = true then maxIdx C false i.1 else 0) = M0 at hunits ⊢
      generalize ∑ i : Fin C.length, (if dynAt C i.1 = true then maxIdx C true i.1 else 0) = M1 at hunits ⊢
      have h1 : M0 + M1 ≤ w * (M0 + M1) := Nat.le_mul_of_pos_left _ hw1
      have h2 : w * (NB + (M0 + M1)) ≤ w * budgeted C := Nat.mul_le_mul_left w hunits
      rw [mul_add] at h2
      omega

theorem SquaringUnits : Accounting.Guarantees.Gates.SquaringUnits := by
  intro ι C p n s w G x slot out hp hn hw1 hsize hreg hcoin hout
  refine UnitsFooling C G x 4 w ((List.range n).map out) hw1 hsize (fun t ht => (hout t ht).1) hreg ?_
    (nibbles_out C p n s G x out hp hn fun t ht => (hout t ht).2)
  intro t ht t' ht' i hi
  rw [hcoin t ht i hi, hcoin t' ht' i hi]
  exact nibbles_byte_congr s t t' (slot i)

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

/-! ## Primitives' output intervals

Each rule's interval holds its primitive's value: `fmaxf` and `fminf` by the operands' signs (`fmaxf_bounds`,
`fminf_bounds`), the truncation by its monotonicity below `128.0` (`rzByteNat_mono`), the clamps and the dither by
arithmetic, and the LUT by a gather's value being a column word or its fill, 0 (`getD_hull0`). -/

/-- `fmaxf` with a non-negative operand is at least it and at most `+inf`, and with two is their larger. -/
lemma fmaxf_bounds {x y : ℤ} (hx : S32 x) (hy : S32 y) :
    (0 ≤ x → x ≤ f32Inf → x ≤ f32Fmaxf x y ∧ f32Fmaxf x y ≤ f32Inf) ∧
    (0 ≤ y → y ≤ f32Inf → y ≤ f32Fmaxf x y ∧ f32Fmaxf x y ≤ f32Inf) ∧
    (0 ≤ x → x ≤ f32Inf → 0 ≤ y → y ≤ f32Inf → f32Fmaxf x y = max x y) := by
  obtain ⟨hx0, hx1⟩ := hx
  obtain ⟨hy0, hy1⟩ := hy
  simp only [f32Inf, s32Min, s32Max] at *
  by_cases hx : x < 0 <;> by_cases hy : y < 0 <;>
    simp only [f32Fmaxf, f32MaxMin, f32NaN, f32Key, f32Mag, f32NanPtx, hx, hy, ite_true, ite_false,
      Bool.and_eq_true, decide_eq_true_eq] <;>
    split_ifs <;> omega

/-- `fminf` with a non-negative operand is at most it, and with two is their smaller. -/
lemma fminf_bounds {x y : ℤ} (hx : S32 x) (hy : S32 y) :
    (0 ≤ x → x ≤ f32Inf → s32Min ≤ f32Fminf x y ∧ f32Fminf x y ≤ x) ∧
    (0 ≤ y → y ≤ f32Inf → s32Min ≤ f32Fminf x y ∧ f32Fminf x y ≤ y) ∧
    (0 ≤ x → x ≤ f32Inf → 0 ≤ y → y ≤ f32Inf → f32Fminf x y = min x y) := by
  obtain ⟨hx0, hx1⟩ := hx
  obtain ⟨hy0, hy1⟩ := hy
  simp only [f32Inf, s32Min, s32Max] at *
  by_cases hx : x < 0 <;> by_cases hy : y < 0 <;>
    simp only [f32Fminf, f32MaxMin, f32NaN, f32Key, f32Mag, f32NanPtx, hx, hy, ite_true, ite_false,
      Bool.and_eq_true, decide_eq_true_eq, Bool.false_eq_true] <;>
    split_ifs <;> omega

theorem FmaxfRange : Accounting.Guarantees.Gates.FmaxfRange := by
  intro a b o h x y hx hy ha hb
  have hm := fmaxf_bounds hx hy
  rcases a with _ | ⟨al, ah⟩ <;> rcases b with _ | ⟨bl, bh⟩ <;>
    simp only [fmaxfRule, Iv.f32Pos, Bool.and_eq_true, decide_eq_true_eq, Rng.Holds, Iv.Mem] at h ha hb ⊢ <;>
    (try split_ifs at h) <;> cases h <;> dsimp only
  all_goals first
    | (rw [hm.2.2 (by omega) (by omega) (by omega) (by omega)]; omega)
    | (have := hm.1 (by omega) (by omega); constructor <;> omega)
    | (have := hm.2.1 (by omega) (by omega); constructor <;> omega)

theorem FminfRange : Accounting.Guarantees.Gates.FminfRange := by
  intro a b o h x y hx hy ha hb
  have hm := fminf_bounds hx hy
  rcases a with _ | ⟨al, ah⟩ <;> rcases b with _ | ⟨bl, bh⟩ <;>
    simp only [fminfRule, Iv.f32Pos, Bool.and_eq_true, decide_eq_true_eq, Rng.Holds, Iv.Mem] at h ha hb ⊢ <;>
    (try split_ifs at h) <;> cases h <;> dsimp only
  all_goals first
    | (rw [hm.2.2 (by omega) (by omega) (by omega) (by omega)]; omega)
    | (have := hm.1 (by omega) (by omega); constructor <;> omega)
    | (have := hm.2.1 (by omega) (by omega); constructor <;> omega)

lemma rzByteNat_mono {u v : ℕ} (h : u ≤ v) (hv : v < 0x43000000) : rzByteNat u ≤ rzByteNat v := by
  obtain ⟨eu, heu⟩ : ∃ e, u / 2 ^ 23 = e := ⟨_, rfl⟩
  obtain ⟨ev, hev⟩ : ∃ e, v / 2 ^ 23 = e := ⟨_, rfl⟩
  have hle : eu ≤ ev := heu ▸ hev ▸ Nat.div_le_div_right h
  have hev' : ev ≤ 133 := by omega
  have heu' : eu ≤ 133 := by omega
  simp only [rzByteNat, heu, hev]
  by_cases h0 : eu < 127
  · simp [h0]
  · interval_cases eu <;> interval_cases ev <;> simp <;> omega

theorem RzByteRange : Accounting.Guarantees.Gates.RzByteRange := by
  intro a o h x ha
  rcases a with _ | ⟨lo, hi⟩
  · cases h
  simp only [rzByteRule, Rng.Holds, Iv.Mem] at h ha ⊢
  split_ifs at h with h1 h2 h3 <;> cases h <;> simp only [f32RzByte] <;> split_ifs <;>
    first
    | omega
    | (have := rzByteNat_mono (u := x.toNat) (v := hi.toNat) (by omega) (by omega); constructor <;> omega)
    | (have := rzByteNat_mono (u := lo.toNat) (v := x.toNat) (by omega) (by omega)
       have := rzByteNat_mono (u := x.toNat) (v := hi.toNat) (by omega) (by omega); constructor <;> omega)

theorem QuantRange : Accounting.Guarantees.Gates.QuantRange := by
  intro lo w rnd v
  simp only [quantize, clampTo, Iv.Mem]
  omega

theorem DitherRange : Accounting.Guarantees.Gates.DitherRange := by
  intro lo w m v R
  have := R.isLt
  simp only [dither, clampTo, Iv.Mem]
  omega

lemma getD_hull0 (T : List ℤ) (i : ℕ) : (hull0 T).Mem (T.getD i 0) := by
  induction T generalizing i with
  | nil => simp [hull0, Iv.Mem]
  | cons t T ih =>
      have := ih (i - 1)
      cases i <;> simp_all [hull0, Iv.Mem]

theorem LutSoftmaxRange : Accounting.Guarantees.Gates.LutSoftmaxRange := by
  intro T d
  exact getD_hull0 T _

/-! ## `Bf16ToF32`, free-affine -/

theorem Bf16ToF32Affine : Accounting.Guarantees.Gates.Bf16ToF32Affine := by
  refine ⟨by decide, fun ⟨u, hu⟩ => ?_⟩
  have hmask : u &&& 0xFFFF = u := by
    rw [show (0xFFFF : ℕ) = 2 ^ 16 - 1 by norm_num, Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt hu]
  simp only [Verity.Prims.bf16ToF32, hmask, Nat.shiftLeft_eq, bf16ToF32Op, IntOp.raw, signed, List.zipWith_cons_cons,
    List.zipWith_nil_right, List.sum_cons, List.sum_nil]
  push_cast
  omega

/-! ## The rows the Python checks agree on

`f32Rows` are `scalar.F32Fmaxf` and `F32Fminf` on pairs of special encodings (s32 readings), `rzRows`
`f32_to_u32_rz_byte` on encodings across the exponents, and `ruleRows` and `rzRuleRows` property/v1's rules
(`stacking.RANGES`) on ranges about the specials. The kernel evaluates the Lean functions on each, and
`test_gate_ranges.py` recomputes each from the Python. -/

def f32Rows : List (ℤ × ℤ × ℤ × ℤ) := [
  (0, 0, 0, 0),
  (0, -2147483648, 0, -2147483648),
  (0, 1, 1, 0),
  (0, 8388607, 8388607, 0),
  (0, 1065353216, 1065353216, 0),
  (0, 1123942400, 1123942400, 0),
  (0, 2139095039, 2139095039, 0),
  (0, 2139095040, 2139095040, 0),
  (0, -8388608, 0, -8388608),
  (0, 2143289344, 0, 0),
  (0, -4194304, 0, 0),
  (0, 2139095041, 0, 0),
  (0, -1082130432, 0, -1082130432),
  (-2147483648, 0, 0, -2147483648),
  (-2147483648, -2147483648, -2147483648, -2147483648),
  (-2147483648, 1, 1, -2147483648),
  (-2147483648, 8388607, 8388607, -2147483648),
  (-2147483648, 1065353216, 1065353216, -2147483648),
  (-2147483648, 1123942400, 1123942400, -2147483648),
  (-2147483648, 2139095039, 2139095039, -2147483648),
  (-2147483648, 2139095040, 2139095040, -2147483648),
  (-2147483648, -8388608, -2147483648, -8388608),
  (-2147483648, 2143289344, -2147483648, -2147483648),
  (-2147483648, -4194304, -2147483648, -2147483648),
  (-2147483648, 2139095041, -2147483648, -2147483648),
  (-2147483648, -1082130432, -2147483648, -1082130432),
  (1, 0, 1, 0),
  (1, -2147483648, 1, -2147483648),
  (1, 1, 1, 1),
  (1, 8388607, 8388607, 1),
  (1, 1065353216, 1065353216, 1),
  (1, 1123942400, 1123942400, 1),
  (1, 2139095039, 2139095039, 1),
  (1, 2139095040, 2139095040, 1),
  (1, -8388608, 1, -8388608),
  (1, 2143289344, 1, 1),
  (1, -4194304, 1, 1),
  (1, 2139095041, 1, 1),
  (1, -1082130432, 1, -1082130432),
  (8388607, 0, 8388607, 0),
  (8388607, -2147483648, 8388607, -2147483648),
  (8388607, 1, 8388607, 1),
  (8388607, 8388607, 8388607, 8388607),
  (8388607, 1065353216, 1065353216, 8388607),
  (8388607, 1123942400, 1123942400, 8388607),
  (8388607, 2139095039, 2139095039, 8388607),
  (8388607, 2139095040, 2139095040, 8388607),
  (8388607, -8388608, 8388607, -8388608),
  (8388607, 2143289344, 8388607, 8388607),
  (8388607, -4194304, 8388607, 8388607),
  (8388607, 2139095041, 8388607, 8388607),
  (8388607, -1082130432, 8388607, -1082130432),
  (1065353216, 0, 1065353216, 0),
  (1065353216, -2147483648, 1065353216, -2147483648),
  (1065353216, 1, 1065353216, 1),
  (1065353216, 8388607, 1065353216, 8388607),
  (1065353216, 1065353216, 1065353216, 1065353216),
  (1065353216, 1123942400, 1123942400, 1065353216),
  (1065353216, 2139095039, 2139095039, 1065353216),
  (1065353216, 2139095040, 2139095040, 1065353216),
  (1065353216, -8388608, 1065353216, -8388608),
  (1065353216, 2143289344, 1065353216, 1065353216),
  (1065353216, -4194304, 1065353216, 1065353216),
  (1065353216, 2139095041, 1065353216, 1065353216),
  (1065353216, -1082130432, 1065353216, -1082130432),
  (1123942400, 0, 1123942400, 0),
  (1123942400, -2147483648, 1123942400, -2147483648),
  (1123942400, 1, 1123942400, 1),
  (1123942400, 8388607, 1123942400, 8388607),
  (1123942400, 1065353216, 1123942400, 1065353216),
  (1123942400, 1123942400, 1123942400, 1123942400),
  (1123942400, 2139095039, 2139095039, 1123942400),
  (1123942400, 2139095040, 2139095040, 1123942400),
  (1123942400, -8388608, 1123942400, -8388608),
  (1123942400, 2143289344, 1123942400, 1123942400),
  (1123942400, -4194304, 1123942400, 1123942400),
  (1123942400, 2139095041, 1123942400, 1123942400),
  (1123942400, -1082130432, 1123942400, -1082130432),
  (2139095039, 0, 2139095039, 0),
  (2139095039, -2147483648, 2139095039, -2147483648),
  (2139095039, 1, 2139095039, 1),
  (2139095039, 8388607, 2139095039, 8388607),
  (2139095039, 1065353216, 2139095039, 1065353216),
  (2139095039, 1123942400, 2139095039, 1123942400),
  (2139095039, 2139095039, 2139095039, 2139095039),
  (2139095039, 2139095040, 2139095040, 2139095039),
  (2139095039, -8388608, 2139095039, -8388608),
  (2139095039, 2143289344, 2139095039, 2139095039),
  (2139095039, -4194304, 2139095039, 2139095039),
  (2139095039, 2139095041, 2139095039, 2139095039),
  (2139095039, -1082130432, 2139095039, -1082130432),
  (2139095040, 0, 2139095040, 0),
  (2139095040, -2147483648, 2139095040, -2147483648),
  (2139095040, 1, 2139095040, 1),
  (2139095040, 8388607, 2139095040, 8388607),
  (2139095040, 1065353216, 2139095040, 1065353216),
  (2139095040, 1123942400, 2139095040, 1123942400),
  (2139095040, 2139095039, 2139095040, 2139095039),
  (2139095040, 2139095040, 2139095040, 2139095040),
  (2139095040, -8388608, 2139095040, -8388608),
  (2139095040, 2143289344, 2139095040, 2139095040),
  (2139095040, -4194304, 2139095040, 2139095040),
  (2139095040, 2139095041, 2139095040, 2139095040),
  (2139095040, -1082130432, 2139095040, -1082130432),
  (-8388608, 0, 0, -8388608),
  (-8388608, -2147483648, -2147483648, -8388608),
  (-8388608, 1, 1, -8388608),
  (-8388608, 8388607, 8388607, -8388608),
  (-8388608, 1065353216, 1065353216, -8388608),
  (-8388608, 1123942400, 1123942400, -8388608),
  (-8388608, 2139095039, 2139095039, -8388608),
  (-8388608, 2139095040, 2139095040, -8388608),
  (-8388608, -8388608, -8388608, -8388608),
  (-8388608, 2143289344, -8388608, -8388608),
  (-8388608, -4194304, -8388608, -8388608),
  (-8388608, 2139095041, -8388608, -8388608),
  (-8388608, -1082130432, -1082130432, -8388608),
  (2143289344, 0, 0, 0),
  (2143289344, -2147483648, -2147483648, -2147483648),
  (2143289344, 1, 1, 1),
  (2143289344, 8388607, 8388607, 8388607),
  (2143289344, 1065353216, 1065353216, 1065353216),
  (2143289344, 1123942400, 1123942400, 1123942400),
  (2143289344, 2139095039, 2139095039, 2139095039),
  (2143289344, 2139095040, 2139095040, 2139095040),
  (2143289344, -8388608, -8388608, -8388608),
  (2143289344, 2143289344, 2147483647, 2147483647),
  (2143289344, -4194304, 2147483647, 2147483647),
  (2143289344, 2139095041, 2147483647, 2147483647),
  (2143289344, -1082130432, -1082130432, -1082130432),
  (-4194304, 0, 0, 0),
  (-4194304, -2147483648, -2147483648, -2147483648),
  (-4194304, 1, 1, 1),
  (-4194304, 8388607, 8388607, 8388607),
  (-4194304, 1065353216, 1065353216, 1065353216),
  (-4194304, 1123942400, 1123942400, 1123942400),
  (-4194304, 2139095039, 2139095039, 2139095039),
  (-4194304, 2139095040, 2139095040, 2139095040),
  (-4194304, -8388608, -8388608, -8388608),
  (-4194304, 2143289344, 2147483647, 2147483647),
  (-4194304, -4194304, 2147483647, 2147483647),
  (-4194304, 2139095041, 2147483647, 2147483647),
  (-4194304, -1082130432, -1082130432, -1082130432),
  (2139095041, 0, 0, 0),
  (2139095041, -2147483648, -2147483648, -2147483648),
  (2139095041, 1, 1, 1),
  (2139095041, 8388607, 8388607, 8388607),
  (2139095041, 1065353216, 1065353216, 1065353216),
  (2139095041, 1123942400, 1123942400, 1123942400),
  (2139095041, 2139095039, 2139095039, 2139095039),
  (2139095041, 2139095040, 2139095040, 2139095040),
  (2139095041, -8388608, -8388608, -8388608),
  (2139095041, 2143289344, 2147483647, 2147483647),
  (2139095041, -4194304, 2147483647, 2147483647),
  (2139095041, 2139095041, 2147483647, 2147483647),
  (2139095041, -1082130432, -1082130432, -1082130432),
  (-1082130432, 0, 0, -1082130432),
  (-1082130432, -2147483648, -2147483648, -1082130432),
  (-1082130432, 1, 1, -1082130432),
  (-1082130432, 8388607, 8388607, -1082130432),
  (-1082130432, 1065353216, 1065353216, -1082130432),
  (-1082130432, 1123942400, 1123942400, -1082130432),
  (-1082130432, 2139095039, 2139095039, -1082130432),
  (-1082130432, 2139095040, 2139095040, -1082130432),
  (-1082130432, -8388608, -1082130432, -8388608),
  (-1082130432, 2143289344, -1082130432, -1082130432),
  (-1082130432, -4194304, -1082130432, -1082130432),
  (-1082130432, 2139095041, -1082130432, -1082130432),
  (-1082130432, -1082130432, -1082130432, -1082130432)]

theorem f32MaxMin_f32Rows :
    f32Rows.all (fun r => f32Fmaxf r.1 r.2.1 == r.2.2.1 && f32Fminf r.1 r.2.1 == r.2.2.2) := by
  decide +kernel

def rzRows : List (ℤ × ℤ) := [
  (-2147483648, 0),
  (-1082130432, 0),
  (-1023541248, 0),
  (-8388608, 0),
  (-4194304, 0),
  (0, 0),
  (1, 0),
  (4194304, 0),
  (8388607, 0),
  (1056964608, 0),
  (1056964609, 0),
  (1061158912, 0),
  (1065353215, 0),
  (1065353216, 1),
  (1065353217, 1),
  (1069547520, 1),
  (1073741823, 1),
  (1073741824, 2),
  (1073741825, 2),
  (1077936128, 3),
  (1082130431, 3),
  (1115684864, 64),
  (1115684865, 64),
  (1119879168, 96),
  (1123942400, 127),
  (1124073471, 127),
  (1124073472, 128),
  (1124073473, 128),
  (1128267776, 192),
  (1132462079, 255),
  (1249902592, 0),
  (1249902593, 0),
  (1254096896, 0),
  (1258291199, 255),
  (1258291200, 0),
  (1258291201, 1),
  (1262485504, 0),
  (1266679807, 255),
  (1266679808, 0),
  (1266679809, 2),
  (1270874112, 0),
  (1275068415, 254),
  (1325400064, 0),
  (1325400065, 0),
  (1329594368, 0),
  (1333788671, 0),
  (1333788672, 255),
  (1333788673, 255),
  (1337982976, 255),
  (1342177279, 255),
  (1342177280, 255),
  (1342177281, 255),
  (1346371584, 255),
  (1350565887, 255),
  (2130706432, 255),
  (2130706433, 255),
  (2134900736, 255),
  (2139095039, 255),
  (2139095040, 255),
  (2139095041, 0),
  (2143289344, 0)]

theorem f32RzByte_rzRows : rzRows.all (fun r => f32RzByte r.1 == r.2) := by
  decide +kernel

def ruleRows : List (Rng × Rng × Rng × Rng) := [
  (none, none, none, none),
  (none, some (0, 0), some (0, 2139095040), some (-2147483648, 0)),
  (none, some (0, 1123942400), some (0, 2139095040), some (-2147483648, 1123942400)),
  (none, some (1065353216, 2139095040), some (1065353216, 2139095040), some (-2147483648, 2139095040)),
  (none, some (2139095040, 2139095040), some (2139095040, 2139095040), some (-2147483648, 2139095040)),
  (none, some (-2147483648, -1), none, none),
  (none, some (2139095041, 2147483647), none, none),
  (none, some (-5, 5), none, none),
  (none, some (-2147483648, 1123942400), none, none),
  (some (0, 0), none, some (0, 2139095040), some (-2147483648, 0)),
  (some (0, 0), some (0, 0), some (0, 0), some (0, 0)),
  (some (0, 0), some (0, 1123942400), some (0, 1123942400), some (0, 0)),
  (some (0, 0), some (1065353216, 2139095040), some (1065353216, 2139095040), some (0, 0)),
  (some (0, 0), some (2139095040, 2139095040), some (2139095040, 2139095040), some (0, 0)),
  (some (0, 0), some (-2147483648, -1), some (0, 2139095040), some (-2147483648, 0)),
  (some (0, 0), some (2139095041, 2147483647), some (0, 2139095040), some (-2147483648, 0)),
  (some (0, 0), some (-5, 5), some (0, 2139095040), some (-2147483648, 0)),
  (some (0, 0), some (-2147483648, 1123942400), some (0, 2139095040), some (-2147483648, 0)),
  (some (0, 1123942400), none, some (0, 2139095040), some (-2147483648, 1123942400)),
  (some (0, 1123942400), some (0, 0), some (0, 1123942400), some (0, 0)),
  (some (0, 1123942400), some (0, 1123942400), some (0, 1123942400), some (0, 1123942400)),
  (some (0, 1123942400), some (1065353216, 2139095040), some (1065353216, 2139095040), some (0, 1123942400)),
  (some (0, 1123942400), some (2139095040, 2139095040), some (2139095040, 2139095040), some (0, 1123942400)),
  (some (0, 1123942400), some (-2147483648, -1), some (0, 2139095040), some (-2147483648, 1123942400)),
  (some (0, 1123942400), some (2139095041, 2147483647), some (0, 2139095040), some (-2147483648, 1123942400)),
  (some (0, 1123942400), some (-5, 5), some (0, 2139095040), some (-2147483648, 1123942400)),
  (some (0, 1123942400), some (-2147483648, 1123942400), some (0, 2139095040), some (-2147483648, 1123942400)),
  (some (1065353216, 2139095040), none, some (1065353216, 2139095040), some (-2147483648, 2139095040)),
  (some (1065353216, 2139095040), some (0, 0), some (1065353216, 2139095040), some (0, 0)),
  (some (1065353216, 2139095040), some (0, 1123942400), some (1065353216, 2139095040), some (0, 1123942400)),
  (some (1065353216, 2139095040), some (1065353216, 2139095040), some (1065353216, 2139095040), some (1065353216, 2139095040)),
  (some (1065353216, 2139095040), some (2139095040, 2139095040), some (2139095040, 2139095040), some (1065353216, 2139095040)),
  (some (1065353216, 2139095040), some (-2147483648, -1), some (1065353216, 2139095040), some (-2147483648, 2139095040)),
  (some (1065353216, 2139095040), some (2139095041, 2147483647), some (1065353216, 2139095040), some (-2147483648, 2139095040)),
  (some (1065353216, 2139095040), some (-5, 5), some (1065353216, 2139095040), some (-2147483648, 2139095040)),
  (some (1065353216, 2139095040), some (-2147483648, 1123942400), some (1065353216, 2139095040), some (-2147483648, 2139095040)),
  (some (2139095040, 2139095040), none, some (2139095040, 2139095040), some (-2147483648, 2139095040)),
  (some (2139095040, 2139095040), some (0, 0), some (2139095040, 2139095040), some (0, 0)),
  (some (2139095040, 2139095040), some (0, 1123942400), some (2139095040, 2139095040), some (0, 1123942400)),
  (some (2139095040, 2139095040), some (1065353216, 2139095040), some (2139095040, 2139095040), some (1065353216, 2139095040)),
  (some (2139095040, 2139095040), some (2139095040, 2139095040), some (2139095040, 2139095040), some (2139095040, 2139095040)),
  (some (2139095040, 2139095040), some (-2147483648, -1), some (2139095040, 2139095040), some (-2147483648, 2139095040)),
  (some (2139095040, 2139095040), some (2139095041, 2147483647), some (2139095040, 2139095040), some (-2147483648, 2139095040)),
  (some (2139095040, 2139095040), some (-5, 5), some (2139095040, 2139095040), some (-2147483648, 2139095040)),
  (some (2139095040, 2139095040), some (-2147483648, 1123942400), some (2139095040, 2139095040), some (-2147483648, 2139095040)),
  (some (-2147483648, -1), none, none, none),
  (some (-2147483648, -1), some (0, 0), some (0, 2139095040), some (-2147483648, 0)),
  (some (-2147483648, -1), some (0, 1123942400), some (0, 2139095040), some (-2147483648, 1123942400)),
  (some (-2147483648, -1), some (1065353216, 2139095040), some (1065353216, 2139095040), some (-2147483648, 2139095040)),
  (some (-2147483648, -1), some (2139095040, 2139095040), some (2139095040, 2139095040), some (-2147483648, 2139095040)),
  (some (-2147483648, -1), some (-2147483648, -1), none, none),
  (some (-2147483648, -1), some (2139095041, 2147483647), none, none),
  (some (-2147483648, -1), some (-5, 5), none, none),
  (some (-2147483648, -1), some (-2147483648, 1123942400), none, none),
  (some (2139095041, 2147483647), none, none, none),
  (some (2139095041, 2147483647), some (0, 0), some (0, 2139095040), some (-2147483648, 0)),
  (some (2139095041, 2147483647), some (0, 1123942400), some (0, 2139095040), some (-2147483648, 1123942400)),
  (some (2139095041, 2147483647), some (1065353216, 2139095040), some (1065353216, 2139095040), some (-2147483648, 2139095040)),
  (some (2139095041, 2147483647), some (2139095040, 2139095040), some (2139095040, 2139095040), some (-2147483648, 2139095040)),
  (some (2139095041, 2147483647), some (-2147483648, -1), none, none),
  (some (2139095041, 2147483647), some (2139095041, 2147483647), none, none),
  (some (2139095041, 2147483647), some (-5, 5), none, none),
  (some (2139095041, 2147483647), some (-2147483648, 1123942400), none, none),
  (some (-5, 5), none, none, none),
  (some (-5, 5), some (0, 0), some (0, 2139095040), some (-2147483648, 0)),
  (some (-5, 5), some (0, 1123942400), some (0, 2139095040), some (-2147483648, 1123942400)),
  (some (-5, 5), some (1065353216, 2139095040), some (1065353216, 2139095040), some (-2147483648, 2139095040)),
  (some (-5, 5), some (2139095040, 2139095040), some (2139095040, 2139095040), some (-2147483648, 2139095040)),
  (some (-5, 5), some (-2147483648, -1), none, none),
  (some (-5, 5), some (2139095041, 2147483647), none, none),
  (some (-5, 5), some (-5, 5), none, none),
  (some (-5, 5), some (-2147483648, 1123942400), none, none),
  (some (-2147483648, 1123942400), none, none, none),
  (some (-2147483648, 1123942400), some (0, 0), some (0, 2139095040), some (-2147483648, 0)),
  (some (-2147483648, 1123942400), some (0, 1123942400), some (0, 2139095040), some (-2147483648, 1123942400)),
  (some (-2147483648, 1123942400), some (1065353216, 2139095040), some (1065353216, 2139095040), some (-2147483648, 2139095040)),
  (some (-2147483648, 1123942400), some (2139095040, 2139095040), some (2139095040, 2139095040), some (-2147483648, 2139095040)),
  (some (-2147483648, 1123942400), some (-2147483648, -1), none, none),
  (some (-2147483648, 1123942400), some (2139095041, 2147483647), none, none),
  (some (-2147483648, 1123942400), some (-5, 5), none, none),
  (some (-2147483648, 1123942400), some (-2147483648, 1123942400), none, none)]

theorem rules_ruleRows :
    ruleRows.all (fun r => fmaxfRule r.1 r.2.1 == r.2.2.1 && fminfRule r.1 r.2.1 == r.2.2.2) := by
  decide +kernel

def rzRuleRows : List (Rng × Rng) := [
  (none, none),
  (some (0, 0), some (0, 0)),
  (some (0, 1123942400), some (0, 127)),
  (some (1065353216, 1123942400), some (1, 127)),
  (some (-5, 5), some (0, 0)),
  (some (-2147483648, -1), some (0, 0)),
  (some (-2147483648, 1124073471), some (0, 127)),
  (some (1056964608, 1078530011), some (0, 3)),
  (some (1123942400, 1124073472), none),
  (some (5, 3), none)]

theorem rzByteRule_rzRuleRows : rzRuleRows.all (fun r => rzByteRule r.1 == r.2) := by
  decide +kernel

/-! ## The exact-int8 profile's primitives -/

theorem F32ToS8RziRange : Accounting.Guarantees.Gates.F32ToS8RziRange := by
  intro a
  unfold f32ToS8Rzi
  split_ifs
  · exact ⟨by norm_num, by norm_num⟩
  · have h1 := min_le_left (128 : ℤ) (truncNat (f32Mag a).toNat)
    have h2 : (0 : ℤ) ≤ min 128 ((truncNat (f32Mag a).toNat : ℕ) : ℤ) := le_min (by norm_num) (Int.natCast_nonneg _)
    exact ⟨by dsimp only; omega, by dsimp only; omega⟩
  · have h1 := min_le_left (127 : ℤ) (truncNat a.toNat)
    have h2 : (0 : ℤ) ≤ min 127 ((truncNat a.toNat : ℕ) : ℤ) := le_min (by norm_num) (Int.natCast_nonneg _)
    exact ⟨by dsimp only; omega, by dsimp only; omega⟩

/-- A float of magnitude at most `127.0` truncates to at most 127. -/
lemma truncNat_le (u : ℕ) (hu : u ≤ 0x42FE0000) : truncNat u ≤ 127 := by
  obtain ⟨e, he⟩ : ∃ e, u / 2 ^ 23 = e := ⟨_, rfl⟩
  have he' : e ≤ 133 := by omega
  simp only [truncNat, he]
  by_cases h0 : e < 127
  · simp [h0]
  · interval_cases e <;> simp <;> omega

lemma fmaxf_negTop (r : ℤ) (hr : S32 r) :
    f32NaN (f32Fmaxf r f32NegTop) = false ∧ S32 (f32Fmaxf r f32NegTop) ∧
      -0x42FE0000 ≤ f32Key (f32Fmaxf r f32NegTop) := by
  have hn : f32NaN f32NegTop = false := by decide
  have hk : f32Key f32NegTop = -0x42FE0000 := by norm_num [f32Key, f32NegTop]
  have hs : S32 f32NegTop := by norm_num [S32, f32NegTop, s32Min, s32Max]
  unfold f32Fmaxf f32MaxMin
  cases hrn : f32NaN r
  · simp only [hn, Bool.and_false, Bool.false_eq_true, ↓reduceIte]
    split_ifs with h1 h2
    · rcases max_choice r f32NegTop with h | h <;> rw [h]
      · exact ⟨hrn, hr, by omega⟩
      · exact ⟨hn, hs, by omega⟩
    · exact ⟨hrn, hr, by omega⟩
    · exact ⟨hn, hs, by omega⟩
  · simp only [hn, Bool.and_false, Bool.false_eq_true, ↓reduceIte]
    exact ⟨by simp, hs, by omega⟩

lemma fminf_top (u : ℤ) (hn : f32NaN u = false) (hu : S32 u) (hk : -0x42FE0000 ≤ f32Key u) :
    f32NaN (f32Fminf u f32Top) = false ∧ S32 (f32Fminf u f32Top) ∧ -0x42FE0000 ≤ f32Key (f32Fminf u f32Top) ∧
      f32Key (f32Fminf u f32Top) ≤ 0x42FE0000 := by
  have tn : f32NaN f32Top = false := by decide
  have tk : f32Key f32Top = 0x42FE0000 := by norm_num [f32Key, f32Top]
  have ts : S32 f32Top := by norm_num [S32, f32Top, s32Min, s32Max]
  unfold f32Fminf f32MaxMin
  simp only [hn, tn, Bool.and_false, Bool.false_eq_true, ↓reduceIte]
  split_ifs with h1 h2
  · rcases min_choice u f32Top with h | h <;> rw [h]
    · exact ⟨hn, hu, hk, by omega⟩
    · exact ⟨tn, ts, by omega, by omega⟩
  · exact ⟨hn, hu, hk, by omega⟩
  · exact ⟨tn, ts, by omega, by omega⟩

lemma wrapS32_mem (z : ℤ) : Iv.Mem (s32Min, s32Max) (wrapS32 z) := by
  unfold wrapS32 Iv.Mem s32Min s32Max
  have h1 := Int.emod_nonneg (z - -2 ^ 31) (by norm_num : (2 : ℤ) ^ 32 ≠ 0)
  have h2 := Int.emod_lt_of_pos (z - -2 ^ 31) (by norm_num : (0 : ℤ) < 2 ^ 32)
  constructor <;> dsimp only <;> omega

theorem Int8ClampRange : Accounting.Guarantees.Gates.Int8ClampRange := by
  intro z
  have hr : S32 (wrapS32 z) := wrapS32_mem z
  generalize wrapS32 z = r at hr ⊢
  obtain ⟨un, us, uk⟩ := fmaxf_negTop r hr
  obtain ⟨vn, vs, vk1, vk2⟩ := fminf_top _ un us uk
  unfold int8Clamp f32ToS8Rzi
  generalize f32Fminf (f32Fmaxf r f32NegTop) f32Top = v at vn vs vk1 vk2 ⊢
  simp only [vn, Bool.false_eq_true, ↓reduceIte]
  unfold f32Key at vk1 vk2
  unfold S32 s32Min at vs
  split_ifs with hv
  · simp only [hv, ↓reduceIte] at vk1
    have hm : f32Mag v = v + 2 ^ 31 := by simp [f32Mag, hv]
    have := truncNat_le (f32Mag v).toNat (by rw [hm]; omega)
    have h2 : (0 : ℤ) ≤ min 128 ((truncNat (f32Mag v).toNat : ℕ) : ℤ) := le_min (by norm_num) (Int.natCast_nonneg _)
    have h3 := min_le_right (128 : ℤ) (truncNat (f32Mag v).toNat)
    exact ⟨by dsimp only; omega, by dsimp only; omega⟩
  · simp only [hv, ↓reduceIte] at vk2
    have := truncNat_le v.toNat (by omega)
    have h2 : (0 : ℤ) ≤ min 127 ((truncNat v.toNat : ℕ) : ℤ) := le_min (by norm_num) (Int.natCast_nonneg _)
    have h3 := min_le_left (127 : ℤ) (truncNat v.toNat)
    exact ⟨by dsimp only; omega, by dsimp only; omega⟩

theorem LutIndexRange : Accounting.Guarantees.Gates.LutIndexRange := by
  intro d a
  refine ⟨le_min (Int.ediv_nonneg (mul_nonneg (le_max_right d 0) (Int.emod_nonneg _ (by norm_num))) (by norm_num))
    (by norm_num), min_le_right _ _⟩

lemma exp2Lut127_mem : ∀ t ∈ exp2Lut127, 0 ≤ t ∧ t ≤ 127 := by
  decide +kernel

theorem Exp2Lut127Range : Accounting.Guarantees.Gates.Exp2Lut127Range := by
  refine ⟨by decide +kernel, fun t => ?_⟩
  rcases Nat.lt_or_ge t exp2Lut127.length with h | h
  · rw [List.getD_eq_getElem _ _ h]; exact exp2Lut127_mem _ (List.getElem_mem h)
  · rw [List.getD_eq_default _ _ h]; exact ⟨le_rfl, by norm_num⟩

theorem S32LowByteRange : Accounting.Guarantees.Gates.S32LowByteRange := by
  refine ⟨fun z => ?_, fun t => ?_⟩ <;> unfold s32LowByte
  · exact ⟨by dsimp only; omega, by dsimp only; omega⟩
  · have := t.isLt; omega

/-! ## Declared bounds, earned -/

theorem DeclaredWindow : Accounting.Guarantees.Gates.DeclaredWindow := by
  intro ι C x hP hD hX
  have hD' : ∀ i (h : i < C.length),
      (∀ j, Holds ((intervals (C.take i)).getD j (0, 0)) ((values (C.take i) x).getD j 0)) → ¬ C[i].op.IsDerived →
        C[i].lo ≤ C[i].raw x (values (C.take i) x) ∧ C[i].raw x (values (C.take i) x) ≤ C[i].hi := by
    intro i hi hinv hnd
    obtain ⟨-, hbud⟩ := hD i hi
    unfold IntGate.raw IntGate.operands
    cases hop : C[i].op with
    | input b s => simp only [IntOp.raw]; exact hX _ (List.getElem_mem hi) b s hop
    | budgeted cls f => simp only [IntOp.raw]; exact hbud cls f hop _ fun j _ => hinv j
    | _ => rw [hop] at hnd; exact absurd trivial hnd
  have hw : InWindow C x := fun i hi => by
    obtain ⟨h1, -, h2⟩ := hP i hi
    have hinv := window_inv' C x hP hD' i hi.le
    exact (gate_window C[i] x _ _ hinv h1 h2 (hD' i hi hinv)).1
  exact ⟨fun i hi _ => ⟨(hw i hi).1, (hw i hi).2.1⟩, hw⟩

lemma getD_intervals_mono (C : IntDag ι) (n j : ℕ) (hj : j < n) :
    ∀ m, n ≤ m → m ≤ C.length → (intervals (C.take m)).getD j (0, 0) = (intervals (C.take n)).getD j (0, 0) := by
  intro m hnm hm
  induction m with
  | zero => omega
  | succ m ih =>
    rcases Nat.eq_or_lt_of_le hnm with h | h
    · rw [h]
    · rw [intervals_take_succ C m (by omega),
        List.getD_append _ _ _ _ (by rw [intervals_take_length C m (by omega)]; omega)]
      exact ih (by omega) (by omega)

lemma getD_intervals_take (C : IntDag ι) (n j : ℕ) (hj : j < n) (hn : n ≤ C.length) :
    (intervals (C.take n)).getD j (0, 0) = (intervals C).getD j (0, 0) := by
  have := getD_intervals_mono C n j hj C.length hn le_rfl
  rw [List.take_length] at this
  exact this.symm

lemma getD_intervals_self (C : IntDag ι) (i : ℕ) (hi : i < C.length) :
    (intervals C).getD i (0, 0) = C[i].derive (intervals (C.take i)) := by
  rw [← getD_intervals_take C (i + 1) i (by omega) hi, intervals_take_succ C i hi,
    List.getD_append_right _ _ _ _ (by rw [intervals_take_length C i hi.le]), intervals_take_length C i hi.le,
    Nat.sub_self, List.getD_cons_zero]

lemma derive_congr (g : IntGate ι) (ivs ivs' : List Iv) (h : ∀ j ∈ g.args, ivs.getD j (0, 0) = ivs'.getD j (0, 0)) :
    g.derive ivs = g.derive ivs' := by
  simp only [IntGate.derive, List.map_congr_left h]

lemma getD_intervals_fix (C : IntDag ι) (i : ℕ) (hi : i < C.length) (hT : ∀ j ∈ C[i].args, j < i) :
    (intervals C).getD i (0, 0) = C[i].derive (intervals C) := by
  rw [getD_intervals_self C i hi]
  exact derive_congr _ _ _ fun j hj => getD_intervals_take C i j (hT j hj) hi.le

lemma getD_intervals_decl (C : IntDag ι) (i : ℕ) (hi : i < C.length) (h : ¬ C[i].op.IsDerived) :
    (intervals C).getD i (0, 0) = (C[i].lo, C[i].hi) := by
  rw [getD_intervals_self C i hi]
  unfold IntGate.derive
  cases hop : C[i].op with
  | input b s => rfl
  | budgeted cls f => rfl
  | _ => rw [hop] at h; exact absurd trivial h

lemma getD_intervals_const (C : IntDag ι) (i : ℕ) (hi : i < C.length) (c : ℤ) (h : C[i].op = .const c) :
    (intervals C).getD i (0, 0) = (c, c) := by
  rw [getD_intervals_self C i hi]
  simp [IntGate.derive, h]

lemma foldl_join_within (lo hi : ℤ) : ∀ (l : List Iv) (s : Iv), s.Within lo hi → (∀ b ∈ l, b.Within lo hi) →
    (l.foldl Iv.join s).Within lo hi
  | [], s, hs, _ => by simpa using hs
  | b :: l, s, hs, hl => by
    simp only [List.foldl_cons]
    have hb := hl b (by simp)
    exact foldl_join_within lo hi l _ ⟨le_min hs.1 hb.1, max_le hs.2 hb.2⟩
      fun b' hb' => hl b' (List.mem_cons_of_mem _ hb')

lemma mul_within {A B Ia Ib : Iv} (hA : A.Within Ia.1 Ia.2) (hA' : A.1 ≤ A.2) (hB : B.Within Ib.1 Ib.2)
    (hB' : B.1 ≤ B.2) : (Iv.mul A B).Within (Iv.mul Ia Ib).1 (Iv.mul Ia Ib).2 := by
  have a1 : Holds Ia A.1 := ⟨hA.1, le_trans hA' hA.2⟩
  have a2 : Holds Ia A.2 := ⟨le_trans hA.1 hA', hA.2⟩
  have b1 : Holds Ib B.1 := ⟨hB.1, le_trans hB' hB.2⟩
  have b2 : Holds Ib B.2 := ⟨le_trans hB.1 hB', hB.2⟩
  have c11 := Holds.mul a1 b1
  have c12 := Holds.mul a1 b2
  have c21 := Holds.mul a2 b1
  have c22 := Holds.mul a2 b2
  exact ⟨le_min (le_min c11.1 c12.1) (le_min c21.1 c22.1), max_le (max_le c11.2 c12.2) (max_le c21.2 c22.2)⟩

lemma foldl_add_within (p q : ℤ) : ∀ (L : List Iv) (s : Iv) (lo hi : ℤ), s.Within lo hi →
    (∀ X ∈ L, X.Within p q) → (L.foldl Iv.add s).Within (lo + L.length * p) (hi + L.length * q)
  | [], s, lo, hi, hs, _ => by simpa using hs
  | X :: L, s, lo, hi, hs, hL => by
    simp only [List.foldl_cons, List.length_cons]
    have hX := hL X (by simp)
    have := foldl_add_within p q L (Iv.add s X) (lo + p) (hi + q) ⟨add_le_add hs.1 hX.1, add_le_add hs.2 hX.2⟩
      fun Y hY => hL Y (List.mem_cons_of_mem _ hY)
    have e1 : lo + p + L.length * p = lo + ((L.length + 1 : ℕ) : ℤ) * p := by push_cast; ring
    have e2 : hi + q + L.length * q = hi + ((L.length + 1 : ℕ) : ℤ) * q := by push_cast; ring
    rw [← e1, ← e2]
    exact this

lemma forall_mem_zipWith {α β γ : Type} (f : α → β → γ) (P : γ → Prop) (A : List α) (B : List β)
    (h : ∀ u ∈ A, ∀ v ∈ B, P (f u v)) : ∀ X ∈ List.zipWith f A B, P X := by
  intro X hX
  obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem hX
  rw [List.getElem_zipWith]
  exact h _ (List.getElem_mem _) _ (List.getElem_mem _)

section chain

variable (pre : IntDag ι) (fa fb : Int8Fmt) (sat : Bool) (prim : String) (S : ℕ) (a b : ℕ → List ℕ)

lemma dotChain_length : (dotChain pre fa fb sat prim S a b).length = pre.length + 1 + S := by
  simp [dotChain]; omega

lemma dotChain_pre (i : ℕ) (h : i < pre.length) (h' : i < (dotChain pre fa fb sat prim S a b).length) :
    (dotChain pre fa fb sat prim S a b)[i] = pre[i] := by
  simp [dotChain, List.getElem_append_left h]

lemma dotChain_zero (h' : pre.length < (dotChain pre fa fb sat prim S a b).length) :
    (dotChain pre fa fb sat prim S a b)[pre.length] = plainGate (.const 0) [] 0 0 "" := by
  simp [dotChain]

lemma dotChain_step (s : ℕ) (h' : pre.length + 1 + s < (dotChain pre fa fb sat prim S a b).length) :
    (dotChain pre fa fb sat prim S a b)[pre.length + 1 + s] =
      ⟨.mma fa fb 32 sat, (pre.length + s) :: (a s ++ b s), s32Min, s32Max, 0, prim, .dot (32 * S) (32 * S) [] s,
        decide (s + 1 = S)⟩ := by
  simp only [dotChain]
  rw [List.getElem_append_right (by omega)]
  have : pre.length + 1 + s - pre.length = s + 1 := by omega
  simp [this]

lemma dotChain_take : (dotChain pre fa fb sat prim S a b).take pre.length = pre := by
  simp [dotChain]

end chain

/-- **The chain's window**: a dot chain on gates `pre` that are proven and earn their bounds, with no affine map or
s8-family step among them, whose operands' derived intervals lie in `Ia` and `Ib` within the steps' formats, and
whose `32·S` products' sum stays in s32, is proven and earns its bounds. -/
lemma dotChain_ok (pre : IntDag ι) (fa fb : Int8Fmt) (sat : Bool) (prim : String) (S : ℕ) (a b : ℕ → List ℕ)
    (Ia Ib : Iv) (hT : Topological pre) (hfree : ∀ g ∈ pre, ¬ g.op.IsMma ∧ ∀ c₀ cs, g.op ≠ .affine c₀ cs)
    (hpre : ∀ i (h : i < pre.length), pre[i].op.IsDerived →
      (pre[i].derive (intervals pre)).Within pre[i].lo pre[i].hi)
    (hdecl : ∀ i (h : i < pre.length), (∀ b s, pre[i].op = .input b s → TypeRange pre[i].lo pre[i].hi) ∧
      ∀ cls f, pre[i].op = .budgeted cls f → ∀ vs : List ℤ,
        (∀ j ∈ pre[i].args, ((intervals pre).getD j (0, 0)).Mem (vs.getD j 0)) →
          Iv.Mem (pre[i].lo, pre[i].hi) (f (pre[i].args.map fun j => vs.getD j 0)))
    (ha : ∀ s < S, (a s).length = 32 ∧ ∀ j ∈ a s, j < pre.length ∧
      ((intervals pre).getD j (0, 0)).Within Ia.1 Ia.2 ∧
        ((intervals pre).getD j (0, 0)).1 ≤ ((intervals pre).getD j (0, 0)).2)
    (hb : ∀ s < S, (b s).length = 32 ∧ ∀ j ∈ b s, j < pre.length ∧
      ((intervals pre).getD j (0, 0)).Within Ib.1 Ib.2 ∧
        ((intervals pre).getD j (0, 0)).1 ≤ ((intervals pre).getD j (0, 0)).2)
    (hfa : Ia.Within fa.lo fa.hi) (hfb : Ib.Within fb.lo fb.hi) (hp : (Iv.mul Ia Ib).1 ≤ 0) (hq : 0 ≤ (Iv.mul Ia Ib).2)
    (hlo : s32Min ≤ 32 * S * (Iv.mul Ia Ib).1) (hhi : 32 * S * (Iv.mul Ia Ib).2 ≤ s32Max) :
    Proven (dotChain pre fa fb sat prim S a b) ∧ Declared (dotChain pre fa fb sat prim S a b) := by
  set C := dotChain pre fa fb sat prim S a b with hC
  set p := (Iv.mul Ia Ib).1
  set q := (Iv.mul Ia Ib).2
  have hlen : C.length = pre.length + 1 + S := dotChain_length ..
  have hIpre : ∀ j, j < pre.length → (intervals C).getD j (0, 0) = (intervals pre).getD j (0, 0) := by
    intro j hj
    rw [← getD_intervals_take C pre.length j hj (by omega), hC, dotChain_take]
  have hTC : ∀ i (h : i < C.length), ∀ j ∈ C[i].args, j < i := by
    intro i hi j hj
    rcases Nat.lt_or_ge i pre.length with h | h
    · have hg : C[i] = pre[i] := dotChain_pre pre fa fb sat prim S a b i h hi
      rw [hg] at hj
      exact hT i h j hj
    rcases Nat.eq_or_lt_of_le h with rfl | h
    · have hg : C[pre.length] = _ := dotChain_zero pre fa fb sat prim S a b hi
      rw [hg] at hj
      simp [plainGate] at hj
    obtain ⟨s, rfl⟩ : ∃ s, i = pre.length + 1 + s := ⟨i - (pre.length + 1), by omega⟩
    have hg : C[pre.length + 1 + s] = _ := dotChain_step pre fa fb sat prim S a b s hi
    rw [hg] at hj
    rcases List.mem_cons.mp hj with rfl | hj
    · omega
    rcases List.mem_append.mp hj with hj | hj
    · have := ((ha s (by omega)).2 j hj).1; omega
    · have := ((hb s (by omega)).2 j hj).1; omega
  have hItake : ∀ i (h : i < C.length), ∀ j ∈ C[i].args,
      (intervals (C.take i)).getD j (0, 0) = (intervals C).getD j (0, 0) :=
    fun i hi j hj => getD_intervals_take C i j (hTC i hi j hj) hi.le
  have hder : ∀ i (h : i < C.length), C[i].derive (intervals (C.take i)) = C[i].derive (intervals C) :=
    fun i hi => derive_congr _ _ _ (hItake i hi)
  have hstep : ∀ s (hs : s < S), ((intervals C).getD (pre.length + 1 + s) (0, 0)) =
      (List.zipWith Iv.mul (((a s).map fun j => (intervals C).getD j (0, 0)))
        ((b s).map fun j => (intervals C).getD j (0, 0))).foldl Iv.add
          ((intervals C).getD (pre.length + s) (0, 0)) := by
    intro s hs
    have hi : pre.length + 1 + s < C.length := by omega
    have hg : C[pre.length + 1 + s] = _ := dotChain_step pre fa fb sat prim S a b s hi
    rw [getD_intervals_fix C _ hi (hTC _ hi), hg]
    simp only [IntGate.derive, List.map_cons, List.headD_cons, List.tail_cons, List.map_append]
    rw [List.take_left' (by rw [List.length_map, (ha s hs).1]), List.drop_left' (by rw [List.length_map, (ha s hs).1])]
  have hmul : ∀ s (hs : s < S), ∀ X ∈ List.zipWith Iv.mul (((a s).map fun j => (intervals C).getD j (0, 0)))
      ((b s).map fun j => (intervals C).getD j (0, 0)), X.Within p q := by
    intro s hs
    refine forall_mem_zipWith _ _ _ _ fun u hu v hv => ?_
    obtain ⟨j, hj, rfl⟩ := List.mem_map.mp hu
    obtain ⟨k, hk, rfl⟩ := List.mem_map.mp hv
    obtain ⟨hj1, hj2, hj3⟩ := (ha s hs).2 j hj
    obtain ⟨hk1, hk2, hk3⟩ := (hb s hs).2 k hk
    rw [hIpre j hj1, hIpre k hk1]
    exact mul_within hj2 hj3 hk2 hk3
  have hzip : ∀ s (hs : s < S), (List.zipWith Iv.mul (((a s).map fun j => (intervals C).getD j (0, 0)))
      ((b s).map fun j => (intervals C).getD j (0, 0))).length = 32 := by
    intro s hs
    simp [List.length_zipWith, (ha s hs).1, (hb s hs).1]
  have hacc : ∀ s, s ≤ S → ((intervals C).getD (pre.length + s) (0, 0)).Within (32 * s * p) (32 * s * q) := by
    intro s
    induction s with
    | zero =>
      intro _
      have hg : C[pre.length] = _ := dotChain_zero pre fa fb sat prim S a b (by rw [dotChain_length]; omega)
      rw [Nat.add_zero, getD_intervals_const C _ (by omega) 0 (by rw [hg]; rfl)]
      simp [Iv.Within]
    | succ s ih =>
      intro hs
      have := foldl_add_within p q _ _ _ _ (ih (by omega)) (hmul s (by omega))
      rw [hzip s (by omega)] at this
      rw [show pre.length + (s + 1) = pre.length + 1 + s by omega, hstep s (by omega)]
      refine ⟨le_trans (le_of_eq (by push_cast; ring)) this.1, le_trans this.2 (le_of_eq (by push_cast; ring))⟩
  have hbound : ∀ s, s < S → s32Min ≤ 32 * ((s + 1 : ℕ) : ℤ) * p ∧ 32 * ((s + 1 : ℕ) : ℤ) * q ≤ s32Max := by
    intro s hs
    have h1 : ((s + 1 : ℕ) : ℤ) ≤ S := by exact_mod_cast hs
    have h0 : (0 : ℤ) ≤ ((s + 1 : ℕ) : ℤ) := Int.natCast_nonneg _
    constructor
    · nlinarith
    · nlinarith
  refine ⟨fun i hi => ?_, fun i hi => ?_⟩
  · rcases Nat.lt_or_ge i pre.length with h | h
    · have hg : C[i] = pre[i] := dotChain_pre pre fa fb sat prim S a b i h hi
      refine ⟨fun hd => ?_, fun c₀ cs hop => ?_, fun fa' fb' k sat' hop => ?_⟩
      · rw [hder i hi]
        rw [derive_congr C[i] (intervals C) (intervals pre) fun j hj => hIpre j (by
          have := hTC i hi j hj; omega)]
        rw [hg] at hd ⊢
        exact hpre i h hd
      · rw [hg] at hop; exact absurd hop ((hfree _ (List.getElem_mem h)).2 c₀ cs)
      · rw [hg] at hop
        exact absurd (by rw [hop]; trivial) (hfree _ (List.getElem_mem h)).1
    rcases Nat.eq_or_lt_of_le h with rfl | h
    · have hg : C[pre.length] = _ := dotChain_zero pre fa fb sat prim S a b hi
      refine ⟨fun _ => ?_, fun c₀ cs hop => ?_, fun fa' fb' k sat' hop => ?_⟩
      · rw [hg]; simp [IntGate.derive, plainGate, Iv.Within]
      · rw [hg] at hop; simp [plainGate] at hop
      · rw [hg] at hop; simp [plainGate] at hop
    obtain ⟨s, rfl⟩ : ∃ s, i = pre.length + 1 + s := ⟨i - (pre.length + 1), by omega⟩
    have hs : s < S := by omega
    have hg : C[pre.length + 1 + s] = _ := dotChain_step pre fa fb sat prim S a b s hi
    refine ⟨fun _ => ?_, fun c₀ cs hop => ?_, fun fa' fb' k sat' hop => ?_⟩
    · rw [hder _ hi, ← getD_intervals_fix C _ hi (hTC _ hi), hstep s hs]
      have := foldl_add_within p q _ _ _ _ (hacc s hs.le) (hmul s hs)
      rw [hzip s hs] at this
      rw [hg]
      obtain ⟨b1, b2⟩ := hbound s hs
      exact ⟨le_trans b1 (le_trans (le_of_eq (by push_cast; ring)) this.1),
        le_trans this.2 (le_trans (le_of_eq (by push_cast; ring)) b2)⟩
    · rw [hg] at hop; simp at hop
    · rw [hg] at hop
      simp only [IntOp.mma.injEq] at hop
      obtain ⟨rfl, rfl, rfl, rfl⟩ := hop
      rw [hg]
      have hla := (ha s hs).1
      have hlb := (hb s hs).1
      simp only [List.length_cons, List.length_append, List.tail_cons]
      refine ⟨by omega, le_rfl, le_rfl, fun j hj => ?_, fun j hj => ?_⟩
      · rw [List.take_left' hla] at hj
        have hj' : j ∈ C[pre.length + 1 + s].args := by
          rw [hg]; exact List.mem_cons_of_mem _ (List.mem_append_left _ hj)
        rw [hItake _ hi j hj']
        obtain ⟨hj1, hj2, -⟩ := (ha s hs).2 j hj
        rw [hIpre j hj1]
        exact ⟨le_trans hfa.1 hj2.1, le_trans hj2.2 hfa.2⟩
      · rw [List.drop_left' hla] at hj
        have hj' : j ∈ C[pre.length + 1 + s].args := by
          rw [hg]; exact List.mem_cons_of_mem _ (List.mem_append_right _ hj)
        rw [hItake _ hi j hj']
        obtain ⟨hj1, hj2, -⟩ := (hb s hs).2 j hj
        rw [hIpre j hj1]
        exact ⟨le_trans hfb.1 hj2.1, le_trans hj2.2 hfb.2⟩
  · rcases Nat.lt_or_ge i pre.length with h | h
    · have hg : C[i] = pre[i] := dotChain_pre pre fa fb sat prim S a b i h hi
      obtain ⟨h1, h2⟩ := hdecl i h
      refine ⟨fun b s hop => ?_, fun cls f hop vs hvs => ?_⟩
      · rw [hg] at hop ⊢; exact h1 b s hop
      · have hvs' : ∀ j ∈ pre[i].args, ((intervals pre).getD j (0, 0)).Mem (vs.getD j 0) := by
          intro j hj
          have hj' : j ∈ C[i].args := by rw [hg]; exact hj
          rw [← hIpre j (by have := hTC i hi j hj'; omega), ← hItake i hi j hj']
          exact hvs j hj'
        rw [hg] at hop ⊢
        exact h2 cls f hop vs hvs'
    rcases Nat.eq_or_lt_of_le h with rfl | h
    · have hg : C[pre.length] = _ := dotChain_zero pre fa fb sat prim S a b hi
      refine ⟨fun b s hop => ?_, fun cls f hop => ?_⟩ <;> rw [hg] at hop <;> simp [plainGate] at hop
    obtain ⟨s, rfl⟩ : ∃ s, i = pre.length + 1 + s := ⟨i - (pre.length + 1), by omega⟩
    have hg : C[pre.length + 1 + s] = _ := dotChain_step pre fa fb sat prim S a b s hi
    refine ⟨fun b s hop => ?_, fun cls f hop => ?_⟩ <;> rw [hg] at hop <;> simp at hop

theorem Int8LinearWindow : Accounting.Guarantees.Gates.Int8LinearWindow := by
  intro S fs fr hS
  suffices hPD : Proven (int8Linear S fs fr) ∧ Declared (int8Linear S fs fr) from
    ⟨hPD.1, hPD.2, fun x hx => (DeclaredWindow _ x hPD.1 hPD.2 hx).2⟩
  set pre := (List.range (4 * (32 * S) + 1)).map (linearGate S fs fr) with hpre
  have hlen : pre.length = 4 * (32 * S) + 1 := by simp [pre]
  have hget : ∀ i (h : i < pre.length), pre[i] = linearGate S fs fr i := by intro i h; simp [pre]
  have hnd : ∀ i (h : i < pre.length), ¬ pre[i].op.IsDerived := by
    intro i h; rw [hget]; unfold linearGate; split_ifs <;> simp [plainGate, word32, IntOp.IsDerived]
  have hI : ∀ i (h : i < pre.length),
      (intervals pre).getD i (0, 0) = ((linearGate S fs fr i).lo, (linearGate S fs fr i).hi) := by
    intro i h; rw [getD_intervals_decl pre i h (hnd i h), hget]
  have hX : ∀ i, 2 * (32 * S) < i → i ≤ 4 * (32 * S) →
      (linearGate S fs fr i).lo = -128 ∧ (linearGate S fs fr i).hi = 127 := by
    intro i h1 h2; unfold linearGate; split_ifs <;> first | omega | simp [plainGate]
  have hS' : (S : ℤ) ≤ 4095 := by exact_mod_cast hS
  refine dotChain_ok pre .s8 .s8 true _ S _ _ (-128, 127) (-128, 127) ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_
  · intro i h j hj
    rw [hget] at hj
    unfold linearGate at hj
    split_ifs at hj <;> simp [plainGate] at hj <;> omega
  · intro g hg
    obtain ⟨i, -, rfl⟩ := List.mem_map.mp hg
    unfold linearGate
    split_ifs <;> simp [plainGate, word32, IntOp.IsMma]
  · intro i h hd; exact absurd hd (hnd i h)
  · intro i h
    rw [hget]
    unfold linearGate
    split_ifs
    · exact ⟨fun _ _ _ => ⟨15, .inl ⟨rfl, rfl⟩⟩, fun _ _ hop => by simp [plainGate] at hop⟩
    · refine ⟨fun _ _ hop => by simp [plainGate, word32] at hop, fun cls f hop vs _ => ?_⟩
      simp only [plainGate, word32, IntOp.budgeted.injEq] at hop
      obtain ⟨-, rfl⟩ := hop
      exact wrapS32_mem _
    · refine ⟨fun _ _ hop => by simp [plainGate, word32] at hop, fun cls f hop vs _ => ?_⟩
      simp only [plainGate, word32, IntOp.budgeted.injEq] at hop
      obtain ⟨-, rfl⟩ := hop
      exact wrapS32_mem _
    · refine ⟨fun _ _ hop => by simp [plainGate] at hop, fun cls f hop vs _ => ?_⟩
      simp only [plainGate, IntOp.budgeted.injEq] at hop
      obtain ⟨-, rfl⟩ := hop
      exact F32ToS8RziRange _
    · exact ⟨fun _ _ _ => ⟨7, .inl ⟨by simp [plainGate], by simp [plainGate]⟩⟩,
        fun _ _ hop => by simp [plainGate] at hop⟩
  · intro s hs
    refine ⟨by simp, fun j hj => ?_⟩
    obtain ⟨t, ht, rfl⟩ := List.mem_map.mp hj
    rw [List.mem_range] at ht
    have hj1 : 2 * (32 * S) + 1 + 32 * s + t < pre.length := by rw [hlen]; omega
    refine ⟨hj1, ?_⟩
    rw [hI _ hj1, (hX _ (by omega) (by omega)).1, (hX _ (by omega) (by omega)).2]
    exact ⟨⟨le_rfl, le_rfl⟩, by norm_num⟩
  · intro s hs
    refine ⟨by simp, fun j hj => ?_⟩
    obtain ⟨t, ht, rfl⟩ := List.mem_map.mp hj
    rw [List.mem_range] at ht
    have hj1 : 3 * (32 * S) + 1 + 32 * s + t < pre.length := by rw [hlen]; omega
    refine ⟨hj1, ?_⟩
    rw [hI _ hj1, (hX _ (by omega) (by omega)).1, (hX _ (by omega) (by omega)).2]
    exact ⟨⟨le_rfl, le_rfl⟩, by norm_num⟩
  · exact ⟨by simp [Int8Fmt.lo], by simp [Int8Fmt.hi]⟩
  · exact ⟨by simp [Int8Fmt.lo], by simp [Int8Fmt.hi]⟩
  · norm_num [Iv.mul]
  · norm_num [Iv.mul]
  · norm_num [Iv.mul, s32Min]; omega
  · norm_num [Iv.mul, s32Max]; omega

lemma pvPre_decl (S n : ℕ) (T : List ℤ) (pre : IntDag ℕ) (hlen : pre.length = 257 + 3 * n + 2 * (32 * S))
    (hget : ∀ i (h : i < pre.length), pre[i] = pvGate S n T i) (htop : Topological pre)
    (hgath : ∀ i (h : i < pre.length), 257 + 2 * n ≤ i → i < 257 + 3 * n →
      pre[i].lo = 0 ∧ pre[i].hi = 127 ∧ (pre[i].derive (intervals pre)).Within 0 127)
    (i : ℕ) (h : i < pre.length) :
    (∀ b s, pre[i].op = .input b s → TypeRange pre[i].lo pre[i].hi) ∧
      ∀ cls f, pre[i].op = .budgeted cls f → ∀ vs : List ℤ,
        (∀ j ∈ pre[i].args, ((intervals pre).getD j (0, 0)).Mem (vs.getD j 0)) →
          Iv.Mem (pre[i].lo, pre[i].hi) (f (pre[i].args.map fun j => vs.getD j 0)) := by
  by_cases hl : 257 + 3 * n ≤ i ∧ i < 257 + 3 * n + 32 * S ∧ 1 ≤ i - (257 + 3 * n) ∧ i - (257 + 3 * n) < n
  · have hg : pre[i] = plainGate (.budgeted .rounding fun v => s32LowByte (v.headD 0)) [i - n] 0 127
        "S32LowByte_v1" := by
      rw [hget]; unfold pvGate; split_ifs <;> first | omega | rfl
    rw [hg]
    refine ⟨fun _ _ hop => by simp [plainGate] at hop, fun cls f hop vs hvs => ?_⟩
    simp only [plainGate, IntOp.budgeted.injEq] at hop
    obtain ⟨-, rfl⟩ := hop
    have he : i - n < pre.length := by omega
    have hm := hvs (i - n) (by simp [plainGate])
    rw [getD_intervals_fix pre _ he (htop _ he)] at hm
    have hw := (hgath _ he (by omega) (by omega)).2.2
    have z0 : 0 ≤ vs.getD (i - n) 0 := le_trans hw.1 hm.1
    have z1 : vs.getD (i - n) 0 ≤ 127 := le_trans hm.2 hw.2
    simp only [plainGate, List.map_cons, List.map_nil, List.headD_cons]
    obtain ⟨t, ht⟩ : ∃ t : ℕ, (t : ℤ) = vs.getD (i - n) 0 := ⟨(vs.getD (i - n) 0).toNat, by omega⟩
    rw [← ht, S32LowByteRange.2 ⟨t, by omega⟩]
    rw [← ht] at z0 z1
    exact ⟨z0, z1⟩
  rw [hget]
  generalize hgi : pvGate S n T i = g
  unfold pvGate at hgi
  split_ifs at hgi <;> subst hgi
  · exact ⟨fun _ _ hop => by simp [plainGate] at hop, fun _ _ hop => by simp [plainGate] at hop⟩
  · exact ⟨fun _ _ _ => ⟨31, .inl ⟨rfl, rfl⟩⟩, fun _ _ hop => by simp [plainGate] at hop⟩
  · refine ⟨fun _ _ hop => by simp [plainGate] at hop, fun cls f hop vs _ => ?_⟩
    simp only [plainGate, IntOp.budgeted.injEq] at hop
    obtain ⟨-, rfl⟩ := hop
    exact LutIndexRange _ _
  · exact ⟨fun _ _ hop => by simp [plainGate] at hop, fun _ _ hop => by simp [plainGate] at hop⟩
  · omega
  · exact ⟨fun _ _ hop => by simp [plainGate] at hop, fun _ _ hop => by simp [plainGate] at hop⟩
  · exact ⟨fun _ _ _ => ⟨7, .inl ⟨by simp [plainGate], by simp [plainGate]⟩⟩, fun _ _ hop => by simp [plainGate] at hop⟩
  · exact ⟨fun _ _ hop => by simp [plainGate] at hop, fun _ _ hop => by simp [plainGate] at hop⟩

theorem LutPvWindow : Accounting.Guarantees.Gates.LutPvWindow := by
  intro S n T hS hT
  suffices hPD : Proven (lutPv S n T) ∧ Declared (lutPv S n T) from
    ⟨hPD.1, hPD.2, fun x hx => (DeclaredWindow _ x hPD.1 hPD.2 hx).2⟩
  set pre := (List.range (257 + 3 * n + 2 * (32 * S))).map (pvGate S n T) with hpre
  have hlen : pre.length = 257 + 3 * n + 2 * (32 * S) := by simp [pre]
  have hget : ∀ i (h : i < pre.length), pre[i] = pvGate S n T i := by intro i h; simp [pre]
  have hTg : ∀ t, 0 ≤ T.getD t 0 ∧ T.getD t 0 ≤ 127 := by
    intro t
    rcases Nat.lt_or_ge t T.length with h | h
    · rw [List.getD_eq_getElem _ _ h]; exact hT _ (List.getElem_mem h)
    · rw [List.getD_eq_default _ _ h]; norm_num
  have htop : Topological pre := by
    intro i h j hj
    rw [hget] at hj
    unfold pvGate at hj
    split_ifs at hj <;> simp [plainGate] at hj <;> omega
  have hIT : ∀ t, t < 256 → (intervals pre).getD t (0, 0) = (T.getD t 0, T.getD t 0) := by
    intro t ht
    have h : t < pre.length := by rw [hlen]; omega
    refine getD_intervals_const pre t h _ ?_
    rw [hget]; simp [pvGate, ht, plainGate]
  have hgath : ∀ i (h : i < pre.length), 257 + 2 * n ≤ i → i < 257 + 3 * n →
      pre[i].lo = 0 ∧ pre[i].hi = 127 ∧ (pre[i].derive (intervals pre)).Within 0 127 := by
    intro i h h1 h2
    have hg : pre[i] = plainGate (.gather (some 8) 0) ((i - n) :: List.range 256) 0 127 "Exp2Lut127_v1" := by
      rw [hget]; unfold pvGate; split_ifs <;> first | omega | rfl
    rw [hg]
    refine ⟨rfl, rfl, ?_⟩
    simp only [IntGate.derive, plainGate, List.map_cons, List.tail_cons]
    refine foldl_join_within 0 127 _ _ ⟨le_rfl, by norm_num⟩ fun b hb => ?_
    obtain ⟨t, ht, rfl⟩ := List.mem_map.mp hb
    rw [List.mem_range] at ht
    rw [hIT t ht]
    exact ⟨(hTg t).1, (hTg t).2⟩
  have hP : ∀ j, j < 32 * S → ((intervals pre).getD (257 + 3 * n + j) (0, 0)).Within 0 127 ∧
      ((intervals pre).getD (257 + 3 * n + j) (0, 0)).1 ≤ ((intervals pre).getD (257 + 3 * n + j) (0, 0)).2 := by
    intro j hj
    have h : 257 + 3 * n + j < pre.length := by rw [hlen]; omega
    by_cases hc : 1 ≤ j ∧ j < n
    · have hg : pre[257 + 3 * n + j] = plainGate (.budgeted .rounding fun v => s32LowByte (v.headD 0))
          [257 + 3 * n + j - n] 0 127 "S32LowByte_v1" := by
        rw [hget]; unfold pvGate; split_ifs <;> first | omega | rfl
      rw [getD_intervals_decl pre _ h (by rw [hg]; simp [plainGate, IntOp.IsDerived]), hg]
      simp [plainGate, Iv.Within]
    · have hg : pre[257 + 3 * n + j] = plainGate (.const 0) [] 0 0 "" := by
        rw [hget]; unfold pvGate; split_ifs <;> first | omega | rfl
      rw [getD_intervals_const pre _ h 0 (by rw [hg]; rfl)]
      simp [Iv.Within]
  have hV : ∀ j, j < 32 * S → ((intervals pre).getD (257 + 3 * n + 32 * S + j) (0, 0)).Within (-128) 127 ∧
      ((intervals pre).getD (257 + 3 * n + 32 * S + j) (0, 0)).1 ≤
        ((intervals pre).getD (257 + 3 * n + 32 * S + j) (0, 0)).2 := by
    intro j hj
    have h : 257 + 3 * n + 32 * S + j < pre.length := by rw [hlen]; omega
    by_cases hc : j < n
    · have hg : pre[257 + 3 * n + 32 * S + j] =
          plainGate (.input .run (257 + 3 * n + 32 * S + j)) [] (-128) 127 "" := by
        rw [hget]; unfold pvGate; split_ifs <;> first | omega | rfl
      rw [getD_intervals_decl pre _ h (by rw [hg]; simp [plainGate, IntOp.IsDerived]), hg]
      simp [plainGate, Iv.Within]
    · have hg : pre[257 + 3 * n + 32 * S + j] = plainGate (.const 0) [] 0 0 "" := by
        rw [hget]; unfold pvGate; split_ifs <;> first | omega | rfl
      rw [getD_intervals_const pre _ h 0 (by rw [hg]; rfl)]
      simp [Iv.Within]
  have hS' : (S : ℤ) ≤ 4128 := by exact_mod_cast hS
  refine dotChain_ok pre .s8 .s8 true _ S _ _ (0, 127) (-128, 127) htop ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_
  · intro g hg
    obtain ⟨i, -, rfl⟩ := List.mem_map.mp hg
    unfold pvGate
    split_ifs <;> simp [plainGate, IntOp.IsMma]
  · intro i h hd
    by_cases hg : 257 + 2 * n ≤ i ∧ i < 257 + 3 * n
    · obtain ⟨hl, hh, hw⟩ := hgath i h hg.1 hg.2
      rw [hl, hh]; exact hw
    rw [hget] at hd ⊢
    unfold pvGate at hd ⊢
    split_ifs at hd ⊢ <;>
      first
      | omega
      | (simp [plainGate, IntOp.IsDerived] at hd; done)
      | simp [IntGate.derive, plainGate, Iv.Within]
  · exact pvPre_decl S n T pre hlen hget htop hgath
  · intro s hs
    refine ⟨by simp, fun j hj => ?_⟩
    obtain ⟨t, ht, rfl⟩ := List.mem_map.mp hj
    rw [List.mem_range] at ht
    have e : 257 + 3 * n + 32 * s + t = 257 + 3 * n + (32 * s + t) := by omega
    rw [e]
    exact ⟨by rw [hlen]; omega, hP _ (by omega)⟩
  · intro s hs
    refine ⟨by simp, fun j hj => ?_⟩
    obtain ⟨t, ht, rfl⟩ := List.mem_map.mp hj
    rw [List.mem_range] at ht
    have e : 257 + 3 * n + 32 * S + 32 * s + t = 257 + 3 * n + 32 * S + (32 * s + t) := by omega
    rw [e]
    exact ⟨by rw [hlen]; omega, hV _ (by omega)⟩
  · exact ⟨by simp [Int8Fmt.lo], by simp [Int8Fmt.hi]⟩
  · exact ⟨by simp [Int8Fmt.lo], by simp [Int8Fmt.hi]⟩
  · norm_num [Iv.mul]
  · norm_num [Iv.mul]
  · norm_num [Iv.mul, s32Min]; omega
  · norm_num [Iv.mul, s32Max]; omega

end Accounting.SecurityProofs.Gates
