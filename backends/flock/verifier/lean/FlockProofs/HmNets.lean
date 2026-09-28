/-
Level 3: where `Flock.HmNets.layout` puts a built circuit's rows. Input bit `i`'s row is `[i]` in A and B; the `k`-th AND
in creation order is row `inRows + k`, its forms' columns (`cols`); output `j` is row `start + j`, its form's columns in A
and the constant in B (`layout_rows`). A committed bit's column is `col`: an input bit its own, the bit after the inputs
committed `k`-th `inRows + k` (`col_input`, `col_and`). `shaLaid_eq` and `hmLaid_eq` are the two nets' layouts as
`layout` of the builder's run, stated without running it.
-/
import Flock.HmNets

namespace Flock.HmNets

theorem le_nextPow2 (n : Nat) : n ≤ nextPow2 n := by
  unfold nextPow2 log2ceil
  split
  · rw [Nat.pow_zero]; omega
  · have := Nat.lt_log2_self (n := n - 1)
    omega

/-- A group of `groupWords n` words holds `n` bits. -/
theorem le_groupWords (n : Nat) : n ≤ groupWords n * WORD := by
  have := le_nextPow2 (max 1 ((n + WORD - 1) / WORD))
  unfold groupWords
  have h2 : n ≤ (n + WORD - 1) / WORD * WORD := by unfold WORD; omega
  have h3 : (n + WORD - 1) / WORD * WORD ≤ max 1 ((n + WORD - 1) / WORD) * WORD :=
    Nat.mul_le_mul_right _ (Nat.le_max_right _ _)
  exact Nat.le_trans h2 (Nat.le_trans h3 (Nat.mul_le_mul_right _ this))

theorem groupWords_pos (n : Nat) : 0 < groupWords n * WORD := by
  unfold groupWords nextPow2 WORD
  exact Nat.mul_pos (Nat.two_pow_pos _) (by decide)

theorem le_roundUp (a s : Nat) (hs : 0 < s) : a ≤ (a + s - 1) / s * s := by
  have h1 := Nat.div_add_mod' (a + s - 1) s
  have h2 := Nat.mod_lt (a + s - 1) hs
  omega

theorem getD_append_left {α : Type} {xs ys : Array α} {i : Nat} {d : α} (h : i < xs.size) :
    (xs ++ ys).getD i d = xs.getD i d := by
  simp [Array.getD_eq_getD_getElem?, Array.getElem?_append, h]

theorem getD_append_right {α : Type} {xs ys : Array α} {i : Nat} {d : α} (h : xs.size ≤ i) :
    (xs ++ ys).getD i d = ys.getD (i - xs.size) d := by
  simp [Array.getD_eq_getD_getElem?, Array.getElem?_append, Nat.not_lt.mpr h]

/-- **The layout's rows** (`ir_lower.layout`): with `nIn` input bits in a group of `inRows` rows and an output group of
`span` rows starting at `start` (the first multiple of `span` past the ANDs), input bit `i`'s row is `[i]` in A and B, the
`k`-th AND's (creation order: `st.rows` reversed) is row `inRows + k` with its forms' columns, output `j`'s is row
`start + j`, its form's columns in A and the constant in B, and the constant is the column after the output group. -/
theorem layout_rows (st : St) (inBits outBits : Array Nat) (outs : Array Lin) :
    let l := layout st inBits outBits outs
    let nIn := inBits.foldl (· + ·) 0
    let inRows := groupWords nIn * WORD
    let span := groupWords (outBits.foldl (· + ·) 0) * WORD
    let start := (inRows + st.rows.length + span - 1) / span * span
    l.constPos = start + span ∧
      (∀ i, i < nIn → l.ra.getD i #[] = #[i] ∧ l.rb.getD i #[] = #[i]) ∧
      (∀ k (hk : k < st.rows.reverse.length),
        l.ra.getD (inRows + k) #[] = cols nIn inRows l.constPos st.rows.reverse[k].a ∧
          l.rb.getD (inRows + k) #[] = cols nIn inRows l.constPos st.rows.reverse[k].b) ∧
      (∀ j (hj : j < outs.size),
        l.ra.getD (start + j) #[] = cols nIn inRows l.constPos outs[j] ∧ l.rb.getD (start + j) #[] = #[l.constPos]) := by
  intro l nIn inRows span start
  have hIn : nIn ≤ inRows := le_groupWords nIn
  have hspan : 0 < span := groupWords_pos _
  have hst : inRows + st.rows.length ≤ start := le_roundUp _ _ hspan
  have hc : l.constPos = start + span := by simp [l, layout, start, span, inRows, nIn]
  let rows := st.rows.reverse.toArray
  let e := fun (n : Nat) => Array.replicate n (#[] : Array Nat)
  let ins := (Array.range nIn).map fun i => #[i]
  have hra : l.ra = ins ++ e (inRows - nIn) ++ rows.map (cols nIn inRows l.constPos ·.a) ++
      e (start - inRows - st.rows.length) ++ outs.map (cols nIn inRows l.constPos) ++ e (span - outs.size) ++
      #[#[l.constPos]] := by
    simp only [l, layout, rows, e, ins, start, span, inRows, nIn, List.size_toArray, List.length_reverse]
  have hrb : l.rb = ins ++ e (inRows - nIn) ++ rows.map (cols nIn inRows l.constPos ·.b) ++
      e (start - inRows - st.rows.length) ++ outs.map (fun _ => #[l.constPos]) ++ e (span - outs.size) ++
      #[#[l.constPos]] := by
    simp only [l, layout, rows, e, ins, start, span, inRows, nIn, List.size_toArray, List.length_reverse]
  have hs : ∀ n, (e n).size = n := fun n => Array.size_replicate
  have hr : rows.size = st.rows.length := by simp [rows]
  have hi : ins.size = nIn := by simp [ins]
  refine ⟨hc, ?_, ?_, ?_⟩
  · intro i hlt
    rw [hra, hrb]
    simp (disch := (simp only [Array.size_append, Array.size_map, hs, hr, hi, List.size_toArray, Array.size_singleton]; omega))
      only [getD_append_left]
    simp [ins, Array.getD_eq_getD_getElem?, hlt]
  · intro k hk
    have hk' : k < st.rows.length := by simpa using hk
    rw [hra, hrb]
    simp (disch := (simp only [Array.size_append, Array.size_map, hs, hr, hi, List.size_toArray, Array.size_singleton]; omega))
      only [getD_append_left, getD_append_right]
    simp only [Array.size_append, hs, hi]
    rw [show inRows + k - (nIn + (inRows - nIn)) = k by omega]
    simp [rows, Array.getD_eq_getD_getElem?, hk']
  · intro j hj
    rw [hra, hrb]
    simp (disch := (simp only [Array.size_append, Array.size_map, hs, hr, hi, List.size_toArray, Array.size_singleton]; omega))
      only [getD_append_left, getD_append_right]
    simp only [Array.size_append, Array.size_map, hs, hi, hr]
    rw [show start + j - (nIn + (inRows - nIn) + st.rows.length + (start - inRows - st.rows.length)) = j by omega]
    simp [Array.getD_eq_getD_getElem?, hj]

theorem col_input {nIn inRows v : Var} (h : v < nIn) : col nIn inRows v = v := by
  simp [col, h]

/-- The bit committed `k`-th after the `nIn` input bits is column `inRows + k`, the `k`-th AND's row. -/
theorem col_and (nIn inRows k : Nat) : col nIn inRows (nIn + k) = inRows + k := by
  have h : ¬ nIn + k < nIn := by omega
  simp only [col, h, ite_false]
  omega

/- By `delta`, not `unfold`: `unfold`'s simplification reduces the projections of the builder's run, which builds the net. -/
theorem shaLaid_eq : shaLaid () =
    layout (compression3.run St.empty).2 SHA_IN SHA_OUT (compression3.run St.empty).1.2.flatten.toArray := by
  delta shaLaid
  rfl

theorem hmLaid_eq : hmLaid () =
    layout ((hm96Row Hm96.Default512.key).run St.empty).2 HM_IN HM_OUT
      ((hm96Row Hm96.Default512.key).run St.empty).1.2.2.2.toArray := by
  delta hmLaid
  rfl

end Flock.HmNets
