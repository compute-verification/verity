module

public import Flock.HmNets
public import Proofs.Flock.Verifier.HmNets
public import Properties.Flock.Nets

/-!
# A built circuit's layout builds its net

`Net.ofRows` succeeds on `HmNets.layout` of a builder's run, at the layout's own header fields, with A and B the sparse
matrices of its rows (`layout_builds`), when the builder's rows output the bits after the inputs in creation order, each
row reads only bits below its own, each output reads only committed bits, and the slot is `2 ^ 18` columns with the
output group inside it. `ofRows_layout` is the same at the level of the rows: every check `Net.ofRows` runs passes on rows
laid out as `layout` lays them (the loops by `checkOrder_ok` and `sparse_ok`). `layout_facts` reads the layout row by row,
which the line lengths read too, and `rows_le` bounds every column by the constant's.
-/

@[expose] public section

open Flock Properties.Flock.Nets

namespace FlockDeploy.Concrete.Nets

theorem forIn_unit_ok {α ε : Type} : ∀ (l : List α) (f : α → Unit → Except ε (ForInStep Unit)),
    (∀ a ∈ l, f a () = .ok (.yield ())) → forIn l () f = .ok ()
  | [], _, _ => rfl
  | a :: l, f, h => by
    rw [List.forIn_cons, h a List.mem_cons_self]
    exact forIn_unit_ok l f fun b hb => h b (List.mem_cons_of_mem a hb)

theorem forIn_ok {α β ε : Type} : ∀ (l : List α) (f : α → β → Except ε (ForInStep β)),
    (∀ a ∈ l, ∀ b, ∃ b', f a b = .ok (.yield b')) → ∀ b, ∃ b', forIn l b f = .ok b'
  | [], _, _, b => ⟨b, rfl⟩
  | a :: l, f, h, b => by
    obtain ⟨b', hb'⟩ := h a List.mem_cons_self b
    rw [List.forIn_cons, hb']
    exact forIn_ok l f (fun c hc => h c (List.mem_cons_of_mem a hc)) b'

theorem bind_ok {α β ε : Type} {x : Except ε α} {g : α → Except ε β} (hx : ∃ a, x = .ok a) (hg : ∀ a, ∃ b, g a = .ok b) :
    ∃ b, (x >>= g) = .ok b := by
  obtain ⟨a, rfl⟩ := hx
  exact hg a

theorem bind_yield_ok {α β ε : Type} {x : Except ε α} {k : α → β} (hx : ∃ a, x = .ok a) :
    ∃ b, (x >>= fun a => pure (ForInStep.yield (k a))) = .ok (.yield b) := by
  obtain ⟨a, rfl⟩ := hx
  exact ⟨k a, rfl⟩

/-- **The row order passes** when every column a computed row reads is the constant, or an earlier column that is not
input padding. -/
theorem checkOrder_ok (name : String) (inCols constPos : Nat) (ra rb : Array (Array Nat))
    (h : ∀ r, inCols ≤ r → r < constPos → ∀ c ∈ ra.getD r #[] ++ rb.getD r #[], c = constPos ∨
      (c < r ∧ (c < inCols && (ra.getD c #[]).isEmpty && (rb.getD c #[]).isEmpty) = false)) :
    Net.checkOrder name inCols constPos ra rb = .ok () := by
  unfold Net.checkOrder
  dsimp only
  rw [Std.Legacy.Range.forIn_eq_forIn_range', forIn_unit_ok]
  · rfl
  intro r hr
  simp only [List.mem_range'_1, Std.Legacy.Range.size, Nat.add_sub_cancel, Nat.div_one] at hr
  rw [← Array.forIn_toList, forIn_unit_ok]
  · rfl
  intro c hc
  by_cases hcp : c = constPos
  · subst hcp
    simp only [beq_self_eq_true, ↓reduceIte]
    rfl
  rcases h r (by omega) (by omega) c (Array.mem_toList_iff.1 hc) with hcp' | ⟨hlt, hpad⟩
  · exact absurd hcp' hcp
  · simp only [beq_eq_false_iff_ne.2 hcp, Bool.false_eq_true, ↓reduceIte, ge_iff_le, Nat.not_le.2 hlt, hpad]
    rfl

/-- **`Sparse.ofRows` succeeds** when every column is below the width. -/
theorem sparse_ok (rows : Array (Array Nat)) (width : Nat) (h : ∀ r ∈ rows, ∀ c ∈ r, c < width) :
    ∃ s, Sparse.ofRows rows width = .ok s := by
  unfold Sparse.ofRows
  dsimp only
  rw [← Array.forIn_toList]
  refine bind_ok (forIn_ok _ _ (fun r hr b => ?_) _) fun _ => ⟨_, rfl⟩
  rw [← Array.forIn_toList]
  refine bind_yield_ok (forIn_ok _ _ (fun c hc d => ⟨pushLe32 d c, ?_⟩) _)
  simp only [ge_iff_le, Nat.not_le.2 (h r (Array.mem_toList_iff.1 hr) c (Array.mem_toList_iff.1 hc)), ↓reduceIte]
  rfl

open Flock.HmNets hiding and

/-- **The layout's sizes, padding and constant row**: both sides have a row per useful column; the input group's padding,
the rows between the ANDs and the output group, and the output group's rows past the outputs are empty in A and B; the
constant row is `[const]` in both. -/
theorem layout_pad (st : St) (inBits outBits : Array Nat) (outs : Array Lin) :
    let l := layout st inBits outBits outs
    let nIn := inBits.foldl (· + ·) 0
    let inRows := groupWords nIn * WORD
    let span := groupWords (outBits.foldl (· + ·) 0) * WORD
    let start := (inRows + st.rows.length + span - 1) / span * span
    outs.size ≤ span →
      l.ra.size = start + span + 1 ∧ l.rb.size = start + span + 1 ∧ l.useful = start + span + 1 ∧
      (∀ i, nIn ≤ i → i < inRows → l.ra.getD i #[] = #[] ∧ l.rb.getD i #[] = #[]) ∧
      (∀ i, inRows + st.rows.length ≤ i → i < start → l.ra.getD i #[] = #[] ∧ l.rb.getD i #[] = #[]) ∧
      (∀ i, start + outs.size ≤ i → i < start + span → l.ra.getD i #[] = #[] ∧ l.rb.getD i #[] = #[]) ∧
      l.ra.getD (start + span) #[] = #[start + span] ∧ l.rb.getD (start + span) #[] = #[start + span] := by
  intro l nIn inRows span start hsz
  have hIn : nIn ≤ inRows := le_groupWords nIn
  have hspan : 0 < span := groupWords_pos _
  have hst : inRows + st.rows.length ≤ start := le_roundUp _ _ hspan
  let rows := st.rows.reverse.toArray
  let e := fun (n : Nat) => Array.replicate n (#[] : Array Nat)
  let ins := (Array.range nIn).map fun i => #[i]
  have hra : l.ra = ins ++ e (inRows - nIn) ++ rows.map (cols nIn inRows (start + span) ·.a) ++
      e (start - inRows - st.rows.length) ++ outs.map (cols nIn inRows (start + span)) ++ e (span - outs.size) ++
      #[#[start + span]] := by
    simp only [l, layout, rows, e, ins, start, span, inRows, nIn, List.size_toArray, List.length_reverse]
  have hrb : l.rb = ins ++ e (inRows - nIn) ++ rows.map (cols nIn inRows (start + span) ·.b) ++
      e (start - inRows - st.rows.length) ++ outs.map (fun _ => #[start + span]) ++ e (span - outs.size) ++
      #[#[start + span]] := by
    simp only [l, layout, rows, e, ins, start, span, inRows, nIn, List.size_toArray, List.length_reverse]
  have hs : ∀ n, (e n).size = n := fun n => Array.size_replicate
  have hr : rows.size = st.rows.length := by simp [rows]
  have hi : ins.size = nIn := by simp [ins]
  have he : ∀ n j, (e n).getD j #[] = #[] := fun n j => by
    simp only [e, Array.getD_eq_getD_getElem?, Array.getElem?_replicate]
    split <;> rfl
  have hu : l.useful = start + span + 1 := by simp [l, layout, start, span, inRows, nIn]
  refine ⟨?_, ?_, hu, fun i h1 h2 => ⟨?_, ?_⟩, fun i h1 h2 => ⟨?_, ?_⟩, fun i h1 h2 => ⟨?_, ?_⟩, ?_, ?_⟩
  all_goals first
    | (rw [hra]; simp only [Array.size_append, Array.size_map, hs, hr, hi, Array.size_singleton]; omega)
    | (rw [hrb]; simp only [Array.size_append, Array.size_map, hs, hr, hi, Array.size_singleton]; omega)
    | skip
  all_goals first
    | (rw [hra]
       simp (disch := (simp only [Array.size_append, Array.size_map, hs, hr, hi, Array.size_singleton]; omega))
         only [getD_append_left, getD_append_right, he])
    | (rw [hrb]
       simp (disch := (simp only [Array.size_append, Array.size_map, hs, hr, hi, Array.size_singleton]; omega))
         only [getD_append_left, getD_append_right, he])
    | skip
  all_goals
    simp only [Array.size_append, Array.size_map, hs, hr, hi]
    rw [show start + span - (nIn + (inRows - nIn) + st.rows.length + (start - inRows - st.rows.length) + outs.size +
      (span - outs.size)) = 0 by omega]
    rfl

theorem isPow2_groupWords (n : Nat) : isPow2 (groupWords n) = true := by
  simp only [isPow2, groupWords, nextPow2, Nat.log2_two_pow, Nat.two_pow_pos, decide_true, beq_self_eq_true, Bool.and_self]

theorem mem_cols {nIn inRows cp c : Nat} {f : Lin} (h : c ∈ cols nIn inRows cp f) :
    c = cp ∨ ∃ v ∈ f.s, c = col nIn inRows v := by
  unfold cols at h
  rcases Array.mem_append.1 h with h | h
  · obtain ⟨v, hv, rfl⟩ := List.mem_map.1 (List.mem_toArray.1 h)
    exact Or.inr ⟨v, hv, rfl⟩
  · split at h
    · exact Or.inl (List.mem_singleton.1 (Array.mem_toList_iff.2 h))
    · simp at h

theorem log2ceil_le (n : Nat) : n ≤ 2 ^ log2ceil n := le_nextPow2 n

theorem isPow2_two_pow (k : Nat) : isPow2 (2 ^ k) = true := by
  simp only [isPow2, Nat.log2_two_pow, Nat.two_pow_pos, decide_true, beq_self_eq_true, Bool.and_self]

/-- A bit below the `k`-th committed after the inputs has a column below that AND's row, outside the input padding. -/
theorem col_lt {nIn inRows k v : Nat} (hIn : nIn ≤ inRows) (hv : v < nIn + k) :
    col nIn inRows v < inRows + k ∧ (col nIn inRows v < nIn ∨ inRows ≤ col nIn inRows v) := by
  by_cases h : v < nIn
  · rw [col_input h]
    omega
  · obtain ⟨d, rfl⟩ : ∃ d, v = nIn + d := ⟨v - nIn, by omega⟩
    rw [col_and]
    omega

/-- **Every column of rows laid out as `layout` lays them is at most the constant's.** -/
theorem rows_le (ra rb : Array (Array Nat)) (nIn inRows cp : Nat) (hsa : ra.size = cp + 1) (hsb : rb.size = cp + 1)
    (hIn : nIn ≤ inRows) (hcp : inRows ≤ cp)
    (hins : ∀ i < nIn, ra.getD i #[] = #[i] ∧ rb.getD i #[] = #[i])
    (hpI : ∀ i, nIn ≤ i → i < inRows → ra.getD i #[] = #[] ∧ rb.getD i #[] = #[])
    (hcon : ra.getD cp #[] = #[cp] ∧ rb.getD cp #[] = #[cp])
    (hord : ∀ r, inRows ≤ r → r < cp → ∀ c ∈ ra.getD r #[] ++ rb.getD r #[], c = cp ∨ (c < r ∧ (c < nIn ∨ inRows ≤ c))) :
    ∀ i, ∀ c ∈ ra.getD i #[] ++ rb.getD i #[], c ≤ cp := by
  intro i c hc
  by_cases h1 : i < nIn
  · rw [(hins i h1).1, (hins i h1).2] at hc
    simp at hc
    omega
  by_cases h2 : i < inRows
  · rw [(hpI i (by omega) h2).1, (hpI i (by omega) h2).2] at hc
    simp at hc
  by_cases h3 : i < cp
  · rcases hord i (by omega) h3 c hc with h | ⟨h, _⟩ <;> omega
  by_cases h4 : i = cp
  · subst h4
    rw [hcon.1, hcon.2] at hc
    simp at hc
    omega
  · have ha : ra.getD i #[] = #[] := by simp [Array.getD_eq_getD_getElem?, hsa, show ¬ i < cp + 1 by omega]
    have hb : rb.getD i #[] = #[] := by simp [Array.getD_eq_getD_getElem?, hsb, show ¬ i < cp + 1 by omega]
    rw [ha, hb] at hc
    simp at hc

/-- **`Net.ofRows` builds a net from rows laid out as `layout` lays them**: `nIn` input rows `[i]·[i]` from column 0
in a group of `inW` words, empty padding to the group's end, a constant row `[cp]·[cp]` at the last column `cp`, and
between them rows that read only the constant or earlier columns outside the padding; one input group and one output
group of power-of-two words, the output group aligned and inside a slot of `2 ^ 18` columns. -/
theorem ofRows_layout (name : String) (inBits outBits : Array Nat) (ra rb : Array (Array Nat))
    (nIn inW outW start cp : Nat) (hsa : ra.size = cp + 1) (hsb : rb.size = cp + 1)
    (hnIn : inBits.foldl (· + ·) 0 = nIn) (hnInW : nIn ≤ inW * WORD) (hout : outBits.foldl (· + ·) 0 ≤ outW * WORD)
    (hinW : isPow2 inW = true) (houtW : isPow2 outW = true) (hal : start / WORD % outW = 0)
    (hst : inW * WORD ≤ start) (hcp : cp = start + outW * WORD)
    (hins : ∀ i < nIn, ra.getD i #[] = #[i] ∧ rb.getD i #[] = #[i])
    (hpI : ∀ i, nIn ≤ i → i < inW * WORD → ra.getD i #[] = #[] ∧ rb.getD i #[] = #[])
    (hcon : ra.getD cp #[] = #[cp] ∧ rb.getD cp #[] = #[cp])
    (hord : ∀ r, inW * WORD ≤ r → r < cp → ∀ c ∈ ra.getD r #[] ++ rb.getD r #[],
      c = cp ∨ (c < r ∧ (c < nIn ∨ inW * WORD ≤ c)))
    (hU : max 7 (log2ceil (cp + 1)) = 18) (hD : start / WORD + outW ≤ 2 ^ 18 / WORD) :
    Net.ofRows name (cp + 1) cp #[⟨0, inW, inBits⟩] #[⟨start / WORD, outW, outBits⟩] ra rb =
      .ok (built name (cp + 1) ⟨0, inW, inBits⟩ ⟨start / WORD, outW, outBits⟩ inW (sparseOf ra) (sparseOf rb)) := by
  have hpad : ∀ c, c < nIn ∨ inW * WORD ≤ c →
      (decide (c < inW * WORD) && (ra.getD c #[]).isEmpty && (rb.getD c #[]).isEmpty) = false := by
    intro c hc
    rcases hc with hc | hc
    · rw [(hins c hc).1]
      simp
    · simp [Nat.not_lt.2 hc]
  have hco : Net.checkOrder "flock-ir-unit/v2" (inW * WORD) cp ra rb = .ok () :=
    checkOrder_ok _ _ _ _ _ fun r h1 h2 c hc => (hord r h1 h2 c hc).imp_right fun ⟨h, h'⟩ => ⟨h, hpad c h'⟩
  have hle := rows_le ra rb nIn (inW * WORD) cp hsa hsb hnInW (by omega) hins hpI hcon hord
  have hw : cp + 1 ≤ 2 ^ 18 := hU ▸ Nat.le_trans (log2ceil_le _) (Nat.pow_le_pow_right (by decide) (Nat.le_max_right _ _))
  have hlt : ∀ (x : Array (Array Nat)), (x = ra ∨ x = rb) → ∀ r ∈ x, ∀ c ∈ r, c < 2 ^ 18 := by
    intro x hx r hr c hc
    obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hr
    have : x.getD i #[] = x[i] := by simp [Array.getD_eq_getD_getElem?, hi]
    have := hle i c (by rcases hx with rfl | rfl <;> simp [this, hc])
    omega
  obtain ⟨a, ha⟩ := sparse_ok ra _ (hlt ra (Or.inl rfl))
  obtain ⟨b, hb⟩ := sparse_ok rb _ (hlt rb (Or.inr rfl))
  have hsA : sparseOf ra = a := by simp [sparseOf, ha, Except.toOption]
  have hsB : sparseOf rb = b := by simp [sparseOf, hb, Except.toOption]
  rw [hsA, hsB]
  have hback : (Option.map (fun g : Group => g.col + g.words) (#[⟨0, inW, inBits⟩] : Array Group).back?).getD 0 = inW := by
    simp
  have hg1 : ((#[⟨0, inW, inBits⟩, ⟨start / WORD, outW, outBits⟩] : Array Group).any
      fun g => !(isPow2 g.words && g.col % g.words == 0)) = false := by
    simp [hinW, houtW, hal]
  have hg2 : ((#[⟨0, inW, inBits⟩] : Array Group).any fun g => decide (g.bits.foldl (· + ·) 0 > g.words * WORD) ||
      decide (g.col + g.words > inW)) = false := by
    simp only [List.any_toArray, List.any_cons, List.any_nil, Bool.or_false, hnIn]
    simp [Nat.not_lt.2 hnInW]
  have hg3 : ((#[⟨start / WORD, outW, outBits⟩] : Array Group).any fun g =>
      decide (g.bits.foldl (· + ·) 0 > g.words * WORD)) = false := by
    simp [Nat.not_lt.2 hout]
  have hfind : (List.range (inW * WORD)).find? (fun i =>
      !((ra.getD i #[] == #[i] && rb.getD i #[] == #[i]) || ((ra.getD i #[]).isEmpty && (rb.getD i #[]).isEmpty))) = none := by
    rw [List.find?_eq_none]
    intro i hi
    rw [List.mem_range] at hi
    by_cases h : i < nIn
    · simp [(hins i h).1, (hins i h).2]
    · simp [(hpI i (by omega) hi).1, (hpI i (by omega) hi).2]
  have hinp : Net.inputsOk #[⟨0, inW, inBits⟩] ra rb = true := by
    simp only [Net.inputsOk]
    simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range, decide_eq_true_eq, Array.all_eq_true', beq_iff_eq]
    refine ⟨fun i hi => by simp at hi, fun g hg j hj => ?_⟩
    simp at hg
    subst hg
    simp only [Nat.zero_mul, Nat.zero_add] at hj ⊢
    rw [hnIn] at hj
    exact hins j hj
  have hd11 : (#[⟨start / WORD, outW, outBits⟩] : Array Group).find? (fun g => decide (g.col + g.words > 2 ^ 18 / WORD)) =
      none := by
    simp [Nat.not_lt.2 hD]
  unfold Net.ofRows
  simp only [hU, hback, List.append_toArray, List.cons_append, List.nil_append, hg1, hg2, hg3, hfind, hco,
    hcon.1, hcon.2, hinp, ha, hb, hd11, Bool.false_eq_true, ↓reduceIte, Bool.not_true, bne_self_eq_false, Bool.or_self,
    show ¬ inW * WORD > cp by omega, show (#[⟨start / WORD, outW, outBits⟩] : Array Group).isEmpty = false from rfl]
  rfl

/-- **A built circuit's layout, row by row**: when the builder's rows output the bits after the `nIn` inputs in creation
order, each row reads only bits below its own, and each output reads only committed bits and fits its group, the layout
has a row per useful column on both sides, its groups and constant at its header fields, input rows `[i]·[i]`, empty
padding, the `k`-th AND's forms' columns at row `inRows + k`, output `j`'s form's columns and the constant at row
`start + j`, and every computed row reads only the constant or earlier columns outside the input padding. -/
theorem layout_facts (st : St) (inBits outBits : Array Nat) (outs : Array Lin) (nIn inW outW start : Nat)
    (hnIn : inBits.foldl (· + ·) 0 = nIn) (hinW : groupWords nIn = inW)
    (houtW : groupWords (outBits.foldl (· + ·) 0) = outW)
    (hstart : (inW * WORD + st.rows.length + outW * WORD - 1) / (outW * WORD) * (outW * WORD) = start)
    (hn : st.next = nIn + st.rows.length)
    (hout : ∀ k (hk : k < st.rows.reverse.length), st.rows.reverse[k].out = nIn + k)
    (hrow : ∀ r ∈ st.rows, (∀ v ∈ r.a.s, v < r.out) ∧ ∀ v ∈ r.b.s, v < r.out)
    (ho : ∀ f ∈ outs.toList, ∀ v ∈ f.s, v < st.next) (hsz : outs.size ≤ outW * WORD) (l : Laid)
    (hl : layout st inBits outBits outs = l) :
    l.ra.size = start + outW * WORD + 1 ∧ l.rb.size = start + outW * WORD + 1 ∧
      l.useful = start + outW * WORD + 1 ∧ l.constPos = start + outW * WORD ∧
      l.inGroups = #[⟨0, inW, inBits⟩] ∧ l.outGroups = #[⟨start / WORD, outW, outBits⟩] ∧
      start / WORD % outW = 0 ∧ inW * WORD + st.rows.length ≤ start ∧ nIn ≤ inW * WORD ∧
      (∀ i < nIn, l.ra.getD i #[] = #[i] ∧ l.rb.getD i #[] = #[i]) ∧
      (∀ i, nIn ≤ i → i < inW * WORD → l.ra.getD i #[] = #[] ∧ l.rb.getD i #[] = #[]) ∧
      (∀ i, inW * WORD + st.rows.length ≤ i → i < start → l.ra.getD i #[] = #[] ∧ l.rb.getD i #[] = #[]) ∧
      (∀ i, start + outs.size ≤ i → i < start + outW * WORD → l.ra.getD i #[] = #[] ∧ l.rb.getD i #[] = #[]) ∧
      (∀ k (hk : k < st.rows.reverse.length),
        l.ra.getD (inW * WORD + k) #[] = cols nIn (inW * WORD) (start + outW * WORD) st.rows.reverse[k].a ∧
          l.rb.getD (inW * WORD + k) #[] = cols nIn (inW * WORD) (start + outW * WORD) st.rows.reverse[k].b) ∧
      (∀ j (hj : j < outs.size), l.ra.getD (start + j) #[] = cols nIn (inW * WORD) (start + outW * WORD) outs[j] ∧
        l.rb.getD (start + j) #[] = #[start + outW * WORD]) ∧
      (l.ra.getD (start + outW * WORD) #[] = #[start + outW * WORD] ∧
        l.rb.getD (start + outW * WORD) #[] = #[start + outW * WORD]) ∧
      ∀ r, inW * WORD ≤ r → r < start + outW * WORD → ∀ c ∈ l.ra.getD r #[] ++ l.rb.getD r #[],
        c = start + outW * WORD ∨ (c < r ∧ (c < nIn ∨ inW * WORD ≤ c)) := by
  subst hl
  obtain ⟨hsa, hsb, hu, hpI, hpM, hpO, hca, hcb⟩ := layout_pad st inBits outBits outs (by rw [houtW]; exact hsz)
  obtain ⟨hc, hins, hands, houts⟩ := layout_rows st inBits outBits outs
  have hog : (layout st inBits outBits outs).outGroups = #[⟨start / WORD, outW, outBits⟩] := by
    subst hnIn hinW houtW hstart
    simp [layout]
  have hig : (layout st inBits outBits outs).inGroups = #[⟨0, inW, inBits⟩] := by
    subst hnIn hinW
    rfl
  simp only [hnIn, hinW, houtW, hstart] at hsa hsb hu hpI hpM hpO hca hcb hc hins hands houts
  have hspan : 0 < outW * WORD := houtW ▸ groupWords_pos _
  have hst : inW * WORD + st.rows.length ≤ start := hstart ▸ le_roundUp _ _ hspan
  have hIn : nIn ≤ inW * WORD := hinW ▸ le_groupWords _
  rw [hc] at hands houts
  have hal : start / WORD % outW = 0 := by
    rw [← hstart]
    generalize (inW * WORD + st.rows.length + outW * WORD - 1) / (outW * WORD) = q
    rw [← Nat.mul_assoc, Nat.mul_div_cancel _ (by decide : 0 < WORD)]
    exact Nat.mul_mod_left _ _
  refine ⟨hsa, hsb, hu, hc, hig, hog, hal, hst, hIn, hins, hpI, hpM, hpO, hands, houts, ⟨hca, hcb⟩, ?_⟩
  generalize layout st inBits outBits outs = l at *
  intro r h1 h2 c hc
  by_cases hA : r < inW * WORD + st.rows.length
  · obtain ⟨k, rfl⟩ : ∃ k, r = inW * WORD + k := ⟨r - inW * WORD, by omega⟩
    have hk : k < st.rows.reverse.length := by simp; omega
    obtain ⟨ha, hb⟩ := hands k hk
    have hmem : st.rows.reverse[k] ∈ st.rows := List.mem_reverse.1 (List.getElem_mem hk)
    obtain ⟨hra, hrb⟩ := hrow _ hmem
    have ho' := hout k hk
    have key : ∀ f : Lin, (∀ v ∈ f.s, v < st.rows.reverse[k].out) → c ∈ cols nIn (inW * WORD) (start + outW * WORD) f →
        c = start + outW * WORD ∨ (c < inW * WORD + k ∧ (c < nIn ∨ inW * WORD ≤ c)) := by
      intro f hf hcf
      rcases mem_cols hcf with h | ⟨v, hv, rfl⟩
      · exact Or.inl h
      · have hv' : (v : Nat) < nIn + k := ho' ▸ hf v hv
        exact Or.inr (col_lt hIn hv')
    rcases Array.mem_append.1 hc with hc | hc
    · rw [ha] at hc; exact key _ hra hc
    · rw [hb] at hc; exact key _ hrb hc
  by_cases hM : r < start
  · rw [(hpM r (by omega) hM).1, (hpM r (by omega) hM).2] at hc
    simp at hc
  by_cases hO : r < start + outs.size
  · obtain ⟨j, rfl⟩ : ∃ j, r = start + j := ⟨r - start, by omega⟩
    have hj : j < outs.size := by omega
    obtain ⟨ha, hb⟩ := houts j hj
    rcases Array.mem_append.1 hc with hc | hc
    · rw [ha] at hc
      rcases mem_cols hc with h | ⟨v, hv, rfl⟩
      · exact Or.inl h
      · have hv' : (v : Nat) < nIn + st.rows.length := hn ▸ ho _ (Array.getElem_mem_toList hj) v hv
        have := col_lt hIn hv'
        omega
    · rw [hb] at hc
      exact Or.inl (List.mem_singleton.1 (Array.mem_toList_iff.2 hc))
  · rw [(hpO r (by omega) h2).1, (hpO r (by omega) h2).2] at hc
    simp at hc

/-- **A built circuit's layout builds its net**: when the builder's rows output the bits after the `nIn` inputs in
creation order, each row reads only bits below its own, each output reads only committed bits and the outputs fit their
group, and the slot is `2 ^ 18` columns with the output group inside it, `Net.ofRows` succeeds on the layout at its own
header fields, with A and B the sparse matrices of its rows. -/
theorem layout_builds (name : String) (st : St) (inBits outBits : Array Nat) (outs : Array Lin)
    (nIn inW outW start : Nat) (hnIn : inBits.foldl (· + ·) 0 = nIn) (hinW : groupWords nIn = inW)
    (houtW : groupWords (outBits.foldl (· + ·) 0) = outW)
    (hstart : (inW * WORD + st.rows.length + outW * WORD - 1) / (outW * WORD) * (outW * WORD) = start)
    (hn : st.next = nIn + st.rows.length)
    (hout : ∀ k (hk : k < st.rows.reverse.length), st.rows.reverse[k].out = nIn + k)
    (hrow : ∀ r ∈ st.rows, (∀ v ∈ r.a.s, v < r.out) ∧ ∀ v ∈ r.b.s, v < r.out)
    (ho : ∀ f ∈ outs.toList, ∀ v ∈ f.s, v < st.next) (hsz : outs.size ≤ outW * WORD)
    (hU : max 7 (log2ceil (start + outW * WORD + 1)) = 18) (hD : start / WORD + outW ≤ 2 ^ 18 / WORD) :
    netOf name (layout st inBits outBits outs) = .ok (built name (start + outW * WORD + 1) ⟨0, inW, inBits⟩
      ⟨start / WORD, outW, outBits⟩ inW (sparseOf (layout st inBits outBits outs).ra)
        (sparseOf (layout st inBits outBits outs).rb)) := by
  obtain ⟨hsa, hsb, hu, hc, hig, hog, hal, hst, hIn, hins, hpI, -, -, -, -, hcon, hord⟩ :=
    layout_facts st inBits outBits outs nIn inW outW start hnIn hinW houtW hstart hn hout hrow ho hsz _ rfl
  generalize layout st inBits outBits outs = l at *
  unfold netOf
  rw [hu, hc, hig, hog]
  exact ofRows_layout name inBits outBits l.ra l.rb nIn inW outW start (start + outW * WORD) hsa hsb hnIn hIn
    (houtW ▸ le_groupWords _) (hinW ▸ isPow2_groupWords _) (houtW ▸ isPow2_groupWords _) hal (by omega) rfl hins hpI
    hcon hord hU hD

end FlockDeploy.Concrete.Nets
