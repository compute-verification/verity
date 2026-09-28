import Proofs.Flock.Soundness.Lowering
import Flock.Net
-- level3 already realizes the lemmas this file's `simp` calls need (`Std.Legacy.Range.mk.congr_simp`), so they are
-- reused rather than declared a second time, which the audit's kernel replay refuses
import Proofs.Flock.Level3.LookupRows

/-!
# The rows the verifier parses, as the audit's `Rows`

The executable's parser (`Flock.Net.parse`) holds every expanded circuit's rows to #144's order
(`Net.checkOrder`, PR #147): a computed row, `inCols ≤ r < constPos`, reads in `A` and in `B` only
the constant or an earlier column that is not input padding (`RowOrder`). `checkOrder_ok` turns an
accepted check into that property, and `Rows.ofOrdered` builds the audit's `Rows` from rows that
have it, with `topo` proved: what #144 named, the parser now checks.
-/

namespace FlockSoundness

open Flock

/-! ## Loops that only continue or throw -/

theorem forIn_list_yield_ok {α ε : Type} {l : List α} {f : α → PUnit → Except ε (ForInStep PUnit)}
    (hf : ∀ a s, f a PUnit.unit = .ok s → s = .yield PUnit.unit)
    (h : forIn l PUnit.unit f = .ok PUnit.unit) : ∀ a ∈ l, f a PUnit.unit = .ok (.yield PUnit.unit) := by
  induction l with
  | nil => intro a ha; cases ha
  | cons x l ih =>
    rw [List.forIn_cons] at h
    rcases hx : f x PUnit.unit with e | s
    · rw [hx] at h; cases h
    · have hs := hf x s hx
      subst hs
      rw [hx] at h
      intro a ha
      rcases List.mem_cons.1 ha with rfl | ha
      · exact hx
      · exact ih h a ha

theorem forIn_array_yield_ok {α ε : Type} {xs : Array α} {f : α → PUnit → Except ε (ForInStep PUnit)}
    (hf : ∀ a s, f a PUnit.unit = .ok s → s = .yield PUnit.unit)
    (h : forIn xs PUnit.unit f = .ok PUnit.unit) : ∀ a ∈ xs.toList, f a PUnit.unit = .ok (.yield PUnit.unit) :=
  forIn_list_yield_ok hf (by rw [Array.forIn_toList]; exact h)

theorem forIn_range_yield_ok {ε : Type} {lo hi : ℕ} {f : ℕ → PUnit → Except ε (ForInStep PUnit)}
    (hf : ∀ a s, f a PUnit.unit = .ok s → s = .yield PUnit.unit)
    (h : forIn [lo:hi] PUnit.unit f = .ok PUnit.unit) :
    ∀ r, lo ≤ r → r < hi → f r PUnit.unit = .ok (.yield PUnit.unit) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  intro r hlo hhi
  refine forIn_list_yield_ok hf h r ?_
  simp only [Std.Legacy.Range.size, List.mem_range'_1]
  omega

/-! ## The order `Net.checkOrder` checks -/

/-- Column `c` is **input padding**: an input column whose rows are empty in `A` and in `B`. -/
def InputPadding (inCols : ℕ) (ra rb : Array (Array ℕ)) (c : ℕ) : Prop :=
  c < inCols ∧ (ra.getD c #[]).isEmpty = true ∧ (rb.getD c #[]).isEmpty = true

/-- **Row order** (#144's `Rows.topo`, as the parser checks it): every computed row
`inCols ≤ r < constPos` reads, in `A` and in `B`, only the constant or an earlier column that is not
input padding. -/
def RowOrder (inCols constPos : ℕ) (ra rb : Array (Array ℕ)) : Prop :=
  ∀ r, inCols ≤ r → r < constPos → ∀ c ∈ (ra.getD r #[]).toList ++ (rb.getD r #[]).toList,
    c = constPos ∨ (c < r ∧ ¬ InputPadding inCols ra rb c)

/-- **The parser's order check, accepted, is `RowOrder`.** -/
theorem checkOrder_ok {name : String} {inCols constPos : ℕ} {ra rb : Array (Array ℕ)}
    (h : Net.checkOrder name inCols constPos ra rb = .ok ()) : RowOrder inCols constPos ra rb := by
  unfold Net.checkOrder at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  · rename_i v hloop
    cases v
    have hout := forIn_range_yield_ok (fun a s hs => by
      split at hs
      · cases hs
      · cases hs; rfl) hloop
    intro r hlo hhi c hc
    have hr := hout r hlo hhi
    split at hr
    · cases hr
    · rename_i w hinner
      cases w
      have hin := forIn_array_yield_ok (fun a s hs => by
        split at hs
        · cases hs; rfl
        · split at hs
          · simp [throw, throwThe, MonadExceptOf.throw] at hs
          · split at hs
            · simp [throw, throwThe, MonadExceptOf.throw] at hs
            · cases hs; rfl) hinner c (by rw [Array.toList_append]; exact hc)
      by_cases hk : c = constPos
      · exact Or.inl hk
      · refine Or.inr ?_
        simp only [hk, beq_iff_eq, ite_false] at hin
        split at hin
        · simp [throw, throwThe, MonadExceptOf.throw] at hin
        · rename_i hlt
          split at hin
          · simp [throw, throwThe, MonadExceptOf.throw] at hin
          · rename_i hpad
            refine ⟨by omega, fun ⟨h1, h2, h3⟩ => hpad ?_⟩
            rw [h2, h3]
            simp [h1]

/-! ## The audit's `Rows` from ordered rows -/

/-- A column a computed row reads is the constant or an earlier column, so below `constPos + 1`. -/
theorem RowOrder.lt {inCols constPos : ℕ} {ra rb : Array (Array ℕ)} (h : RowOrder inCols constPos ra rb)
    {r : ℕ} (hlo : inCols ≤ r) (hhi : r < constPos) {c : ℕ}
    (hc : c ∈ (ra.getD r #[]).toList ++ (rb.getD r #[]).toList) : c = constPos ∨ c < r :=
  (h r hlo hhi c hc).imp id And.left

/-- **The template's rows, from rows the parser ordered**: columns below `min inCols constPos` are
its inputs, rows from there to `constPos − 1` are computed, and column `constPos` is its constant.
(A net whose constant sits among its input rows has no computed rows.) -/
def Rows.ofOrdered (inCols constPos : ℕ) (ra rb : Array (Array ℕ)) (h : RowOrder inCols constPos ra rb) :
    Rows where
  nIn := min inCols constPos
  nComp := constPos - min inCols constPos
  a i := (ra.getD (min inCols constPos + i.val) #[]).toList.pmap (fun c hc => ⟨c, hc⟩) fun c hc => by
    have hi := i.isLt
    rcases h.lt (r := min inCols constPos + i.val) (by omega) (by omega) (List.mem_append_left _ hc) with
      hc' | hc' <;> omega
  b i := (rb.getD (min inCols constPos + i.val) #[]).toList.pmap (fun c hc => ⟨c, hc⟩) fun c hc => by
    have hi := i.isLt
    rcases h.lt (r := min inCols constPos + i.val) (by omega) (by omega) (List.mem_append_right _ hc) with
      hc' | hc' <;> omega
  topo i c hc := by
    have hi := i.isLt
    have hlo : inCols ≤ min inCols constPos + i.val := by omega
    have hhi : min inCols constPos + i.val < constPos := by omega
    have hone : min inCols constPos + (constPos - min inCols constPos) = constPos := by omega
    rcases List.mem_append.1 hc with hc | hc <;>
    · obtain ⟨c₀, hc₀, rfl⟩ := List.mem_pmap.1 hc
      rcases h.lt hlo hhi
          (by first | exact List.mem_append_left _ hc₀ | exact List.mem_append_right _ hc₀) with
        h' | h'
      · exact Or.inr (by show c₀ = _; rw [hone]; exact h')
      · exact Or.inl h'

section ofOrdered

variable {inCols constPos : ℕ} {ra rb : Array (Array ℕ)} {h : RowOrder inCols constPos ra rb}

theorem Rows.ofOrdered_nIn (hle : inCols ≤ constPos) : (Rows.ofOrdered inCols constPos ra rb h).nIn = inCols := by
  show min inCols constPos = inCols
  exact Nat.min_eq_left hle

theorem Rows.ofOrdered_a (i : Fin (Rows.ofOrdered inCols constPos ra rb h).nComp) :
    ((Rows.ofOrdered inCols constPos ra rb h).a i).map Fin.val =
      (ra.getD ((Rows.ofOrdered inCols constPos ra rb h).nIn + i.val) #[]).toList := by
  simp [Rows.ofOrdered, List.map_pmap]

theorem Rows.ofOrdered_b (i : Fin (Rows.ofOrdered inCols constPos ra rb h).nComp) :
    ((Rows.ofOrdered inCols constPos ra rb h).b i).map Fin.val =
      (rb.getD ((Rows.ofOrdered inCols constPos ra rb h).nIn + i.val) #[]).toList := by
  simp [Rows.ofOrdered, List.map_pmap]

theorem Rows.ofOrdered_comp (i : Fin (Rows.ofOrdered inCols constPos ra rb h).nComp) :
    ((Rows.ofOrdered inCols constPos ra rb h).comp i).val =
      (Rows.ofOrdered inCols constPos ra rb h).nIn + i.val := rfl

theorem Rows.ofOrdered_one : (Rows.ofOrdered inCols constPos ra rb h).one.val = constPos := by
  show min inCols constPos + (constPos - min inCols constPos) = constPos
  omega

end ofOrdered

/-! ## What `Net.parse` accepted -/

theorem except_bind_ok {α β ε : Type} {x : Except ε α} {f : α → Except ε β} {v : β}
    (h : (x >>= f) = .ok v) : ∃ a, x = .ok a ∧ f a = .ok v := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem throw_bind_ne {α β : Type} {e : String} {f : α → Except String β} {v : β} :
    (throw e >>= f : Except String β) ≠ .ok v := by
  intro h; cases h

theorem ite_throw_ok {α β : Type} {c : Prop} [Decidable c] {e : String} {f : α → Except String β}
    {g : Except String β} {v : β} (h : (if c then (throw e >>= f) else g) = .ok v) : g = .ok v := by
  by_cases hc : c
  · rw [ite_eq_left_of_eq_true _ _ (eq_true hc)] at h; cases h
  · rwa [ite_eq_right_of_eq_false _ _ (eq_false hc)] at h

theorem parse_tail {α : Type} {x y : Except String Sparse} {mk : Sparse → Sparse → α} {v : α}
    (h : (do let a ← x; let b ← y; pure (mk a b)) = .ok v) : ∃ a b, x = .ok a ∧ y = .ok b ∧ v = mk a b := by
  obtain ⟨a, ha, h⟩ := except_bind_ok h
  obtain ⟨b, hb, h⟩ := except_bind_ok h
  cases h
  exact ⟨a, b, ha, hb, rfl⟩

/-- **A net `Net.ofRows` built**: the parser's (`parse_built`), or a typed class's from `derive`'s rows
(`Typed.netOfD`). -/
def Built (net : Net) : Prop :=
  ∃ name useful constPos ig og ra rb, Net.ofRows name useful constPos ig og ra rb = .ok net

set_option maxHeartbeats 1000000 in
/-- **`Net.parse` builds its net with `Net.ofRows`**, after reading its rows and trailing lines. -/
theorem parse_built {lines : Array String} {net : Net} (h : Net.parse lines = .ok net) : Built net := by
  unfold Net.parse at h
  simp only at h
  replace h := ite_throw_ok h
  obtain ⟨useful, -, h⟩ := except_bind_ok h
  obtain ⟨constPos, -, h⟩ := except_bind_ok h
  obtain ⟨ig, -, h⟩ := except_bind_ok h
  obtain ⟨og, -, h⟩ := except_bind_ok h
  replace h := ite_throw_ok (ite_throw_ok (ite_throw_ok (ite_throw_ok (ite_throw_ok h))))
  obtain ⟨s, -, h⟩ := except_bind_ok h
  refine ⟨(tokens (lines.getD 0 "")).getD 1 "", useful, constPos, ig.1, og.1, s.1, s.2, ?_⟩
  by_cases hL : (lines.getD (1 + useful) "").startsWith "LEAVES " = true
  · rw [ite_eq_left_of_eq_true _ _ (eq_true hL)] at h
    rcases hj : Lean.Json.parse ((lines.getD (1 + useful) "").drop 7).toString with e | j
    · rw [hj] at h; exact absurd h throw_bind_ne
    · rw [hj] at h
      exact ite_throw_ok (ite_throw_ok h)
  · rw [ite_eq_right_of_eq_false _ _ (eq_false hL)] at h
    exact ite_throw_ok (ite_throw_ok h)

/-- **What `Net.ofRows` built**: its order check passed on the very rows it stored as `A` and `B`. -/
theorem built_checkOrder {net : Net} (h : Built net) :
    ∃ ra rb, Net.checkOrder "flock-ir-unit/v2" (net.inWords * WORD) net.constPos ra rb = .ok () ∧
      Sparse.ofRows ra (2 ^ net.unitLog) = .ok net.a ∧ Sparse.ofRows rb (2 ^ net.unitLog) = .ok net.b := by
  obtain ⟨_, _, _, _, _, ra, rb, ho⟩ := h
  obtain ⟨-, ha, hb, -, hc, -, -, -, -, -, hco, -, -, -⟩ := Net.ofRows_ok ho
  exact ⟨ra, rb, by rw [hc]; exact hco, ha, hb⟩

/-- **Every net `Net.ofRows` builds is ordered**: the rows it stored as `A` and `B` have `RowOrder`. -/
theorem built_rowOrder {net : Net} (h : Built net) :
    ∃ ra rb, RowOrder (net.inWords * WORD) net.constPos ra rb ∧
      Sparse.ofRows ra (2 ^ net.unitLog) = .ok net.a ∧ Sparse.ofRows rb (2 ^ net.unitLog) = .ok net.b :=
  let ⟨ra, rb, hco, ha, hb⟩ := built_checkOrder h
  ⟨ra, rb, checkOrder_ok hco, ha, hb⟩

/-- **The parser orders every net it accepts**: the rows it stored as `A` and `B` have `RowOrder`,
so #144's `Rows.topo` for a parsed net is an executable check, not a hypothesis. -/
theorem parse_rowOrder {lines : Array String} {net : Net} (h : Net.parse lines = .ok net) :
    ∃ ra rb, RowOrder (net.inWords * WORD) net.constPos ra rb ∧
      Sparse.ofRows ra (2 ^ net.unitLog) = .ok net.a ∧ Sparse.ofRows rb (2 ^ net.unitLog) = .ok net.b :=
  built_rowOrder (parse_built h)

/-! ## `Sparse`: the bytes `ofRows` writes, and `row` reads them back -/

/-- The four little-endian bytes `pushLe32` appends. -/
def le32 (x : ℕ) : List UInt8 :=
  [(x % 256).toUInt8, ((x >>> 8) % 256).toUInt8, ((x >>> 16) % 256).toUInt8, ((x >>> 24) % 256).toUInt8]

theorem pushLe32_data (b : ByteArray) (x : ℕ) : (pushLe32 b x).data.toList = b.data.toList ++ le32 x := by
  simp [pushLe32, Std.Legacy.Range.forIn_eq_forIn_range', List.range', le32, ByteArray.push, Id.run]
  rw [show ∀ v : ByteArray, (pure v : Id ByteArray) = v from fun _ => rfl]
  simp

theorem getElem!_of_some {a : Array UInt8} {i : ℕ} {v : UInt8} (h : a[i]? = some v) : a[i]! = v := by
  simp [getElem!_def, h]

theorem testBit_mod256 (y k : ℕ) : (y % 256).testBit k = (decide (k < 8) && y.testBit k) :=
  Nat.testBit_mod_two_pow y 8 k

/-- The four bytes of `x < 2^32`, reassembled. -/
theorem le32_or (x : ℕ) (hx : x < 2 ^ 32) :
    (x % 256 ||| (x >>> 8 % 256) <<< 8 ||| (x >>> 16 % 256) <<< 16 ||| (x >>> 24 % 256) <<< 24) = x := by
  apply Nat.eq_of_testBit_eq
  intro n
  simp only [Nat.testBit_or, Nat.testBit_shiftLeft, testBit_mod256, Nat.testBit_shiftRight]
  have hx' : 32 ≤ n → x.testBit n = false := fun hn =>
    Nat.testBit_eq_false_of_lt (lt_of_lt_of_le hx (Nat.pow_le_pow_right (by norm_num) hn))
  rcases Nat.lt_or_ge n 8 with h8 | h8
  · simp [h8, show ¬ 8 ≤ n by omega, show ¬ 16 ≤ n by omega, show ¬ 24 ≤ n by omega]
  rcases Nat.lt_or_ge n 16 with h16 | h16
  · simp [h8, show ¬ n < 8 by omega, show n - 8 < 8 by omega, show 8 + (n - 8) = n by omega,
      show ¬ 16 ≤ n by omega, show ¬ 24 ≤ n by omega]
  rcases Nat.lt_or_ge n 24 with h24 | h24
  · simp [h16, show ¬ n < 8 by omega, show ¬ n - 8 < 8 by omega, show n - 16 < 8 by omega,
      show 16 + (n - 16) = n by omega, show ¬ 24 ≤ n by omega]
  rcases Nat.lt_or_ge n 32 with h32 | h32
  · simp [h24, show ¬ n < 8 by omega, show ¬ n - 8 < 8 by omega, show ¬ n - 16 < 8 by omega,
      show n - 24 < 8 by omega, show 24 + (n - 24) = n by omega]
  · simp [hx' h32, show ¬ n < 8 by omega, show ¬ n - 8 < 8 by omega, show ¬ n - 16 < 8 by omega,
      show ¬ n - 24 < 8 by omega]

/-- `rd32` reads back what `pushLe32` wrote, below `2^32`. -/
theorem rd32_le32 (b : ByteArray) (o x : ℕ) (hx : x < 2 ^ 32)
    (hb : ∀ k (hk : k < 4), b.data[o + k]? = some ((le32 x).get ⟨k, by simp [le32]; omega⟩)) :
    rd32 b o = x := by
  have h0 := hb 0 (by omega)
  have h1 := hb 1 (by omega)
  have h2 := hb 2 (by omega)
  have h3 := hb 3 (by omega)
  simp only [Nat.add_zero] at h0
  simp only [rd32, ByteArray.get!, getElem!_of_some h0, getElem!_of_some h1, getElem!_of_some h2,
    getElem!_of_some h3]
  simp only [le32, List.get_eq_getElem, List.getElem_cons_zero, List.getElem_cons_succ]
  simpa using le32_or x hx

/-! ## `Sparse.ofRows`, read back -/

/-- The bytes `ofRows` writes for one row. -/
def encRow (r : Array ℕ) : List UInt8 := r.toList.flatMap le32

theorem le32_length (x : ℕ) : (le32 x).length = 4 := rfl

theorem encRow_length (r : Array ℕ) : (encRow r).length = 4 * r.size := by
  simp only [encRow, List.length_flatMap, le32_length, List.map_const', List.sum_replicate, smul_eq_mul,
    Array.length_toList]
  ring

/-- Where each row starts in the data, from byte `base`. -/
def rowStarts : ℕ → List (Array ℕ) → List ℕ
  | _, [] => []
  | base, r :: L => base :: rowStarts (base + 4 * r.size) L

theorem forIn_cols_ok {width : ℕ} {f : ℕ → ByteArray → Except String (ForInStep ByteArray)}
    (hf : ∀ c s v, f c s = .ok v → c < width ∧ v = .yield (pushLe32 s c)) :
    ∀ (l : List ℕ) (d d' : ByteArray), forIn l d f = .ok d' →
      d'.data.toList = d.data.toList ++ l.flatMap le32 ∧ ∀ c ∈ l, c < width := by
  intro l
  induction l with
  | nil => intro d d' h; cases h; simp
  | cons c l ih =>
    intro d d' h
    rw [List.forIn_cons] at h
    rcases hc : f c d with e | v
    · rw [hc] at h; cases h
    · obtain ⟨hlt, rfl⟩ := hf c d v hc
      rw [hc] at h
      obtain ⟨h1, h2⟩ := ih _ _ h
      refine ⟨by rw [h1, pushLe32_data]; simp, ?_⟩
      intro x hx
      rcases List.mem_cons.1 hx with rfl | hx
      · exact hlt
      · exact h2 x hx

theorem forIn_rows_ok {width : ℕ}
    {F : Array ℕ → ByteArray × Array ℕ × Array ℕ → Except String (ForInStep (ByteArray × Array ℕ × Array ℕ))}
    (hF : ∀ r st v, F r st = .ok v → ∃ d', d'.data.toList = st.1.data.toList ++ encRow r ∧
      (∀ c ∈ r.toList, c < width) ∧ v = .yield (d', st.2.1.push st.1.size, st.2.2.push r.size)) :
    ∀ (L : List (Array ℕ)) (st st' : ByteArray × Array ℕ × Array ℕ), forIn L st F = .ok st' →
      st'.1.data.toList = st.1.data.toList ++ L.flatMap encRow ∧
      st'.2.1.toList = st.2.1.toList ++ rowStarts st.1.size L ∧
      st'.2.2.toList = st.2.2.toList ++ L.map Array.size ∧
      ∀ r ∈ L, ∀ c ∈ r.toList, c < width := by
  intro L
  induction L with
  | nil => intro st st' h; cases h; simp [rowStarts]
  | cons r L ih =>
    intro st st' h
    rw [List.forIn_cons] at h
    rcases hr : F r st with e | v
    · rw [hr] at h; cases h
    · obtain ⟨d', hd, hw, rfl⟩ := hF r st v hr
      rw [hr] at h
      obtain ⟨h1, h2, h3, h4⟩ := ih _ _ h
      have hsize : d'.size = st.1.size + 4 * r.size := by
        have := congrArg List.length hd
        simp only [List.length_append, Array.length_toList, encRow_length] at this
        exact this
      refine ⟨by rw [h1, hd]; simp, ?_, by rw [h3]; simp, ?_⟩
      · rw [h2]; simp [rowStarts, hsize]
      · intro x hx c hc
        rcases List.mem_cons.1 hx with rfl | hx
        · exact hw c hc
        · exact h4 x hx c hc

/-- **What `Sparse.ofRows` built**: the rows' `le32` bytes in order, each row's start and length, and
every column below the width. -/
theorem ofRows_spec {rows : Array (Array ℕ)} {width : ℕ} {m : Sparse} (h : Sparse.ofRows rows width = .ok m) :
    m.data.data.toList = rows.toList.flatMap encRow ∧ m.start.toList = rowStarts 0 rows.toList ∧
      m.len.toList = rows.toList.map Array.size ∧ ∀ r ∈ rows.toList, ∀ c ∈ r.toList, c < width := by
  unfold Sparse.ofRows at h
  simp only at h
  obtain ⟨st, hst, h⟩ := except_bind_ok h
  cases h
  rw [← Array.forIn_toList] at hst
  have := forIn_rows_ok (width := width) (fun r st v hv => by
    obtain ⟨d, hd, hv⟩ := except_bind_ok hv
    cases hv
    rw [← Array.forIn_toList] at hd
    obtain ⟨h1, h2⟩ := forIn_cols_ok (width := width) (fun c s v hv => by
      by_cases hc : c ≥ width
      · rw [ite_eq_left_of_eq_true _ _ (eq_true hc)] at hv; cases hv
      · rw [ite_eq_right_of_eq_false _ _ (eq_false hc)] at hv
        cases hv
        exact ⟨by omega, rfl⟩) _ _ _ hd
    exact ⟨d, by rw [h1]; rfl, h2, rfl⟩) _ _ _ hst
  obtain ⟨h1, h2, h3, h4⟩ := this
  exact ⟨by simpa using h1, by simpa using h2, by simpa using h3, h4⟩

theorem rowStarts_getElem? : ∀ (L : List (Array ℕ)) (base i : ℕ), i < L.length →
    (rowStarts base L)[i]? = some (base + 4 * ((L.take i).map Array.size).sum)
  | [], _, _, h => absurd h (by simp)
  | r :: L, base, 0, _ => by simp [rowStarts]
  | r :: L, base, i + 1, h => by
      simp only [rowStarts, List.getElem?_cons_succ]
      rw [rowStarts_getElem? L _ i (by simpa using h)]
      simp only [List.take_succ_cons, List.map_cons, List.sum_cons]
      congr 1
      ring

theorem rowStarts_length : ∀ (L : List (Array ℕ)) (base : ℕ), (rowStarts base L).length = L.length
  | [], _ => rfl
  | r :: L, base => by simp [rowStarts, rowStarts_length L]

theorem flatMap_le32_getElem? : ∀ (l : List ℕ) (t k : ℕ), t < l.length → k < 4 →
    (l.flatMap le32)[4 * t + k]? = (le32 (l.getD t 0))[k]?
  | [], _, _, h, _ => absurd h (by simp)
  | x :: l, 0, k, _, hk => by
      simp only [List.flatMap_cons, Nat.mul_zero, Nat.zero_add, List.getD_cons_zero]
      rw [List.getElem?_append_left (by rw [le32_length]; exact hk)]
  | x :: l, t + 1, k, ht, hk => by
      simp only [List.flatMap_cons, List.getD_cons_succ]
      rw [List.getElem?_append_right (by rw [le32_length]; omega), le32_length,
        show 4 * (t + 1) + k - 4 = 4 * t + k by omega]
      exact flatMap_le32_getElem? l t k (by simpa using ht) hk

theorem flatMap_encRow_length (L : List (Array ℕ)) :
    (L.flatMap encRow).length = 4 * (L.map Array.size).sum := by
  induction L with
  | nil => rfl
  | cons r L ih => simp [List.flatMap_cons, encRow_length, ih]; ring

/-- **`Sparse.ofRows` reads back its rows**: row `i` of what it built is `rows[i]` (empty past the
last row), when every column is below `2^32`. -/
theorem ofRows_row {rows : Array (Array ℕ)} {width : ℕ} {m : Sparse} (h : Sparse.ofRows rows width = .ok m)
    (hw : width ≤ 2 ^ 32) (i : ℕ) : m.row i = rows.getD i #[] := by
  obtain ⟨hd, hs, hl, hc⟩ := ofRows_spec h
  have hlen : m.len.getD i 0 = (rows.getD i #[]).size := by
    simp only [Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, hl, List.getElem?_map]
    cases rows.toList[i]? <;> rfl
  by_cases hi : i < rows.size
  · have hri : rows.getD i #[] = rows[i] := by simp [Array.getD_eq_getD_getElem?, hi]
    have hst : m.start.getD i 0 = 4 * ((rows.toList.take i).map Array.size).sum := by
      simp only [Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, hs]
      rw [rowStarts_getElem? _ 0 i (by simpa using hi)]
      simp
    apply Array.ext
    · simp [Sparse.row, hlen]
    · intro t h1 h2
      simp only [Sparse.row, Array.getElem_map, Array.getElem_range]
      have ht : t < rows[i].size := by rw [hri] at h2; exact h2
      simp only [hri]
      have hlt : rows[i][t] < 2 ^ 32 :=
        lt_of_lt_of_le (hc rows[i] (Array.getElem_mem_toList hi) _ (Array.getElem_mem_toList ht)) hw
      refine rd32_le32 _ _ _ hlt fun k hk => ?_
      rw [hst]
      have hsplit : rows.toList.flatMap encRow = (rows.toList.take i).flatMap encRow ++
          (encRow rows[i] ++ (rows.toList.drop (i + 1)).flatMap encRow) := by
        conv_lhs => rw [← List.take_append_drop i rows.toList]
        rw [List.flatMap_append, List.drop_eq_getElem_cons (by simpa using hi), List.flatMap_cons]
        simp
      rw [← Array.getElem?_toList, hd, hsplit, List.getElem?_append_right (by rw [flatMap_encRow_length]; omega),
        flatMap_encRow_length, show 4 * ((rows.toList.take i).map Array.size).sum + 4 * t + k -
          4 * ((rows.toList.take i).map Array.size).sum = 4 * t + k by omega,
        List.getElem?_append_left (by rw [encRow_length]; omega), encRow,
        flatMap_le32_getElem? _ t k (by simpa using ht) hk]
      simp [ht]
  · have hri : rows.getD i #[] = #[] := by simp [Array.getD_eq_getD_getElem?, hi]
    rw [hri] at hlen ⊢
    simp [Sparse.row, hlen]

/-! ## The audit's `Rows` for a built net -/

/-- **The rows of a net `Net.ofRows` built** (the parser's, or a typed class's), as the audit's `Rows`: its computed rows
are `net.a.row r`, `net.b.row r` (`Rows.ofNet_a`, `Rows.ofNet_b`), and `topo` holds because `Net.ofRows` checked it
(`built_rowOrder`). -/
noncomputable def Rows.ofNet {net : Net} (h : Built net) : Rows :=
  Rows.ofOrdered (net.inWords * WORD) net.constPos (Classical.choose (built_rowOrder h))
    (Classical.choose (Classical.choose_spec (built_rowOrder h)))
    (Classical.choose_spec (Classical.choose_spec (built_rowOrder h))).1

section ofNet

variable {net : Net} (h : Built net)

theorem Rows.ofNet_nIn : (Rows.ofNet h).nIn = min (net.inWords * WORD) net.constPos := rfl

theorem Rows.ofNet_one : (Rows.ofNet h).one.val = net.constPos := Rows.ofOrdered_one

theorem Rows.ofNet_comp (i : Fin (Rows.ofNet h).nComp) :
    ((Rows.ofNet h).comp i).val = (Rows.ofNet h).nIn + i.val := rfl

/-- Computed row `i` reads, in `A`, exactly the columns of the net's row `nIn + i`. -/
theorem Rows.ofNet_a (hu : net.unitLog ≤ 32) (i : Fin (Rows.ofNet h).nComp) :
    ((Rows.ofNet h).a i).map Fin.val = (net.a.row ((Rows.ofNet h).nIn + i.val)).toList := by
  have hs := Classical.choose_spec (Classical.choose_spec (built_rowOrder h))
  refine (Rows.ofOrdered_a i).trans ?_
  rw [ofRows_row hs.2.1 (Nat.pow_le_pow_right (by norm_num) hu)]
  rfl

/-- Computed row `i` reads, in `B`, exactly the columns of the net's row `nIn + i`. -/
theorem Rows.ofNet_b (hu : net.unitLog ≤ 32) (i : Fin (Rows.ofNet h).nComp) :
    ((Rows.ofNet h).b i).map Fin.val = (net.b.row ((Rows.ofNet h).nIn + i.val)).toList := by
  have hs := Classical.choose_spec (Classical.choose_spec (built_rowOrder h))
  refine (Rows.ofOrdered_b i).trans ?_
  rw [ofRows_row hs.2.2 (Nat.pow_le_pow_right (by norm_num) hu)]
  rfl

end ofNet

/-! ## An instance's rows: `k` copies of the template's, stacked -/

namespace Rows

variable (R : Rows) (k : ℕ)

theorem mul_add_lt {u k n c : ℕ} (hu : u < k) (hc : c < n) : u * n + c < k * n := by
  have : u * n + n ≤ k * n := by
    calc u * n + n = (u + 1) * n := by ring
      _ ≤ k * n := Nat.mul_le_mul_right _ hu
  omega

/-- The columns of `k` stacked copies. -/
abbrev stackW : ℕ := k * R.nIn + k * (R.nComp + 1) + 1

/-- **Copy `u`'s column `c`, in the stacked order**: its input `c` is input `u·nIn + c`; its computed
row `c − nIn` follows the copy's constant row; its constant is the copy's constant row. -/
def stackCol {u : ℕ} (hu : u < k) (c : Fin R.w) : Fin (R.stackW k) :=
  if hc : c.val < R.nIn then ⟨u * R.nIn + c.val, by
    have := mul_add_lt hu hc; show _ < k * R.nIn + k * (R.nComp + 1) + 1; omega⟩
  else if hc' : c.val < R.nIn + R.nComp then
    ⟨k * R.nIn + (u * (R.nComp + 1) + (1 + (c.val - R.nIn))), by
      have := mul_add_lt hu (show 1 + (c.val - R.nIn) < R.nComp + 1 by omega)
      show _ < k * R.nIn + k * (R.nComp + 1) + 1; omega⟩
  else ⟨k * R.nIn + (u * (R.nComp + 1) + 0), by
      have := mul_add_lt hu (show 0 < R.nComp + 1 by omega)
      show _ < k * R.nIn + k * (R.nComp + 1) + 1; omega⟩

theorem stack_copy_lt {i : ℕ} (hi : i < k * (R.nComp + 1)) : i / (R.nComp + 1) < k :=
  (Nat.div_lt_iff_lt_mul (by omega)).2 hi

/-- Row `i` of the stack, from a copy's rows `f`: the copy's constant row reads the shared constant,
and the copy's computed row `j` reads its own columns. -/
def stackRow (f : Fin R.nComp → List (Fin R.w)) (i : Fin (k * (R.nComp + 1))) : List (Fin (R.stackW k)) :=
  if hj : i.val % (R.nComp + 1) = 0 then [⟨k * R.nIn + k * (R.nComp + 1), by
    show _ < k * R.nIn + k * (R.nComp + 1) + 1; omega⟩]
  else (f ⟨i.val % (R.nComp + 1) - 1, by have := Nat.mod_lt i.val (show 0 < R.nComp + 1 by omega); omega⟩).map
    (R.stackCol k (R.stack_copy_lt k i.isLt))

theorem stackRow_topo (f : Fin R.nComp → List (Fin R.w))
    (hf : ∀ j, ∀ c ∈ f j, c.val < R.nIn + j.val ∨ c.val = R.nIn + R.nComp)
    (i : Fin (k * (R.nComp + 1))) (c : Fin (R.stackW k)) (hc : c ∈ R.stackRow k f i) :
    c.val < k * R.nIn + i.val ∨ c.val = k * R.nIn + k * (R.nComp + 1) := by
  have hdm := Nat.div_add_mod' i.val (R.nComp + 1)
  have hmod := Nat.mod_lt i.val (show 0 < R.nComp + 1 by omega)
  unfold stackRow at hc
  split at hc
  · simp only [List.mem_singleton] at hc
    exact Or.inr (by rw [hc])
  · rename_i hj
    obtain ⟨c₀, hc₀, rfl⟩ := List.mem_map.1 hc
    refine Or.inl ?_
    have ht := hf _ c₀ hc₀
    simp only at ht
    unfold stackCol
    split
    · rename_i h1
      have := mul_add_lt (R.stack_copy_lt k i.isLt) h1
      simp only
      omega
    · split
      · simp only
        omega
      · simp only
        omega

/-- **An instance's rows**: `k` copies of the template's rows `R`, in the order `Rows` needs. First
every copy's inputs; then, copy by copy, the copy's constant (a computed row reading the shared
constant) and its computed rows; then the shared constant. -/
def stack : Rows where
  nIn := k * R.nIn
  nComp := k * (R.nComp + 1)
  a := R.stackRow k R.a
  b := R.stackRow k R.b
  topo i c hc := by
    rcases List.mem_append.1 hc with hc | hc
    · exact R.stackRow_topo k R.a (fun j c hc => R.topo j c (List.mem_append_left _ hc)) i c hc
    · exact R.stackRow_topo k R.b (fun j c hc => R.topo j c (List.mem_append_right _ hc)) i c hc

end Rows

/-! ## An instance placed in a statement -/

section Place

variable (S : Model.Statement) (R : Rows) (k : ℕ) (base : ℕ → ℕ)
  (hb : ∀ u < k, base u + (R.nIn + R.nComp) < 2 ^ S.kLog)

/-- Copy `u`'s column `c` in the block: `base u + c`. -/
def posIn {u : ℕ} (hu : u < k) (c : Fin R.w) : Fin (2 ^ S.kLog) :=
  ⟨base u + c.val, by have := hb u hu; have := c.isLt; simp only [Rows.w] at this; omega⟩

/-- **Where the stacked columns sit in the block**: copy `u`'s inputs and computed rows at
`base u + c`, its constant row at its slot's constant `base u + (nIn + nComp)`, and the shared
constant at the pin. -/
def stackPlace (x : Fin (R.stackW k)) : Fin (2 ^ S.kLog) :=
  if h1 : x.val < k * R.nIn then
    ⟨base (x.val / R.nIn) + x.val % R.nIn, by
      have hn : 0 < R.nIn := Nat.pos_of_ne_zero fun h => by simp [h] at h1
      have hu : x.val / R.nIn < k := (Nat.div_lt_iff_lt_mul hn).2 h1
      have := hb _ hu; have := Nat.mod_lt x.val hn; omega⟩
  else if h2 : x.val < k * R.nIn + k * (R.nComp + 1) then
    ⟨base ((x.val - k * R.nIn) / (R.nComp + 1)) +
        (if (x.val - k * R.nIn) % (R.nComp + 1) = 0 then R.nIn + R.nComp
          else R.nIn + ((x.val - k * R.nIn) % (R.nComp + 1) - 1)), by
      have hu : (x.val - k * R.nIn) / (R.nComp + 1) < k := (Nat.div_lt_iff_lt_mul (by omega)).2 (by omega)
      have := hb _ hu
      have := Nat.mod_lt (x.val - k * R.nIn) (show 0 < R.nComp + 1 by omega)
      split <;> omega⟩
  else S.pin

theorem div_mod_copy {u n c : ℕ} (hc : c < n) : (u * n + c) / n = u ∧ (u * n + c) % n = c := by
  have hn : 0 < n := by omega
  refine ⟨?_, ?_⟩
  · rw [Nat.add_comm, Nat.add_mul_div_right _ _ hn, Nat.div_eq_of_lt hc, Nat.zero_add]
  · rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hc]

/-- A copy's column, stacked and then placed, is `base u + c`. -/
theorem stackPlace_stackCol {u : ℕ} (hu : u < k) (c : Fin R.w) :
    stackPlace S R k base hb (R.stackCol k hu c) = posIn S R k base hb hu c := by
  have hc := c.isLt
  simp only [Rows.w] at hc
  unfold Rows.stackCol
  split
  · rename_i h
    have hlt := Rows.mul_add_lt hu h
    obtain ⟨hd, hm⟩ := div_mod_copy (u := u) h
    unfold stackPlace
    rw [dite_eq_left (show u * R.nIn + c.val < k * R.nIn from hlt)]
    simp only [posIn, hd, hm]
  · split
    · rename_i h h'
      have hlt := Rows.mul_add_lt hu (show 1 + (c.val - R.nIn) < R.nComp + 1 by omega)
      obtain ⟨hd, hm⟩ := div_mod_copy (u := u) (show 1 + (c.val - R.nIn) < R.nComp + 1 by omega)
      unfold stackPlace
      simp only
      rw [dite_eq_right (by omega), dite_eq_left (by omega)]
      simp only [posIn, Nat.add_sub_cancel_left, hd, hm, show 1 + (c.val - R.nIn) ≠ 0 by omega, ite_false]
      congr 2
      omega
    · rename_i h h'
      have hlt := Rows.mul_add_lt hu (show 0 < R.nComp + 1 by omega)
      obtain ⟨hd, hm⟩ := div_mod_copy (u := u) (show 0 < R.nComp + 1 by omega)
      unfold stackPlace
      simp only
      rw [dite_eq_right (by omega), dite_eq_left (by omega)]
      simp only [posIn, Nat.add_sub_cancel_left, hd, hm, ite_true]
      congr 2
      omega

theorem stackPlace_last (h : k * R.nIn + k * (R.nComp + 1) < R.stackW k) :
    stackPlace S R k base hb ⟨k * R.nIn + k * (R.nComp + 1), h⟩ = S.pin := by
  unfold stackPlace
  rw [dite_eq_right (by simp), dite_eq_right (by simp)]

theorem stackPlace_one : stackPlace S R k base hb (R.stack k).one = S.pin :=
  stackPlace_last S R k base hb _

/-- Stacked computed row `i` of copy `u = i / (nComp + 1)` sits at the copy's constant when it is the
copy's constant row, and at the copy's computed row otherwise. -/
theorem stackPlace_comp (i : Fin (R.stack k).nComp) :
    stackPlace S R k base hb ((R.stack k).comp i) =
      if hj : i.val % (R.nComp + 1) = 0 then posIn S R k base hb (R.stack_copy_lt k i.isLt) R.one
      else posIn S R k base hb (R.stack_copy_lt k i.isLt)
        (R.comp ⟨i.val % (R.nComp + 1) - 1, by
          have := Nat.mod_lt i.val (show 0 < R.nComp + 1 by omega); omega⟩) := by
  have hi : i.val < k * (R.nComp + 1) := i.isLt
  have hc : ((R.stack k).comp i).val = k * R.nIn + i.val := rfl
  have hn : (R.stack k).nIn = k * R.nIn := rfl
  unfold stackPlace
  rw [dite_eq_right (by omega), dite_eq_left (by omega)]
  apply Fin.ext
  by_cases hj : i.val % (R.nComp + 1) = 0
  · rw [dite_eq_left hj]
    simp only [Rows.comp, hn, Nat.add_sub_cancel_left, hj, ite_true, posIn, Rows.one]
  · rw [dite_eq_right hj]
    simp only [Rows.comp, hn, Nat.add_sub_cancel_left, hj, ite_false, posIn]

/-- **An instance's rows placed in a statement.** If, at every copy `u`'s computed rows, the level-0
matrices read the template's rows shifted to the copy's slot (`hrowA`, `hrowB`), and every copy's
constant row and the pin's own row read the pin (`hconstA`, …), then the stacked rows are placed at
`stackPlace`. -/
theorem placement_stack
    (hrowA : ∀ u (hu : u < k) (i : Fin R.nComp) (j : Fin (2 ^ S.kLog)),
      S.A₀ (posIn S R k base hb hu (R.comp i)) j = (((R.a i).map (posIn S R k base hb hu)).count j : ZMod 2))
    (hrowB : ∀ u (hu : u < k) (i : Fin R.nComp) (j : Fin (2 ^ S.kLog)),
      S.B₀ (posIn S R k base hb hu (R.comp i)) j = (((R.b i).map (posIn S R k base hb hu)).count j : ZMod 2))
    (hconstA : ∀ u (hu : u < k) (j : Fin (2 ^ S.kLog)),
      S.A₀ (posIn S R k base hb hu R.one) j = if j = S.pin then 1 else 0)
    (hconstB : ∀ u (hu : u < k) (j : Fin (2 ^ S.kLog)),
      S.B₀ (posIn S R k base hb hu R.one) j = if j = S.pin then 1 else 0)
    (hpinA : ∀ j, S.A₀ S.pin j = if j = S.pin then 1 else 0)
    (hpinB : ∀ j, S.B₀ S.pin j = if j = S.pin then 1 else 0) :
    Placement S (R.stack k) (stackPlace S R k base hb) := by
  have key : ∀ (M : Matrix (Fin (2 ^ S.kLog)) (Fin (2 ^ S.kLog)) (ZMod 2)) (f : Fin R.nComp → List (Fin R.w))
      (hrow : ∀ u (hu : u < k) (i : Fin R.nComp) (j : Fin (2 ^ S.kLog)),
        M (posIn S R k base hb hu (R.comp i)) j = (((f i).map (posIn S R k base hb hu)).count j : ZMod 2))
      (hconst : ∀ u (hu : u < k) (j : Fin (2 ^ S.kLog)),
        M (posIn S R k base hb hu R.one) j = if j = S.pin then 1 else 0)
      (i : Fin (R.stack k).nComp) (j : Fin (2 ^ S.kLog)),
      M (stackPlace S R k base hb ((R.stack k).comp i)) j =
        (((R.stackRow k f i).map (stackPlace S R k base hb)).count j : ZMod 2) := by
    intro M f hrow hconst i j
    rw [stackPlace_comp]
    unfold Rows.stackRow
    split
    · rename_i hj
      rw [hconst]
      simp only [List.map_cons, List.map_nil, stackPlace_last, List.count_singleton]
      by_cases h : j = S.pin
      · simp [h]
      · simp [h, Ne.symm h]
    · rename_i hj
      rw [hrow, List.map_map]
      congr 3
      funext c
      exact (stackPlace_stackCol S R k base hb _ c).symm
  exact {
    hA := key S.A₀ R.a hrowA hconstA
    hB := key S.B₀ R.b hrowB hconstB
    hA₁ := fun j => by rw [stackPlace_one]; exact hpinA j
    hB₁ := fun j => by rw [stackPlace_one]; exact hpinB j }

end Place

end FlockSoundness