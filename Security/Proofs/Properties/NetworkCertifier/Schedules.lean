import Properties.NetworkCertifier.Schedules
import Proofs.Properties.NetworkCertifier.RowsStep

/-!
# The schedule the rows audit accepts fits the rows format

`schedulesCoveredAt` proves `Properties.NetworkCertifier.SchedulesCoveredAt`. An accepted run's egress record has
`H = W·T` rows of at most `S = lanes·r` cells (`framesOf`), each `r` of them a connection (`connsOf`), and admits the
ordinals `1 … N` with `N` at most the connections its rows hold, so at most `lanes·W·T`. Each row is the one
`expectedRows` renders from the schedule and data it shows, so connection `i` holds `r` cells of bucket `b` exactly when
`a_i ≤ b < a_i + D + k_i`, `k_i` its lane's buckets (`rowsAt_length`), and its data, its cells but the padding, fit the
buckets from `a_i + D` to `H` (`data_le`): its lane ends in the run. A bucket of the run then holds its row's cells, at
most `S`; one of the `D` after it at most the last's; and one later only the cells of one connection past the run, which
`wardenOf` admits `D` buckets apart. `rows_step` (`RowsStep.lean`) walks the audit's `do` blocks as Flock's `walk_step`
does, and names each bound value after its binder.

The bound holds for every well-formed format of the period's connections and `T`-bucket windows (`auditCovered`).
`Fits` reads only a format's connections and its windows, so the network certifier's own format, `cellRows`, fits a
schedule exactly when `unaryFormat`, a well-formed one, does (`cellRows_fits_iff`), and the proof reads it there.
-/

open NetworkCertifier.Rows

namespace Properties.NetworkCertifier.RowsAudit

/-! ### Lists, folds and sums -/

theorem list_mapM_forall₂ {α β ε : Type} {f : α → Except ε β} :
    ∀ {l : List α} {l' : List β}, l.mapM f = .ok l' → List.Forall₂ (fun a b => f a = .ok b) l l'
  | [], l', h => by
    simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact .nil
  | a :: l, l', h => by
    rw [List.mapM_cons] at h
    obtain ⟨b, hb, h⟩ := ebind_ok h
    obtain ⟨bs, hbs, h⟩ := ebind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact .cons hb (list_mapM_forall₂ hbs)

theorem array_mapM_forall₂ {α β ε : Type} {f : α → Except ε β} {xs : Array α} {ys : Array β}
    (h : xs.mapM f = .ok ys) : List.Forall₂ (fun a b => f a = .ok b) xs.toList ys.toList := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : xs.toList.mapM f with
  | error e => rw [hl] at h; cases h
  | ok l =>
    rw [hl] at h
    simp only [Functor.map, Except.map, Except.ok.injEq] at h
    subst h
    simpa using list_mapM_forall₂ hl

theorem forall₂_length {α β : Type} {R : α → β → Prop} {l : List α} {l' : List β} (h : List.Forall₂ R l l') :
    l.length = l'.length := by
  induction h with
  | nil => rfl
  | cons _ _ ih => simp [ih]

theorem forall₂_mem_right {α β : Type} {R : α → β → Prop} {l : List α} {l' : List β}
    (h : List.Forall₂ R l l') : ∀ b ∈ l', ∃ a ∈ l, R a b := by
  induction h with
  | nil => intro b hb; simp at hb
  | cons hab _ ih =>
    intro b hb
    rcases List.mem_cons.1 hb with rfl | hb
    · exact ⟨_, List.mem_cons_self, hab⟩
    · obtain ⟨a, ha, hr⟩ := ih b hb
      exact ⟨a, List.mem_cons_of_mem _ ha, hr⟩

theorem option_list_mapM_length {α β : Type} {f : α → Option β} :
    ∀ {l : List α} {l' : List β}, l.mapM f = some l' → l'.length = l.length
  | [], l', h => by
    rw [List.mapM_nil] at h
    cases h
    rfl
  | a :: l, l', h => by
    rw [List.mapM_cons] at h
    cases hb : f a with
    | none => rw [hb] at h; cases h
    | some b =>
      rw [hb] at h
      cases hl : l.mapM f with
      | none => rw [hl] at h; cases h
      | some bs =>
        rw [hl] at h
        have h' : some (b :: bs) = some l' := h
        cases h'
        simp [option_list_mapM_length hl]

theorem option_array_mapM_size {α β : Type} {f : α → Option β} {xs : Array α} {ys : Array β}
    (h : xs.mapM f = some ys) : ys.size = xs.size := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : xs.toList.mapM f with
  | none => rw [hl] at h; cases h
  | some l =>
    rw [hl] at h
    have h' : some l.toArray = some ys := h
    cases h'
    simp [option_list_mapM_length hl]

theorem option_list_mapM_id {α : Type} :
    ∀ {l : List (Option α)} {l' : List α}, l.mapM (fun a => a) = some l' → l = l'.map some
  | [], l', h => by
    rw [List.mapM_nil] at h
    cases h
    rfl
  | a :: l, l', h => by
    rw [List.mapM_cons] at h
    cases a with
    | none => cases h
    | some b =>
      cases hl : l.mapM (fun a => a) with
      | none => rw [hl] at h; cases h
      | some bs =>
        rw [hl] at h
        have h' : some (b :: bs) = some l' := h
        cases h'
        rw [option_list_mapM_id hl, List.map_cons]

theorem option_array_mapM_id {α : Type} {xs : Array (Option α)} {ys : Array α}
    (h : xs.mapM (m := Option) (fun a => a) = some ys) : xs = ys.map some := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : xs.toList.mapM (fun a => a) with
  | none => rw [hl] at h; cases h
  | some l =>
    rw [hl] at h
    have h' : some l.toArray = some ys := h
    cases h'
    apply Array.ext'
    rw [option_list_mapM_id hl]
    simp

theorem foldl_inv {α β : Type} (P : β → Prop) (f : β → α → β) (hf : ∀ b a, P b → P (f b a)) :
    ∀ (l : List α) (b : β), P b → P (l.foldl f b)
  | [], _, hb => hb
  | a :: l, b, hb => foldl_inv P f hf l (f b a) (hf b a hb)

theorem foldl_inv_mem {α β : Type} (P : β → Prop) (f : β → α → β) :
    ∀ (l : List α), (∀ b a, a ∈ l → P b → P (f b a)) → ∀ b, P b → P (l.foldl f b)
  | [], _, _, hb => hb
  | a :: l, hf, b, hb =>
    foldl_inv_mem P f l (fun b' a' ha' hb' => hf b' a' (List.mem_cons_of_mem _ ha') hb') (f b a)
      (hf b a List.mem_cons_self hb)

theorem foldl_sum {α β : Type} (f : β → α → β) (m : β → ℕ) (w : α → ℕ) (P : β → Prop)
    (hf : ∀ b a, P b → P (f b a) ∧ m (f b a) = m b + w a) :
    ∀ (l : List α) (b : β), P b → P (l.foldl f b) ∧ m (l.foldl f b) = m b + (l.map w).sum
  | [], _, hb => ⟨hb, by simp⟩
  | a :: l, b, hb => by
    obtain ⟨h1, h2⟩ := hf b a hb
    obtain ⟨h3, h4⟩ := foldl_sum f m w P hf l (f b a) h1
    refine ⟨h3, ?_⟩
    rw [List.foldl_cons, h4, h2, List.map_cons, List.sum_cons]
    omega

theorem foldl_size_le {α : Type} (k : ℕ) (xs : Array (Array α)) (hk : ∀ x ∈ xs.toList, x.size ≤ k) :
    xs.foldl (fun a x => a + x.size) 0 ≤ k * xs.size := by
  rw [← Array.foldl_toList, ← Array.length_toList]
  suffices ∀ (l : List (Array α)) (acc : ℕ), (∀ x ∈ l, x.size ≤ k) →
      l.foldl (fun a x => a + x.size) acc ≤ acc + k * l.length by
    simpa using this xs.toList 0 hk
  intro l
  induction l with
  | nil => intro acc _; simp
  | cons x l ih =>
    intro acc h
    have := ih (acc + x.size) (fun y hy => h y (List.mem_cons_of_mem _ hy))
    have hx := h x List.mem_cons_self
    simp only [List.foldl_cons, List.length_cons, Nat.mul_add, Nat.mul_one]
    omega

theorem length_flatten_const {α : Type} {k : ℕ} :
    ∀ {L : List (List α)}, (∀ y ∈ L, y.length = k) → L.flatten.length = L.length * k
  | [], _ => by simp
  | y :: L, h => by
    have hy := h y List.mem_cons_self
    have ih := length_flatten_const (fun z hz => h z (List.mem_cons_of_mem _ hz))
    simp only [List.flatten_cons, List.length_append, List.length_cons, Nat.add_mul, Nat.one_mul, hy, ih]
    omega

theorem countP_eq_sum_map {α : Type} (p : α → Bool) :
    ∀ l : List α, l.countP p = (l.map fun a => if p a then 1 else 0).sum
  | [] => by simp
  | a :: l => by
    rw [List.countP_cons, countP_eq_sum_map p l, List.map_cons, List.sum_cons, Nat.add_comm]

theorem countP_const {α : Type} (c : Bool) (l : List α) : l.countP (fun _ => c) = if c then l.length else 0 := by
  cases c <;> simp

theorem sum_map_le {α : Type} {f g : α → ℕ} : ∀ {l : List α}, (∀ a ∈ l, f a ≤ g a) → (l.map f).sum ≤ (l.map g).sum
  | [], _ => by simp
  | a :: l, h => by
    have h1 := h a List.mem_cons_self
    have h2 := sum_map_le (l := l) fun b hb => h b (List.mem_cons_of_mem _ hb)
    simp only [List.map_cons, List.sum_cons]
    omega

theorem sum_map_eq_zero {α : Type} {f : α → ℕ} : ∀ {l : List α}, (∀ a ∈ l, f a = 0) → (l.map f).sum = 0
  | [], _ => by simp
  | a :: l, h => by
    have h1 := h a List.mem_cons_self
    have h2 := sum_map_eq_zero (l := l) fun b hb => h b (List.mem_cons_of_mem _ hb)
    simp only [List.map_cons, List.sum_cons]
    omega

/-- A sum over distinct indices of terms zero but at `i` is the term at `i`. -/
theorem sum_map_single {f : ℕ → ℕ} {i : ℕ} :
    ∀ {l : List ℕ}, l.Nodup → i ∈ l → (∀ j ∈ l, j ≠ i → f j = 0) → (l.map f).sum = f i
  | [], _, hi, _ => by simp at hi
  | j :: l, hl, hi, h => by
    rw [List.nodup_cons] at hl
    simp only [List.map_cons, List.sum_cons]
    by_cases hj : j = i
    · have h0 := sum_map_eq_zero (l := l) fun k hk =>
        h k (List.mem_cons_of_mem _ hk) fun hki => hl.1 (by rw [hj, ← hki]; exact hk)
      rw [hj]
      omega
    · have hi' : i ∈ l := by
        rcases List.mem_cons.1 hi with h' | h'
        · exact absurd h'.symm hj
        · exact h'
      have h1 := sum_map_single hl.2 hi' fun k hk => h k (List.mem_cons_of_mem _ hk)
      have h2 := h j List.mem_cons_self hj
      omega

theorem sum_range_ge (k r : ℕ) : ∀ H, ((List.range H).map fun b => if k ≤ b then r else 0).sum = (H - k) * r
  | 0 => by simp
  | H + 1 => by
    simp only [List.range_succ, List.map_append, List.map_cons, List.map_nil, List.sum_append, List.sum_cons,
      List.sum_nil, Nat.add_zero, sum_range_ge k r H]
    split_ifs with h
    · rw [show H + 1 - k = H - k + 1 by omega, Nat.add_mul, Nat.one_mul]
    · simp [show H + 1 - k = 0 by omega, show H - k = 0 by omega]

/-- `M` intervals of `D` buckets each, `D` apart from `c0`, cover `c0 … c0 + D·M − 1` once. -/
theorem sum_stride (c0 D b r : ℕ) : ∀ M,
    ((List.range M).map fun t => if c0 + D * t ≤ b ∧ b < c0 + D * t + D then r else 0).sum =
      if c0 ≤ b ∧ b < c0 + D * M then r else 0
  | 0 => by
    simp only [List.range_zero, List.map_nil, List.sum_nil]
    split_ifs <;> omega
  | M + 1 => by
    simp only [List.range_succ, List.map_append, List.map_cons, List.map_nil, List.sum_append, List.sum_cons,
      List.sum_nil, Nat.add_zero, sum_stride c0 D b r M, Nat.mul_add, Nat.mul_one]
    split_ifs <;> omega

/-- The cells `j < k·r` of a lane in bucket `a + ⌊j/r⌋ = b` are `r` in each of the `k` buckets from `a`. -/
theorem countP_div {r : ℕ} (hr : 0 < r) (a b : ℕ) : ∀ k,
    (List.range (k * r)).countP (fun j => a + j / r == b) = if a ≤ b ∧ b < a + k then r else 0
  | 0 => by
    rw [Nat.zero_mul, List.range_zero, List.countP_nil]
    split_ifs <;> omega
  | k + 1 => by
    rw [Nat.add_mul, Nat.one_mul, List.range_add, List.countP_append, countP_div hr a b k, List.countP_map]
    have hc : (List.range r).countP ((fun j => a + j / r == b) ∘ (k * r + ·)) =
        (List.range r).countP (fun _ => a + k == b) :=
      List.countP_congr fun t ht => by
        have ht : t < r := List.mem_range.1 ht
        have : (k * r + t) / r = k := by
          rw [Nat.add_comm, Nat.add_mul_div_right _ _ hr, Nat.div_eq_of_lt ht, Nat.zero_add]
        simp only [Function.comp_apply, this]
    rw [hc, countP_const, List.length_range]
    simp only [beq_iff_eq]
    split_ifs <;> omega

theorem map_zipIdx_fst {α β : Type} (f : α → β) (l : List α) : l.zipIdx.map (fun x => f x.1) = l.map f := by
  have := congrArg (List.map f) (List.zipIdx_map_fst 0 l)
  rw [List.map_map] at this
  exact this

/-! ### Arrays -/

theorem getD_of_lt {α : Type} {xs : Array α} {i : ℕ} (d : α) (h : i < xs.size) : xs.getD i d = xs[i] := by
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem h, Option.getD_some]

theorem getD_of_ge {α : Type} {xs : Array α} {i : ℕ} (d : α) (h : xs.size ≤ i) : xs.getD i d = d := by
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none h, Option.getD_none]

theorem getD_mem {α : Type} {xs : Array α} {i : ℕ} {d : α} (h : i < xs.size) : xs.getD i d ∈ xs.toList := by
  rw [getD_of_lt d h]
  exact Array.getElem_mem_toList h

theorem getD_irrel {α : Type} {xs : Array α} {i : ℕ} (h : i < xs.size) (d d' : α) : xs.getD i d = xs.getD i d' := by
  rw [getD_of_lt d h, getD_of_lt d' h]

theorem getD_modify {α : Type} {xs : Array α} {x b : ℕ} {f : α → α} {d : α} (hb : b < xs.size) :
    (xs.modify x f).getD b d = if x = b then f (xs.getD b d) else xs.getD b d := by
  have hb' : b < (xs.modify x f).size := by rw [Array.size_modify]; exact hb
  rw [getD_of_lt d hb', getD_of_lt d hb, Array.getElem_modify]

theorem toList_getD {α : Type} (xs : Array α) (d : α) :
    xs.toList = (List.range xs.size).map fun b => xs.getD b d := by
  apply List.ext_getElem
  · simp
  · intro n h1 _
    have hn : n < xs.size := by simpa using h1
    simp only [List.getElem_map, List.getElem_range, Array.getElem_toList, getD_of_lt d hn]

/-- Folding `· ++ g b` into rows over the buckets `a … a + n − 1` appends `g b` to row `b` when it is among them. -/
theorem foldl_modify_append {α : Type} (g : ℕ → Array α) (b : ℕ) :
    ∀ (n a : ℕ) (rows : Array (Array α)), b < rows.size →
      ((List.range' a n).foldl (fun rows b => rows.modify b (· ++ g b)) rows).size = rows.size ∧
        ((List.range' a n).foldl (fun rows b => rows.modify b (· ++ g b)) rows).getD b #[] =
          if a ≤ b ∧ b < a + n then rows.getD b #[] ++ g b else rows.getD b #[]
  | 0, a, rows, _ => ⟨rfl, (ite_eq_right (show ¬ (a ≤ b ∧ b < a + 0) by omega)).symm⟩
  | n + 1, a, rows, hb => by
    have hs : (rows.modify a (· ++ g a)).size = rows.size := Array.size_modify
    obtain ⟨h1, h2⟩ := foldl_modify_append g b n (a + 1) (rows.modify a (· ++ g a)) (by rw [hs]; exact hb)
    rw [List.range'_succ, List.foldl_cons]
    refine ⟨h1.trans hs, ?_⟩
    rw [h2, getD_modify hb]
    by_cases hab : a = b
    · rw [ite_eq_right (show ¬ (a + 1 ≤ b ∧ b < a + 1 + n) by omega), ite_eq_left hab,
        ite_eq_left (show a ≤ b ∧ b < a + (n + 1) by omega), hab]
    · by_cases hc : a + 1 ≤ b ∧ b < a + 1 + n
      · rw [ite_eq_left hc, ite_eq_right hab, ite_eq_left (show a ≤ b ∧ b < a + (n + 1) by omega)]
      · rw [ite_eq_right hc, ite_eq_right hab, ite_eq_right (show ¬ (a ≤ b ∧ b < a + (n + 1)) by omega)]

/-! ### The audit's egress pass -/

/-- Cell `c` is connection `i + 1`'s and not padding: one of the payloads `egressOf` gives it. -/
def connData (i : ℕ) (c : Cell) : Bool := c.conn - 1 == i && c.kind != .pad

def dataOne (i : ℕ) (c : Cell) : ℕ := if connData i c then 1 else 0

theorem dataOne_eq (i : ℕ) (c : Cell) : dataOne i c = if c.conn - 1 = i ∧ c.kind ≠ Kind.pad then 1 else 0 := by
  unfold dataOne connData
  by_cases h1 : c.conn - 1 = i <;> by_cases h2 : c.kind = Kind.pad <;> simp [h1, h2]

theorem egressOf_sizes (N : ℕ) (rows : Array (Array Cell)) :
    (egressOf N rows).1.size = N ∧ (egressOf N rows).2.size = N := by
  unfold egressOf
  refine foldl_inv (fun acc : Array (Option ℕ) × Array (Array ByteArray) => acc.1.size = N ∧ acc.2.size = N) _ ?_
    _ _ (by simp)
  rintro acc ⟨row, b⟩ hacc
  try dsimp only
  rw [← Array.foldl_toList]
  refine foldl_inv (fun acc : Array (Option ℕ) × Array (Array ByteArray) => acc.1.size = N ∧ acc.2.size = N) _ ?_
    _ _ hacc
  rintro ⟨first, data⟩ c hc
  have hc : first.size = N ∧ data.size = N := hc
  try dsimp only
  cases hk : c.kind == Kind.pad <;> simp [hc]

/-- Each connection's first bucket in `egressOf`'s rows is one of them. -/
theorem egressOf_first (N : ℕ) (rows : Array (Array Cell)) :
    ∀ j < N, ∀ b, (egressOf N rows).1.getD j none = some b → b < rows.size := by
  unfold egressOf
  refine And.right (foldl_inv_mem (fun acc : Array (Option ℕ) × Array (Array ByteArray) =>
      acc.1.size = N ∧ ∀ j < N, ∀ b, acc.1.getD j none = some b → b < rows.size) _ _ ?step _ ⟨?sz, ?emp⟩)
  case sz => simp
  case emp =>
    intro j hj b h
    try dsimp only at h
    have hj' : j < (Array.replicate N (none : Option ℕ)).size := by simpa using hj
    rw [getD_of_lt none hj', Array.getElem_replicate] at h
    cases h
  case step =>
    rintro acc ⟨row, b⟩ hrb hacc
    have hb : b < rows.size := (List.mem_zipIdx' hrb).1
    try dsimp only
    rw [← Array.foldl_toList]
    refine foldl_inv (fun acc : Array (Option ℕ) × Array (Array ByteArray) =>
      acc.1.size = N ∧ ∀ j < N, ∀ b', acc.1.getD j none = some b' → b' < rows.size) _ ?_ _ _ hacc
    rintro ⟨first, data⟩ c hp
    have hs : first.size = N := hp.1
    have hf : ∀ j < N, ∀ b', first.getD j none = some b' → b' < rows.size := hp.2
    try dsimp only
    refine ⟨by rw [Array.size_modify]; exact hs, fun j hj b' h => ?_⟩
    have hj' : j < first.size := by rw [hs]; exact hj
    rw [getD_modify hj'] at h
    by_cases hx : c.conn - 1 = j
    · rw [ite_eq_left hx] at h
      cases hf' : first.getD j none with
      | none =>
        rw [hf'] at h
        have h' : some b = some b' := h
        cases h'
        exact hb
      | some b'' =>
        rw [hf'] at h
        have h' : some b'' = some b' := h
        cases h'
        exact hf j hj _ hf'
    · rw [ite_eq_right hx] at h
      exact hf j hj b' h

/-- Connection `i + 1`'s data in `egressOf` is its cells but the padding, row by row. -/
theorem egressOf_count {N i : ℕ} (hi : i < N) (rows : Array (Array Cell)) :
    ((egressOf N rows).2.getD i #[]).size = (rows.toList.map fun row => row.toList.countP (connData i)).sum := by
  unfold egressOf
  refine Eq.trans (foldl_sum _ (fun acc : Array (Option ℕ) × Array (Array ByteArray) => (acc.2.getD i #[]).size)
    (fun x : Array Cell × ℕ => (x.1.toList.map (dataOne i)).sum)
    (fun acc : Array (Option ℕ) × Array (Array ByteArray) => acc.2.size = N) ?step _ _ ?init).2 ?fin
  case init => simp
  case step =>
    rintro acc ⟨row, b⟩ hp
    try dsimp only
    rw [← Array.foldl_toList]
    refine foldl_sum _ (fun acc : Array (Option ℕ) × Array (Array ByteArray) => (acc.2.getD i #[]).size) (dataOne i)
      (fun acc : Array (Option ℕ) × Array (Array ByteArray) => acc.2.size = N) ?_ _ _ hp
    rintro ⟨first, data⟩ c hp
    have hp : data.size = N := hp
    have hb : i < data.size := by rw [hp]; exact hi
    try dsimp only
    cases hk : c.kind == Kind.pad
    · rw [ite_eq_right Bool.false_ne_true, getD_modify hb]
      refine ⟨by rw [Array.size_modify]; exact hp, ?_⟩
      by_cases hx : c.conn - 1 = i
      · have h1 : dataOne i c = 1 := by rw [dataOne_eq, ite_eq_left ⟨hx, ne_of_beq_false hk⟩]
        rw [ite_eq_left hx, h1, Array.size_push]
      · have h0 : dataOne i c = 0 := by rw [dataOne_eq, ite_eq_right fun h => hx h.1]
        rw [ite_eq_right hx, h0, Nat.add_zero]
    · have h0 : dataOne i c = 0 := by rw [dataOne_eq, ite_eq_right fun h => h.2 (beq_iff_eq.mp hk)]
      rw [ite_eq_left (rfl : true = true), h0, Nat.add_zero]
      exact ⟨hp, rfl⟩
  case fin =>
    have h0 : ((Array.replicate N (#[] : Array ByteArray)).getD i #[]).size = 0 := by
      rw [getD_of_lt (xs := Array.replicate N (#[] : Array ByteArray)) #[] (by simpa using hi),
        Array.getElem_replicate, Array.size_empty]
    try dsimp only
    rw [h0, Nat.zero_add, map_zipIdx_fst (fun row : Array Cell => (row.toList.map (dataOne i)).sum) rows.toList]
    exact congrArg List.sum (List.map_congr_left fun row _ => (countP_eq_sum_map (connData i) row.toList).symm)

/-- What the audit reads off `egressOf` once every connection has a first bucket. -/
theorem egressOf_facts {N : ℕ} {rows : Array (Array Cell)} {adm : Array ℕ}
    (h : (egressOf N rows).1.mapM (m := Option) (fun a => a) = some adm) :
    adm.size = N ∧ (egressOf N rows).2.size = N ∧ (∀ i < N, adm.getD i 0 < rows.size) ∧
      ∀ i < N, ((egressOf N rows).2.getD i #[]).size =
        (rows.toList.map fun row => row.toList.countP (connData i)).sum := by
  have hs := egressOf_sizes N rows
  have hid := option_array_mapM_id h
  have hadm : adm.size = N := (option_array_mapM_size h).trans hs.1
  refine ⟨hadm, hs.2, fun i hi => ?_, fun i hi => egressOf_count hi rows⟩
  have hi' : i < adm.size := by rw [hadm]; exact hi
  exact egressOf_first N rows i hi _ (by rw [hid]; simp [hi'])

/-- A row's connections are its cells' groups of `r`. -/
theorem connsOf_size {r : ℕ} {what : String} {b : ℕ} {row : Array Cell} {cs : Array ℕ}
    (h : connsOf r what b row = .ok cs) : cs.size = row.size / r := by
  unfold connsOf at h
  dsimp only at h
  try simp only [throw_bind'] at h
  repeat' rows_step
  all_goals
    obtain rfl := pure_ok h
    simp

/-- A direction's frames are `T` rows a window, each of at most `S` frames. -/
theorem framesOf_ok {p : Params} {S : ℕ} {what : String} {records : Array Record}
    {fr : Array (Array ByteArray)} (h : framesOf p S what records = .ok fr) :
    fr.size = records.size * p.window ∧ ∀ row ∈ fr.toList, row.size ≤ S := by
  unfold framesOf at h
  obtain ⟨windows, hw, h⟩ := ebind_ok h
  obtain rfl := pure_ok h
  have hw := list_mapM_forall₂ hw
  have hwin : ∀ y ∈ windows, y.length = p.window ∧ ∀ row ∈ y, row.size ≤ S := by
    intro y hy
    obtain ⟨⟨r, w⟩, -, h⟩ := forall₂_mem_right hw y hy
    dsimp only at h
    try simp only [throw_bind'] at h
    repeat' rows_step
    all_goals
      have h2 := list_mapM_forall₂ h
      have hT : r.grid.size = p.window := by simpa using hc
      refine ⟨?_, fun row hrow => ?_⟩
      · have := forall₂_length h2
        simp only [List.length_zipIdx, Array.length_toList] at this
        omega
      · obtain ⟨⟨grow, t⟩, -, h⟩ := forall₂_mem_right h2 row hrow
        dsimp only at h
        try simp only [throw_bind'] at h
        repeat' rows_step
        all_goals
          obtain rfl := pure_ok h
          simp only [Bool.and_eq_true, beq_iff_eq] at hc
          exact Array.size_filterMap_le.trans hc.1.le
  refine ⟨?_, fun row hrow => ?_⟩
  · rw [List.size_toArray, length_flatten_const (fun y hy => (hwin y hy).1), ← forall₂_length hw]
    simp
  · have hrow : row ∈ windows.flatten := hrow
    obtain ⟨y, hy, hry⟩ := List.mem_flatten.1 hrow
    exact (hwin y hy).2 row hry

/-- A direction's cells are its frames, row for row. -/
theorem cellsOf_ok {p : Params} {what : String} {fr : Array (Array ByteArray)} {eg : Array (Array Cell)}
    (h : cellsOf p what fr = .ok eg) :
    eg.size = fr.size ∧ ∀ c ∈ eg.toList, ∃ row ∈ fr.toList, c.size = row.size := by
  unfold cellsOf at h
  have h2 := array_mapM_forall₂ h
  refine ⟨?_, fun c hc => ?_⟩
  · have := forall₂_length h2
    simp only [List.length_zipIdx, Array.length_toList] at this
    exact this.symm
  · obtain ⟨⟨row, b⟩, hx, hr⟩ := forall₂_mem_right h2 c hc
    have hm : (row, b) ∈ fr.toList.zipIdx := hx
    refine ⟨row, List.fst_mem_of_mem_zipIdx hm, ?_⟩
    try dsimp only at hr
    have := forall₂_length (array_mapM_forall₂ hr)
    simp only [Array.length_toList] at this
    exact this.symm

/-! ### Lanes and the rendered rows -/

theorem laneCells_ge {r : ℕ} (hr : 0 < r) (n : ℕ) : n ≤ laneCells r n := by
  unfold laneCells
  have h1 := Nat.div_add_mod (n + r - 1) r
  have h2 := Nat.mod_lt (n + r - 1) hr
  rw [Nat.mul_comm]
  omega

theorem laneCells_zero {r : ℕ} (hr : 0 < r) : laneCells r 0 = 0 := by
  unfold laneCells
  rw [Nat.zero_add, Nat.div_eq_of_lt (by omega), Nat.zero_mul]

theorem laneOf_size {r : ℕ} (hr : 0 < r) (o : ℕ) (data : Array ByteArray) :
    (laneOf r o data).size = laneCells r data.size := by
  unfold laneOf
  by_cases h : data.size = 0
  · rw [ite_eq_left h]
    simp [h, laneCells_zero hr]
  · rw [ite_eq_right h]
    have := laneCells_ge hr data.size
    simp only [Array.size_append, Array.size_push, Array.size_map, Array.size_extract, Array.size_replicate]
    omega

theorem laneOf_conn (r o : ℕ) (data : Array ByteArray) : ∀ x ∈ (laneOf r o data).toList, x.conn = o := by
  intro x hx
  unfold laneOf at hx
  split at hx
  · simp at hx
  · dsimp only at hx
    rw [Array.toList_append, Array.toList_push] at hx
    rcases List.mem_append.1 hx with hx | hx
    · rcases List.mem_append.1 hx with hx | hx
      · rw [Array.toList_map] at hx
        obtain ⟨a, -, rfl⟩ := List.mem_map.1 hx
        rfl
      · rw [List.mem_singleton] at hx
        subst hx
        rfl
    · rw [Array.toList_replicate] at hx
      obtain ⟨-, rfl⟩ := List.mem_replicate.1 hx
      rfl

theorem laneFrames_size {p : Params} {a o b k : ℕ} {lane : Array Cell} (hl : lane.size = k * p.slots) (h1 : a ≤ b)
    (h2 : b < a + p.delay + k) : (laneFrames p a o lane b).size = p.slots := by
  unfold laneFrames
  rw [ite_eq_right (show ¬ b < a by omega)]
  by_cases h3 : b < a + p.delay
  · rw [ite_eq_left h3, Array.size_replicate]
  · rw [ite_eq_right h3, Array.size_extract, hl]
    have : (b - a - p.delay + 1) * p.slots ≤ k * p.slots := Nat.mul_le_mul_right _ (by omega)
    rw [Nat.add_mul, Nat.one_mul] at this ⊢
    omega

theorem laneFrames_conn {p : Params} {a o b : ℕ} {lane : Array Cell} (hl : ∀ y ∈ lane.toList, y.conn = o) :
    ∀ x ∈ (laneFrames p a o lane b).toList, x.conn = o := by
  intro x hx
  unfold laneFrames at hx
  split at hx
  · simp at hx
  · split at hx
    · rw [Array.toList_replicate] at hx
      obtain ⟨-, rfl⟩ := List.mem_replicate.1 hx
      rfl
    · rw [Array.toList_extract, List.extract_eq_drop_take'] at hx
      exact hl x (List.mem_of_mem_take (List.mem_of_mem_drop hx))

/-- A connection's data cells in bucket `b`: none before `a + D`, at most `r` from it. -/
theorem laneFrames_count {p : Params} {a o b : ℕ} {lane : Array Cell} (i : ℕ) :
    (laneFrames p a o lane b).toList.countP (connData i) ≤ if a + p.delay ≤ b then p.slots else 0 := by
  unfold laneFrames
  by_cases h1 : b < a
  · rw [ite_eq_left h1]
    simp
  · rw [ite_eq_right h1]
    by_cases h2 : b < a + p.delay
    · rw [ite_eq_left h2, Array.toList_replicate, List.countP_replicate]
      simp [connData, Cell.pad]
    · rw [ite_eq_right h2, ite_eq_left (show a + p.delay ≤ b by omega)]
      refine Nat.le_trans List.countP_le_length ?_
      rw [Array.length_toList, Array.size_extract, Nat.add_mul, Nat.one_mul]
      omega

theorem laneFrames_count_ne {p : Params} {a o b i : ℕ} {lane : Array Cell} (ho : o - 1 ≠ i)
    (hl : ∀ y ∈ lane.toList, y.conn = o) : (laneFrames p a o lane b).toList.countP (connData i) = 0 := by
  rw [List.countP_eq_zero]
  intro x hx
  simp [connData, laneFrames_conn hl x hx, ho]

/-- One step of `expectedRows`: connection `i` adds its cells to the buckets from its admission to its lane's last. -/
def rowsStep (p : Params) (s : Schedule) (data : Array (Array ByteArray)) (H : ℕ) (rows : Array (Array Cell))
    (i : ℕ) : Array (Array Cell) :=
  (List.range' (s.adm.getD i 0) (min H (s.adm.getD i 0 + p.delay + s.cells.getD i 0 / p.slots) - s.adm.getD i 0)).foldl
    (fun rows b => rows.modify b
      (· ++ laneFrames p (s.adm.getD i 0) (i + 1) (laneOf p.slots (i + 1) (data.getD i #[])) b)) rows

/-- Connection `i`'s cells in `expectedRows`' row `b`. -/
def rowsAt (p : Params) (s : Schedule) (data : Array (Array ByteArray)) (H b i : ℕ) : List Cell :=
  if s.adm.getD i 0 ≤ b ∧ b < min H (s.adm.getD i 0 + p.delay + s.cells.getD i 0 / p.slots) then
    (laneFrames p (s.adm.getD i 0) (i + 1) (laneOf p.slots (i + 1) (data.getD i #[])) b).toList
  else []

theorem expectedRows_eq (p : Params) (s : Schedule) (data : Array (Array ByteArray)) (H : ℕ) :
    expectedRows p s data H = (List.range s.adm.size).foldl (rowsStep p s data H) (Array.replicate H #[]) := rfl

theorem rowsStep_one {p : Params} {s : Schedule} {data : Array (Array ByteArray)} {H b : ℕ}
    {rows : Array (Array Cell)} (i : ℕ) (hb : b < rows.size) :
    (rowsStep p s data H rows i).size = rows.size ∧
      ((rowsStep p s data H rows i).getD b #[]).toList = (rows.getD b #[]).toList ++ rowsAt p s data H b i := by
  obtain ⟨h1, h2⟩ := foldl_modify_append
    (laneFrames p (s.adm.getD i 0) (i + 1) (laneOf p.slots (i + 1) (data.getD i #[]))) b
    (min H (s.adm.getD i 0 + p.delay + s.cells.getD i 0 / p.slots) - s.adm.getD i 0) (s.adm.getD i 0) rows hb
  refine ⟨h1, ?_⟩
  unfold rowsStep rowsAt
  rw [h2]
  split_ifs
  all_goals first | omega | rw [Array.toList_append] | rw [List.append_nil]

theorem rowsStep_fold {p : Params} {s : Schedule} {data : Array (Array ByteArray)} {H b : ℕ} :
    ∀ (l : List ℕ) (rows : Array (Array Cell)), b < rows.size →
      (l.foldl (rowsStep p s data H) rows).size = rows.size ∧
        ((l.foldl (rowsStep p s data H) rows).getD b #[]).toList =
          (rows.getD b #[]).toList ++ l.flatMap (rowsAt p s data H b)
  | [], rows, _ => ⟨rfl, by simp⟩
  | i :: l, rows, hb => by
    obtain ⟨h1, h2⟩ := rowsStep_one (p := p) (s := s) (data := data) (H := H) i hb
    obtain ⟨h3, h4⟩ := rowsStep_fold l (rowsStep p s data H rows i) (by rw [h1]; exact hb)
    rw [List.foldl_cons, List.flatMap_cons]
    exact ⟨h3.trans h1, by rw [h4, h2, List.append_assoc]⟩

/-- **Row `b` of `expectedRows`** is each connection's cells in it, connection by connection. -/
theorem expectedRows_getD {p : Params} {s : Schedule} {data : Array (Array ByteArray)} {H b : ℕ} (hb : b < H) :
    (expectedRows p s data H).size = H ∧
      ((expectedRows p s data H).getD b #[]).toList = (List.range s.adm.size).flatMap (rowsAt p s data H b) := by
  have hb' : b < (Array.replicate H (#[] : Array Cell)).size := by simpa using hb
  obtain ⟨h1, h2⟩ := rowsStep_fold (p := p) (s := s) (data := data) (List.range s.adm.size) _ hb'
  rw [expectedRows_eq]
  refine ⟨by rw [h1, Array.size_replicate], ?_⟩
  rw [h2, getD_of_lt _ hb', Array.getElem_replicate, Array.toList_empty, List.nil_append]

/-! ### The schedule `scheduleOf` shows -/

/-- Connection `i + 1`'s lane's buckets, `⌈n/r⌉` for its `n` data cells. -/
def kOf (p : Params) (data : Array (Array ByteArray)) (i : ℕ) : ℕ := ((data.getD i #[]).size + p.slots - 1) / p.slots

theorem sched_adm {p : Params} {adm : Array ℕ} {data : Array (Array ByteArray)} :
    (scheduleOf p adm data).adm = adm := rfl

theorem cells_getD {p : Params} {adm : Array ℕ} {data : Array (Array ByteArray)} {i : ℕ} (hi : i < data.size) :
    (scheduleOf p adm data).cells.getD i 0 = laneCells p.slots (data.getD i #[]).size := by
  simp [scheduleOf, hi]

theorem cells_getD_ge {p : Params} {adm : Array ℕ} {data : Array (Array ByteArray)} {i : ℕ} (hi : data.size ≤ i) :
    (scheduleOf p adm data).cells.getD i 0 = 0 := by
  simp [scheduleOf, hi]

theorem cells_div {p : Params} {adm : Array ℕ} {data : Array (Array ByteArray)} {i : ℕ} (hr : 0 < p.slots)
    (hi : i < data.size) : (scheduleOf p adm data).cells.getD i 0 / p.slots = kOf p data i := by
  rw [cells_getD hi]
  unfold laneCells kOf
  exact Nat.mul_div_cancel _ hr

theorem cells_dvd {p : Params} {adm : Array ℕ} {data : Array (Array ByteArray)} {i : ℕ} :
    p.slots ∣ (scheduleOf p adm data).cells.getD i 0 := by
  by_cases hi : i < data.size
  · rw [cells_getD hi]
    exact Nat.dvd_mul_left _ _
  · rw [cells_getD_ge (by omega)]
    exact Nat.dvd_zero _

theorem laneOf_size' {p : Params} {data : Array (Array ByteArray)} {i : ℕ} (hr : 0 < p.slots) :
    (laneOf p.slots (i + 1) (data.getD i #[])).size = kOf p data i * p.slots :=
  laneOf_size hr _ _

/-- Connection `i + 1` holds `r` cells of each bucket of the run from its admission to its lane's last. -/
theorem rowsAt_length {p : Params} {adm : Array ℕ} {data : Array (Array ByteArray)} {H b i : ℕ} (hr : 0 < p.slots)
    (hb : b < H) (hi : i < data.size) :
    (rowsAt p (scheduleOf p adm data) data H b i).length =
      if adm.getD i 0 ≤ b ∧ b < adm.getD i 0 + p.delay + kOf p data i then p.slots else 0 := by
  unfold rowsAt
  rw [sched_adm, cells_div hr hi]
  by_cases hc : adm.getD i 0 ≤ b ∧ b < adm.getD i 0 + p.delay + kOf p data i
  · rw [ite_eq_left (show adm.getD i 0 ≤ b ∧ b < min H (adm.getD i 0 + p.delay + kOf p data i) by omega), ite_eq_left hc,
      Array.length_toList, laneFrames_size (laneOf_size' hr) hc.1 hc.2]
  · rw [ite_eq_right (show ¬ (adm.getD i 0 ≤ b ∧ b < min H (adm.getD i 0 + p.delay + kOf p data i)) by omega), ite_eq_right hc,
      List.length_nil]

theorem rowsAt_count_self (p : Params) (s : Schedule) (data : Array (Array ByteArray)) (H b i : ℕ) :
    (rowsAt p s data H b i).countP (connData i) ≤ if s.adm.getD i 0 + p.delay ≤ b then p.slots else 0 := by
  unfold rowsAt
  split
  · exact laneFrames_count i
  · simp

theorem rowsAt_count_ne (p : Params) (s : Schedule) (data : Array (Array ByteArray)) (H b : ℕ) {i i' : ℕ}
    (h : i' ≠ i) : (rowsAt p s data H b i').countP (connData i) = 0 := by
  unfold rowsAt
  split
  · exact laneFrames_count_ne (by omega) (laneOf_conn _ _ _)
  · simp

/-- **Each lane ends in the run**: connection `i + 1`'s data, one cell each, fit the buckets from `a_i + D` to `H`. -/
theorem data_le {p : Params} {H N : ℕ} {eg : Array (Array Cell)} {adm : Array ℕ} {data : Array (Array ByteArray)}
    (heg : eg.size = H) (hadm : adm.size = N)
    (hX : ∀ b < H, eg.getD b #[] = (expectedRows p (scheduleOf p adm data) data H).getD b #[])
    {i : ℕ} (hi : i < N)
    (hcount : (data.getD i #[]).size = (eg.toList.map fun row => row.toList.countP (connData i)).sum) :
    (data.getD i #[]).size ≤ (H - (adm.getD i 0 + p.delay)) * p.slots := by
  rw [hcount, toList_getD eg #[], List.map_map, heg, ← sum_range_ge (adm.getD i 0 + p.delay) p.slots H]
  apply sum_map_le
  intro b hb
  have hb : b < H := List.mem_range.1 hb
  show (eg.getD b #[]).toList.countP (connData i) ≤ if adm.getD i 0 + p.delay ≤ b then p.slots else 0
  rw [hX b hb, (expectedRows_getD hb).2, List.countP_flatMap, sched_adm, hadm]
  have hs := sum_map_single (f := List.countP (connData i) ∘ rowsAt p (scheduleOf p adm data) data H b)
    (l := List.range N) List.nodup_range (List.mem_range.2 hi) (fun j _ hj => rowsAt_count_ne p _ data H b hj)
  rw [hs]
  exact rowsAt_count_self p (scheduleOf p adm data) data H b i

/-! ### The load of a bucket -/

/-- A lane of whole buckets loads `r` frames in each bucket from its admission to its last. -/
theorem laneLoad_eq {κ : Type} (s : NetTiming.WardenSchedule κ) (c : κ) (b k : ℕ) (hr : 0 < s.r)
    (hc : s.cells c = k * s.r) :
    NetTiming.laneLoad s c b = if s.adm c ≤ b ∧ b < s.adm c + s.D + k then s.r else 0 := by
  have ht : NetTiming.tailPads s c b = 0 := by
    unfold NetTiming.tailPads
    rw [hc]
    rcases k with _ | k'
    · simp
    · have h1 : (k' + 1) * s.r - 1 = (s.r - 1) + k' * s.r := by
        rw [Nat.add_mul, Nat.one_mul]
        omega
      rw [h1, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt (by omega : s.r - 1 < s.r), Nat.sub_self, ite_self]
  unfold NetTiming.laneLoad
  rw [ht, hc, ← List.countP_eq_length_filter]
  unfold NetTiming.WardenSchedule.bucket
  rw [countP_div hr]
  split_ifs <;> omega

/-- Connection `i`'s load in bucket `b` on `wardenOf`'s schedule. -/
def loadAt (p : Params) (n : ℕ) (s : Schedule) (b i : ℕ) : ℕ :=
  if s.adm.getD i (n * p.window + p.delay * (i + 1)) ≤ b ∧
      b < s.adm.getD i (n * p.window + p.delay * (i + 1)) + p.delay + s.cells.getD i 0 / p.slots then p.slots
  else 0

theorem laneLoad_wardenOf {p : Params} {n : ℕ} {adm : Array ℕ} {data : Array (Array ByteArray)} (hr : 0 < p.slots)
    (c : Fin (periodConns p n)) (b : ℕ) :
    NetTiming.laneLoad (wardenOf p n (scheduleOf p adm data)) c b = loadAt p n (scheduleOf p adm data) b c.val := by
  have hc : (wardenOf p n (scheduleOf p adm data)).cells c =
      (scheduleOf p adm data).cells.getD c.val 0 / p.slots * (wardenOf p n (scheduleOf p adm data)).r :=
    (Nat.div_mul_cancel cells_dvd).symm
  exact (laneLoad_eq (wardenOf p n (scheduleOf p adm data)) c b _ hr hc).trans rfl

theorem loadAt_real {p : Params} {n N : ℕ} {adm : Array ℕ} {data : Array (Array ByteArray)} {b i : ℕ}
    (hr : 0 < p.slots) (hi : i < N) (hadm : adm.size = N) (hd : data.size = N) :
    loadAt p n (scheduleOf p adm data) b i =
      if adm.getD i 0 ≤ b ∧ b < adm.getD i 0 + p.delay + kOf p data i then p.slots else 0 := by
  have hi1 : i < adm.size := by omega
  have hi2 : i < data.size := by omega
  unfold loadAt
  rw [sched_adm, getD_irrel hi1 (n * p.window + p.delay * (i + 1)) 0, cells_div hr hi2]

theorem loadAt_fake {p : Params} {n N : ℕ} {adm : Array ℕ} {data : Array (Array ByteArray)} {b i : ℕ}
    (hi : N ≤ i) (hadm : adm.size = N) (hd : data.size = N) :
    loadAt p n (scheduleOf p adm data) b i =
      if n * p.window + p.delay * (i + 1) ≤ b ∧ b < n * p.window + p.delay * (i + 1) + p.delay then p.slots
      else 0 := by
  have hi1 : adm.size ≤ i := by omega
  have hi2 : data.size ≤ i := by omega
  unfold loadAt
  rw [sched_adm, getD_of_ge _ hi1, cells_getD_ge hi2, Nat.zero_div,
    Nat.add_zero (n * p.window + p.delay * (i + 1) + p.delay)]

theorem find_eq {α : Type} [BEq α] [LawfulBEq α] {H : ℕ} {f g : ℕ → α}
    (h : ∀ b, List.find? (fun b => f b != g b) (List.range H) = some b → False) : ∀ b < H, f b = g b := by
  intro b hb
  cases hf : List.find? (fun b => f b != g b) (List.range H) with
  | some b' => exact (h b' hf).elim
  | none =>
    have := List.find?_eq_none.1 hf b (List.mem_range.2 hb)
    simpa using this

/-- **The schedule fits**: from what an accepted audit reads off the egress record. -/
theorem fits_of {p : Params} {C : Type} {L K W N : ℕ} {eg : Array (Array Cell)} {adm : Array ℕ}
    {data : Array (Array ByteArray)}
    (F : NetTiming.RowsFormat (Fin (periodConns p W)) C L K) (hF : F.Wf) (hT : F.T = p.window)
    (hr : 0 < p.slots) (hl : 0 < p.lanes) (hH : 0 < W * p.window)
    (heg : eg.size = W * p.window) (hrow : ∀ b < W * p.window, (eg.getD b #[]).size ≤ p.lanes * p.slots)
    (hadm : adm.size = N) (hdata : data.size = N) (hN : N ≤ periodConns p W)
    (hfirst : ∀ i < N, adm.getD i 0 < W * p.window)
    (hcount : ∀ i < N,
      (data.getD i #[]).size = (eg.toList.map fun row => row.toList.countP (connData i)).sum)
    (hX : ∀ b < W * p.window,
      eg.getD b #[] = (expectedRows p (scheduleOf p adm data) data (W * p.window)).getD b #[]) :
    F.Fits (p.lanes * p.slots) W (wardenOf p W (scheduleOf p adm data)) := by
  have hF1 : ∀ i < N, 0 < kOf p data i → adm.getD i 0 + p.delay + kOf p data i ≤ W * p.window := by
    intro i hi hk
    have hle := data_le (p := p) heg hadm hX hi (hcount i hi)
    have hlt : kOf p data i < W * p.window - (adm.getD i 0 + p.delay) + 1 := by
      unfold kOf
      rw [Nat.div_lt_iff_lt_mul hr, Nat.add_mul, Nat.one_mul]
      omega
    omega
  refine And.intro (fun c j hj => ?_) (fun b => ?_)
  · rw [hT]
    have hj' : j < (scheduleOf p adm data).cells.getD c.val 0 := hj
    have hc : c.val < N := by
      by_contra hc
      have : (scheduleOf p adm data).cells.getD c.val 0 = 0 := cells_getD_ge (by omega)
      omega
    have hcells : (scheduleOf p adm data).cells.getD c.val 0 = kOf p data c.val * p.slots :=
      cells_getD (by omega)
    have hk : j / p.slots < kOf p data c.val := by
      rw [Nat.div_lt_iff_lt_mul hr]
      omega
    have h1 := hF1 c.val hc (Nat.lt_of_le_of_lt (Nat.zero_le _) hk)
    show adm.getD c.val (W * p.window + p.delay * (c.val + 1)) + p.delay + j / p.slots < W * p.window
    rw [getD_irrel (show c.val < adm.size by omega) _ 0]
    omega
  · have e1 : (F.conns.map fun c => NetTiming.laneLoad (wardenOf p W (scheduleOf p adm data)) c b) =
        (F.conns.map Fin.val).map (loadAt p W (scheduleOf p adm data) b) := by
      rw [List.map_map]
      exact List.map_congr_left fun c _ => laneLoad_wardenOf hr c b
    have hnd : (F.conns.map Fin.val).Nodup :=
      List.Pairwise.map Fin.val (fun a b (h : a ≠ b) hv => h (Fin.eq_of_val_eq hv)) hF.1
    have hperm : (F.conns.map Fin.val).Perm (List.range (periodConns p W)) := by
      rw [List.perm_ext_iff_of_nodup hnd List.nodup_range]
      intro x
      rw [List.mem_range, List.mem_map]
      exact ⟨fun ⟨c, _, hc⟩ => hc ▸ c.isLt, fun hx => ⟨⟨x, hx⟩, hF.2.1 _, rfl⟩⟩
    rw [e1, (hperm.map (loadAt p W (scheduleOf p adm data) b)).sum_nat]
    obtain ⟨M, hM⟩ : ∃ M, periodConns p W = N + M := ⟨periodConns p W - N, by omega⟩
    rw [hM, List.range_add, List.map_append, List.sum_append, List.map_map]
    have hR : (List.range N).map (loadAt p W (scheduleOf p adm data) b) =
        (List.range N).map fun i =>
          if adm.getD i 0 ≤ b ∧ b < adm.getD i 0 + p.delay + kOf p data i then p.slots else 0 :=
      List.map_congr_left fun i hi => loadAt_real hr (List.mem_range.1 hi) hadm hdata
    have hFk : ((List.range M).map (loadAt p W (scheduleOf p adm data) b ∘ (N + ·))).sum =
        if W * p.window + p.delay * (N + 1) ≤ b ∧ b < W * p.window + p.delay * (N + 1) + p.delay * M
        then p.slots else 0 := by
      rw [← sum_stride (W * p.window + p.delay * (N + 1)) p.delay b p.slots M]
      refine congrArg List.sum (List.map_congr_left fun t _ => ?_)
      show loadAt p W (scheduleOf p adm data) b (N + t) = _
      rw [loadAt_fake (by omega : N ≤ N + t) hadm hdata]
      simp only [Nat.mul_add, Nat.mul_one]
      split_ifs <;> omega
    have hS : ∀ b' < W * p.window, ((List.range N).map fun i =>
        if adm.getD i 0 ≤ b' ∧ b' < adm.getD i 0 + p.delay + kOf p data i then p.slots else 0).sum ≤
          p.lanes * p.slots := by
      intro b' hb'
      have h1 := hrow b' hb'
      rw [hX b' hb', ← Array.length_toList, (expectedRows_getD hb').2, List.length_flatMap, sched_adm, hadm] at h1
      have e : ∀ i ∈ List.range N,
          (if adm.getD i 0 ≤ b' ∧ b' < adm.getD i 0 + p.delay + kOf p data i then p.slots else 0) =
            (rowsAt p (scheduleOf p adm data) data (W * p.window) b' i).length := fun i hi =>
        Eq.symm (rowsAt_length hr hb' (by rw [hdata]; exact List.mem_range.1 hi))
      rw [List.map_congr_left e]
      exact h1
    have hReal : ∀ b', W * p.window ≤ b' → ((List.range N).map fun i =>
        if adm.getD i 0 ≤ b' ∧ b' < adm.getD i 0 + p.delay + kOf p data i then p.slots else 0).sum ≤
          ((List.range N).map fun i =>
            if adm.getD i 0 ≤ W * p.window - 1 ∧ W * p.window - 1 < adm.getD i 0 + p.delay + kOf p data i
            then p.slots else 0).sum := by
      intro b' hb'
      apply sum_map_le
      intro i hi
      have hi : i < N := List.mem_range.1 hi
      have h1 := hfirst i hi
      have h2 := hF1 i hi
      split_ifs <;> omega
    have hReal0 : ∀ b', W * p.window + p.delay ≤ b' → ((List.range N).map fun i =>
        if adm.getD i 0 ≤ b' ∧ b' < adm.getD i 0 + p.delay + kOf p data i then p.slots else 0).sum = 0 := by
      intro b' hb'
      apply sum_map_eq_zero
      intro i hi
      have hi : i < N := List.mem_range.1 hi
      have h1 := hfirst i hi
      have h2 := hF1 i hi
      split_ifs <;> omega
    rw [hR, hFk]
    have hc0 : p.delay ≤ p.delay * (N + 1) := Nat.le_mul_of_pos_right _ (by omega)
    have hSr : p.slots ≤ p.lanes * p.slots := Nat.le_mul_of_pos_left _ hl
    by_cases hb1 : b < W * p.window
    · have := hS b hb1
      split_ifs <;> omega
    · by_cases hb2 : b < W * p.window + p.delay
      · have h1 := hReal b (by omega)
        have h2 := hS (W * p.window - 1) (by omega)
        split_ifs <;> omega
      · rw [hReal0 b (by omega)]
        split_ifs <;> omega

/-- **An accepted run admits at most `lanes·W·T` connections, and its schedule fits the rows format.** -/
theorem auditCovered {p : Params} {C : Type} {L K : ℕ} {ingress egress fails : Array Record} {sum : Summary}
    (F : NetTiming.RowsFormat (Fin (periodConns p egress.size)) C L K) (hF : F.Wf) (hT : F.T = p.window)
    (h : audit p ingress egress fails = .ok sum) :
    sum.schedule.adm.size ≤ periodConns p egress.size ∧
      F.Fits (p.lanes * p.slots) egress.size (wardenOf p egress.size sum.schedule) := by
  unfold audit at h
  dsimp only at h
  try simp only [throw_bind'] at h
  repeat' rows_step
  all_goals
    obtain rfl := pure_ok h
    have hpre := ‹(decide (0 < egress.size) && _ && _) = true›
    have hpar := ‹(decide (0 < p.slots) && _ && _ && _ && _) = true›
    simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hpre hpar
    have hW : 0 < egress.size := by omega
    have hr : 0 < p.slots := by omega
    have hl : 0 < p.lanes := by omega
    have hT' : 0 < p.window := by omega
    have hFr := framesOf_ok ‹framesOf p (p.lanes * p.slots) "egress" egress = .ok _›
    have hC := cellsOf_ok h_eg
    have hE := array_mapM_forall₂ h_egConns
    have heg : eg.size = egress.size * p.window := hC.1.trans hFr.1
    have hsize : egConns.size = eg.size := by
      have := forall₂_length hE
      simpa using this.symm
    have hcs : ∀ cs ∈ egConns.toList, cs.size ≤ p.lanes := by
      intro cs hcs
      obtain ⟨⟨row, b⟩, hx, hr⟩ := forall₂_mem_right hE cs hcs
      have hr' : connsOf p.slots "egress" b row = .ok cs := hr
      have hm : (row, b) ∈ eg.toList.zipIdx := hx
      obtain ⟨frow, hfrow, hsz⟩ := hC.2 row (List.fst_mem_of_mem_zipIdx hm)
      rw [connsOf_size hr']
      apply Nat.div_le_of_le_mul
      rw [hsz, Nat.mul_comm]
      exact hFr.2 frow hfrow
    have hsum := foldl_size_le p.lanes egConns hcs
    have hNle : Array.foldl (fun n cs => max n (cs.back?.getD 0)) 0 egConns ≤ periodConns p egress.size := by
      unfold periodConns
      rw [← heg, ← hsize]
      omega
    obtain ⟨hadm, hdata, hfirst, hcount⟩ := egressOf_facts ‹Array.mapM _ _ = some _›
    have hrow : ∀ b < egress.size * p.window, (eg.getD b #[]).size ≤ p.lanes * p.slots := by
      intro b hb
      obtain ⟨row, hrow, hsz⟩ := hC.2 _ (getD_mem (by omega : b < eg.size))
      rw [hsz]
      exact hFr.2 row hrow
    have hX := find_eq (α := Array Cell)
      ‹∀ b, List.find? _ (List.range (egress.size * p.window)) = some b → False›
    exact ⟨Nat.le_trans (Nat.le_of_eq hadm) hNle,
      fits_of F hF hT hr hl (Nat.mul_pos hW hT') heg hrow hadm hdata hNle
        (fun i hi => Nat.lt_of_lt_of_eq (hfirst i hi) heg) hcount hX⟩

/-! ### The network certifier's format, read at a well-formed one -/

def kindTag : NetTiming.CellView Unit → ℕ
  | .pad => 0
  | .fail => 1
  | .data _ => 2

def kindOfTag (m : ℕ) : NetTiming.CellView Unit :=
  if m = 0 then .pad else if m = 1 then .fail else .data ()

theorem kindOfTag_tag (v : NetTiming.CellView Unit) : kindOfTag (kindTag v) = v := by
  cases v <;> rfl

theorem kindTag_lt (v : NetTiming.CellView Unit) : kindTag v < 3 := by
  cases v <;> simp [kindTag]

/-- A well-formed rows format of `T`-bucket windows over `n` connections, each frame its connection and its kind in
unary. -/
def unaryFormat (n T : ℕ) : NetTiming.RowsFormat (Fin n) Unit 1 1 where
  conns := List.finRange n
  enc x := List.replicate (3 * x.1.val + kindTag x.2) true
  dec l := if h : l.length / 3 < n then some (⟨l.length / 3, h⟩, kindOfTag (l.length % 3)) else none
  link := 0
  T := T
  jitter _ _ _ := 0
  status _ _ _ _ := .complete

theorem unaryFormat_wf (n T : ℕ) : (unaryFormat n T).Wf := by
  unfold NetTiming.RowsFormat.Wf
  refine ⟨List.nodup_finRange n, List.mem_finRange, ?_⟩
  rintro ⟨c, v⟩
  have ht := kindTag_lt v
  have h1 : (3 * c.val + kindTag v) / 3 = c.val := by omega
  have h2 : (3 * c.val + kindTag v) % 3 = kindTag v := by omega
  simp only [unaryFormat, List.length_replicate]
  split <;> rename_i h
  · simp only [Option.some.injEq, Prod.mk.injEq]
    exact ⟨Fin.ext h1, by rw [h2, kindOfTag_tag]⟩
  · exact (h (by rw [h1]; exact c.isLt)).elim

/-- A schedule fits `cellRows` exactly when it fits `unaryFormat` of the same connections and windows: `Fits` reads
only a format's connections and its windows' buckets. -/
theorem cellRows_fits_iff {L K : ℕ} (B N : ℕ) (link : Fin L) (T : ℕ)
    (jitter : NetTiming.WardenSchedule (Fin N) → (Fin N → NetTiming.Arrivals (NetTiming.FrameCell (B - 9))) → ℕ → ℕ)
    (status : NetTiming.WardenSchedule (Fin N) → (Fin N → NetTiming.Arrivals (NetTiming.FrameCell (B - 9))) → ℕ →
      Fin K → NetTiming.Status) (S n : ℕ) (s : NetTiming.WardenSchedule (Fin N)) :
    (NetTiming.cellRows B N link T jitter status).Fits S n s ↔ (unaryFormat N T).Fits S n s :=
  Iff.rfl

end Properties.NetworkCertifier.RowsAudit

namespace Properties.NetworkCertifier

/-- **`SchedulesCoveredAt`**: the schedule an accepted run's records show is registered and fits the network
certifier's rows format, read at `unaryFormat` (`cellRows_fits_iff`). -/
theorem schedulesCoveredAt : SchedulesCoveredAt := by
  intro p ingress egress fails sum L K link jitter status h
  obtain ⟨h1, h2⟩ :=
    RowsAudit.auditCovered (RowsAudit.unaryFormat _ p.window) (RowsAudit.unaryFormat_wf _ _) rfl h
  refine ⟨h1, ?_⟩
  show _ ∧ _
  exact ⟨Set.mem_singleton _, (RowsAudit.cellRows_fits_iff _ _ link _ jitter status _ _ _).2 h2⟩

end Properties.NetworkCertifier

/-! ### The witnesses

`SchedulesCoveredAt`'s hypotheses hold at the deployed parameters (`warden.WardenParams`: `B = 256`, `r = 4`, `D = 10`,
16 lanes, `T = 600`, `egress_cells = 4096`): the audit accepts a window in which no connection is admitted
(`schedulesCoveredAt_honest`). It accepts a run that carries a connection too, at `r = D = 1`, one lane and `T = 2`
(`schedulesCoveredAt_honestLane`): connection 1 is admitted at bucket 0, pads its delay and ends in bucket 1. The
lane's 256-byte cells at the deployed parameters are past what the kernel evaluates in a declaration's budget. Its
conclusion fails for a schedule the audit never shows (`schedulesCoveredAt_nontrivial`): seventeen connections admitted
in bucket 0 at the deployed parameters put 68 frames in its 64 slots. The audit's verdicts are evaluated by the kernel
(`decide +kernel`) through `ok_of_view`, which reads an accepted summary off its fields. -/

namespace Properties.NetworkCertifier.RowsAudit

theorem ok_of_view {p : Params} {ingress egress fails : Array Record} {adm cells : Array ℕ} {i e : ℕ}
    (h : ((audit p ingress egress fails).toOption.map fun s =>
      (s.schedule.adm, s.schedule.cells, s.ingressCells, s.egressCells)) = some (adm, cells, i, e)) :
    audit p ingress egress fails = .ok ⟨⟨adm, cells⟩, i, e⟩ := by
  cases hx : audit p ingress egress fails with
  | error _ => rw [hx] at h; simp [Except.toOption] at h
  | ok s =>
    rw [hx] at h
    obtain ⟨⟨a, c⟩, i', e'⟩ := s
    simp [Except.toOption] at h
    obtain ⟨rfl, rfl, rfl, rfl⟩ := h
    rfl

end Properties.NetworkCertifier.RowsAudit

namespace Properties.NetworkCertifier

/-- **`SchedulesCoveredAt` meets the deployed parameters**: the audit accepts a window of empty rows. -/
theorem schedulesCoveredAt_honest :
    audit ⟨256, 4, 10, 16, 600, 4096⟩ #[⟨none, Array.replicate 600 (Array.replicate 64 none)⟩]
        #[⟨none, Array.replicate 600 (Array.replicate 64 none)⟩]
        #[⟨none, Array.replicate 600 (Array.replicate 16 none)⟩] =
      .ok ⟨⟨#[], #[]⟩, 0, 0⟩ :=
  RowsAudit.ok_of_view (by decide +kernel)

/-- **`SchedulesCoveredAt` meets a run with a connection**: at `r = D = 1`, one lane and `T = 2`, connection 1's
ingress is its `END` in bucket 0, and its egress pads its delay in bucket 0 and ends in bucket 1. -/
theorem schedulesCoveredAt_honestLane :
    audit ⟨10, 1, 1, 1, 2, 1⟩
        #[⟨none, #[#[some ((⟨.fin, 1, .empty⟩ : Cell).encode 10)], #[none]]⟩]
        #[⟨none, #[#[some ((Cell.pad 1).encode 10)], #[some ((⟨.fin, 1, .empty⟩ : Cell).encode 10)]]⟩]
        #[⟨none, #[#[none], #[none]]⟩] =
      .ok ⟨⟨#[0], #[1]⟩, 1, 2⟩ :=
  RowsAudit.ok_of_view (by decide +kernel)

/-- **`SchedulesCoveredAt` fails without the audit**: seventeen connections admitted in bucket 0 at the deployed
parameters, as a verifier that accepted everything could report, overload the 64 slots of the network certifier's
format of 600-bucket windows. -/
theorem schedulesCoveredAt_nontrivial :
    ¬ (wardenOf ⟨256, 4, 10, 16, 600, 4096⟩ 1 ⟨Array.replicate 17 0, Array.replicate 17 4⟩ ∈
      {s | (s.r, s.D) ∈ ({(4, 10)} : Set (ℕ × ℕ)) ∧
        (NetTiming.cellRows (L := 1) (K := 1) 256 (periodConns ⟨256, 4, 10, 16, 600, 4096⟩ 1) 0 600
          (fun _ _ _ => 0) (fun _ _ _ _ => .complete)).Fits 64 1 s}) := by
  intro hm
  simp only [Set.mem_ofPred_eq, NetTiming.RowsFormat.Fits] at hm
  exact absurd (hm.2.2 0) (by decide +kernel)

end Properties.NetworkCertifier

