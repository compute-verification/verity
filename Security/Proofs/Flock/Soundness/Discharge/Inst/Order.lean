import Proofs.Flock.Soundness.Discharge.Inst.Place
import Proofs.Flock.Soundness.ExecLookup
import Flock.ProvedScope

/-!
# Every column with a row is in the order, once the wires are acyclic

`order c` (`Inst/Rows.lean`) lists a template instance's logical inputs, then pass by pass every column whose row reads
only listed columns. `ProvedScope.wiresAcyclic` fills, from the units' leaf ports and the leaf cuts, each input port a wire
reaches once every input port its source port reads (`outReads`) is filled, and accepts when every port is filled and
every slot's rows read only port bits and earlier rows. Then every column with a row is in the order (`order_complete`):
- the order is a fixpoint: a column that is not in it is not ready (`order_stable`);
- a filled port's bits are in the order (`Filled`), by induction on the passes of the fill (`AvOk`): a wire-fed bit's
  row is the copy of its source bit (`addSelf` cancels its own entry), and the source port's bits are computed from the
  ports `outReads` names (`ok_mem_order`, a lookup slot's from its input port, `lookup_mem_order`);
- a computed row reads only filled ports' bits, earlier rows and the constant.
-/

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Inst

open Compose

/-! ## The order is a fixpoint -/

/-- No column left out of `l` is ready. -/
def Stable (c : Circuit) (l : List ℕ) : Prop := ∀ x ∈ cols c, x ∉ l → ready c l x = false

/-- The columns left out of `l`. -/
def left (c : Circuit) (l : List ℕ) : ℕ := ((cols c).filter fun y => !decide (y ∈ l)).length

theorem step_stable {c : Circuit} {l : List ℕ} (h : Stable c l) : step c l = l := by
  unfold step
  rw [List.filter_eq_nil_iff.2 fun x hx hp => ?_, List.append_nil]
  simp only [Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not] at hp
  rw [h x hx hp.1] at hp
  exact Bool.false_ne_true hp.2

theorem left_lt {c : Circuit} {l : List ℕ} {x : ℕ} (hx : x ∈ cols c) (hxl : x ∉ l) (hr : ready c l x = true) :
    left c (step c l) < left c l := by
  have hsub : ∀ y, y ∈ l → y ∈ step c l := fun y hy => List.mem_append_left _ hy
  have hxs : x ∈ step c l :=
    List.mem_append_right _ (List.mem_filter.2 ⟨hx, by simp [hxl, hr]⟩)
  have he : (cols c).filter (fun y => !decide (y ∈ step c l)) =
      ((cols c).filter fun y => !decide (y ∈ l)).filter fun y => !decide (y ∈ step c l) := by
    rw [List.filter_filter]
    apply List.filter_congr
    intro y _
    by_cases h : y ∈ step c l
    · simp [h]
    · simp [h, mt (hsub y) h]
  unfold left
  rw [he]
  exact List.length_filter_lt_length_iff_exists.2 ⟨x, List.mem_filter.2 ⟨hx, by simp [hxl]⟩, by simp [hxs]⟩

theorem iter_stable (c : Circuit) (l : List ℕ) :
    ∀ k, Stable c ((step c)^[k] l) ∨ left c ((step c)^[k] l) + k ≤ left c l := by
  intro k
  induction k with
  | zero => exact Or.inr (by simp)
  | succ k ih =>
    rw [Function.iterate_succ_apply']
    by_cases hs : Stable c ((step c)^[k] l)
    · rw [step_stable hs]; exact Or.inl hs
    · right
      have hk := ih.resolve_left hs
      unfold Stable at hs
      push Not at hs
      obtain ⟨x, hx, hxl, hr⟩ := hs
      have := left_lt hx hxl (by simpa using hr)
      omega

/-- **The order is a fixpoint**: no column left out of it is ready. -/
theorem order_stable (c : Circuit) : Stable c (order c) := by
  rcases iter_stable c (inCols c) (cols c).length with h | h
  · exact h
  · have hl : left c (inCols c) ≤ (cols c).length := List.length_filter_le _ _
    have h0 : left c (order c) = 0 := by unfold order; omega
    intro x hx hxl
    unfold left at h0
    have := List.length_eq_zero_iff.1 h0
    have hm : x ∈ (cols c).filter fun y => !decide (y ∈ order c) := List.mem_filter.2 ⟨hx, by simp [hxl]⟩
    rw [this] at hm
    exact absurd hm List.not_mem_nil

theorem mem_order_of_ready {c : Circuit} {x : ℕ} (hx : x ∈ cols c) (hr : ready c (order c) x = true) :
    x ∈ order c := by
  by_contra h
  rw [order_stable c x hx h] at hr
  exact Bool.false_ne_true hr

theorem code_mem_cols {c : Circuit} {n i o : ℕ} (hn : n < c.nets.size) (hi : i < c.perVu.getD n 0)
    (ho : o < netW c n) : code n i o ∈ cols c :=
  List.mem_flatMap.2 ⟨n, List.mem_range.2 hn, List.mem_flatMap.2 ⟨i, List.mem_range.2 hi,
    List.mem_map.2 ⟨o, List.mem_range.2 ho, rfl⟩⟩⟩

theorem inCols_sub_order (c : Circuit) : ∀ x ∈ inCols c, x ∈ order c := fun _ hx => (inv_order c).1.subset hx

theorem ready_some {c : Circuit} {l : List ℕ} {x : ℕ} {p : List ℕ × List ℕ} (h : row c x = some (some p))
    (hp : ∀ y ∈ p.1 ++ p.2, y ∈ l) : ready c l x = true := by
  unfold ready
  rw [h]
  simp only [List.all_eq_true, decide_eq_true_eq]
  exact hp

theorem row_code (c : Circuit) (n i o : ℕ) : row c (code n i o) = rowAt c n i o := by
  simp [row]

/-! ## Each column's row -/

section rows

variable {c : Circuit} {n i : ℕ} (hn : n < c.nets.size) (hsha : n ≠ c.shaNet) (hhm : n ≠ c.hmNet)
  (hi : i < c.perVu.getD n 0)
include hn hsha hhm hi

theorem rowAt_const_eq (hw : (netOf c n).constPos < netW c n) : rowAt c n i (netOf c n).constPos = some none := by
  unfold rowAt
  rw [ite_eq_left ⟨hn, hsha, hhm, hi, hw⟩, ite_eq_left rfl]

theorem rowAt_comp_eq {o : ℕ} (hlo : netIn c n ≤ o) (hhi : o < (netOf c n).constPos) (hw : o < netW c n) :
    rowAt c n i o = some (some (reads c n i (entA c n o), reads c n i (entB c n o))) := by
  unfold rowAt
  rw [ite_eq_left ⟨hn, hsha, hhm, hi, hw⟩, ite_eq_right (Nat.ne_of_lt hhi), ite_eq_left ⟨hlo, hhi⟩]

theorem rowAt_feed_eq {o s : ℕ} (hlo : o < netIn c n) (hc : netIn c n ≤ (netOf c n).constPos) (hw : o < netW c n)
    (hf : feed c n i o = some s) :
    rowAt c n i o = some (some (reads c n i (addSelf (entA c n o) o) ++ [s],
      reads c n i (addSelf (entB c n o) o) ++ [s])) := by
  unfold rowAt
  rw [ite_eq_left ⟨hn, hsha, hhm, hi, hw⟩, ite_eq_right (by omega), ite_eq_right (by omega), hf]

/-- **A slot's constant is in the order.** -/
theorem const_mem_order (hw : (netOf c n).constPos < netW c n) : code n i (netOf c n).constPos ∈ order c := by
  refine mem_order_of_ready (code_mem_cols hn hi hw) ?_
  unfold ready
  rw [row_code, rowAt_const_eq hn hsha hhm hi hw]

/-- **A slot's computed rows are in the order**, each that is `Good`, when every read of a `Good` computed row is the
constant, an earlier `Good` computed row, or a column already in the order. -/
theorem comp_mem_order (hw : (netOf c n).constPos < netW c n) (Good : ℕ → Prop)
    (hstep : ∀ r, Good r → netIn c n ≤ r → r < (netOf c n).constPos → ∀ j, j < netW c n →
      (entA c n r j = 1 ∨ entB c n r j = 1) →
      j = (netOf c n).constPos ∨ (netIn c n ≤ j ∧ j < r ∧ Good j) ∨ code n i j ∈ order c) :
    ∀ r, Good r → netIn c n ≤ r → r < (netOf c n).constPos → code n i r ∈ order c := by
  intro r
  induction r using Nat.strong_induction_on with
  | _ r ih =>
    intro hg hlo hhi
    have hrw : r < netW c n := lt_trans hhi hw
    have hrow : row c (code n i r) = some (some (reads c n i (entA c n r), reads c n i (entB c n r))) := by
      rw [row_code]; exact rowAt_comp_eq hn hsha hhm hi hlo hhi hrw
    refine mem_order_of_ready (code_mem_cols hn hi hrw) (ready_some hrow fun y hy => ?_)
    have hread : ∃ j, j < netW c n ∧ (entA c n r j = 1 ∨ entB c n r j = 1) ∧ y = code n i j := by
      rcases List.mem_append.1 hy with hy | hy <;>
      · obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hy
        obtain ⟨hj, he⟩ := List.mem_filter.1 hj
        exact ⟨j, List.mem_range.1 hj, by simp only [decide_eq_true_eq] at he; tauto, rfl⟩
    obtain ⟨j, hj, he, rfl⟩ := hread
    rcases hstep r hg hlo hhi j hj he with h | ⟨h1, h2, h3⟩ | h
    · rw [h]; exact const_mem_order hn hsha hhm hi hw
    · exact ih j h2 h3 h1 (lt_trans h2 hhi)
    · exact h

end rows

/-! ## Filled ports -/

/-- **Input port `p` of net `n`'s slot `i` is filled**: its bits are in the order. -/
def Filled (c : Circuit) (n i p : ℕ) : Prop :=
  ∀ t < ((portCols (netOf c n).inGroups).getD p default).2,
    code n i (((portCols (netOf c n).inGroups).getD p default).1 + t) ∈ order c

/-- Every bit of `m` is a bit of `m'`. -/
def Sub (m m' : ℕ) : Prop := ∀ b, m.testBit b = true → m'.testBit b = true

/-- **A mask of filled ports** in net `n`'s slot `i`: no bit 0 (no read of a column no port carries, nor out of order),
and bit `p + 1` only for a filled port `p`. -/
def Ok (c : Circuit) (n i m : ℕ) : Prop :=
  m.testBit 0 = false ∧ ∀ p < (portCols (netOf c n).inGroups).size, m.testBit (p + 1) = true → Filled c n i p

theorem Ok.mono {c : Circuit} {n i m m' : ℕ} (h : Ok c n i m') (hs : Sub m m') : Ok c n i m := by
  refine ⟨?_, fun p hp hb => h.2 p hp (hs _ hb)⟩
  cases hb : m.testBit 0
  · rfl
  · have := hs 0 hb
    rw [h.1] at this
    exact absurd this Bool.false_ne_true

theorem foldl_or {α : Type} (g : α → ℕ) : ∀ (l : List α) (init : ℕ),
    Sub init (l.foldl (fun m x => m ||| g x) init) ∧ ∀ x ∈ l, Sub (g x) (l.foldl (fun m x => m ||| g x) init)
  | [], _ => ⟨fun _ h => h, fun _ h => absurd h List.not_mem_nil⟩
  | a :: l, init => by
    simp only [List.foldl_cons]
    obtain ⟨h1, h2⟩ := foldl_or g l (init ||| g a)
    refine ⟨fun b hb => h1 b (by rw [Nat.testBit_or, hb, Bool.true_or]), fun x hx => ?_⟩
    rcases List.mem_cons.1 hx with rfl | hx
    · exact fun b hb => h1 b (by rw [Nat.testBit_or, hb, Bool.or_true])
    · exact h2 x hx

theorem foldl_mem_inv {α β : Type} (P : β → Prop) (f : β → α → β) :
    ∀ (l : List α) (b : β), P b → (∀ b, P b → ∀ a ∈ l, P (f b a)) → P (l.foldl f b)
  | [], _, hb, _ => hb
  | a :: l, b, hb, h => foldl_mem_inv P f l (f b a) (h b hb a List.mem_cons_self)
      fun b' hb' a' ha' => h b' hb' a' (List.mem_cons_of_mem _ ha')

theorem nat_fold_eq {β : Type} (k : ℕ) (f : ℕ → β → β) (init : β) :
    Nat.fold k (fun i _ acc => f i acc) init = (List.range k).foldl (fun acc i => f i acc) init := by
  induction k with
  | zero => rfl
  | succ k ih => rw [Nat.fold_succ, ih, List.range_succ, List.foldl_append]; rfl

theorem foldRow_eq {β : Type} (m : Sparse) (i : ℕ) (f : β → ℕ → β) (init : β) :
    ProvedScope.foldRow m i f init = (m.row i).toList.foldl f init := by
  unfold ProvedScope.foldRow Sparse.row
  rw [nat_fold_eq _ (fun t acc => f acc (rd32 m.data (m.start.getD i 0 + 4 * t))), Array.toList_map,
    Array.toList_range, List.foldl_map]

theorem getD_setIfInBounds {α : Type} (xs : Array α) (j x : ℕ) (v d : α) :
    (xs.setIfInBounds j v).getD x d = if j = x ∧ j < xs.size then v else xs.getD x d := by
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_setIfInBounds, Array.getD_eq_getD_getElem?]
  by_cases h1 : j = x
  · subst h1
    by_cases h2 : j < xs.size
    · simp [h2]
    · simp [h2]
  · simp [h1]

theorem getD_push_lt {α : Type} (xs : Array α) (v d : α) {i : ℕ} (h : i < xs.size) :
    (xs.push v).getD i d = xs.getD i d := by
  simp [Array.getD_eq_getD_getElem?, Array.getElem?_push, Nat.ne_of_lt h]

theorem getD_push_size {α : Type} (xs : Array α) (v d : α) : (xs.push v).getD xs.size d = v := by
  simp [Array.getD_eq_getD_getElem?]

theorem entry_mem {m : Sparse} {rows : Array (Array ℕ)} {width : ℕ} (h : Sparse.ofRows rows width = .ok m) {r j : ℕ}
    (he : sparseEntry m r j ≠ 0) : j ∈ (m.row r).toList := by
  rw [sparseEntry_ofRows h] at he
  by_contra hj
  rw [List.count_eq_zero.2 hj, Nat.cast_zero] at he
  exact he rfl

theorem zmod2_add_ne {a b : ZMod 2} (h : a + b = 1) : a ≠ 0 ∨ b ≠ 0 := by
  revert a b; decide

/-! ## What a stage's output ports read (`outReads`) -/

section masks

variable (net : Net)

/-- `outReads`' input rows. -/
abbrev mN : ℕ := min (net.inWords * WORD) net.constPos

/-- `outReads`' input mask: bit `p + 1` at port `p`'s bits, `1` at another input column. -/
def mIn : Array ℕ :=
  (Array.range (portCols net.inGroups).size).foldl (fun m p =>
      (Array.range ((portCols net.inGroups).getD p default).2).foldl (fun m x =>
        m.setIfInBounds (((portCols net.inGroups).getD p default).1 + x) (1 <<< (p + 1))) m)
    (Array.replicate (mN net) 1)

/-- `outReads`' mask of column `x`, against the computed rows' masks `rows`. -/
def mCol (rows : Array ℕ) (x : ℕ) : ℕ :=
  if x < mN net then (mIn net).getD x 1 else if x < net.constPos then rows.getD (x - mN net) 1
  else if x == net.constPos then 0 else 1

/-- `outReads`' union of two masks. -/
def mUnion (m y : ℕ) : ℕ := if y == 0 || y == m then m else if m == 0 then y else m ||| y

/-- The mask of computed row `mN + i`, against the masks `rows` of the rows before it. -/
def mRow (rows : Array ℕ) (i : ℕ) : ℕ :=
  ProvedScope.foldRow net.b (mN net + i) (fun m x => mUnion m (mCol net rows x))
    (ProvedScope.foldRow net.a (mN net + i) (fun m x => mUnion m (mCol net rows x)) 0)

/-- `outReads`' computed rows' masks. -/
def mRows : Array ℕ := Nat.fold (net.constPos - mN net) (fun i _ rows => rows.push (mRow net rows i)) #[]

/-- `outReads`' output ports' masks. -/
def mMasks : Array ℕ :=
  (portCols net.outGroups).map fun q => (Array.range q.2).foldl (fun m x => mUnion m (mCol net (mRows net) (q.1 + x))) 0

theorem outReads_eq (h : net.lookup.isSome = false) :
    ProvedScope.outReads net = ((mMasks net).map fun m => (Array.range (portCols net.inGroups).size).filter
      fun p => m.testBit (p + 1), (mRows net).all (· % 2 == 0) && (mMasks net).all (· % 2 == 0)) := by
  unfold ProvedScope.outReads
  simp only [h, Bool.false_eq_true, ↓reduceIte]
  rfl

theorem mUnion_eq (m y : ℕ) : mUnion m y = m ||| y := by
  unfold mUnion
  split
  · rename_i h
    simp only [Bool.or_eq_true, beq_iff_eq] at h
    rcases h with rfl | rfl
    · rw [Nat.or_zero]
    · rw [Nat.or_self]
  · split
    · rename_i _ h
      simp only [beq_iff_eq] at h
      subst h
      rw [Nat.zero_or]
    · rfl

theorem mCol_push (rows : Array ℕ) (m y : ℕ) (hy : y < mN net + rows.size ∨ net.constPos ≤ y) :
    mCol net (rows.push m) y = mCol net rows y := by
  unfold mCol
  split
  · rfl
  · split
    · rw [getD_push_lt _ _ _ (by omega)]
    · rfl

theorem mCol_even {rows : Array ℕ} {y : ℕ} (h : (mCol net rows y).testBit 0 = false) :
    y < mN net + rows.size ∨ y = net.constPos := by
  unfold mCol at h
  split at h
  · left; omega
  · split at h
    · by_contra hc
      rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (by omega)] at h
      exact absurd h (by decide)
    · split at h
      · rename_i h3
        exact Or.inr (by simpa using h3)
      · exact absurd h (by decide)

/-- The first `k` computed rows' masks: each read of one is the constant or an earlier column, its mask inside the
row's, unless the row's mask has bit 0. -/
def MJ (k : ℕ) (rows : Array ℕ) : Prop :=
  rows.size = k ∧ ∀ i < k, ∀ y ∈ (net.a.row (mN net + i)).toList ++ (net.b.row (mN net + i)).toList,
    (rows.getD i 0).testBit 0 = true ∨ ((y < mN net + i ∨ y = net.constPos) ∧ Sub (mCol net rows y) (rows.getD i 0))

theorem mRow_sub (rows : Array ℕ) (k : ℕ) :
    ∀ y ∈ (net.a.row (mN net + k)).toList ++ (net.b.row (mN net + k)).toList, Sub (mCol net rows y) (mRow net rows k) := by
  have hfun : (fun m x => mUnion m (mCol net rows x)) = fun m x => m ||| mCol net rows x := by
    funext m x; exact mUnion_eq _ _
  unfold mRow
  rw [foldRow_eq, foldRow_eq, hfun]
  intro y hy
  obtain ⟨-, ha2⟩ := foldl_or (mCol net rows) (net.a.row (mN net + k)).toList 0
  obtain ⟨hb1, hb2⟩ := foldl_or (mCol net rows) (net.b.row (mN net + k)).toList
    ((net.a.row (mN net + k)).toList.foldl (fun m x => m ||| mCol net rows x) 0)
  rcases List.mem_append.1 hy with hy | hy
  · exact fun b h => hb1 b (ha2 y hy b h)
  · exact hb2 y hy

theorem mJ_push {k : ℕ} {rows : Array ℕ} (h : MJ net k rows) : MJ net (k + 1) (rows.push (mRow net rows k)) := by
  obtain ⟨hs, hJ⟩ := h
  refine ⟨by rw [Array.size_push, hs], fun i hi y hy => ?_⟩
  by_cases hik : i = k
  · subst hik
    rw [← hs, getD_push_size, hs]
    by_cases hodd : (mRow net rows i).testBit 0 = true
    · exact Or.inl hodd
    · right
      have hev : (mCol net rows y).testBit 0 = false := by
        cases h : (mCol net rows y).testBit 0
        · rfl
        · exact absurd (mRow_sub net rows i y hy 0 h) hodd
      have hlt := mCol_even net hev
      rw [hs] at hlt
      refine ⟨hlt, ?_⟩
      rw [mCol_push net rows _ y (by rw [hs]; omega)]
      exact mRow_sub net rows i y hy
  · have hik' : i < k := by omega
    rw [getD_push_lt _ _ _ (by rw [hs]; exact hik')]
    rcases hJ i hik' y hy with h | ⟨hlt, hs'⟩
    · exact Or.inl h
    · right
      refine ⟨hlt, ?_⟩
      rw [mCol_push net rows _ y (by rw [hs]; omega)]
      exact hs'

theorem mRows_J : MJ net (net.constPos - mN net) (mRows net) := by
  have he := nat_fold_eq (net.constPos - mN net) (fun i rows => rows.push (mRow net rows i)) #[]
  unfold mRows
  rw [he]
  generalize net.constPos - mN net = K
  induction K with
  | zero => exact ⟨rfl, fun i hi => absurd hi (Nat.not_lt_zero _)⟩
  | succ k ih =>
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
    exact mJ_push net ih

theorem mask_sub (q : ℕ × ℕ) {x : ℕ} (hx : x < q.2) :
    Sub (mCol net (mRows net) (q.1 + x))
      ((Array.range q.2).foldl (fun m x => mUnion m (mCol net (mRows net) (q.1 + x))) 0) := by
  have hfun : (fun m x => mUnion m (mCol net (mRows net) (q.1 + x))) =
      fun m x => m ||| mCol net (mRows net) (q.1 + x) := by
    funext m x; exact mUnion_eq _ _
  rw [hfun, ← Array.foldl_toList, Array.toList_range]
  exact (foldl_or (fun x => mCol net (mRows net) (q.1 + x)) _ 0).2 x (List.mem_range.2 hx)

/-- An input column's mask is `1`, or bit `p + 1` alone, for a port `p` it is a bit of. -/
def MInOk (m : Array ℕ) : Prop :=
  ∀ x, m.getD x 1 = 1 ∨ ∃ p t, p < (portCols net.inGroups).size ∧ t < ((portCols net.inGroups).getD p default).2 ∧
    x = ((portCols net.inGroups).getD p default).1 + t ∧ m.getD x 1 = 2 ^ (p + 1)

theorem mIn_ok : MInOk net (mIn net) := by
  unfold mIn
  rw [← Array.foldl_toList, Array.toList_range]
  refine foldl_mem_inv (MInOk net) _ _ _ (fun x => Or.inl ?_) fun m hm p hp => ?_
  · simp only [Array.getD_eq_getD_getElem?, Array.getElem?_replicate]
    split <;> rfl
  · rw [← Array.foldl_toList, Array.toList_range]
    refine foldl_mem_inv (MInOk net) _ _ _ hm fun m' hm' t ht x => ?_
    rw [Nat.one_shiftLeft, getD_setIfInBounds]
    split
    · rename_i h
      exact Or.inr ⟨p, t, List.mem_range.1 hp, List.mem_range.1 ht, h.1.symm, rfl⟩
    · exact hm' x

end masks

section built

variable {c : Circuit} {n i : ℕ} (hn : n < c.nets.size) (hsha : n ≠ c.shaNet) (hhm : n ≠ c.hmNet)
  (hi : i < c.perVu.getD n 0)
include hn hsha hhm hi

/-- **A built slot's columns of filled ports' masks are in the order**: an input column of a filled port, a computed row
reading only filled ports' bits, earlier such rows and the constant, or the constant. -/
theorem ok_mem_order (hb : Built (netOf c n)) (hc : netIn c n ≤ (netOf c n).constPos)
    (hw : (netOf c n).constPos < netW c n) :
    ∀ x, Ok c n i (mCol (netOf c n) (mRows (netOf c n)) x) → code n i x ∈ order c := by
  have hN : mN (netOf c n) = netIn c n := min_eq_left hc
  have hin : ∀ x < netIn c n, Ok c n i ((mIn (netOf c n)).getD x 1) → code n i x ∈ order c := by
    intro x _ hok
    rcases mIn_ok (netOf c n) x with h1 | ⟨p, t, hp, ht, rfl, h2⟩
    · rw [h1] at hok; exact absurd hok.1 (by decide)
    · rw [h2] at hok
      exact hok.2 p hp (by rw [Nat.testBit_two_pow]; simp) t ht
  have hcol_in : ∀ x < netIn c n, mCol (netOf c n) (mRows (netOf c n)) x = (mIn (netOf c n)).getD x 1 := by
    intro x hx
    unfold mCol
    rw [ite_eq_left (by rw [hN]; exact hx)]
  obtain ⟨hsz, hJ⟩ := mRows_J (netOf c n)
  have hcol_comp : ∀ r, netIn c n ≤ r → r < (netOf c n).constPos →
      mCol (netOf c n) (mRows (netOf c n)) r = (mRows (netOf c n)).getD (r - netIn c n) 0 := by
    intro r hlo hhi
    unfold mCol
    rw [ite_eq_right (by rw [hN]; omega), ite_eq_left hhi, hN, Array.getD_eq_getD_getElem?,
      Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (by rw [hsz, hN]; omega)]
    rfl
  obtain ⟨ra, rb, -, hra, hrb⟩ := built_rowOrder hb
  have hlk : (netOf c n).lookup = none := by
    obtain ⟨_, _, _, _, _, _, _, ho⟩ := hb
    exact (Net.ofRows_ok ho).2.2.2.2.2.2.2.2.2.2.2.2.1
  have hcomp := comp_mem_order hn hsha hhm hi hw (fun r => Ok c n i (mCol (netOf c n) (mRows (netOf c n)) r))
    (by
      intro r hg hlo hhi j _ he
      have hj : j ∈ ((netOf c n).a.row (mN (netOf c n) + (r - netIn c n))).toList ++
          ((netOf c n).b.row (mN (netOf c n) + (r - netIn c n))).toList := by
        rw [hN, Nat.add_sub_cancel' hlo]
        rcases he with he | he
        · exact List.mem_append_left _ (entry_mem hra (by unfold entA at he; rw [he]; decide))
        · refine List.mem_append_right _ (entry_mem hrb ?_)
          unfold entB at he
          rw [hlk, lookupEntry, add_zero] at he
          rw [he]; decide
      rw [hcol_comp r hlo hhi] at hg
      rcases hJ (r - netIn c n) (by rw [hN]; omega) j hj with h | ⟨hlt, hs⟩
      · rw [hg.1] at h; exact absurd h Bool.false_ne_true
      · have hok := hg.mono hs
        rw [hN, Nat.add_sub_cancel' hlo] at hlt
        rcases hlt with hlt | hlt
        · by_cases hj' : netIn c n ≤ j
          · exact Or.inr (Or.inl ⟨hj', hlt, hok⟩)
          · exact Or.inr (Or.inr (hin j (by omega) (by rw [← hcol_in j (by omega)]; exact hok)))
        · exact Or.inl hlt)
  intro x hok
  by_cases h1 : x < netIn c n
  · exact hin x h1 (by rw [← hcol_in x h1]; exact hok)
  by_cases h2 : x < (netOf c n).constPos
  · exact hcomp x hok (by omega) h2
  by_cases h3 : x = (netOf c n).constPos
  · rw [h3]; exact const_mem_order hn hsha hhm hi hw
  · unfold mCol at hok
    rw [ite_eq_right (by rw [hN]; exact h1), ite_eq_right h2, ite_eq_right (by simpa using h3)] at hok
    exact absurd hok.1 (by decide)

end built

/-! ## A lookup slot -/

/-- A decoder of two bits or more ends with its combining rows, past the rows it was given. -/
theorem decode_first (bd : Lookup.Builder) (bits : List ℕ) (fuel : ℕ) (h2 : 2 ≤ bits.length)
    (hlen : bits.length ≤ fuel + 1) : bd.a.size ≤ (Lookup.decode bd bits fuel).2.2 := by
  cases fuel with
  | zero => omega
  | succ f =>
    have hs : ∀ b, bits ≠ [b] := by rintro b rfl; simp at h2
    rw [decode_combine _ _ _ hs]
    dsimp only
    have htl : (bits.take (bits.length / 2)).length = bits.length / 2 := by simp; omega
    have hdl : (bits.drop (bits.length / 2)).length = bits.length - bits.length / 2 := by simp
    obtain ⟨A1, -⟩ := decode_spec (K := ZMod 2) (fun _ => 0) f bd (bits.take (bits.length / 2))
      (by intro e; have := congrArg List.length e; rw [htl] at this; simp at this; omega) (by rw [htl]; omega)
    obtain ⟨A2, -⟩ := decode_spec (K := ZMod 2) (fun _ => 0) f (Lookup.decode bd (bits.take (bits.length / 2)) f).1
      (bits.drop (bits.length / 2))
      (by intro e; have := congrArg List.length e; rw [hdl] at this; simp at this; omega) (by rw [hdl]; omega)
    exact A1.size.1.trans A2.size.1

/-- **What a lookup slot's rows read**: its input port is its first `nb` columns, each a self row; a computed row reads
the constant or an earlier column, an input column only of the port; and its output port is computed rows. -/
theorem lookup_facts {nm : String} {tb : ByteArray} {nb lo bits : ℕ} {net : Net}
    (hb : Lookup.build nm tb nb lo bits = .ok net) (hlog : net.unitLog ≤ 32) :
    portCols net.inGroups = #[(0, nb)] ∧ net.inWords * WORD = 128 ∧
      (∀ x < nb, ∀ j, sparseEntry net.a x j = (if j = x then 1 else 0) ∧
        sparseEntry net.b x j + lookupEntry net.lookup x j = if j = x then 1 else 0) ∧
      (∀ r, 128 ≤ r → r < net.constPos → ∀ j,
        (sparseEntry net.a r j = 1 ∨ sparseEntry net.b r j + lookupEntry net.lookup r j = 1) →
        j = net.constPos ∨ (j < r ∧ (j < 128 → j < nb))) ∧
      ∃ o0, portCols net.outGroups = #[(o0, bits)] ∧ 128 ≤ o0 ∧ o0 + bits < net.constPos := by
  obtain ⟨-, hig, hiw, -⟩ := build_facts hb
  obtain ⟨hlo, hln, hb32, hn, -, d1, d2, r3, hd1, hd2, -, -, ⟨k, hr3a, hr3b, hr3m⟩, hA, hB, -, hconst, hout, hlk,
    nm', hO⟩ := build_spec hb
  have hw : 2 ^ net.unitLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by norm_num) hlog
  have I0 := inputRows_eq nb
  have hI0a : (inputRows nb).a.size = 128 := by rw [I0]; simp
  have hI0b : (inputRows nb).b.size = 128 := by rw [I0]; simp
  have hlo0 : List.range lo ≠ [] := by simp; omega
  have hhi0 : (List.range (nb - lo)).map (lo + ·) ≠ [] := by simp; omega
  obtain ⟨A1, -, F1, -⟩ := decode_spec (K := ZMod 2) (fun _ => 0) lo (inputRows nb) (List.range lo) hlo0 (by simp)
  rw [← hd1] at A1 F1
  simp only [List.length_range] at F1
  obtain ⟨A2, -⟩ := decode_spec (K := ZMod 2) (fun _ => 0) (nb - lo) d1.1 ((List.range (nb - lo)).map (lo + ·)) hhi0
    (by simp)
  rw [← hd2] at A2
  have hfirst : 128 ≤ d1.2.2 := by
    have := decode_first (inputRows nb) (List.range lo) lo (by simp; omega) (by simp)
    rw [← hd1, hI0a] at this
    exact this
  have hlotop : d1.2.2 + 2 ^ lo = d1.1.a.size := (F1 hlo).1
  have PE := productRows_eq d2.2.1 bits d2.1
  set P := productRows d2.2.1 bits d2.1 with hP
  have AP : Appends d2.1 P := Appends.of_eq _ _ (by rw [PE]) (by rw [PE]) (by simp)
  have A3 : Appends P r3 := Appends.of_eq _ _ hr3a hr3b rfl
  set F := outputRows d2.1.a.size d2.2.1.size bits r3 with hF
  have FE := outputRows_eq d2.1.a.size d2.2.1.size bits r3
  rw [← hF] at FE
  have AF : Appends r3 F := Appends.of_eq _ _ (by rw [FE]) (by rw [FE]) (by simp)
  have e0 : (inputRows nb).a.size = (inputRows nb).b.size := by rw [hI0a, hI0b]
  have AF0 := A1.trans (A2.trans (AP.trans (A3.trans AF)))
  have eF := AF0.size.2 e0
  have hFs : F.a.size = r3.a.size + 128 := by rw [FE]; simp
  have h2s := A2.size.1
  have hr3 : 128 ≤ r3.a.size := by have := (A1.trans (A2.trans (AP.trans A3))).size.1; omega
  have hra := ofRows_row hA hw
  have hrb := ofRows_row hB hw
  have hord := checkOrder_ok hO
  have hin : ∀ x < 128, (fixKonst F.a.size (F.a.push #[F.a.size])).getD x #[] = (if x < nb then #[x] else #[]) ∧
      (fixKonst F.a.size (F.b.push #[F.a.size])).getD x #[] = (if x < nb then #[x] else #[]) := by
    intro x hx
    have e := AF0.getD e0 (by rw [hI0a]; exact hx)
    have hK : x ≠ Lookup.KONST := by unfold Lookup.KONST; omega
    rw [fixKonst_getD _ _ _ (by omega), fixKonst_getD _ _ _ (by omega), e.1, e.2, I0]
    by_cases hxn : x < nb
    · simp [Array.getD_eq_getD_getElem?, hx, hxn, hK]
    · simp [Array.getD_eq_getD_getElem?, hx, hxn]
  have hlk0 : ∀ r j, lookupEntry net.lookup r j ≠ 0 → d2.1.a.size ≤ r ∧ d1.2.2 ≤ j ∧ j < d1.2.2 + 2 ^ lo := by
    intro r j h
    rw [hlk] at h
    simp only [lookupEntry] at h
    split at h
    · rename_i hc; exact ⟨hc.1, hc.2.2.1, hc.2.2.2.1⟩
    · exact absurd rfl h
  refine ⟨by rw [hig]; rfl, by rw [hiw]; rfl, fun x hx j => ?_, fun r hr1 hr2 j hj => ?_, ?_⟩
  · have hl0 : lookupEntry net.lookup x j = 0 := by
      by_contra h
      have := hlk0 x j h
      omega
    rw [hl0, add_zero, sparseEntry_ofRows hA, sparseEntry_ofRows hB, hra, hrb, (hin x (by omega)).1,
      (hin x (by omega)).2, ite_eq_left hx, count_singleton_zmod]
    exact ⟨rfl, rfl⟩
  · rw [hconst] at hr2 ⊢
    have hord' : j ∈ ((fixKonst F.a.size (F.a.push #[F.a.size])).getD r #[]).toList ++
        ((fixKonst F.a.size (F.b.push #[F.a.size])).getD r #[]).toList → j = F.a.size ∨ (j < r ∧ (j < 128 → j < nb)) := by
      intro hm
      rcases hord r hr1 hr2 j hm with h | ⟨hlt, hpad⟩
      · exact Or.inl h
      · refine Or.inr ⟨hlt, fun h128 => ?_⟩
        by_contra hnb
        apply hpad
        refine ⟨h128, ?_, ?_⟩
        · rw [(hin j h128).1, ite_eq_right hnb]; rfl
        · rw [(hin j h128).2, ite_eq_right hnb]; rfl
    rcases hj with hj | hj
    · have := entry_mem hA (r := r) (j := j) (by rw [hj]; decide)
      rw [hra] at this
      exact hord' (List.mem_append_left _ this)
    · rcases zmod2_add_ne hj with hj | hj
      · have := entry_mem hB (r := r) (j := j) hj
        rw [hrb] at this
        exact hord' (List.mem_append_right _ this)
      · obtain ⟨h1, h2, h3⟩ := hlk0 r j hj
        exact Or.inr ⟨by omega, fun _ => by omega⟩
  · refine ⟨r3.a.size / 128 * WORD, by rw [hout]; rfl, ?_, ?_⟩
    · show 128 ≤ r3.a.size / 128 * 128
      omega
    · rw [hconst]
      show r3.a.size / 128 * 128 + bits < F.a.size
      omega

section lookup

variable {c : Circuit} {n i : ℕ} (hn : n < c.nets.size) (hsha : n ≠ c.shaNet) (hhm : n ≠ c.hmNet)
  (hi : i < c.perVu.getD n 0)
include hn hsha hhm hi

/-- **A lookup slot whose input port is filled has its computed rows in the order.** -/
theorem lookup_mem_order {nm : String} {tb : ByteArray} {nb lo bits : ℕ}
    (hb : Lookup.build nm tb nb lo bits = .ok (netOf c n)) (hlog : (netOf c n).unitLog ≤ 32)
    (hw : (netOf c n).constPos < netW c n) (hf : Filled c n i 0) :
    ∀ r, netIn c n ≤ r → r < (netOf c n).constPos → code n i r ∈ order c := by
  obtain ⟨hpc, hiw, -, hrd, -⟩ := lookup_facts hb hlog
  have hnetIn : netIn c n = 128 := hiw
  have hf' : ∀ t < nb, code n i t ∈ order c := by
    intro t ht
    have := hf t (by rw [hpc]; exact ht)
    rw [hpc] at this
    simpa using this
  intro r h1 h2
  refine comp_mem_order hn hsha hhm hi hw (fun _ => True) (fun r _ hlo hhi j _ he => ?_) r trivial h1 h2
  rw [hnetIn] at hlo
  rcases hrd r hlo hhi j he with h | ⟨hlt, h128⟩
  · exact Or.inl h
  · by_cases hj : 128 ≤ j
    · exact Or.inr (Or.inl ⟨by rw [hnetIn]; exact hj, hlt, trivial⟩)
    · exact Or.inr (Or.inr (hf' j (h128 (by omega))))

end lookup

/-! ## A net with slots -/

/-- **What the order's completeness reads of a net with slots**: its rows in `rd32`'s words, its input rows before its
constant and its constant in its slot, its input ports inside its input rows and apart, and built by `Net.ofRows` or
`Lookup.build`. -/
structure NetFit (c : Circuit) (n : ℕ) : Prop where
  log : (netOf c n).unitLog ≤ 32
  inConst : netIn c n ≤ (netOf c n).constPos
  constW : (netOf c n).constPos < netW c n
  ports : ∀ q ∈ (portCols (netOf c n).inGroups).toList, q.1 + q.2 ≤ netIn c n
  nodup : (portBits (netOf c n).inGroups).Nodup
  kind : Built (netOf c n) ∨ ∃ nm tb nb lo bits, Lookup.build nm tb nb lo bits = .ok (netOf c n)

theorem port_mem {gs : Array Flock.Group} {p t : ℕ} (hp : p < (portCols gs).size) (ht : t < (portCols gs)[p]!.2) :
    (portCols gs)[p]!.1 + t ∈ portBits gs := by
  unfold portBits
  refine List.mem_flatMap.2 ⟨(portCols gs)[p]!, ?_, List.mem_range'_1.2 ⟨by omega, by omega⟩⟩
  rw [getElem!_pos (portCols gs) p hp]
  exact Array.getElem_mem_toList hp

/-- **An input port bit's own rows are self rows**, `[x]·[x]`. -/
theorem port_rows {c : Circuit} {n : ℕ} (hF : NetFit c n) {o : ℕ} (ho : o ∈ portBits (netOf c n).inGroups) (j : ℕ) :
    entA c n o j = (if j = o then 1 else 0) ∧ entB c n o j = if j = o then 1 else 0 := by
  rcases hF.kind with hb | ⟨nm, tb, nb, lo, bits, hb⟩
  · have hlk : (netOf c n).lookup = none := by
      obtain ⟨_, _, _, _, _, _, _, ho⟩ := hb
      exact (Net.ofRows_ok ho).2.2.2.2.2.2.2.2.2.2.2.2.1
    obtain ⟨ha, hb'⟩ := Aliased.port_self hb hF.log ho j
    refine ⟨ha, ?_⟩
    unfold entB
    rw [hlk, lookupEntry, add_zero]
    exact hb'
  · obtain ⟨hpc, -, hself, -⟩ := lookup_facts hb hF.log
    have hon : o < nb := by
      unfold portBits at ho
      rw [hpc] at ho
      simpa using ho
    exact hself o hon j

theorem reads_addSelf {c : Circuit} {n i o : ℕ} {e : ℕ → ZMod 2} (he : ∀ j, e j = if j = o then 1 else 0) :
    reads c n i (addSelf e o) = [] := by
  unfold reads
  rw [List.filter_eq_nil_iff.2 fun j _ hj => ?_, List.map_nil]
  simp only [decide_eq_true_eq, addSelf, he] at hj
  by_cases h : j = o
  · rw [ite_eq_left h] at hj; exact absurd hj (by decide)
  · rw [ite_eq_right h] at hj; exact absurd hj (by decide)

/-! ## The fill `wiresAcyclic` runs -/

section fill

variable (c : Circuit)

/-- `wiresAcyclic`'s slots of net `n`: none for the rows' circuits. -/
def fSlots (n : ℕ) : ℕ := if (n == c.shaNet || n == c.hmNet) = true then 0 else c.perVu.getD n 0

/-- `wiresAcyclic`'s input ports per net. -/
def fPorts : Array ℕ := c.nets.map fun e => (portCols e.2.inGroups).size

/-- One step of `wiresAcyclic`'s port numbering. -/
def fBaseStep (b : Array ℕ) (n : ℕ) : Array ℕ := b.push (b.back?.getD 0 + fSlots c n * (fPorts c).getD n 0)

/-- `wiresAcyclic`'s first port number of each net. -/
def fBase : Array ℕ := (Array.range c.nets.size).foldl (fBaseStep c) #[0]

/-- `wiresAcyclic`'s number of input port `p` of net `n`'s slot `i`. -/
def fNode (n i p : ℕ) : ℕ := (fBase c).getD n 0 + i * (fPorts c).getD n 0 + p

/-- `wiresAcyclic`'s input ports: of a slot outside the rows' circuits. -/
def fIn (n i p : ℕ) : Bool := decide (i < fSlots c n) && decide (p < (fPorts c).getD n 0)

/-- `wiresAcyclic`'s reads of each net's output ports. -/
def fReads : Array (Array (Array ℕ) × Bool) :=
  (Array.range c.nets.size).map fun n => if fSlots c n == 0 then (#[], true) else ProvedScope.outReads (c.nets.getD n default).2

/-- `wiresAcyclic`'s ports filled to start: the units' leaf ports and the leaf cuts. -/
def fAv0 : Array Bool :=
  c.leafCuts.foldl (fun av e => if fIn c c.unitNet e.1 e.2.1 then av.set! (fNode c c.unitNet e.1 e.2.1) true else av)
    ((Array.range c.upv).foldl (fun av u => (Array.range (c.unit.inGroups.getD 0 default).bits.size).foldl (fun av p =>
      if fIn c c.unitNet u p then av.set! (fNode c c.unitNet u p) true else av) av)
      (Array.replicate ((fBase c).getD c.nets.size 0) false))

/-- `wiresAcyclic`'s wire ready to fill its port: its source slot's ports its source port reads are filled. -/
def fReady (av : Array Bool) (w : Wire) : Bool :=
  decide (w.src.2.1 < fSlots c w.src.1) && match ((fReads c).getD w.src.1 (#[], false)).1[w.src.2.2]? with
    | some ps => ps.all fun q => av.getD (fNode c w.src.1 w.src.2.1 q) false
    | none => false

/-- `wiresAcyclic`'s port a wire fills. -/
def fDst (w : Wire) : Option ℕ :=
  if fIn c w.dst.1 w.dst.2.1 w.dst.2.2 then some (fNode c w.dst.1 w.dst.2.1 w.dst.2.2) else none

theorem wiresAcyclic_eq : ProvedScope.wiresAcyclic c =
    ((ProvedScope.fillLoop (ProvedScope.fillPass (fReady c) (fDst c) c.wires) (c.wires.size + 1) (fAv0 c)).all id &&
      (Array.range c.nets.size).all fun n => fSlots c n == 0 || ((fReads c).getD n (#[], false)).2) := rfl

theorem fSlots_pos {n : ℕ} (h : 0 < fSlots c n) :
    n ≠ c.shaNet ∧ n ≠ c.hmNet ∧ fSlots c n = c.perVu.getD n 0 := by
  unfold fSlots at h ⊢
  split at h
  · omega
  · rename_i hr
    rw [ite_eq_right hr]
    simp only [Bool.or_eq_true, beq_iff_eq, not_or] at hr
    exact ⟨hr.1, hr.2, rfl⟩

theorem fSlots_of {n : ℕ} (hsha : n ≠ c.shaNet) (hhm : n ≠ c.hmNet) : fSlots c n = c.perVu.getD n 0 := by
  unfold fSlots
  rw [ite_eq_right (by simp [hsha, hhm])]

theorem fPorts_getD (n : ℕ) :
    (fPorts c).getD n 0 = if n < c.nets.size then (portCols (netOf c n).inGroups).size else 0 := by
  unfold fPorts
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_map]
  split
  · rename_i h
    rw [Array.getElem?_eq_getElem h]
    simp [netOf, Array.getD_eq_getD_getElem?, h]
  · rename_i h
    rw [Array.getElem?_eq_none (by omega)]
    rfl

/-- The ports numbered before net `k`'s. -/
def pre (k : ℕ) : ℕ := ((List.range k).map fun m => fSlots c m * (fPorts c).getD m 0).sum

theorem pre_succ (k : ℕ) : pre c (k + 1) = pre c k + fSlots c k * (fPorts c).getD k 0 := by
  unfold pre
  rw [List.range_succ, List.map_append, List.sum_append]
  simp

theorem pre_mono {a b : ℕ} (h : a ≤ b) : pre c a ≤ pre c b := by
  induction b, h using Nat.le_induction with
  | base => exact le_refl _
  | succ b _ ih => rw [pre_succ]; omega

theorem fBase_spec : (fBase c).size = c.nets.size + 1 ∧ ∀ k ≤ c.nets.size, (fBase c).getD k 0 = pre c k := by
  unfold fBase
  rw [← Array.foldl_toList, Array.toList_range]
  suffices h : ∀ K, ((List.range K).foldl (fBaseStep c) #[0]).size = K + 1 ∧
      ∀ k ≤ K, ((List.range K).foldl (fBaseStep c) #[0]).getD k 0 = pre c k from h _
  intro K
  induction K with
  | zero =>
    refine ⟨rfl, fun k hk => ?_⟩
    obtain rfl : k = 0 := by omega
    rfl
  | succ K ih =>
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
    obtain ⟨hs, hg⟩ := ih
    generalize (List.range K).foldl (fBaseStep c) #[0] = b at hs hg
    have hback : b.back?.getD 0 = pre c K := by
      rw [Array.back?_eq_getElem?, hs, Nat.add_sub_cancel, ← Array.getD_eq_getD_getElem?]
      exact hg K le_rfl
    unfold fBaseStep
    refine ⟨by rw [Array.size_push, hs], fun k hk => ?_⟩
    rcases Nat.lt_or_ge k (K + 1) with hk' | hk'
    · rw [getD_push_lt _ _ _ (by omega)]
      exact hg k (by omega)
    · obtain rfl : k = K + 1 := by omega
      have hsz := getD_push_size b (b.back?.getD 0 + fSlots c K * (fPorts c).getD K 0) 0
      rw [hs] at hsz
      rw [hsz, hback, pre_succ]

theorem fIn_spec {n i p : ℕ} (h : fIn c n i p = true) :
    n < c.nets.size ∧ n ≠ c.shaNet ∧ n ≠ c.hmNet ∧ i < c.perVu.getD n 0 ∧
      p < (portCols (netOf c n).inGroups).size := by
  unfold fIn at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨h1, h2⟩ := h
  rw [fPorts_getD] at h2
  split at h2
  · rename_i hn
    obtain ⟨hs, hh, he⟩ := fSlots_pos c (n := n) (by omega)
    rw [he] at h1
    exact ⟨hn, hs, hh, h1, h2⟩
  · omega

theorem fIn_of {n i p : ℕ} (hn : n < c.nets.size) (hsha : n ≠ c.shaNet) (hhm : n ≠ c.hmNet)
    (hi : i < c.perVu.getD n 0) (hp : p < (portCols (netOf c n).inGroups).size) : fIn c n i p = true := by
  unfold fIn
  rw [fSlots_of c hsha hhm, fPorts_getD, ite_eq_left hn]
  simp only [Bool.and_eq_true, decide_eq_true_eq]
  exact ⟨hi, hp⟩

theorem fNode_lt {n i p : ℕ} (h : fIn c n i p = true) : pre c n ≤ fNode c n i p ∧ fNode c n i p < pre c (n + 1) := by
  obtain ⟨hn, -, -, -, -⟩ := fIn_spec c h
  unfold fIn at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  unfold fNode
  rw [(fBase_spec c).2 n hn.le, pre_succ]
  have := Nat.mul_le_mul_right ((fPorts c).getD n 0) (show i + 1 ≤ fSlots c n from h.1)
  rw [Nat.succ_mul] at this
  constructor <;> omega

theorem mul_add_inj {k a b a' b' : ℕ} (hb : b < k) (hb' : b' < k) (h : a * k + b = a' * k + b') : a = a' ∧ b = b' := by
  have hk : 0 < k := by omega
  have h1 : (b + a * k) / k = (b' + a' * k) / k := by rw [Nat.add_comm b, Nat.add_comm b', h]
  rw [Nat.add_mul_div_right _ _ hk, Nat.add_mul_div_right _ _ hk, Nat.div_eq_of_lt hb, Nat.div_eq_of_lt hb'] at h1
  simp only [Nat.zero_add] at h1
  subst h1
  exact ⟨rfl, by omega⟩

theorem fNode_inj {n i p n' i' p' : ℕ} (h : fIn c n i p = true) (h' : fIn c n' i' p' = true)
    (he : fNode c n i p = fNode c n' i' p') : n = n' ∧ i = i' ∧ p = p' := by
  have b := fNode_lt c h
  have b' := fNode_lt c h'
  have hnn : n = n' := by
    by_contra hne
    rcases Nat.lt_or_gt_of_ne hne with hlt | hlt
    · have := pre_mono c (show n + 1 ≤ n' by omega); omega
    · have := pre_mono c (show n' + 1 ≤ n by omega); omega
  subst hnn
  unfold fIn at h h'
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h h'
  unfold fNode at he
  have he' : i * (fPorts c).getD n 0 + p = i' * (fPorts c).getD n 0 + p' := by omega
  exact ⟨rfl, mul_add_inj h.2 h'.2 he'⟩

theorem fNode_size {n i p : ℕ} (h : fIn c n i p = true) : fNode c n i p < pre c c.nets.size :=
  lt_of_lt_of_le (fNode_lt c h).2 (pre_mono c (fIn_spec c h).1)

/-- **The fill's invariant**: every filled port's bits are in the order. -/
def AvOk (av : Array Bool) : Prop :=
  av.size = pre c c.nets.size ∧ ∀ n i p, fIn c n i p = true → av.getD (fNode c n i p) false = true → Filled c n i p

theorem avOk_set {av : Array Bool} (h : AvOk c av) {n i p : ℕ} (hin : fIn c n i p = true) (hf : Filled c n i p) :
    AvOk c (av.set! (fNode c n i p) true) := by
  refine ⟨by rw [Array.set!_eq_setIfInBounds, Array.size_setIfInBounds, h.1], fun n' i' p' hin' hg => ?_⟩
  rw [Array.set!_eq_setIfInBounds, getD_setIfInBounds] at hg
  split at hg
  · rename_i he
    obtain ⟨rfl, rfl, rfl⟩ := fNode_inj c hin hin' he.1
    exact hf
  · exact h.2 n' i' p' hin' hg

variable {c}

/-- **A unit's input port no wire reaches is filled**: its bits are logical inputs. -/
theorem in_filled (hF : NetFit c c.unitNet) {u p : ℕ} (hu : u < c.upv)
    (hp : p < (portCols (netOf c c.unitNet).inGroups).size)
    (hnw : (c.wires.toList.filter fun w => decide (w.dst = (c.unitNet, u, p))) = []) : Filled c c.unitNet u p := by
  intro t ht
  rw [FlockSoundness.Layout.getD_portCols hp] at ht ⊢
  have hq := hF.ports (portCols (netOf c c.unitNet).inGroups)[p]!
    (by rw [getElem!_pos (portCols _) p hp]; exact Array.getElem_mem_toList hp)
  have hW : 0 < netIn c c.unitNet := by omega
  set o := (portCols (netOf c c.unitNet).inGroups)[p]!.1 + t with ho
  have hon : o < netIn c c.unitNet := by omega
  have hdiv : (u * netIn c c.unitNet + o) / netIn c c.unitNet = u := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hW, Nat.div_eq_of_lt hon, Nat.zero_add]
  have hmod : (u * netIn c c.unitNet + o) % netIn c c.unitNet = o := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hon]
  have hfeed : feed c c.unitNet u o = none := by
    unfold feed
    split
    · unfold wireAt
      rw [ho, portOf_eq hF.nodup hp ht, hnw]
      rfl
    · rfl
  apply inCols_sub_order
  refine List.mem_map.2 ⟨u * netIn c c.unitNet + o, List.mem_filter.2 ⟨List.mem_range.2 ?_, ?_⟩, ?_⟩
  · have := Nat.mul_le_mul_right (netIn c c.unitNet) (show u + 1 ≤ c.upv by omega)
    rw [Nat.succ_mul] at this
    omega
  · rw [hdiv, hmod, hfeed]; rfl
  · unfold inCode
    rw [hdiv, hmod]

/-- The wire into a port `HmRow.check` wired is its only one. -/
theorem filter_wire (C : CheckFacts c) {w : Wire} (hw : w ∈ c.wires.toList) (hsha : w.dst.1 ≠ c.shaNet)
    (hhm : w.dst.1 ≠ c.hmNet) :
    (c.wires.toList.filter fun w' => decide (w'.dst = (w.dst.1, w.dst.2.1, w.dst.2.2))) = [w] := by
  obtain ⟨hd1, hd2, hd3⟩ := C.wires w hw
  have hle : (if w.dst.1 = c.unitNet then (c.unit.inGroups.getD 0 default).bits.size else 0) ≤ w.dst.2.2 := by
    split
    · exact C.wire_leaf w hw (by assumption)
    · exact Nat.zero_le _
  have hone := C.wired w.dst.1 hd1 hsha hhm w.dst.2.1 hd2 w.dst.2.2 hle hd3
  have hmem : w ∈ c.wires.toList.filter fun w' => decide (w'.dst = (w.dst.1, w.dst.2.1, w.dst.2.2)) :=
    List.mem_filter.2 ⟨hw, by simp⟩
  have hlen : (c.wires.toList.filter fun w' => decide (w'.dst = (w.dst.1, w.dst.2.1, w.dst.2.2))).length = 1 := by
    have := List.length_pos_of_mem hmem
    omega
  obtain ⟨a, ha⟩ := List.length_eq_one_iff.1 hlen
  rw [ha] at hmem ⊢
  rw [List.mem_singleton.1 hmem]

variable (hF : ∀ n < c.nets.size, 0 < c.perVu.getD n 0 → NetFit c n) (C : CheckFacts c)
  (hpv : c.perVu.getD c.unitNet 0 = c.upv)
include hF C hpv

theorem avOk_init : AvOk c (fAv0 c) := by
  unfold fAv0
  rw [← Array.foldl_toList]
  refine foldl_mem_inv (AvOk c) _ _ _ ?_ fun av hav e he => ?_
  · rw [← Array.foldl_toList, Array.toList_range]
    refine foldl_mem_inv (AvOk c) _ _ _ ⟨?_, fun n i p _ hg => ?_⟩ fun av hav u hu => ?_
    · rw [Array.size_replicate, (fBase_spec c).2 _ le_rfl]
    · simp only [Array.getD_eq_getD_getElem?, Array.getElem?_replicate] at hg
      split at hg <;> exact absurd hg Bool.false_ne_true
    · rw [← Array.foldl_toList, Array.toList_range]
      refine foldl_mem_inv (AvOk c) _ _ _ hav fun av hav p hp => ?_
      split
      · rename_i hin
        obtain ⟨hun, -, -, hiu, hpu⟩ := fIn_spec c hin
        refine avOk_set c hav hin (in_filled (hF _ hun (by omega)) (List.mem_range.1 hu) hpu ?_)
        refine List.filter_eq_nil_iff.2 fun w hw hd => ?_
        have hd' : w.dst = (c.unitNet, u, p) := of_decide_eq_true hd
        have := C.wire_leaf w hw (by rw [hd'])
        rw [hd'] at this
        exact absurd (List.mem_range.1 hp) (Nat.not_lt.2 this)
      · exact hav
  · split
    · rename_i hin
      obtain ⟨hun, hsha, hhm, hiu, hpu⟩ := fIn_spec c hin
      refine avOk_set c hav hin (in_filled (hF _ hun (by omega)) (C.cuts e he).1 hpu ?_)
      have hone := C.wired c.unitNet hun hsha hhm e.1 hiu e.2.1 (by rw [ite_eq_left rfl]; exact C.cut_leaf e he) hpu
      have hc : 0 < (c.leafCuts.toList.filter fun e' => (c.unitNet, e'.1, e'.2.1) = (c.unitNet, e.1, e.2.1)).length :=
        List.length_pos_of_mem (List.mem_filter.2 ⟨he, by simp⟩)
      exact List.length_eq_zero_iff.1 (by omega)
    · exact hav

omit hF C hpv in
/-- The reads `wiresAcyclic` checked, of every net with slots. -/
abbrev ReadsOk : Prop := ∀ n < c.nets.size, fSlots c n ≠ 0 → (ProvedScope.outReads (netOf c n)).2 = true

omit hF C hpv in
theorem readsOk_of (h : ((Array.range c.nets.size).all fun n => fSlots c n == 0 || ((fReads c).getD n (#[], false)).2) =
    true) : ReadsOk (c := c) := by
  intro n hn hs
  rw [Array.all_eq_true'] at h
  have := h n (Array.mem_range.2 hn)
  simp only [Bool.or_eq_true, beq_iff_eq, hs, false_or] at this
  unfold fReads at this
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_map, Array.getElem?_range, ite_eq_left hn] at this
  simp only [Option.map_some, Option.getD_some, beq_iff_eq, hs, ↓reduceIte] at this
  exact this

omit C hpv in
/-- **A ready wire's source port's bits are in the order**: computed from the filled ports its source port reads. -/
theorem src_mem (hR : ReadsOk (c := c)) {av : Array Bool} (hav : AvOk c av) {w : Wire} (hr : fReady c av w = true) :
    ∀ t < ((portCols (netOf c w.src.1).outGroups).getD w.src.2.2 default).2,
      code w.src.1 w.src.2.1 (((portCols (netOf c w.src.1).outGroups).getD w.src.2.2 default).1 + t) ∈ order c := by
  unfold fReady at hr
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hr
  obtain ⟨hsl, hm⟩ := hr
  obtain ⟨hsha, hhm, hse⟩ := fSlots_pos c (n := w.src.1) (by omega)
  rw [hse] at hsl
  have hsn : w.src.1 < c.nets.size := by
    by_contra h
    have hnone : (fReads c)[w.src.1]? = none := Array.getElem?_eq_none (by unfold fReads; simp; omega)
    rw [Array.getD_eq_getD_getElem?, hnone] at hm
    simp at hm
  have hrd : (fReads c).getD w.src.1 (#[], false) = ProvedScope.outReads (netOf c w.src.1) := by
    unfold fReads
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_map, Array.getElem?_range, ite_eq_left hsn]
    simp only [Option.map_some, Option.getD_some, beq_iff_eq]
    rw [ite_eq_right (by omega)]
  rw [hrd] at hm
  have hF' := hF _ hsn (by omega)
  have hfill : ∀ p < (portCols (netOf c w.src.1).inGroups).size, av.getD (fNode c w.src.1 w.src.2.1 p) false = true →
      Filled c w.src.1 w.src.2.1 p := fun p hp hg => hav.2 _ _ _ (fIn_of c hsn hsha hhm (by omega) hp) hg
  rcases hF'.kind with hb | ⟨nm, tb, nb, lo, bits, hb⟩
  · have hlk : (netOf c w.src.1).lookup = none := by
      obtain ⟨_, _, _, _, _, _, _, ho⟩ := hb
      exact (Net.ofRows_ok ho).2.2.2.2.2.2.2.2.2.2.2.2.1
    have hok := hR _ hsn (by omega)
    rw [outReads_eq _ (by rw [hlk]; rfl)] at hm hok
    simp only [Bool.and_eq_true, Array.all_eq_true', beq_iff_eq] at hok
    revert hm
    rcases hq : (portCols (netOf c w.src.1).outGroups)[w.src.2.2]? with _ | q
    · simp [mMasks, hq]
    · intro hm
      simp only [mMasks, Array.getElem?_map, hq, Option.map_some, Array.all_eq_true', Array.mem_filter,
        Array.mem_range, and_imp] at hm
      have hqd : (portCols (netOf c w.src.1).outGroups).getD w.src.2.2 default = q := by
        rw [Array.getD_eq_getD_getElem?, hq]; rfl
      rw [hqd]
      set m := (Array.range q.2).foldl (fun m x => mUnion m (mCol (netOf c w.src.1) (mRows (netOf c w.src.1)) (q.1 + x))) 0
      have hmo : Ok c w.src.1 w.src.2.1 m := by
        refine ⟨?_, fun p hp hb => hfill p hp (hm p hp hb)⟩
        have : m % 2 = 0 := hok.2 m (Array.mem_map.2 ⟨q, Array.mem_of_getElem? hq, rfl⟩)
        rw [Nat.testBit_zero]; simp [this]
      intro t ht
      exact ok_mem_order hsn hsha hhm hsl hb hF'.inConst hF'.constW _ (hmo.mono (mask_sub _ q ht))
  · have hlk : (netOf c w.src.1).lookup.isSome = true := by
      obtain ⟨-, -, -, -, -, _, _, _, -, -, -, -, -, -, -, -, -, -, hlk, -⟩ := build_spec hb
      rw [hlk]; rfl
    obtain ⟨hpc, hiw, -, -, o0, hpo, h128, hob⟩ := lookup_facts hb hF'.log
    have hro : ProvedScope.outReads (netOf c w.src.1) = ((portCols (netOf c w.src.1).outGroups).map fun _ =>
        Array.range (portCols (netOf c w.src.1).inGroups).size, true) := by
      unfold ProvedScope.outReads
      rw [ite_eq_left hlk]
    rw [hro, hpo, hpc] at hm
    have hsp : w.src.2.2 = 0 ∧ av.getD (fNode c w.src.1 w.src.2.1 0) false = true := by
      rcases hx : w.src.2.2 with _ | k
      · rw [hx] at hm
        refine ⟨by omega, ?_⟩
        simpa [Array.all_iff_forall, Array.getD_eq_getD_getElem?] using hm
      · rw [hx] at hm
        simp at hm
    have hf0 : Filled c w.src.1 w.src.2.1 0 := hfill 0 (by rw [hpc]; exact Nat.one_pos) hsp.2
    have hcomp := lookup_mem_order hsn hsha hhm hsl hb hF'.log hF'.constW hf0
    rw [hsp.1, hpo]
    show ∀ t < bits, code w.src.1 w.src.2.1 (o0 + t) ∈ order c
    intro t ht
    exact hcomp _ (by show _ * _ ≤ _; rw [hiw]; omega) (by omega)

omit hpv in
/-- **A wire's port is filled once its source port's bits are in the order**: each bit of the port copies its source
port's bit at the same place. -/
theorem fill_filled {w : Wire} (hw : w ∈ c.wires.toList) (hin : fIn c w.dst.1 w.dst.2.1 w.dst.2.2 = true)
    (hs : ∀ t < ((portCols (netOf c w.src.1).outGroups).getD w.src.2.2 default).2,
      code w.src.1 w.src.2.1 (((portCols (netOf c w.src.1).outGroups).getD w.src.2.2 default).1 + t) ∈ order c) :
    Filled c w.dst.1 w.dst.2.1 w.dst.2.2 := by
  obtain ⟨hn, hsha, hhm, hi, hp⟩ := fIn_spec c hin
  have hF' := hF _ hn (by omega)
  obtain ⟨-, -, hsp, hwid⟩ := C.wire_src w hw
  intro t ht
  rw [FlockSoundness.Layout.getD_portCols hp] at ht ⊢
  obtain ⟨o, ho⟩ : ∃ o, o = (portCols (netOf c w.dst.1).inGroups)[w.dst.2.2]!.1 + t := ⟨_, rfl⟩
  rw [← ho]
  have hob : o ∈ portBits (netOf c w.dst.1).inGroups := by rw [ho]; exact port_mem hp ht
  have hq := hF'.ports (portCols (netOf c w.dst.1).inGroups)[w.dst.2.2]!
    (by rw [getElem!_pos (portCols _) _ hp]; exact Array.getElem_mem_toList hp)
  have hon : o < netIn c w.dst.1 := by omega
  have hsrc : ∀ t' < ((portCols (netOf c w.dst.1).inGroups)[w.dst.2.2]!).2,
      code w.src.1 w.src.2.1 (((portCols (netOf c w.src.1).outGroups).getD w.src.2.2 default).1 + t') ∈ order c :=
    fun t' ht' => hs t' (by rw [FlockSoundness.Layout.getD_portCols hsp, ← hwid]; exact ht')
  have hfeed : feed c w.dst.1 w.dst.2.1 o = some (code w.src.1 w.src.2.1
      (((portCols (netOf c w.src.1).outGroups).getD w.src.2.2 default).1 + t)) := by
    unfold feed wireAt
    rw [ite_eq_left hob, ho, portOf_eq hF'.nodup hp ht, filter_wire C hw hsha hhm]
    simp
  have hcw := hF'.constW
  have hic := hF'.inConst
  have hrow := rowAt_feed_eq hn hsha hhm hi hon hic (by omega) hfeed
  rw [reads_addSelf fun j => (port_rows hF' hob j).1, reads_addSelf fun j => (port_rows hF' hob j).2] at hrow
  refine mem_order_of_ready (code_mem_cols hn hi (by omega)) (ready_some
    (by rw [row_code]; exact hrow) fun y hy => ?_)
  simp only [List.nil_append, List.mem_append, List.mem_singleton, or_self] at hy
  rw [hy]
  exact hsrc t ht

omit hpv in
/-- **A pass of the fill keeps the invariant.** -/
theorem avOk_pass (hR : ReadsOk (c := c)) {av : Array Bool} (hav : AvOk c av) :
    AvOk c (ProvedScope.fillPass (fReady c) (fDst c) c.wires av).1 := by
  unfold ProvedScope.fillPass
  rw [← Array.foldl_toList]
  refine foldl_mem_inv (fun b : Array Bool × Bool => AvOk c b.1) _ _ _ hav fun b hb w hw => ?_
  obtain ⟨av', ch⟩ := b
  dsimp only at hb ⊢
  split
  · rename_i d hd
    unfold fDst at hd
    split at hd
    · rename_i hin
      cases hd
      split
      · rename_i hc
        simp only [Bool.and_eq_true] at hc
        exact avOk_set c hb hin (fill_filled hF C hw hin (src_mem hF hR hb hc.2))
      · exact hb
    · cases hd
  · exact hb

omit hpv in
/-- **The fill keeps the invariant**, through every pass. -/
theorem avOk_loop (hR : ReadsOk (c := c)) (fuel : ℕ) :
    ∀ {av : Array Bool}, AvOk c av →
      AvOk c (ProvedScope.fillLoop (ProvedScope.fillPass (fReady c) (fDst c) c.wires) fuel av) := by
  induction fuel with
  | zero => intro av hav; exact hav
  | succ f ih =>
    intro av hav
    have hp := avOk_pass hF C hR hav
    rw [ProvedScope.fillLoop]
    split
    · rename_i av' h
      rw [h] at hp
      exact ih hp
    · rename_i av' h
      rw [h] at hp
      exact hp

/-- **Every input port of every slot outside the rows' circuits is filled** when `wiresAcyclic` passed, and every net
with slots reads in order. -/
theorem all_filled (hw : ProvedScope.wiresAcyclic c = true) :
    ReadsOk (c := c) ∧ ∀ n i p, fIn c n i p = true → Filled c n i p := by
  rw [wiresAcyclic_eq, Bool.and_eq_true] at hw
  have hR := readsOk_of hw.2
  refine ⟨hR, fun n i p hin => ?_⟩
  have hav := avOk_loop hF C hR (c.wires.size + 1) (avOk_init hF C hpv)
  refine hav.2 n i p hin ?_
  have hall := hw.1
  rw [Array.all_eq_true'] at hall
  have hlt := fNode_size c hin
  rw [← hav.1] at hlt
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hlt, Option.getD_some]
  exact hall _ (Array.getElem_mem hlt)

/-- **Order completeness**: when `wiresAcyclic` passed, every column of the template instance with a row is in the
logical order. -/
theorem order_complete_of (hw : ProvedScope.wiresAcyclic c = true) :
    ∀ n i o, rowAt c n i o ≠ none → code n i o ∈ order c := by
  obtain ⟨hR, hfill⟩ := all_filled hF C hpv hw
  intro n i o hr
  obtain ⟨hn, hsha, hhm, hi, hwo⟩ := rowAt_dom hr
  have hF' := hF n hn (by omega)
  have hports : ∀ p < (portCols (netOf c n).inGroups).size, Filled c n i p :=
    fun p hp => hfill n i p (fIn_of c hn hsha hhm hi hp)
  by_cases hc : o = (netOf c n).constPos
  · rw [hc]; exact const_mem_order hn hsha hhm hi hF'.constW
  by_cases hm : netIn c n ≤ o ∧ o < (netOf c n).constPos
  · rcases hF'.kind with hb | ⟨nm, tb, nb, lo, bits, hb⟩
    · have hlk : (netOf c n).lookup = none := by
        obtain ⟨_, _, _, _, _, _, _, ho⟩ := hb
        exact (Net.ofRows_ok ho).2.2.2.2.2.2.2.2.2.2.2.2.1
      have hok := hR n hn (by rw [fSlots_of c hsha hhm]; omega)
      rw [outReads_eq _ (by rw [hlk]; rfl)] at hok
      simp only [Bool.and_eq_true, Array.all_eq_true', beq_iff_eq] at hok
      obtain ⟨hsz, -⟩ := mRows_J (netOf c n)
      have hN : mN (netOf c n) = netIn c n := min_eq_left hF'.inConst
      have hk : o - mN (netOf c n) < (mRows (netOf c n)).size := by rw [hsz, hN]; omega
      refine ok_mem_order hn hsha hhm hi hb hF'.inConst hF'.constW o ⟨?_, fun p hp _ => hports p hp⟩
      have hcol : mCol (netOf c n) (mRows (netOf c n)) o = (mRows (netOf c n))[o - mN (netOf c n)] := by
        unfold mCol
        rw [ite_eq_right (by rw [hN]; omega), ite_eq_left hm.2, Array.getD_eq_getD_getElem?,
          Array.getElem?_eq_getElem hk, Option.getD_some]
      rw [hcol, Nat.testBit_zero]
      have := hok.1 _ (Array.getElem_mem hk)
      simp [this]
    · exact lookup_mem_order hn hsha hhm hi hb hF'.log hF'.constW
        (hports 0 (by rw [(lookup_facts hb hF'.log).1]; exact Nat.one_pos)) o hm.1 hm.2
  · rcases hrr : rowAt c n i o with _ | _ | p
    · exact absurd hrr hr
    · exact absurd (rowAt_const hrr) hc
    · rcases (rowAt_comp hrr).2 with h | ⟨-, s, hf, -⟩
      · exact absurd ⟨h.1, h.2.1⟩ hm
      · obtain ⟨p, t, hp, ht, rfl⟩ := Aliased.mem_portBits (feed_some hf).1
        have := hports p hp t (by rw [FlockSoundness.Layout.getD_portCols hp]; exact ht)
        rwa [FlockSoundness.Layout.getD_portCols hp] at this

end fill

/-- **Order completeness** of an accepted circuit whose wires `wiresAcyclic` passed: every column of the template
instance with a row is in the logical order. -/
theorem order_complete {tags : Tags} {st : Stmt} (L : FlockSoundness.Layout st) (P : ParseFacts tags st.c)
    (C : CheckFacts st.c) (hg : 0 < st.c.g) (hw : ProvedScope.wiresAcyclic st.c = true) :
    ∀ n i o, rowAt st.c n i o ≠ none → code n i o ∈ order st.c :=
  order_complete_of (fun n hn hi => ⟨netLog_le L C hn hg hi, (net_const L P C hn hg hi).1, (L.fit n).1,
      (nets_ok P n).1.2.2, portBits_nodup_net P hn, P.nets _ (getD_mem_toList hn)⟩) C
    (perVu_unit C L.unit_mem L.unit_kind L.unit_count hg) hw

end FlockSoundness.Discharge.Inst
