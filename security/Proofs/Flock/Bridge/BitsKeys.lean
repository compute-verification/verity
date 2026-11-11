module

public import Properties.Flock.Circuit.Keys
public import Spot.Bits

/-!
# Bits' source keys are the bridge's (`Spot.Bits.srcKey`, `keyIdx`)

`Spot.Bits.srcAt` and `Aliased.srcAt` name the same column Δ copies into a unit input. Bits looks a port's
leaf cut and wire up in the circuit's `Index` and a leaf word's row port by binary search (`lastLe`); the bridge
takes the first leaf cut and wire of a list and the last row port by `find?`. `srcOf` maps Bits' sources to the
bridge's, constructor for constructor:

* `lastLe_eq`: on the row ports' first words, which ascend (`starts_mono`), the binary search is the bridge's search.
* `cuts_eq`, `wires_eq`: the `Index`'s leaf cut and wire of a port are its first ones.
* `srcAt_eq`, `srcKey_eq`: the bridge's source is `srcOf` of Bits'.
* `keyIdx_eq`: Bits' key of an input is the bridge's.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Flock.Bridge

open FlockSoundness FlockSoundness.Discharge

/-- **Bits' source as the bridge's**, constructor for constructor. -/
def srcOf : Spot.Bits.Src → Aliased.Src
  | .zero => .zero
  | .msg p byte bit => .msg p byte bit
  | .wire net slot col => .wire net slot col
  | .lit col => .lit col
  | .pad k off => .pad k off

theorem srcOf_inj {s s' : Spot.Bits.Src} : srcOf s' = srcOf s ↔ s' = s := by
  refine ⟨fun h => ?_, fun h => h ▸ rfl⟩
  cases s <;> cases s' <;> simp_all [srcOf]

theorem srcOf_eq_zero {s : Spot.Bits.Src} : srcOf s = .zero ↔ s = .zero := by
  cases s <;> simp [srcOf]

private theorem srcOf_ite (q : Prop) [Decidable q] (a b : Spot.Bits.Src) :
    srcOf (if q then a else b) = if q then srcOf a else srcOf b := by
  by_cases h : q <;> simp [h]

private theorem getElem!_eq_getD {α : Type} [Inhabited α] (a : Array α) (i : ℕ) : a[i]! = a.getD i default := by
  simp only [Array.getD_eq_getD_getElem?, getElem!_def]
  cases a[i]? <;> rfl

private theorem ite_decide {α : Type} (P : Prop) [Decidable P] (a b : α) :
    (if decide P = true then a else b) = if P then a else b := by
  by_cases h : P <;> simp [h]

private theorem getElem!_int (a : Array Int) (i : ℕ) : a[i]! = a.getD i 0 := getElem!_eq_getD a i

/-! ## The row port of a leaf word -/

private def lastLeBody (s : Array ℕ) (x : ℕ) : Unit → ℕ × ℕ → Id (ForInStep (ℕ × ℕ)) :=
  fun _ st => if st.1 < st.2 then
      if s.getD ((st.1 + st.2) / 2) 0 ≤ x then pure (.yield ((st.1 + st.2) / 2 + 1, st.2))
      else pure (.yield (st.1, (st.1 + st.2) / 2))
    else pure (.done (st.1, st.2))

private theorem lastLe_loop (s : Array ℕ) (x : ℕ) :
    Spot.Bits.lastLe s x = (Lean.Loop.forIn {} (0, s.size) (lastLeBody s x)).run.1 - 1 := rfl

/-- The binary search keeps every index below `lo` at most `x` and every one from `hi` above it, and ends with them
equal. -/
private theorem lastLe_spec (s : Array ℕ) (x : ℕ)
    (hs : ∀ i j, i ≤ j → j < s.size → s.getD i 0 ≤ s.getD j 0) :
    ∀ n lo hi, hi - lo ≤ n → lo ≤ hi → hi ≤ s.size → (∀ i < lo, s.getD i 0 ≤ x) →
      (∀ i, hi ≤ i → i < s.size → x < s.getD i 0) →
      ∃ r, (Lean.Loop.forIn {} (lo, hi) (lastLeBody s x)).run = (r, r) ∧ r ≤ s.size ∧
        (∀ i < r, s.getD i 0 ≤ x) ∧ (∀ i, r ≤ i → i < s.size → x < s.getD i 0)
  | n, lo, hi, hn, hlh, hhs, hlo, hhi => by
    rw [Lean.Loop.forIn_eq_of_monadTail]
    by_cases hlt : lo < hi
    · have hm1 : lo ≤ (lo + hi) / 2 := by omega
      have hm2 : (lo + hi) / 2 < hi := by omega
      by_cases hx : s.getD ((lo + hi) / 2) 0 ≤ x
      · have hb : lastLeBody s x () (lo, hi) = pure (.yield ((lo + hi) / 2 + 1, hi)) := by
          simp only [lastLeBody, hlt, hx, ite_true]
        rw [hb]
        match n with
        | 0 => omega
        | n + 1 =>
          exact lastLe_spec s x hs n _ _ (by omega) (by omega) hhs
            (fun i hi' => Nat.le_trans (hs i _ (by omega) (by omega)) hx) hhi
      · have hb : lastLeBody s x () (lo, hi) = pure (.yield (lo, (lo + hi) / 2)) := by
          simp only [lastLeBody, hlt, hx, ite_true, ite_false]
        rw [hb]
        match n with
        | 0 => omega
        | n + 1 =>
          exact lastLe_spec s x hs n _ _ (by omega) hm1 (by omega) hlo
            (fun i h1 h2 => Nat.lt_of_lt_of_le (Nat.lt_of_not_le hx) (hs _ i h1 h2))
    · have hb : lastLeBody s x () (lo, hi) = pure (.done (lo, hi)) := by
        simp only [lastLeBody, hlt, ite_false]
      rw [hb]
      exact ⟨lo, by rw [show hi = lo by omega]; rfl, by omega, hlo, fun i h1 h2 => hhi i (by omega) h2⟩

private theorem find?_range_reverse (p : ℕ → Bool) {r : ℕ} :
    ∀ n, r ≤ n → (∀ i < r, p i) → (∀ i, r ≤ i → i < n → p i = false) →
      ((List.range n).reverse.find? p).getD 0 = r - 1
  | 0, hr, _, _ => by simp at hr; simp [hr]
  | n + 1, hr, h1, h2 => by
    rw [List.range_succ, List.reverse_append, List.reverse_singleton, List.singleton_append, List.find?_cons]
    by_cases hn : r = n + 1
    · subst hn; simp [h1 n (by omega)]
    · rw [h2 n (by omega) (by omega)]
      exact find?_range_reverse p n (by omega) h1 fun i hi hi' => h2 i hi (by omega)

/-- **On an ascending array, `lastLe` is the last index whose entry is at most `x`**, or 0. -/
theorem lastLe_eq (s : Array ℕ) (x : ℕ) (hs : ∀ i j, i ≤ j → j < s.size → s.getD i 0 ≤ s.getD j 0) :
    Spot.Bits.lastLe s x = ((List.range s.size).reverse.find? fun q => s.getD q 0 ≤ x).getD 0 := by
  obtain ⟨r, he, hr, h1, h2⟩ := lastLe_spec s x hs _ 0 s.size (Nat.le_refl _) (Nat.zero_le _) (Nat.le_refl _)
    (fun _ h => absurd h (Nat.not_lt_zero _)) (fun _ h1 h2 => absurd h2 (by omega))
  rw [lastLe_loop, he]
  exact (find?_range_reverse _ s.size hr (fun i hi => decide_eq_true (h1 i hi))
    fun i hi hi' => decide_eq_false (Nat.not_le_of_lt (h2 i hi hi'))).symm

private theorem starts_inv {α : Type} (w : α → ℕ) : ∀ (l : List α) (a : Array ℕ) (t : ℕ),
    (∀ i j, i ≤ j → j < a.size → a.getD i 0 ≤ a.getD j 0) → (∀ i < a.size, a.getD i 0 ≤ t) →
    let s := (l.foldl (fun (acc : Array ℕ × ℕ) p => (acc.1.push acc.2, acc.2 + w p)) (a, t)).1
    ∀ i j, i ≤ j → j < s.size → s.getD i 0 ≤ s.getD j 0
  | [], _, _, h1, _ => h1
  | p :: l, a, t, h1, h2 => by
    refine starts_inv w l (a.push t) (t + w p) (fun i j hij hj => ?_) (fun i hi => ?_)
    · simp only [Array.getD_eq_getD_getElem?, Array.getElem?_push, Array.size_push] at hj ⊢
      by_cases hja : j = a.size
      · subst hja
        by_cases hia : i = a.size
        · simp [hia]
        · have := h2 i (by omega)
          simp only [Array.getD_eq_getD_getElem?] at this
          simp [hia, this]
      · have hia : i ≠ a.size := by omega
        have := h1 i j hij (by omega)
        simp only [Array.getD_eq_getD_getElem?] at this
        simp [hia, hja, this]
    · simp only [Array.getD_eq_getD_getElem?, Array.getElem?_push, Array.size_push] at hi ⊢
      by_cases hia : i = a.size
      · simp [hia]
      · have := h2 i (by omega)
        simp only [Array.getD_eq_getD_getElem?] at this
        simp [hia]; omega

/-- The row ports' first words ascend. -/
theorem starts_mono (c : Flock.Circuit) :
    ∀ i j, i ≤ j → j < (Typed.starts c).size → (Typed.starts c).getD i 0 ≤ (Typed.starts c).getD j 0 := by
  unfold Typed.starts
  rw [← Array.foldl_toList]
  exact starts_inv Port.words c.ports.toList #[] 0 (fun _ _ _ h => absurd h (Nat.not_lt_zero _))
    (fun _ h => absurd h (Nat.not_lt_zero _))

/-! ## A port's leaf cut and wire -/

/-- A fold that inserts each entry's key unless present keeps each key's first entry. -/
private theorem foldl_first {β K V : Type} [BEq K] [Hashable K] [LawfulBEq K] [LawfulHashable K]
    (P : β → Bool) (key : β → K) (val : β → V) (f : Std.HashMap K V → β → Std.HashMap K V)
    (hf : ∀ m b, f m b = if P b && !m.contains (key b) then m.insert (key b) (val b) else m) :
    ∀ (l : List β) (m : Std.HashMap K V) (a : K),
      (l.foldl f m)[a]? = (m[a]?).or ((l.find? fun b => P b && key b == a).map val)
  | [], m, a => by simp
  | b :: l, m, a => by
    rw [List.foldl_cons, foldl_first P key val f hf l, List.find?_cons, hf]
    have hca : m.contains (key b) = (m[key b]?).isSome := Std.HashMap.contains_eq_isSome_getElem?
    by_cases hk : key b = a
    · subst hk
      cases hm : m[key b]? with
      | none =>
        have hc : m.contains (key b) = false := by rw [hca, hm]; rfl
        cases P b
        · simp [hc, hm]
        · simp [hc]
      | some v =>
        have hc : m.contains (key b) = true := by rw [hca, hm]; rfl
        cases P b <;> simp [hc, hm]
    · have hb : (key b == a) = false := beq_eq_false_iff_ne.mpr hk
      by_cases hc : P b && !m.contains (key b) <;> simp [hc, hb, Std.HashMap.getElem?_insert]

private theorem find?_congr {α : Type} {p q : α → Bool} : ∀ {l : List α}, (∀ a ∈ l, p a = q a) →
    l.find? p = l.find? q
  | [], _ => rfl
  | a :: l, h => by
    rw [List.find?_cons, List.find?_cons, h a (by simp), find?_congr (l := l) fun b hb => h b (by simp [hb])]

/-- **The `Index`'s leaf cut of copy `k`'s port `p`** is its first. -/
theorem cuts_eq (c : Flock.Circuit) (k p : ℕ) :
    (Spot.Bits.Index.of c).cuts[(k, p)]? =
      (c.leafCuts.toList.find? fun e => decide (e.1 = k ∧ e.2.1 = p)).map fun e => e.2.2 := by
  show (c.leafCuts.foldl _ {})[(k, p)]? = _
  rw [← Array.foldl_toList, foldl_first (fun _ => true) (fun e : ℕ × ℕ × ℕ => (e.1, e.2.1)) (fun e => e.2.2) _
    (fun m e => by obtain ⟨k, p, w⟩ := e; by_cases h : m.contains (k, p) <;> simp [h])]
  rw [Std.HashMap.getElem?_empty, Option.none_or]
  refine congrArg _ (find?_congr fun e _ => Bool.eq_iff_iff.mpr ?_)
  simp [Prod.ext_iff]

/-- **The `Index`'s wire into copy `k`'s port `p`** is the first. -/
theorem wires_eq (c : Flock.Circuit) (k p : ℕ) :
    (Spot.Bits.Index.of c).wires[(k, p)]? =
      (c.wires.toList.find? fun w => decide (w.dst = (c.unitNet, k, p))).map Flock.Wire.src := by
  show (c.wires.foldl _ {})[(k, p)]? = _
  rw [← Array.foldl_toList, foldl_first (fun w : Flock.Wire => w.dst.1 == c.unitNet)
    (fun w => (w.dst.2.1, w.dst.2.2)) Flock.Wire.src _ (fun _ _ => rfl)]
  rw [Std.HashMap.getElem?_empty, Option.none_or]
  refine congrArg _ (find?_congr fun w _ => Bool.eq_iff_iff.mpr ?_)
  obtain ⟨⟨d1, d2, d3⟩, _⟩ := w
  simp [Prod.ext_iff]

/-! ## Sources and keys -/

/-- The bridge's row port of leaf word `x`: the last whose first word is at most `x`. -/
private def portAt (c : Flock.Circuit) (x : ℕ) : ℕ :=
  ((List.range (Typed.starts c).size).reverse.find? fun q => (Typed.starts c)[q]! ≤ x).getD 0

private theorem msgAt_eq (c : Flock.Circuit) (x t : ℕ) (wide : Bool) :
    srcOf (Spot.Bits.msgAt c (Spot.Bits.Index.of c) x t wide) =
      if (wide || decide (t < 16)) && decide (16 * (x - (Typed.starts c)[portAt c x]!) + t <
          c.ports[portAt c x]!.bits) then
        .msg (portAt c x) (2 * (x - (Typed.starts c)[portAt c x]!) + t / 8) (t % 8)
      else .zero := by
  unfold Spot.Bits.msgAt portAt
  rw [show (Spot.Bits.Index.of c).starts = Typed.starts c from rfl, lastLe_eq _ _ (starts_mono c)]
  simp only [getElem!_eq_getD, srcOf_ite]
  rfl

/-- **The bridge's source of copy `k`'s port `p`, bit `t`, is Bits'**, under `srcOf`. -/
theorem srcAt_eq (c : Flock.Circuit) (k p t : ℕ) :
    Aliased.srcAt c k p t = srcOf (Spot.Bits.srcAt c (Spot.Bits.Index.of c) k p t) := by
  unfold Aliased.srcAt Spot.Bits.srcAt
  by_cases h1 : p < (c.unit.inGroups.getD 0 default).bits.size
  · simp only [h1, ite_true, getElem!_int, srcOf_ite]
    by_cases hl : (c.leavesIn.getD k #[]).getD p 0 < 0
    · simp only [hl, ite_true, srcOf]
    · simp only [hl, ite_false]
      rw [msgAt_eq]
      simp only [portAt, Bool.true_or, Bool.true_and]
      exact (ite_decide _ _ _).symm
  · simp only [h1, ite_false]
    rw [cuts_eq]
    cases hf : c.leafCuts.toList.filter (fun e => decide (e.1 = k ∧ e.2.1 = p)) with
    | cons e _ =>
      have he : c.leafCuts.toList.find? (fun e => decide (e.1 = k ∧ e.2.1 = p)) = some e := by
        rw [← List.head?_filter, hf]; rfl
      rw [he]
      simp only [Option.map_some]
      rw [msgAt_eq]
      simp only [portAt, Bool.false_or]
      rfl
    | nil =>
      have he : c.leafCuts.toList.find? (fun e => decide (e.1 = k ∧ e.2.1 = p)) = none := by
        rw [← List.head?_filter, hf]; rfl
      rw [he, wires_eq]
      simp only [Option.map_none]
      cases hw : c.wires.toList.filter (fun w => decide (w.dst = (c.unitNet, k, p))) with
      | cons w _ =>
        have hw' : c.wires.toList.find? (fun w => decide (w.dst = (c.unitNet, k, p))) = some w := by
          rw [← List.head?_filter, hw]; rfl
        rw [hw']
        obtain ⟨_, net, slot, j⟩ := w
        rfl
      | nil =>
        have hw' : c.wires.toList.find? (fun w => decide (w.dst = (c.unitNet, k, p))) = none := by
          rw [← List.head?_filter, hw]; rfl
        rw [hw']
        rfl

private theorem getElem!_pair (a : Array (ℕ × ℕ)) (i : ℕ) : a[i]! = a.getD i (0, 0) := getElem!_eq_getD a i

/-- **The bridge's key of stacked input `x` is Bits'**, under `srcOf`. -/
theorem srcKey_eq (c : Flock.Circuit) (x : ℕ) :
    Aliased.srcKey c x = srcOf (Spot.Bits.srcKey c (Spot.Bits.Index.of c) x) := by
  unfold Aliased.srcKey Spot.Bits.srcKey FlockSoundness.portOf FlockSoundness.portBits
  simp only [getElem!_pair]
  generalize Flock.portCols c.unit.inGroups = pc
  generalize x % (c.unit.inWords * Flock.WORD) = off
  cases hf : (List.range pc.size).find? fun p =>
      decide ((pc.getD p (0, 0)).1 ≤ off ∧ off < (pc.getD p (0, 0)).1 + (pc.getD p (0, 0)).2) with
  | none =>
    have hn : off ∉ pc.toList.flatMap fun q => List.range' q.1 q.2 := fun hm => by
      obtain ⟨q, hq, hq'⟩ := List.mem_flatMap.mp hm
      obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.mp hq
      have h := List.find?_eq_none.mp hf i (List.mem_range.mpr (by simpa using hi))
      simp only [Array.getD_eq_getD_getElem?] at h hq'
      simp only [Array.length_toList] at hi
      simp [List.mem_range'_1, hi] at h hq'
      omega
    simp only [hn, ite_false]
    rfl
  | some p =>
    have hp := List.find?_some hf
    have hpm := List.mem_range.mp (List.mem_of_find?_eq_some hf)
    have hm : off ∈ pc.toList.flatMap fun q => List.range' q.1 q.2 := by
      refine List.mem_flatMap.mpr ⟨pc[p], List.getElem_mem (by simpa using hpm), ?_⟩
      simp only [Array.getD_eq_getD_getElem?, decide_eq_true_eq] at hp
      simp [List.mem_range'_1, hpm] at hp ⊢
      omega
    simp only [hm, ite_true, Option.getD_some]
    exact srcAt_eq _ _ _ _

private theorem finRange_map_val (n : ℕ) : (List.finRange n).map Fin.val = List.range n :=
  List.ext_getElem (by simp) fun i _ _ => by simp

/-- **Bits' key of input `x` is the bridge's.** -/
theorem keyIdx_eq (c : Flock.Circuit) {B x : ℕ} (hx : x < B) :
    Spot.Bits.keyIdx c B x = (Aliased.keyIdx c B ⟨x, hx⟩).map Fin.val := by
  unfold Spot.Bits.keyIdx Aliased.keyIdx
  have hz : Aliased.srcKey c x = .zero ↔
      Spot.Bits.srcKey c (Spot.Bits.Index.of c) x = .zero := by
    rw [srcKey_eq, srcOf_eq_zero]
  by_cases h0 : Spot.Bits.srcKey c (Spot.Bits.Index.of c) x = .zero
  · rw [ite_eq_left_of_eq_true _ _ (eq_true h0), ite_eq_left_of_eq_true _ _ (eq_true (hz.2 h0))]
    rfl
  · rw [ite_eq_right_of_eq_false _ _ (eq_false h0), ite_eq_right_of_eq_false _ _ (eq_false (mt hz.1 h0)),
      ← finRange_map_val, List.find?_map]
    refine congrArg (Option.map Fin.val) (find?_congr fun i _ => ?_)
    show decide (Spot.Bits.srcKey c (Spot.Bits.Index.of c) i.val =
        Spot.Bits.srcKey c (Spot.Bits.Index.of c) x) =
      decide (Aliased.srcKey c i.val = Aliased.srcKey c x)
    rw [decide_eq_decide, srcKey_eq c i.val, srcKey_eq c x, srcOf_inj]

end Flock.Bridge
