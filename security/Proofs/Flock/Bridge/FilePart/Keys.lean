module

public import Spot.FilePart
public import Properties.Flock.Circuit.Keys
public import Init.Internal.Order.While
public import Std.Data.HashMap.Lemmas

/-!
# The code's source keys are the claims'

`srcKey_toA`: the key `Spot.FilePart.ofCircuit` reads for stacked input `x`,
`Spot.FilePart.srcKey c (Index.of c) x`, is `Aliased.srcKey c x` under the constructor map `Src.toA`.
`lastLe`'s binary search is the last row-word start at most `x` on the ascending starts (`lastLe_spec`, `starts_mono`),
and `Index.of`'s maps are the first leaf cut and the first wire into the unit net (`cuts_get`, `wires_get`).
-/

@[expose] public section

namespace Spot.FilePart

open Std FlockSoundness.Discharge

deriving instance ReflBEq, LawfulBEq for Src

/-- A code key as the claims' key (`Aliased.Src`): the same constructor. -/
def Src.toA : Src → Aliased.Src
  | .zero => .zero
  | .msg p b t => .msg p b t
  | .wire n s c => .wire n s c
  | .lit c => .lit c
  | .pad k o => .pad k o

theorem Src.toA_inj {a b : Src} (h : a.toA = b.toA) : a = b := by
  cases a <;> cases b <;> simp_all [toA]

theorem Src.toA_eq_zero {a : Src} : a.toA = .zero ↔ a = .zero := by
  cases a <;> simp [toA]

/-! ## `lastLe` is the last start at most `x` -/

section lastLe

/-- `lastLe`'s loop body. -/
def lastLeBody (s : Array ℕ) (x : ℕ) : Unit → ℕ × ℕ → Id (ForInStep (ℕ × ℕ)) := fun _ st =>
  if st.1 < st.2 then
    if s.getD ((st.1 + st.2) / 2) 0 ≤ x then pure (.yield ((st.1 + st.2) / 2 + 1, st.2))
    else pure (.yield (st.1, (st.1 + st.2) / 2))
  else pure (.done st)

theorem lastLe_eq (s : Array ℕ) (x : ℕ) :
    lastLe s x = (forIn Lean.Loop.mk (0, s.size) (lastLeBody s x)).1 - 1 := rfl

theorem forIn_loop {β : Type} (b : β) (f : Unit → β → Id (ForInStep β)) :
    forIn Lean.Loop.mk b f = Lean.Loop.forIn Lean.Loop.mk b f := rfl

theorem lastLe_loop (s : Array ℕ) (x : ℕ) (hmono : ∀ i j, i ≤ j → j < s.size → s.getD i 0 ≤ s.getD j 0) :
    ∀ (d lo hi : ℕ), hi - lo ≤ d → lo ≤ hi → hi ≤ s.size → (∀ i < lo, s.getD i 0 ≤ x) →
      (∀ i, hi ≤ i → i < s.size → x < s.getD i 0) →
      ∃ P, (forIn Lean.Loop.mk (lo, hi) (lastLeBody s x)).1 = P ∧ P ≤ s.size ∧ (∀ i < P, s.getD i 0 ≤ x) ∧
        (∀ i, P ≤ i → i < s.size → x < s.getD i 0) := by
  intro d
  induction d with
  | zero =>
    intro lo hi hd hle hhi hlo hhi'
    have heq : lo = hi := by omega
    subst heq
    refine ⟨lo, ?_, hhi, hlo, hhi'⟩
    rw [forIn_loop, Lean.Loop.forIn_eq_of_monadTail]
    simp [lastLeBody]; rfl
  | succ d ih =>
    intro lo hi hd hle hhi hlo hhi'
    rw [forIn_loop, Lean.Loop.forIn_eq_of_monadTail]
    by_cases hlt : lo < hi
    · have hmid : (lo + hi) / 2 < hi := by omega
      by_cases hs : s.getD ((lo + hi) / 2) 0 ≤ x
      · simp only [lastLeBody, hlt, hs, ite_true, pure_bind]
        exact ih _ _ (by omega) (by omega) hhi (fun i hi' => by
          by_cases h : i = (lo + hi) / 2
          · subst h; exact hs
          · exact Nat.le_trans (hmono i _ (by omega) (by omega)) hs) hhi'
      · simp only [lastLeBody, hlt, hs, ite_true, ite_false, pure_bind]
        exact ih _ _ (by omega) (by omega) (by omega) hlo (fun i hi' hi'' =>
          Nat.lt_of_lt_of_le (Nat.lt_of_not_le hs) (hmono _ i hi' hi''))
    · have heq : lo = hi := by omega
      subst heq
      refine ⟨lo, ?_, hhi, hlo, hhi'⟩
      simp [lastLeBody]; rfl

theorem find?_range_reverse (n P : ℕ) (p : ℕ → Bool) (hP : P ≤ n) (hlt : ∀ i < P, p i = true)
    (hge : ∀ i, P ≤ i → i < n → p i = false) : ((List.range n).reverse.find? p).getD 0 = P - 1 := by
  have hr : List.range n = List.range P ++ List.range' P (n - P) := by
    obtain ⟨k, rfl⟩ : ∃ k, n = P + k := ⟨n - P, by omega⟩
    simp [List.range_add, List.range'_eq_map_range]
  rw [hr, List.reverse_append, List.find?_append]
  have hnone : (List.range' P (n - P)).reverse.find? p = none := by
    rw [List.find?_eq_none]
    intro i hi
    rw [List.mem_reverse, List.mem_range'_1] at hi
    simp [hge i (by omega) (by omega)]
  rw [hnone, Option.none_or]
  cases P with
  | zero => simp
  | succ P =>
    rw [List.range_succ, List.reverse_append, List.reverse_singleton, List.singleton_append]
    simp [hlt P (by omega)]

/-- **`lastLe` on an ascending array** is the last index whose entry is at most `x`, or 0. -/
theorem lastLe_spec (s : Array ℕ) (x : ℕ) (hmono : ∀ i j, i ≤ j → j < s.size → s.getD i 0 ≤ s.getD j 0) :
    lastLe s x = ((List.range s.size).reverse.find? fun q => decide (s[q]! ≤ x)).getD 0 := by
  obtain ⟨P, hP, hPs, hlo, hhi⟩ := lastLe_loop s x hmono (s.size - 0) 0 s.size (Nat.le_refl _) (Nat.zero_le _)
    (Nat.le_refl _) (fun i h => absurd h (Nat.not_lt_zero _)) (fun i hi hi' => absurd hi' (by omega))
  rw [lastLe_eq, hP, find?_range_reverse s.size P _ hPs]
  · intro i hi
    have := hlo i hi
    have hi' : i < s.size := by omega
    simp only [decide_eq_true_eq, getElem!_pos s i hi']
    simpa [Array.getD, hi'] using this
  · intro i hi hi'
    have := hhi i hi hi'
    simp only [decide_eq_false_iff_not, Nat.not_le, getElem!_pos s i hi']
    simpa [Array.getD, hi'] using this

end lastLe

/-! ## The row words' starts ascend -/

theorem getD_push (a : Array ℕ) (v i : ℕ) :
    (a.push v).getD i 0 = if i < a.size then a.getD i 0 else if i = a.size then v else 0 := by
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_push]
  by_cases h : i < a.size
  · have h' : i ≠ a.size := by omega
    simp [h, h']
  · by_cases h' : i = a.size
    · simp [h']
    · simp [h, h']

theorem starts_foldl (l : List Flock.Port) : ∀ (acc : Array ℕ × ℕ),
    (∀ i j, i ≤ j → j < acc.1.size → acc.1.getD i 0 ≤ acc.1.getD j 0) → (∀ i < acc.1.size, acc.1.getD i 0 ≤ acc.2) →
    ∀ i j, i ≤ j →
      j < (l.foldl (fun (acc : Array ℕ × ℕ) p => (acc.1.push acc.2, acc.2 + p.words)) acc).1.size →
      (l.foldl (fun (acc : Array ℕ × ℕ) p => (acc.1.push acc.2, acc.2 + p.words)) acc).1.getD i 0 ≤
        (l.foldl (fun (acc : Array ℕ × ℕ) p => (acc.1.push acc.2, acc.2 + p.words)) acc).1.getD j 0 := by
  induction l with
  | nil => intro acc hm _; exact hm
  | cons p l ih =>
    intro acc hm hb
    refine ih _ (fun i j hij hj => ?_) (fun i hi => ?_)
    · simp only [Array.size_push] at hj
      rw [getD_push, getD_push]
      by_cases hj' : j < acc.1.size
      · have hi' : i < acc.1.size := by omega
        rw [if_pos hi', if_pos hj']; exact hm i j hij hj'
      · have hjs : j = acc.1.size := by omega
        rw [if_neg hj', if_pos hjs]
        by_cases hi' : i < acc.1.size
        · rw [if_pos hi']; exact hb i hi'
        · have his : i = acc.1.size := by omega
          rw [if_neg hi', if_pos his]
          exact Nat.le_refl _
    · simp only [Array.size_push] at hi
      rw [getD_push]
      by_cases hi' : i < acc.1.size
      · rw [if_pos hi']; exact Nat.le_trans (hb i hi') (Nat.le_add_right _ _)
      · rw [if_neg hi', if_pos (by omega)]; exact Nat.le_add_right _ _

theorem starts_mono (c : Flock.Circuit) :
    ∀ i j, i ≤ j → j < (Flock.KnownRead.starts c).size → (Flock.KnownRead.starts c).getD i 0 ≤ (Flock.KnownRead.starts c).getD j 0 := by
  unfold Flock.KnownRead.starts
  rw [← Array.foldl_toList]
  exact starts_foldl _ _ (fun i j _ hj => by simp at hj) (fun i hi => by simp at hi)

theorem typed_starts (c : Flock.Circuit) : FlockSoundness.Typed.starts c = Flock.KnownRead.starts c := rfl

/-! ## The first-match maps -/

theorem foldl_first {α K V : Type} [BEq K] [Hashable K] [LawfulBEq K] [LawfulHashable K]
    (P : α → Bool) (kf : α → K) (vf : α → V) :
    ∀ (l : List α) (m : HashMap K V) (k : K),
      (l.foldl (fun m a => if P a && !m.contains (kf a) then m.insert (kf a) (vf a) else m) m)[k]? =
        m[k]?.or ((l.find? fun a => P a && kf a == k).map vf)
  | [], m, k => by simp
  | a :: l, m, k => by
    rw [List.foldl_cons, foldl_first P kf vf l]
    by_cases hP : P a
    · by_cases hk : kf a = k
      · subst hk
        by_cases hc : m.contains (kf a)
        · obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 (HashMap.contains_eq_isSome_getElem?.symm.trans hc)
          simp [hP, hc, hv]
        · have hn : m[kf a]? = none := by
            rw [← Option.not_isSome_iff_eq_none, ← HashMap.contains_eq_isSome_getElem?]; simpa using hc
          simp [hP, hc, hn]
      · have hb : (kf a == k) = false := by simpa using hk
        by_cases hc : m.contains (kf a)
        · simp [hP, hc, hb]
        · simp [hP, hc, hb, HashMap.getElem?_insert]
    · simp [hP]

theorem foldl_first' {α K V : Type} [BEq K] [Hashable K] [LawfulBEq K] [LawfulHashable K]
    (kf : α → K) (vf : α → V) (l : List α) (k : K) :
      (l.foldl (fun (m : HashMap K V) a => if m.contains (kf a) then m else m.insert (kf a) (vf a)) ∅)[k]? =
        (l.find? fun a => kf a == k).map vf := by
  have h := foldl_first (fun _ => true) kf vf l ∅ k
  have he : (fun (m : HashMap K V) a => if (fun _ => true) a && !m.contains (kf a) then m.insert (kf a) (vf a) else m) =
      fun m a => if m.contains (kf a) then m else m.insert (kf a) (vf a) := by
    funext m a
    cases m.contains (kf a) <;> rfl
  rw [he] at h
  simpa using h

theorem cuts_get (c : Flock.Circuit) (k p : ℕ) :
    (Index.of c).cuts[(k, p)]? = (c.leafCuts.toList.find? fun e => decide (e.1 = k ∧ e.2.1 = p)).map (·.2.2) := by
  show (c.leafCuts.foldl (fun (m : HashMap (ℕ × ℕ) ℕ) e =>
    if m.contains (e.1, e.2.1) then m else m.insert (e.1, e.2.1) e.2.2) ∅)[(k, p)]? = _
  rw [← Array.foldl_toList, foldl_first']
  congr 2
  funext e
  rw [Bool.eq_iff_iff, beq_iff_eq]
  simp [Prod.ext_iff]

theorem wires_get (c : Flock.Circuit) (k p : ℕ) :
    (Index.of c).wires[(k, p)]? = (c.wires.toList.find? fun w => decide (w.dst = (c.unitNet, k, p))).map (·.src) := by
  show (c.wires.foldl (fun (m : HashMap (ℕ × ℕ) (ℕ × ℕ × ℕ)) w =>
    if (w.dst.1 == c.unitNet) && !m.contains (w.dst.2.1, w.dst.2.2) then m.insert (w.dst.2.1, w.dst.2.2) w.src
    else m) ∅)[(k, p)]? = _
  rw [← Array.foldl_toList, foldl_first, HashMap.getElem?_empty, Option.none_or]
  congr 2
  funext w
  rw [Bool.eq_iff_iff]
  simp [Prod.ext_iff]

/-! ## The code's key is the claims' -/

/-- `msgAt`'s key, at any `f` it unfolds to (`rfl` at a call): the claims' row-word key. -/
theorem msg_toA (c : Flock.Circuit) (x t : ℕ) (wide : Bool) (f : Src)
    (hf : f =
      let q := lastLe (Flock.KnownRead.starts c) x
      let s := (Flock.KnownRead.starts c).getD q 0
      if (wide || decide (t < 16)) && decide (16 * (x - s) + t < (c.ports.getD q default).bits) then
        .msg q (2 * (x - s) + t / 8) (t % 8)
      else .zero) :
    f.toA =
      let q := ((List.range (FlockSoundness.Typed.starts c).size).reverse.find? (fun q => (FlockSoundness.Typed.starts c)[q]! ≤ x)).getD 0
      if (wide || decide (t < 16)) && decide (16 * (x - (FlockSoundness.Typed.starts c)[q]!) + t < c.ports[q]!.bits) then
        .msg q (2 * (x - (FlockSoundness.Typed.starts c)[q]!) + t / 8) (t % 8)
      else .zero := by
  subst hf
  rw [lastLe_spec _ x (starts_mono c), typed_starts]
  simp only [Array.getElem!_eq_getD]
  rw [apply_ite Src.toA]
  rfl

theorem default_int : (default : Int) = 0 := rfl

theorem default_nat : (default : ℕ) = 0 := rfl

theorem srcAt_toA (c : Flock.Circuit) (k p t : ℕ) : (srcAt c (Index.of c) k p t).toA = Aliased.srcAt c k p t := by
  unfold srcAt Aliased.srcAt
  by_cases hp : p < (c.unit.inGroups.getD 0 default).bits.size
  · simp only [hp, ite_true, Array.getElem!_eq_getD, default_int]
    by_cases hl : (c.leavesIn.getD k #[]).getD p 0 < 0
    · simp only [hl, ite_true]; rfl
    · simp only [hl, ite_false]
      refine (msg_toA c ((c.leavesIn.getD k #[]).getD p 0).toNat t true _ ?_).trans ?_
      · rfl
      · simp only [Bool.true_or, Bool.true_and, decide_eq_true_eq, Array.getElem!_eq_getD]
  · simp only [hp, ite_false]
    rw [cuts_get]
    cases he : c.leafCuts.toList.filter (fun e => decide (e.1 = k ∧ e.2.1 = p)) with
    | cons e rest =>
      have hf : c.leafCuts.toList.find? (fun e => decide (e.1 = k ∧ e.2.1 = p)) = some e := by
        rw [← List.head?_filter, he]; rfl
      rw [hf]
      simp only [Option.map_some]
      refine (msg_toA c e.2.2 t false _ ?_).trans ?_
      · rfl
      · simp only [Bool.false_or, Array.getElem!_eq_getD]
    | nil =>
      have hf : c.leafCuts.toList.find? (fun e => decide (e.1 = k ∧ e.2.1 = p)) = none := by
        rw [← List.head?_filter, he]; rfl
      rw [hf]
      simp only [Option.map_none]
      rw [wires_get]
      cases hw : c.wires.toList.filter (fun w => decide (w.dst = (c.unitNet, k, p))) with
      | cons w rest =>
        have hf : c.wires.toList.find? (fun w => decide (w.dst = (c.unitNet, k, p))) = some w := by
          rw [← List.head?_filter, hw]; rfl
        rw [hf]
        rfl
      | nil =>
        have hf : c.wires.toList.find? (fun w => decide (w.dst = (c.unitNet, k, p))) = none := by
          rw [← List.head?_filter, hw]; rfl
        rw [hf]
        rfl

theorem mem_portBits (gs : Array Flock.Group) (off : ℕ) :
    off ∈ FlockSoundness.portBits gs ↔ ∃ p, p < (Flock.portCols gs).size ∧ (Flock.portCols gs)[p]!.1 ≤ off ∧
      off < (Flock.portCols gs)[p]!.1 + (Flock.portCols gs)[p]!.2 := by
  unfold FlockSoundness.portBits
  simp only [List.mem_flatMap, Array.mem_toList_iff, List.mem_range'_1]
  constructor
  · rintro ⟨q, hq, h⟩
    obtain ⟨p, hp, rfl⟩ := Array.mem_iff_getElem.1 hq
    exact ⟨p, hp, by rw [getElem!_pos _ p hp]; exact h⟩
  · rintro ⟨p, hp, h⟩
    rw [getElem!_pos _ p hp] at h
    exact ⟨_, Array.getElem_mem hp, h⟩

/-- **The code's key of stacked input `x` is the claims'** (`Aliased.srcKey`). -/
theorem srcKey_toA (c : Flock.Circuit) (x : ℕ) : (srcKey c (Index.of c) x).toA = Aliased.srcKey c x := by
  unfold srcKey Aliased.srcKey
  have hcol : ∀ p, (Flock.portCols c.unit.inGroups).getD p (0, 0) = (Flock.portCols c.unit.inGroups)[p]! := fun p =>
    Array.getElem!_eq_getD.symm
  simp only [hcol]
  split
  · rename_i p hp
    have hmem := List.mem_of_find?_eq_some hp
    have hpred := List.find?_some hp
    rw [List.mem_range] at hmem
    rw [decide_eq_true_eq] at hpred
    have hin : x % (c.unit.inWords * Flock.WORD) ∈ FlockSoundness.portBits c.unit.inGroups := (mem_portBits _ _).2 ⟨p, hmem, hpred⟩
    have hof : FlockSoundness.portOf (Flock.portCols c.unit.inGroups) (x % (c.unit.inWords * Flock.WORD)) = p := by
      unfold FlockSoundness.portOf; rw [hp]; rfl
    rw [if_pos hin, hof, srcAt_toA]
  · rename_i hn
    have hin : x % (c.unit.inWords * Flock.WORD) ∉ FlockSoundness.portBits c.unit.inGroups := by
      rw [mem_portBits]
      rintro ⟨p, hp, h⟩
      rw [List.find?_eq_none] at hn
      exact absurd (decide_eq_true h) (hn p (List.mem_range.2 hp))
    rw [if_neg hin]
    rfl

end Spot.FilePart
