module

public import Proofs.Properties.InferenceOnly.ReadsPieces
import all Init.Data.Array.QSort.Basic
import Batteries.Tactic.OpenPrivate
import Mathlib.Analysis.SpecialFunctions.Pow.NNReal
import Mathlib.Data.NNRat.Floor
import Mathlib.Tactic.ENatToNat
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.Measurability.Init
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Tactic.Polynomial.Basic
import Mathlib.Tactic.ReduceModChar
import Proofs.Core.QSort
import Proofs.Properties.InferenceOnly.Lemmas
import Proofs.Properties.Poub.Remote.Sort
import Std.Data.HashSet.Lemmas

/-!
# What `Desc.segDown` gives a segment

`Desc.segDown f s` resolves segment `s` with `f`, a piece at a time: what it gives names, position by position, what `f`
gives each position of `s` (`segDown_rel`), and each segment it gives is one `f` gives a piece of `s`'s copies, or `s`
with each piece the one piece `f` gives it, reshaped to its rows (`segDown_mem`). The loops that append what a function
gives each element (`forIn_trace`, `forIn_traceG`) are read here too.
-/

@[expose] public section

namespace InferenceOnlyVerifier.Properties.SecurityProofs.ReadsExact

open InferenceOnlyVerifier.Desc (Piece Seg PS Tgt)
open Flock.Program (Part Seq)

/-- A loop appending what `F` gives each element: the outputs, element by element, after the start. -/
theorem forIn_trace {β : Type} (F : β → Except String (Array Seg)) :
    ∀ (l : List β) (acc out : Array Seg), forIn l acc (fun q (r : Array Seg) => do
        let v ← F q
        pure (ForInStep.yield (r ++ v))) = .ok out →
      ∃ outs : List (Array Seg), List.Forall₂ (fun q v => F q = .ok v) l outs ∧ out = outs.foldl (· ++ ·) acc
  | [], acc, out, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    exact ⟨[], .nil, h.symm⟩
  | x :: l, acc, out, h => by
    rw [List.forIn_cons] at h
    obtain ⟨y, hy, h⟩ := Walk.except_bind_ok h
    obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
    simp only [pure, Except.pure, Except.ok.injEq] at hy
    subst hy
    obtain ⟨outs, h1, h2⟩ := forIn_trace F l (acc ++ v) out h
    exact ⟨v :: outs, .cons hv h1, h2⟩

theorem forall2_append {α β : Type} {R : α → β → Prop} : ∀ {l1 : List α} {m1 : List β} {l2 : List α}
    {m2 : List β}, List.Forall₂ R l1 m1 → List.Forall₂ R l2 m2 → List.Forall₂ R (l1 ++ l2) (m1 ++ m2)
  | _, _, _, _, .nil, h2 => h2
  | _, _, _, _, .cons h hrest, h2 => .cons h (forall2_append hrest h2)

theorem segsPos_foldl : ∀ (outs : List (Array Seg)) (acc : Array Seg),
    segsPos (outs.foldl (· ++ ·) acc) = segsPos acc ++ outs.flatMap segsPos
  | [], acc => by simp
  | v :: outs, acc => by
    rw [List.foldl_cons, segsPos_foldl outs, segsPos_append, List.flatMap_cons, List.append_assoc]

theorem mem_foldl : ∀ (outs : List (Array Seg)) (acc : Array Seg) (g : Seg),
    g ∈ (outs.foldl (· ++ ·) acc).toList → g ∈ acc.toList ∨ ∃ v ∈ outs, g ∈ v.toList
  | [], acc, g, h => .inl h
  | v :: outs, acc, g, h => by
    rw [List.foldl_cons] at h
    rcases mem_foldl outs _ g h with h | ⟨w, hw, hg⟩
    · rw [Array.toList_append, List.mem_append] at h
      rcases h with h | h
      · exact .inl h
      · exact .inr ⟨v, List.mem_cons_self, h⟩
    · exact .inr ⟨w, List.mem_cons_of_mem _ hw, hg⟩

theorem forall2_mem {α β : Type} {R : α → β → Prop} : ∀ {l : List α} {m : List β}, List.Forall₂ R l m →
    ∀ b ∈ m, ∃ a ∈ l, R a b
  | _, _, .nil, _, hb => absurd hb (by simp)
  | _, _, .cons (a := a) (b := b') h hrest, b, hb => by
    rcases List.mem_cons.1 hb with rfl | hb
    · exact ⟨a, List.mem_cons_self, h⟩
    · obtain ⟨a', ha', h'⟩ := forall2_mem hrest b hb
      exact ⟨a', List.mem_cons_of_mem _ ha', h'⟩

/-- Two such loops nested, the inner one over `G b`. -/
theorem forIn_traceG {β : Type} (F : Piece → Except String (Array Seg)) (G : β → Array Piece) :
    ∀ (l : List β) (acc out : Array Seg), forIn l acc (fun (b : β) (r : Array Seg) => do
        let r' ← forIn (G b) r (fun q (r : Array Seg) => do
          let v ← F q
          pure (ForInStep.yield (r ++ v)))
        pure (ForInStep.yield r')) = .ok out →
      ∃ outs : List (Array Seg), List.Forall₂ (fun q v => F q = .ok v) (l.flatMap fun b => (G b).toList) outs ∧
        out = outs.foldl (· ++ ·) acc
  | [], acc, out, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    exact ⟨[], .nil, h.symm⟩
  | b :: l, acc, out, h => by
    rw [List.forIn_cons] at h
    obtain ⟨y, hy, h⟩ := Walk.except_bind_ok h
    obtain ⟨r', hr', hy⟩ := Walk.except_bind_ok hy
    simp only [pure, Except.pure, Except.ok.injEq] at hy
    subst hy
    rw [← Array.forIn_toList] at hr'
    obtain ⟨o1, h1, e1⟩ := forIn_trace F _ acc r' hr'
    obtain ⟨o2, h2, e2⟩ := forIn_traceG F G l r' out h
    refine ⟨o1 ++ o2, ?_, ?_⟩
    · rw [List.flatMap_cons]; exact forall2_append h1 h2
    · rw [e2, e1, List.foldl_append]

theorem list_mapM_ok {α β ε : Type} {f : α → Except ε β} :
    ∀ (l : List α) (r : List β), l.mapM f = .ok r → List.Forall₂ (fun a b => f a = .ok b) l r
  | [], r, h => by
    simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h; exact .nil
  | a :: l, r, h => by
    rw [List.mapM_cons] at h
    obtain ⟨b, hb, h⟩ := Walk.except_bind_ok h
    obtain ⟨bs, hbs, h⟩ := Walk.except_bind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact .cons hb (list_mapM_ok l bs hbs)

theorem array_mapM_ok {α β ε : Type} {f : α → Except ε β} (l : Array α) (r : Array β) (h : l.mapM f = .ok r) :
    List.Forall₂ (fun a b => f a = .ok b) l.toList r.toList := by
  rw [Array.mapM_eq_mapM_toList] at h
  rcases hx : l.toList.mapM f with e | r'
  · rw [hx] at h; cases h
  · rw [hx] at h
    have : r = r'.toArray := by
      simp only [Functor.map, Except.map, Except.ok.injEq] at h; exact h.symm
    subst this
    exact list_mapM_ok _ _ hx

theorem list_mapM_some {α β : Type} {f : α → Option β} :
    ∀ (l : List α) (r : List β), l.mapM f = some r → List.Forall₂ (fun a b => f a = some b) l r
  | [], r, h => by
    simp only [List.mapM_nil, pure, Option.some.injEq] at h
    subst h; exact .nil
  | a :: l, r, h => by
    rw [List.mapM_cons] at h
    simp only [bind, Option.bind_eq_some_iff, pure, Option.some.injEq] at h
    obtain ⟨b, hb, bs, hbs, rfl⟩ := h
    exact .cons hb (list_mapM_some l bs hbs)

theorem array_mapM_some {α β : Type} {f : α → Option β} (l : Array α) (r : Array β) (h : l.mapM f = some r) :
    List.Forall₂ (fun a b => f a = some b) l.toList r.toList := by
  rw [Array.mapM_eq_mapM_toList] at h
  rcases hx : l.toList.mapM f with _ | r'
  · rw [hx] at h; cases h
  · rw [hx] at h
    have : r = r'.toArray := by
      simp only [Functor.map, Option.map, Option.some.injEq] at h; exact h.symm
    subst this
    exact list_mapM_some _ _ hx

/-- A list `l` related by `A` to `r`, and `l.zip r` by `B` to `ps`: each of `ps` related to its element of `l`. -/
theorem forall2_zip {α β γ : Type} {A : α → β → Prop} {B : α × β → γ → Prop} :
    ∀ {l : List α} {r : List β} {ps : List γ}, List.Forall₂ A l r → List.Forall₂ B (l.zip r) ps →
      List.Forall₂ (fun p q => q ∈ l ∧ ∃ b, A q b ∧ B (q, b) p) ps l
  | _, _, _, .nil, h => by cases h; exact .nil
  | _, _, _, .cons ha hrest, h => by
    rw [List.zip_cons_cons] at h
    cases h with
    | cons hb h' =>
      exact .cons ⟨List.mem_cons_self, _, ha, hb⟩
        ((forall2_zip hrest h').imp fun _ _ ⟨hm, hx⟩ => ⟨List.mem_cons_of_mem _ hm, hx⟩)

theorem forall2_mem_imp {α β : Type} {R : α → β → Prop} :
    ∀ {l : List α} {m : List β}, List.Forall₂ R l m → List.Forall₂ (fun a b => a ∈ l ∧ R a b) l m
  | _, _, .nil => .nil
  | _, _, .cons h hrest =>
    .cons ⟨List.mem_cons_self, h⟩ ((forall2_mem_imp hrest).imp fun _ _ ⟨ha, hr⟩ => ⟨List.mem_cons_of_mem _ ha, hr⟩)

theorem forall2_flatMap {α β γ δ : Type} {R : γ → δ → Prop} {S : α → β → Prop} {F : α → List γ} {G : β → List δ}
    (hFG : ∀ a b, S a b → List.Forall₂ R (F a) (G b)) :
    ∀ {l : List α} {m : List β}, List.Forall₂ S l m → List.Forall₂ R (l.flatMap F) (m.flatMap G)
  | _, _, .nil => .nil
  | _, _, .cons h hrest => by
    rw [List.flatMap_cons, List.flatMap_cons]
    exact forall2_append (hFG _ _ h) (forall2_flatMap hFG hrest)

theorem forall2_flatMap_same {α γ δ : Type} {R : γ → δ → Prop} {F : α → List γ} {G : α → List δ} :
    ∀ {l : List α}, (∀ a ∈ l, List.Forall₂ R (F a) (G a)) → List.Forall₂ R (l.flatMap F) (l.flatMap G)
  | [], _ => .nil
  | a :: l, h => by
    rw [List.flatMap_cons, List.flatMap_cons]
    exact forall2_append (h a List.mem_cons_self) (forall2_flatMap_same fun b hb => h b (List.mem_cons_of_mem _ hb))

/-- What `F` gives each piece, laid end to end, related position by position to the pieces' positions. -/
theorem rel_trace {R : Tgt × Int → Tgt × Int → Prop} {F : Piece → Except String (Array Seg)} :
    ∀ {qs : List Piece} {outs : List (Array Seg)}, List.Forall₂ (fun q v => F q = .ok v) qs outs →
      (∀ q ∈ qs, ∀ v, F q = .ok v → List.Forall₂ R (segsPos v) (pos q)) →
      List.Forall₂ R (outs.flatMap segsPos) (qs.flatMap pos)
  | _, _, .nil, _ => .nil
  | _, _, .cons (a := q) (b := v) hv hrest, hF => by
    rw [List.flatMap_cons, List.flatMap_cons]
    exact forall2_append (hF q List.mem_cons_self v hv)
      (rel_trace hrest fun q' hq' => hF q' (List.mem_cons_of_mem _ hq'))

theorem forall2_sl {α β : Type} {R : α → β → Prop} {l : List α} {m : List β} (h : List.Forall₂ R l m) (a b : Nat) :
    List.Forall₂ R (sl l a b) (sl m a b) :=
  List.forall₂_take _ (List.forall₂_drop _ h)

theorem sl_flatMap_range {α : Type} (f : Nat → List α) (L : Nat) (hf : ∀ r, (f r).length = L) (n m : Nat)
    (hm : m < n) : sl ((List.range n).flatMap f) (m * L) ((m + 1) * L) = f m := by
  apply List.ext_getElem?
  intro t
  have hL : (m + 1) * L - m * L = L := by rw [Nat.succ_mul]; omega
  simp only [sl, hL, List.getElem?_take, List.getElem?_drop, flatMap_range_getElem? f L hf]
  by_cases ht : t < L
  · have h1 : m * L + t < n * L := by
      have : (m + 1) * L ≤ n * L := Nat.mul_le_mul_right _ hm
      rw [Nat.succ_mul] at this; omega
    have h2 : (m * L + t) / L = m := by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt ht, Nat.zero_add]
    have h3 : (m * L + t) % L = t := by
      rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt ht]
    simp [ht, h1, h2, h3]
  · rw [if_neg ht]; symm; apply List.getElem?_eq_none; rw [hf]; omega

/-- Row `m` of a piece: its positions `[m·len, (m+1)·len)`. -/
theorem pos_row_sl (q : Piece) (m : Nat) (hm : m < q.rows) :
    pos (q.row m) = sl (pos q) (m * q.len) ((m + 1) * q.len) := by
  rw [pos_eq_rows q, sl_flatMap_range _ q.len (fun r => by rw [pos_row]; simp) q.rows m hm]

/-- `y` reshaped to `rows` rows of `len`: row `m` its positions `[m·len, (m+1)·len)`. -/
theorem reshape_row (y : Piece) (rows len : Nat) (p : Piece) (h : y.reshape rows len = some p) (m : Nat)
    (hm : m < rows) : pos (p.row m) = sl (pos y) (m * len) ((m + 1) * len) := by
  unfold Desc.Piece.reshape at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, beq_iff_eq] at hc
    obtain ⟨h1, h2⟩ := hc
    cases h
    subst h1 h2
    exact pos_row_sl _ m hm
  · split at h
    · rename_i _ hc
      simp only [Bool.and_eq_true, List.isEmpty_iff, beq_iff_eq] at hc
      obtain ⟨⟨hr, hlv⟩, hl⟩ := hc
      cases h
      have hb : (m + 1) * len ≤ rows * len := Nat.mul_le_mul_right _ hm
      have hy : pos y = (List.range (rows * len)).map fun (t : Nat) => (y.tgt, y.base + (t : Int) * y.stride) := by
        rw [pos_eq_rows y, hr, List.range_one, List.flatMap_singleton, pos_row, hl]
        simp [Desc.Piece.rowOff, hlv]
      have hL : (m + 1) * len - m * len = len := by rw [Nat.succ_mul]; omega
      rw [hy, sl_map_range _ _ _ _ hb, pos_row, List.range'_eq_map_range, List.map_map, hL]
      dsimp only
      apply List.map_congr_left
      intro t _
      simp only [Function.comp_apply, Desc.Piece.rowOff, hlv, List.isEmpty_nil, ↓reduceIte, Prod.mk.injEq,
        true_and]
      push_cast
      ring
    · cases h

theorem reshape_sub (y : Piece) (rows len : Nat) (p : Piece) (h : y.reshape rows len = some p) (m : Nat)
    (hm : m < rows) : ∀ x ∈ pos (p.row m), x ∈ pos y := by
  intro x hx
  rw [reshape_row y rows len p h m hm] at hx
  exact List.mem_of_mem_drop (List.mem_of_mem_take hx)

/-- `segDown`'s test that a rowed segment's piece resolves to one piece of its rows. -/
theorem one_spec {s : Seg} {q : Piece} {r : Array Seg} {p : Piece}
    (h : (if r.size == 1 && q.rows == s.reps then r[0]!.single.bind (·.reshape q.rows q.len) else none) = some p) :
    q.rows = s.reps ∧ ∃ x y, r = #[x] ∧ x.single = some y ∧ y.reshape q.rows q.len = some p := by
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, beq_iff_eq] at hc
    rw [Option.bind_eq_some_iff] at h
    obtain ⟨y, hy, hre⟩ := h
    refine ⟨hc.2, r[0]!, y, ?_, hy, hre⟩
    rcases r with ⟨_ | ⟨x, _ | ⟨x2, l⟩⟩⟩ <;> simp at hc ⊢
  · cases h

theorem range_size (n : Nat) : List.range' 0 [:n].size = List.range' 0 n := by
  simp [Std.Legacy.Range.size]

/-- **What `segDown` gives a segment names what `f` gives each of its positions**: related position by position (`R`)
to `s`'s, where each piece `f` resolves, its positions `s`'s, gives segments so related to its positions. -/
theorem segDown_rel {R : Tgt × Int → Tgt × Int → Prop} (f : Piece → Except String (Array Seg)) (s : Seg)
    (hf : ∀ q out, (∀ x ∈ pos q, x ∈ segPos s) → f q = .ok out → List.Forall₂ R (segsPos out) (pos q))
    (out : Array Seg) (h : Desc.segDown f s = .ok out) : List.Forall₂ R (segsPos out) (segPos s) := by
  unfold Desc.segDown at h
  cases hr : s.lanes
  · simp only [hr, Bool.not_false, ↓reduceIte] at h
    obtain ⟨-, -, h⟩ := Walk.except_bind_ok h
    obtain ⟨o, ho, h⟩ := Walk.except_bind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    rw [← Array.forIn_toList] at ho
    obtain ⟨outs, hF, rfl⟩ := forIn_trace f _ #[] o ho
    rw [segsPos_foldl, segsPos_empty, List.nil_append]
    have hL : segPos s = ((Array.range s.reps).flatMap s.copy).toList.flatMap pos := by
      simp only [segPos, posAll, Array.toList_flatMap, Array.toList_range, List.flatMap_assoc]
    rw [hL]
    exact rel_trace hF fun q hq v hv => hf q v (fun x hx => by rw [hL]; exact List.mem_flatMap.2 ⟨q, hq, hx⟩) hv
  · simp only [hr, Bool.not_true, Bool.false_eq_true, ↓reduceIte] at h
    obtain ⟨rs, hrs, h⟩ := Walk.except_bind_ok h
    have hrs2 := array_mapM_ok _ _ hrs
    have hsub : ∀ q ∈ s.ps.toList, q.rows = s.reps → ∀ x ∈ pos q, x ∈ segPos s := by
      intro q hq hrq x hx
      rw [pos_eq_rows q, hrq] at hx
      obtain ⟨m, hm, hx⟩ := List.mem_flatMap.1 hx
      refine List.mem_flatMap.2 ⟨m, hm, ?_⟩
      rw [copy_rowed s hr, posAll_map]
      exact List.mem_flatMap.2 ⟨q, hq, hx⟩
    split at h
    · rename_i ps hps
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      have hpair := forall2_zip hrs2 (by rw [← Array.toList_zip]; exact array_mapM_some _ _ hps)
      rw [segsPos_singleton]
      simp only [segPos]
      apply forall2_flatMap_same
      intro m hm
      rw [copy_rowed s hr, copy_rowed { s with ps := ps, shift := 0 } hr, posAll_map, posAll_map]
      refine forall2_flatMap (fun p q hpq => ?_) hpair
      obtain ⟨hq, r, hfr, hone⟩ := hpq
      obtain ⟨hrq, x, y, rfl, hy, hre⟩ := one_spec hone
      have hxy := hf q #[x] (hsub q hq hrq) hfr
      rw [segsPos_singleton, single_pos x y hy] at hxy
      have hmq : m < q.rows := by rw [hrq]; exact List.mem_range.1 hm
      rw [reshape_row y q.rows q.len p hre m hmq, pos_row_sl q m hmq]
      exact forall2_sl hxy _ _
    · obtain ⟨-, -, h⟩ := Walk.except_bind_ok h
      obtain ⟨o, ho, h⟩ := Walk.except_bind_ok h
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      rw [Std.Legacy.Range.forIn_eq_forIn_range'] at ho
      obtain ⟨outs, hF, rfl⟩ := forIn_traceG f s.copy _ #[] o ho
      rw [segsPos_foldl, segsPos_empty, List.nil_append]
      have hL : segPos s = ((List.range' 0 s.reps).flatMap fun m => (s.copy m).toList).flatMap pos := by
        simp only [segPos, posAll, List.flatMap_assoc, List.range_eq_range']
      rw [range_size] at hF
      rw [hL]
      exact rel_trace hF fun q hq v hv => hf q v (fun x hx => by rw [hL]; exact List.mem_flatMap.2 ⟨q, hq, hx⟩) hv

theorem reshape_dims (y : Piece) (rows len : Nat) (p : Piece) (h : y.reshape rows len = some p) :
    p.rows = rows ∧ p.len = len := by
  unfold Desc.Piece.reshape at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, beq_iff_eq] at hc
    cases h; exact hc
  · split at h
    · cases h; exact ⟨rfl, rfl⟩
    · cases h

theorem copy_count (s : Seg) (m : Nat) (q : Piece) (hq : q ∈ (s.copy m).toList) :
    ∃ p ∈ s.ps.toList, q.count = p.count ∨ q.count = p.len := by
  unfold Desc.Seg.copy at hq
  split at hq
  · rw [Array.toList_map, List.mem_map] at hq
    obtain ⟨p, hp, rfl⟩ := hq
    exact ⟨p, hp, .inr (by simp [Desc.Piece.count, Desc.Piece.row])⟩
  · unfold Desc.shiftGates at hq
    rw [Array.toList_map, List.mem_map] at hq
    obtain ⟨p, hp, rfl⟩ := hq
    refine ⟨p, hp, .inl ?_⟩
    split <;> rfl

theorem forall2_true {α β : Type} : ∀ {l : List α} {m : List β}, l.length = m.length →
    List.Forall₂ (fun _ _ => True) l m
  | [], [], _ => .nil
  | _ :: _, _ :: _, h => .cons trivial (forall2_true (by simpa using h))
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h

/-- **Each segment `segDown` gives** is one `f` gives a piece of `s`'s copies, of positive count and its positions
`s`'s; or `s` with its pieces each reshaped from what `f` gives it, each position one `f` gives such a piece, each piece's
positions those of a piece of one segment `f` gives such a piece, its pieces `s`'s counts and its copies `s`'s length. -/
theorem segDown_mem (f : Piece → Except String (Array Seg)) (s : Seg)
    (hpos : ∀ p ∈ s.ps.toList, 0 < p.count)
    (out : Array Seg) (h : Desc.segDown f s = .ok out) (g : Seg) (hg : g ∈ out.toList) :
    (∃ q v, 0 < q.count ∧ (∀ x ∈ pos q, x ∈ segPos s) ∧ f q = .ok v ∧ g ∈ v.toList) ∨
    ((∀ x ∈ segPos g, ∃ q v, 0 < q.count ∧ (∀ x ∈ pos q, x ∈ segPos s) ∧ f q = .ok v ∧ x ∈ segsPos v) ∧
      (∀ p ∈ g.ps.toList, ∃ q v, 0 < q.count ∧ (∀ x ∈ pos q, x ∈ segPos s) ∧ f q = .ok v ∧
        ∃ x ∈ v.toList, ∃ y ∈ x.ps.toList, ∀ z ∈ pos p, z ∈ pos y) ∧
      (∀ p ∈ g.ps.toList, ∃ q ∈ s.ps.toList, p.count = q.count) ∧ g.reps = s.reps ∧ g.len = s.len ∧
      g.shift = 0 ∧ (posAll (g.copy 0)).length = (posAll (s.copy 0)).length ∧ ∀ p ∈ g.ps.toList, p.rows = g.reps) := by
  have hcopy : ∀ L : List Nat, ∀ q ∈ L.flatMap (fun m => (s.copy m).toList), 0 < q.count := by
    intro L q hq
    obtain ⟨m, -, hq⟩ := List.mem_flatMap.1 hq
    obtain ⟨p, hp, e | e⟩ := copy_count s m q hq
    · rw [e]; exact hpos p hp
    · rw [e]; have := hpos p hp; unfold Desc.Piece.count at this
      exact Nat.pos_of_ne_zero (Nat.mul_ne_zero_iff.1 (Nat.pos_iff_ne_zero.1 this)).2
  unfold Desc.segDown at h
  cases hr : s.lanes
  · simp only [hr, Bool.not_false, ↓reduceIte] at h
    obtain ⟨-, -, h⟩ := Walk.except_bind_ok h
    obtain ⟨o, ho, h⟩ := Walk.except_bind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    rw [← Array.forIn_toList] at ho
    obtain ⟨outs, hF, rfl⟩ := forIn_trace f _ #[] o ho
    rcases mem_foldl outs #[] g hg with hg | ⟨v, hv, hgv⟩
    · simp at hg
    obtain ⟨q, hq, hfq⟩ := forall2_mem hF v hv
    have hL : segPos s = ((Array.range s.reps).flatMap s.copy).toList.flatMap pos := by
      simp only [segPos, posAll, Array.toList_flatMap, Array.toList_range, List.flatMap_assoc]
    have hq' : q ∈ (List.range s.reps).flatMap fun m => (s.copy m).toList := by
      simpa only [Array.toList_flatMap, Array.toList_range] using hq
    exact .inl ⟨q, v, hcopy _ q hq', fun x hx => by rw [hL]; exact List.mem_flatMap.2 ⟨q, hq, hx⟩, hfq, hgv⟩
  · simp only [hr, Bool.not_true, Bool.false_eq_true, ↓reduceIte] at h
    obtain ⟨rs, hrs, h⟩ := Walk.except_bind_ok h
    have hrs2 := array_mapM_ok _ _ hrs
    have hsub : ∀ q ∈ s.ps.toList, q.rows = s.reps → ∀ x ∈ pos q, x ∈ segPos s := by
      intro q hq hrq x hx
      rw [pos_eq_rows q, hrq] at hx
      obtain ⟨m, hm, hx⟩ := List.mem_flatMap.1 hx
      refine List.mem_flatMap.2 ⟨m, hm, ?_⟩
      rw [copy_rowed s hr, posAll_map]
      exact List.mem_flatMap.2 ⟨q, hq, hx⟩
    split at h
    · rename_i ps hps
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      have := List.mem_singleton.1 hg
      subst this
      have hpair := forall2_zip hrs2 (by rw [← Array.toList_zip]; exact array_mapM_some _ _ hps)
      have hone : ∀ p ∈ ps.toList, ∃ q ∈ s.ps.toList, q.rows = s.reps ∧ ∃ x y, f q = .ok #[x] ∧
          x.single = some y ∧ y.reshape q.rows q.len = some p := by
        intro p hp
        obtain ⟨q, -, hq, r, hfr, hone⟩ := forall2_mem hpair.flip p hp
        obtain ⟨hrq, x, y, rfl, hy, hre⟩ := one_spec hone
        exact ⟨q, hq, hrq, x, y, hfr, hy, hre⟩
      refine .inr ⟨fun x hx => ?_, fun p hp => ?_, fun p hp => ?_, rfl, rfl, rfl, ?_, fun p hp => ?_⟩
      · obtain ⟨m, hm, hx⟩ := List.mem_flatMap.1 hx
        rw [copy_rowed { s with ps := ps, shift := 0 } hr, posAll_map] at hx
        obtain ⟨p, hp, hx⟩ := List.mem_flatMap.1 hx
        obtain ⟨q, hq, hrq, x', y, hfr, hy, hre⟩ := hone p hp
        have hmq : m < q.rows := by rw [hrq]; exact List.mem_range.1 hm
        refine ⟨q, #[x'], hpos q hq, hsub q hq hrq, hfr, ?_⟩
        rw [segsPos_singleton, single_pos x' y hy]
        exact reshape_sub y _ _ p hre m hmq x hx
      · obtain ⟨q, hq, hrq, x', y, hfr, hy, hre⟩ := hone p hp
        refine ⟨q, #[x'], hpos q hq, hsub q hq hrq, hfr, x', by simp, y, single_mem hy, fun z hz => ?_⟩
        rw [pos_eq_rows p] at hz
        obtain ⟨m, hm, hz⟩ := List.mem_flatMap.1 hz
        obtain ⟨h1, -⟩ := reshape_dims y _ _ p hre
        exact reshape_sub y _ _ p hre m (by rw [← h1]; exact List.mem_range.1 hm) z hz
      · obtain ⟨q, hq, -, x', y, -, -, hre⟩ := hone p hp
        obtain ⟨h1, h2⟩ := reshape_dims y _ _ p hre
        exact ⟨q, hq, by simp [Desc.Piece.count, h1, h2]⟩
      · rw [copy_rowed s hr, copy_rowed { s with ps := ps, shift := 0 } hr, posAll_map, posAll_map]
        refine List.Forall₂.length_eq (R := fun _ _ => True) (forall2_flatMap (fun p q hpq => ?_) hpair)
        obtain ⟨-, r, -, hone⟩ := hpq
        obtain ⟨-, x, y, -, -, hre⟩ := one_spec hone
        obtain ⟨-, h2⟩ := reshape_dims y _ _ p hre
        exact forall2_true (by simp [pos_row, h2])
      · obtain ⟨q, -, hrq, x', y, -, -, hre⟩ := hone p hp
        exact (reshape_dims y _ _ p hre).1.trans hrq
    · obtain ⟨-, -, h⟩ := Walk.except_bind_ok h
      obtain ⟨o, ho, h⟩ := Walk.except_bind_ok h
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      rw [Std.Legacy.Range.forIn_eq_forIn_range'] at ho
      obtain ⟨outs, hF, rfl⟩ := forIn_traceG f s.copy _ #[] o ho
      rw [range_size] at hF
      rcases mem_foldl outs #[] g hg with hg | ⟨v, hv, hgv⟩
      · simp at hg
      obtain ⟨q, hq, hfq⟩ := forall2_mem hF v hv
      have hL : segPos s = ((List.range' 0 s.reps).flatMap fun m => (s.copy m).toList).flatMap pos := by
        simp only [segPos, posAll, List.flatMap_assoc, List.range_eq_range']
      exact .inl ⟨q, v, hcopy _ q hq, fun x hx => by rw [hL]; exact List.mem_flatMap.2 ⟨q, hq, hx⟩, hfq, hgv⟩

/-- A segment of positive count is of pieces of positive count. -/
def CountsOk (g : Seg) : Prop := 0 < g.count → ∀ p ∈ g.ps.toList, 0 < p.count

theorem shiftGates_tgt {δ : Int} {ps : Array Piece} {q : Piece} (h : q ∈ (Desc.shiftGates δ ps).toList) :
    ∃ g ∈ ps.toList, q.tgt = g.tgt := by
  unfold Desc.shiftGates at h
  rw [Array.toList_map, List.mem_map] at h
  obtain ⟨g, hg, rfl⟩ := h
  refine ⟨g, hg, ?_⟩
  split <;> rfl

theorem copy_tgt {s : Seg} {m : Nat} {q : Piece} (h : q ∈ (s.copy m).toList) : ∃ g ∈ s.ps.toList, q.tgt = g.tgt := by
  unfold Desc.Seg.copy at h
  split at h
  · rw [Array.toList_map, List.mem_map] at h
    obtain ⟨g, hg, rfl⟩ := h
    exact ⟨g, hg, rfl⟩
  · exact shiftGates_tgt h

theorem forIn_congr_mem {β σ : Type} (l : List β) {f g : β → σ → Except String (ForInStep σ)}
    (h : ∀ x ∈ l, f x = g x) (init : σ) : forIn l init f = forIn l init g := by
  induction l generalizing init with
  | nil => rfl
  | cons x l ih =>
    rw [List.forIn_cons, List.forIn_cons, h x List.mem_cons_self]
    congr 1
    funext r
    cases r with
    | done => rfl
    | yield b => exact ih (fun y hy => h y (List.mem_cons_of_mem _ hy)) b

theorem forIn_congr_arr {β σ : Type} (l : Array β) {f g : β → σ → Except String (ForInStep σ)}
    (h : ∀ x ∈ l.toList, f x = g x) (init : σ) : forIn l init f = forIn l init g := by
  rw [← Array.forIn_toList, ← Array.forIn_toList]
  exact forIn_congr_mem _ h init

theorem array_mapM_congr {α β : Type} {f g : α → Except String β} (l : Array α) (h : ∀ x ∈ l.toList, f x = g x) :
    l.mapM f = l.mapM g := by
  rw [Array.mapM_eq_mapM_toList, Array.mapM_eq_mapM_toList]
  congr 1
  generalize l.toList = L at h ⊢
  induction L with
  | nil => rfl
  | cons x L ih => rw [List.mapM_cons, List.mapM_cons, h x List.mem_cons_self, ih (fun y hy => h y (List.mem_cons_of_mem _ hy))]

/-- `segDown` applies its resolver only to pieces of `s`'s targets. -/
theorem segDown_congr {f f' : Piece → Except String (Array Seg)} (s : Seg)
    (H : ∀ q, (∃ g ∈ s.ps.toList, q.tgt = g.tgt) → f q = f' q) : Desc.segDown f s = Desc.segDown f' s := by
  have hc : ∀ m, ∀ q ∈ (s.copy m).toList, f q = f' q := fun m q hq => H q (copy_tgt hq)
  unfold Desc.segDown
  cases hr : s.lanes
  · simp only [Bool.not_false, ↓reduceIte]
    congr 1
    funext u
    congr 1
    apply forIn_congr_arr
    intro q hq
    funext r
    simp only [Array.toList_flatMap, List.mem_flatMap] at hq
    obtain ⟨m, -, hq⟩ := hq
    rw [hc m q hq]
  · simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte]
    rw [array_mapM_congr s.ps (fun q hq => H q ⟨q, hq, rfl⟩)]
    congr 1
    funext rs
    split
    · rfl
    congr 1
    funext u
    congr 1
    rw [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.forIn_eq_forIn_range']
    apply forIn_congr_mem
    intro m _
    funext r
    congr 1
    apply forIn_congr_arr
    intro q hq
    funext r'
    rw [hc m q hq]

theorem countsOk_one (x : Piece) : CountsOk (Seg.one x) := by
  intro h p hp
  simp only [Seg.one, List.mem_singleton, List.toList_toArray] at hp
  subst hp
  simpa [Desc.Seg.count, Seg.one] using h

theorem ofPart_counts (q : Part) : ∀ g ∈ (Desc.ofPart q).toList, CountsOk g := by
  have hone : ∀ (X : Array Piece) g, g ∈ (X.map Seg.one).toList → CountsOk g := fun X g hg => by
    rw [Array.toList_map, List.mem_map] at hg
    obtain ⟨x, -, rfl⟩ := hg
    exact countsOk_one x
  have hone1 : ∀ (x : Piece) g, g ∈ #[Seg.one x].toList → CountsOk g := fun x g hg => by
    simp at hg; subst hg; exact countsOk_one x
  cases q with
  | affine p i b c => exact hone1 _
  | strided p i b inner os is c =>
    simp only [Desc.ofPart]
    split_ifs
    · intro g hg; simp at hg
    · exact hone1 _
    · exact hone1 _
    · exact hone _
  | explicit it => exact hone _
  | grid run lanes =>
    simp only [Desc.ofPart, Desc.ofGrid]
    split
    · intro g hg
      rw [Array.toList_map, List.mem_map] at hg
      obtain ⟨l, -, rfl⟩ := hg
      exact countsOk_one _
    · intro g hg
      simp at hg
      subst hg
      intro h p hp
      · rw [Array.toList_map, List.mem_map] at hp
        obtain ⟨l, -, rfl⟩ := hp
        obtain ⟨-, -, -, h4, h5, -⟩ := lanePiece_spec run ((lanes[0]?.map Flock.Program.Lane.periods).getD 0) l
        simp only [Desc.Seg.count] at h
        rw [Desc.Piece.count, h4, h5]
        rw [Nat.pos_iff_ne_zero, Nat.mul_ne_zero_iff, Nat.mul_ne_zero_iff] at h
        exact Nat.mul_pos (Nat.pos_of_ne_zero h.1) (Nat.pos_of_ne_zero h.2.2)

theorem partsFold_counts : ∀ (l : List Part) (acc : Array Seg), (∀ g ∈ acc.toList, CountsOk g) →
    ∀ g ∈ (l.foldl (fun acc q => acc ++ Desc.ofPart q) acc).toList, CountsOk g
  | [], _, h => h
  | q :: l, acc, h => by
    rw [List.foldl_cons]
    refine partsFold_counts l _ fun g hg => ?_
    rw [Array.toList_append, List.mem_append] at hg
    rcases hg with hg | hg
    · exact h g hg
    · exact ofPart_counts q g hg

/-- **A sequence's segments** (`Desc.ofSeq`) are each of one copy at least, of pieces of positive count. -/
theorem ofSeq_counts (s : Seq) : ∀ g ∈ (Desc.ofSeq s).segs.toList, 1 ≤ g.reps ∧ ∀ p ∈ g.ps.toList, 0 < p.count := by
  intro g hg
  unfold Desc.ofSeq at hg
  rw [of_segs, Array.toList_filter, List.mem_filter] at hg
  obtain ⟨hg, hc⟩ := hg
  rw [← Array.foldl_toList] at hg
  have hc' : 0 < g.count := by simpa using hc
  refine ⟨?_, partsFold_counts _ _ (by simp) g hg hc'⟩
  unfold Desc.Seg.count at hc'
  exact Nat.pos_of_ne_zero (Nat.mul_ne_zero_iff.1 (Nat.pos_iff_ne_zero.1 hc')).1

/-- Rows `[lo, lo + n)` of a piece name what its rows `lo + m`, `m < n`, name. -/
theorem mem_pos_rowsFrom (p : Piece) (lo n : Nat) (x : Tgt × Int) :
    x ∈ pos (p.rowsFrom lo n) ↔ ∃ m < n, x ∈ pos (p.row (lo + m)) := by
  rw [pos_eq_rows, (rowsFrom_spec p lo n).2.2.2.1, List.mem_flatMap]
  constructor
  · rintro ⟨m, hm, hx⟩
    exact ⟨m, List.mem_range.1 hm, by rwa [← pos_row_rowsFrom p lo n m (List.mem_range.1 hm)]⟩
  · rintro ⟨m, hm, hx⟩
    exact ⟨m, List.mem_range.2 hm, by rwa [pos_row_rowsFrom p lo n m hm]⟩

/-- A rowed segment names what its pieces' first `reps` rows name. -/
theorem mem_segPos_rowed (s : Seg) (hr : s.lanes = true) (x : Tgt × Int) :
    x ∈ segPos s ↔ ∃ p ∈ s.ps.toList, x ∈ pos (p.rowsFrom 0 s.reps) := by
  unfold segPos
  simp only [List.mem_flatMap, List.mem_range, copy_rowed s hr, posAll_map, mem_pos_rowsFrom, Nat.zero_add]
  exact ⟨fun ⟨m, hm, p, hp, hx⟩ => ⟨p, hp, m, hm, hx⟩, fun ⟨p, hp, m, hm, hx⟩ => ⟨m, hm, p, hp, hx⟩⟩

/-- A piece's first `n` rows, `n` at most its rows, name what it names. -/
theorem pos_rowsFrom_sub (p : Piece) {n : Nat} (hn : n ≤ p.rows) : ∀ z ∈ pos (p.rowsFrom 0 n), z ∈ pos p := by
  intro z hz
  obtain ⟨m, hm, hz⟩ := (mem_pos_rowsFrom p 0 n z).1 hz
  rw [Nat.zero_add] at hz
  rw [pos_eq_rows p]
  exact List.mem_flatMap.2 ⟨m, List.mem_range.2 (by omega), hz⟩

end InferenceOnlyVerifier.Properties.SecurityProofs.ReadsExact
