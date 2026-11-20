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

/-- Two such loops nested, over a sequence's segments and each one's pieces. -/
theorem forIn_trace2 (F : Piece → Except String (Array Seg)) :
    ∀ (l : List Seg) (acc out : Array Seg), forIn l acc (fun (s : Seg) (r : Array Seg) => do
        let r' ← forIn s.ps r (fun q (r : Array Seg) => do
          let v ← F q
          pure (ForInStep.yield (r ++ v)))
        pure (ForInStep.yield r')) = .ok out →
      ∃ outs : List (Array Seg), List.Forall₂ (fun q v => F q = .ok v) (l.flatMap fun s => s.ps.toList) outs ∧
        out = outs.foldl (· ++ ·) acc
  | [], acc, out, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    exact ⟨[], .nil, h.symm⟩
  | s :: l, acc, out, h => by
    rw [List.forIn_cons] at h
    obtain ⟨y, hy, h⟩ := Walk.except_bind_ok h
    obtain ⟨r', hr', hy⟩ := Walk.except_bind_ok hy
    simp only [pure, Except.pure, Except.ok.injEq] at hy
    subst hy
    rw [← Array.forIn_toList] at hr'
    obtain ⟨o1, h1, e1⟩ := forIn_trace F _ acc r' hr'
    obtain ⟨o2, h2, e2⟩ := forIn_trace2 F l r' out h
    refine ⟨o1 ++ o2, ?_, ?_⟩
    · rw [List.flatMap_cons]; exact forall2_append h1 h2
    · rw [e2, e1, List.foldl_append]

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
theorem reshape_row (y : Piece) (rows len : Nat) (p : Piece) (h : y.reshape rows len = some p)
    (hlen : (pos y).length = rows * len) (m : Nat) (hm : m < rows) :
    pos (p.row m) = sl (pos y) (m * len) ((m + 1) * len) := by
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
      simp only [Bool.and_eq_true, decide_eq_true_eq, List.isEmpty_iff, beq_iff_eq] at hc
      obtain ⟨⟨hr, hlv⟩, hl⟩ := hc
      cases h
      rcases Nat.eq_zero_or_pos len with h0 | h0
      · subst h0
        simp [pos_row, sl]
      have hc : y.rows = 1 := by
        rw [pos_length, Desc.Piece.count, hl] at hlen
        have : 0 < rows * len := Nat.mul_pos (by omega) h0
        rcases Nat.lt_or_ge y.rows 1 with h | h
        · have : y.rows = 0 := by omega
          rw [this, Nat.zero_mul] at hlen; omega
        · omega
      rw [pos_eq_rows y, hc, List.range_one, List.flatMap_singleton, pos_row, hl, sl_map_range _ _ _ _ (by
        rw [Nat.succ_mul]; have : m * len + len ≤ rows * len := by
          have := Nat.mul_le_mul_right len (Nat.succ_le_of_lt hm); rw [Nat.succ_mul] at this; exact this
        omega), pos_row, List.range'_eq_map_range, List.map_map]
      have hL : (m + 1) * len - m * len = len := by rw [Nat.succ_mul]; omega
      rw [hL]
      apply List.map_congr_left
      intro t _
      simp only [Function.comp_apply, Desc.Piece.rowOff, hlv, List.isEmpty_nil, ↓reduceIte, Prod.mk.injEq,
        true_and]
      push_cast
      ring
    · cases h

/-- **What `segDown` gives a segment names what `f` gives each of its positions**: related position by position (`R`)
to `s`'s, where each piece `f` resolves, its positions `s`'s, gives segments so related to its positions, and a rowed
`s` has as many copies as each of its pieces rows. -/
theorem segDown_rel {R : Tgt × Int → Tgt × Int → Prop} (f : Piece → Except String (Array Seg)) (s : Seg)
    (hs : s.rowed = true → ∀ p ∈ s.ps.toList, p.rows = s.reps)
    (hf : ∀ q out, (∀ x ∈ pos q, x ∈ segPos s) → f q = .ok out → List.Forall₂ R (segsPos out) (pos q))
    (out : Array Seg) (h : Desc.segDown f s = .ok out) : List.Forall₂ R (segsPos out) (segPos s) := by
  unfold Desc.segDown at h
  cases hr : s.rowed
  · simp only [hr, Bool.not_false, ↓reduceIte] at h
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
    have hsub : ∀ q ∈ s.ps.toList, ∀ x ∈ pos q, x ∈ segPos s := by
      intro q hq x hx
      rw [pos_eq_rows q, hs hr q hq] at hx
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
      rw [copy_rowed _ hr, copy_rowed s hr, posAll_map, posAll_map]
      refine forall2_flatMap (fun p q hpq => ?_) hpair
      obtain ⟨hq, r, hfr, hone⟩ := hpq
      rcases r with ⟨_ | ⟨x, _ | ⟨x2, l⟩⟩⟩ <;> simp at hone
      obtain ⟨y, hy, hre⟩ := hone
      have hxy := hf q #[x] (hsub q hq) hfr
      rw [segsPos_singleton, single_pos x y hy] at hxy
      have hmq : m < q.rows := by rw [hs hr q hq]; exact List.mem_range.1 hm
      have hlen : (pos y).length = q.rows * q.len := by rw [hxy.length_eq, pos_length]; rfl
      rw [reshape_row y q.rows q.len p hre hlen m hmq, pos_row_sl q m hmq]
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
      simp only [Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one] at hF
      rw [hL]
      exact rel_trace hF fun q hq v hv => hf q v (fun x hx => by rw [hL]; exact List.mem_flatMap.2 ⟨q, hq, hx⟩) hv

end InferenceOnlyVerifier.Properties.SecurityProofs.ReadsExact
