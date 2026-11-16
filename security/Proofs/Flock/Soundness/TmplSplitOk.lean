module

public import Proofs.Flock.Soundness.TmplSplitStep
import Mathlib.Combinatorics.Matroid.Init
import Mathlib.Data.Nat.Totient
import Proofs.Flock.Level3.FoldedBase
import Proofs.Flock.Level3.Nodes
import Proofs.Flock.Soundness.TmplFoldRef
import Std.Data.HashMap.Lemmas

/-!
# What a walked table means

`Spec`: a class's table, placed at any chunk origin `H`, values (`tabExp`, `tabSum`) to its copies' `refVal`s, as
`ExpOk` and `SumOk` say of `foldDef`'s `Out`; its open inputs are keyed by an input of its Definition and one of its
copies (`UOk`). `StOk`: every cached table means its class's, at every depth past its Definition.

- `rows_val`, `binds_val`, `slots_tabExp`: a Definition's own rows, a call's bindings and a run of lanes spread over the
  64 slots, valued.
- `walkChild_ok`: a callee's table, re-keyed or walked directly, means its class's at the caller's chunk origin.
-/

@[expose] public section

open Flock Flock.Tmpl FlockLevel3 GF128 Std

namespace FlockSoundness.TmplFold

theorem isZero_eq {a : F128} (h : a.isZero = true) : a = 0 := by
  simp only [F128.isZero, Bool.and_eq_true, beq_iff_eq] at h
  cases a
  simp_all <;> rfl

/-! ## Open inputs' keys -/

/-- A class's copies: one while summed, else one per low part. -/
def nkOf (β : ℕ) (as : Array ℕ) : ℕ := if 6 ≤ β then 1 else as.size

/-- Every open input entry's input is one of `ins` and its copy one of `nk`. -/
def UOk (ins nk : ℕ) (T : Tab) : Prop := ∀ x, x ∈ T.U → x.1 < ins ∧ x.2.1 < nk

theorem mem_bump3 (m : HashMap (ℕ × ℕ × ℕ) F128) (k x : ℕ × ℕ × ℕ) (v : F128) (h : x ∈ bump m k v) : x = k ∨ x ∈ m := by
  unfold bump at h
  rcases HashMap.mem_insert.1 h with h | h
  · exact Or.inl (beq_iff_eq.1 h).symm
  · exact Or.inr h

theorem ref_U (D : Def) (pt : FoldPt) (summed : Bool) (A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ) {nk : ℕ}
    (o : Tab) {k : ℕ} (hk : k < nk) (cx : ℕ) (w : F128) (v : ℕ) (ho : UOk D.ins nk o) :
    UOk D.ins nk (Tab.ref D pt summed A B as gL pos o k cx w v) := by
  unfold Tab.ref
  split_ifs
  · exact ho
  · exact ho
  · intro x hx
    rcases mem_bump3 _ _ _ _ hx with rfl | hx
    · exact ⟨show v - D.cols < D.ins by omega, hk⟩
    · exact ho x hx
  · exact ho
  · exact ho

theorem foldl_inv {α δ : Type} (Q : δ → Prop) (f : δ → α → δ) :
    ∀ (l : List α) (d : δ), (∀ d a, a ∈ l → Q d → Q (f d a)) → Q d → Q (l.foldl f d)
  | [], _, _, hd => hd
  | a :: l, d, h, hd => foldl_inv Q f l _ (fun d' a' ha' => h d' a' (List.mem_cons_of_mem _ ha'))
      (h d a (List.mem_cons_self ..) hd)

theorem fold_inv {κ δ : Type} [BEq κ] [Hashable κ] (Q : δ → Prop) (step : δ → κ → F128 → δ) (m : HashMap κ F128)
    (init : δ) (h : ∀ d (e : κ × F128), e ∈ m.toList → Q d → Q (step d e.1 e.2)) (h0 : Q init) :
    Q (m.fold step init) := by
  rw [HashMap.fold_eq_foldl_toList]
  exact foldl_inv Q (fun d e => step d e.1 e.2) m.toList init (fun d e he hd => h d e he hd) h0

theorem bind_U (D : Def) (pt : FoldPt) (summed : Bool) (A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ) {nk : ℕ}
    (c : Call) (o : Tab) (j : ℕ) (ch : Tab) (rm : ℕ → ℕ × F128) (wk : ℕ → ℕ → List (ℕ × F128))
    (hwk : ∀ e ∈ ch.U.toList, ∀ kf ∈ wk e.1.2.1 j, kf.1 < nk) (ho : UOk D.ins nk o) :
    UOk D.ins nk (Tab.bind (Tab.ref D pt summed A B as gL pos) c o j ch rm wk) := by
  unfold Tab.bind
  dsimp only
  refine fold_inv (UOk D.ins nk) _ ch.U _ (fun o' e he hq => ?_) ho
  obtain ⟨⟨t', kc, cx⟩, v⟩ := e
  dsimp only
  generalize rm cx = p
  obtain ⟨px, fx⟩ := p
  refine foldl_inv (UOk D.ins nk) _ _ o' (fun o'' kf hkf hq' => ?_) hq
  obtain ⟨k, f⟩ := kf
  exact ref_U D pt summed A B as gL pos o'' (hwk _ he (k, f) hkf) _ _ _ hq'

theorem binds_U (D : Def) (pt : FoldPt) (summed : Bool) (A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ) {nk : ℕ}
    (c : Call) (o : Tab) (count : ℕ) (ch : Tab) (rm : ℕ → ℕ × F128) (wk : ℕ → ℕ → List (ℕ × F128))
    (hwk : ∀ j, ∀ e ∈ ch.U.toList, ∀ kf ∈ wk e.1.2.1 j, kf.1 < nk) (ho : UOk D.ins nk o) :
    UOk D.ins nk (Tab.binds (Tab.bind (Tab.ref D pt summed A B as gL pos) c) o count ch rm wk) :=
  foldl_inv (UOk D.ins nk) _ _ o (fun o' j _ hq => bind_U D pt summed A B as gL pos c o' j ch rm wk (hwk j) hq) ho

theorem flush_U (summed : Bool) (o : Tab) (cx cy : ℕ) (w : F128) (acc : Array F128) {ins nk : ℕ}
    (ho : UOk ins nk o) : UOk ins nk (Tab.flush summed o cx cy w acc) := by
  unfold Tab.flush
  dsimp only
  split_ifs <;> exact ho

theorem refs_U (D : Def) (pt : FoldPt) (summed : Bool) (A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ) {nk : ℕ}
    (o : Tab) {k : ℕ} (hk : k < nk) (cx : ℕ) (w : F128) (vs : Array ℕ) (ho : UOk D.ins nk o) :
    UOk D.ins nk (Tab.refs D pt summed A B as gL pos o k cx w vs) := by
  unfold Tab.refs
  split_ifs
  · rw [← Array.foldl_toList]
    exact foldl_inv (UOk D.ins nk) _ _ _ (fun _ _ _ h => ref_U D pt summed A B as gL pos _ hk _ _ _ h) ho
  · rw [← Array.foldl_toList]
    refine flush_U _ _ _ _ _ _ (foldl_inv (fun st : Tab × ℕ × Array F128 => UOk D.ins nk st.1) _ _ _
      (fun st v _ h => ?_) ho)
    simp only [Tab.refStep]
    split_ifs <;> first
      | exact h
      | exact flush_U _ _ _ _ _ _ h
      | exact ref_U D pt summed A B as gL pos _ hk _ _ _ h

theorem flushK_U (as : Array ℕ) (nk' : ℕ) (o : Tab) (cx cy : ℕ) (ws acc : Array F128) {ins nk : ℕ}
    (ho : UOk ins nk o) : UOk ins nk (Tab.flushK as nk' o cx cy ws acc) := by
  unfold Tab.flushK
  exact ho

theorem flushP_U (o : Tab) (cx cy : ℕ) (pd : Array F128) {ins nk : ℕ} (ho : UOk ins nk o) :
    UOk ins nk (Tab.flushP o cx cy pd) := by
  unfold Tab.flushP
  exact ho

theorem flushKP_U (as : Array ℕ) (nk' : ℕ) (st : Tab × ℕ × ℕ × Array F128) (cx cy : ℕ) (ws acc : Array F128)
    {ins nk : ℕ} (ho : UOk ins nk st.1) : UOk ins nk (Tab.flushKP as nk' st cx cy ws acc).1 := by
  obtain ⟨o, px, py, pd⟩ := st
  simp only [Tab.flushKP]
  split_ifs
  · exact ho
  · exact flushP_U _ _ _ _ ho

theorem refsK_U (D : Def) (pt : FoldPt) (A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ) (fac : ℕ → F128) {nk : ℕ}
    (cx : ℕ) (ws : Array F128) (st : Tab × ℕ × ℕ × Array F128) (vs : Array ℕ) (ho : UOk D.ins nk st.1) :
    UOk D.ins nk (Tab.refsK D pt A B as gL pos fac nk cx ws st vs).1 := by
  obtain ⟨o, px, py, pd⟩ := st
  unfold Tab.refsK
  dsimp only
  rw [← Array.foldl_toList]
  refine flushKP_U _ _ _ _ _ _ _ (foldl_inv (fun st : Tab × ℕ × Array F128 => UOk D.ins nk st.1) _ _ _
    (fun st v _ h => ?_) ho)
  simp only [Tab.refStepK]
  split_ifs <;> first
    | exact h
    | exact flushK_U _ _ _ _ _ _ _ h
    | exact foldl_inv (UOk D.ins nk) _ _ _
        (fun o' k hk h' => ref_U D pt false A B as gL pos o' (List.mem_range.mp hk) _ _ _ h') h

theorem refDK_U (D : Def) (pt : FoldPt) (A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ) (fac : ℕ → F128) {nk : ℕ}
    (cx : ℕ) (ws : Array F128) (st : Tab × ℕ × ℕ × Array F128) (v : ℕ) (ho : UOk D.ins nk st.1) :
    UOk D.ins nk (Tab.refDK D pt A B as gL pos fac nk cx ws st v).1 := by
  obtain ⟨o, px, py, pd⟩ := st
  simp only [Tab.refDK]
  split_ifs
  · exact ho
  · exact flushP_U _ _ _ _ ho
  · exact foldl_inv (UOk D.ins nk) _ _ _
      (fun o' k hk h' => ref_U D pt false A B as gL pos o' (List.mem_range.mp hk) _ _ _ h') ho

theorem halfK_U (D : Def) (pt : FoldPt) (A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ) (fac : ℕ → F128) {nk : ℕ}
    (cx : ℕ) (ws : Array F128) (st : Tab × ℕ × ℕ × Array F128) (vs : Array ℕ) (ho : UOk D.ins nk st.1) :
    UOk D.ins nk (Tab.halfK D pt A B as gL pos fac nk cx ws st vs).1 := by
  unfold Tab.halfK
  split_ifs
  · rw [← Array.foldl_toList]
    exact foldl_inv (fun st : Tab × ℕ × ℕ × Array F128 => UOk D.ins nk st.1) _ _ _
      (fun st' v _ h => refDK_U D pt A B as gL pos fac cx ws st' v h) ho
  · exact refsK_U D pt A B as gL pos fac cx ws st vs ho

theorem rowsK_U (D : Def) (pt : FoldPt) (A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ) (fac : ℕ → F128)
    (rowLow : ℕ → ℕ → F128) (alpha : F128) {nk : ℕ} (rows : Array (ℕ × Array ℕ × Array ℕ)) (o : Tab)
    (ho : UOk D.ins nk o) : UOk D.ins nk (Tab.rowsK D pt A B as gL pos fac rowLow alpha nk rows o) := by
  have hw : ∀ (T : Tab) (n : ℕ), UOk D.ins nk T → UOk D.ins nk { T with work := n } := fun _ _ h => h
  unfold Tab.rowsK
  rw [← Array.foldl_toList]
  refine flushP_U _ _ _ _ (foldl_inv (fun st : Tab × ℕ × ℕ × Array F128 => UOk D.ins nk st.1) _ _ _
    (fun st r _ h => ?_) ho)
  obtain ⟨x, ra, rb⟩ := r
  simp only [Tab.rowK]
  exact hw _ _ (halfK_U D pt A B as gL pos fac _ _ _ _ (halfK_U D pt A B as gL pos fac _ _ _ _ h))

theorem colT_U (D : Def) (pt : FoldPt) (A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ) {nk : ℕ} (hk : 0 < nk)
    (cx : ℕ) (w : F128) (st : Tab × Array F128) (v : ℕ) (ho : UOk D.ins nk st.1) :
    UOk D.ins nk (Tab.colT D pt A B as gL pos cx w st v).1 := by
  obtain ⟨o, W⟩ := st
  simp only [Tab.colT]
  split_ifs
  · exact ho
  · exact ref_U D pt false A B as gL pos o hk _ _ _ ho

theorem rowT_U (D : Def) (pt : FoldPt) (A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ) {nk : ℕ} (hk : 0 < nk)
    (rowLow : ℕ → ℕ → F128) (alpha : F128) (cx : ℕ) (st : Tab × Array F128) (r : ℕ × Array ℕ × Array ℕ)
    (ho : UOk D.ins nk st.1) : UOk D.ins nk (Tab.rowT D pt A B as gL pos rowLow alpha cx st r).1 := by
  have hw : ∀ (T : Tab) (n : ℕ), UOk D.ins nk T → UOk D.ins nk { T with work := n } := fun _ _ h => h
  obtain ⟨x, ra, rb⟩ := r
  simp only [Tab.rowT, ← Array.foldl_toList]
  refine hw _ _ (foldl_inv (fun st : Tab × Array F128 => UOk D.ins nk st.1) _ _ _
    (fun st' v _ h => colT_U D pt A B as gL pos hk cx _ st' v h)
    (foldl_inv (fun st : Tab × Array F128 => UOk D.ins nk st.1) _ _ _
      (fun st' v _ h => colT_U D pt A B as gL pos hk cx _ st' v h) ho))

theorem colD_U (A B : ℕ) (as : Array ℕ) (pos : ℕ → ℕ) (fac : ℕ → F128) (cx : ℕ) (x : F128)
    (st : Tab × ℕ × ℕ × Array F128) (v : ℕ) {ins nk : ℕ} (ho : UOk ins nk st.1) :
    UOk ins nk (Tab.colD A B as pos fac cx x st v).1 := by
  obtain ⟨o, px, py, pd⟩ := st
  simp only [Tab.colD]
  split_ifs
  · exact ho
  · exact flushP_U _ _ _ _ ho

theorem colsP_U (D : Def) (pt : FoldPt) (A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ) (fac : ℕ → F128)
    (cx : ℕ) (W : Array F128) : ∀ (n v : ℕ) (st : Tab × ℕ × ℕ × Array F128), UOk D.ins 1 st.1 →
      UOk D.ins 1 (Tab.colsP D pt A B as gL pos fac cx W n v st).1
  | 0, _, _, ho => ho
  | n + 1, v, st, ho => by
    simp only [Tab.colsP]
    refine colsP_U D pt A B as gL pos fac cx W n (v + 1) _ ?_
    split_ifs
    · exact ho
    · exact colD_U A B as pos fac cx _ st v ho

theorem rowsT_U (D : Def) (pt : FoldPt) (A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ) (fac : ℕ → F128)
    (rowLow : ℕ → ℕ → F128) (alpha : F128) (cx : ℕ) (rows : Array (ℕ × Array ℕ × Array ℕ)) (o : Tab)
    (ho : UOk D.ins 1 o) : UOk D.ins 1 (Tab.rowsT D pt A B as gL pos fac rowLow alpha cx rows o) := by
  unfold Tab.rowsT
  rw [← Array.foldl_toList]
  exact flushP_U _ _ _ _ (colsP_U D pt A B as gL pos fac cx _ _ _ _
    (foldl_inv (fun st : Tab × Array F128 => UOk D.ins 1 st.1) _ _ _
      (fun st r _ h => rowT_U D pt A B as gL pos Nat.one_pos rowLow alpha cx st r h) ho))

theorem rowsC_U (D : Def) (pt : FoldPt) (A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ) (fac : ℕ → F128)
    (rowLow : ℕ → ℕ → F128) (alpha : F128) {nk : ℕ} (rows : Array (ℕ × Array ℕ × Array ℕ)) (o : Tab)
    (ho : UOk D.ins nk o) : UOk D.ins nk (Tab.rowsC D pt A B as gL pos fac rowLow alpha nk rows o) := by
  unfold Tab.rowsC
  dsimp only
  split_ifs with hc
  · simp only [Bool.and_eq_true, beq_iff_eq] at hc
    obtain ⟨⟨rfl, -⟩, -⟩ := hc
    exact rowsT_U D pt A B as gL pos fac rowLow alpha _ rows o ho
  · exact rowsK_U D pt A B as gL pos fac rowLow alpha rows o ho

theorem rows_U (D : Def) (pt : FoldPt) (summed : Bool) (A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ)
    (fac : ℕ → F128) (rowLow : ℕ → ℕ → F128) (alpha : F128) (nk : ℕ) (rows : Array (ℕ × Array ℕ × Array ℕ)) (o : Tab)
    (ho : UOk D.ins nk o) :
    UOk D.ins nk (Tab.rows D pt summed A B as gL pos fac rowLow alpha nk rows o) := by
  have hw : ∀ (T : Tab) (n : ℕ), UOk D.ins nk T → UOk D.ins nk { T with work := n } := fun _ _ h => h
  unfold Tab.rows
  split_ifs
  · exact rowsC_U D pt A B as gL pos fac rowLow alpha rows o ho
  · rw [← Array.foldl_toList]
    refine foldl_inv (UOk D.ins nk) _ _ o (fun o' r _ hq => ?_) ho
    obtain ⟨x, ra, rb⟩ := r
    dsimp only
    refine hw _ _ (foldl_inv (UOk D.ins nk) _ _ o' (fun o'' k hk hq' => ?_) hq)
    have hk := List.mem_range.mp hk
    exact refs_U D pt summed A B as gL pos _ hk _ _ _ (refs_U D pt summed A B as gL pos _ hk _ _ _ hq')

/-! ## Values of the walk's pure steps -/

theorem foldl_val {α δ : Type} (V : δ → GF128) (f : δ → α → δ) (G : α → GF128) (h : ∀ o a, V (f o a) = V o + G a) :
    ∀ (l : List α) (o : δ), V (l.foldl f o) = V o + (l.map G).sum
  | [], o => by simp
  | a :: l, o => by rw [List.foldl_cons, foldl_val V f G h l, h, List.map_cons, List.sum_cons, add_assoc]

/-- `foldl_val` under an invariant every step keeps. -/
theorem foldl_val_inv {α δ : Type} (I : δ → Prop) (V : δ → GF128) (f : δ → α → δ) (G : α → GF128)
    (h : ∀ d a, I d → I (f d a) ∧ V (f d a) = V d + G a) :
    ∀ (l : List α) (d : δ), I d → I (l.foldl f d) ∧ V (l.foldl f d) = V d + (l.map G).sum
  | [], d, hd => by simp [hd]
  | a :: l, d, hd => by
    obtain ⟨h1, h2⟩ := h d a hd
    obtain ⟨h3, h4⟩ := foldl_val_inv I V f G h l _ h1
    refine ⟨h3, ?_⟩
    rw [List.foldl_cons, h4, h2, List.map_cons, List.sum_cons, add_assoc]

/-! ## A row's references, summed in runs

`Tab.refs` sums an own column's factors along a run of references in one column chunk and adds the run's sums at its
end (`Tab.flush`). A run's *pending* value (`pendX`, `pendS`) is what its flush will add; each step keeps the table's
value plus the pending one equal to the per-reference sum `Tab.ref` would have reached. -/

/-- **A run's sums, added** (explicit): slot `s`'s sum at the row's and the run's chunks. -/
theorem flush_tabExp (pt : FoldPt) {s : ℕ} (hs : s < 64) (L H : ℕ) (iv : ℕ → ℕ → GF128) (o : Tab) (cx cy : ℕ)
    (w : F128) (acc : Array F128) :
    tabExp pt s L H iv (Tab.flush false o cx cy w acc) =
      tabExp pt s L H iv o + of w * of (acc.getD s 0) * (hiE pt L H cx * hiG pt L H cy) := by
  unfold Tab.flush tabExp
  rw [ite_eq_right Bool.false_ne_true]
  dsimp only
  rw [foldl_val (fun (m : HashMap (ℕ × ℕ × ℕ) F128) =>
      hsum m fun e => if e.2.2 = s then hiE pt L H e.1 * hiG pt L H e.2.1 else 0) _
      (fun s' => if s' = s then of w * of (acc.getD s' 0) * (hiE pt L H cx * hiG pt L H cy) else 0)
      (fun m s' => ?_), list_range_sum, Finset.sum_ite_eq' (Finset.range 64) s,
    ite_eq_left (Finset.mem_range.2 hs)]
  · ring
  · by_cases h1 : (acc.getD s' 0).isZero = true
    · rw [ite_eq_left h1, isZero_eq h1, of_zero']
      split_ifs <;> ring
    · rw [ite_eq_right h1, hsum_bump, of_mul]
      dsimp only
      split_ifs <;> ring

/-- **A run's sum, added** (summed): at the row's and the run's chunks. -/
theorem flush_tabSum (pt : FoldPt) (L H : ℕ) (gl gc : GF128) (iv : ℕ → GF128) (o : Tab) (cx cy : ℕ) (w : F128)
    (acc : Array F128) :
    tabSum pt L H gl gc iv (Tab.flush true o cx cy w acc) =
      tabSum pt L H gl gc iv o + gl * (of w * of (acc.getD 0 0) * (hiE pt L H cx * hiG pt L H cy)) := by
  unfold Tab.flush
  rw [ite_eq_left rfl]
  dsimp only
  by_cases h : (acc.getD 0 0).isZero = true
  · rw [ite_eq_left h, isZero_eq h, of_zero']
    ring
  · rw [ite_eq_right h]
    unfold tabSum
    dsimp only
    rw [hsum_bump, of_mul]
    dsimp only
    ring

/-- A run's pending value at slot `s` (explicit). -/
noncomputable def pendX (pt : FoldPt) (s L H cx : ℕ) (w : F128) (st : Tab × ℕ × Array F128) : GF128 :=
  of w * of (st.2.2.getD s 0) * (hiE pt L H cx * hiG pt L H st.2.1)

/-- A run's pending value (summed). -/
noncomputable def pendS (pt : FoldPt) (L H cx : ℕ) (gl : GF128) (w : F128) (st : Tab × ℕ × Array F128) : GF128 :=
  gl * (of w * of (st.2.2.getD 0 0) * (hiE pt L H cx * hiG pt L H st.2.1))

theorem refStep_size (D : Def) (pt : FoldPt) (summed : Bool) (A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ)
    (k cx : ℕ) (w : F128) (st : Tab × ℕ × Array F128) (v : ℕ) (hst : st.2.2.size = 64) :
    (Tab.refStep D pt summed A B as gL pos k cx w st v).2.2.size = 64 := by
  simp only [Tab.refStep]
  split_ifs <;> simp [hst]

/-- **One reference, summed in its run** (explicit): `Tab.ref`'s term. -/
theorem refStep_exp (D : Def) (pt : FoldPt) {s : ℕ} (hs : s < 64) (L H A B : ℕ) (as : Array ℕ) (gL : Half)
    (pos : ℕ → ℕ) (iv : ℕ → ℕ → GF128) (k cx : ℕ) (w : F128) (st : Tab × ℕ × Array F128) (hst : st.2.2.size = 64)
    (v : ℕ) :
    tabExp pt s L H iv (Tab.refStep D pt false A B as gL pos k cx w st v).1 +
        pendX pt s L H cx w (Tab.refStep D pt false A B as gL pos k cx w st v) =
      tabExp pt s L H iv st.1 + pendX pt s L H cx w st + of w * (hiE pt L H cx * refX D pt s L H A B as gL pos iv k v) := by
  obtain ⟨o, cy, acc⟩ := st
  dsimp only at hst
  by_cases hv : v < D.cols
  · have hi : (as.getD k 0 + B * (pos v % A)) % 64 < acc.size := by
      rw [hst]; exact Nat.mod_lt _ (by norm_num)
    have hi' : (as.getD k 0 + B * (pos v % A)) % 64 < (Array.replicate 64 (0 : F128)).size := by
      rw [Array.size_replicate]; exact Nat.mod_lt _ (by norm_num)
    simp only [Tab.refStep, pendX, refX, hv, ↓reduceIte, Bool.false_eq_true]
    by_cases hc : pos v / A = cy
    · simp only [hc, beq_self_eq_true, ↓reduceIte]
      rw [getD_modify _ _ hi]
      split_ifs
      · rw [of_add]; ring
      · ring
    · simp only [beq_eq_false_iff_ne.2 hc, Bool.false_eq_true, ↓reduceIte]
      rw [flush_tabExp pt hs, getD_modify _ _ hi', getD_replicate]
      split_ifs
      · rw [of_add, of_zero']; ring
      · rw [of_zero']; ring
  · simp only [Tab.refStep, pendX, hv, ↓reduceIte]
    rw [ref_tabExp]
    ring

/-- **One reference, summed in its run** (summed): `Tab.ref`'s term. -/
theorem refStep_sum (D : Def) (pt : FoldPt) (L H A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ) (gl gc : GF128)
    (iv : ℕ → GF128) (k cx : ℕ) (w : F128) (st : Tab × ℕ × Array F128) (hst : st.2.2.size = 64) (v : ℕ) :
    tabSum pt L H gl gc iv (Tab.refStep D pt true A B as gL pos k cx w st v).1 +
        pendS pt L H cx gl w (Tab.refStep D pt true A B as gL pos k cx w st v) =
      tabSum pt L H gl gc iv st.1 + pendS pt L H cx gl w st + of w * (hiE pt L H cx * refS D pt L H A gL pos gl gc iv v) := by
  obtain ⟨o, cy, acc⟩ := st
  dsimp only at hst
  by_cases hv : v < D.cols
  · have hi : 0 < acc.size := by omega
    have hi0 : 0 < (Array.replicate 64 (0 : F128)).size := by simp
    simp only [Tab.refStep, pendS, refS, hv, ↓reduceIte]
    by_cases hc : pos v / A = cy
    · simp only [hc, beq_self_eq_true, ↓reduceIte]
      rw [getD_modify _ _ hi, ite_eq_left rfl, of_add]
      ring
    · simp only [beq_eq_false_iff_ne.2 hc, Bool.false_eq_true, ↓reduceIte]
      rw [flush_tabSum, getD_modify _ _ hi0, ite_eq_left rfl, getD_replicate, of_add, of_zero']
      ring
  · simp only [Tab.refStep, pendS, hv, ↓reduceIte]
    rw [ref_tabSum]
    ring

/-- **A row's references** (explicit): each adds `Tab.ref`'s term, summed in runs or not. -/
theorem refs_tabExp (D : Def) (pt : FoldPt) {s : ℕ} (hs : s < 64) (L H A B : ℕ) (as : Array ℕ) (gL : Half)
    (pos : ℕ → ℕ) (iv : ℕ → ℕ → GF128) (o : Tab) (k cx : ℕ) (w : F128) (vs : Array ℕ) :
    tabExp pt s L H iv (Tab.refs D pt false A B as gL pos o k cx w vs) = tabExp pt s L H iv o +
      (vs.toList.map fun v => of w * (hiE pt L H cx * refX D pt s L H A B as gL pos iv k v)).sum := by
  unfold Tab.refs
  split_ifs
  · rw [← Array.foldl_toList]
    exact foldl_val (tabExp pt s L H iv) _ _ (fun o v => by rw [ref_tabExp, mul_assoc]) _ _
  · rw [← Array.foldl_toList]
    have key : ∀ (l : List ℕ) (st : Tab × ℕ × Array F128), st.2.2.size = 64 →
        tabExp pt s L H iv (l.foldl (Tab.refStep D pt false A B as gL pos k cx w) st).1 +
            pendX pt s L H cx w (l.foldl (Tab.refStep D pt false A B as gL pos k cx w) st) =
          tabExp pt s L H iv st.1 + pendX pt s L H cx w st +
            (l.map fun v => of w * (hiE pt L H cx * refX D pt s L H A B as gL pos iv k v)).sum := by
      intro l
      induction l with
      | nil => intro st _; simp
      | cons v l ih =>
        intro st hst
        rw [List.foldl_cons, ih _ (refStep_size D pt false A B as gL pos k cx w st v hst),
          refStep_exp D pt hs L H A B as gL pos iv k cx w st hst v, List.map_cons, List.sum_cons]
        ring
    have h := key vs.toList (o, 0, Array.replicate 64 0) (by simp)
    unfold pendX at h
    dsimp only
    rw [flush_tabExp pt hs, h]
    simp only [getD_replicate, of_zero', mul_zero, zero_mul, add_zero]

/-- **A row's references** (summed): each adds `Tab.ref`'s term, summed in runs or not. -/
theorem refs_tabSum (D : Def) (pt : FoldPt) (L H A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ) (gl gc : GF128)
    (iv : ℕ → GF128) (o : Tab) (k cx : ℕ) (w : F128) (vs : Array ℕ) :
    tabSum pt L H gl gc iv (Tab.refs D pt true A B as gL pos o k cx w vs) = tabSum pt L H gl gc iv o +
      (vs.toList.map fun v => of w * (hiE pt L H cx * refS D pt L H A gL pos gl gc iv v)).sum := by
  unfold Tab.refs
  split_ifs
  · rw [← Array.foldl_toList]
    exact foldl_val (tabSum pt L H gl gc iv) _ _ (fun o v => by rw [ref_tabSum, mul_assoc]) _ _
  · rw [← Array.foldl_toList]
    have key : ∀ (l : List ℕ) (st : Tab × ℕ × Array F128), st.2.2.size = 64 →
        tabSum pt L H gl gc iv (l.foldl (Tab.refStep D pt true A B as gL pos k cx w) st).1 +
            pendS pt L H cx gl w (l.foldl (Tab.refStep D pt true A B as gL pos k cx w) st) =
          tabSum pt L H gl gc iv st.1 + pendS pt L H cx gl w st +
            (l.map fun v => of w * (hiE pt L H cx * refS D pt L H A gL pos gl gc iv v)).sum := by
      intro l
      induction l with
      | nil => intro st _; simp
      | cons v l ih =>
        intro st hst
        rw [List.foldl_cons, ih _ (refStep_size D pt true A B as gL pos k cx w st v hst),
          refStep_sum D pt L H A B as gL pos gl gc iv k cx w st hst v, List.map_cons, List.sum_cons]
        ring
    have h := key vs.toList (o, 0, Array.replicate 64 0) (by simp)
    unfold pendS at h
    dsimp only
    rw [flush_tabSum, h]
    simp only [getD_replicate, of_zero', mul_zero, zero_mul, add_zero]

/-! ## A row's references at every copy at once -/

/-- A low part below `B`, with `B` dividing 64, stays in the slot: `a + B·u` has slot `a + (B·u mod 64)` and the
64-block of `B·u`. -/
theorem low_split {B a : ℕ} (hB : 64 % B = 0) (ha : a < B) (u : ℕ) :
    (a + B * u) % 64 = a + B * u % 64 ∧ (a + B * u) / 64 = B * u / 64 := by
  have hd : B ∣ 64 := Nat.dvd_of_mod_eq_zero hB
  obtain ⟨c, hc⟩ : B ∣ B * u % 64 := (Nat.dvd_mod_iff hd).2 (Nat.dvd_mul_right B u)
  obtain ⟨d, hd'⟩ := hd
  have hlt : B * c < B * d := by
    rw [← hc, ← hd']
    exact Nat.mod_lt _ (by norm_num)
  have hcd : c + 1 ≤ d := Nat.lt_of_mul_lt_mul_left hlt
  have h1 : B * c + B ≤ 64 := by
    rw [hd']
    calc B * c + B = B * (c + 1) := by ring
      _ ≤ B * d := Nat.mul_le_mul_left B hcd
  have key : ∀ w y : ℕ, w % 64 = y → y + B ≤ 64 → (a + w) % 64 = a + w % 64 ∧ (a + w) / 64 = w / 64 := by
    intro w y h2 h3
    constructor <;> omega
  exact key _ _ hc h1

/-- Copy `k`'s factor at slot `s` for a run's sum at `t` (explicit): `ws[k]` and the chunks' high factors where
`as[k] + t` is `s`. -/
noncomputable def slotK (pt : FoldPt) (as : Array ℕ) (s L H cx cy : ℕ) (ws : Array F128) (k t : ℕ) : GF128 :=
  if as.getD k 0 + t = s then of (ws.getD k 0) * (hiE pt L H cx * hiG pt L H cy) else 0

/-- A run's pending value at every copy (explicit, slot `s`): its sums at column chunk `cy`. -/
noncomputable def pendK (pt : FoldPt) (as : Array ℕ) (nk s L H cx : ℕ) (ws : Array F128) (cy : ℕ)
    (acc : Array F128) : GF128 :=
  ∑ k ∈ Finset.range nk, ∑ t ∈ Finset.range 64, of (acc.getD t 0) * slotK pt as s L H cx cy ws k t

theorem sum_modify_add (acc : Array F128) {i n : ℕ} (hi : i < acc.size) (hin : i < n) (g : F128) (c : ℕ → GF128) :
    ∑ t ∈ Finset.range n, of ((acc.modify i (· + g)).getD t 0) * c t =
      ∑ t ∈ Finset.range n, of (acc.getD t 0) * c t + of g * c i := by
  have h : ∀ t, of ((acc.modify i (· + g)).getD t 0) * c t =
      of (acc.getD t 0) * c t + if i = t then of g * c t else 0 := by
    intro t
    rw [getD_modify _ _ hi]
    split_ifs <;> simp only [of_add, add_mul, add_zero]
  rw [Finset.sum_congr rfl (fun t _ => h t), Finset.sum_add_distrib, Finset.sum_ite_eq (Finset.range n) i,
    ite_eq_left (Finset.mem_range.2 hin)]

theorem sum_modify_add' (acc : Array F128) {i n : ℕ} (hi : i < acc.size) (hin : i < n) (g : F128) (c : ℕ → GF128) :
    ∑ t ∈ Finset.range n, of ((acc.modify i (g + ·)).getD t 0) * c t =
      ∑ t ∈ Finset.range n, of (acc.getD t 0) * c t + of g * c i := by
  have h : ∀ t, of ((acc.modify i (g + ·)).getD t 0) * c t =
      of (acc.getD t 0) * c t + if i = t then of g * c t else 0 := by
    intro t
    rw [getD_modify _ _ hi]
    split_ifs
    · simp only [of_add]
      ring
    · simp only [add_zero]
  rw [Finset.sum_congr rfl (fun t _ => h t), Finset.sum_add_distrib, Finset.sum_ite_eq (Finset.range n) i,
    ite_eq_left (Finset.mem_range.2 hin)]

/-- **A run's sums, added at every copy** (explicit). -/
theorem flushK_tabExp (pt : FoldPt) (s L H : ℕ) (iv : ℕ → ℕ → GF128) (as : Array ℕ) (nk : ℕ) (o : Tab) (cx cy : ℕ)
    (ws acc : Array F128) :
    tabExp pt s L H iv (Tab.flushK as nk o cx cy ws acc) =
      tabExp pt s L H iv o + pendK pt as nk s L H cx ws cy acc := by
  unfold Tab.flushK tabExp pendK
  dsimp only
  rw [foldl_val (fun (m : HashMap (ℕ × ℕ × ℕ) F128) =>
      hsum m fun e => if e.2.2 = s then hiE pt L H e.1 * hiG pt L H e.2.1 else 0) _
      (fun t => ∑ k ∈ Finset.range nk, of (acc.getD t 0) * slotK pt as s L H cx cy ws k t)
      (fun m t => ?_), list_range_sum, Finset.sum_comm (s := Finset.range 64)]
  · ring
  · by_cases h1 : (acc.getD t 0).isZero = true
    · rw [ite_eq_left h1, isZero_eq h1, of_zero']
      simp
    · rw [ite_eq_right h1, foldl_val (fun (m : HashMap (ℕ × ℕ × ℕ) F128) =>
          hsum m fun e => if e.2.2 = s then hiE pt L H e.1 * hiG pt L H e.2.1 else 0) _
          (fun k => of (acc.getD t 0) * slotK pt as s L H cx cy ws k t) (fun m k => ?_), list_range_sum]
      rw [hsum_bump, of_mul]
      unfold slotK
      dsimp only
      split_ifs <;> ring

theorem refStepK_size (D : Def) (pt : FoldPt) (A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ) (fac : ℕ → F128)
    (nk cx : ℕ) (ws : Array F128) (st : Tab × ℕ × Array F128) (v : ℕ) (hst : st.2.2.size = 64) :
    (Tab.refStepK D pt A B as gL pos fac nk cx ws st v).2.2.size = 64 := by
  simp only [Tab.refStepK]
  split_ifs <;> simp [hst]

/-- **One reference at every copy, summed in its run**: `Tab.ref`'s term at each copy. -/
theorem refStepK_exp (D : Def) (pt : FoldPt) {s : ℕ} (L H A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ)
    (fac : ℕ → F128) (iv : ℕ → ℕ → GF128) {nk : ℕ} (hB : 64 % B = 0) (has : ∀ k < nk, as.getD k 0 < B)
    (hfac : ∀ v < D.cols, fac v = gL.at (B * (pos v % A) / 64)) (cx : ℕ) (ws : Array F128)
    (st : Tab × ℕ × Array F128) (hst : st.2.2.size = 64) (v : ℕ) :
    tabExp pt s L H iv (Tab.refStepK D pt A B as gL pos fac nk cx ws st v).1 +
        pendK pt as nk s L H cx ws (Tab.refStepK D pt A B as gL pos fac nk cx ws st v).2.1
          (Tab.refStepK D pt A B as gL pos fac nk cx ws st v).2.2 =
      tabExp pt s L H iv st.1 + pendK pt as nk s L H cx ws st.2.1 st.2.2 +
        ∑ k ∈ Finset.range nk, of (ws.getD k 0) * (hiE pt L H cx * refX D pt s L H A B as gL pos iv k v) := by
  obtain ⟨o, cy, acc⟩ := st
  dsimp only at hst ⊢
  by_cases hv : v < D.cols
  · have hi64 : B * (pos v % A) % 64 < 64 := Nat.mod_lt _ (by norm_num)
    have hrx : ∀ k ∈ Finset.range nk, of (ws.getD k 0) * (hiE pt L H cx * refX D pt s L H A B as gL pos iv k v) =
        of (gL.at (B * (pos v % A) / 64)) * slotK pt as s L H cx (pos v / A) ws k (B * (pos v % A) % 64) := by
      intro k hk
      obtain ⟨h1, h2⟩ := low_split hB (has k (Finset.mem_range.1 hk)) (pos v % A)
      simp only [refX, slotK, hv, ↓reduceIte, h1, h2]
      split_ifs <;> ring
    rw [Finset.sum_congr rfl hrx]
    simp only [Tab.refStepK, hv, ↓reduceIte, hfac v hv]
    by_cases hc : pos v / A = cy
    · simp only [hc, beq_self_eq_true, ↓reduceIte]
      unfold pendK
      rw [add_assoc, ← Finset.sum_add_distrib]
      congr 1
      exact Finset.sum_congr rfl fun k _ => sum_modify_add acc (by rw [hst]; exact hi64) hi64 _ _
    · simp only [beq_eq_false_iff_ne.2 hc, Bool.false_eq_true, ↓reduceIte]
      rw [flushK_tabExp, add_assoc, add_assoc]
      congr 2
      unfold pendK
      refine Finset.sum_congr rfl fun k _ => ?_
      rw [sum_modify_add _ (by simpa using hi64) hi64]
      simp only [getD_replicate, of_zero', zero_mul, Finset.sum_const_zero, zero_add]
  · simp only [Tab.refStepK, hv, ↓reduceIte]
    have h := foldl_val (tabExp pt s L H iv) (fun o k => Tab.ref D pt false A B as gL pos o k cx (ws.getD k 0) v)
      (fun k => of (ws.getD k 0) * (hiE pt L H cx * refX D pt s L H A B as gL pos iv k v))
      (fun o k => by rw [ref_tabExp, mul_assoc]) (List.range nk) o
    rw [h, list_range_sum]
    ring

/-- **A row's references at every copy, summed in runs** (explicit, each copy's low part below `B`, `B` dividing 64):
the table's value plus the runs' pending value gains `Tab.ref`'s term at each copy for each reference. -/
theorem refStepK_fold (D : Def) (pt : FoldPt) {s : ℕ} (L H A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ)
    (fac : ℕ → F128) (iv : ℕ → ℕ → GF128) {nk : ℕ} (hB : 64 % B = 0) (has : ∀ k < nk, as.getD k 0 < B)
    (hfac : ∀ v < D.cols, fac v = gL.at (B * (pos v % A) / 64)) (cx : ℕ) (ws : Array F128) :
    ∀ (l : List ℕ) (st : Tab × ℕ × Array F128), st.2.2.size = 64 →
      (l.foldl (Tab.refStepK D pt A B as gL pos fac nk cx ws) st).2.2.size = 64 ∧
        tabExp pt s L H iv (l.foldl (Tab.refStepK D pt A B as gL pos fac nk cx ws) st).1 +
            pendK pt as nk s L H cx ws (l.foldl (Tab.refStepK D pt A B as gL pos fac nk cx ws) st).2.1
              (l.foldl (Tab.refStepK D pt A B as gL pos fac nk cx ws) st).2.2 =
          tabExp pt s L H iv st.1 + pendK pt as nk s L H cx ws st.2.1 st.2.2 +
            (l.map fun v => ∑ k ∈ Finset.range nk,
              of (ws.getD k 0) * (hiE pt L H cx * refX D pt s L H A B as gL pos iv k v)).sum := by
  intro l
  induction l with
  | nil => intro st hst; simp [hst]
  | cons v l ih =>
    intro st hst
    obtain ⟨h1, h2⟩ := ih _ (refStepK_size D pt A B as gL pos fac nk cx ws st v hst)
    refine ⟨h1, ?_⟩
    rw [List.foldl_cons, h2, refStepK_exp D pt L H A B as gL pos fac iv hB has hfac cx ws st hst v, List.map_cons,
      List.sum_cons]
    ring

theorem getD_map_range {α : Type} (f : ℕ → α) (n : ℕ) {k : ℕ} (hk : k < n) (d : α) :
    ((Array.range n).map f).getD k d = f k := by
  simp [Array.getD, hk]

/-- A function read from a table of its own values is itself. -/
theorem tabled_eq {α : Type} [Inhabited α] (n : ℕ) (f : ℕ → α) : tabled ((Array.range n).map f) f = f := by
  funext y
  by_cases h : y < n
  · have h' : y < ((Array.range n).map f).size := by simpa using h
    simp only [tabled, h', ↓reduceIte]
    exact getD_map_range f n h default
  · have h' : ¬ y < ((Array.range n).map f).size := by simpa using h
    simp only [tabled, h', ↓reduceIte]

theorem getD_map_map_range {α β : Type} (f : ℕ → α) (g : α → β) (n : ℕ) {k : ℕ} (hk : k < n) (d : β) :
    (((Array.range n).map f).map g).getD k d = g (f k) := by
  simp [Array.getD, hk]

/-! ## Rows on pending sums

`Tab.rowsK` carries sums of one row chunk and column chunk from row to row, as `Tab.refsK` carries a run's within a row
before adding the last to them; their pending value (`pendP`) is what their flush will add, and each step keeps the
table's value plus the pending one equal to the per-reference sum. -/

/-- The pending sums' value at slot `s` (explicit): `pd[s]` at their chunks. -/
noncomputable def pendP (pt : FoldPt) (s L H : ℕ) (st : Tab × ℕ × ℕ × Array F128) : GF128 :=
  of (st.2.2.2.getD s 0) * (hiE pt L H st.2.1 * hiG pt L H st.2.2.1)

/-- **Pending sums, added** (explicit): slot `s`'s sum at their chunks. -/
theorem flushP_tabExp (pt : FoldPt) {s : ℕ} (hs : s < 64) (L H : ℕ) (iv : ℕ → ℕ → GF128) (o : Tab) (cx cy : ℕ)
    (pd : Array F128) :
    tabExp pt s L H iv (Tab.flushP o cx cy pd) =
      tabExp pt s L H iv o + of (pd.getD s 0) * (hiE pt L H cx * hiG pt L H cy) := by
  unfold Tab.flushP tabExp
  dsimp only
  rw [foldl_val (fun (m : HashMap (ℕ × ℕ × ℕ) F128) =>
      hsum m fun e => if e.2.2 = s then hiE pt L H e.1 * hiG pt L H e.2.1 else 0) _
      (fun s' => if s' = s then of (pd.getD s' 0) * (hiE pt L H cx * hiG pt L H cy) else 0)
      (fun m s' => ?_), list_range_sum, Finset.sum_ite_eq' (Finset.range 64) s,
    ite_eq_left (Finset.mem_range.2 hs)]
  · ring
  · by_cases h1 : (pd.getD s' 0).isZero = true
    · rw [ite_eq_left h1, isZero_eq h1, of_zero']
      split_ifs <;> ring
    · rw [ite_eq_right h1, hsum_bump]
      dsimp only
      split_ifs <;> ring

theorem flushP_pend (pt : FoldPt) {s : ℕ} (hs : s < 64) (L H : ℕ) (iv : ℕ → ℕ → GF128)
    (st : Tab × ℕ × ℕ × Array F128) :
    tabExp pt s L H iv (Tab.flushP st.1 st.2.1 st.2.2.1 st.2.2.2) = tabExp pt s L H iv st.1 + pendP pt s L H st :=
  flushP_tabExp pt hs L H iv _ _ _ _

/-- **`addK`'s sums**: slot `s` gains `ws[j] · f` for each of the `n` copies `j` from `k` on whose slot `as[j] + t` is
`s` (each times `c`). -/
theorem addK_spec (as : Array ℕ) (ws : Array F128) (t : ℕ) (f : F128) {s : ℕ} (hs : s < 64) (c : GF128) :
    ∀ (n k : ℕ) (pd : Array F128), pd.size = 64 →
      (addK as ws t f n k pd).size = 64 ∧
        of ((addK as ws t f n k pd).getD s 0) * c = of (pd.getD s 0) * c +
          ∑ j ∈ Finset.range n, if as.getD (k + j) 0 + t = s then of (ws.getD (k + j) 0) * of f * c else 0
  | 0, k, pd, hpd => by simp [addK, hpd]
  | n + 1, k, pd, hpd => by
    obtain ⟨h1, h2⟩ := addK_spec as ws t f hs c n (k + 1) (pd.modify (as.getD k 0 + t) (· + ws.getD k 0 * f))
      (by rw [Array.size_modify, hpd])
    simp only [addK]
    refine ⟨h1, ?_⟩
    have hm : (pd.modify (as.getD k 0 + t) (· + ws.getD k 0 * f)).getD s 0 =
        if as.getD k 0 + t = s then pd.getD s 0 + ws.getD k 0 * f else pd.getD s 0 := by
      by_cases he : as.getD k 0 + t = s
      · rw [getD_modify _ _ (by rw [hpd]; omega)]
      · rw [ite_eq_right he]
        generalize as.getD k 0 + t = i at he ⊢
        simp only [Array.getD_eq_getD_getElem?, Array.getElem?_modify, ite_eq_right he]
    rw [h2, hm, Finset.sum_range_succ']
    simp only [Nat.add_zero, show ∀ j, k + 1 + j = k + (j + 1) from fun j => by omega]
    split_ifs <;> (try simp only [of_add, of_mul]) <;> ring

/-- **A run's sums, on pending sums** (explicit): slot `s` gains their value at every copy, at the run's chunks. -/
theorem addRun_spec (pt : FoldPt) {s : ℕ} (hs : s < 64) (L H : ℕ) (as : Array ℕ) (nk : ℕ) (ws acc : Array F128)
    (cx cy : ℕ) (pd : Array F128) (hpd : pd.size = 64) :
    (addRun as nk ws acc pd).size = 64 ∧
      of ((addRun as nk ws acc pd).getD s 0) * (hiE pt L H cx * hiG pt L H cy) =
        of (pd.getD s 0) * (hiE pt L H cx * hiG pt L H cy) + pendK pt as nk s L H cx ws cy acc := by
  have key : ∀ (l : List ℕ) (pd : Array F128), pd.size = 64 →
      (l.foldl (addRunStep as nk ws acc) pd).size = 64 ∧
        of ((l.foldl (addRunStep as nk ws acc) pd).getD s 0) * (hiE pt L H cx * hiG pt L H cy) =
          of (pd.getD s 0) * (hiE pt L H cx * hiG pt L H cy) +
            (l.map fun t => ∑ k ∈ Finset.range nk, of (acc.getD t 0) * slotK pt as s L H cx cy ws k t).sum := by
    intro l
    induction l with
    | nil => intro pd hpd; simp [hpd]
    | cons t l ih =>
      intro pd hpd
      rw [List.foldl_cons, List.map_cons, List.sum_cons]
      by_cases h1 : (acc.getD t 0).isZero = true
      · have hz : ∑ k ∈ Finset.range nk, of (acc.getD t 0) * slotK pt as s L H cx cy ws k t = 0 := by
          simp [isZero_eq h1, of_zero']
        have hstep : addRunStep as nk ws acc pd t = pd := by simp only [addRunStep, h1, ↓reduceIte]
        rw [hstep, hz, zero_add]
        exact ih pd hpd
      · have hstep : addRunStep as nk ws acc pd t = addK as ws t (acc.getD t 0) nk 0 pd := by
          simp only [addRunStep, h1, Bool.false_eq_true, ↓reduceIte]
        obtain ⟨h3, h4⟩ := addK_spec as ws t (acc.getD t 0) hs (hiE pt L H cx * hiG pt L H cy) nk 0 pd hpd
        obtain ⟨h5, h6⟩ := ih _ h3
        rw [hstep]
        refine ⟨h5, ?_⟩
        rw [h6, h4, add_assoc]
        congr 2
        refine Finset.sum_congr rfl fun k _ => ?_
        simp only [Nat.zero_add, slotK]
        split_ifs <;> ring
  obtain ⟨h1, h2⟩ := key (List.range 64) pd hpd
  refine ⟨h1, ?_⟩
  unfold addRun pendK
  rw [h2, list_range_sum, Finset.sum_comm]

/-- **A run's sums, added to pending sums** (explicit): their value at every copy, at the run's chunks, the pending
sums of other chunks first added. -/
theorem flushKP_pend (pt : FoldPt) {s : ℕ} (hs : s < 64) (L H : ℕ) (iv : ℕ → ℕ → GF128) (as : Array ℕ) (nk : ℕ)
    (st : Tab × ℕ × ℕ × Array F128) (hst : st.2.2.2.size = 64) (cx cy : ℕ) (ws acc : Array F128) :
    (Tab.flushKP as nk st cx cy ws acc).2.2.2.size = 64 ∧
      tabExp pt s L H iv (Tab.flushKP as nk st cx cy ws acc).1 + pendP pt s L H (Tab.flushKP as nk st cx cy ws acc) =
        tabExp pt s L H iv st.1 + pendP pt s L H st + pendK pt as nk s L H cx ws cy acc := by
  obtain ⟨o, px, py, pd⟩ := st
  dsimp only at hst ⊢
  obtain ⟨h1, h2⟩ := addRun_spec pt hs L H as nk ws acc cx cy pd hst
  obtain ⟨h1', h2'⟩ := addRun_spec pt hs L H as nk ws acc cx cy (Array.replicate 64 0) (by simp)
  by_cases hc : px = cx ∧ py = cy
  · obtain ⟨hx, hy⟩ := hc
    simp only [Tab.flushKP, hx, hy, beq_self_eq_true, Bool.and_self, ↓reduceIte]
    refine ⟨h1, ?_⟩
    unfold pendP
    dsimp only
    rw [h2]
    ring
  · have hc' : (px == cx && py == cy) = false := Bool.eq_false_iff.2 fun h => hc (by simpa using h)
    simp only [Tab.flushKP, hc', Bool.false_eq_true, ↓reduceIte]
    refine ⟨h1', ?_⟩
    unfold pendP
    dsimp only
    rw [h2', flushP_tabExp pt hs, getD_replicate, of_zero']
    ring

/-- **A row's references at every copy, on pending sums** (explicit, each copy's low part below `B`, `B` dividing 64):
`Tab.refsK`'s runs, the last added to the pending sums, each reference `Tab.ref`'s term at each copy. -/
theorem refsK_exp (D : Def) (pt : FoldPt) {s : ℕ} (hs : s < 64) (L H A B : ℕ) (as : Array ℕ) (gL : Half)
    (pos : ℕ → ℕ) (fac : ℕ → F128) (iv : ℕ → ℕ → GF128) {nk : ℕ} (hB : 64 % B = 0)
    (has : ∀ k < nk, as.getD k 0 < B) (hfac : ∀ v < D.cols, fac v = gL.at (B * (pos v % A) / 64)) (cx : ℕ)
    (ws : Array F128) (st : Tab × ℕ × ℕ × Array F128) (hst : st.2.2.2.size = 64) (vs : Array ℕ) :
    (Tab.refsK D pt A B as gL pos fac nk cx ws st vs).2.2.2.size = 64 ∧
      tabExp pt s L H iv (Tab.refsK D pt A B as gL pos fac nk cx ws st vs).1 +
          pendP pt s L H (Tab.refsK D pt A B as gL pos fac nk cx ws st vs) =
        tabExp pt s L H iv st.1 + pendP pt s L H st + (vs.toList.map fun v => ∑ k ∈ Finset.range nk,
          of (ws.getD k 0) * (hiE pt L H cx * refX D pt s L H A B as gL pos iv k v)).sum := by
  obtain ⟨o, px, py, pd⟩ := st
  dsimp only at hst ⊢
  unfold Tab.refsK
  dsimp only
  rw [← Array.foldl_toList]
  obtain ⟨-, h⟩ := refStepK_fold D pt (s := s) L H A B as gL pos fac iv hB has hfac cx ws vs.toList
    (o, py, Array.replicate 64 0) (by simp)
  obtain ⟨h1, h2⟩ := flushKP_pend pt hs L H iv as nk
    ((vs.toList.foldl (Tab.refStepK D pt A B as gL pos fac nk cx ws) (o, py, Array.replicate 64 0)).1, px, py, pd) hst
    cx (vs.toList.foldl (Tab.refStepK D pt A B as gL pos fac nk cx ws) (o, py, Array.replicate 64 0)).2.1 ws
    (vs.toList.foldl (Tab.refStepK D pt A B as gL pos fac nk cx ws) (o, py, Array.replicate 64 0)).2.2
  refine ⟨h1, ?_⟩
  rw [h2]
  have hz : pendK pt as nk s L H cx ws py (Array.replicate 64 0) = 0 := by
    simp only [pendK, getD_replicate, of_zero', zero_mul, Finset.sum_const_zero]
  dsimp only at h
  rw [hz, add_zero] at h
  unfold pendP
  dsimp only
  linear_combination h

/-- **One reference at every copy, on pending sums** (explicit): `Tab.ref`'s term at each copy. -/
theorem refDK_exp (D : Def) (pt : FoldPt) {s : ℕ} (hs : s < 64) (L H A B : ℕ) (as : Array ℕ) (gL : Half)
    (pos : ℕ → ℕ) (fac : ℕ → F128) (iv : ℕ → ℕ → GF128) {nk : ℕ} (hB : 64 % B = 0)
    (has : ∀ k < nk, as.getD k 0 < B) (hfac : ∀ v < D.cols, fac v = gL.at (B * (pos v % A) / 64)) (cx : ℕ)
    (ws : Array F128) (st : Tab × ℕ × ℕ × Array F128) (hst : st.2.2.2.size = 64) (v : ℕ) :
    (Tab.refDK D pt A B as gL pos fac nk cx ws st v).2.2.2.size = 64 ∧
      tabExp pt s L H iv (Tab.refDK D pt A B as gL pos fac nk cx ws st v).1 +
          pendP pt s L H (Tab.refDK D pt A B as gL pos fac nk cx ws st v) =
        tabExp pt s L H iv st.1 + pendP pt s L H st +
          ∑ k ∈ Finset.range nk, of (ws.getD k 0) * (hiE pt L H cx * refX D pt s L H A B as gL pos iv k v) := by
  obtain ⟨o, px, py, pd⟩ := st
  dsimp only at hst ⊢
  by_cases hv : v < D.cols
  · have hrx : ∑ k ∈ Finset.range nk, of (ws.getD k 0) * (hiE pt L H cx * refX D pt s L H A B as gL pos iv k v) =
        ∑ j ∈ Finset.range nk, if as.getD (0 + j) 0 + B * (pos v % A) % 64 = s then
          of (ws.getD (0 + j) 0) * of (fac v) * (hiE pt L H cx * hiG pt L H (pos v / A)) else 0 := by
      refine Finset.sum_congr rfl fun k hk => ?_
      obtain ⟨h1, h2⟩ := low_split hB (has k (Finset.mem_range.1 hk)) (pos v % A)
      simp only [refX, hv, ↓reduceIte, h1, h2, hfac v hv, Nat.zero_add]
      split_ifs <;> ring
    rw [hrx]
    obtain ⟨h1, h2⟩ := addK_spec as ws (B * (pos v % A) % 64) (fac v) hs
      (hiE pt L H cx * hiG pt L H (pos v / A)) nk 0 pd hst
    obtain ⟨h1', h2'⟩ := addK_spec as ws (B * (pos v % A) % 64) (fac v) hs
      (hiE pt L H cx * hiG pt L H (pos v / A)) nk 0 (Array.replicate 64 0) (by simp)
    by_cases hc : px = cx ∧ py = pos v / A
    · obtain ⟨hx, hy⟩ := hc
      simp only [Tab.refDK, hv, ↓reduceIte, hx, hy, beq_self_eq_true, Bool.and_self]
      refine ⟨h1, ?_⟩
      unfold pendP
      dsimp only
      rw [h2]
      ring
    · have hc' : (px == cx && py == pos v / A) = false := Bool.eq_false_iff.2 fun h => hc (by simpa using h)
      simp only [Tab.refDK, hv, ↓reduceIte, hc', Bool.false_eq_true]
      refine ⟨h1', ?_⟩
      unfold pendP
      dsimp only
      rw [h2', flushP_tabExp pt hs, getD_replicate, of_zero']
      ring
  · simp only [Tab.refDK, hv, ↓reduceIte]
    refine ⟨hst, ?_⟩
    have h := foldl_val (tabExp pt s L H iv) (fun o k => Tab.ref D pt false A B as gL pos o k cx (ws.getD k 0) v)
      (fun k => of (ws.getD k 0) * (hiE pt L H cx * refX D pt s L H A B as gL pos iv k v))
      (fun o k => by rw [ref_tabExp, mul_assoc]) (List.range nk) o
    unfold pendP
    dsimp only
    rw [h, list_range_sum]
    ring

/-- **A row's references at every copy, on pending sums** (explicit): `Tab.ref`'s term at each copy. -/
theorem halfK_exp (D : Def) (pt : FoldPt) {s : ℕ} (hs : s < 64) (L H A B : ℕ) (as : Array ℕ) (gL : Half)
    (pos : ℕ → ℕ) (fac : ℕ → F128) (iv : ℕ → ℕ → GF128) {nk : ℕ} (hB : 64 % B = 0)
    (has : ∀ k < nk, as.getD k 0 < B) (hfac : ∀ v < D.cols, fac v = gL.at (B * (pos v % A) / 64)) (cx : ℕ)
    (ws : Array F128) (st : Tab × ℕ × ℕ × Array F128) (hst : st.2.2.2.size = 64) (vs : Array ℕ) :
    (Tab.halfK D pt A B as gL pos fac nk cx ws st vs).2.2.2.size = 64 ∧
      tabExp pt s L H iv (Tab.halfK D pt A B as gL pos fac nk cx ws st vs).1 +
          pendP pt s L H (Tab.halfK D pt A B as gL pos fac nk cx ws st vs) =
        tabExp pt s L H iv st.1 + pendP pt s L H st + (vs.toList.map fun v => ∑ k ∈ Finset.range nk,
          of (ws.getD k 0) * (hiE pt L H cx * refX D pt s L H A B as gL pos iv k v)).sum := by
  unfold Tab.halfK
  split_ifs
  · rw [← Array.foldl_toList]
    have key : ∀ (l : List ℕ) (st : Tab × ℕ × ℕ × Array F128), st.2.2.2.size = 64 →
        (l.foldl (Tab.refDK D pt A B as gL pos fac nk cx ws) st).2.2.2.size = 64 ∧
          tabExp pt s L H iv (l.foldl (Tab.refDK D pt A B as gL pos fac nk cx ws) st).1 +
              pendP pt s L H (l.foldl (Tab.refDK D pt A B as gL pos fac nk cx ws) st) =
            tabExp pt s L H iv st.1 + pendP pt s L H st + (l.map fun v => ∑ k ∈ Finset.range nk,
              of (ws.getD k 0) * (hiE pt L H cx * refX D pt s L H A B as gL pos iv k v)).sum := by
      intro l
      induction l with
      | nil => intro st hst; simp [hst]
      | cons v l ih =>
        intro st hst
        obtain ⟨h1, h2⟩ := refDK_exp D pt hs L H A B as gL pos fac iv hB has hfac cx ws st hst v
        obtain ⟨h3, h4⟩ := ih _ h1
        refine ⟨h3, ?_⟩
        rw [List.foldl_cons, h4, h2, List.map_cons, List.sum_cons]
        ring
    exact key vs.toList st hst
  · exact refsK_exp D pt hs L H A B as gL pos fac iv hB has hfac cx ws st hst vs

/-- **One row at every copy, on pending sums** (explicit): each of its references, copy by copy. -/
theorem rowK_exp (D : Def) (pt : FoldPt) {s : ℕ} (hs : s < 64) (L H A B : ℕ) (as : Array ℕ) (gL : Half)
    (pos : ℕ → ℕ) (fac : ℕ → F128) (iv : ℕ → ℕ → GF128) {nk : ℕ} (hB : 64 % B = 0)
    (has : ∀ k < nk, as.getD k 0 < B) (hfac : ∀ v < D.cols, fac v = gL.at (B * (pos v % A) / 64))
    (rowLow : ℕ → ℕ → F128) (alpha : F128) (st : Tab × ℕ × ℕ × Array F128) (hst : st.2.2.2.size = 64)
    (r : ℕ × Array ℕ × Array ℕ) :
    (Tab.rowK D pt A B as gL pos fac rowLow alpha nk st r).2.2.2.size = 64 ∧
      tabExp pt s L H iv (Tab.rowK D pt A B as gL pos fac rowLow alpha nk st r).1 +
          pendP pt s L H (Tab.rowK D pt A B as gL pos fac rowLow alpha nk st r) =
        tabExp pt s L H iv st.1 + pendP pt s L H st + ∑ k ∈ Finset.range nk,
          ((r.2.1.toList.map fun v => of (alpha * rowLow k (pos r.1)) *
              (hiE pt L H (pos r.1 / A) * refX D pt s L H A B as gL pos iv k v)).sum +
            (r.2.2.toList.map fun v => of (rowLow k (pos r.1)) *
              (hiE pt L H (pos r.1 / A) * refX D pt s L H A B as gL pos iv k v)).sum) := by
  have hW : ∀ (T : Tab) (n : ℕ), tabExp pt s L H iv { T with work := n } = tabExp pt s L H iv T := fun _ _ => rfl
  have hP : ∀ (T : Tab) (st' : Tab × ℕ × ℕ × Array F128), pendP pt s L H (T, st'.2) = pendP pt s L H st' :=
    fun _ _ => rfl
  obtain ⟨x, ra, rb⟩ := r
  obtain ⟨h1, h2⟩ := halfK_exp D pt hs L H A B as gL pos fac iv hB has hfac (pos x / A)
    (((Array.range nk).map fun k => rowLow k (pos x)).map (alpha * ·)) st hst ra
  obtain ⟨h3, h4⟩ := halfK_exp D pt hs L H A B as gL pos fac iv hB has hfac (pos x / A)
    ((Array.range nk).map fun k => rowLow k (pos x)) _ h1 rb
  simp only [Tab.rowK]
  refine ⟨h3, ?_⟩
  rw [hW, hP, h4, h2, add_assoc, list_sum_comm, list_sum_comm, ← Finset.sum_add_distrib]
  congr 1
  refine Finset.sum_congr rfl fun k hk => ?_
  simp only [getD_map_range _ _ (Finset.mem_range.1 hk), getD_map_map_range _ _ _ (Finset.mem_range.1 hk)]

/-- **Explicit rows at every copy, on pending sums** (each copy's low part below `B`, `B` dividing 64): each row's
references, copy by copy. -/
theorem rowsK_tabExp (D : Def) (pt : FoldPt) {s : ℕ} (hs : s < 64) (L H A B : ℕ) (as : Array ℕ) (gL : Half)
    (pos : ℕ → ℕ) (fac : ℕ → F128) (iv : ℕ → ℕ → GF128) {nk : ℕ} (hB : 64 % B = 0)
    (has : ∀ k < nk, as.getD k 0 < B) (hfac : ∀ v < D.cols, fac v = gL.at (B * (pos v % A) / 64))
    (rowLow : ℕ → ℕ → F128) (alpha : F128) (rows : Array (ℕ × Array ℕ × Array ℕ)) (o : Tab) :
    tabExp pt s L H iv (Tab.rowsK D pt A B as gL pos fac rowLow alpha nk rows o) =
      tabExp pt s L H iv o + (rows.toList.map fun r => ∑ k ∈ Finset.range nk,
        ((r.2.1.toList.map fun v => of (alpha * rowLow k (pos r.1)) *
            (hiE pt L H (pos r.1 / A) * refX D pt s L H A B as gL pos iv k v)).sum +
          (r.2.2.toList.map fun v => of (rowLow k (pos r.1)) *
            (hiE pt L H (pos r.1 / A) * refX D pt s L H A B as gL pos iv k v)).sum)).sum := by
  unfold Tab.rowsK
  rw [← Array.foldl_toList]
  have key : ∀ (l : List (ℕ × Array ℕ × Array ℕ)) (st : Tab × ℕ × ℕ × Array F128), st.2.2.2.size = 64 →
      tabExp pt s L H iv (l.foldl (Tab.rowK D pt A B as gL pos fac rowLow alpha nk) st).1 +
          pendP pt s L H (l.foldl (Tab.rowK D pt A B as gL pos fac rowLow alpha nk) st) =
        tabExp pt s L H iv st.1 + pendP pt s L H st + (l.map fun r => ∑ k ∈ Finset.range nk,
          ((r.2.1.toList.map fun v => of (alpha * rowLow k (pos r.1)) *
              (hiE pt L H (pos r.1 / A) * refX D pt s L H A B as gL pos iv k v)).sum +
            (r.2.2.toList.map fun v => of (rowLow k (pos r.1)) *
              (hiE pt L H (pos r.1 / A) * refX D pt s L H A B as gL pos iv k v)).sum)).sum := by
    intro l
    induction l with
    | nil => intro st _; simp
    | cons r l ih =>
      intro st hst
      obtain ⟨h1, h2⟩ := rowK_exp D pt hs L H A B as gL pos fac iv hB has hfac rowLow alpha st hst r
      rw [List.foldl_cons, ih _ h1, h2, List.map_cons, List.sum_cons]
      ring
  have h := key rows.toList (o, 0, 0, Array.replicate 64 0) (by simp)
  dsimp only at h ⊢
  rw [flushP_pend pt hs, h]
  simp only [pendP, getD_replicate, of_zero', zero_mul, add_zero]

/-! ## Rows column by column

`Tab.rowsT` sums each row's weight into its own columns' `W` and multiplies each column's sum by its factor once. The
sums' value (`pendW`) is what that pass will add, and each reference keeps the table's value plus theirs equal to the
per-reference sum. -/

/-- `Tab.rowsT`'s column sums' value at slot `s` (explicit): `W[u]` times own column `u`'s term at copy `0`. -/
noncomputable def pendW (D : Def) (pt : FoldPt) (s L H A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ)
    (iv : ℕ → ℕ → GF128) (cx : ℕ) (W : Array F128) : GF128 :=
  ∑ u ∈ Finset.range D.cols, of (W.getD u 0) * (hiE pt L H cx * refX D pt s L H A B as gL pos iv 0 u)

/-- **One reference, column by column** (explicit): `Tab.ref`'s term at copy `0`. -/
theorem colT_exp (D : Def) (pt : FoldPt) (s L H A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ)
    (iv : ℕ → ℕ → GF128) (cx : ℕ) (w : F128) (st : Tab × Array F128) (hst : st.2.size = D.cols) (v : ℕ) :
    (Tab.colT D pt A B as gL pos cx w st v).2.size = D.cols ∧
      tabExp pt s L H iv (Tab.colT D pt A B as gL pos cx w st v).1 +
          pendW D pt s L H A B as gL pos iv cx (Tab.colT D pt A B as gL pos cx w st v).2 =
        tabExp pt s L H iv st.1 + pendW D pt s L H A B as gL pos iv cx st.2 +
          of w * (hiE pt L H cx * refX D pt s L H A B as gL pos iv 0 v) := by
  obtain ⟨o, W⟩ := st
  dsimp only at hst ⊢
  by_cases hv : v < D.cols
  · simp only [Tab.colT, hv, ↓reduceIte]
    refine ⟨by rw [Array.size_modify, hst], ?_⟩
    unfold pendW
    rw [sum_modify_add' W (by rw [hst]; exact hv) hv]
    ring
  · simp only [Tab.colT, hv, ↓reduceIte]
    refine ⟨hst, ?_⟩
    rw [ref_tabExp]
    ring

/-- **One row, column by column** (explicit): each of its references at copy `0`. -/
theorem rowT_exp (D : Def) (pt : FoldPt) (s L H A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ)
    (iv : ℕ → ℕ → GF128) (rowLow : ℕ → ℕ → F128) (alpha : F128) (cx : ℕ) (st : Tab × Array F128)
    (hst : st.2.size = D.cols) (r : ℕ × Array ℕ × Array ℕ) :
    (Tab.rowT D pt A B as gL pos rowLow alpha cx st r).2.size = D.cols ∧
      tabExp pt s L H iv (Tab.rowT D pt A B as gL pos rowLow alpha cx st r).1 +
          pendW D pt s L H A B as gL pos iv cx (Tab.rowT D pt A B as gL pos rowLow alpha cx st r).2 =
        tabExp pt s L H iv st.1 + pendW D pt s L H A B as gL pos iv cx st.2 +
          ((r.2.1.toList.map fun v => of (alpha * rowLow 0 (pos r.1)) *
              (hiE pt L H cx * refX D pt s L H A B as gL pos iv 0 v)).sum +
            (r.2.2.toList.map fun v => of (rowLow 0 (pos r.1)) *
              (hiE pt L H cx * refX D pt s L H A B as gL pos iv 0 v)).sum) := by
  have hW : ∀ (T : Tab) (n : ℕ), tabExp pt s L H iv { T with work := n } = tabExp pt s L H iv T := fun _ _ => rfl
  have key := fun (w : F128) => foldl_val_inv (fun st : Tab × Array F128 => st.2.size = D.cols)
    (fun st => tabExp pt s L H iv st.1 + pendW D pt s L H A B as gL pos iv cx st.2)
    (Tab.colT D pt A B as gL pos cx w)
    (fun v => of w * (hiE pt L H cx * refX D pt s L H A B as gL pos iv 0 v))
    (fun st v hst => colT_exp D pt s L H A B as gL pos iv cx w st hst v)
  obtain ⟨x, ra, rb⟩ := r
  obtain ⟨h1, h2⟩ := key (alpha * rowLow 0 (pos x)) ra.toList st hst
  obtain ⟨h3, h4⟩ := key (rowLow 0 (pos x)) rb.toList _ h1
  simp only [Tab.rowT, ← Array.foldl_toList]
  refine ⟨h3, ?_⟩
  rw [hW, h4, h2]
  ring

/-- **One column's sum, on pending sums** (explicit): own column `v`'s term at copy `0`, its sum `x` the weight. -/
theorem colD_exp (D : Def) (pt : FoldPt) {s : ℕ} (hs : s < 64) (L H A B : ℕ) (as : Array ℕ) (gL : Half)
    (pos : ℕ → ℕ) (fac : ℕ → F128) (iv : ℕ → ℕ → GF128) (hB : 64 % B = 0) (has : as.getD 0 0 < B)
    (hfac : ∀ v < D.cols, fac v = gL.at (B * (pos v % A) / 64)) (cx : ℕ) (x : F128)
    (st : Tab × ℕ × ℕ × Array F128) (hst : st.2.2.2.size = 64) {v : ℕ} (hv : v < D.cols) :
    (Tab.colD A B as pos fac cx x st v).2.2.2.size = 64 ∧
      tabExp pt s L H iv (Tab.colD A B as pos fac cx x st v).1 + pendP pt s L H (Tab.colD A B as pos fac cx x st v) =
        tabExp pt s L H iv st.1 + pendP pt s L H st +
          of x * (hiE pt L H cx * refX D pt s L H A B as gL pos iv 0 v) := by
  obtain ⟨o, px, py, pd⟩ := st
  dsimp only at hst ⊢
  obtain ⟨h1, h2⟩ := low_split hB has (pos v % A)
  have ht : as.getD 0 0 + B * (pos v % A) % 64 < 64 := by rw [← h1]; exact Nat.mod_lt _ (by norm_num)
  have hrx : of x * (hiE pt L H cx * refX D pt s L H A B as gL pos iv 0 v) =
      if as.getD 0 0 + B * (pos v % A) % 64 = s then
        of x * of (fac v) * (hiE pt L H cx * hiG pt L H (pos v / A)) else 0 := by
    simp only [refX, hv, ↓reduceIte, h1, h2, hfac v hv]
    split_ifs <;> ring
  have hg : ∀ q : Array F128, q.size = 64 →
      of ((q.modify (as.getD 0 0 + B * (pos v % A) % 64) (x * fac v + ·)).getD s 0) =
        of (q.getD s 0) + if as.getD 0 0 + B * (pos v % A) % 64 = s then of x * of (fac v) else 0 := fun q hq => by
    have hi : as.getD 0 0 + B * (pos v % A) % 64 < q.size := by rw [hq]; exact ht
    rw [getD_modify _ _ hi]
    split_ifs <;> (try simp only [of_add, of_mul]) <;> ring
  by_cases hc : px = cx ∧ py = pos v / A
  · obtain ⟨hx, hy⟩ := hc
    simp only [Tab.colD, hx, hy, beq_self_eq_true, Bool.and_self, ↓reduceIte]
    refine ⟨by rw [Array.size_modify, hst], ?_⟩
    unfold pendP
    dsimp only
    rw [hg pd hst, hrx]
    split_ifs <;> ring
  · have hc' : (px == cx && py == pos v / A) = false := Bool.eq_false_iff.2 fun h => hc (by simpa using h)
    simp only [Tab.colD, hc', Bool.false_eq_true, ↓reduceIte]
    refine ⟨by simp, ?_⟩
    unfold pendP
    dsimp only
    rw [hg (Array.replicate 64 0) (by simp), flushP_tabExp pt hs, getD_replicate, of_zero', hrx]
    split_ifs <;> ring

/-- **Column sums, multiplied** (explicit): own columns `v .. v + n`, each sum `W[v + j]` times its term at copy `0`. -/
theorem colsP_exp (D : Def) (pt : FoldPt) {s : ℕ} (hs : s < 64) (L H A B : ℕ) (as : Array ℕ) (gL : Half)
    (pos : ℕ → ℕ) (fac : ℕ → F128) (iv : ℕ → ℕ → GF128) (hB : 64 % B = 0) (has : as.getD 0 0 < B)
    (hfac : ∀ v < D.cols, fac v = gL.at (B * (pos v % A) / 64)) (cx : ℕ) (W : Array F128) :
    ∀ (n v : ℕ) (st : Tab × ℕ × ℕ × Array F128), v + n ≤ D.cols → st.2.2.2.size = 64 →
      (Tab.colsP D pt A B as gL pos fac cx W n v st).2.2.2.size = 64 ∧
        tabExp pt s L H iv (Tab.colsP D pt A B as gL pos fac cx W n v st).1 +
            pendP pt s L H (Tab.colsP D pt A B as gL pos fac cx W n v st) =
          tabExp pt s L H iv st.1 + pendP pt s L H st + ∑ j ∈ Finset.range n,
            of (W.getD (v + j) 0) * (hiE pt L H cx * refX D pt s L H A B as gL pos iv 0 (v + j))
  | 0, v, st, _, hst => by simp [Tab.colsP, hst]
  | n + 1, v, st, hn, hst => by
    have hstep : (if (W.getD v 0).isZero = true then st
          else Tab.colD A B as pos fac cx (W.getD v 0) st v).2.2.2.size = 64 ∧
        tabExp pt s L H iv (if (W.getD v 0).isZero = true then st
            else Tab.colD A B as pos fac cx (W.getD v 0) st v).1 +
          pendP pt s L H (if (W.getD v 0).isZero = true then st
            else Tab.colD A B as pos fac cx (W.getD v 0) st v) =
        tabExp pt s L H iv st.1 + pendP pt s L H st +
          of (W.getD v 0) * (hiE pt L H cx * refX D pt s L H A B as gL pos iv 0 v) := by
      split_ifs with hz
      · rw [isZero_eq hz, of_zero', zero_mul, add_zero]
        exact ⟨hst, rfl⟩
      · exact colD_exp D pt hs L H A B as gL pos fac iv hB has hfac cx (W.getD v 0) st hst (by omega)
    obtain ⟨h1, h2⟩ := hstep
    obtain ⟨h3, h4⟩ := colsP_exp D pt hs L H A B as gL pos fac iv hB has hfac cx W n (v + 1) _ (by omega) h1
    simp only [Tab.colsP]
    refine ⟨h3, ?_⟩
    rw [h4, h2, Finset.sum_range_succ']
    simp only [Nat.add_zero, show ∀ j, v + 1 + j = v + (j + 1) from fun j => by omega]
    ring

/-- **Explicit rows of one row chunk at one copy, column by column** (its low part below `B`, `B` dividing 64): each
row's references at copy `0`. -/
theorem rowsT_tabExp (D : Def) (pt : FoldPt) {s : ℕ} (hs : s < 64) (L H A B : ℕ) (as : Array ℕ) (gL : Half)
    (pos : ℕ → ℕ) (fac : ℕ → F128) (iv : ℕ → ℕ → GF128) (hB : 64 % B = 0) (has : as.getD 0 0 < B)
    (hfac : ∀ v < D.cols, fac v = gL.at (B * (pos v % A) / 64)) (rowLow : ℕ → ℕ → F128) (alpha : F128) (cx : ℕ)
    (rows : Array (ℕ × Array ℕ × Array ℕ)) (o : Tab) :
    tabExp pt s L H iv (Tab.rowsT D pt A B as gL pos fac rowLow alpha cx rows o) =
      tabExp pt s L H iv o + (rows.toList.map fun r =>
        (r.2.1.toList.map fun v => of (alpha * rowLow 0 (pos r.1)) *
            (hiE pt L H cx * refX D pt s L H A B as gL pos iv 0 v)).sum +
          (r.2.2.toList.map fun v => of (rowLow 0 (pos r.1)) *
            (hiE pt L H cx * refX D pt s L H A B as gL pos iv 0 v)).sum).sum := by
  obtain ⟨-, h2⟩ := foldl_val_inv (fun st : Tab × Array F128 => st.2.size = D.cols)
    (fun st => tabExp pt s L H iv st.1 + pendW D pt s L H A B as gL pos iv cx st.2)
    (Tab.rowT D pt A B as gL pos rowLow alpha cx) _
    (fun st r hst => rowT_exp D pt s L H A B as gL pos iv rowLow alpha cx st hst r)
    rows.toList (o, Array.replicate D.cols 0) (by simp)
  unfold Tab.rowsT
  rw [← Array.foldl_toList]
  generalize rows.toList.foldl (Tab.rowT D pt A B as gL pos rowLow alpha cx) (o, Array.replicate D.cols 0) = st
    at h2 ⊢
  obtain ⟨-, h4⟩ := colsP_exp D pt hs L H A B as gL pos fac iv hB has hfac cx st.2 D.cols 0
    (st.1, 0, 0, Array.replicate 64 0) (by omega) (by simp)
  dsimp only at h2 h4 ⊢
  rw [flushP_pend pt hs, h4]
  simp only [pendP, getD_replicate, of_zero', zero_mul, add_zero, Nat.zero_add]
  unfold pendW at h2
  simp only [getD_replicate, of_zero', zero_mul, Finset.sum_const_zero, add_zero] at h2
  exact h2

/-- **Explicit rows at every copy** (each copy's low part below `B`, `B` dividing 64), column by column
(`Tab.rowsT`) or row by row (`Tab.rowsK`): each row's references, copy by copy. -/
theorem rowsC_tabExp (D : Def) (pt : FoldPt) {s : ℕ} (hs : s < 64) (L H A B : ℕ) (as : Array ℕ) (gL : Half)
    (pos : ℕ → ℕ) (fac : ℕ → F128) (iv : ℕ → ℕ → GF128) {nk : ℕ} (hB : 64 % B = 0)
    (has : ∀ k < nk, as.getD k 0 < B) (hfac : ∀ v < D.cols, fac v = gL.at (B * (pos v % A) / 64))
    (rowLow : ℕ → ℕ → F128) (alpha : F128) (rows : Array (ℕ × Array ℕ × Array ℕ)) (o : Tab) :
    tabExp pt s L H iv (Tab.rowsC D pt A B as gL pos fac rowLow alpha nk rows o) =
      tabExp pt s L H iv o + (rows.toList.map fun r => ∑ k ∈ Finset.range nk,
        ((r.2.1.toList.map fun v => of (alpha * rowLow k (pos r.1)) *
            (hiE pt L H (pos r.1 / A) * refX D pt s L H A B as gL pos iv k v)).sum +
          (r.2.2.toList.map fun v => of (rowLow k (pos r.1)) *
            (hiE pt L H (pos r.1 / A) * refX D pt s L H A B as gL pos iv k v)).sum)).sum := by
  unfold Tab.rowsC
  dsimp only
  split_ifs with hc
  · simp only [Bool.and_eq_true, beq_iff_eq] at hc
    obtain ⟨⟨rfl, -⟩, hall⟩ := hc
    have hall' : rows.toList.all (fun r => pos r.1 / A == pos (rows.getD 0 (0, #[], #[])).1 / A) = true := by
      rw [Array.all_toList]
      exact hall
    have hcx : ∀ r ∈ rows.toList, pos r.1 / A = pos (rows.getD 0 (0, #[], #[])).1 / A := fun r hr => by
      simpa using List.all_eq_true.1 hall' r hr
    rw [rowsT_tabExp D pt hs L H A B as gL pos fac iv hB (has 0 Nat.one_pos) hfac]
    congr 1
    refine congrArg List.sum (List.map_congr_left fun r hr => ?_)
    rw [Finset.sum_range_one, hcx r hr]
  · exact rowsK_tabExp D pt hs L H A B as gL pos fac iv hB has hfac rowLow alpha rows o

/-- **A Definition's own rows**, through `Tab.refs`, which adds `of w · Z k cx v` for each reference, or, while
explicit with each copy's low part below `B` and `B` dividing 64, through `Tab.rowsC`, which adds them at every copy:
each row's references, copy by copy. -/
theorem rows_val (V : Tab → GF128) (hW : ∀ (o : Tab) (n : ℕ), V { o with work := n } = V o)
    (D : Def) (pt : FoldPt) (summed : Bool) (A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ) (fac : ℕ → F128)
    (Z : ℕ → ℕ → ℕ → GF128)
    (hR : ∀ o k cx w vs, V (Tab.refs D pt summed A B as gL pos o k cx w vs) =
      V o + (vs.toList.map fun v => of w * Z k cx v).sum) {nk : ℕ}
    (hK : summed = false → 64 % B = 0 → (∀ k < nk, as.getD k 0 < B) → ∀ rowLow alpha rows o,
      V (Tab.rowsC D pt A B as gL pos fac rowLow alpha nk rows o) =
        V o + (rows.toList.map fun r => ∑ k ∈ Finset.range nk,
          ((r.2.1.toList.map fun v => of (alpha * rowLow k (pos r.1)) * Z k (pos r.1 / A) v).sum +
            (r.2.2.toList.map fun v => of (rowLow k (pos r.1)) * Z k (pos r.1 / A) v).sum)).sum)
    (rowLow : ℕ → ℕ → F128) (alpha : F128) (rows : Array (ℕ × Array ℕ × Array ℕ)) (o : Tab) :
    V (Tab.rows D pt summed A B as gL pos fac rowLow alpha nk rows o) =
      V o + (rows.toList.map fun r => ∑ k ∈ Finset.range nk,
        ((r.2.1.toList.map fun v => of (alpha * rowLow k (pos r.1)) * Z k (pos r.1 / A) v).sum +
          (r.2.2.toList.map fun v => of (rowLow k (pos r.1)) * Z k (pos r.1 / A) v).sum)).sum := by
  unfold Tab.rows
  split_ifs with hc
  · simp at hc
    obtain ⟨⟨hs, hB⟩, has'⟩ := hc
    exact hK hs hB (fun k hk => by simpa using has' k hk) rowLow alpha rows o
  · rw [← Array.foldl_toList]
    refine foldl_val V _ (fun r : ℕ × Array ℕ × Array ℕ => ∑ k ∈ Finset.range nk,
        ((r.2.1.toList.map fun v => of (alpha * rowLow k (pos r.1)) * Z k (pos r.1 / A) v).sum +
          (r.2.2.toList.map fun v => of (rowLow k (pos r.1)) * Z k (pos r.1 / A) v).sum)) (fun o' r => ?_)
      rows.toList o
    obtain ⟨x, ra, rb⟩ := r
    dsimp only
    rw [hW, ← list_range_sum]
    refine foldl_val V _ (fun k : ℕ => (ra.toList.map fun v => of (alpha * rowLow k (pos x)) * Z k (pos x / A) v).sum +
        (rb.toList.map fun v => of (rowLow k (pos x)) * Z k (pos x / A) v).sum) (fun o'' k => ?_) (List.range nk) o'
    rw [hR, hR, add_assoc]

/-- **A call's bindings**, copy by copy. -/
theorem binds_val (V : Tab → GF128) (bind : Tab → ℕ → Tab → (ℕ → ℕ × F128) → (ℕ → ℕ → List (ℕ × F128)) → Tab)
    (ch : Tab) (rm : ℕ → ℕ × F128) (wk : ℕ → ℕ → List (ℕ × F128)) (G : ℕ → GF128)
    (h : ∀ o j, V (bind o j ch rm wk) = V o + G j) (o : Tab) (count : ℕ) :
    V (Tab.binds bind o count ch rm wk) = V o + ∑ j ∈ Finset.range count, G j := by
  unfold Tab.binds
  rw [foldl_val V _ G (fun o j => h o j), list_range_sum]

theorem sum_filter_range (p : ℕ → Bool) (g : ℕ → GF128) : ∀ n,
    (((List.range n).filter p).map g).sum = ∑ k ∈ Finset.range n, if p k then g k else 0
  | 0 => by simp
  | n + 1 => by
    rw [List.range_succ, List.filter_append, List.map_append, List.sum_append, sum_filter_range p g n,
      Finset.sum_range_succ]
    by_cases h : p n <;> simp [h]

/-- Over the nonzero slots of `W`, the slot-`s` entry's term is all: the rest are another slot's, or zero. -/
theorem slots_pick (W : Array F128) {s : ℕ} (hs : s < 64) (X : ℕ → GF128) (Y : GF128)
    (hX : ∀ s', W.getD s' 0 = 0 → X s' = 0) :
    ((((List.range 64).filter fun s' => !(W.getD s' 0).isZero).map fun s' => X s' * (if s' = s then Y else 0)).sum) =
      X s * Y := by
  rw [sum_filter_range, Finset.sum_eq_single s (fun b _ hb => by simp [hb]) (fun h => absurd (Finset.mem_range.2 hs) h)]
  by_cases hz : (W.getD s 0).isZero = true
  · simp [hz, hX s (isZero_eq hz)]
  · rw [ite_eq_left (by simpa using hz), ite_eq_left rfl]

/-- **A run of lanes spread over the slots**: the child's summed entries at slot `s` by `W[s]`, the constant's at `qc`'s
slot by `EL`, each re-keyed. -/
theorem slots_tabExp (pt : FoldPt) {s : ℕ} (hs : s < 64) (L H : ℕ) (iv : ℕ → ℕ → GF128) (W : Array F128) (EL : F128)
    (n : ℕ) (o ch : Tab) (rm cm : ℕ → ℕ × F128) :
    tabExp pt s L H iv (Tab.slots pt W EL n o ch rm cm) = tabExp pt s L H iv o +
      (of (W.getD s 0) * hsum ch.S (fun e =>
          of (rm e.1).2 * hiE pt L H (rm e.1).1 * (of (cm e.2).2 * hiG pt L H (cm e.2).1)) +
        of EL * Gs pt s pt.qc * hsum ch.C (fun c => of (rm c).2 * hiE pt L H (rm c).1)) := by
  unfold Tab.slots tabExp
  dsimp only
  rw [fold_val (fun mp : HashMap (ℕ × ℕ × ℕ) F128 =>
        hsum mp (fun e => if e.2.2 = s then hiE pt L H e.1 * hiG pt L H e.2.1 else 0)) _
      (fun (e : ℕ × ℕ) (v : F128) => (((List.range 64).filter fun s' => !(W.getD s' 0).isZero).map fun s' =>
        of (W.getD s' 0 * v * (rm e.1).2 * (cm e.2).2) *
          (if s' = s then hiE pt L H (rm e.1).1 * hiG pt L H (cm e.2).1 else 0)).sum) ?_,
    hsum_fold_bump ch.C o.PC _ fun k v => (((rm k).1, pt.qc % 64), EL * v * (rm k).2 * pt.g pt.qc)]
  · have e1 : ∀ e : (ℕ × ℕ) × F128, (((List.range 64).filter fun s' => !(W.getD s' 0).isZero).map fun s' =>
          of (W.getD s' 0 * e.2 * (rm e.1.1).2 * (cm e.1.2).2) *
            (if s' = s then hiE pt L H (rm e.1.1).1 * hiG pt L H (cm e.1.2).1 else 0)).sum =
        of (W.getD s 0) * (of e.2 * (of (rm e.1.1).2 * hiE pt L H (rm e.1.1).1 *
          (of (cm e.1.2).2 * hiG pt L H (cm e.1.2).1))) := by
      intro e
      rw [slots_pick W hs (fun s' => of (W.getD s' 0 * e.2 * (rm e.1.1).2 * (cm e.1.2).2)) _
        (fun s' h0 => by simp only [h0, of_mul, of_zero', zero_mul])]
      simp only [of_mul]
      ring
    have e2 : ∀ e : ℕ × F128, of (EL * e.2 * (rm e.1).2 * pt.g pt.qc) *
        (if pt.qc % 64 = s then hiE pt L H (rm e.1).1 else 0) =
        of EL * Gs pt s pt.qc * (of e.2 * (of (rm e.1).2 * hiE pt L H (rm e.1).1)) := by
      intro e
      unfold Gs
      split_ifs <;> simp only [of_mul, mul_zero, zero_mul] <;> ring
    unfold hsum
    simp only [e1, e2, List.sum_map_mul_left]
    ring
  · rintro mp ⟨cx, cy⟩ v
    dsimp only
    exact foldl_val (fun mp : HashMap (ℕ × ℕ × ℕ) F128 =>
        hsum mp (fun e => if e.2.2 = s then hiE pt L H e.1 * hiG pt L H e.2.1 else 0))
      (fun mp s' => bump mp ((rm cx).1, (cm cy).1, s') (W.getD s' 0 * v * (rm cx).2 * (cm cy).2))
      (fun s' => of (W.getD s' 0 * v * (rm cx).2 * (cm cy).2) *
        (if s' = s then hiE pt L H (rm cx).1 * hiG pt L H (cm cy).1 else 0))
      (fun mp s' => hsum_bump _ _ _ _) _ mp

/-- The slot weights `W` the walk spreads a run of lanes by: `W[s]` is `Σ E(a) · eq(r, ⌊a/64⌋)` over the lanes at `s`. -/
theorem slotW_get (pt : FoldPt) {s : ℕ} (hs : s < 64) (β' : ℕ) (lows : List ℕ) :
    of ((lows.foldl (fun W a => addAt W a (eLow pt β' a * Tmpl.eqAt pt.r 0 (β' - 6) (a / 64)))
      (Array.replicate 64 0)).getD s 0) = (lows.map fun a => of (eLow pt β' a) * gLow pt s β' a).sum := by
  have h := foldl_addAt_get lows (fun a => a) (fun a => eLow pt β' a * Tmpl.eqAt pt.r 0 (β' - 6) (a / 64))
    (Array.replicate 64 0) (s := s) (by simp [hs])
  rw [h, getD_replicate, of_zero', zero_add]
  congr 1
  refine List.map_congr_left fun a _ => ?_
  unfold gLow
  split_ifs <;> simp [of_mul]

theorem slotEL (pt : FoldPt) (β' : ℕ) : ∀ (lows : List ℕ) (x : F128),
    of (lows.foldl (fun EL a => EL + eLow pt β' a) x) = of x + (lows.map fun a => of (eLow pt β' a)).sum
  | [], x => by simp
  | a :: l, x => by rw [List.foldl_cons, slotEL pt β' l, of_add, List.map_cons, List.sum_cons, add_assoc]

/-! ## What a class's table means -/

/-- **What a class's table means**: placed at chunk origin `H` (`h = H · 2^al + off`), an explicit class's `tabExp` is
its copies' `refVal`s, and a summed class's `tabSum` times `E(a)` is the copy at low part `a`'s. -/
def Spec (t : Tmpl) (lays : Array Lay) (pt : FoldPt) (s fuel d β : ℕ) (as : Array ℕ) (off al : ℕ) (T : Tab) : Prop :=
  UOk (t.defs.getD d default).ins (nkOf β as) T ∧
    (¬ 6 ≤ β → ∀ (H : ℕ) (ip : ℕ → List ℕ),
      tabExp pt s (β + al - 6) H (fun t' k => Gs pt s ((ip k).getD t' pt.qc)) T =
        ∑ k ∈ Finset.range as.size,
          refVal t lays pt s fuel d (as.getD k 0 + 2 ^ β * (H * 2 ^ al + off)) (2 ^ β) (ip k)) ∧
    (6 ≤ β → ∀ (H a : ℕ), a < 2 ^ β → ∀ ip : List ℕ,
      of (eLow pt β a) * tabSum pt (β + al - 6) H (gLow pt s β a) (Gs pt s pt.qc)
          (fun t' => Gs pt s (ip.getD t' pt.qc)) T =
        refVal t lays pt s fuel d (a + 2 ^ β * (H * 2 ^ al + off)) (2 ^ β) ip)

/-- A class the walk folds: chunks of at least 64 slots inside the point, explicit low parts below `2^β`. -/
def Pre (pt : FoldPt) (β al : ℕ) (as : Array ℕ) : Prop :=
  6 ≤ β + al ∧ β + al - 6 ≤ pt.rho.size ∧ β + al - 6 ≤ pt.r.size ∧ (¬ 6 ≤ β → ∀ k < as.size, as.getD k 0 < 2 ^ β)

/-- **The cache**: every cached half is its `eq` table, and every cached table its class's, at offset 0 in its
Definition's `palign` chunks, at every depth past its Definition. -/
def StOk (t : Tmpl) (lays : Array Lay) (pt : FoldPt) (s : ℕ) (st : SplitSt) : Prop :=
  HalvesOk pt st ∧ ∀ e β' (as' : Array ℕ) T, st.tabs.get? (e, β', as') = some T →
    Pre pt β' (lays.getD e default).palign.log2 as' →
    ∀ f, e < f → Spec t lays pt s f e β' as' 0 (lays.getD e default).palign.log2 T

/-- What a walk of depth `fuel` establishes. -/
def WalkOk (t : Tmpl) (lays : Array Lay) (multi : Array Bool) (pt : FoldPt) (s fuel : ℕ) : Prop :=
  ∀ d β (as : Array ℕ) off al (st : SplitSt) T st', (walk t lays multi pt fuel d β as off al).run st = .ok (T, st') →
    StOk t lays pt s st → d < fuel → Pre pt β al as → StOk t lays pt s st' ∧ Spec t lays pt s fuel d β as off al T

theorem sv_pure_ok {σ α : Type} {x a : α} {st st' : σ} (h : (pure x : StateT σ V α).run st = .ok (a, st')) :
    x = a ∧ st = st' := by
  simp only [StateT.run_pure] at h
  cases h
  exact ⟨rfl, rfl⟩

theorem sv_get_ok {σ : Type} {a st st' : σ} (h : (get : StateT σ V σ).run st = .ok (a, st')) : a = st ∧ st' = st := by
  simp only [StateT.run_get] at h
  cases h
  exact ⟨rfl, rfl⟩

theorem sv_modify_ok {σ : Type} {f : σ → σ} {st st' : σ} {u : PUnit}
    (h : (modify f : StateT σ V PUnit).run st = .ok (u, st')) : st' = f st := by
  simp only [StateT.run_modify] at h
  cases h
  rfl

theorem sv_throw {α : Type} (e : String) (st : SplitSt) : (throw e : SV α).run st = .error e := rfl

theorem spec_stable {t : Tmpl} {lays : Array Lay} {pt : FoldPt} {s : ℕ}
    (hcall : ∀ d i, i < (t.defs.getD d default).calls.size → ((t.defs.getD d default).calls.getD i default).def_ < d)
    {F f e β : ℕ} {as : Array ℕ} {off al : ℕ} {T : Tab} (h : Spec t lays pt s F e β as off al T) (hF : e < F)
    (hf : e < f) : Spec t lays pt s f e β as off al T := by
  obtain ⟨hU, hX, hS⟩ := h
  refine ⟨hU, fun h6 H ip => ?_, fun h6 H a ha ip => ?_⟩
  · rw [hX h6 H ip]
    exact Finset.sum_congr rfl fun k _ => refVal_stable hcall F f e _ _ _ hF hf
  · rw [hS h6 H a ha ip]
    exact refVal_stable hcall F f e _ _ _ hF hf

theorem stOk_insert {t : Tmpl} {lays : Array Lay} {pt : FoldPt} {s : ℕ} {st : SplitSt} (hst : StOk t lays pt s st)
    {e β' : ℕ} {as' : Array ℕ} {T : Tab}
    (hT : ∀ f, e < f → Spec t lays pt s f e β' as' 0 (lays.getD e default).palign.log2 T) (n1 n2 : ℕ) :
    StOk t lays pt s { st with tabs := st.tabs.insert (e, β', as') T, tables := n1, tableWork := n2 } := by
  refine ⟨hst.1, fun e2 b2 a2 T2 hg hp f hf => ?_⟩
  dsimp only at hg
  rw [HashMap.get?_eq_getElem?, HashMap.getElem?_insert] at hg
  by_cases hk : (e, β', as') = (e2, b2, a2)
  · rw [ite_eq_left (by simpa using hk)] at hg
    simp only [Prod.mk.injEq] at hk
    obtain ⟨rfl, rfl, rfl⟩ := hk
    cases hg
    exact hT f hf
  · rw [ite_eq_right (by simpa using hk)] at hg
    exact hst.2 e2 b2 a2 T2 (by rwa [HashMap.get?_eq_getElem?]) hp f hf

theorem cachedTab_ok {t : Tmpl} {lays : Array Lay} {multi : Array Bool} {pt : FoldPt} {s : ℕ}
    (hcall : ∀ d i, i < (t.defs.getD d default).calls.size → ((t.defs.getD d default).calls.getD i default).def_ < d)
    {F : ℕ} (hw : WalkOk t lays multi pt s F) {e β' : ℕ} {as' : Array ℕ} {st st' : SplitSt} {T : Tab}
    (h : (cachedTab (walk t lays multi pt F) e β' as' (lays.getD e default).palign.log2).run st = .ok (T, st'))
    (hst : StOk t lays pt s st) (he : e < F) (hpc : Pre pt β' (lays.getD e default).palign.log2 as') :
    StOk t lays pt s st' ∧ Spec t lays pt s F e β' as' 0 (lays.getD e default).palign.log2 T := by
  unfold cachedTab at h
  obtain ⟨s0, st0, hg, h1⟩ := sv_bind_ok h
  clear h
  obtain ⟨rfl, rfl⟩ := sv_get_ok hg
  split at h1
  · rename_i T' hk
    obtain ⟨rfl, rfl⟩ := sv_pure_ok h1
    exact ⟨hst, hst.2 _ _ _ _ hk hpc F he⟩
  · obtain ⟨T2, st4, hw1, h⟩ := sv_bind_ok h1
    obtain ⟨u, st5, hmod, h⟩ := sv_bind_ok h
    have hmod := sv_modify_ok hmod
    subst hmod
    obtain ⟨rfl, rfl⟩ := sv_pure_ok h
    obtain ⟨hst4, hsp⟩ := hw e β' as' 0 _ _ T2 st4 hw1 hst he hpc
    exact ⟨stOk_insert hst4 (fun f hf => spec_stable hcall hsp he hf) _ _, hsp⟩

/-- **A callee's table**, re-keyed from its cached table or walked directly: its class's at the caller's chunk origin. -/
theorem walkChild_ok {t : Tmpl} {lays : Array Lay} {multi : Array Bool} {pt : FoldPt} {s : ℕ}
    (hT : pt.rhoT = Tmpl.halves pt.rho) (hR : pt.rT = Tmpl.halves pt.r)
    (hcall : ∀ d i, i < (t.defs.getD d default).calls.size → ((t.defs.getD d default).calls.getD i default).def_ < d)
    {F : ℕ} (hw : WalkOk t lays multi pt s F) {e β' : ℕ} {as' : Array ℕ} {oc al' : ℕ} {st st' : SplitSt}
    {ch : Tab} {rm cm : ℕ → ℕ × F128}
    (h : (walkChild lays multi pt (walk t lays multi pt F) e β' as' oc al').run st = .ok ((ch, rm, cm), st'))
    (hst : StOk t lays pt s st) (he : e < F) (hpre : Pre pt β' al' as') :
    StOk t lays pt s st' ∧ ∃ al'' off'', Spec t lays pt s F e β' as' off'' al'' ch ∧ ∀ H, ∃ Hc,
      Hc * 2 ^ al'' + off'' = H * 2 ^ al' + oc ∧
      (∀ c, of (rm c).2 * hiE pt (β' + al' - 6) H (rm c).1 = hiE pt (β' + al'' - 6) Hc c) ∧
      (∀ c, of (cm c).2 * hiG pt (β' + al' - 6) H (cm c).1 = hiG pt (β' + al'' - 6) Hc c) := by
  unfold walkChild at h
  dsimp only at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hc
    obtain ⟨⟨⟨_, hoc⟩, hle⟩, h6c⟩ := hc
    have hpc : Pre pt β' (lays.getD e default).palign.log2 as' :=
      ⟨h6c, by have := hpre.2.1; omega, by have := hpre.2.2.1; omega, hpre.2.2.2⟩
    obtain ⟨T, st1, hm, h⟩ := sv_bind_ok h
    obtain ⟨mρ, st2, h1, h⟩ := sv_bind_ok h
    obtain ⟨mr, st3, h2, h⟩ := sv_bind_ok h
    obtain ⟨hx, rfl⟩ := sv_pure_ok h
    simp only [Prod.mk.injEq] at hx
    obtain ⟨rfl, rfl, rfl⟩ := hx
    obtain ⟨hst1, hsp⟩ := cachedTab_ok hcall hw hm hst he hpc
    obtain ⟨rfl, hH2, ht2⟩ := half_ok h1 hst1.1
    obtain ⟨rfl, hH3, ht3⟩ := half_ok h2 hH2
    have hA : 2 ^ al' = 2 ^ (al' - (lays.getD e default).palign.log2) * 2 ^ (lays.getD e default).palign.log2 := by
      rw [← Nat.pow_add, Nat.sub_add_cancel hle]
    have hoc' : oc / 2 ^ (lays.getD e default).palign.log2 * 2 ^ (lays.getD e default).palign.log2 = oc :=
      Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hoc)
    refine ⟨⟨hH3, by rw [ht3, ht2]; exact hst1.2⟩, _, 0, hsp, fun H =>
      ⟨H * 2 ^ (al' - (lays.getD e default).palign.log2) + oc / 2 ^ (lays.getD e default).palign.log2,
        by rw [hA, Nat.add_zero, Nat.add_mul, Nat.mul_assoc, hoc'], fun c => ?_, fun c => ?_⟩⟩
    · simp only [rekeyAt, ↓reduceIte]
      unfold hiE
      rw [hT, rekey pt.rho hle (by have := hpre.2.1; omega) hoc H c,
        show β' + (lays.getD e default).palign.log2 - 6 + (al' - (lays.getD e default).palign.log2) = β' + al' - 6 by
          omega]
    · simp only [rekeyAt, Bool.false_eq_true, ↓reduceIte]
      unfold hiG
      rw [hR, rekey pt.r hle (by have := hpre.2.2.1; omega) hoc H c,
        show β' + (lays.getD e default).palign.log2 - 6 + (al' - (lays.getD e default).palign.log2) = β' + al' - 6 by
          omega]
  · obtain ⟨T, st1, hw1, h⟩ := sv_bind_ok h
    obtain ⟨hx, rfl⟩ := sv_pure_ok h
    simp only [Prod.mk.injEq] at hx
    obtain ⟨rfl, rfl, rfl⟩ := hx
    obtain ⟨hst1, hsp⟩ := hw e β' as' oc al' _ T _ hw1 hst he hpre
    refine ⟨hst1, al', oc, hsp, fun H => ⟨H, rfl, fun c => ?_, fun c => ?_⟩⟩ <;>
      simp only [ident] <;> rw [← one_eq, one_mul]

/-! ## A call's branches, valued -/

theorem hsum_empty {κ : Type} [BEq κ] [LawfulBEq κ] [Hashable κ] [LawfulHashable κ] [DecidableEq κ] (f : κ → GF128) :
    hsum ({} : HashMap κ F128) f = 0 := by
  simp [hsum]

theorem tabExp_empty (pt : FoldPt) (s L H : ℕ) (iv : ℕ → ℕ → GF128) : tabExp pt s L H iv {} = 0 := by
  unfold tabExp
  simp only [hsum_empty, add_zero]

theorem tabSum_empty (pt : FoldPt) (L H : ℕ) (gl gc : GF128) (iv : ℕ → GF128) : tabSum pt L H gl gc iv {} = 0 := by
  unfold tabSum
  simp only [hsum_empty, mul_zero, add_zero]

theorem mem_of_toList {κ : Type} [BEq κ] [LawfulBEq κ] [Hashable κ] [LawfulHashable κ] {m : HashMap κ F128}
    {e : κ × F128} (he : e ∈ m.toList) : e.1 ∈ m := by
  have h := (HashMap.mem_toList_iff_getElem?_eq_some).1 he
  exact HashMap.mem_iff_isSome_getElem?.2 (by rw [h]; rfl)

theorem tabExp_congr_U (pt : FoldPt) (s L H : ℕ) (iv iv' : ℕ → ℕ → GF128) (T : Tab)
    (h : ∀ x, x ∈ T.U → iv x.1 x.2.1 = iv' x.1 x.2.1) : tabExp pt s L H iv T = tabExp pt s L H iv' T := by
  unfold tabExp hsum
  congr 1
  refine congrArg List.sum (List.map_congr_left fun e he => ?_)
  dsimp only
  rw [h e.1 (mem_of_toList he)]

theorem tabSum_congr_U (pt : FoldPt) (L H : ℕ) (gl gc : GF128) (iv iv' : ℕ → GF128) (T : Tab)
    (h : ∀ x, x ∈ T.U → iv x.1 = iv' x.1) : tabSum pt L H gl gc iv T = tabSum pt L H gl gc iv' T := by
  unfold tabSum hsum
  congr 1
  refine congrArg List.sum (List.map_congr_left fun e he => ?_)
  dsimp only
  rw [h e.1 (mem_of_toList he)]

/-- `rm`, `cm` re-key a child's chunks at origin `Hc` (high factors from `Lc`) into its caller's at `H` (from `L`). -/
def ReKey (pt : FoldPt) (L H Lc Hc : ℕ) (rm cm : ℕ → ℕ × F128) : Prop :=
  (∀ c, of (rm c).2 * hiE pt L H (rm c).1 = hiE pt Lc Hc c) ∧ (∀ c, of (cm c).2 * hiG pt L H (cm c).1 = hiG pt Lc Hc c)

/-- A child's own entries, added re-keyed, are its own entries at its origin. -/
theorem addChild_exp (pt : FoldPt) (s L H Lc Hc : ℕ) (iv : ℕ → ℕ → GF128) (o ch : Tab) (rm cm : ℕ → ℕ × F128)
    (ls lc : F128) (hk : ReKey pt L H Lc Hc rm cm) :
    tabExp pt s L H iv (o.addChild ch rm cm ls lc) = tabExp pt s L H iv o +
      (hsum ch.P (fun e => if e.2.2 = s then hiE pt Lc Hc e.1 * hiG pt Lc Hc e.2.1 else 0) +
        hsum ch.PC (fun e => if e.2 = s then hiE pt Lc Hc e.1 else 0)) := by
  rw [addChild_tabExp]
  simp only [hk.1, hk.2]

theorem addChild_sum (pt : FoldPt) (L H Lc Hc : ℕ) (gl gc : GF128) (iv : ℕ → GF128) (o ch : Tab)
    (rm cm : ℕ → ℕ × F128) (ls lc : F128) (hk : ReKey pt L H Lc Hc rm cm) :
    tabSum pt L H gl gc iv (o.addChild ch rm cm ls lc) = tabSum pt L H gl gc iv o +
      (gl * of ls * hsum ch.S (fun e => hiE pt Lc Hc e.1 * hiG pt Lc Hc e.2) +
        gc * of lc * hsum ch.C (fun c => hiE pt Lc Hc c)) := by
  rw [addChild_tabSum]
  simp only [hk.1, hk.2]

/-- **A copy placed directly** (`ms = 1`), explicit: the child's table at its origin, its inputs read through `ref`. -/
theorem child_exp1 (D : Def) (pt : FoldPt) (s L H Lc Hc A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ)
    (iv : ℕ → ℕ → GF128) (c : Call) (o ch : Tab) (j : ℕ) (rm cm : ℕ → ℕ × F128) (hk : ReKey pt L H Lc Hc rm cm) :
    tabExp pt s L H iv (Tab.bind (Tab.ref D pt false A B as gL pos) c (o.addChild ch rm cm 1 1) j ch rm
        fun kc _ => [(kc, 1)]) = tabExp pt s L H iv o +
      tabExp pt s Lc Hc (fun t' kc => refX D pt s L H A B as gL pos iv kc ((c.bound id j).toArray.getD t' 0)) ch := by
  rw [bind_tabExp, addChild_exp pt s L H Lc Hc iv o ch rm cm 1 1 hk]
  unfold tabExp hsum
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero]
  have e1 : ∀ e : (ℕ × ℕ × ℕ) × F128, of (1 * e.2 * (rm e.1.2.2).2) * hiE pt L H (rm e.1.2.2).1 *
      refX D pt s L H A B as gL pos iv e.1.2.1 ((c.bound id j).toArray.getD e.1.1 0) =
      of e.2 * (hiE pt Lc Hc e.1.2.2 * refX D pt s L H A B as gL pos iv e.1.2.1 ((c.bound id j).toArray.getD e.1.1 0)) := by
    intro e
    rw [← hk.1]
    simp only [of_mul, ← one_eq]
    ring
  simp only [e1]
  ring

theorem child_sum1 (D : Def) (pt : FoldPt) (L H Lc Hc A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ)
    (gl gc : GF128) (iv : ℕ → GF128) (c : Call) (o ch : Tab) (j : ℕ) (rm cm : ℕ → ℕ × F128)
    (hk : ReKey pt L H Lc Hc rm cm) :
    tabSum pt L H gl gc iv (Tab.bind (Tab.ref D pt true A B as gL pos) c (o.addChild ch rm cm 1 1) j ch rm
        fun kc _ => [(kc, 1)]) = tabSum pt L H gl gc iv o +
      tabSum pt Lc Hc gl gc (fun t' => refS D pt L H A gL pos gl gc iv ((c.bound id j).toArray.getD t' 0)) ch := by
  rw [bind_tabSum, addChild_sum pt L H Lc Hc gl gc iv o ch rm cm 1 1 hk]
  unfold tabSum hsum
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero]
  have e1 : ∀ e : (ℕ × ℕ × ℕ) × F128, of (1 * e.2 * (rm e.1.2.2).2) * hiE pt L H (rm e.1.2.2).1 *
      refS D pt L H A gL pos gl gc iv ((c.bound id j).toArray.getD e.1.1 0) =
      of e.2 * (hiE pt Lc Hc e.1.2.2 * refS D pt L H A gL pos gl gc iv ((c.bound id j).toArray.getD e.1.1 0)) := by
    intro e
    rw [← hk.1]
    simp only [of_mul, ← one_eq]
    ring
  simp only [e1, ← one_eq, mul_one]
  ring

/-- **A run of lanes below an explicit class** (`β + m < 6`): the child's copy `kc` is lane `kc mod count` of copy
`⌊kc / count⌋`, bound by that lane. -/
theorem child_expL (D : Def) (pt : FoldPt) (s L H Lc Hc A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ)
    (iv : ℕ → ℕ → GF128) (c : Call) (o ch : Tab) (count nk : ℕ) (rm cm : ℕ → ℕ × F128)
    (hk : ReKey pt L H Lc Hc rm cm) (hU : ∀ x, x ∈ ch.U → x.2.1 < nk * count) :
    tabExp pt s L H iv (Tab.binds (Tab.bind (Tab.ref D pt false A B as gL pos) c) (o.addChild ch rm cm 0 0) count ch rm
        fun kc j => if kc % count == j then [(kc / count, 1)] else []) = tabExp pt s L H iv o +
      tabExp pt s Lc Hc (fun t' kc => refX D pt s L H A B as gL pos iv (kc / count)
        ((c.bound id (kc % count)).toArray.getD t' 0)) ch := by
  rw [binds_val _ _ ch rm _ _ (fun o j => bind_tabExp D pt s L H A B as gL pos iv c o j ch rm _),
    addChild_exp pt s L H Lc Hc iv o ch rm cm 0 0 hk, ← list_sum_comm]
  unfold tabExp hsum
  have e1 : ∀ e ∈ ch.U.toList, (∑ j ∈ Finset.range count,
      (((if e.1.2.1 % count == j then [(e.1.2.1 / count, (1 : F128))] else []).map fun kf =>
        of (kf.2 * e.2 * (rm e.1.2.2).2) * hiE pt L H (rm e.1.2.2).1 *
          refX D pt s L H A B as gL pos iv kf.1 ((c.bound id j).toArray.getD e.1.1 0)).sum)) =
      of e.2 * (hiE pt Lc Hc e.1.2.2 * refX D pt s L H A B as gL pos iv (e.1.2.1 / count)
        ((c.bound id (e.1.2.1 % count)).toArray.getD e.1.1 0)) := by
    intro e he
    have hc : 0 < count := Nat.pos_of_ne_zero fun h0 => by
      have := hU e.1 (mem_of_toList he); rw [h0, Nat.mul_zero] at this; omega
    rw [Finset.sum_eq_single (e.1.2.1 % count) (fun j _ hj => by simp [Ne.symm hj])
      (fun h => absurd (Finset.mem_range.2 (Nat.mod_lt _ hc)) h)]
    simp only [beq_self_eq_true, ↓reduceIte, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero]
    rw [← hk.1]
    simp only [of_mul, ← one_eq]
    ring
  rw [List.map_congr_left e1]
  ring

/-- **A run of lanes below a summed class**: the child's summed entries by the lane sums `ls`, `lc`, its inputs bound
lane by lane with weight `eq(ρ[β−6 ..], j)`. -/
theorem child_sumL (D : Def) (pt : FoldPt) (L H Lc Hc A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ)
    (gl gc : GF128) (iv : ℕ → GF128) (c : Call) (o ch : Tab) (count : ℕ) (rm cm : ℕ → ℕ × F128) (ls lc : F128)
    (E : ℕ → F128) (hk : ReKey pt L H Lc Hc rm cm) :
    tabSum pt L H gl gc iv (Tab.binds (Tab.bind (Tab.ref D pt true A B as gL pos) c) (o.addChild ch rm cm ls lc)
        count ch rm fun _ j => [(0, E j)]) = tabSum pt L H gl gc iv o +
      (gl * of ls * hsum ch.S (fun e => hiE pt Lc Hc e.1 * hiG pt Lc Hc e.2) +
        gc * of lc * hsum ch.C (fun c => hiE pt Lc Hc c) +
        ∑ j ∈ Finset.range count, of (E j) * hsum ch.U (fun e => hiE pt Lc Hc e.2.2 *
          refS D pt L H A gL pos gl gc iv ((c.bound id j).toArray.getD e.1 0))) := by
  rw [binds_val _ _ ch rm _ _ (fun o j => bind_tabSum D pt L H A B as gL pos gl gc iv c o j ch rm _),
    addChild_sum pt L H Lc Hc gl gc iv o ch rm cm ls lc hk]
  unfold hsum
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero]
  have e1 : ∀ j, ∀ e : (ℕ × ℕ × ℕ) × F128, of (E j * e.2 * (rm e.1.2.2).2) * hiE pt L H (rm e.1.2.2).1 *
      refS D pt L H A gL pos gl gc iv ((c.bound id j).toArray.getD e.1.1 0) =
      of (E j) * (of e.2 * (hiE pt Lc Hc e.1.2.2 * refS D pt L H A gL pos gl gc iv ((c.bound id j).toArray.getD e.1.1 0))) := by
    intro j e
    rw [← hk.1]
    simp only [of_mul]
    ring
  simp only [e1, List.sum_map_mul_left]
  ring

/-- **A run of lanes from an explicit class into a summed one**: the child's summed entries spread by `W`, `EL`, its
inputs bound for each copy `k` and lane `j` with weight `E k j`. -/
theorem child_expW (D : Def) (pt : FoldPt) {s : ℕ} (hs : s < 64) (L H Lc Hc A B : ℕ) (as : Array ℕ) (gL : Half)
    (pos : ℕ → ℕ) (iv : ℕ → ℕ → GF128) (c : Call) (o ch : Tab) (count nk n : ℕ) (rm cm : ℕ → ℕ × F128)
    (W : Array F128) (EL : F128) (E : ℕ → ℕ → F128) (hk : ReKey pt L H Lc Hc rm cm) :
    tabExp pt s L H iv (Tab.binds (Tab.bind (Tab.ref D pt false A B as gL pos) c) (Tab.slots pt W EL n o ch rm cm)
        count ch rm fun _ j => (List.range nk).map fun k => (k, E k j)) = tabExp pt s L H iv o +
      (of (W.getD s 0) * hsum ch.S (fun e => hiE pt Lc Hc e.1 * hiG pt Lc Hc e.2) +
        of EL * Gs pt s pt.qc * hsum ch.C (fun c => hiE pt Lc Hc c) +
        ∑ j ∈ Finset.range count, ∑ k ∈ Finset.range nk, of (E k j) * hsum ch.U (fun e => hiE pt Lc Hc e.2.2 *
          refX D pt s L H A B as gL pos iv k ((c.bound id j).toArray.getD e.1 0))) := by
  rw [binds_val _ _ ch rm _ _ (fun o j => bind_tabExp D pt s L H A B as gL pos iv c o j ch rm _),
    slots_tabExp pt hs L H iv W EL n o ch rm cm]
  simp only [hk.1, hk.2]
  have e1 : ∀ j, ∀ e : (ℕ × ℕ × ℕ) × F128, (((List.range nk).map fun k => (k, E k j)).map fun kf =>
      of (kf.2 * e.2 * (rm e.1.2.2).2) * hiE pt L H (rm e.1.2.2).1 *
        refX D pt s L H A B as gL pos iv kf.1 ((c.bound id j).toArray.getD e.1.1 0)).sum =
      ∑ k ∈ Finset.range nk, of (E k j) * (of e.2 * (hiE pt Lc Hc e.1.2.2 *
        refX D pt s L H A B as gL pos iv k ((c.bound id j).toArray.getD e.1.1 0))) := by
    intro j e
    rw [List.map_map, list_range_sum]
    refine Finset.sum_congr rfl fun k _ => ?_
    simp only [Function.comp]
    rw [← hk.1]
    simp only [of_mul]
    ring
  unfold hsum
  simp only [e1]
  have e2 : ∀ j, (ch.U.toList.map fun e => ∑ k ∈ Finset.range nk, of (E k j) * (of e.2 * (hiE pt Lc Hc e.1.2.2 *
      refX D pt s L H A B as gL pos iv k ((c.bound id j).toArray.getD e.1.1 0)))).sum =
      ∑ k ∈ Finset.range nk, of (E k j) * (ch.U.toList.map fun e => of e.2 * (hiE pt Lc Hc e.1.2.2 *
        refX D pt s L H A B as gL pos iv k ((c.bound id j).toArray.getD e.1.1 0))).sum := by
    intro j
    rw [list_sum_comm]
    exact Finset.sum_congr rfl fun k _ => List.sum_map_mul_left _ _ _
  simp only [e2]
  ring

end FlockSoundness.TmplFold
