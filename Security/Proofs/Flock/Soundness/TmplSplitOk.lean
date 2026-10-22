import Proofs.Flock.Soundness.TmplSplitStep

/-!
# What a walked table means (PRF-133, part 3, stage 3)

`Spec`: a class's table, placed at any chunk origin `H`, values (`tabExp`, `tabSum`) to its copies' `refVal`s, as
`ExpOk` and `SumOk` say of `foldDef`'s `Out`; its open inputs are keyed by an input of its Definition and one of its
copies (`UOk`). `StOk`: every cached table means its class's, at every depth past its Definition.

- `rows_val`, `binds_val`, `slots_tabExp`: a Definition's own rows, a call's bindings and a run of lanes spread over the
  64 slots, valued.
- `walkChild_ok`: a callee's table, re-keyed or walked directly, means its class's at the caller's chunk origin.
-/

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

theorem rows_U (D : Def) (pt : FoldPt) (summed : Bool) (A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ)
    (rowLow : ℕ → ℕ → F128) (alpha : F128) (nk A' : ℕ) (rows : Array (ℕ × Array ℕ × Array ℕ)) (o : Tab)
    (ho : UOk D.ins nk o) :
    UOk D.ins nk (Tab.rows (Tab.ref D pt summed A B as gL pos) rowLow alpha nk A' pos rows o) := by
  unfold Tab.rows
  rw [← Array.foldl_toList]
  refine foldl_inv (UOk D.ins nk) _ _ o (fun o' r _ hq => ?_) ho
  obtain ⟨x, ra, rb⟩ := r
  dsimp only
  have hw : ∀ (T : Tab) (n : ℕ), UOk D.ins nk T → UOk D.ins nk { T with work := n } := fun _ _ h => h
  refine hw _ _ (foldl_inv (UOk D.ins nk) _ _ o' (fun o'' k hk hq' => ?_) hq)
  have hk := List.mem_range.mp hk
  dsimp only
  rw [← Array.foldl_toList, ← Array.foldl_toList]
  exact foldl_inv (UOk D.ins nk) _ _ _ (fun _ _ _ h => ref_U D pt summed A B as gL pos _ hk _ _ _ h)
    (foldl_inv (UOk D.ins nk) _ _ _ (fun _ _ _ h => ref_U D pt summed A B as gL pos _ hk _ _ _ h) hq')

/-! ## Values of the walk's pure steps -/

theorem foldl_val {α δ : Type} (V : δ → GF128) (f : δ → α → δ) (G : α → GF128) (h : ∀ o a, V (f o a) = V o + G a) :
    ∀ (l : List α) (o : δ), V (l.foldl f o) = V o + (l.map G).sum
  | [], o => by simp
  | a :: l, o => by rw [List.foldl_cons, foldl_val V f G h l, h, List.map_cons, List.sum_cons, add_assoc]

/-- **A Definition's own rows**, through a `ref` that adds `of w · Z k cx v`: each row's references, copy by copy. -/
theorem rows_val (V : Tab → GF128) (hW : ∀ (o : Tab) (n : ℕ), V { o with work := n } = V o)
    (ref : Tab → ℕ → ℕ → F128 → ℕ → Tab) (Z : ℕ → ℕ → ℕ → GF128)
    (hV : ∀ o k cx w v, V (ref o k cx w v) = V o + of w * Z k cx v)
    (rowLow : ℕ → ℕ → F128) (alpha : F128) (nk A : ℕ) (pos : ℕ → ℕ) (rows : Array (ℕ × Array ℕ × Array ℕ))
    (o : Tab) :
    V (Tab.rows ref rowLow alpha nk A pos rows o) = V o + (rows.toList.map fun r => ∑ k ∈ Finset.range nk,
      ((r.2.1.toList.map fun v => of (alpha * rowLow k (pos r.1)) * Z k (pos r.1 / A) v).sum +
        (r.2.2.toList.map fun v => of (rowLow k (pos r.1)) * Z k (pos r.1 / A) v).sum)).sum := by
  unfold Tab.rows
  rw [← Array.foldl_toList]
  refine foldl_val V _ (fun r => ∑ k ∈ Finset.range nk,
      ((r.2.1.toList.map fun v => of (alpha * rowLow k (pos r.1)) * Z k (pos r.1 / A) v).sum +
        (r.2.2.toList.map fun v => of (rowLow k (pos r.1)) * Z k (pos r.1 / A) v).sum)) (fun o' r => ?_) _ o
  obtain ⟨x, ra, rb⟩ := r
  dsimp only
  rw [hW, ← list_range_sum]
  refine foldl_val V _ (fun k => (ra.toList.map fun v => of (alpha * rowLow k (pos x)) * Z k (pos x / A) v).sum +
      (rb.toList.map fun v => of (rowLow k (pos x)) * Z k (pos x / A) v).sum) (fun o'' k => ?_) _ o'
  dsimp only
  rw [← Array.foldl_toList, ← Array.foldl_toList, foldl_val V _ _ (fun o v => hV o k (pos x / A) _ v),
    foldl_val V _ _ (fun o v => hV o k (pos x / A) _ v), add_assoc]

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
  · simp [hz]

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
  rw [fold_val (fun mp => hsum mp (fun e => if e.2.2 = s then hiE pt L H e.1 * hiG pt L H e.2.1 else 0)) _
      (fun e v => (((List.range 64).filter fun s' => !(W.getD s' 0).isZero).map fun s' =>
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
    generalize rm cx = p
    generalize cm cy = q
    obtain ⟨px, fx⟩ := p
    obtain ⟨py, fy⟩ := q
    exact foldl_val (fun mp => hsum mp (fun e => if e.2.2 = s then hiE pt L H e.1 * hiG pt L H e.2.1 else 0)) _ _
      (fun mp s' => hsum_bump _ _ _ _) _ mp

/-- The slot weights `W` the walk spreads a run of lanes by: `W[s]` is `Σ E(a) · eq(r, ⌊a/64⌋)` over the lanes at `s`. -/
theorem slotW_get (pt : FoldPt) {s : ℕ} (hs : s < 64) (β' : ℕ) (lows : List ℕ) :
    of ((lows.foldl (fun W a => addAt W a (eLow pt β' a * Tmpl.eqAt pt.r 0 (β' - 6) (a / 64)))
      (Array.replicate 64 0)).getD s 0) = (lows.map fun a => of (eLow pt β' a) * gLow pt s β' a).sum := by
  have h := foldl_addAt_get lows (fun a => a) (fun a => eLow pt β' a * Tmpl.eqAt pt.r 0 (β' - 6) (a / 64))
    (Array.replicate 64 0) (s := s) (by simp [hs])
  simp only at h
  rw [h, getD_replicate, of_zero', zero_add]
  congr 1
  refine List.map_congr_left fun a _ => ?_
  unfold gLow
  split_ifs <;> simp [of_mul]

theorem slotEL (pt : FoldPt) (β' : ℕ) : ∀ (lows : List ℕ) (x : F128),
    of (lows.foldl (fun EL a => EL + eLow pt β' a) x) = of x + (lows.map fun a => of (eLow pt β' a)).sum
  | [], x => by simp
  | a :: l, x => by rw [List.foldl_cons, slotEL pt β' l, of_add, List.map_cons, List.sum_cons, add_assoc]

end FlockSoundness.TmplFold
