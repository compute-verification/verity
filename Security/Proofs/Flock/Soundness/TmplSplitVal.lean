import Proofs.Flock.Soundness.TmplSplitTab

/-!
# A class's table, valued at a chunk origin

`walk` folds a class at local offset `off` in chunks of `2^al`; placed at chunk origin `H` (so at `h = H · 2^al + off`
in `foldDef`'s terms), an entry at row chunk `cx` and column chunk `cy` has high factors `eq(ρ[L ..], H + cx)` and
`eq(r[L ..], H + cy)`, `L = β + al − 6` (`hiE`, `hiG`). `tabExp` and `tabSum` value a table as `expVal` and `sumVal`
value an `Out`.

- `e_chunk`, `g_chunk` (explicit) and `e_chunk_sum`, `g_chunk_sum` (summed): a position's `E` and slot weights are the
  low factors `walk` reads from its `eq` halves times the high factors.
- `ref_tabExp`, `ref_tabSum`: `Tab.ref` adds its reference's term.
-/

open Flock Flock.Tmpl FlockLevel3 GF128 Std

namespace FlockSoundness.TmplFold

/-- Chunk `H + c`'s high `E` factor, `eq(ρ[L ..], H + c)`. -/
noncomputable def hiE (pt : FoldPt) (L H c : ℕ) : GF128 := of (Tmpl.eqTail pt.rhoT L (H + c))

/-- Chunk `H + c`'s high slot factor, `eq(r[L ..], H + c)`. -/
noncomputable def hiG (pt : FoldPt) (L H c : ℕ) : GF128 := of (Tmpl.eqTail pt.rT L (H + c))

/-- An explicit class's table at chunk origin `H`, slot `s`: its own entries at `s` by their chunks' high factors, the
constant's by its row chunk's, and copy `k`'s open input `t` by `iv t k`. -/
noncomputable def tabExp (pt : FoldPt) (s L H : ℕ) (iv : ℕ → ℕ → GF128) (T : Tab) : GF128 :=
  hsum T.P (fun e => if e.2.2 = s then hiE pt L H e.1 * hiG pt L H e.2.1 else 0) +
    hsum T.PC (fun e => if e.2 = s then hiE pt L H e.1 else 0) +
    hsum T.U (fun e => hiE pt L H e.2.2 * iv e.1 e.2.1)

/-- A summed class's table at chunk origin `H`: its own entries by `gl`, the constant's by `gc`, open input `t` by
`iv t`, each by its chunks' high factors. -/
noncomputable def tabSum (pt : FoldPt) (L H : ℕ) (gl gc : GF128) (iv : ℕ → GF128) (T : Tab) : GF128 :=
  gl * hsum T.S (fun e => hiE pt L H e.1 * hiG pt L H e.2) + gc * hsum T.C (fun c => hiE pt L H c) +
    hsum T.U (fun e => hiE pt L H e.2.2 * iv e.1)

/-! ## A position's factors, split at its chunk -/

theorem chunk_pos (β al a H P : ℕ) :
    a + 2 ^ β * (H * 2 ^ al + P) = (a + 2 ^ β * (P % 2 ^ al)) + 2 ^ (β + al) * (H + P / 2 ^ al) := by
  have h := Nat.mod_add_div P (2 ^ al)
  rw [Nat.pow_add]
  calc a + 2 ^ β * (H * 2 ^ al + P) = a + 2 ^ β * (H * 2 ^ al + (P % 2 ^ al + 2 ^ al * (P / 2 ^ al))) := by rw [h]
    _ = _ := by ring

theorem chunk_low {β al a P : ℕ} (ha : a < 2 ^ β) : a + 2 ^ β * (P % 2 ^ al) < 2 ^ (β + al) :=
  lane_lt ha (Nat.mod_lt _ (Nat.two_pow_pos _))

theorem e_chunk (pt : FoldPt) (hT : pt.rhoT = Tmpl.halves pt.rho) {β al a : ℕ} (h6 : 6 ≤ β + al)
    (hL : β + al - 6 ≤ pt.rho.size) (ha : a < 2 ^ β) (H P : ℕ) :
    of (pt.e (a + 2 ^ β * (H * 2 ^ al + P))) =
      of (pt.lw.getD ((a + 2 ^ β * (P % 2 ^ al)) % 64) 0 *
        (eqHalf pt.rho 0 (β + al - 6)).at ((a + 2 ^ β * (P % 2 ^ al)) / 64)) * hiE pt (β + al - 6) H (P / 2 ^ al) := by
  have hq := chunk_low (al := al) (P := P) ha
  unfold FoldPt.e hiE
  rw [chunk_pos, split_mod _ _ _ h6, split_div _ _ _ h6, hT, of_mul, eqTail_split' pt.rho _ _ _ hL (split_lo _ _ h6 hq)]
  simp only [of_mul]
  rw [eqHalf_at _ _ _ _ (split_lo _ _ h6 hq)]
  ring

theorem g_chunk (pt : FoldPt) (hR : pt.rT = Tmpl.halves pt.r) {s β al a : ℕ} (h6 : 6 ≤ β + al)
    (hL : β + al - 6 ≤ pt.r.size) (ha : a < 2 ^ β) (H P : ℕ) :
    Gs pt s (a + 2 ^ β * (H * 2 ^ al + P)) =
      (if (a + 2 ^ β * (P % 2 ^ al)) % 64 = s then
        of ((eqHalf pt.r 0 (β + al - 6)).at ((a + 2 ^ β * (P % 2 ^ al)) / 64)) else 0) *
        hiG pt (β + al - 6) H (P / 2 ^ al) := by
  have hq := chunk_low (al := al) (P := P) ha
  unfold Gs FoldPt.g hiG
  rw [chunk_pos, split_mod _ _ _ h6, split_div _ _ _ h6]
  split_ifs
  · rw [hR, eqTail_split' pt.r _ _ _ hL (split_lo _ _ h6 hq), eqHalf_at _ _ _ _ (split_lo _ _ h6 hq)]
  · rw [zero_mul]

theorem e_chunk_sum (pt : FoldPt) (hT : pt.rhoT = Tmpl.halves pt.rho) {β al a : ℕ} (h6 : 6 ≤ β)
    (hL : β + al - 6 ≤ pt.rho.size) (ha : a < 2 ^ β) (H P : ℕ) :
    of (pt.e (a + 2 ^ β * (H * 2 ^ al + P))) =
      of (eLow pt β a) * of ((eqHalf pt.rho (β - 6) al).at (P % 2 ^ al)) * hiE pt (β + al - 6) H (P / 2 ^ al) := by
  have hP := Nat.mod_add_div P (2 ^ al)
  unfold hiE
  rw [e_split pt hT h6 (by omega) ha, hT,
    show H * 2 ^ al + P = P % 2 ^ al + 2 ^ al * (H + P / 2 ^ al) by
      calc H * 2 ^ al + P = H * 2 ^ al + (P % 2 ^ al + 2 ^ al * (P / 2 ^ al)) := by rw [hP]
        _ = _ := by ring,
    eqTail_split_at pt.rho _ _ _ _ (by omega) (Nat.mod_lt _ (Nat.two_pow_pos _)),
    eqHalf_at _ _ _ _ (Nat.mod_lt _ (Nat.two_pow_pos _)), show β - 6 + al = β + al - 6 by omega]
  ring

theorem g_chunk_sum (pt : FoldPt) (hR : pt.rT = Tmpl.halves pt.r) {s β al a : ℕ} (h6 : 6 ≤ β)
    (hL : β + al - 6 ≤ pt.r.size) (ha : a < 2 ^ β) (H P : ℕ) :
    Gs pt s (a + 2 ^ β * (H * 2 ^ al + P)) =
      gLow pt s β a * of ((eqHalf pt.r (β - 6) al).at (P % 2 ^ al)) * hiG pt (β + al - 6) H (P / 2 ^ al) := by
  have hP := Nat.mod_add_div P (2 ^ al)
  unfold hiG
  rw [g_split pt hR h6 (by omega) ha, hR,
    show H * 2 ^ al + P = P % 2 ^ al + 2 ^ al * (H + P / 2 ^ al) by
      calc H * 2 ^ al + P = H * 2 ^ al + (P % 2 ^ al + 2 ^ al * (P / 2 ^ al)) := by rw [hP]
        _ = _ := by ring,
    eqTail_split_at pt.r _ _ _ _ (by omega) (Nat.mod_lt _ (Nat.two_pow_pos _)),
    eqHalf_at _ _ _ _ (Nat.mod_lt _ (Nat.two_pow_pos _)), show β - 6 + al = β + al - 6 by omega]
  ring

/-! ## A reference's step -/

/-- An explicit reference's slot-`s` weight: an own column's slot factor, an input's `iv`, the constant's slot weight. -/
noncomputable def refX (D : Def) (pt : FoldPt) (s L H A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ)
    (iv : ℕ → ℕ → GF128) (k v : ℕ) : GF128 :=
  if v < D.cols then
    (if (as.getD k 0 + B * (pos v % A)) % 64 = s then
      of (gL.at ((as.getD k 0 + B * (pos v % A)) / 64)) * hiG pt L H (pos v / A) else 0)
  else if v < D.cols + D.ins then iv (v - D.cols) k
  else if pt.qc % 64 = s then of (pt.g pt.qc) else 0

/-- A summed reference's weight: an own column's by `gl`, an input's `iv`, the constant's `gc`. -/
noncomputable def refS (D : Def) (pt : FoldPt) (L H A : ℕ) (gL : Half) (pos : ℕ → ℕ) (gl gc : GF128)
    (iv : ℕ → GF128) (v : ℕ) : GF128 :=
  if v < D.cols then gl * of (gL.at (pos v % A)) * hiG pt L H (pos v / A)
  else if v < D.cols + D.ins then iv (v - D.cols) else gc

/-- **An explicit reference's step**: an own column adds at its slot, an input opens, the constant adds at `qc`'s slot. -/
theorem ref_tabExp (D : Def) (pt : FoldPt) (s L H A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ)
    (iv : ℕ → ℕ → GF128) (o : Tab) (k cx : ℕ) (w : F128) (v : ℕ) :
    tabExp pt s L H iv (Tab.ref D pt false A B as gL pos o k cx w v) = tabExp pt s L H iv o +
      of w * hiE pt L H cx * refX D pt s L H A B as gL pos iv k v := by
  unfold Tab.ref tabExp refX
  by_cases h1 : v < D.cols
  · simp only [h1, ↓reduceIte, Bool.false_eq_true, hsum_bump]
    split_ifs <;> simp only [of_mul] <;> ring
  · by_cases h2 : v < D.cols + D.ins
    · simp only [h1, h2, ↓reduceIte, hsum_bump]
      ring
    · simp only [h1, h2, ↓reduceIte, Bool.false_eq_true, hsum_bump]
      split_ifs <;> simp only [of_mul] <;> ring

/-- **A summed reference's step**: an own column adds to the own entries, an input opens, the constant adds. -/
theorem ref_tabSum (D : Def) (pt : FoldPt) (L H A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ)
    (gl gc : GF128) (iv : ℕ → GF128) (o : Tab) (k cx : ℕ) (w : F128) (v : ℕ) :
    tabSum pt L H gl gc iv (Tab.ref D pt true A B as gL pos o k cx w v) = tabSum pt L H gl gc iv o +
      of w * hiE pt L H cx * refS D pt L H A gL pos gl gc iv v := by
  unfold Tab.ref tabSum refS
  by_cases h1 : v < D.cols
  · simp only [h1, ↓reduceIte, hsum_bump, of_mul]
    ring
  · by_cases h2 : v < D.cols + D.ins
    · simp only [h1, h2, ↓reduceIte, hsum_bump]
      ring
    · simp only [h1, h2, ↓reduceIte, hsum_bump]
      ring

/-! ## A child's table, added and bound -/

/-- A table step that bumps each entry into `init` at a re-keyed key adds each entry's re-keyed term. -/
theorem hsum_fold_bump {κ κ' : Type} [BEq κ] [Hashable κ] [BEq κ'] [LawfulBEq κ'] [Hashable κ'] [LawfulHashable κ']
    [DecidableEq κ'] (m : HashMap κ F128) (init : HashMap κ' F128) (f : κ' → GF128) (g : κ → F128 → κ' × F128) :
    hsum (m.fold (fun acc k v => bump acc (g k v).1 (g k v).2) init) f =
      hsum init f + (m.toList.map fun e => of (g e.1 e.2).2 * f (g e.1 e.2).1).sum :=
  fold_val (fun acc => hsum acc f) _ (fun k v => of (g k v).2 * f (g k v).1)
    (fun acc k v => hsum_bump acc f _ _) m init

/-- **A child's table added** at re-keyed chunks (`rm` rows, `cm` columns, each a chunk and a factor). -/
theorem addChild_tabExp (pt : FoldPt) (s L H : ℕ) (iv : ℕ → ℕ → GF128) (o ch : Tab) (rm cm : ℕ → ℕ × F128)
    (ls lc : F128) :
    tabExp pt s L H iv (o.addChild ch rm cm ls lc) = tabExp pt s L H iv o +
      (hsum ch.P (fun e => if e.2.2 = s then
          of (rm e.1).2 * hiE pt L H (rm e.1).1 * (of (cm e.2.1).2 * hiG pt L H (cm e.2.1).1) else 0) +
        hsum ch.PC (fun e => if e.2 = s then of (rm e.1).2 * hiE pt L H (rm e.1).1 else 0)) := by
  unfold Tab.addChild tabExp
  dsimp only
  rw [hsum_fold_bump ch.P o.P _ fun k v => (((rm k.1).1, (cm k.2.1).1, k.2.2), v * (rm k.1).2 * (cm k.2.1).2),
    hsum_fold_bump ch.PC o.PC _ fun k v => (((rm k.1).1, k.2), v * (rm k.1).2)]
  unfold hsum
  have e1 : ∀ e : (ℕ × ℕ × ℕ) × F128, of (e.2 * (rm e.1.1).2 * (cm e.1.2.1).2) *
      (if e.1.2.2 = s then hiE pt L H (rm e.1.1).1 * hiG pt L H (cm e.1.2.1).1 else 0) =
      of e.2 * (if e.1.2.2 = s then
        of (rm e.1.1).2 * hiE pt L H (rm e.1.1).1 * (of (cm e.1.2.1).2 * hiG pt L H (cm e.1.2.1).1) else 0) := by
    intro e; split_ifs <;> simp only [of_mul, mul_zero] <;> ring
  have e2 : ∀ e : (ℕ × ℕ) × F128, of (e.2 * (rm e.1.1).2) * (if e.1.2 = s then hiE pt L H (rm e.1.1).1 else 0) =
      of e.2 * (if e.1.2 = s then of (rm e.1.1).2 * hiE pt L H (rm e.1.1).1 else 0) := by
    intro e; split_ifs <;> simp only [of_mul, mul_zero] <;> ring
  simp only [e1, e2]
  ring

theorem addChild_tabSum (pt : FoldPt) (L H : ℕ) (gl gc : GF128) (iv : ℕ → GF128) (o ch : Tab)
    (rm cm : ℕ → ℕ × F128) (ls lc : F128) :
    tabSum pt L H gl gc iv (o.addChild ch rm cm ls lc) = tabSum pt L H gl gc iv o +
      (gl * of ls * hsum ch.S (fun e =>
          of (rm e.1).2 * hiE pt L H (rm e.1).1 * (of (cm e.2).2 * hiG pt L H (cm e.2).1)) +
        gc * of lc * hsum ch.C (fun c => of (rm c).2 * hiE pt L H (rm c).1)) := by
  unfold Tab.addChild tabSum
  dsimp only
  rw [hsum_fold_bump ch.S o.S _ fun k v => (((rm k.1).1, (cm k.2).1), ls * v * (rm k.1).2 * (cm k.2).2),
    hsum_fold_bump ch.C o.C _ fun k v => ((rm k).1, lc * v * (rm k).2)]
  unfold hsum
  have e1 : ∀ e : (ℕ × ℕ) × F128, of (ls * e.2 * (rm e.1.1).2 * (cm e.1.2).2) *
      (hiE pt L H (rm e.1.1).1 * hiG pt L H (cm e.1.2).1) =
      of ls * (of e.2 * (of (rm e.1.1).2 * hiE pt L H (rm e.1.1).1 * (of (cm e.1.2).2 * hiG pt L H (cm e.1.2).1))) := by
    intro e; simp only [of_mul]; ring
  have e2 : ∀ e : ℕ × F128, of (lc * e.2 * (rm e.1).2) * hiE pt L H (rm e.1).1 =
      of lc * (of e.2 * (of (rm e.1).2 * hiE pt L H (rm e.1).1)) := by
    intro e; simp only [of_mul]; ring
  simp only [e1, e2, List.sum_map_mul_left]
  ring

/-- **The re-key**: a child table at chunk origin `Hc = H · 2^(al' − alc) + oc / 2^alc` (offset `oc` a multiple of its
chunks `2^alc`), at its chunk `c`, has the caller's chunk `⌊P / 2^al'⌋` factor times `eq` of the coordinates between,
`P = oc + c · 2^alc`. -/
theorem rekey (x : Array F128) {Lc alc al' oc : ℕ} (hle : alc ≤ al') (hk : Lc + (al' - alc) ≤ x.size)
    (hoc : oc % 2 ^ alc = 0) (H c : ℕ) :
    of (Tmpl.eqTail (Tmpl.halves x) Lc (H * 2 ^ (al' - alc) + oc / 2 ^ alc + c)) =
      of ((eqHalf x Lc (al' - alc)).at ((oc + c * 2 ^ alc) % 2 ^ al' / 2 ^ alc)) *
        of (Tmpl.eqTail (Tmpl.halves x) (Lc + (al' - alc)) (H + (oc + c * 2 ^ alc) / 2 ^ al')) := by
  have hA : 2 ^ al' = 2 ^ alc * 2 ^ (al' - alc) := by rw [← Nat.pow_add, Nat.add_sub_cancel' hle]
  have hdiv : (oc + c * 2 ^ alc) / 2 ^ alc = oc / 2 ^ alc + c := by
    rw [Nat.add_mul_div_right _ _ (Nat.two_pow_pos _)]
  have hlo : (oc + c * 2 ^ alc) % 2 ^ al' / 2 ^ alc = (oc + c * 2 ^ alc) / 2 ^ alc % 2 ^ (al' - alc) := by
    rw [hA, Nat.mod_mul_right_div_self]
  have hhi : (oc + c * 2 ^ alc) / 2 ^ al' = (oc + c * 2 ^ alc) / 2 ^ alc / 2 ^ (al' - alc) := by
    rw [hA, Nat.div_div_eq_div_mul]
  have hu := Nat.mod_lt ((oc + c * 2 ^ alc) / 2 ^ alc) (Nat.two_pow_pos (al' - alc))
  rw [hlo, hhi, eqHalf_at _ _ _ _ hu, ← eqTail_split_at x Lc (al' - alc) _ _ hk hu, hdiv]
  congr 2
  have hm := Nat.mod_add_div (oc / 2 ^ alc + c) (2 ^ (al' - alc))
  calc H * 2 ^ (al' - alc) + oc / 2 ^ alc + c = (oc / 2 ^ alc + c) % 2 ^ (al' - alc) +
        2 ^ (al' - alc) * ((oc / 2 ^ alc + c) / 2 ^ (al' - alc)) + H * 2 ^ (al' - alc) := by rw [hm]; ring
    _ = _ := by ring

/-- The terms a list of low parts and factors `(k, f)` adds through `ref`, each with weight `f · v · fx`. -/
theorem foldl_ref_exp (D : Def) (pt : FoldPt) (s L H A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ)
    (iv : ℕ → ℕ → GF128) (px : ℕ) (v fx : F128) (u : ℕ) :
    ∀ (l : List (ℕ × F128)) (o : Tab),
      tabExp pt s L H iv (l.foldl (fun o kf => Tab.ref D pt false A B as gL pos o kf.1 px (kf.2 * v * fx) u) o) =
        tabExp pt s L H iv o +
          (l.map fun kf => of (kf.2 * v * fx) * hiE pt L H px * refX D pt s L H A B as gL pos iv kf.1 u).sum
  | [], o => by simp
  | kf :: l, o => by
    rw [List.foldl_cons, foldl_ref_exp D pt s L H A B as gL pos iv px v fx u l, ref_tabExp, List.map_cons,
      List.sum_cons, add_assoc]

theorem foldl_ref_sum (D : Def) (pt : FoldPt) (L H A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ)
    (gl gc : GF128) (iv : ℕ → GF128) (px : ℕ) (v fx : F128) (u : ℕ) :
    ∀ (l : List (ℕ × F128)) (o : Tab),
      tabSum pt L H gl gc iv (l.foldl (fun o kf => Tab.ref D pt true A B as gL pos o kf.1 px (kf.2 * v * fx) u) o) =
        tabSum pt L H gl gc iv o +
          (l.map fun kf => of (kf.2 * v * fx) * hiE pt L H px * refS D pt L H A gL pos gl gc iv u).sum
  | [], o => by simp
  | kf :: l, o => by
    rw [List.foldl_cons, foldl_ref_sum D pt L H A B as gL pos gl gc iv px v fx u l, ref_tabSum, List.map_cons,
      List.sum_cons, add_assoc]

/-- **A copy's bindings**: each open input entry of the child, through `wk`'s low parts and factors, adds `ref`'s term
for the caller reference bound to it. -/
theorem bind_tabExp (D : Def) (pt : FoldPt) (s L H A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ)
    (iv : ℕ → ℕ → GF128) (c : Call) (o : Tab) (j : ℕ) (ch : Tab) (rm : ℕ → ℕ × F128)
    (wk : ℕ → ℕ → List (ℕ × F128)) :
    tabExp pt s L H iv (Tab.bind (Tab.ref D pt false A B as gL pos) c o j ch rm wk) = tabExp pt s L H iv o +
      (ch.U.toList.map fun e => ((wk e.1.2.1 j).map fun kf =>
        of (kf.2 * e.2 * (rm e.1.2.2).2) * hiE pt L H (rm e.1.2.2).1 *
          refX D pt s L H A B as gL pos iv kf.1 ((c.bound id j).toArray.getD e.1.1 0)).sum).sum := by
  unfold Tab.bind
  dsimp only
  rw [fold_val (tabExp pt s L H iv) _ (fun e v => ((wk e.2.1 j).map fun kf =>
      of (kf.2 * v * (rm e.2.2).2) * hiE pt L H (rm e.2.2).1 *
        refX D pt s L H A B as gL pos iv kf.1 ((c.bound id j).toArray.getD e.1 0)).sum) ?_]
  · rfl
  · rintro o' ⟨t', kc, cx⟩ v
    dsimp only
    generalize rm cx = p
    obtain ⟨px, fx⟩ := p
    exact foldl_ref_exp D pt s L H A B as gL pos iv px v fx _ _ o'

theorem bind_tabSum (D : Def) (pt : FoldPt) (L H A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ)
    (gl gc : GF128) (iv : ℕ → GF128) (c : Call) (o : Tab) (j : ℕ) (ch : Tab) (rm : ℕ → ℕ × F128)
    (wk : ℕ → ℕ → List (ℕ × F128)) :
    tabSum pt L H gl gc iv (Tab.bind (Tab.ref D pt true A B as gL pos) c o j ch rm wk) = tabSum pt L H gl gc iv o +
      (ch.U.toList.map fun e => ((wk e.1.2.1 j).map fun kf =>
        of (kf.2 * e.2 * (rm e.1.2.2).2) * hiE pt L H (rm e.1.2.2).1 *
          refS D pt L H A gL pos gl gc iv ((c.bound id j).toArray.getD e.1.1 0)).sum).sum := by
  unfold Tab.bind
  dsimp only
  rw [fold_val (tabSum pt L H gl gc iv) _ (fun e v => ((wk e.2.1 j).map fun kf =>
      of (kf.2 * v * (rm e.2.2).2) * hiE pt L H (rm e.2.2).1 *
        refS D pt L H A gL pos gl gc iv ((c.bound id j).toArray.getD e.1 0)).sum) ?_]
  · rfl
  · rintro o' ⟨t', kc, cx⟩ v
    dsimp only
    generalize rm cx = p
    obtain ⟨px, fx⟩ := p
    exact foldl_ref_sum D pt L H A B as gL pos gl gc iv px v fx _ _ o'

end FlockSoundness.TmplFold
