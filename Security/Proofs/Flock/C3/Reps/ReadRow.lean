import Proofs.Flock.C3.Reps.ReadCol

/-!
# The opening unit reads its row columns (`openRowRead`)

`OpenRowRead LANES H`, proved. The unit is `unitLaid LANES H`, `recOpenNet`'s run from the empty state laid out.

* The builder (`rowReads_unit`): `recOpen` computes the residuals first, and residual 0's product `l` multiplies
  `coef`'s element `l` by the row's (`consStructure`). Karatsuba's leaves include each position's pair of bits
  (`mem_leaves`), so `leafSums` ANDs `coef`'s bit `c` with the row's, two input bits, which `and` makes a row
  (`reads_and`). Every builder after it keeps the rows it starts from (`Keeps`: `fresh` puts its row in front), so that
  row stays (`Reads`, `RowReads`).
* The layout (`laid_reads`): the `k`-th row in creation order is the laid-out row `inRows + k`, below the constant, its
  forms' columns those of their variables, an input bit's its own (`layout_rows`, `col_input`).
* The net (`readCol_of_laid`): a laid-out row below the constant is a computed row of `Rows.ofNet`, reading the same
  columns, at `unitLog ≤ 32`.
-/

namespace FlockC3.Reps.RowRead

section Builder

open Flock.HmNets hiding and
open Flock.RecOpen

/-! ## The rows a builder starts from stay -/

/-- `m` keeps the rows it starts from: `fresh` only puts rows in front. -/
def Keeps {α : Type} (m : M α) : Prop := ∀ s : St, s.rows <:+ (m.run s).2.rows

namespace Keeps

variable {α β : Type}

theorem pure' (a : α) : Keeps (pure a : M α) := fun s => List.suffix_refl s.rows

theorem bind' {m : M α} {f : α → M β} (hm : Keeps m) (hf : ∀ a, Keeps (f a)) : Keeps (m >>= f) :=
  fun s => (hm s).trans (hf (m.run s).1 (m.run s).2)

theorem ite' {c : Prop} [Decidable c] {m₁ m₂ : M α} (h₁ : Keeps m₁) (h₂ : Keeps m₂) :
    Keeps (if c then m₁ else m₂) := by
  split
  · exact h₁
  · exact h₂

theorem mapM' (f : β → M α) (hf : ∀ b, Keeps (f b)) : ∀ l : List β, Keeps (l.mapM f)
  | [] => by rw [List.mapM_nil]; exact pure' _
  | b :: l => by
      rw [List.mapM_cons]
      exact bind' (hf b) fun _ => bind' (mapM' f hf l) fun _ => pure' _

end Keeps

theorem keeps_fresh (a b : Lin) : Keeps (fresh a b) := fun s => by
  show s.rows <:+ ⟨s.next, a, b⟩ :: s.rows
  exact List.suffix_cons _ _

theorem keeps_and (a b : Lin) : Keeps (Flock.HmNets.and a b) := by
  unfold Flock.HmNets.and
  exact Keeps.ite' (Keeps.pure' _) (Keeps.ite' (Keeps.pure' _) (Keeps.ite' (Keeps.pure' _) (keeps_fresh a b)))

theorem keeps_maj (x y w : Lin) : Keeps (maj x y w) := by
  unfold maj
  dsimp only
  exact Keeps.ite' (Keeps.ite' (Keeps.pure' _) (Keeps.ite' (Keeps.pure' _) (Keeps.pure' _)))
    (Keeps.bind' (keeps_and _ _) fun _ => Keeps.pure' _)

theorem keeps_commit (f : Lin) : Keeps (commit f) := by
  unfold commit
  exact Keeps.ite' (Keeps.pure' _) (keeps_fresh _ _)

theorem keeps_chW (e f g : Word) : Keeps (chW e f g) :=
  Keeps.mapM' _ (fun _ => Keeps.bind' (keeps_and _ _) fun _ => Keeps.pure' _) _

theorem keeps_majW (a b c : Word) : Keeps (majW a b c) := Keeps.mapM' _ (fun _ => keeps_maj _ _ _) _

theorem keeps_commitW (w : Word) : Keeps (commitW w) := Keeps.mapM' _ keeps_commit _

theorem keeps_carryCommit (i : Nat) (c : Lin) : Keeps (carryCommit i c) := by
  unfold carryCommit
  exact Keeps.ite' (keeps_commit c) (Keeps.pure' _)

theorem keeps_add2Go : ∀ (a b : Word) (c : Lin) (i : Nat), Keeps (add2Go a b c i)
  | x :: a, y :: b, c, i => by
      unfold add2Go
      exact Keeps.ite' (Keeps.pure' _) (Keeps.bind' (keeps_maj x y c) fun c₁ =>
        Keeps.bind' (keeps_carryCommit i c₁) fun c₂ =>
          Keeps.bind' (keeps_add2Go a b c₂ (i + 1)) fun _ => Keeps.pure' _)
  | [], _, _, _ => by unfold add2Go; exact Keeps.pure' _
  | _ :: _, [], _, _ => by unfold add2Go; exact Keeps.pure' _

theorem keeps_add2 (a b : Word) : Keeps (add2 a b) := keeps_add2Go a b _ 0

theorem keeps_addN : ∀ (acc : Word) (xs : List Word), Keeps (addN acc xs)
  | _, [] => Keeps.pure' _
  | acc, x :: xs => Keeps.bind' (keeps_add2 acc x) fun acc' => keeps_addN acc' xs

theorem keeps_schedStep (w : List Word) : Keeps (schedStep w) :=
  Keeps.bind' (keeps_addN _ _) fun _ => Keeps.bind' (keeps_commitW _) fun _ => Keeps.pure' _

theorem keeps_schedule : ∀ (n : Nat) (w : List Word), Keeps (schedule n w)
  | 0, _ => Keeps.pure' _
  | n + 1, w => Keeps.bind' (keeps_schedStep w) fun w' => keeps_schedule n w'

theorem keeps_roundW (k : BitVec 64) (wt : Word) (r : Regs Word) : Keeps (roundW k wt r) :=
  Keeps.bind' (keeps_chW _ _ _) fun _ => Keeps.bind' (keeps_addN _ _) fun _ =>
    Keeps.bind' (keeps_majW _ _ _) fun _ => Keeps.bind' (keeps_add2 _ _) fun _ =>
      Keeps.bind' (keeps_add2 _ _) fun _ => Keeps.bind' (keeps_commitW _) fun _ =>
        Keeps.bind' (keeps_add2 _ _) fun _ => Keeps.bind' (keeps_commitW _) fun _ => Keeps.pure' _

theorem keeps_rounds (w : List Word) : ∀ (n t : Nat) (r : Regs Word), Keeps (rounds w n t r)
  | 0, _, _ => Keeps.pure' _
  | n + 1, t, r => Keeps.bind' (keeps_roundW _ _ r) fun r' => keeps_rounds w n (t + 1) r'

theorem keeps_compress (cv msg : List Word) : Keeps (compress cv msg) :=
  Keeps.bind' (keeps_schedule _ _) fun _ => Keeps.bind' (keeps_rounds _ _ _ _) fun _ =>
    Keeps.mapM' _ (fun _ => keeps_add2 _ _) _

theorem keeps_leafSums : ∀ (ps : List (List Lin × List Lin)) (sums : List Lin), Keeps (leafSums ps sums)
  | [], _ => Keeps.pure' _
  | _ :: ps, _ => Keeps.bind' (Keeps.mapM' _ (fun _ => keeps_and _ _) _) fun _ => keeps_leafSums ps _

theorem keeps_residual (S : GfS) (W V : List Lin) (r : Res) : Keeps (Flock.RecOpen.residual S W V r) := by
  unfold Flock.RecOpen.residual
  exact Keeps.ite' (Keeps.pure' _)
    (Keeps.bind' (keeps_leafSums _ _) fun _ => Keeps.bind' (keeps_commitW _) fun _ => Keeps.pure' _)

theorem keeps_residualsGo (S : GfS) (W V : List Lin) : ∀ rs : List Res, Keeps (residualsGo S W V rs)
  | [] => Keeps.pure' _
  | r :: rs => Keeps.bind' (keeps_residual S W V r) fun _ =>
      Keeps.bind' (keeps_residualsGo S W V rs) fun _ => Keeps.pure' _

theorem keeps_shaBlock (st blk : List Lin) : Keeps (shaBlock st blk) :=
  Keeps.bind' (keeps_compress _ _) fun _ => Keeps.pure' _

theorem keeps_shaBlocks (msg : List Lin) : ∀ (n k : Nat) (st : List Lin), Keeps (shaBlocks msg n k st)
  | 0, _, _ => Keeps.pure' _
  | n + 1, k, _ => Keeps.bind' (keeps_shaBlock _ _) fun st' => keeps_shaBlocks msg n (k + 1) st'

theorem keeps_shaFrom (cv : List (BitVec 64)) (bits : List Lin) (pre : Nat) : Keeps (shaFrom cv bits pre) :=
  Keeps.bind' (keeps_shaBlocks _ _ _ _) fun _ => Keeps.pure' _

theorem keeps_swap (d : Lin) (a s : List Lin) : Keeps (swap d a s) :=
  Keeps.bind' (Keeps.mapM' _ (fun _ => keeps_and _ _) _) fun _ => Keeps.pure' _

theorem keeps_climbFrom (d : List Lin) : ∀ (h : Nat) (cur : List Lin) (ss : List (List Lin)), Keeps (climbFrom d h cur ss)
  | _, _, [] => Keeps.pure' _
  | h, _, _ :: rest => Keeps.bind' (keeps_swap _ _ _) fun _ =>
      Keeps.bind' (keeps_shaFrom _ _ _) fun cur' => keeps_climbFrom d (h + 1) cur' rest

theorem keeps_climbRow (H : Nat) (row aux d : List Lin) : Keeps (climbRow H row aux d) :=
  Keeps.bind' (keeps_shaFrom _ _ _) fun _ => Keeps.bind' (keeps_climbFrom _ _ _ _) fun _ => Keeps.pure' _

/-! ## A row reading a variable -/

/-- A row of `s` reads variable `v`, in `A` or in `B`. -/
def RowReads (s : St) (v : Var) : Prop := ∃ r ∈ s.rows, v ∈ r.a.s ++ r.b.s

/-- Running `m` from any state leaves a row reading `v`. -/
def Reads {α : Type} (m : M α) (v : Var) : Prop := ∀ s : St, RowReads (m.run s).2 v

theorem RowReads.mono {s t : St} (h : s.rows <:+ t.rows) {v : Var} (hr : RowReads s v) : RowReads t v :=
  let ⟨r, hm, hv⟩ := hr
  ⟨r, h.subset hm, hv⟩

namespace Reads

variable {α β : Type}

theorem bind_left {m : M α} {f : α → M β} {v : Var} (hm : Reads m v) (hf : ∀ a, Keeps (f a)) : Reads (m >>= f) v :=
  fun s => (hm s).mono (hf (m.run s).1 (m.run s).2)

theorem bind_right {m : M α} {f : α → M β} {v : Var} (hf : ∀ a, Reads (f a) v) : Reads (m >>= f) v :=
  fun s => hf (m.run s).1 (m.run s).2

theorem ite_neg {c : Prop} [Decidable c] {m₁ m₂ : M α} {v : Var} (hc : ¬c) (h : Reads m₂ v) :
    Reads (if c then m₁ else m₂) v := by
  split
  · next h' => exact absurd h' hc
  · exact h

theorem mem_mapM (f : β → M α) (hf : ∀ b, Keeps (f b)) {v : Var} {p : β} (hp : Reads (f p) v) :
    ∀ l : List β, p ∈ l → Reads (l.mapM f) v
  | [], h => by simp at h
  | b :: l, h => by
      rw [List.mapM_cons]
      rcases List.mem_cons.1 h with rfl | h
      · exact bind_left hp fun _ => Keeps.bind' (Keeps.mapM' f hf l) fun _ => Keeps.pure' _
      · exact bind_right fun _ => bind_left (mem_mapM f hf hp l h) fun _ => Keeps.pure' _

end Reads

theorem reads_fresh {a b : Lin} {v : Var} (hv : v ∈ a.s ++ b.s) : Reads (fresh a b) v := fun s => by
  show RowReads ⟨s.next + 1, ⟨s.next, a, b⟩ :: s.rows⟩ v
  exact ⟨⟨s.next, a, b⟩, List.mem_cons.2 (Or.inl rfl), hv⟩

/-- An AND of two non-constant forms of different supports is a row. -/
theorem reads_and {a b : Lin} {v : Var} (ha : a.s ≠ []) (hb : b.s ≠ []) (hab : a.s ≠ b.s) (hv : v ∈ a.s ++ b.s) :
    Reads (Flock.HmNets.and a b) v := by
  have h1 : ¬ a.isConst = true := by simpa [Lin.isConst] using ha
  have h2 : ¬ b.isConst = true := by simpa [Lin.isConst] using hb
  unfold Flock.HmNets.and
  exact Reads.ite_neg h1 (Reads.ite_neg h2 (Reads.ite_neg hab (reads_fresh hv)))

/-! ## Residual 0's leaves -/

theorem getD_take {l : List Lin} {k i : Nat} (h : i < k) : (l.take k).getD i Lin.zero = l.getD i Lin.zero := by
  simp [List.getD_eq_getElem?_getD, h]

theorem getD_drop (l : List Lin) (k i : Nat) : (l.drop k).getD i Lin.zero = l.getD (k + i) Lin.zero := by
  simp [List.getD_eq_getElem?_getD, List.getElem?_drop]

/-- **Karatsuba's leaves include each position's pair of operand bits**. -/
theorem mem_leaves : ∀ (n : Nat) (a b : List Lin) (i : Nat), i < 2 ^ n →
    (a.getD i Lin.zero, b.getD i Lin.zero) ∈ leaves n a b
  | 0, a, b, i, hi => by
      rw [Nat.pow_zero] at hi
      obtain rfl : i = 0 := by omega
      cases a <;> cases b <;> simp [leaves]
  | n + 1, a, b, i, hi => by
      rw [leaves]
      by_cases h : i < 2 ^ n
      · have := mem_leaves n (a.take (2 ^ n)) (b.take (2 ^ n)) i h
        rw [getD_take h, getD_take h] at this
        exact List.mem_append_left _ (List.mem_append_left _ this)
      · have h2 : i - 2 ^ n < 2 ^ n := by rw [Nat.pow_succ] at hi; omega
        have := mem_leaves n (a.drop (2 ^ n)) (b.drop (2 ^ n)) (i - 2 ^ n) h2
        rw [getD_drop, getD_drop, Nat.add_sub_cancel' (by omega)] at this
        exact List.mem_append_left _ (List.mem_append_right _ this)

theorem reads_leafSums {v : Var} {p : List Lin × List Lin} {x y : Lin} (hxy : (x, y) ∈ leaves 7 p.1 p.2)
    (hx : x.s ≠ []) (hy : y.s ≠ []) (hne : x.s ≠ y.s) (hv : v ∈ x.s ++ y.s) :
    ∀ (ps : List (List Lin × List Lin)) (sums : List Lin), p ∈ ps → Reads (leafSums ps sums) v
  | [], _, h => by simp at h
  | q :: ps, sums, h => by
      rw [leafSums]
      rcases List.mem_cons.1 h with rfl | h
      · exact Reads.bind_left (Reads.mem_mapM (fun xy : Lin × Lin => Flock.HmNets.and xy.1 xy.2)
          (fun _ => keeps_and _ _) (p := (x, y)) (reads_and hx hy hne hv) _ hxy) fun _ => keeps_leafSums _ _
      · exact Reads.bind_right fun _ => reads_leafSums hxy hx hy hne hv ps _ h

theorem reads_residual (S : GfS) (W V : List Lin) (r : Res) (hne : ¬ r.mul.isEmpty = true) {v : Var}
    (h : Reads (leafSums (r.mul.map fun ab => (opForms S W V ab.1, opForms S W V ab.2))
      (List.replicate (3 ^ 7) Lin.zero)) v) : Reads (Flock.RecOpen.residual S W V r) v := by
  unfold Flock.RecOpen.residual
  exact Reads.ite_neg hne (Reads.bind_left h fun _ => Keeps.bind' (keeps_commitW _) fun _ => Keeps.pure' _)

theorem reads_residualsGo (S : GfS) (W V : List Lin) {v : Var} {r : Res} (hr : Reads (Flock.RecOpen.residual S W V r) v) :
    ∀ rs : List Res, r ∈ rs → Reads (residualsGo S W V rs) v
  | [], h => by simp at h
  | r' :: rs, h => by
      rw [residualsGo]
      rcases List.mem_cons.1 h with rfl | h
      · exact Reads.bind_left hr fun _ => Keeps.bind' (keeps_residualsGo S W V rs) fun _ => Keeps.pure' _
      · exact Reads.bind_right fun _ => Reads.bind_left (reads_residualsGo S W V hr rs h) fun _ => Keeps.pure' _

/-! ## The operand forms -/

theorem ops_w {LANES l : Nat} (hl : l < LANES) : (consStructure LANES).ops.getD l [] = [Term.w l] := by
  simp [consStructure, List.getD_eq_getElem?_getD, List.getElem?_append_left, hl]

theorem ops_v {LANES i : Nat} (hi : i < 2 * LANES) : (consStructure LANES).ops.getD (LANES + i) [] = [Term.v i] := by
  simp [consStructure, List.getD_eq_getElem?_getD, hi]

theorem length_termForms (W V : List Lin) (t : Term) : (termForms W V t).length = 128 := by
  cases t <;> simp [termForms]

theorem getD_opForms {S : GfS} {W V : List Lin} {k : Nat} {t : Term} (hk : S.ops.getD k [] = [t]) {j : Nat}
    (hj : j < 128) : (opForms S W V k).getD j Lin.zero = Lin.zero ^^^ (termForms W V t).getD j Lin.zero := by
  rw [opForms, hk]
  have hl := length_termForms W V t
  simp only [xorAll, xorL, List.map_cons, List.map_nil, List.foldl_cons, List.foldl_nil, List.getD_eq_getElem?_getD,
    List.getElem?_zipWith, List.getElem?_replicate, hj, ↓reduceIte,
    List.getElem?_eq_getElem (show j < (termForms W V t).length by omega), Option.getD_some]

theorem getD_termForms_w (W V : List Lin) (l : Nat) {j : Nat} (hj : j < 128) :
    (termForms W V (.w l)).getD j Lin.zero = W.getD (128 * l + j) Lin.zero := by
  simp [termForms, List.getD_eq_getElem?_getD, hj]

theorem getD_termForms_v (W V : List Lin) (i : Nat) {j : Nat} (hj : j < 128) :
    (termForms W V (.v i)).getD j Lin.zero = V.getD (128 * i + j) Lin.zero := by
  simp [termForms, List.getD_eq_getElem?_getD, hj]

theorem getD_inputs (n b i : Nat) (hi : i < n) :
    ((List.range n).map fun k => Lin.var (b + k)).getD i Lin.zero = Lin.var (b + i) := by
  simp [List.getD_eq_getElem?_getD, hi]

theorem s_zero_xor_var (u : Var) : (Lin.zero ^^^ Lin.var u).s = [u] := by
  show Flock.HmNets.symm [] [u] = [u]
  rw [Flock.HmNets.symm.eq_1]

/-! ## Residual 0 reads the row -/

/-- **Residual 0 ANDs bit `c` of `V` with bit `c` of `W`**: its product `l = c / 128` multiplies `V`'s element `l` by
`W`'s, and Karatsuba's leaf `c % 128` is their bits `c`; where those are variables `w ≠ u`, a row reads `u`. -/
theorem reads_residualForms {LANES : Nat} {W V : List Lin} {c : Nat} (hc : c < 128 * LANES) {u w : Var}
    (hW : W.getD c Lin.zero = Lin.var u) (hV : V.getD c Lin.zero = Lin.var w) (huw : w ≠ u) :
    Reads (residualForms (consStructure LANES) W V) u := by
  have hl : c / 128 < LANES := by omega
  have hj : c % 128 < 128 := Nat.mod_lt _ (by decide)
  have hcj : 128 * (c / 128) + c % 128 = c := Nat.div_add_mod c 128
  have hx : (opForms (consStructure LANES) W V (LANES + 0 * LANES + c / 128)).getD (c % 128) Lin.zero =
      Lin.zero ^^^ Lin.var w := by
    rw [show LANES + 0 * LANES + c / 128 = LANES + c / 128 by omega, getD_opForms (ops_v (by omega)) hj,
      getD_termForms_v _ _ _ hj, hcj, hV]
  have hy : (opForms (consStructure LANES) W V (c / 128)).getD (c % 128) Lin.zero = Lin.zero ^^^ Lin.var u := by
    rw [getD_opForms (ops_w hl) hj, getD_termForms_w _ _ _ hj, hcj, hW]
  have hxy := mem_leaves 7 (opForms (consStructure LANES) W V (LANES + 0 * LANES + c / 128))
    (opForms (consStructure LANES) W V (c / 128)) (c % 128) (Nat.lt_of_lt_of_le hj (by decide))
  rw [hx, hy] at hxy
  have hs : ∀ v : Var, (Lin.zero ^^^ Lin.var v).s ≠ [] := fun v => by rw [s_zero_xor_var]; exact List.cons_ne_nil _ _
  have hne : (Lin.zero ^^^ Lin.var w).s ≠ (Lin.zero ^^^ Lin.var u).s := by
    rw [s_zero_xor_var, s_zero_xor_var]
    exact fun h => huw (List.cons.inj h).1
  have hv : u ∈ (Lin.zero ^^^ Lin.var w).s ++ (Lin.zero ^^^ Lin.var u).s := by
    rw [s_zero_xor_var, s_zero_xor_var]
    exact List.mem_append_right _ (List.mem_cons.2 (Or.inl rfl))
  have hr0 := reads_leafSums (p := (opForms (consStructure LANES) W V (LANES + 0 * LANES + c / 128),
    opForms (consStructure LANES) W V (c / 128))) hxy (hs w) (hs u) hne hv
    (((List.range LANES).map fun l => (LANES + 0 * LANES + l, l)).map fun ab =>
      (opForms (consStructure LANES) W V ab.1, opForms (consStructure LANES) W V ab.2)) (List.replicate (3 ^ 7) Lin.zero)
    (List.mem_map.2 ⟨(LANES + 0 * LANES + c / 128, c / 128), List.mem_map.2 ⟨c / 128, List.mem_range.2 hl, rfl⟩, rfl⟩)
  exact reads_residualsGo _ _ _ (r := Res.mk ((List.range LANES).map fun l => (LANES + 0 * LANES + l, l))
      [Term.w (LANES + 0)])
    (reads_residual _ _ _ _ (by simp; omega) hr0) _ (List.mem_map.2 ⟨0, List.mem_range.2 (by decide), rfl⟩)

theorem reads_recOpen {LANES H : Nat} {row aux acc coef dirs : List Lin} {c : Nat} (hc : c < 128 * LANES) {u w : Var}
    (hrow : c < row.length) (hW : row.getD c Lin.zero = Lin.var u) (hV : coef.getD c Lin.zero = Lin.var w)
    (huw : w ≠ u) : Reads (recOpen LANES H row aux acc coef dirs) u := by
  have hW' : (row ++ acc.take 256).getD c Lin.zero = Lin.var u := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_append_left hrow, ← List.getD_eq_getElem?_getD, hW]
  unfold recOpen
  exact Reads.bind_left (reads_residualForms hc hW' hV huw)
    fun _ => Keeps.bind' (keeps_climbRow _ _ _ _) fun _ => Keeps.pure' _

theorem recOpenNet_run (LANES H : Nat) (p : Ports) (s : St) :
    (recOpenNet LANES H p).run s =
      (recOpen LANES H ((List.range p.row).map fun i => Lin.var (s.next + i))
        ((List.range p.aux).map fun i => Lin.var (s.next + p.row + i))
        ((List.range p.acc).map fun i => Lin.var (s.next + p.row + p.aux + i))
        ((List.range p.coef).map fun i => Lin.var (s.next + p.row + p.aux + p.acc + i))
        ((List.range p.dirs).map fun i => Lin.var (s.next + p.row + p.aux + p.acc + p.coef + i))).run
        ⟨s.next + p.row + p.aux + p.acc + p.coef + p.dirs, s.rows⟩ := rfl

/-- **The opening's builder leaves a row reading each row bit.** -/
theorem rowReads_unit (LANES H : Nat) {c : Nat} (hc : c < 128 * LANES) :
    RowReads ((recOpenNet LANES H (ports LANES H)).run St.empty).2 c := by
  rw [recOpenNet_run]
  refine reads_recOpen hc (by simpa [ports] using hc) ?_ (getD_inputs _ _ _ (by simp only [ports]; omega)) ?_ _
  · rw [getD_inputs _ _ _ (by simpa [ports] using hc)]
    simp [St.empty]
  · simp only [St.empty, ports]
    generalize max 1 ((H + 1) / 2) = m
    change @Ne Nat _ _
    omega

/-! ## In the laid-out unit -/

theorem mem_cols {nIn inRows cp : Nat} {f : Lin} {v : Var} (hv : v ∈ f.s) (hlt : v < nIn) :
    v ∈ (cols nIn inRows cp f).toList := by
  simp only [cols, Array.toList_append]
  exact List.mem_append_left _ (List.mem_map.2 ⟨v, hv, col_input hlt⟩)

/-- **The laid-out unit has a computed row reading each row bit**: the `k`-th AND, at row `inRows + k`. -/
theorem laid_reads (LANES H : Nat) {c : Nat} (hc : c < 128 * LANES) :
    ∃ k, groupWords ((wordPorts ((ports LANES H).bits.foldl (· + ·) 0)).foldl (· + ·) 0) * Flock.WORD + k <
        (unitLaid LANES H).constPos ∧
      c ∈ ((unitLaid LANES H).ra.getD
          (groupWords ((wordPorts ((ports LANES H).bits.foldl (· + ·) 0)).foldl (· + ·) 0) * Flock.WORD + k) #[]).toList ++
        ((unitLaid LANES H).rb.getD
          (groupWords ((wordPorts ((ports LANES H).bits.foldl (· + ·) 0)).foldl (· + ·) 0) * Flock.WORD + k) #[]).toList := by
  obtain ⟨r, hr, hv⟩ := rowReads_unit LANES H hc
  obtain ⟨k, hk, hrk⟩ := List.mem_iff_getElem.1 (List.mem_reverse.2 hr)
  obtain ⟨h1, -, h3, -⟩ := layout_rows ((recOpenNet LANES H (ports LANES H)).run St.empty).2
    (wordPorts ((ports LANES H).bits.foldl (· + ·) 0)) (wordPorts 2048) (recOpenOuts LANES H (ports LANES H)).toArray
  have hspan := groupWords_pos ((wordPorts 2048).foldl (· + ·) 0)
  have hst := le_roundUp (groupWords ((wordPorts ((ports LANES H).bits.foldl (· + ·) 0)).foldl (· + ·) 0) * Flock.WORD +
    ((recOpenNet LANES H (ports LANES H)).run St.empty).2.rows.length) _ hspan
  have hlt : c < (wordPorts ((ports LANES H).bits.foldl (· + ·) 0)).foldl (· + ·) 0 := by
    rw [foldl_wordPorts, foldl_ports]
    generalize max 1 ((H + 1) / 2) = m
    omega
  refine ⟨k, ?_, ?_⟩
  · show _ < (layout _ _ _ _).constPos
    rw [h1]
    simp only [List.length_reverse] at hk
    omega
  · obtain ⟨ha, hb⟩ := h3 k hk
    show c ∈ ((layout _ _ _ _).ra.getD _ #[]).toList ++ ((layout _ _ _ _).rb.getD _ #[]).toList
    rw [ha, hb, hrk]
    rcases List.mem_append.1 hv with hv | hv
    · exact List.mem_append_left _ (mem_cols hv hlt)
    · exact List.mem_append_right _ (mem_cols hv hlt)

theorem inWords_unitLaid {LANES H : Nat} {name : String} {net : Flock.Net}
    (hn : netOf name (unitLaid LANES H) = .ok net) :
    net.inWords = groupWords ((wordPorts ((ports LANES H).bits.foldl (· + ·) 0)).foldl (· + ·) 0) := by
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, hiw, -⟩ := Flock.Net.ofRows_ok hn
  rw [hiw]
  show 0 + _ = _
  rw [Nat.zero_add]
  rfl

end Builder

section Net

open FlockSoundness FlockSoundness.Audit

/-- **A laid-out row below the constant is a computed row of the net**, reading the same columns: the net is
`Net.ofRows`' (`netOf`) of the layout's rows, its computed rows `Rows.ofNet`'s from its input rows' end, at
`unitLog ≤ 32` (`ofRows_row`, `Rows.ofNet_a`, `Rows.ofNet_b`). -/
theorem readCol_of_laid {net : Flock.Net} (hU : Built net) {name : String} {l : Flock.HmNets.Laid}
    (hl : Flock.HmNets.netOf name l = .ok net) (hu : net.unitLog ≤ 32) {k c : ℕ}
    (hk : net.inWords * Flock.WORD + k < l.constPos)
    (hc : c ∈ (l.ra.getD (net.inWords * Flock.WORD + k) #[]).toList ++
      (l.rb.getD (net.inWords * Flock.WORD + k) #[]).toList) :
    FlockVBridge.OfVStar.ReadCol (Rows.ofNet hU) c := by
  obtain ⟨-, ha, hb, -, hcp, -, -, hle, -⟩ := Flock.Net.ofRows_ok hl
  have hnIn : (Rows.ofNet hU).nIn = net.inWords * Flock.WORD := by rw [Rows.ofNet_nIn]; omega
  have h1 : (Rows.ofNet hU).nIn + (Rows.ofNet hU).nComp = l.constPos := by
    rw [← hcp]; exact Rows.ofNet_one hU
  have hw : (2 : ℕ) ^ net.unitLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by norm_num) hu
  have hj : k < (Rows.ofNet hU).nComp := by omega
  have hij : (Rows.ofNet hU).nIn + (⟨k, hj⟩ : Fin _).val = net.inWords * Flock.WORD + k := by simp only; omega
  have hra : ((Rows.ofNet hU).a ⟨k, hj⟩).map Fin.val = (l.ra.getD (net.inWords * Flock.WORD + k) #[]).toList := by
    have h := Rows.ofNet_a hU hu ⟨k, hj⟩
    rw [ofRows_row ha hw, hij] at h
    exact h
  have hrb : ((Rows.ofNet hU).b ⟨k, hj⟩).map Fin.val = (l.rb.getD (net.inWords * Flock.WORD + k) #[]).toList := by
    have h := Rows.ofNet_b hU hu ⟨k, hj⟩
    rw [ofRows_row hb hw, hij] at h
    exact h
  rw [← hra, ← hrb, ← List.map_append] at hc
  obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hc
  exact ⟨⟨k, hj⟩, x, hx, rfl⟩

end Net

end FlockC3.Reps.RowRead

namespace FlockC3.Reps

/-- **The opening unit reads its row columns** (`OpenRowRead`): residual 0's AND of `coef`'s bit `c` and the row's
(`laid_reads`) is a computed row of the unit's net (`readCol_of_laid`). -/
theorem openRowRead (LANES H : ℕ) : OpenRowRead LANES H := by
  intro net hU hn hu c hc
  obtain ⟨k, hk, hmem⟩ := RowRead.laid_reads LANES H hc
  rw [← RowRead.inWords_unitLaid hn] at hk hmem
  exact RowRead.readCol_of_laid hU hn hu hk hmem

end FlockC3.Reps
