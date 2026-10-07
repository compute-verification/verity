import Proofs.Flock.Soundness.Types.Flat
import Proofs.Flock.Soundness.Lowering

/-!
# A unit's rows in logical order (verified-lowering phase 1, 1d step 1)

The verifier folds a unit's rows at physical columns; the audit reads them as `Rows`, whose columns are in logical order:
the unit's input bits, then its computed rows, each reading only earlier columns or `one`. `Rows.compose` builds the
audit's `Rows` from a `Logical`: the columns in logical order (`order`), and each column's row as the block holds it
(`row`), or `none` for a row that copies the block's `one`. So the audit's rows are the verifier's, read in another order:
- logical column `ℓ < |order|` is physical column `order[ℓ]`, and the last logical column is the block's `one`;
- computed row `k` is the row at `order[nIn + k]`, each column it reads mapped to its place in `order`.

Placement (for every `Logical` whose rows are `Ordered`):
- `placement_of_realizes`: `Placement` for `Rows.compose` at `composePlace` (each logical column at its physical
  column's block position, `one` at the pin), when the block's matrices at those positions read the rows' columns'
  positions, a row that copies `one` and the pin's row read the pin.

For every flat type `derive` accepts, whose `Logical` is `ofDerived` (`order`, the rows, the constant copying `one` as Δ
binds it):
- `compose_ordered`: the rows are `Ordered`;
- `Rows.compose_eval`: satisfied composed rows, with `one` at 1, carry the type's outputs (`Types.eval`) on the output
  columns. This is `Types.compose_sound` read through the logical order.
-/

open Flock Flock.Derive

namespace FlockSoundness

open Audit

namespace Compose

/-- **A unit's rows in logical order**, each at its physical column: `order` lists the columns, the unit's `nIn` input
bits first, and `row c` is column `c`'s row as the block holds it, or `none` for a row that copies the block's `one`. -/
structure Logical where
  nIn : ℕ
  order : List ℕ
  row : ℕ → Option (List ℕ × List ℕ)

variable (L : Logical)

/-- Physical column `c`'s logical column: its place in `order` (`|order|`, the block's `one`, for a column not in it). -/
def lidx (c : ℕ) : ℕ := L.order.idxOf c

/-- Computed row `k`'s columns, `A` and `B`, as logical columns: the row at `order[nIn + k]` with each column mapped by
`lidx`, or `one` on both sides for a row that copies it. -/
def lrow (k : ℕ) : List ℕ × List ℕ :=
  match L.row (L.order.getD (L.nIn + k) 0) with
  | none => ([L.order.length], [L.order.length])
  | some p => (p.1.map (lidx L), p.2.map (lidx L))

/-- **The logical order**: `order` starts with the `nIn` input columns, each computed row reads only logical columns
before its own, or `one`, and every column a row reads is in `order`. -/
structure Ordered : Prop where
  le : L.nIn ≤ L.order.length
  topo : ∀ k < L.order.length - L.nIn, ∀ c ∈ (lrow L k).1 ++ (lrow L k).2, c < L.nIn + k ∨ c = L.order.length
  mem : ∀ k < L.order.length - L.nIn, ∀ p, L.row (L.order.getD (L.nIn + k) 0) = some p →
    ∀ c ∈ p.1 ++ p.2, c ∈ L.order

variable {L}

theorem lrow_none {k : ℕ} (h : L.row (L.order.getD (L.nIn + k) 0) = none) :
    lrow L k = ([L.order.length], [L.order.length]) := by
  unfold lrow; rw [h]

theorem lrow_some {k : ℕ} {p : List ℕ × List ℕ} (h : L.row (L.order.getD (L.nIn + k) 0) = some p) :
    lrow L k = (p.1.map (lidx L), p.2.map (lidx L)) := by
  unfold lrow; rw [h]

theorem lidx_getD (hn : L.order.Nodup) {ℓ : ℕ} (hℓ : ℓ < L.order.length) : lidx L (L.order.getD ℓ 0) = ℓ := by
  unfold lidx
  rw [List.getD_eq_getElem _ _ hℓ]
  exact hn.idxOf_getElem ℓ hℓ

theorem lidx_lt_of_mem_take {c ℓ : ℕ} (hc : c ∈ L.order.take ℓ) : lidx L c < ℓ := by
  unfold lidx
  rw [← List.take_append_drop ℓ L.order, List.idxOf_append_of_mem hc]
  exact lt_of_lt_of_le (List.idxOf_lt_length_iff.2 hc) (List.length_take_le _ _)

/-- **`Ordered` from reads earlier in `order`**: each computed row, at `order[ℓ]` for `ℓ ≥ nIn`, reads only columns that
come before it in `order`. -/
theorem ordered_of_take (hle : L.nIn ≤ L.order.length)
    (hread : ∀ ℓ < L.order.length, L.nIn ≤ ℓ → ∀ p, L.row (L.order.getD ℓ 0) = some p →
      ∀ c ∈ p.1 ++ p.2, c ∈ L.order.take ℓ) : Ordered L := by
  refine ⟨hle, fun k hk c hc => ?_, fun k hk p hp c hc => List.mem_of_mem_take (hread _ (by omega) (by omega) p hp c hc)⟩
  cases hr : L.row (L.order.getD (L.nIn + k) 0) with
  | none =>
    rw [lrow_none hr] at hc
    simp only [List.mem_append, List.mem_singleton, or_self] at hc
    exact Or.inr hc
  | some p =>
    rw [lrow_some hr] at hc
    simp only [List.mem_append, List.mem_map] at hc
    rcases hc with ⟨x, hx, rfl⟩ | ⟨x, hx, rfl⟩
    · exact Or.inl (lidx_lt_of_mem_take (hread _ (by omega) (by omega) p hr x (List.mem_append_left _ hx)))
    · exact Or.inl (lidx_lt_of_mem_take (hread _ (by omega) (by omega) p hr x (List.mem_append_right _ hx)))

end Compose

open Compose

/-- **A unit's rows, in logical order**: the audit's `Rows`, whose columns are `order`'s, then `one`. -/
def Rows.compose (L : Logical) (h : Ordered L) : Rows where
  nIn := L.nIn
  nComp := L.order.length - L.nIn
  a k := (lrow L k).1.pmap (fun c hc => ⟨c, hc⟩) fun c hc => by
    have := h.le
    rcases h.topo k k.isLt c (List.mem_append_left _ hc) with h' | h' <;> omega
  b k := (lrow L k).2.pmap (fun c hc => ⟨c, hc⟩) fun c hc => by
    have := h.le
    rcases h.topo k k.isLt c (List.mem_append_right _ hc) with h' | h' <;> omega
  topo k c hc := by
    have := h.le
    rcases List.mem_append.1 hc with hc | hc <;>
    · obtain ⟨c₀, hc₀, rfl⟩ := List.mem_pmap.1 hc
      rcases h.topo k k.isLt c₀
          (by first | exact List.mem_append_left _ hc₀ | exact List.mem_append_right _ hc₀) with h' | h'
      · exact Or.inl h'
      · exact Or.inr (by show c₀ = _; omega)

namespace Rows

variable {L : Logical} {h : Ordered L}

theorem compose_nIn : (Rows.compose L h).nIn = L.nIn := rfl

theorem compose_w : (Rows.compose L h).w = L.order.length + 1 := by
  show L.nIn + (L.order.length - L.nIn) + 1 = _
  have := h.le
  omega

theorem compose_comp (k : Fin (Rows.compose L h).nComp) : ((Rows.compose L h).comp k).val = L.nIn + k.val := rfl

theorem compose_one : (Rows.compose L h).one.val = L.order.length := by
  show L.nIn + (L.order.length - L.nIn) = _
  have := h.le
  omega

theorem compose_a (k : Fin (Rows.compose L h).nComp) : ((Rows.compose L h).a k).map Fin.val = (lrow L k).1 := by
  simp [Rows.compose, List.map_pmap]

theorem compose_b (k : Fin (Rows.compose L h).nComp) : ((Rows.compose L h).b k).map Fin.val = (lrow L k).2 := by
  simp [Rows.compose, List.map_pmap]

end Rows

/-! ## Placing the composed rows -/

section Place

variable (S : Model.Statement) (L : Logical) (h : Ordered L) (pos : ℕ → ℕ)
  (hb : ∀ ℓ < L.order.length, pos (L.order.getD ℓ 0) < 2 ^ S.kLog)

/-- **Where each logical column sits in the block**: its physical column's position `pos`, and `one` at the pin. -/
def composePlace (ℓ : Fin (Rows.compose L h).w) : Fin (2 ^ S.kLog) :=
  if hℓ : ℓ.val < L.order.length then ⟨pos (L.order.getD ℓ.val 0), hb _ hℓ⟩ else S.pin

variable {S L h pos hb}

theorem composePlace_val (ℓ : Fin (Rows.compose L h).w) :
    (composePlace S L h pos hb ℓ).val = if ℓ.val < L.order.length then pos (L.order.getD ℓ.val 0) else S.pin.val := by
  unfold composePlace
  split <;> simp_all

/-- A row's logical reads, placed, are its physical reads' positions. -/
theorem map_composePlace (M : List (Fin (Rows.compose L h).w)) (cs : List ℕ) (hcs : ∀ c ∈ cs, c ∈ L.order)
    (hM : M.map Fin.val = cs.map (lidx L)) : (M.map (composePlace S L h pos hb)).map Fin.val = cs.map pos := by
  have : (Fin.val ∘ composePlace S L h pos hb) =
      (fun ℓ => if ℓ < L.order.length then pos (L.order.getD ℓ 0) else S.pin.val) ∘ Fin.val := by
    funext ℓ; exact composePlace_val ℓ
  rw [List.map_map, this, ← List.map_map, hM, List.map_map]
  refine List.map_congr_left fun c hc => ?_
  have hlt : lidx L c < L.order.length := List.idxOf_lt_length_iff.2 (hcs c hc)
  simp only [Function.comp]
  rw [ite_eq_left hlt]
  unfold lidx at hlt ⊢
  rw [List.getD_eq_getElem _ _ hlt, List.getElem_idxOf]

theorem count_composePlace (M : List (Fin (Rows.compose L h).w)) (cs : List ℕ) (hcs : ∀ c ∈ cs, c ∈ L.order)
    (hM : M.map Fin.val = cs.map (lidx L)) (j : Fin (2 ^ S.kLog)) :
    (((M.map (composePlace S L h pos hb)).count j : ℕ) : ZMod 2) = (((cs.map pos).count j.val : ℕ) : ZMod 2) := by
  rw [← List.count_map_of_injective _ Fin.val Fin.val_injective, map_composePlace M cs hcs hM]

/-- **The composed rows, placed.** Suppose the block's matrices, at each computed row's position, read exactly the
positions of the row's columns, or the pin for a row that copies `one`, and the pin's row reads the pin. Then
`Rows.compose` is placed at `composePlace`. -/
theorem placement_of_realizes
    (hrowA : ∀ k < L.order.length - L.nIn, ∀ p, L.row (L.order.getD (L.nIn + k) 0) = some p →
      ∀ r j : Fin (2 ^ S.kLog), r.val = pos (L.order.getD (L.nIn + k) 0) →
        S.A₀ r j = (((p.1.map pos).count j.val : ℕ) : ZMod 2))
    (hrowB : ∀ k < L.order.length - L.nIn, ∀ p, L.row (L.order.getD (L.nIn + k) 0) = some p →
      ∀ r j : Fin (2 ^ S.kLog), r.val = pos (L.order.getD (L.nIn + k) 0) →
        S.B₀ r j = (((p.2.map pos).count j.val : ℕ) : ZMod 2))
    (honeA : ∀ k < L.order.length - L.nIn, L.row (L.order.getD (L.nIn + k) 0) = none →
      ∀ r j : Fin (2 ^ S.kLog), r.val = pos (L.order.getD (L.nIn + k) 0) → S.A₀ r j = if j = S.pin then 1 else 0)
    (honeB : ∀ k < L.order.length - L.nIn, L.row (L.order.getD (L.nIn + k) 0) = none →
      ∀ r j : Fin (2 ^ S.kLog), r.val = pos (L.order.getD (L.nIn + k) 0) → S.B₀ r j = if j = S.pin then 1 else 0)
    (hpinA : ∀ j, S.A₀ S.pin j = if j = S.pin then 1 else 0)
    (hpinB : ∀ j, S.B₀ S.pin j = if j = S.pin then 1 else 0) :
    Placement S (Rows.compose L h) (composePlace S L h pos hb) := by
  have hle := h.le
  have hone : composePlace S L h pos hb (Rows.compose L h).one = S.pin := by
    unfold composePlace; rw [dite_eq_right (by rw [Rows.compose_one]; omega)]
  have hcomp : ∀ k : Fin (Rows.compose L h).nComp,
      (composePlace S L h pos hb ((Rows.compose L h).comp k)).val = pos (L.order.getD (L.nIn + k.val) 0) := by
    intro k
    have := k.isLt
    rw [composePlace_val, Rows.compose_comp,
      ite_eq_left (by show L.nIn + k.val < _; simp only [Rows.compose] at this; omega)]
  -- a row that copies `one` reads the pin
  have hcopy : ∀ M : List (Fin (Rows.compose L h).w), M.map Fin.val = [L.order.length] →
      M.map (composePlace S L h pos hb) = [S.pin] := by
    intro M hM
    cases M with
    | nil => cases hM
    | cons x M' =>
      cases M' with
      | nil =>
        simp only [List.map_cons, List.map_nil, List.cons.injEq, and_true] at hM ⊢
        have : x = (Rows.compose L h).one := Fin.ext (by rw [Rows.compose_one]; exact hM)
        rw [this, hone]
      | cons y M'' => simp at hM
  have key : ∀ (Mx : Matrix (Fin (2 ^ S.kLog)) (Fin (2 ^ S.kLog)) (ZMod 2)) (sel : List ℕ × List ℕ → List ℕ)
      (hsel : sel = Prod.fst ∨ sel = Prod.snd) (f : Fin (Rows.compose L h).nComp → List (Fin (Rows.compose L h).w))
      (hf : ∀ k, (f k).map Fin.val = sel (lrow L k.val))
      (hrow : ∀ k < L.order.length - L.nIn, ∀ p, L.row (L.order.getD (L.nIn + k) 0) = some p →
        ∀ r j : Fin (2 ^ S.kLog), r.val = pos (L.order.getD (L.nIn + k) 0) →
          Mx r j = ((((sel p).map pos).count j.val : ℕ) : ZMod 2))
      (hcst : ∀ k < L.order.length - L.nIn, L.row (L.order.getD (L.nIn + k) 0) = none →
        ∀ r j : Fin (2 ^ S.kLog), r.val = pos (L.order.getD (L.nIn + k) 0) → Mx r j = if j = S.pin then 1 else 0)
      (k : Fin (Rows.compose L h).nComp) (j : Fin (2 ^ S.kLog)),
      Mx (composePlace S L h pos hb ((Rows.compose L h).comp k)) j =
        (((f k).map (composePlace S L h pos hb)).count j : ZMod 2) := by
    intro Mx sel hsel f hf hrow hcst k j
    have hk : k.val < L.order.length - L.nIn := k.isLt
    cases hr : L.row (L.order.getD (L.nIn + k.val) 0) with
    | none =>
      rw [hcst k.val hk hr _ j (hcomp k)]
      have hM : (f k).map Fin.val = [L.order.length] := by
        rw [hf, lrow_none hr]; rcases hsel with rfl | rfl <;> rfl
      rw [hcopy (f k) hM]
      by_cases hj : j = S.pin
      · simp [hj]
      · simp [hj, Ne.symm hj]
    | some p =>
      rw [hrow k.val hk p hr _ j (hcomp k)]
      have hmem := h.mem k.val hk p hr
      have hsub : ∀ c ∈ sel p, c ∈ L.order := by
        intro c hc
        rcases hsel with rfl | rfl
        · exact hmem c (List.mem_append_left _ hc)
        · exact hmem c (List.mem_append_right _ hc)
      have hM : (f k).map Fin.val = (sel p).map (lidx L) := by
        rw [hf, lrow_some hr]; rcases hsel with rfl | rfl <;> rfl
      exact (count_composePlace (f k) (sel p) hsub hM j).symm
  exact {
    hA := key S.A₀ Prod.fst (Or.inl rfl) (Rows.compose L h).a (fun k => Rows.compose_a k) hrowA honeA
    hB := key S.B₀ Prod.snd (Or.inr rfl) (Rows.compose L h).b (fun k => Rows.compose_b k) hrowB honeB
    hA₁ := fun j => by rw [hone]; exact hpinA j
    hB₁ := fun j => by rw [hone]; exact hpinB j }

end Place

/-! ## Values on the physical columns -/

namespace Compose

variable {L : Logical} {h : Ordered L}

/-- **The composed rows' values on the physical columns**: a column in `order` carries its logical column's value, and
every other column 0. -/
def physical (z : Fin (Rows.compose L h).w → Bool) (c : ℕ) : Bool :=
  if hc : lidx L c < L.order.length then z ⟨lidx L c, by rw [Rows.compose_w]; omega⟩ else false

theorem physical_getD (hn : L.order.Nodup) (z : Fin (Rows.compose L h).w → Bool) {ℓ : ℕ} (hℓ : ℓ < L.order.length) :
    physical z (L.order.getD ℓ 0) = z ⟨ℓ, by rw [Rows.compose_w]; omega⟩ := by
  unfold physical
  split
  · exact congrArg z (Fin.ext (lidx_getD hn hℓ))
  · rename_i h'; exact absurd (by rw [lidx_getD hn hℓ]; exact hℓ) h'

theorem physical_not_mem (z : Fin (Rows.compose L h).w → Bool) {c : ℕ} (hc : c ∉ L.order) : physical z c = false := by
  unfold physical
  rw [dite_eq_right (by unfold lidx; rw [List.idxOf_eq_length hc]; omega)]

/-- A row's logical reads, on the composed values, are its physical reads on the physical values. -/
theorem xorSum_physical (z : Fin (Rows.compose L h).w → Bool) (M : List (Fin (Rows.compose L h).w)) (cs : List ℕ)
    (hM : M.map Fin.val = cs.map (lidx L)) (hcs : ∀ c ∈ cs, c ∈ L.order) : xorSum M z = xorSum cs (physical z) := by
  let zN : ℕ → Bool := fun ℓ => if hℓ : ℓ < (Rows.compose L h).w then z ⟨ℓ, hℓ⟩ else false
  have h1 : xorSum M z = xorSum (M.map Fin.val) zN := by
    rw [xorSum_map]; exact xorSum_congr fun x _ => by simp only [Function.comp, zN]; rw [dite_eq_left x.isLt]
  rw [h1, hM, xorSum_map]
  refine xorSum_congr fun c hc => ?_
  have hlt : lidx L c < L.order.length := List.idxOf_lt_length_iff.2 (hcs c hc)
  have hw : lidx L c < (Rows.compose L h).w := by rw [Rows.compose_w]; omega
  simp only [Function.comp, zN, physical, hlt, hw, dite_true]

end Compose

/-! ## Flat types: the rows `derive` lays out -/

namespace Compose

variable (d : Derived)

/-- **A flat type's rows in logical order**: `derive`'s `order` and rows, the constant copying `one` (Δ binds the unit's
constant to the pin). -/
abbrev ofDerived : Logical :=
  ⟨d.nIn, d.order, fun c => if c = d.geo.const then none else some (d.rows.getD c ([], []))⟩

theorem order_length : d.order.length = d.nIn + 1 + d.geo.nAnd + d.nOut := by
  simp [Derived.order]
  omega

theorem order_getD_in {i : ℕ} (hi : i < d.nIn) : d.order.getD i 0 = d.inCol i := by
  simp [Derived.order, List.getD_eq_getElem?_getD, List.getElem?_append_left, hi]

theorem order_getD_const : d.order.getD d.nIn 0 = d.geo.const := by
  simp [Derived.order, List.getD_eq_getElem?_getD]

theorem order_getD_and {k : ℕ} (hk : k < d.geo.nAnd) : d.order.getD (d.nIn + 1 + k) 0 = d.geo.andBase + k := by
  unfold Derived.order
  rw [List.getD_eq_getElem?_getD, List.append_assoc, List.append_assoc,
    List.getElem?_append_right (by simp; omega), List.length_map, List.length_range,
    show d.nIn + 1 + k - d.nIn = k + 1 by omega, List.singleton_append, List.getElem?_cons_succ,
    List.getElem?_append_left (by simpa using hk)]
  simp [hk]

theorem order_getD_out {o : ℕ} (ho : o < d.nOut) :
    d.order.getD (d.nIn + 1 + d.geo.nAnd + o) 0 = d.outCol o := by
  unfold Derived.order
  rw [List.getD_eq_getElem?_getD, List.getElem?_append_right (by simp; omega)]
  simp only [List.length_append, List.length_map, List.length_range, List.length_singleton,
    show d.nIn + 1 + d.geo.nAnd + o - (d.nIn + 1 + d.geo.nAnd) = o by omega, List.getElem?_map,
    List.getElem?_range ho, Option.map_some, Option.getD_some]

variable {d}

theorem ofDerived_row {c : ℕ} (hc : c ≠ d.geo.const) : (ofDerived d).row c = some (d.rows.getD c ([], [])) := by
  show (if c = d.geo.const then _ else _) = _
  rw [ite_eq_right hc]

theorem ofDerived_row_const {c : ℕ} (hc : c = d.geo.const) : (ofDerived d).row c = none := by
  show (if c = d.geo.const then _ else _) = _
  rw [ite_eq_left hc]

end Compose

section flat

variable {t : CircuitType} {inSplit outSplit : List ℕ} {d : Derived}

/-- The logical columns a flat type's rows read: an input wire's is its bit, an AND wire's its row after the constant,
and the constant's is its place after the inputs. -/
theorem lidx_reads (hd : derive t inSplit outSplit = .ok d) :
    (∀ w < t.inBits + d.geo.nAnd, lidx (ofDerived d) (d.geo.col t.inBits w) =
      if w < t.inBits then w else d.nIn + 1 + (w - t.inBits)) ∧ lidx (ofDerived d) d.geo.const = d.nIn := by
  have hn : (ofDerived d).order.Nodup := Types.order_nodup hd
  have hlen := order_length d
  obtain ⟨-, -, hdd⟩ := Types.derive_ok hd
  have hnIn : d.nIn = t.inBits := by rw [hdd]
  refine ⟨fun w hw => ?_, by rw [← order_getD_const d, lidx_getD hn (by simp; omega)]⟩
  by_cases hwi : w < t.inBits
  · rw [ite_eq_left hwi]
    have : d.geo.col t.inBits w = d.order.getD w 0 := by
      rw [order_getD_in d (by omega)]; simp [Geo.col, hwi, Derived.inCol]
    rw [this, lidx_getD hn (by simp; omega)]
  · rw [ite_eq_right hwi]
    have : d.geo.col t.inBits w = d.order.getD (d.nIn + 1 + (w - t.inBits)) 0 := by
      rw [order_getD_and d (by omega)]; simp [Geo.col, hwi]
    rw [this, lidx_getD hn (by simp; omega)]

theorem lidx_form (hd : derive t inSplit outSplit = .ok d) {f : Form} {m : ℕ} (hm : m ≤ d.geo.nAnd)
    (hf : ∀ w ∈ f.ws.toList, w < t.inBits + m) :
    ∀ x ∈ d.geo.formCols t.inBits f, lidx (ofDerived d) x < d.nIn + 1 + m := by
  obtain ⟨hwire, hconst⟩ := lidx_reads hd
  obtain ⟨-, -, hdd⟩ := Types.derive_ok hd
  have hnIn : d.nIn = t.inBits := by rw [hdd]
  intro x hx
  simp only [Geo.formCols, List.mem_append, List.mem_map] at hx
  rcases hx with ⟨w, hw, rfl⟩ | hx
  · have := hf w hw
    rw [hwire w (by omega)]
    split <;> omega
  · split at hx
    · rw [List.mem_singleton] at hx; rw [hx, hconst]; omega
    · cases hx

/-- **A flat type's composed rows read only earlier columns**, or `one`: its constant copies `one`; an AND row reads the
inputs, the constant and the ANDs before it; an output copy reads any wire and the constant. -/
theorem compose_ordered (hd : derive t inSplit outSplit = .ok d) : Ordered (ofDerived d) := by
  have hlen := order_length d
  have hconst := (lidx_reads hd).2
  obtain ⟨hflat, hgeo, hdd⟩ := Types.derive_ok hd
  have hnIn : d.nIn = t.inBits := by rw [hdd]
  have hnOut : d.nOut = t.outBits := by rw [hdd]
  have hG : d.geo = geo t inSplit outSplit := by rw [hdd]
  obtain ⟨hc, -, -, hout, -⟩ := Types.geoOk_facts hgeo
  rw [← hG] at hc hout
  have hrows : ∀ r ≤ d.geo.const, d.rows.getD r ([], []) = d.geo.rowAt t r := by
    intro r hr
    rw [hdd] at hr ⊢
    simp only [List.getD_eq_getElem?_getD]
    rw [Types.rows_get hr]
    rfl
  have hnAnd : d.geo.nAnd = t.items.size := by rw [hG]; rfl
  -- row `k`'s physical columns, other than the constant's: each maps to a logical column below `nIn + k`
  have hreads : ∀ k < d.order.length - d.nIn, d.order.getD (d.nIn + k) 0 ≠ d.geo.const →
      ∀ x ∈ (d.rows.getD (d.order.getD (d.nIn + k) 0) ([], [])).1 ++
        (d.rows.getD (d.order.getD (d.nIn + k) 0) ([], [])).2, lidx (ofDerived d) x < d.nIn + k := by
    intro k hk hne x hx
    have h0 : k ≠ 0 := by rintro rfl; rw [Nat.add_zero, order_getD_const] at hne; exact hne rfl
    by_cases h1 : k < 1 + d.geo.nAnd
    · obtain ⟨k', rfl⟩ : ∃ k', k = 1 + k' := ⟨k - 1, by omega⟩
      have hk' : k' < d.geo.nAnd := by omega
      rw [show d.nIn + (1 + k') = d.nIn + 1 + k' by omega, order_getD_and d hk', hrows _ (by omega)] at hx
      obtain ⟨a, b, hab, ha, hb⟩ := Types.flatOk_items hflat k' (by omega)
      rw [Types.rowAt_and hk' hc (by rw [Array.getElem?_eq_getElem (by omega), hab])] at hx
      rcases List.mem_append.1 hx with hx | hx
      · have := lidx_form hd (by omega) ha x hx; omega
      · have := lidx_form hd (by omega) hb x hx; omega
    · obtain ⟨o, rfl⟩ : ∃ o, k = 1 + d.geo.nAnd + o := ⟨k - (1 + d.geo.nAnd), by omega⟩
      have ho : o < d.nOut := by omega
      obtain ⟨c₀, hc₀, hc₁, hc₂, hbc⟩ := hout o (by omega)
      have hoc : d.outCol o = c₀ := by simp [Derived.outCol, hc₀]
      rw [show d.nIn + (1 + d.geo.nAnd + o) = d.nIn + 1 + d.geo.nAnd + o by omega, order_getD_out d ho, hoc,
        hrows _ (by omega), Types.rowAt_out_some hc₁ hc₂ hbc] at hx
      obtain ⟨hosz, hof⟩ := Types.flatOk_outputs hflat
      rcases List.mem_append.1 hx with hx | hx
      · have := lidx_form hd (le_refl _) (f := t.outputs[o]?.getD default) (fun w hw => by
          rw [Array.getElem?_eq_getElem (by omega),
            Option.getD_some] at hw
          rw [hnAnd]; exact hof o (by omega) w hw) x hx
        omega
      · rw [List.mem_singleton] at hx; rw [hx, hconst]; omega
  refine ⟨by show d.nIn ≤ d.order.length; omega, fun k hk c hc' => ?_, fun k hk p hp x hx => ?_⟩
  · show c < d.nIn + k ∨ c = d.order.length
    by_cases hck : d.order.getD (d.nIn + k) 0 = d.geo.const
    · rw [lrow_none (ofDerived_row_const hck)] at hc'
      simp only [List.mem_append, List.mem_singleton, or_self] at hc'
      exact Or.inr hc'
    · rw [lrow_some (ofDerived_row hck)] at hc'
      simp only [List.mem_append, List.mem_map] at hc'
      rcases hc' with ⟨x, hx, rfl⟩ | ⟨x, hx, rfl⟩
      · exact Or.inl (hreads k hk hck x (List.mem_append_left _ hx))
      · exact Or.inl (hreads k hk hck x (List.mem_append_right _ hx))
  · have hp' : (ofDerived d).row (d.order.getD (d.nIn + k) 0) = some p := hp
    have hk' : k < d.order.length - d.nIn := hk
    have hck : d.order.getD (d.nIn + k) 0 ≠ d.geo.const := by
      intro hck; rw [ofDerived_row_const hck] at hp'; cases hp'
    rw [ofDerived_row hck, Option.some.injEq] at hp'
    subst hp'
    have : d.order.idxOf x < d.nIn + k := hreads k hk' hck x hx
    exact List.idxOf_lt_length_iff.1 (show d.order.idxOf x < d.order.length by omega)

/-! ## What a flat type's composed rows compute -/

namespace Compose

/-- The wire values below `w + |items|` depend only on the values below `w`. -/
theorem run_congr : ∀ (items : List Item) (v v' : ℕ → Bool) (w : ℕ), Types.ReadsEarlier w items →
    (∀ j < w, v j = v' j) → ∀ j < w + items.length, Types.run v w items j = Types.run v' w items j
  | [], _, _, _, _, hv, j, hj => hv j (by simpa using hj)
  | it :: rest, v, v', w, hr, hv, j, hj => by
    simp only [Types.run]
    refine run_congr rest _ _ (w + 1) (fun k hk f hf x hx => ?_) (fun j' hj' => ?_) j (by simp at hj; omega)
    · have := hr (k + 1) (by simp; omega) f (by simpa using hf) x hx; omega
    · by_cases hjw : j' = w
      · rw [ite_eq_left hjw, ite_eq_left hjw]
        exact Types.itemVal_congr fun f hf x hx => hv x (by have := hr 0 (by simp) f (by simpa using hf) x hx; omega)
      · rw [ite_eq_right hjw, ite_eq_right hjw]; exact hv j' (by omega)

/-- A flat type's outputs depend only on its input bits. -/
theorem eval_congr {t : CircuitType} (hflat : flatOk t = true) {x x' : ℕ → Bool} (hx : ∀ i < t.inBits, x i = x' i) :
    Types.eval t x = Types.eval t x' := by
  obtain ⟨-, hof⟩ := Types.flatOk_outputs hflat
  unfold Types.eval
  refine List.map_congr_left fun f hf => ?_
  obtain ⟨o, ho, rfl⟩ := List.mem_iff_getElem.1 hf
  have ho' : o < t.outputs.size := by simpa using ho
  refine Types.formVal_congr fun y hy => ?_
  rw [Array.getElem_toList] at hy
  exact run_congr _ _ _ _ (Types.flatOk_reads hflat) hx y (by have := hof o ho' y hy; simpa using this)

/-- **The composed rows' values, on the physical columns, satisfy `derive`'s rows**: an input bit's row and the constant's
read only their own column, a column outside `order` is a zero row carrying 0, and every other row is a composed row. -/
theorem sat_physical (hd : derive t inSplit outSplit = .ok d) {h : Ordered (ofDerived d)}
    {z : Fin (Rows.compose (ofDerived d) h).w → Bool} (hz : (Rows.compose (ofDerived d) h).Sat z) :
    Types.Sat d.rows (physical z) := by
  have hn : (ofDerived d).order.Nodup := Types.order_nodup hd
  have hlen := order_length d
  have hle := h.le
  obtain ⟨-, hgeo, hdd⟩ := Types.derive_ok hd
  have hnIn : d.nIn = t.inBits := by rw [hdd]
  have hG : d.geo = geo t inSplit outSplit := by rw [hdd]
  obtain ⟨hc, hin, -, -, -⟩ := Types.geoOk_facts hgeo
  rw [← hG] at hc hin
  have hrows : ∀ r ≤ d.geo.const, d.rows.getD r ([], []) = d.geo.rowAt t r := by
    intro r hr
    rw [hdd] at hr ⊢
    simp only [List.getD_eq_getElem?_getD]
    rw [Types.rows_get hr]
    rfl
  have hself : ∀ r, physical z r = (xorSum [r] (physical z) && xorSum [r] (physical z)) := by
    intro r; simp [xorSum]
  intro r p hp
  have hpr : p = d.rows.getD r ([], []) := by rw [List.getD_eq_getElem?_getD, hp, Option.getD_some]
  by_cases hm : r ∈ d.order
  · have hlt := List.idxOf_lt_length_iff.2 hm
    obtain ⟨ℓ, hℓ, rfl⟩ : ∃ ℓ, ℓ < d.order.length ∧ d.order.getD ℓ 0 = r :=
      ⟨d.order.idxOf r, hlt, by rw [List.getD_eq_getElem _ _ hlt]; exact List.getElem_idxOf hlt⟩
    by_cases hin' : ℓ < d.nIn
    · obtain ⟨c, hcO, hc1, hbc⟩ := hin ℓ (by omega)
      have hcol : d.order.getD ℓ 0 = c := by rw [order_getD_in d hin']; simp [Derived.inCol, hcO]
      rw [hpr, hcol, hrows c (by omega), Types.rowAt_in_some hc1 hc hbc]
      exact hself c
    · obtain ⟨k, rfl⟩ : ∃ k, ℓ = d.nIn + k := ⟨ℓ - d.nIn, by omega⟩
      by_cases hck : d.order.getD (d.nIn + k) 0 = d.geo.const
      · rw [hpr, hck, hrows _ le_rfl, Types.rowAt_const]
        exact hself _
      · have hk : k < (Rows.compose (ofDerived d) h).nComp := by show k < d.order.length - d.nIn; omega
        have hzk := hz ⟨k, hk⟩
        have hrow := ofDerived_row hck
        have hmem := h.mem k hk _ hrow
        rw [xorSum_physical z _ _ (by rw [Rows.compose_a, lrow_some hrow])
            (fun c hc => hmem c (List.mem_append_left _ hc)),
          xorSum_physical z _ _ (by rw [Rows.compose_b, lrow_some hrow])
            (fun c hc => hmem c (List.mem_append_right _ hc))] at hzk
        rw [physical_getD hn z hℓ, hpr]
        exact hzk
  · have hp0 : p = ([], []) := by
      by_contra hne; exact hm (Types.order_covers hd r p hp hne)
    rw [hp0, physical_not_mem z hm]
    rfl

end Compose

/-- **The composed rows compute the type.** Column values that satisfy `Rows.compose` of a flat type's `derive`, with
`one` at 1, carry on output `o`'s logical column (after the inputs, the constant and the ANDs) the type's output `o` on
the input columns' values. -/
theorem Rows.compose_eval (hd : derive t inSplit outSplit = .ok d) {h : Ordered (ofDerived d)}
    {z : Fin (Rows.compose (ofDerived d) h).w → Bool} (hz : (Rows.compose (ofDerived d) h).Sat z)
    (h1 : z (Rows.compose (ofDerived d) h).one = true) {x : ℕ → Bool}
    (hx : ∀ i : Fin (Rows.compose (ofDerived d) h).w, i.val < d.nIn → x i.val = z i) {o : ℕ}
    (ho : o < t.outputs.size) (ℓ : Fin (Rows.compose (ofDerived d) h).w) (hℓ : ℓ.val = d.nIn + 1 + d.geo.nAnd + o) :
    z ℓ = (Types.eval t x).getD o false := by
  have hn : (ofDerived d).order.Nodup := Types.order_nodup hd
  have hlen := order_length d
  have hle := h.le
  obtain ⟨hflat, -, hdd⟩ := Types.derive_ok hd
  have hnIn : d.nIn = t.inBits := by rw [hdd]
  have hnOut : d.nOut = t.outBits := by rw [hdd]
  obtain ⟨hosz, -⟩ := Types.flatOk_outputs hflat
  -- the constant carries 1: its row copies `one`
  have hconst : physical z d.geo.const = true := by
    have h0 : 0 < (Rows.compose (ofDerived d) h).nComp := by show 0 < d.order.length - d.nIn; omega
    have hz0 := hz ⟨0, h0⟩
    have hlr : lrow (ofDerived d) 0 = ([d.order.length], [d.order.length]) :=
      lrow_none (ofDerived_row_const (by rw [Nat.add_zero]; exact order_getD_const d))
    have hone : ∀ M : List (Fin (Rows.compose (ofDerived d) h).w), M.map Fin.val = [d.order.length] →
        xorSum M z = true := by
      intro M hM
      cases M with
      | nil => cases hM
      | cons x M' =>
        cases M' with
        | nil =>
          simp only [List.map_cons, List.map_nil, List.cons.injEq, and_true] at hM
          have : x = (Rows.compose (ofDerived d) h).one := Fin.ext (by rw [Rows.compose_one]; exact hM)
          simp [xorSum, this, h1]
        | cons y M'' => simp at hM
    rw [hone _ (by rw [Rows.compose_a, hlr]), hone _ (by rw [Rows.compose_b, hlr])] at hz0
    rw [← order_getD_const d, physical_getD hn z (by simp; omega)]
    exact hz0
  have hmain := Types.compose_sound hd (physical z) (Compose.sat_physical hd hz) hconst o ho
  rw [← order_getD_out d (show o < d.nOut by omega), physical_getD hn z (by simp; omega)] at hmain
  have hb : d.nIn + 1 + d.geo.nAnd + o < (Rows.compose (ofDerived d) h).w := by rw [Rows.compose_w]; simp; omega
  rw [show ℓ = ⟨_, hb⟩ from Fin.ext hℓ, hmain]
  congr 1
  refine eval_congr hflat fun i hi => ?_
  rw [← order_getD_in d (show i < d.nIn by omega), physical_getD hn z (by simp; omega)]
  exact (hx ⟨i, by rw [Rows.compose_w]; simp; omega⟩ (by show i < d.nIn; omega)).symm

end flat

end FlockSoundness
