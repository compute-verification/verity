import Flock.DeriveCheck
import Proofs.Flock.Soundness.Types.Flat

/-!
# Derived rows compute their circuit type, over the type DAG

`Flock.DeriveAll.deriveChecked` is the function the verifier runs to lay a statement's circuit types out as rows (Project
doc `verified-lowering-design.md` §1.2, S3c): `deriveAll` builds every layout's rows, and its check walks each layout's type
item by item against them. From the walk alone:

* **`layout_sound`**: column values that satisfy a placed or inlined layout's rows as the verifier folds them
  (`Flock.DeriveAll.fullRow`: a read's product rows with their table-direct side), with its constant at 1, carry on each
  output column the layout's type's output (`evalT`: its callees by digest, its reads by the table's words) on its input
  columns;
* **`unit_sound`**: the same for the unit's rows as the block holds them (Δ applied, `Flock.DeriveAll.blockRow`), with the
  block's constant at 1.

The check also covers the unit's logical order (`Flock.DeriveAll.order`, the list `Rows.compose` reads), so
**`order_sound`** and **`inputs_self`** give its facts: no column repeats, the inputs come first with self rows, each row
after them reads only earlier columns (and `one`), and every column outside the order has a zero row or none.

So all of this holds of whatever `deriveAll` builds: nothing here is about how it builds. The flat case is `compose_sound`
(`Types/Flat.lean`), and on a flat type `evalT` is its `eval` (**`evalT_flat`**), so there the two say the same. Reads are
checked inline (the decoders' one-hot minterms, the product rows' table-direct side, the output rows, bits constant across
the table) and placed: a placed read's segment embeds a generated `table/v2` layout, checked the same way in its own
coordinates, and **`genOk_sound`** gives that its output columns carry the table's bits at the index its input columns
spell.
-/

namespace FlockSoundness.Types.Dag

open Flock Flock.DeriveAll Audit

/-- Rows given as a function of the column, satisfied by column values `z`. -/
def SatAt (R : ℕ → Option (List ℕ × List ℕ)) (z : ℕ → Bool) : Prop :=
  ∀ c r, R c = some r → z c = (xorSum r.1 z && xorSum r.2 z)

theorem xorSum_nil {ι : Type} (v : ι → Bool) : xorSum ([] : List ι) v = false := rfl
theorem xorSum_cons {ι : Type} (a : ι) (l : List ι) (v : ι → Bool) :
    xorSum (a :: l) v = Bool.xor (v a) (xorSum l v) := rfl

theorem xorSum_perm {l m : List Nat} (h : l.Perm m) (v : ℕ → Bool) : xorSum l v = xorSum m v := by
  induction h with
  | nil => rfl
  | cons x _ ih => rw [xorSum_cons, xorSum_cons, ih]
  | swap x y l => rw [xorSum_cons, xorSum_cons, xorSum_cons, xorSum_cons, Bool.xor_left_comm]
  | trans _ _ ih1 ih2 => rw [ih1, ih2]

/-! ## Column forms -/

/-- A column form's value: its columns' XOR, and its constant. -/
def cfVal (z : ℕ → Bool) (f : CF) : Bool := Bool.xor (xorSum f.1 z) (f.2 % 2 == 1)

theorem xorSum_xorCols (z : ℕ → Bool) : ∀ a b : List Nat, xorSum (xorCols a b) z = Bool.xor (xorSum a z) (xorSum b z)
  | [], m => by simp [xorCols, xorSum]
  | a :: l, [] => by simp [xorCols, xorSum]
  | a :: l, b :: m => by
    unfold xorCols
    split
    · rw [xorSum_cons, xorSum_xorCols z l (b :: m)]
      simp only [xorSum_cons]
      cases z a <;> cases z b <;> cases xorSum l z <;> cases xorSum m z <;> rfl
    · split
      · rw [xorSum_cons, xorSum_xorCols z (a :: l) m]
        simp only [xorSum_cons]
        cases z a <;> cases z b <;> cases xorSum l z <;> cases xorSum m z <;> rfl
      · have hab : a = b := by omega
        subst hab
        rw [xorSum_xorCols z l m]
        simp only [xorSum_cons]
        cases z a <;> cases xorSum l z <;> cases xorSum m z <;> rfl
termination_by a b => a.length + b.length

theorem mod2_xor (x y : Nat) : ((x + y) % 2 % 2 == 1) = Bool.xor (x % 2 == 1) (y % 2 == 1) := by
  rcases Nat.mod_two_eq_zero_or_one x with hx | hx <;> rcases Nat.mod_two_eq_zero_or_one y with hy | hy <;>
    simp [Nat.add_mod, hx, hy]

theorem cfVal_xorCF (z : ℕ → Bool) (a b : CF) : cfVal z (xorCF a b) = Bool.xor (cfVal z a) (cfVal z b) := by
  unfold cfVal xorCF
  simp only
  rw [xorSum_xorCols, mod2_xor]
  cases xorSum a.1 z <;> cases xorSum b.1 z <;> cases (a.2 % 2 == 1) <;> cases (b.2 % 2 == 1) <;> rfl

theorem cfVal_foldl (z : ℕ → Bool) (wire : Array CF) :
    ∀ (l : List Nat) (acc : CF), cfVal z (l.foldl (fun acc w => xorCF acc (wire.getD w ([], 0))) acc) =
      Bool.xor (cfVal z acc) (xorSum l fun w => cfVal z (wire.getD w ([], 0)))
  | [], acc => by simp [xorSum]
  | w :: l, acc => by
    rw [List.foldl_cons, cfVal_foldl z wire l, cfVal_xorCF, xorSum_cons, Bool.xor_assoc]

/-- A form over the caller's wires, as columns, is worth the form on the wires' values. -/
theorem cfVal_formOf (z : ℕ → Bool) (wire : Array CF) (f : Form) (hc : f.c ≤ 1) :
    cfVal z (formOf wire f) = formVal (fun w => cfVal z (wire.getD w ([], 0))) f := by
  unfold formOf formVal
  rw [← Array.foldl_toList, cfVal_foldl]
  have : cfVal z ([], f.c % 2) = decide (f.c = 1) := by
    unfold cfVal; rcases Nat.lt_or_ge f.c 1 with h | h
    · have : f.c = 0 := by omega
      simp [this, xorSum]
    · have : f.c = 1 := by omega
      simp [this, xorSum]
  rw [this, Bool.xor_comm]

theorem formOf_snd_lt (wire : Array CF) (f : Form) : (formOf wire f).2 < 2 := by
  unfold formOf
  rw [← Array.foldl_toList]
  suffices ∀ (l : List Nat) (acc : CF), acc.2 < 2 →
      (l.foldl (fun acc w => xorCF acc (wire.getD w ([], 0))) acc).2 < 2 from this _ _ (Nat.mod_lt _ (by omega))
  intro l
  induction l with
  | nil => intro acc h; exact h
  | cons w l ih => intro acc _; exact ih _ (Nat.mod_lt _ (by omega))

/-- A resolved row side, with the constant's column at 1, is worth its column form. -/
theorem xorSum_resolve (z : ℕ → Bool) (k : Nat) (hk : z k = true) (f : CF) (hf : f.2 < 2) :
    xorSum (resolve k (ent f)) z = cfVal z f := by
  unfold resolve ent cfVal
  rcases (show f.2 = 0 ∨ f.2 = 1 by omega) with h | h
  · simp [h]
  · simp only [h, beq_self_eq_true, ↓reduceIte, Nat.one_mod]
    rw [xorSum_perm (List.mergeSort_perm _ _), xorSum_append, xorSum_cons, xorSum_nil, hk]
    rfl

/-! ## What a type computes -/

/-- The number that bits `bs` spell, least significant first. -/
def bitsIdx : List Bool → Nat
  | [] => 0
  | b :: bs => (if b then 1 else 0) + 2 * bitsIdx bs

/-- The values an item appends to the wire values `vals`: an AND's one, a call's callee outputs (by `ev`), a read's table
bits. -/
def itemOut (types : String → Option CircuitType) (words : String → Option (Array Nat))
    (ev : CircuitType → (ℕ → Bool) → List Bool) (t : CircuitType) (vals : List Bool) : Item → List Bool
  | .and a b => [formVal (vals.getD · false) a && formVal (vals.getD · false) b]
  | .call k args =>
    match (t.calls[k]?).bind types with
    | some u => (List.range u.outBits).map fun o =>
        (ev u fun i => formVal (vals.getD · false) (args.getD i default)).getD o false
    | none => []
  | .read k idx =>
    match t.tables[k]? with
    | some (sha, _, vb) =>
      let i := bitsIdx (idx.toList.map (formVal (vals.getD · false)))
      (List.range vb).map fun j => (((words sha).getD #[]).getD i 0).testBit j
    | none => []
  | .bad _ => []

/-- The wire values once `items` of type `t` run after `vals`. -/
def runT (types : String → Option CircuitType) (words : String → Option (Array Nat))
    (ev : CircuitType → (ℕ → Bool) → List Bool) (t : CircuitType) : List Bool → List Item → List Bool
  | vals, [] => vals
  | vals, it :: rest => runT types words ev t (vals ++ itemOut types words ev t vals it) rest

/-- **What a circuit type computes** on inputs `x`, its callees by digest in `types`, its tables' words by SHA-512 in
`words`; `fuel` bounds the call depth. -/
def evalT (types : String → Option CircuitType) (words : String → Option (Array Nat)) :
    Nat → CircuitType → (ℕ → Bool) → List Bool
  | 0, _, _ => []
  | fuel + 1, t, x => t.outputs.toList.map (formVal ((runT types words (fun u y => evalT types words fuel u y) t
      ((List.range t.inBits).map x) t.items.toList).getD · false))

theorem evalT_congr (types : String → Option CircuitType) (words : String → Option (Array Nat)) (F : Nat)
    (u : CircuitType) {x y : ℕ → Bool} (h : ∀ i < u.inBits, x i = y i) : evalT types words F u x = evalT types words F u y := by
  cases F with
  | zero => rfl
  | succ F =>
    simp only [evalT]
    rw [List.map_congr_left fun i hi => h i (List.mem_range.mp hi)]

theorem evalT_getD (types : String → Option CircuitType) (words : String → Option (Array Nat)) (F : Nat)
    (u : CircuitType) (x : ℕ → Bool) {o : Nat} (ho : o < u.outputs.size) :
    (evalT types words (F + 1) u x).getD o false = formVal ((runT types words (fun u y => evalT types words F u y) u
      ((List.range u.inBits).map x) u.items.toList).getD · false) u.outputs[o] := by
  simp [evalT, List.getD_eq_getElem?_getD, ho]

/-! ## The wires' column forms carry the wire values -/

theorem getD_eq (a : Array CF) (w : Nat) : a.getD w ([], 0) = a[w]?.getD ([], 0) := by
  unfold Array.getD; split <;> simp_all

/-- The walk's invariant: wire `w`'s column form is worth wire `w`'s value. -/
def Inv (z : ℕ → Bool) (wire : Array CF) (vals : List Bool) : Prop :=
  wire.size = vals.length ∧ ∀ w, cfVal z (wire.getD w ([], 0)) = vals.getD w false

theorem cfVal_nil (z : ℕ → Bool) : cfVal z ([], 0) = false := rfl

theorem cfVal_col (z : ℕ → Bool) (c : Nat) : cfVal z ([c], 0) = z c := by
  simp [cfVal, xorSum]

theorem Inv.vals {z : ℕ → Bool} {wire : Array CF} {vals : List Bool} (h : Inv z wire vals) :
    (fun w => cfVal z (wire.getD w ([], 0))) = (vals.getD · false) := funext h.2

theorem Inv.append {z : ℕ → Bool} {wire X : Array CF} {vals Y : List Bool} (h : Inv z wire vals)
    (hs : X.size = Y.length) (hX : ∀ o, cfVal z (X.getD o ([], 0)) = Y.getD o false) :
    Inv z (wire ++ X) (vals ++ Y) := by
  refine ⟨by simp [h.1, hs], fun w => ?_⟩
  rw [getD_eq]
  by_cases hw : w < wire.size
  · rw [Array.getElem?_append_left hw, ← getD_eq, h.2 w, List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD,
      List.getElem?_append_left (by rw [← h.1]; exact hw)]
  · rw [Array.getElem?_append_right (by omega), ← getD_eq, hX, List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD,
      List.getElem?_append_right (by rw [← h.1]; omega), h.1]

theorem Inv.push {z : ℕ → Bool} {wire : Array CF} {vals : List Bool} (h : Inv z wire vals) (x : CF) (b : Bool)
    (hx : cfVal z x = b) : Inv z (wire.push x) (vals ++ [b]) := by
  rw [← Array.append_singleton]
  refine h.append (by simp) fun o => ?_
  rcases o with _ | o
  · simp [Array.getD, hx]
  · simp [Array.getD, cfVal_nil]

/-- A form over the caller's wires, as columns, is worth the form on the wire values. -/
theorem Inv.formOf {z : ℕ → Bool} {wire : Array CF} {vals : List Bool} (h : Inv z wire vals) (f : Form) (hc : f.c ≤ 1) :
    cfVal z (DeriveAll.formOf wire f) = formVal (vals.getD · false) f := by
  rw [cfVal_formOf z wire f hc, h.vals]

/-! ## What the check's row tests give -/

/-- On column values `z` that satisfy a layout's rows as the verifier folds them: each tested row's equation, each
product row's with its table-direct side, each constant copy at 1, each binding copy equal to its source once the callee's
constant is 1, and the layout's constant at 1. -/
structure Facts (words : String → Option (Array Nat)) (k : Chk) (z : ℕ → Bool) : Prop where
  row : ∀ c r, k.rowIs c r = true → z c = (xorSum r.1 z && xorSum r.2 z)
  prod : ∀ c a rec, k.prodIs c a rec = true →
    z c = (xorSum a z && xorSum (readSide ((words rec.table).getD #[]) rec c) z)
  one : ∀ c, k.oneAt c = true → z c = true
  bind : ∀ c src cb, k.bindAt c src cb = true → z cb = true → z c = xorSum src z
  const : z k.d.const = true

theorem fullRow_of_uncovered {words : String → Option (Array Nat)} {d : D} {c : Nat}
    (h : (d.reads.find? (·.covers c)).isNone = true) : fullRow words d c = d.rows[c]? := by
  unfold fullRow
  rw [Option.isNone_iff_eq_none.mp h]

theorem rowIs_inner (words : String → Option (Array Nat)) {d : D} {c : Nat} {r : List Nat × List Nat}
    (h : Chk.rowIs ⟨d, false⟩ c r = true) : fullRow words d c = some r := by
  simp only [Chk.rowIs, Bool.and_eq_true, beq_iff_eq, Bool.not_false, Bool.true_or, and_true] at h
  rw [fullRow_of_uncovered h.2, h.1]

theorem oneAt_inner (words : String → Option (Array Nat)) {d : D} {c : Nat} (h : Chk.oneAt ⟨d, false⟩ c = true) :
    fullRow words d c = some ([d.const], [d.const]) := by
  simp only [Chk.oneAt, Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true, beq_iff_eq] at h
  rw [fullRow_of_uncovered h.2, h.1]

theorem bindAt_inner (words : String → Option (Array Nat)) {d : D} {c : Nat} {src : List Nat} {cb : Nat}
    (h : Chk.bindAt ⟨d, false⟩ c src cb = true) : fullRow words d c = some (src, [cb]) := by
  simp only [Chk.bindAt, Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true, beq_iff_eq] at h
  rw [fullRow_of_uncovered h.2, h.1]

theorem prodIs_full (words : String → Option (Array Nat)) {d : D} {u : Bool} {c : Nat} {a : List Nat} {rec : ReadRec}
    (h : Chk.prodIs ⟨d, u⟩ c a rec = true) :
    fullRow words d c = some (a, readSide ((words rec.table).getD #[]) rec c) ∧
      (u = true → (d.delta.find? (·.1 == c)).isNone = true) := by
  simp only [Chk.prodIs, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
  refine ⟨?_, fun hu => by subst hu; simpa using h.2⟩
  unfold fullRow
  rw [h.1.2, h.1.1]
  rfl

theorem rowIs_unit (words : String → Option (Array Nat)) {d : D} {c : Nat} {r : List Nat × List Nat}
    (h : Chk.rowIs ⟨d, true⟩ c r = true) (one : Nat) : blockRow words d one c = some r := by
  simp only [Chk.rowIs, Bool.and_eq_true, beq_iff_eq, Bool.not_true, Bool.false_or, Option.isNone_iff_eq_none] at h
  unfold blockRow
  rw [h.2]
  exact (fullRow_of_uncovered (by rw [h.1.2]; rfl)).trans h.1.1

theorem oneAt_unit (words : String → Option (Array Nat)) {d : D} {c : Nat} (h : Chk.oneAt ⟨d, true⟩ c = true)
    (one : Nat) : blockRow words d one c = some ([one], [one]) := by
  simp only [Chk.oneAt, ↓reduceIte, beq_iff_eq] at h
  simp [blockRow, h]

theorem bindAt_unit (words : String → Option (Array Nat)) {d : D} {c : Nat} {src : List Nat} {cb : Nat}
    (h : Chk.bindAt ⟨d, true⟩ c src cb = true) (one : Nat) : blockRow words d one c = some (src, src) := by
  simp only [Chk.bindAt, ↓reduceIte, beq_iff_eq] at h
  simp [blockRow, h]

theorem prodIs_unit (words : String → Option (Array Nat)) {d : D} {c : Nat} {a : List Nat} {rec : ReadRec}
    (h : Chk.prodIs ⟨d, true⟩ c a rec = true) (one : Nat) :
    blockRow words d one c = some (a, readSide ((words rec.table).getD #[]) rec c) := by
  obtain ⟨hf, hd⟩ := prodIs_full words h
  unfold blockRow
  rw [Option.isNone_iff_eq_none.mp (hd rfl)]
  exact hf

theorem facts_inner {words : String → Option (Array Nat)} {d : D} {z : ℕ → Bool} (hz : SatAt (fullRow words d) z)
    (h1 : z d.const = true) : Facts words ⟨d, false⟩ z where
  row c r h := hz c r (rowIs_inner words h)
  prod c a rec h := hz c _ (prodIs_full words h).1
  one c h := by
    rw [hz c _ (oneAt_inner words h)]
    simp [xorSum, h1]
  bind c src cb h hcb := by
    rw [hz c _ (bindAt_inner words h)]
    simp [xorSum, hcb]
  const := h1

theorem facts_unit {words : String → Option (Array Nat)} {d : D} {one : Nat} {z : ℕ → Bool}
    (hz : SatAt (blockRow words d one) z) (h1 : z one = true) (hc : Chk.oneAt ⟨d, true⟩ d.const = true) :
    Facts words ⟨d, true⟩ z where
  row c r h := hz c r (rowIs_unit words h one)
  prod c a rec h := hz c _ (prodIs_unit words h one)
  one c h := by
    rw [hz c _ (oneAt_unit words h one)]
    simp [xorSum, h1]
  bind c src cb h _ := by
    rw [hz c _ (bindAt_unit words h one)]
    simp
  const := by
    rw [hz _ _ (oneAt_unit words hc one)]
    simp [xorSum, h1]

/-! ## A layout's check, unfolded -/

theorem getD_of_getElem? {α : Type} {a : Array α} {i : Nat} {x d : α} (h : a[i]? = some x) : a.getD i d = x := by
  obtain ⟨hi, hx⟩ := Array.getElem?_eq_some_iff.mp h
  simp [Array.getD, hi, hx]

theorem checkLayout_spec {types : String → Option CircuitType} {words : String → Option (Array Nat)}
    {ls : Array Layout} {done : Array D} {unit jj : Nat} (h : checkLayout types words ls done unit jj = true)
    {t : CircuitType} (ht : (ls.getD jj default).type.bind types = some t) :
    t.outputs.size = t.outBits ∧
      (∀ o (ho : o < t.outputs.size), t.outputs[o].c ≤ 1) ∧
      (done.getD jj default).inCols.size = t.inBits ∧ (done.getD jj default).outCols.size = t.outputs.size ∧
      Chk.oneAt ⟨done.getD jj default, jj == unit⟩ (done.getD jj default).const = true ∧
      (∀ w < t.inBits, Chk.rowIs ⟨done.getD jj default, jj == unit⟩ ((done.getD jj default).inCols.getD w 0)
        ([(done.getD jj default).inCols.getD w 0], [(done.getD jj default).inCols.getD w 0]) = true) ∧
      ∃ wire, walk types words ls done unit jj ⟨done.getD jj default, jj == unit⟩ jj true t (ls.getD jj default)
          t.items.toList 0 0 ((Array.range t.inBits).map fun w => ([(done.getD jj default).inCols.getD w 0], 0))
          (done.getD jj default).items.toList = some (wire, []) ∧
        ∀ o < t.outputs.size, Chk.rowIs ⟨done.getD jj default, jj == unit⟩ ((done.getD jj default).outCols.getD o 0)
          (resolve (done.getD jj default).const (ent (formOf wire (t.outputs.getD o default))),
            [(done.getD jj default).const]) = true := by
  unfold checkLayout at h
  dsimp only at h
  split at h
  · rename_i t' _ ht'
    rw [ht] at ht'
    cases ht'
    simp only [Bool.and_eq_true, beq_iff_eq, List.all_eq_true, List.mem_range] at h
    obtain ⟨⟨⟨⟨⟨⟨hos, hoc⟩, hin⟩, hout⟩, hone⟩, hins⟩, hw⟩ := h
    refine ⟨hos, fun o ho => by simpa using (Array.all_eq_true.mp hoc) o ho, hin, hout, hone, hins, ?_⟩
    split at hw
    · rename_i _ wire hwk
      exact ⟨wire, hwk, fun o ho => List.all_eq_true.mp hw o (List.mem_range.mpr ho)⟩
    · cases hw
  · rename_i hn
    rw [ht] at hn
    cases hn
  · cases h

/-- A generated read layout's check: it is not the unit, and its rows pass `genOk`. -/
theorem genLayout_spec {types : String → Option CircuitType} {words : String → Option (Array Nat)}
    {ls : Array Layout} {done : Array D} {unit jj : Nat} (h : checkLayout types words ls done unit jj = true)
    {sha : String} {n vb lo : Nat} (hown : (ls.getD jj default).own = .gen sha n vb lo) :
    jj ≠ unit ∧ genOk words ⟨done.getD jj default, false⟩ sha n vb lo = true := by
  unfold checkLayout at h
  dsimp only at h
  split at h
  · rename_i heq _
    rw [hown] at heq
    cases heq
  · rename_i heq _
    rw [hown] at heq
    cases heq
    simp only [Bool.and_eq_true, bne_iff_ne, ne_eq] at h
    obtain ⟨hne, hg⟩ := h
    have hk : (jj == unit) = false := by simp [hne]
    rw [hk] at hg
    exact ⟨hne, hg⟩
  · cases h

theorem genOk_facts {words : String → Option (Array Nat)} {d : D} {u : Bool} {sha : String} {n vb lo : Nat}
    (h : genOk words ⟨d, u⟩ sha n vb lo = true) :
    d.inCols = Array.range n ∧ d.outCols.size = vb ∧ Chk.oneAt ⟨d, u⟩ d.const = true ∧
      ((words sha).getD #[]).size = 2 ^ n ∧ 2 ≤ lo ∧ lo < n ∧ ∀ i < n, Chk.rowIs ⟨d, u⟩ i ([i], [i]) = true := by
  unfold genOk at h
  dsimp only at h
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq, List.all_eq_true, List.mem_range] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨hin, hout⟩, hone⟩, hws⟩, hlo⟩, hn⟩, hins⟩, -⟩ := h
  exact ⟨hin, hout, hone, hws, hlo, hn, hins⟩

/-! ## A placed call -/

/-- Layout `jj`'s rows compute its type, at every fuel past its index. -/
def LayoutSound (types : String → Option CircuitType) (words : String → Option (Array Nat)) (ls : Array Layout)
    (done : Array D) (jj : Nat) : Prop :=
  ∀ u, (ls.getD jj default).type.bind types = some u → ∀ z, SatAt (fullRow words (done.getD jj default)) z →
    z (done.getD jj default).const = true → ∀ F, jj < F → ∀ o < u.outputs.size,
      z ((done.getD jj default).outCols.getD o 0) =
        (evalT types words F u fun i => z ((done.getD jj default).inCols.getD i 0)).getD o false

theorem getD_map_col (a : Array Nat) (base o : Nat) :
    (a.map fun col => (([base + col], 0) : CF)).getD o ([], 0) = if o < a.size then ([base + a.getD o 0], 0) else ([], 0) := by
  unfold Array.getD
  by_cases h : o < a.size <;> simp [h]

theorem list_getD_map_range (n o : Nat) (g : ℕ → Bool) :
    ((List.range n).map g).getD o false = if o < n then g o else false := by
  by_cases h : o < n <;> simp [List.getD_eq_getElem?_getD, h]

theorem readSide_shift (ws : Array Nat) (r : ReadRec) (base c : Nat) :
    readSide ws (r.shift base) (base + c) = (readSide ws r c).map (· + base) := by
  unfold readSide ReadRec.shift
  simp only
  rw [show base + c - (r.prod0 + base) = c - r.prod0 by omega, List.map_map]
  congr 1
  funext l
  simp only [Function.comp]
  omega

/-- **A placed callee's rows, as `placedOk` embeds them**: they hold at its base, its constant is 1 there, and each bound
input carries the call's input form. -/
theorem placed_core {words : String → Option (Array Nat)} {k : Chk} {z : ℕ → Bool} (hf : Facts words k z) {sub : D}
    {base : Nat} {ex : List Nat} {wire : Array CF} {vals : List Bool} (hinv : Inv z wire vals) {args : Array Form}
    (hargs : ∀ i (hi : i < args.size), args[i].c ≤ 1) (hone' : Chk.oneAt ⟨sub, false⟩ sub.const = true)
    (hins' : ∀ w < args.size, Chk.rowIs ⟨sub, false⟩ (sub.inCols.getD w 0)
      ([sub.inCols.getD w 0], [sub.inCols.getD w 0]) = true)
    (hp : placedOk k sub base ex wire args = true) :
    SatAt (fullRow words sub) (fun c => z (c + base)) ∧ z (sub.const + base) = true ∧
      ∀ i < args.size, z (sub.inCols.getD i 0 + base) = formVal (vals.getD · false) (args.getD i default) := by
  unfold placedOk at hp
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range] at hp
  obtain ⟨⟨hcb, hbind⟩, hemb⟩ := hp
  have hcb1 : z (base + sub.const) = true := hf.one _ hcb
  refine ⟨?_, by rw [Nat.add_comm]; exact hcb1, ?_⟩
  -- the callee's rows hold at its base
  · intro c r hr
    have hrows : ∃ p, sub.rows[c]? = some p := by
      unfold fullRow at hr
      split at hr
      · cases hq : sub.rows[c]? with
        | none => rw [hq] at hr; cases hr
        | some p => exact ⟨p, rfl⟩
      · exact ⟨r, hr⟩
    obtain ⟨p, hp⟩ := hrows
    have hc : c < sub.rows.size := (Array.getElem?_eq_some_iff.mp hp).1
    have hrc := hemb c hc
    simp only [Bool.or_eq_true, beq_iff_eq, List.any_eq_true, List.mem_range, Bool.and_eq_true,
      Bool.not_eq_true'] at hrc
    rcases hrc with (hc0 | ⟨i, hi, -, hic⟩) | hrow
    · subst hc0
      rw [oneAt_inner words hone'] at hr
      cases hr
      simp [xorSum]
    · have := rowIs_inner words (hins' i (by omega))
      rw [hic, hr] at this
      cases this
      simp [xorSum]
    · split at hrow
      · rename_i rec hrec
        have hz := hf.prod _ _ _ hrow
        have hfr : fullRow words sub c = some (p.1, readSide ((words rec.table).getD #[]) rec c) := by
          unfold fullRow; rw [hrec, hp]; rfl
        rw [hfr] at hr
        cases hr
        show z (c + base) = _
        rw [Nat.add_comm, hz, getD_of_getElem? hp, readSide_shift]
        simp only [xorSum_map, ReadRec.shift]
        rfl
      · rename_i hrec
        rw [fullRow_of_uncovered (by rw [hrec]; rfl), hp] at hr
        cases hr
        rw [getD_of_getElem? hp] at hrow
        show z (c + base) = _
        rw [Nat.add_comm, hf.row _ _ hrow]
        simp only [shiftRow, xorSum_map]
        rfl
  -- its inputs carry the call's input forms
  · intro i hi
    have hbi := hbind i (by omega)
    have hgi : args.getD i default = args[i] := by simp [Array.getD, hi]
    rw [hgi] at hbi ⊢
    rw [Nat.add_comm]
    split at hbi
    · split at hbi
      · rename_i w hws
        simp only [Bool.and_eq_true, beq_iff_eq] at hbi
        obtain ⟨hc0, hww⟩ := hbi
        have := hinv.2 w
        rw [getD_of_getElem? hww, cfVal_col] at this
        rw [this]
        unfold formVal
        rw [hws, hc0]
        simp [xorSum]
      · cases hbi
    · rw [hf.bind _ _ _ hbi hcb1, xorSum_resolve z _ hf.const _ (formOf_snd_lt _ _)]
      exact hinv.formOf _ (hargs i hi)

theorem placed_sound {types : String → Option CircuitType} {words : String → Option (Array Nat)} {ls : Array Layout}
    {done : Array D} {unit : Nat} {k : Chk} {z : ℕ → Bool} (hf : Facts words k z) {jj base : Nat} {ex : List Nat}
    {wire : Array CF} {vals : List Bool} (hinv : Inv z wire vals) {args : Array Form} {u : CircuitType}
    (hcheck : checkLayout types words ls done unit jj = true) (hjj : jj ≠ unit)
    (hty : (ls.getD jj default).type.bind types = some u) (hsound : LayoutSound types words ls done jj)
    (hsz : args.size = u.inBits) (hargs : ∀ i (hi : i < args.size), args[i].c ≤ 1)
    (hout : (done.getD jj default).outCols.size = u.outBits)
    (hp : placedOk k (done.getD jj default) base ex wire args = true) (F : Nat) (hF : jj < F) (o : Nat) :
    cfVal z (((done.getD jj default).outCols.map fun col => (([base + col], 0) : CF)).getD o ([], 0)) =
      ((List.range u.outBits).map fun o =>
        (evalT types words F u fun i => formVal (vals.getD · false) (args.getD i default)).getD o false).getD o false := by
  obtain ⟨hos', -, -, -, hone', hins', -⟩ := checkLayout_spec hcheck hty
  have hjj' : (jj == unit) = false := by simp [hjj]
  rw [hjj'] at hone' hins'
  unfold LayoutSound at hsound
  generalize done.getD jj default = sub at *
  obtain ⟨hsat, hcb1, hx⟩ := placed_core hf hinv hargs hone' (fun w hw => hins' w (by omega)) hp
  have hS := hsound u hty _ hsat hcb1 F hF
  rw [getD_map_col, list_getD_map_range]
  by_cases ho : o < sub.outCols.size
  · rw [ite_eq_left ho, ite_eq_left (by omega), cfVal_col, Nat.add_comm, hS o (by omega)]
    exact congrArg (fun l => List.getD l o false) (evalT_congr types words F u fun i hi => hx i (by omega))
  · rw [ite_eq_right ho, ite_eq_right (by omega)]
    rfl

/-! ## S3c-2: reads. The decoder -/

theorem bitsIdx_append : ∀ (l m : List Bool), bitsIdx (l ++ m) = bitsIdx l + 2 ^ l.length * bitsIdx m
  | [], m => by simp [bitsIdx]
  | b :: l, m => by
    simp only [List.cons_append, bitsIdx, bitsIdx_append l m, List.length_cons, Nat.pow_succ]
    rw [Nat.mul_add, ← Nat.mul_assoc, Nat.mul_comm 2 (2 ^ l.length)]
    omega

theorem bitsIdx_lt : ∀ l : List Bool, bitsIdx l < 2 ^ l.length
  | [] => by simp [bitsIdx]
  | b :: l => by
    have := bitsIdx_lt l
    simp only [bitsIdx, List.length_cons, Nat.pow_succ]
    split <;> omega

theorem split_idx {lo hi i p : Nat} (hp : 0 < p) (hlo : lo < p) :
    (decide (lo = i % p) && decide (hi = i / p)) = decide (lo + p * hi = i) := by
  by_cases h : lo + p * hi = i
  · subst h
    simp [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hlo, Nat.add_mul_div_left _ _ hp, Nat.div_eq_of_lt hlo]
  · simp only [h, decide_false, Bool.and_eq_false_iff, decide_eq_false_iff_not]
    by_cases h1 : lo = i % p
    · right; intro h2; apply h; rw [h1, h2]; exact Nat.mod_add_div i p
    · left; exact h1

theorem cfVal_neg (z : ℕ → Bool) (f : CF) (hf : f.2 < 2) : cfVal z (f.1, (f.2 + 1) % 2) = !cfVal z f := by
  unfold cfVal
  rcases (show f.2 = 0 ∨ f.2 = 1 by omega) with h | h <;> simp [h]

theorem blockRows_spec (words : String → Option (Array Nat)) (k : Chk) (z : ℕ → Bool) (hf : Facts words k z)
    (lo hi : Array CF) (h : Nat)
    (hlo : ∀ a, (lo.getD a ([], 0)).2 < 2) (hhi : ∀ a, (hi.getD a ([], 0)).2 < 2) :
    ∀ cnt i refs acc ms refs', blockRows k lo hi h cnt i refs acc = some (ms, refs') →
      ms.size = acc.size + cnt ∧ (∀ m < acc.size, ms.getD m ([], 0) = acc.getD m ([], 0)) ∧
      ∀ m < cnt, ∃ col, ms.getD (acc.size + m) ([], 0) = ([col], 0) ∧
        z col = (cfVal z (lo.getD ((i + m) % 2 ^ h) ([], 0)) && cfVal z (hi.getD ((i + m) / 2 ^ h) ([], 0)))
  | 0, i, refs, acc, ms, refs', hb => by
    simp only [blockRows, Option.some.injEq, Prod.mk.injEq] at hb
    obtain ⟨rfl, rfl⟩ := hb
    exact ⟨rfl, fun _ _ => rfl, fun m hm => absurd hm (Nat.not_lt_zero _)⟩
  | cnt + 1, i, refs, acc, ms, refs', hb => by
    match refs, hb with
    | .row col :: refs, hb =>
      unfold blockRows at hb
      split at hb
      · rename_i hr
        obtain ⟨h1, h2, h3⟩ := blockRows_spec words k z hf lo hi h hlo hhi cnt (i + 1) refs _ ms refs' hb
        have hz := hf.row _ _ hr
        rw [xorSum_resolve z _ hf.const _ (hlo _), xorSum_resolve z _ hf.const _ (hhi _)] at hz
        refine ⟨by simp [h1]; omega, fun m hm => ?_, fun m hm => ?_⟩
        · rw [h2 m (by simp; omega)]
          simp [Array.getD, hm, Nat.lt_succ_of_lt hm, Array.getElem_push_lt hm]
        · cases m with
          | zero =>
            refine ⟨col, ?_, by simpa using hz⟩
            rw [Nat.add_zero, h2 acc.size (by simp)]
            simp [Array.getD]
          | succ m =>
            obtain ⟨col', hcol, hzc⟩ := h3 m (by omega)
            refine ⟨col', by simpa [Nat.add_assoc, Nat.add_comm 1 m] using hcol, ?_⟩
            simpa [Nat.add_assoc, Nat.add_comm 1 m] using hzc
      · cases hb
    | [], hb => simp [blockRows] at hb
    | .seg _ _ _ :: _, hb => simp [blockRows] at hb

theorem getD_extract (a : Array CF) (s e i : Nat) (hi : i < e - s) (he : e ≤ a.size) :
    (a.extract s e).getD i ([], 0) = a.getD (s + i) ([], 0) := by
  have h2 : s + i < a.size := by omega
  have h3 : i < min e a.size - s := by omega
  simp [Array.getD, h2, h3, Array.getElem_extract]

theorem getD_extract_lt2 (a : Array CF) (s e : Nat) (he : e ≤ a.size) (h : ∀ x, (a.getD x ([], 0)).2 < 2) :
    ∀ x, ((a.extract s e).getD x ([], 0)).2 < 2 := by
  intro x
  by_cases hx : x < e - s
  · rw [getD_extract a s e x hx he]; exact h _
  · have : ¬ x < min e a.size - s := by omega
    simp [Array.getD, this]

theorem toList_split (a : Array CF) (h : Nat) :
    a.toList = (a.extract 0 h).toList ++ (a.extract h a.size).toList := by
  simp only [Array.toList_extract, List.extract, Nat.sub_zero, List.drop_zero]
  rw [List.take_of_length_le (l := List.drop h a.toList) (by simp)]
  exact (List.take_append_drop h a.toList).symm

theorem chkDecode_spec (words : String → Option (Array Nat)) (k : Chk) (z : ℕ → Bool) (hf : Facts words k z) :
    ∀ (n : Nat) (leaves : Array CF), leaves.size = n → ∀ refs ms refs', (∀ a, (leaves.getD a ([], 0)).2 < 2) →
      chkDecode k leaves refs = some (ms, refs') →
      ms.size = 2 ^ (max leaves.size 1) ∧ (∀ a, (ms.getD a ([], 0)).2 < 2) ∧
      ∀ i < ms.size, cfVal z (ms.getD i ([], 0)) = decide (bitsIdx (leaves.toList.map (cfVal z)) = i) := by
  intro n
  induction n using Nat.strongRecOn with
  | ind n ih =>
  intro leaves hn refs ms refs' hlv hd
  unfold chkDecode at hd
  split at hd
  · rename_i hs
    simp only [Option.some.injEq, Prod.mk.injEq] at hd
    obtain ⟨rfl, rfl⟩ := hd
    have hf0 := hlv 0
    generalize hfd : leaves.getD 0 ([], 0) = f at hf0
    have hmax : max leaves.size 1 = 1 := by omega
    have g0 : (#[(f.1, (f.2 + 1) % 2), f] : Array CF).getD 0 ([], 0) = (f.1, (f.2 + 1) % 2) := rfl
    have g1 : (#[(f.1, (f.2 + 1) % 2), f] : Array CF).getD 1 ([], 0) = f := rfl
    have hl : leaves.toList.map (cfVal z) = [] ∧ f = ([], 0) ∨ leaves.toList.map (cfVal z) = [cfVal z f] := by
      rcases (show leaves.size = 0 ∨ leaves.size = 1 by omega) with h0 | h1
      · left
        refine ⟨by simp [List.eq_nil_of_length_eq_zero (by simpa using h0 : leaves.toList.length = 0)], ?_⟩
        rw [← hfd]; simp [Array.getD, h0]
      · right
        obtain ⟨x, hx⟩ := List.length_eq_one_iff.mp (by simpa using h1 : leaves.toList.length = 1)
        have : x = f := by
          rw [← hfd, Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, hx]
          rfl
        rw [hx, this]; rfl
    refine ⟨by simp [hmax], fun a => ?_, fun i hi => ?_⟩
    · rcases a with _ | _ | a
      · rw [g0]; simp only; omega
      · rw [g1]; exact hf0
      · simp [Array.getD, show ¬ (a + 1 + 1 < 2) by omega]
    · have hi2 : i < 2 := by simpa using hi
      rcases (show i = 0 ∨ i = 1 by omega) with rfl | rfl
      · rw [g0, cfVal_neg z f hf0]
        rcases hl with ⟨hl, rfl⟩ | hl
        · rw [hl]; simp [cfVal, xorSum, bitsIdx]
        · rw [hl]; cases hc : cfVal z f <;> simp [bitsIdx, hc]
      · rw [g1]
        rcases hl with ⟨hl, rfl⟩ | hl
        · rw [hl]; simp [cfVal, xorSum, bitsIdx]
        · rw [hl]; cases hc : cfVal z f <;> simp [bitsIdx, hc]
  · rename_i hs
    dsimp only at hd
    split at hd
    · cases hd
    · rename_i loM refs1 hlo
      split at hd
      · cases hd
      · rename_i hiM refs2 hhi
        have hh1 : 1 ≤ leaves.size / 2 := by omega
        have hh2 : leaves.size / 2 < leaves.size := by omega
        have hlsz : (leaves.extract 0 (leaves.size / 2)).size = leaves.size / 2 := by
          simp [Array.size_extract]; omega
        have hhsz : (leaves.extract (leaves.size / 2) leaves.size).size = leaves.size - leaves.size / 2 := by
          simp [Array.size_extract]
        obtain ⟨l1, l2, l3⟩ := ih (leaves.size / 2) (by omega) _ hlsz refs loM refs1
          (getD_extract_lt2 _ _ _ (by omega) hlv) hlo
        obtain ⟨h1', h2', h3'⟩ := ih (leaves.size - leaves.size / 2) (by omega) _ hhsz refs1 hiM refs2
          (getD_extract_lt2 _ _ _ (Nat.le_refl _) hlv) (by simpa using hhi)
        rw [hlsz, show max (leaves.size / 2) 1 = leaves.size / 2 by omega] at l1
        rw [hhsz, show max (leaves.size - leaves.size / 2) 1 = leaves.size - leaves.size / 2 by omega] at h1'
        obtain ⟨b1, -, b3⟩ := blockRows_spec words k z hf loM hiM (leaves.size / 2) l2 h2' _ _ _ _ _ _ hd
        simp only [Array.size_empty, Nat.zero_add] at b1 b3
        have hmax : max leaves.size 1 = leaves.size := by omega
        refine ⟨by rw [b1, hmax], fun a => ?_, fun i hiu => ?_⟩
        · by_cases ha : a < ms.size
          · obtain ⟨col, hc, -⟩ := b3 a (by omega)
            rw [hc]; simp
          · simp [Array.getD, ha]
        · obtain ⟨col, hc, hzc⟩ := b3 i (by omega)
          rw [hc, cfVal_col, hzc]
          have hpow : 2 ^ leaves.size = 2 ^ (leaves.size / 2) * 2 ^ (leaves.size - leaves.size / 2) := by
            rw [← Nat.pow_add]; congr 1; omega
          have hi' : i < 2 ^ (leaves.size / 2) * 2 ^ (leaves.size - leaves.size / 2) := by omega
          have hq : i / 2 ^ (leaves.size / 2) < 2 ^ (leaves.size - leaves.size / 2) :=
            (Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _)).mpr (by rw [Nat.mul_comm]; exact hi')
          rw [l3 _ (by rw [l1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)), h3' _ (by rw [h1']; exact hq)]
          have hlt := bitsIdx_lt ((leaves.extract 0 (leaves.size / 2)).toList.map (cfVal z))
          simp only [List.length_map, Array.length_toList, hlsz] at hlt
          rw [split_idx (Nat.two_pow_pos _) hlt, toList_split leaves (leaves.size / 2), List.map_append,
            bitsIdx_append]
          simp [hlsz, Nat.min_eq_left (Nat.le_of_lt hh2)]

/-! ## S3c-2: reads. One-hot sums -/

/-- The XOR of a one-hot family over a list without repeats is whether the hot index is in the list. -/
theorem xorSum_onehot {v : ℕ → Bool} {x : Nat} (hv : ∀ l, v l = decide (x = l)) :
    ∀ L : List Nat, L.Nodup → xorSum L v = decide (x ∈ L)
  | [], _ => by simp [xorSum]
  | a :: L, hnd => by
    rw [xorSum_cons, xorSum_onehot hv L (List.nodup_cons.mp hnd).2, hv]
    by_cases hxa : x = a
    · subst hxa
      have : x ∉ L := (List.nodup_cons.mp hnd).1
      simp [this]
    · simp [hxa]

/-- The XOR over `h < m` of `[x = h] ∧ q h` is `q x` when `x < m`. -/
theorem xorSum_select {x m : Nat} (hx : x < m) (q : ℕ → Bool) :
    xorSum (List.range m) (fun h => decide (x = h) && q h) = q x := by
  have key : ∀ L : List Nat, L.Nodup → xorSum L (fun h => decide (x = h) && q h) = (decide (x ∈ L) && q x) := by
    intro L
    induction L with
    | nil => intro _; simp [xorSum]
    | cons a L ih =>
      intro hnd
      rw [xorSum_cons, ih (List.nodup_cons.mp hnd).2]
      by_cases hxa : x = a
      · subst hxa
        have : x ∉ L := (List.nodup_cons.mp hnd).1
        simp [this]
      · simp [hxa]
  rw [key _ List.nodup_range]
  simp [hx]

/-! ## S3c-2: reads. Product and output rows -/

theorem prodRows_spec (words : String → Option (Array Nat)) (k : Chk) (z : ℕ → Bool) (hf : Facts words k z)
    (rec : ReadRec) (hiM : Array CF) (nh : Nat) (hhi : ∀ a, (hiM.getD a ([], 0)).2 < 2) :
    ∀ cnt x refs refs', prodRows k rec hiM nh cnt x refs = some refs' →
      ∀ m < cnt, z (rec.prod0 + (x + m)) = (cfVal z (hiM.getD ((x + m) % nh) ([], 0)) &&
        xorSum (readSide ((words rec.table).getD #[]) rec (rec.prod0 + (x + m))) z)
  | 0, _, _, _, _ => fun m hm => absurd hm (Nat.not_lt_zero _)
  | cnt + 1, x, refs, refs', hp => by
    match refs, hp with
    | .row col :: refs, hp =>
      unfold prodRows at hp
      split at hp
      · rename_i hc
        simp only [Bool.and_eq_true, beq_iff_eq] at hc
        obtain ⟨hcol, hprod⟩ := hc
        intro m hm
        cases m with
        | zero =>
          have := hf.prod _ _ _ hprod
          rw [xorSum_resolve z _ hf.const _ (hhi _)] at this
          rw [Nat.add_zero, ← hcol, this]
        | succ m =>
          have := prodRows_spec words k z hf rec hiM nh hhi cnt (x + 1) refs refs' hp m (by omega)
          rw [show x + (m + 1) = x + 1 + m by omega]
          exact this
      · cases hp
    | [], hp => simp [prodRows] at hp
    | .seg _ _ _ :: _, hp => simp [prodRows] at hp

theorem outRows_spec (k : Chk) (z : ℕ → Bool) (words : String → Option (Array Nat)) (hf : Facts words k z)
    (rec : ReadRec) (ws : Array Nat) (kk nh : Nat) (q : ℕ → Bool)
    (hq : ∀ j < kk, xorSum ((List.range nh).map (rec.prod0 + j * nh + ·)) z = q j)
    (hconst : ∀ j, kk ≤ j → (∀ i < ws.size, (ws.getD i 0).testBit j = (ws.getD 0 0).testBit j) →
      (ws.getD 0 0).testBit j = q j) :
    ∀ cnt j refs acc outs refs', outRows k rec ws kk nh cnt j refs acc = some (outs, refs') →
      outs.size = acc.size + cnt ∧ (∀ m < acc.size, outs.getD m ([], 0) = acc.getD m ([], 0)) ∧
      ∀ m < cnt, cfVal z (outs.getD (acc.size + m) ([], 0)) = q (j + m)
  | 0, j, refs, acc, outs, refs', ho => by
    simp only [outRows, Option.some.injEq, Prod.mk.injEq] at ho
    obtain ⟨rfl, rfl⟩ := ho
    exact ⟨rfl, fun _ _ => rfl, fun m hm => absurd hm (Nat.not_lt_zero _)⟩
  | cnt + 1, j, refs, acc, outs, refs', ho => by
    unfold outRows at ho
    split at ho
    · rename_i hj
      split at ho
      · rename_i col refs'' 
        split at ho
        · rename_i hr
          obtain ⟨h1, h2, h3⟩ := outRows_spec k z words hf rec ws kk nh q hq hconst cnt (j + 1) refs'' _ outs refs' ho
          have hz := hf.row _ _ hr
          refine ⟨by simp [h1]; omega, fun m hm => ?_, fun m hm => ?_⟩
          · rw [h2 m (by simp; omega)]
            simp [Array.getD, hm, Nat.lt_succ_of_lt hm, Array.getElem_push_lt hm]
          · cases m with
            | zero =>
              have hx : (acc.push ([col], 0)).getD acc.size ([], 0) = ([col], 0) := by simp [Array.getD]
              simp only [Nat.add_zero]
              rw [h2 acc.size (by simp), hx, cfVal_col, hz, hq j hj]
              simp [xorSum, hf.const]
            | succ m =>
              have := h3 m (by omega)
              simpa [Nat.add_assoc, Nat.add_comm 1 m] using this
        · cases ho
      · cases ho
    · rename_i hj
      split at ho
      · rename_i hall
        obtain ⟨h1, h2, h3⟩ := outRows_spec k z words hf rec ws kk nh q hq hconst cnt (j + 1) refs _ outs refs' ho
        have hc := hconst j (by omega) (fun i hi => by
          have := List.all_eq_true.mp hall i (List.mem_range.mpr hi)
          simpa using this)
        refine ⟨by simp [h1]; omega, fun m hm => ?_, fun m hm => ?_⟩
        · rw [h2 m (by simp; omega)]
          simp [Array.getD, hm, Nat.lt_succ_of_lt hm, Array.getElem_push_lt hm]
        · cases m with
          | zero =>
            have hx : (acc.push ([], if (ws.getD 0 0).testBit j then 1 else 0)).getD acc.size ([], 0) =
                ([], if (ws.getD 0 0).testBit j then 1 else 0) := by simp [Array.getD]
            simp only [Nat.add_zero]
            rw [h2 acc.size (by simp), hx, ← hc]
            cases (ws.getD 0 0).testBit j <;> simp [cfVal, xorSum]
          | succ m =>
            have := h3 m (by omega)
            simpa [Nat.add_assoc, Nat.add_comm 1 m] using this
      · cases ho

/-! ## S3c-2: reads. The table-direct side -/

/-- The table-direct side's XOR, the low minterms one-hot at `lo`: bit `j` of word `h·2^lo + lo`. -/
theorem xorSum_readSide (ws : Array Nat) (r : ReadRec) {z : ℕ → Bool} {lo : Nat} (hlo : lo < 2 ^ r.lo)
    (hm : ∀ l, l < 2 ^ r.lo → z (r.loTop + l) = decide (lo = l)) {j h : Nat} (hh : h < 2 ^ (r.n - r.lo)) :
    xorSum (readSide ws r (r.prod0 + j * 2 ^ (r.n - r.lo) + h)) z =
      (ws.getD (h * 2 ^ r.lo + lo) 0).testBit j := by
  have hpos : 0 < 2 ^ (r.n - r.lo) := Nat.two_pow_pos _
  have e1 : (r.prod0 + j * 2 ^ (r.n - r.lo) + h - r.prod0) % 2 ^ (r.n - r.lo) = h := by
    rw [show r.prod0 + j * 2 ^ (r.n - r.lo) + h - r.prod0 = h + 2 ^ (r.n - r.lo) * j by
      rw [Nat.mul_comm]; omega]
    rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hh]
  have e2 : (r.prod0 + j * 2 ^ (r.n - r.lo) + h - r.prod0) / 2 ^ (r.n - r.lo) = j := by
    rw [show r.prod0 + j * 2 ^ (r.n - r.lo) + h - r.prod0 = h + 2 ^ (r.n - r.lo) * j by
      rw [Nat.mul_comm]; omega]
    rw [Nat.add_mul_div_left _ _ hpos, Nat.div_eq_of_lt hh, Nat.zero_add]
  unfold readSide
  rw [e1, e2, xorSum_map]
  have hv : ∀ l, ((z ∘ (r.loTop + ·)) l) = (decide (lo = l) || decide (2 ^ r.lo ≤ l) && z (r.loTop + l)) := by
    intro l
    by_cases hl : l < 2 ^ r.lo
    · simp [hm l hl, show ¬ 2 ^ r.lo ≤ l by omega]
    · have : lo ≠ l := by omega
      simp [this, show 2 ^ r.lo ≤ l by omega]
  -- only indices below 2^lo appear, where the family is one-hot at lo
  have hsub : ∀ l ∈ (List.range (2 ^ r.lo)).filter
      (fun l => (ws.getD (h * 2 ^ r.lo + l) 0).testBit j), (z ∘ (r.loTop + ·)) l = decide (lo = l) := by
    intro l hl
    have := List.mem_range.mp (List.mem_filter.mp hl).1
    simp [hm l this]
  rw [xorSum_congr hsub]
  rw [xorSum_onehot (v := fun l => decide (lo = l)) (fun _ => rfl) _ (List.nodup_range.filter _)]
  simp [List.mem_filter, hlo]

/-- An output row's side: the XOR over the high minterms `h` of bit `j`'s product rows, one-hot at `hi`. -/
theorem xorSum_outRow (r : ReadRec) {z : ℕ → Bool} (j : Nat) (q : ℕ → Bool) {hi : Nat}
    (hhi : hi < 2 ^ (r.n - r.lo))
    (hp : ∀ h < 2 ^ (r.n - r.lo), z (r.prod0 + j * 2 ^ (r.n - r.lo) + h) = (decide (hi = h) && q h)) :
    xorSum ((List.range (2 ^ (r.n - r.lo))).map (r.prod0 + j * 2 ^ (r.n - r.lo) + ·)) z = q hi := by
  rw [xorSum_map]
  refine (xorSum_congr (w := fun h => decide (hi = h) && q h) fun h hh => ?_).trans (xorSum_select hhi q)
  exact hp h (List.mem_range.mp hh)

/-- **A read's product rows**: with its high minterms one-hot at `hiIdx` and its low minterms (its record's columns
`loTop + l`) one-hot at `loIdx`, bit `j`'s product rows sum to bit `j` of word `loIdx + 2^lo · hiIdx` of the table. -/
theorem prod_sums (words : String → Option (Array Nat)) (k : Chk) (z : ℕ → Bool) (hf : Facts words k z)
    (rec : ReadRec) (ws : Array Nat) (hws : (words rec.table).getD #[] = ws) {n lo : Nat} (hrn : rec.n = n)
    (hrlo : rec.lo = lo) (loM hiM : Array CF) {loIdx hiIdx : Nat} (hloI : loIdx < 2 ^ lo)
    (hhiI : hiIdx < 2 ^ (n - lo))
    (hloM : ∀ l < 2 ^ lo, cfVal z (loM.getD l ([], 0)) = decide (loIdx = l))
    (hhiM : ∀ h < 2 ^ (n - lo), cfVal z (hiM.getD h ([], 0)) = decide (hiIdx = h))
    (hhi2 : ∀ a, (hiM.getD a ([], 0)).2 < 2)
    (hlocols : ∀ l < 2 ^ lo, loM.getD l ([], 0) = ([rec.loTop + l], 0)) {kk : Nat} {refs refs' : List Ref}
    (hpr : prodRows k rec hiM (2 ^ (n - lo)) (kk * 2 ^ (n - lo)) 0 refs = some refs') :
    ∀ j < kk, xorSum ((List.range (2 ^ (n - lo))).map (rec.prod0 + j * 2 ^ (n - lo) + ·)) z =
      (ws.getD (loIdx + 2 ^ lo * hiIdx) 0).testBit j := by
  have hp := prodRows_spec words k z hf rec hiM _ hhi2 _ 0 _ refs' hpr
  have hloc : ∀ l, l < 2 ^ rec.lo → z (rec.loTop + l) = decide (loIdx = l) := by
    intro l hl
    rw [hrlo] at hl
    have := hloM l hl
    rw [hlocols l hl, cfVal_col] at this
    exact this
  have hprod : ∀ j < kk, ∀ h < 2 ^ (n - lo),
      z (rec.prod0 + j * 2 ^ (n - lo) + h) = (decide (hiIdx = h) && (ws.getD (h * 2 ^ lo + loIdx) 0).testBit j) := by
    intro j hj h hh
    have hx := hp (j * 2 ^ (n - lo) + h) (by
      have : j * 2 ^ (n - lo) + h < (j + 1) * 2 ^ (n - lo) := by rw [Nat.succ_mul]; omega
      calc j * 2 ^ (n - lo) + h < (j + 1) * 2 ^ (n - lo) := this
        _ ≤ kk * 2 ^ (n - lo) := Nat.mul_le_mul_right _ hj)
    rw [Nat.zero_add, show (j * 2 ^ (n - lo) + h) % 2 ^ (n - lo) = h by
      rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hh], hhiM h hh, ← Nat.add_assoc] at hx
    rw [hx, hws]
    congr 1
    have hn' : rec.n - rec.lo = n - lo := by rw [hrn, hrlo]
    have := xorSum_readSide ws rec (lo := loIdx) (by rw [hrlo]; exact hloI) hloc (j := j) (h := h)
      (by rw [hn']; exact hh)
    rw [hn', hrlo] at this
    exact this
  intro j hj
  have := xorSum_outRow rec j (fun h => (ws.getD (h * 2 ^ lo + loIdx) 0).testBit j) (hi := hiIdx)
    (by rw [hrn, hrlo]; exact hhiI) (fun h hh => by rw [hrn, hrlo] at hh ⊢; exact hprod j hj h hh)
  rw [hrn, hrlo] at this
  rw [this, Nat.mul_comm hiIdx, Nat.add_comm]

/-- **An inline read's outputs.** If its decoders' minterms are one-hot at `loIdx` and `hiIdx`, the read's product and
output rows (`readOk`) carry bits `0 … vb − 1` of word `loIdx + 2^lo · hiIdx` of the table. -/
theorem readOk_spec (words : String → Option (Array Nat)) (k : Chk) (z : ℕ → Bool) (hf : Facts words k z)
    (sha : String) (n lo vb : Nat) (loM hiM : Array CF) (refs : List Ref) (outs : Array CF) (refs' : List Ref)
    (hn : lo < n) (hsize : ((words sha).getD #[]).size = 2 ^ n) {loIdx hiIdx : Nat} (hloI : loIdx < 2 ^ lo)
    (hhiI : hiIdx < 2 ^ (n - lo))
    (hloM : ∀ l < 2 ^ lo, cfVal z (loM.getD l ([], 0)) = decide (loIdx = l))
    (hhiM : ∀ h < 2 ^ (n - lo), cfVal z (hiM.getD h ([], 0)) = decide (hiIdx = h))
    (hhi2 : ∀ a, (hiM.getD a ([], 0)).2 < 2)
    (hr : readOk k ((words sha).getD #[]) sha n lo vb loM hiM refs = some (outs, refs')) :
    outs.size = vb ∧ ∀ j < vb, cfVal z (outs.getD j ([], 0)) =
      (((words sha).getD #[]).getD (loIdx + 2 ^ lo * hiIdx) 0).testBit j := by
  generalize hws : (words sha).getD #[] = ws at *
  have hidx : loIdx + 2 ^ lo * hiIdx < ws.size := by
    rw [hsize]
    have : 2 ^ n = 2 ^ lo * 2 ^ (n - lo) := by rw [← Nat.pow_add, Nat.add_sub_cancel' (Nat.le_of_lt hn)]
    rw [this]
    calc loIdx + 2 ^ lo * hiIdx < 2 ^ lo + 2 ^ lo * hiIdx := by omega
      _ = 2 ^ lo * (hiIdx + 1) := by rw [Nat.mul_add, Nat.mul_one, Nat.add_comm]
      _ ≤ 2 ^ lo * 2 ^ (n - lo) := Nat.mul_le_mul_left _ hhiI
  -- a bit constant across the table is the bit of any word
  have hconst : ∀ j, (∀ i < ws.size, (ws.getD i 0).testBit j = (ws.getD 0 0).testBit j) →
      (ws.getD 0 0).testBit j = (ws.getD (loIdx + 2 ^ lo * hiIdx) 0).testBit j :=
    fun j h => (h _ hidx).symm
  unfold readOk at hr
  dsimp only at hr
  split at hr
  · rename_i hk0
    obtain ⟨h1, -, h3⟩ := outRows_spec k z words hf default ws 0 (2 ^ (n - lo))
      (fun j => (ws.getD (loIdx + 2 ^ lo * hiIdx) 0).testBit j)
      (fun j hj => absurd hj (Nat.not_lt_zero _)) (fun j _ h => hconst j h) vb 0 refs #[] outs refs' hr
    exact ⟨by simpa using h1, fun j hj => by simpa using h3 j (by simpa using hj)⟩
  · rename_i hk0
    split at hr
    · rename_i prod0 _
      split at hr
      · cases hr
      · rename_i rec hrec
        split at hr
        · rename_i hc
          simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq, List.all_eq_true, List.mem_range] at hc
          obtain ⟨⟨⟨⟨⟨⟨htab, hrn⟩, hrlo⟩, hrk⟩, hrp⟩, hkv⟩, hlocols⟩ := hc
          split at hr
          · cases hr
          · rename_i refs'' hpr
            have hwsr : (words rec.table).getD #[] = ws := by rw [htab]; exact hws
            obtain ⟨h1, -, h3⟩ := outRows_spec k z words hf rec ws _ _
              (fun j => (ws.getD (loIdx + 2 ^ lo * hiIdx) 0).testBit j)
              (prod_sums words k z hf rec ws hwsr hrn hrlo loM hiM hloI hhiI hloM hhiM hhi2
                (fun l hl => by simpa using hlocols l hl) hpr)
              (fun j _ h => hconst j h) vb 0 refs'' #[] outs refs' hr
            exact ⟨by simpa using h1, fun j hj => by simpa using h3 j (by simpa using hj)⟩
        · cases hr
    · cases hr

/-! ## S3c-2b: placed reads -/

theorem genOuts_spec (words : String → Option (Array Nat)) (k : Chk) (z : ℕ → Bool) (hf : Facts words k z)
    (rec : ReadRec) (ws : Array Nat) (vb kk nh : Nat) (q : ℕ → Bool)
    (hq : ∀ j < kk, xorSum ((List.range nh).map (rec.prod0 + j * nh + ·)) z = q j)
    (hconst : ∀ j, kk ≤ j → (∀ i < ws.size, (ws.getD i 0).testBit j = (ws.getD 0 0).testBit j) →
      (ws.getD 0 0).testBit j = q j)
    (h : genOuts k ws rec vb kk nh = true) : ∀ j < vb, z (k.d.outCols.getD j 0) = q j := by
  intro j hj
  unfold genOuts at h
  have hjj := List.all_eq_true.mp h j (List.mem_range.mpr hj)
  dsimp only at hjj
  split at hjj
  · rename_i hjk
    have hz := hf.row _ _ hjj
    simp only at hz
    rw [hz, hq j hjk]
    simp [xorSum, hf.const]
  · rename_i hjk
    simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range, beq_iff_eq] at hjj
    obtain ⟨hall, hrow⟩ := hjj
    rw [hf.row _ _ hrow, ← hconst j (by omega) hall]
    cases (ws.getD 0 0).testBit j <;> simp [xorSum, hf.const]

/-- **A generated read layout computes its table**: on column values that satisfy its rows as the verifier folds them,
with its constant at 1, output bit `o` is bit `o` of the table's word that its input columns `0 … n − 1` index. -/
theorem genOk_sound (words : String → Option (Array Nat)) (d : D) {sha : String} {n vb lo : Nat}
    (h : genOk words ⟨d, false⟩ sha n vb lo = true) {z : ℕ → Bool} (hz : SatAt (fullRow words d) z)
    (h1 : z d.const = true) :
    ∀ o < vb, z (d.outCols.getD o 0) = (((words sha).getD #[]).getD (bitsIdx ((List.range n).map z)) 0).testBit o := by
  have hf := facts_inner hz h1
  obtain ⟨-, -, -, hws, hlo2, hlon, -⟩ := genOk_facts h
  -- the index: its low bits, then its high bits
  have hidx : bitsIdx ((List.range n).map z) =
      bitsIdx ((List.range lo).map z) + 2 ^ lo * bitsIdx ((List.range (n - lo)).map fun i => z (lo + i)) := by
    have hn : n = lo + (n - lo) := by omega
    conv => lhs; rw [hn]
    simp only [List.range_add, List.map_append, bitsIdx_append, List.map_map, List.length_map, List.length_range,
      Function.comp_def]
  unfold genOk at h
  dsimp only at h
  simp only [Bool.and_eq_true] at h
  obtain ⟨-, hm⟩ := h
  generalize hwsd : (words sha).getD #[] = ws at hm hws ⊢
  split at hm
  · cases hm
  · rename_i loM refs1 hdlo
    split at hm
    · cases hm
    · rename_i hiM refs2 hdhi
      have hlv1 : ∀ a, (((Array.range lo).map fun i => (([i], 0) : CF)).getD a ([], 0)).2 < 2 := by
        intro a; by_cases ha : a < lo <;> simp [Array.getD, ha]
      have hlv2 : ∀ a, (((Array.range (n - lo)).map fun i => (([lo + i], 0) : CF)).getD a ([], 0)).2 < 2 := by
        intro a; by_cases ha : a < n - lo <;> simp [Array.getD, ha]
      obtain ⟨dl1, -, dl3⟩ := chkDecode_spec words _ z hf _ _ rfl _ loM refs1 hlv1 hdlo
      obtain ⟨dh1, dh2, dh3⟩ := chkDecode_spec words _ z hf _ _ rfl _ hiM refs2 hlv2 hdhi
      simp only [Array.size_map, Array.size_range] at dl1 dh1
      rw [show max lo 1 = lo by omega] at dl1
      rw [show max (n - lo) 1 = n - lo by omega] at dh1
      have hv1 : ((Array.range lo).map fun i => (([i], 0) : CF)).toList.map (cfVal z) = (List.range lo).map z := by
        simp [cfVal_col]
      have hv2 : ((Array.range (n - lo)).map fun i => (([lo + i], 0) : CF)).toList.map (cfVal z) =
          (List.range (n - lo)).map fun i => z (lo + i) := by
        simp [cfVal_col]
      rw [hv1] at dl3
      rw [hv2] at dh3
      obtain ⟨loIdx, hloDef⟩ : ∃ x, bitsIdx ((List.range lo).map z) = x := ⟨_, rfl⟩
      obtain ⟨hiIdx, hhiDef⟩ : ∃ x, bitsIdx ((List.range (n - lo)).map fun i => z (lo + i)) = x := ⟨_, rfl⟩
      have hloI : loIdx < 2 ^ lo := by rw [← hloDef]; simpa using bitsIdx_lt ((List.range lo).map z)
      have hhiI : hiIdx < 2 ^ (n - lo) := by
        rw [← hhiDef]; simpa using bitsIdx_lt ((List.range (n - lo)).map fun i => z (lo + i))
      rw [hloDef] at dl3
      rw [hhiDef] at dh3
      rw [hloDef, hhiDef] at hidx
      have hidxlt : loIdx + 2 ^ lo * hiIdx < ws.size := by
        rw [← hidx, hws]; simpa using bitsIdx_lt ((List.range n).map z)
      have hconst : ∀ j, (∀ i < ws.size, (ws.getD i 0).testBit j = (ws.getD 0 0).testBit j) →
          (ws.getD 0 0).testBit j = (ws.getD (loIdx + 2 ^ lo * hiIdx) 0).testBit j := fun j h => (h _ hidxlt).symm
      intro o ho
      rw [hidx]
      split at hm
      · simp only [Bool.and_eq_true] at hm
        exact genOuts_spec words _ z hf default ws vb 0 _ (fun j => (ws.getD (loIdx + 2 ^ lo * hiIdx) 0).testBit j)
          (fun j hj => absurd hj (Nat.not_lt_zero _)) (fun j _ h => hconst j h) hm.2 o ho
      · split at hm
        · rename_i prod0 _
          split at hm
          · cases hm
          · rename_i rec hrec
            simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq, List.all_eq_true, List.mem_range] at hm
            obtain ⟨⟨⟨⟨⟨⟨⟨⟨htab, hrn⟩, hrlo⟩, hrk⟩, hrp⟩, hkv⟩, hlocols⟩, hpr⟩, hg⟩ := hm
            split at hpr
            · rename_i hpr'
              have hwsr : (words rec.table).getD #[] = ws := by rw [htab]; exact hwsd
              exact genOuts_spec words _ z hf rec ws vb _ _ (fun j => (ws.getD (loIdx + 2 ^ lo * hiIdx) 0).testBit j)
                (prod_sums words _ z hf rec ws hwsr hrn hrlo loM hiM hloI hhiI
                  (fun l hl => dl3 l (by rw [dl1]; exact hl)) (fun h hh => dh3 h (by rw [dh1]; exact hh)) dh2
                  (fun l hl => by simpa using hlocols l hl) hpr')
                (fun j _ h => hconst j h) hg o ho
            · cases hpr
        · cases hm

/-- **A placed read's outputs**: its segment embeds a generated read layout (`placedOk`, every index bit bound), so its
output columns carry the table's bits at the index its forms spell. -/
theorem placed_read_sound {types : String → Option CircuitType} {words : String → Option (Array Nat)}
    {ls : Array Layout} {done : Array D} {unit : Nat} {k : Chk} {z : ℕ → Bool} (hf : Facts words k z)
    {jj base : Nat} {wire : Array CF} {vals : List Bool} (hinv : Inv z wire vals) {idx : Array Form} {sha : String}
    {n vb lo : Nat} (hcheck : checkLayout types words ls done unit jj = true)
    (hown : (ls.getD jj default).own = .gen sha n vb lo) (hsz : idx.size = n)
    (hargs : ∀ i (hi : i < idx.size), idx[i].c ≤ 1)
    (hp : placedOk k (done.getD jj default) base [] wire idx = true) (o : Nat) :
    cfVal z (((done.getD jj default).outCols.map fun col => (([base + col], 0) : CF)).getD o ([], 0)) =
      ((List.range vb).map fun j =>
        (((words sha).getD #[]).getD (bitsIdx (idx.toList.map (formVal (vals.getD · false)))) 0).testBit j).getD o
        false := by
  obtain ⟨-, hg⟩ := genLayout_spec hcheck hown
  obtain ⟨hin, hout, hone, -, -, -, hins⟩ := genOk_facts hg
  generalize done.getD jj default = sub at *
  have hcol : ∀ i < n, sub.inCols.getD i 0 = i := by
    intro i hi
    rw [hin]
    simp [Array.getD, hi]
  obtain ⟨hsat, hcb1, hx⟩ := placed_core hf hinv hargs hone
    (fun w hw => by rw [hcol w (by omega)]; exact hins w (by omega)) hp
  have hS := genOk_sound words sub hg hsat hcb1
  have hvals : ((List.range n).map fun c => z (c + base)) = idx.toList.map (formVal (vals.getD · false)) := by
    apply List.ext_getElem (by simp [hsz])
    intro i h1 h2
    simp only [List.length_map, List.length_range] at h1
    have := hx i (by omega)
    rw [hcol i h1] at this
    simp [this, Array.getD, show i < idx.size by omega]
  rw [getD_map_col, list_getD_map_range]
  by_cases ho : o < sub.outCols.size
  · rw [ite_eq_left ho, ite_eq_left (by omega), cfVal_col, Nat.add_comm base]
    rw [hS o (by omega), hvals]
  · rw [ite_eq_right ho, ite_eq_right (by omega)]
    rfl

/-! ## The walk -/

theorem inv_args {z : ℕ → Bool} {wire : Array CF} {vals : List Bool} (hinv : Inv z wire vals) {args : Array Form}
    {n : Nat} (hsz : args.size = n) (hargs : ∀ i (hi : i < args.size), args[i].c ≤ 1) :
    Inv z (args.map (formOf wire)) ((List.range n).map fun i => formVal (vals.getD · false) (args.getD i default)) := by
  refine ⟨by simp [hsz], fun w => ?_⟩
  rw [list_getD_map_range]
  by_cases hw : w < args.size
  · rw [ite_eq_left (by omega)]
    rw [show (args.map (formOf wire)).getD w ([], 0) = formOf wire args[w] by simp [Array.getD, hw],
      show args.getD w default = args[w] by simp [Array.getD, hw]]
    exact hinv.formOf _ (hargs w hw)
  · rw [ite_eq_right (by omega)]
    simp [Array.getD, hw, cfVal_nil]

theorem walk_sound {types : String → Option CircuitType} {words : String → Option (Array Nat)} {ls : Array Layout}
    {done : Array D} {unit j : Nat} {k : Chk} {z : ℕ → Bool} (hf : Facts words k z)
    (hcallee : ∀ jj < j, jj ≠ unit →
      checkLayout types words ls done unit jj = true ∧ LayoutSound types words ls done jj) :
    ∀ jl top t l items c r wire refs wire' refs',
      walk types words ls done unit j k jl top t l items c r wire refs = some (wire', refs') →
      ∀ F, jl ≤ F → (top = true → jl = j) → ∀ vals, Inv z wire vals →
        Inv z wire' (runT types words (fun u y => evalT types words F u y) t vals items) := by
  intro jl
  induction jl using Nat.strongRecOn with
  | ind jl ihj =>
  intro top t l items
  induction items with
  | nil =>
    intro c r wire refs wire' refs' h F _ _ vals hinv
    unfold walk at h
    cases h
    exact hinv
  | cons it rest ih =>
    intro c r wire refs wire' refs' h F hF htop vals hinv
    cases it with
    | and fa fb =>
      unfold walk at h
      split at h
      · split at h
        · rename_i hc
          simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
          obtain ⟨⟨hfa, hfb⟩, hrow⟩ := hc
          have hz := hf.row _ _ hrow
          simp only [runT, itemOut]
          refine ih _ _ _ _ _ _ h F hF htop _ (hinv.push _ _ ?_)
          rw [cfVal_col, hz]
          simp only
          rw [xorSum_resolve z _ hf.const _ (formOf_snd_lt _ _), xorSum_resolve z _ hf.const _ (formOf_snd_lt _ _),
            hinv.formOf fa hfa, hinv.formOf fb hfb]
        · cases h
      · cases h
    | call kk args =>
      unfold walk at h
      split at h
      · cases h
      · rename_i u hu
        dsimp only at h
        split at h
        · split at h
          · rename_i hlt
            split at h
            · rename_i hc
              split at h
              · rename_i w'' refs'' hw
                simp only [Bool.and_eq_true, beq_iff_eq, Array.all_eq_true, decide_eq_true_eq] at hc
                obtain ⟨⟨⟨hsz, hargs⟩, hos⟩, houts⟩ := hc
                obtain ⟨F', rfl⟩ : ∃ F', F = F' + 1 := ⟨F - 1, by omega⟩
                have hin := ihj _ hlt false u _ _ 0 0 _ _ w'' refs'' hw F' (by omega) (by simp) _
                  (inv_args hinv hsz hargs)
                simp only [runT, itemOut, hu]
                refine ih _ _ _ _ _ _ h (F' + 1) hF htop _ (hinv.append (by simp [hos]) fun o => ?_)
                rw [list_getD_map_range]
                by_cases ho : o < u.outputs.size
                · rw [ite_eq_left (by omega), evalT_getD _ _ _ _ _ ho,
                    show (u.outputs.map (formOf w'')).getD o ([], 0) = formOf w'' u.outputs[o] by simp [Array.getD, ho]]
                  exact hin.formOf _ (houts o ho)
                · rw [ite_eq_right (by omega)]
                  simp [Array.getD, ho, cfVal_nil]
              · cases h
            · cases h
          · cases h
        · split at h
          · rename_i jj base ex refs''
            split at h
            · rename_i hc
              simp only [Bool.and_eq_true, beq_iff_eq, bne_iff_ne, ne_eq, decide_eq_true_eq, Array.all_eq_true] at hc
              obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨htop', hjj⟩, hjjl⟩, hjju⟩, hty⟩, hsz⟩, hargs⟩, -⟩, hout⟩, hp⟩ := hc
              obtain ⟨hchk, hsound⟩ := hcallee jj hjjl hjju
              have hty' : (ls.getD jj default).type.bind types = some u := by rw [hty]; exact hu
              simp only [runT, itemOut, hu]
              refine ih _ _ _ _ _ _ h F hF htop _
                (hinv.append (by rw [Array.size_map, List.length_map, List.length_range]; exact hout) fun o => ?_)
              exact placed_sound hf hinv hchk hjju hty' hsound hsz hargs hout hp F (by have := htop htop'; omega) o
            · cases h
          · cases h
    | read kk idx =>
      unfold walk at h
      split at h
      · cases h
      · rename_i sha n vb htab
        dsimp only at h
        split at h
        · rename_i jj _ _
          split at h
          · rename_i jj' base ex refs''
            split at h
            · rename_i sha' n' vb' lo' hown
              split at h
              · rename_i hc
                simp only [Bool.and_eq_true, beq_iff_eq, bne_iff_ne, ne_eq, decide_eq_true_eq, Array.all_eq_true]
                  at hc
                obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨htop', -⟩, hjjl⟩, hjju⟩, hsha⟩, hn⟩, hvb⟩, -⟩, hsz⟩, hargs⟩, hout⟩, hp⟩ := hc
                subst hsha hn hvb
                obtain ⟨hchk, -⟩ := hcallee jj hjjl hjju
                simp only [runT, itemOut, htab]
                refine ih _ _ _ _ _ _ h F hF htop _
                  (hinv.append (by rw [Array.size_map, List.length_map, List.length_range]; exact hout) fun o => ?_)
                exact placed_read_sound hf hinv hchk hown hsz hargs hp o
              · cases h
            · cases h
          · cases h
        · split at h
          · rename_i hc
            simp only [Bool.and_eq_true, beq_iff_eq, Array.all_eq_true, decide_eq_true_eq] at hc
            obtain ⟨⟨⟨⟨hsz, hargs⟩, hlo2⟩, hlon⟩, hws⟩ := hc
            split at h
            · cases h
            · rename_i loM refs1 hdlo
              split at h
              · cases h
              · rename_i hiM refs2 hdhi
                split at h
                · rename_i outs refs3 hro
                  generalize hlo : (l.reads.getD r default).lo = lo at *
                  -- the index forms, as columns, carry the index bits
                  have hfs : ∀ b, cfVal z ((idx.map (formOf wire)).getD b ([], 0)) =
                      (idx.toList.map (formVal (vals.getD · false))).getD b false := by
                    intro b
                    by_cases hb : b < idx.size
                    · rw [show (idx.map (formOf wire)).getD b ([], 0) = formOf wire idx[b] by simp [Array.getD, hb],
                        hinv.formOf _ (hargs b hb)]
                      simp [List.getD_eq_getElem?_getD, hb]
                    · simp [Array.getD, hb, List.getD_eq_getElem?_getD, cfVal_nil]
                  have hmap : (idx.map (formOf wire)).toList.map (cfVal z) = idx.toList.map (formVal (vals.getD · false)) := by
                    apply List.ext_getElem (by simp)
                    intro b hb1 hb2
                    have := hfs b
                    simp only [List.length_map, Array.length_toList, Array.size_map] at hb1
                    simp [Array.getD, hb1, List.getD_eq_getElem?_getD] at this
                    simpa using this
                  have hfs2 : ∀ a, ((idx.map (formOf wire)).getD a ([], 0)).2 < 2 := by
                    intro a
                    by_cases ha : a < idx.size
                    · rw [show (idx.map (formOf wire)).getD a ([], 0) = formOf wire idx[a] by simp [Array.getD, ha]]
                      exact formOf_snd_lt _ _
                    · simp [Array.getD, ha]
                  have hsz' : (idx.map (formOf wire)).size = n := by simpa using hsz
                  obtain ⟨dl1, -, dl3⟩ := chkDecode_spec words k z hf _ _ rfl refs loM refs1
                    (getD_extract_lt2 _ _ _ (by rw [hsz']; omega) hfs2) hdlo
                  obtain ⟨dh1, dh2, dh3⟩ := chkDecode_spec words k z hf _ _ rfl refs1 hiM refs2
                    (getD_extract_lt2 _ _ _ (by omega) hfs2) hdhi
                  have hlsz : ((idx.map (formOf wire)).extract 0 lo).size = lo := by
                    simp [Array.size_extract]; omega
                  have hhsz : ((idx.map (formOf wire)).extract lo n).size = n - lo := by
                    simp [Array.size_extract, hsz]
                  rw [hlsz, show max lo 1 = lo by omega] at dl1
                  rw [hhsz, show max (n - lo) 1 = n - lo by omega] at dh1
                  have hloI := bitsIdx_lt (((idx.map (formOf wire)).extract 0 lo).toList.map (cfVal z))
                  have hhiI := bitsIdx_lt (((idx.map (formOf wire)).extract lo n).toList.map (cfVal z))
                  simp only [List.length_map, Array.length_toList, hlsz, hhsz] at hloI hhiI
                  obtain ⟨loIdx, hloDef⟩ : ∃ x, bitsIdx (((idx.map (formOf wire)).extract 0 lo).toList.map (cfVal z)) = x :=
                    ⟨_, rfl⟩
                  obtain ⟨hiIdx, hhiDef⟩ : ∃ x, bitsIdx (((idx.map (formOf wire)).extract lo n).toList.map (cfVal z)) = x :=
                    ⟨_, rfl⟩
                  rw [hloDef] at dl3 hloI
                  rw [hhiDef] at dh3 hhiI
                  obtain ⟨ro1, ro3⟩ := readOk_spec words k z hf sha n lo vb loM hiM refs2 outs refs3 hlon
                    (by simpa using hws) hloI hhiI (fun l hl => dl3 l (by rw [dl1]; exact hl))
                    (fun h hh => dh3 h (by rw [dh1]; exact hh)) dh2 hro
                  -- the index the semantics reads is `loIdx + 2^lo · hiIdx`
                  have hindex : bitsIdx (idx.toList.map (formVal (vals.getD · false))) = loIdx + 2 ^ lo * hiIdx := by
                    rw [← hmap]
                    have hsplit := toList_split (idx.map (formOf wire)) lo
                    rw [show ((idx.map (formOf wire)).extract lo (idx.map (formOf wire)).size) =
                      (idx.map (formOf wire)).extract lo n by rw [hsz']] at hsplit
                    rw [hsplit, List.map_append, bitsIdx_append, hloDef, hhiDef]
                    simp [hlsz, Nat.min_eq_left (show lo ≤ idx.size by omega)]
                  simp only [runT, itemOut, htab]
                  refine ih _ _ _ _ _ _ h F hF htop _ (hinv.append (by simp [ro1]) fun o => ?_)
                  rw [list_getD_map_range, hindex]
                  by_cases ho : o < vb
                  · rw [ite_eq_left ho, ro3 o ho]
                  · rw [ite_eq_right ho]
                    simp [Array.getD, show ¬ o < outs.size by omega, cfVal_nil]
                · cases h
          · cases h
    | bad _ => unfold walk at h; cases h

/-! ## Every layout -/

/-- A checked layout, on column values where its row tests hold: each output column carries the type's output on its
input columns. -/
theorem checked_sound {types : String → Option CircuitType} {words : String → Option (Array Nat)} {ls : Array Layout}
    {done : Array D} {unit jj : Nat} (hcheck : checkLayout types words ls done unit jj = true) {t : CircuitType}
    (ht : (ls.getD jj default).type.bind types = some t) {z : ℕ → Bool}
    (hf : Facts words ⟨done.getD jj default, jj == unit⟩ z)
    (hcallee : ∀ jj' < jj, jj' ≠ unit →
      checkLayout types words ls done unit jj' = true ∧ LayoutSound types words ls done jj')
    (F : Nat) (hF : jj < F) (o : Nat) (ho : o < t.outputs.size) :
    z ((done.getD jj default).outCols.getD o 0) =
      (evalT types words F t fun i => z ((done.getD jj default).inCols.getD i 0)).getD o false := by
  obtain ⟨-, hoc, -, -, -, -, wire, hw, hrows⟩ := checkLayout_spec hcheck ht
  obtain ⟨F', rfl⟩ : ∃ F', F = F' + 1 := ⟨F - 1, by omega⟩
  have hinv0 : Inv z ((Array.range t.inBits).map fun w => ([(done.getD jj default).inCols.getD w 0], 0))
      ((List.range t.inBits).map fun i => z ((done.getD jj default).inCols.getD i 0)) := by
    refine ⟨by simp, fun w => ?_⟩
    rw [list_getD_map_range]
    by_cases hw' : w < t.inBits
    · rw [ite_eq_left hw']
      simp [Array.getD, hw', cfVal_col]
    · rw [ite_eq_right hw']
      simp [Array.getD, hw', cfVal_nil]
  have hinv := walk_sound hf hcallee jj true t _ _ 0 0 _ _ wire [] hw F' (by omega) (fun _ => rfl) _ hinv0
  have hget : t.outputs.getD o default = t.outputs[o] := by simp [Array.getD, ho]
  have hr := hf.row _ _ (hrows o ho)
  have hc1 : z (done.getD jj default).const = true := hf.const
  rw [hr, evalT_getD _ _ _ _ _ ho, xorSum_resolve z _ hf.const _ (formOf_snd_lt _ _), hget,
    hinv.formOf _ (hoc o ho)]
  simp only
  rw [xorSum_cons, xorSum_nil, hc1]
  simp

/-- **Every inner layout's rows compute its type.** -/
theorem layouts_sound_of {types : String → Option CircuitType} {words : String → Option (Array Nat)} {ls : Array Layout}
    {done : Array D} {unit : Nat} (hall : ∀ jj < ls.size, checkLayout types words ls done unit jj = true) :
    ∀ jj < ls.size, jj ≠ unit → LayoutSound types words ls done jj := by
  intro jj
  induction jj using Nat.strongRecOn with
  | ind jj ih =>
  intro hjj hne u hu z hz h1 F hF o ho
  have hk : (jj == unit) = false := by simp [hne]
  exact checked_sound (hall jj hjj) hu (by rw [hk]; exact facts_inner hz h1)
    (fun jj' hlt hne' => ⟨hall jj' (by omega), ih jj' hlt (by omega) hne'⟩) F hF o ho

/-- **The unit's rows compute its type**: column values that satisfy its rows in the block (Δ applied), with the block's
constant `one` at 1, carry on each output column the type's output on its input columns. -/
theorem unit_sound_of {types : String → Option CircuitType} {words : String → Option (Array Nat)} {ls : Array Layout}
    {done : Array D} {unit : Nat} (hall : ∀ jj < ls.size, checkLayout types words ls done unit jj = true)
    (hu : unit < ls.size) {t : CircuitType} (ht : (ls.getD unit default).type.bind types = some t) (one : Nat)
    {z : ℕ → Bool}
    (hz : SatAt (blockRow words (done.getD unit default) one) z)
    (h1 : z one = true) (F : Nat) (hF : unit < F) (o : Nat) (ho : o < t.outputs.size) :
    z ((done.getD unit default).outCols.getD o 0) =
      (evalT types words F t fun i => z ((done.getD unit default).inCols.getD i 0)).getD o false := by
  obtain ⟨-, -, -, -, hone, -⟩ := checkLayout_spec (hall unit hu) ht
  have hk : (unit == unit) = true := by simp
  rw [hk] at hone
  exact checked_sound (hall unit hu) ht (by rw [hk]; exact facts_unit hz h1 hone)
    (fun jj hlt hne => ⟨hall jj (by omega), layouts_sound_of hall jj (by omega) hne⟩) F hF o ho

theorem deriveChecked_checks {types : String → Option CircuitType} {ls : Array Layout} {info : String → Option TableInfo}
    {words : String → Option (Array Nat)} {unit : Nat} {done : Array D}
    (h : deriveChecked types ls info words unit = .ok done) :
    (∀ jj < ls.size, checkLayout types words ls done unit jj = true) ∧ orderChecked words done unit = true ∧
      partsChecked done unit = true := by
  unfold deriveChecked at h
  cases hd : deriveAll types ls info words unit with
  | error e => simp [hd, bind, Except.bind] at h
  | ok done' =>
    simp only [hd, bind, Except.bind] at h
    split at h
    · cases h
    · rename_i hf
      split at h
      · cases h
      · rename_i ho
        split at h
        · cases h
        · rename_i hp
          cases h
          refine ⟨fun jj hjj => ?_, by simpa using ho, by simpa using hp⟩
          have := List.find?_eq_none.mp hf jj (List.mem_range.mpr hjj)
          simpa using this

theorem deriveChecked_spec {types : String → Option CircuitType} {ls : Array Layout} {info : String → Option TableInfo}
    {words : String → Option (Array Nat)} {unit : Nat} {done : Array D}
    (h : deriveChecked types ls info words unit = .ok done) :
    (∀ jj < ls.size, checkLayout types words ls done unit jj = true) ∧ orderChecked words done unit = true :=
  ⟨(deriveChecked_checks h).1, (deriveChecked_checks h).2.1⟩

theorem deriveChecked_parts {types : String → Option CircuitType} {ls : Array Layout} {info : String → Option TableInfo}
    {words : String → Option (Array Nat)} {unit : Nat} {done : Array D}
    (h : deriveChecked types ls info words unit = .ok done) : partsChecked done unit = true :=
  (deriveChecked_checks h).2.2

theorem deriveChecked_all {types : String → Option CircuitType} {ls : Array Layout} {info : String → Option TableInfo}
    {words : String → Option (Array Nat)} {unit : Nat} {done : Array D}
    (h : deriveChecked types ls info words unit = .ok done) :
    ∀ jj < ls.size, checkLayout types words ls done unit jj = true :=
  (deriveChecked_spec h).1

/-! ## The unit's logical order -/

theorem colSet_go (c : Nat) : ∀ (o : List Nat) (s : Std.HashSet Nat),
    (o.foldl (fun s c => s.insert c) s).contains c = true ↔ (s.contains c = true ∨ c ∈ o)
  | [], s => by simp
  | x :: o, s => by
    rw [List.foldl_cons, colSet_go c o, Std.HashSet.contains_insert]
    simp only [Bool.or_eq_true, beq_iff_eq, List.mem_cons]
    constructor
    · intro h
      rcases h with (h | h) | h
      · exact Or.inr (Or.inl h.symm)
      · exact Or.inl h
      · exact Or.inr (Or.inr h)
    · intro h
      rcases h with h | h | h
      · exact Or.inl (Or.inr h)
      · exact Or.inl (Or.inl h.symm)
      · exact Or.inr h

theorem colSet_contains (o : List Nat) (c : Nat) : (colSet o).contains c = true ↔ c ∈ o := by
  unfold colSet
  rw [colSet_go]
  simp [Std.HashSet.contains_empty]

theorem orderOk_spec (words : String → Option (Array Nat)) (u : D) (nIn : Nat) :
    ∀ (o : List Nat) (ℓ : Nat) (seen : Std.HashSet Nat) (pre : List Nat),
    (∀ c, seen.contains c = true ↔ c ∈ pre) → pre.length = ℓ → pre.Nodup → orderOk words u nIn o ℓ seen = true →
    (pre ++ o).Nodup ∧ (∀ c ∈ o, c < u.rows.size) ∧
      ∀ m (hm : m < o.length), nIn ≤ ℓ + m → ∀ x ∈ readsOf words u o[m], x ∈ pre ++ o.take m
  | [], _, _, pre, _, _, hnd, _ => by simpa using hnd
  | c :: rest, ℓ, seen, pre, hs, hl, hnd, h => by
    unfold orderOk at h
    simp only [Bool.and_eq_true, decide_eq_true_eq, Bool.not_eq_true', Bool.or_eq_true, List.all_eq_true] at h
    obtain ⟨⟨⟨hc, hnew⟩, hreads⟩, hrest⟩ := h
    have hcpre : c ∉ pre := fun hm => by
      rw [← hs] at hm
      rw [hm] at hnew
      cases hnew
    have hs' : ∀ x, (seen.insert c).contains x = true ↔ x ∈ pre ++ [c] := by
      intro x
      rw [Std.HashSet.contains_insert]
      simp only [Bool.or_eq_true, beq_iff_eq, List.mem_append, List.mem_singleton, hs]
      constructor
      · intro h; rcases h with h | h
        · exact Or.inr h.symm
        · exact Or.inl h
      · intro h; rcases h with h | h
        · exact Or.inr h
        · exact Or.inl h.symm
    have hnd' : (pre ++ [c]).Nodup :=
      List.nodup_append.mpr ⟨hnd, List.nodup_cons.mpr ⟨by simp, List.nodup_nil⟩, fun a ha b hb hab => by
        rw [List.mem_singleton] at hb; subst hb; subst hab; exact hcpre ha⟩
    obtain ⟨h1, h2, h3⟩ := orderOk_spec words u nIn rest (ℓ + 1) _ (pre ++ [c]) hs' (by simp [hl]) hnd' hrest
    refine ⟨by simpa using h1, fun x hx => ?_, fun m hm hnm x hx => ?_⟩
    · rcases List.mem_cons.mp hx with rfl | hx
      · exact hc
      · exact h2 x hx
    · cases m with
      | zero =>
        have hℓ : ¬ ℓ < nIn := by omega
        have := hreads.resolve_left hℓ x hx
        rw [hs] at this
        simpa using this
      | succ m =>
        have := h3 m (by simpa using hm) (by omega) x (by simpa using hx)
        simpa using this

theorem readsOf_blockRow {words : String → Option (Array Nat)} {u : D} {one c : Nat} {p : List Nat × List Nat}
    (h : blockRow words u one c = some p) : ∀ x ∈ p.1 ++ p.2, x ≠ one → x ∈ readsOf words u c := by
  obtain ⟨a, b⟩ := p
  unfold blockRow at h
  unfold readsOf
  split at h
  · cases h
    intro x hx hne
    simp at hx
    exact absurd hx hne
  · cases h
    intro x hx _
    simpa using hx
  · rw [h]
    intro x hx _
    exact hx

theorem outsideOk_spec {u : D} {o : List Nat} (h : outsideOk u o = true) :
    (∀ c < u.rows.size, c ∉ o → u.rows[c]? = some ([], [])) ∧ (∀ e ∈ u.delta, e.1 ∈ o) ∧
      ∀ r ∈ u.reads, ∀ c, r.covers c = true → c ∈ o := by
  unfold outsideOk at h
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range, Bool.or_eq_true, beq_iff_eq, Array.all_eq_true] at h
  obtain ⟨⟨h1, h2⟩, h3⟩ := h
  refine ⟨fun c hc hco => ?_, fun e he => ?_, fun r hr c hc => ?_⟩
  · rcases h1 c hc with h' | h'
    · exact absurd ((colSet_contains o c).mp h') hco
    · exact h'
  · obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.mp he
    exact (colSet_contains o _).mp (h2 i hi)
  · obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.mp hr
    simp only [ReadRec.covers, Bool.and_eq_true, decide_eq_true_eq] at hc
    have := h3 i hi (c - u.reads[i].prod0) (by omega)
    rw [show u.reads[i].prod0 + (c - u.reads[i].prod0) = c by omega] at this
    exact (colSet_contains o c).mp this

theorem inputs_self_of {types : String → Option CircuitType} {words : String → Option (Array Nat)} {ls : Array Layout}
    {done : Array D} {unit : Nat} (hall : ∀ jj < ls.size, checkLayout types words ls done unit jj = true)
    (hu : unit < ls.size) {t : CircuitType} (ht : (ls.getD unit default).type.bind types = some t) (one : Nat) :
    ∀ i < (done.getD unit default).inCols.size,
      blockRow words (done.getD unit default) one ((done.getD unit default).inCols.getD i 0) =
        some ([(done.getD unit default).inCols.getD i 0], [(done.getD unit default).inCols.getD i 0]) := by
  obtain ⟨-, -, hin, -, -, hins, -⟩ := checkLayout_spec (hall unit hu) ht
  have hk : (unit == unit) = true := by simp
  rw [hk] at hins
  intro i hi
  exact rowIs_unit words (hins i (by omega)) one

/-- **The unit's rows are in logical order**, as `Rows.compose` reads them. If `deriveChecked` accepts, then for `u` the
unit's rows and `order done u`: no column repeats; each is one of `u`'s rows; the order starts with `u`'s input columns;
each row after them reads, besides `one`, only columns before it; and every column outside the order has a zero row or
none. -/
theorem order_sound {types : String → Option CircuitType} {ls : Array Layout} {info : String → Option TableInfo}
    {words : String → Option (Array Nat)} {unit : Nat} {done : Array D}
    (hd : deriveChecked types ls info words unit = .ok done) (one : Nat) :
    (order done (done.getD unit default)).Nodup ∧
      (∀ c ∈ order done (done.getD unit default), c < (done.getD unit default).rows.size) ∧
      (order done (done.getD unit default)).take (done.getD unit default).inCols.size =
        (done.getD unit default).inCols.toList ∧
      (∀ ℓ (hℓ : ℓ < (order done (done.getD unit default)).length), (done.getD unit default).inCols.size ≤ ℓ →
        ∀ p, blockRow words (done.getD unit default) one (order done (done.getD unit default))[ℓ] = some p →
          ∀ x ∈ p.1 ++ p.2, x ≠ one → x ∈ (order done (done.getD unit default)).take ℓ) ∧
      ∀ c p, c ∉ order done (done.getD unit default) → blockRow words (done.getD unit default) one c = some p →
        p = ([], []) := by
  have ho := (deriveChecked_spec hd).2
  unfold orderChecked at ho
  simp only [Bool.and_eq_true] at ho
  generalize done.getD unit default = u at *
  obtain ⟨hnd, hlt, hrd⟩ := orderOk_spec words u u.inCols.size (order done u) 0 {} []
    (by simp [Std.HashSet.contains_empty]) rfl List.nodup_nil ho.1
  obtain ⟨hz, hdel, hrds⟩ := outsideOk_spec ho.2
  refine ⟨by simpa using hnd, hlt, by simp [order], fun ℓ hℓ hn p hp x hx hne => ?_, fun c p hc hp => ?_⟩
  · simpa using hrd ℓ hℓ (by omega) x (readsOf_blockRow hp x hx hne)
  · unfold blockRow at hp
    split at hp
    · rename_i x hf
      have hx : x = c := by simpa using Array.find?_some hf
      exact absurd (hx ▸ hdel _ (Array.mem_of_find?_eq_some hf)) hc
    · rename_i x s hf
      have hx : x = c := by simpa using Array.find?_some hf
      exact absurd (hx ▸ hdel _ (Array.mem_of_find?_eq_some hf)) hc
    · -- no read covers a column outside the order, so the row is the derived one
      have hnc : (u.reads.find? (·.covers c)).isNone = true := by
        cases hf : u.reads.find? (·.covers c) with
        | none => rfl
        | some r => exact absurd (hrds r (Array.mem_of_find?_eq_some hf) c (by simpa using Array.find?_some hf)) hc
      rw [fullRow_of_uncovered hnc] at hp
      have hcs : c < u.rows.size := (Array.getElem?_eq_some_iff.mp hp).1
      rw [hz c hcs hc] at hp
      exact (Option.some.inj hp).symm

/-- **With `one` past every row, `[one]·[one]` rows are exactly Δ's constant copies.** -/
theorem blockRow_one_iff {types : String → Option CircuitType} {ls : Array Layout} {info : String → Option TableInfo}
    {words : String → Option (Array Nat)} {unit : Nat} {done : Array D}
    (hd : deriveChecked types ls info words unit = .ok done) {t : CircuitType}
    (ht : (ls.getD unit default).type.bind types = some t) {one : Nat}
    (hone : (done.getD unit default).rows.size ≤ one) (c : Nat) :
    blockRow words (done.getD unit default) one c = some ([one], [one]) ↔
      (done.getD unit default).delta.find? (·.1 == c) = some (c, none) := by
  have hall := (deriveChecked_spec hd).1
  have ho := (deriveChecked_spec hd).2
  have hu : unit < ls.size := by
    by_cases h : unit < ls.size
    · exact h
    · have : ls.getD unit default = default := by simp [Array.getD, h]
      rw [this] at ht; cases ht
  obtain ⟨-, -, hin, -, -, hins, -⟩ := checkLayout_spec (hall unit hu) ht
  have hk : (unit == unit) = true := by simp
  rw [hk] at hins
  have hout := (order_sound hd one).2.2.2.2
  unfold orderChecked at ho
  simp only [Bool.and_eq_true] at ho
  generalize done.getD unit default = u at *
  obtain ⟨-, hlt, hrd⟩ := orderOk_spec words u u.inCols.size (order done u) 0 {} []
    (by simp [Std.HashSet.contains_empty]) rfl List.nodup_nil ho.1
  obtain ⟨-, hdel, -⟩ := outsideOk_spec ho.2
  have hpre : ∀ ℓ (hℓ : ℓ < (order done u).length), ℓ < u.inCols.size → (order done u)[ℓ] = u.inCols.getD ℓ 0 := by
    intro ℓ hℓ hin'
    simp [order, List.getElem_append_left (by simpa using hin' : ℓ < u.inCols.toList.length), Array.getD, hin']
  -- a column past every row is in no row the order reads
  have hnot : ∀ ℓ (hℓ : ℓ < (order done u).length), u.inCols.size ≤ ℓ →
      one ∉ readsOf words u (order done u)[ℓ] := by
    intro ℓ hℓ hn hm
    have := hrd ℓ hℓ (by omega) one hm
    have := hlt one (List.mem_of_mem_take (by simpa using this))
    omega
  constructor
  · intro h
    cases hf : u.delta.find? (·.1 == c) with
    | none =>
      exfalso
      have hr : fullRow words u c = some ([one], [one]) := by simpa [blockRow, hf] using h
      by_cases hco : c ∈ order done u
      · obtain ⟨ℓ, hℓ, hcℓ⟩ := List.getElem_of_mem hco
        by_cases hin' : ℓ < u.inCols.size
        · have hs := rowIs_unit words (hins ℓ (by omega)) one
          rw [← hpre ℓ hℓ hin', hcℓ, h] at hs
          have : one = c := by simpa using hs
          have := hlt c hco
          omega
        · have := hnot ℓ hℓ (by omega)
          rw [hcℓ] at this
          simp [readsOf, hf, hr] at this
      · have := hout c _ hco h
        simp at this
    | some e =>
      obtain ⟨x, s⟩ := e
      have hx : x = c := by simpa using Array.find?_some hf
      subst hx
      cases s with
      | none => rfl
      | some s =>
        exfalso
        have hss : s = [one] := by
          have : (s, s) = ([one], [one]) := by simpa [blockRow, hf] using h
          simpa using this
        have hco : x ∈ order done u := hdel _ (Array.mem_of_find?_eq_some hf)
        obtain ⟨ℓ, hℓ, hcℓ⟩ := List.getElem_of_mem hco
        by_cases hin' : ℓ < u.inCols.size
        · have hs := hins ℓ (by omega)
          simp only [Chk.rowIs, Bool.and_eq_true, beq_iff_eq, Bool.not_true, Bool.false_or,
            Option.isNone_iff_eq_none] at hs
          rw [← hpre ℓ hℓ hin', hcℓ, hf] at hs
          cases hs.2
        · have := hnot ℓ hℓ (by omega)
          rw [hcℓ] at this
          simp [readsOf, hf, hss] at this
  · intro h
    simp [blockRow, h]

/-! ## The statements -/

theorem lt_of_type {ls : Array Layout} {types : String → Option CircuitType} {j : ℕ} {t : CircuitType}
    (ht : (ls.getD j default).type.bind types = some t) : j < ls.size := by
  by_contra hj
  have : ls.getD j default = default := by simp [Array.getD, hj]
  rw [this] at ht
  cases ht

/-- **The unit's derived rows compute its type**, over the type DAG. If `deriveChecked` accepts the layouts, column
values that satisfy the unit's rows as the block holds them (Δ applied), with the block's constant `one` at 1, carry on
each output column the unit type's output on its input columns. -/
theorem unit_sound {types : String → Option CircuitType} {ls : Array Layout} {info : String → Option TableInfo}
    {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D}
    (hd : deriveChecked types ls info words unit = .ok done) {t : CircuitType}
    (ht : (ls.getD unit default).type.bind types = some t) (one : ℕ) (z : ℕ → Bool)
    (hz : SatAt (blockRow words (done.getD unit default) one) z) (h1 : z one = true) (F : ℕ) (hF : unit < F) (o : ℕ)
    (ho : o < t.outputs.size) :
    z ((done.getD unit default).outCols.getD o 0) =
      (evalT types words F t fun i => z ((done.getD unit default).inCols.getD i 0)).getD o false :=
  unit_sound_of (deriveChecked_all hd) (lt_of_type ht) ht one hz h1 F hF o ho

/-- **Every other layout's derived rows compute its type.** If `deriveChecked` accepts the layouts, column values that
satisfy layout `j`'s rows as the verifier folds them (a placed or inlined callee's, in its own coordinates), with its
constant at 1, carry on each output column its type's output on its input columns. -/
theorem layout_sound {types : String → Option CircuitType} {ls : Array Layout} {info : String → Option TableInfo}
    {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D}
    (hd : deriveChecked types ls info words unit = .ok done) {j : ℕ} (hj : j ≠ unit) {t : CircuitType}
    (ht : (ls.getD j default).type.bind types = some t) (z : ℕ → Bool) (hz : SatAt (fullRow words (done.getD j default)) z)
    (h1 : z (done.getD j default).const = true) (F : ℕ) (hF : j < F) (o : ℕ) (ho : o < t.outputs.size) :
    z ((done.getD j default).outCols.getD o 0) =
      (evalT types words F t fun i => z ((done.getD j default).inCols.getD i 0)).getD o false :=
  layouts_sound_of (deriveChecked_all hd) j (lt_of_type ht) hj t ht z hz h1 F hF o ho

/-- **The unit's input rows are self rows** in the block, whatever `one` is. -/
theorem inputs_self {types : String → Option CircuitType} {ls : Array Layout} {info : String → Option TableInfo}
    {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D}
    (hd : deriveChecked types ls info words unit = .ok done) {t : CircuitType}
    (ht : (ls.getD unit default).type.bind types = some t) (one : ℕ) :
    ∀ i < (done.getD unit default).inCols.size,
      blockRow words (done.getD unit default) one ((done.getD unit default).inCols.getD i 0) =
        some ([(done.getD unit default).inCols.getD i 0], [(done.getD unit default).inCols.getD i 0]) :=
  inputs_self_of (deriveChecked_all hd) (lt_of_type ht) ht one

/-! ## Flat types -/

/-- On AND items that read only earlier wires, `runT`'s wire list is `run`'s wire values. -/
theorem runT_ands (types : String → Option CircuitType) (words : String → Option (Array ℕ))
    (ev : CircuitType → (ℕ → Bool) → List Bool) (t : CircuitType) :
    ∀ (items : List Item) (vals : List Bool) (v : ℕ → Bool),
      (∀ k (hk : k < items.length), ∃ a b, items[k] = .and a b ∧ (∀ x ∈ a.ws.toList, x < vals.length + k) ∧
        (∀ x ∈ b.ws.toList, x < vals.length + k)) →
      (∀ j < vals.length, vals.getD j false = v j) →
      (runT types words ev t vals items).length = vals.length + items.length ∧
        ∀ j < vals.length + items.length, (runT types words ev t vals items).getD j false = run v vals.length items j
  | [], vals, v, _, hv => ⟨by simp [runT], fun j hj => by simpa [runT, run] using hv j (by simpa using hj)⟩
  | it :: rest, vals, v, hit, hv => by
    obtain ⟨a, b, hab, ha, hb⟩ := hit 0 (by simp)
    simp only [List.getElem_cons_zero, Nat.add_zero] at hab ha hb
    subst hab
    have hval : (formVal (vals.getD · false) a && formVal (vals.getD · false) b) = itemVal v (.and a b) := by
      simp only [itemVal]
      rw [formVal_congr fun x hx => hv x (ha x hx), formVal_congr fun x hx => hv x (hb x hx)]
    obtain ⟨h1, h2⟩ := runT_ands types words ev t rest (vals ++ [itemVal v (.and a b)])
      (fun j => if j = vals.length then itemVal v (.and a b) else v j)
      (fun k hk => by
        obtain ⟨a', b', h', ha', hb'⟩ := hit (k + 1) (by simp; omega)
        refine ⟨a', b', by simpa using h', fun x hx => ?_, fun x hx => ?_⟩
        · have := ha' x hx; simp; omega
        · have := hb' x hx; simp; omega)
      (fun j hj => by
        simp only [List.length_append, List.length_singleton] at hj
        by_cases hjw : j = vals.length
        · subst hjw
          simp [List.getD_eq_getElem?_getD]
        · have hj' : j < vals.length := by omega
          simp only [hjw, ↓reduceIte]
          rw [← hv j hj', List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD,
            List.getElem?_append_left hj'])
    simp only [runT, itemOut]
    rw [hval]
    refine ⟨by rw [h1]; simp; omega, fun j hj => ?_⟩
    simp only [List.length_cons] at hj
    rw [h2 j (by simp; omega)]
    simp [run]

/-- **A flat type's `evalT` is S2's `eval`**, at any fuel past zero: for a type `flatOk` accepts (AND items over
earlier wires, output forms over the wires), `Types.Dag`'s statements say what `Types.Flat`'s do. -/
theorem evalT_flat (types : String → Option CircuitType) (words : String → Option (Array ℕ)) {t : CircuitType}
    (h : Flock.Derive.flatOk t = true) (F : ℕ) (x : ℕ → Bool) : evalT types words (F + 1) t x = eval t x := by
  have hlen : ((List.range t.inBits).map x).length = t.inBits := by simp
  obtain ⟨-, h2⟩ := runT_ands types words (fun u y => evalT types words F u y) t t.items.toList
    ((List.range t.inBits).map x) x
    (fun k hk => by
      obtain ⟨a, b, hab, ha, hb⟩ := flatOk_items h k (by simpa using hk)
      exact ⟨a, b, by simpa using hab, fun y hy => by rw [hlen]; exact ha y hy, fun y hy => by rw [hlen]; exact hb y hy⟩)
    (fun j hj => by rw [hlen] at hj; simp [List.getD_eq_getElem?_getD, hj])
  simp only [evalT, eval]
  apply List.map_congr_left
  intro f hf
  apply formVal_congr
  intro y hy
  obtain ⟨o, ho, rfl⟩ := List.getElem_of_mem hf
  have hy' := (flatOk_outputs h).2 o (by simpa using ho) y (by simpa using hy)
  rw [h2 y (by rw [hlen]; simpa using hy'), hlen]

end FlockSoundness.Types.Dag
