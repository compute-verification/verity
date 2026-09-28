import Flock.DeriveCheck
import FlockSoundness.Types.Flat

/-!
# Derived rows compute their circuit type, over the type DAG

`Flock.DeriveAll.deriveChecked` is the function the verifier runs to lay a statement's circuit types out as rows (Project
doc `verified-lowering-design.md` §1.2, S3c): `deriveAll` builds every layout's rows, and its check walks each layout's type
item by item against them. From the walk alone:

* **`layout_sound`**: column values that satisfy a placed or inlined layout's rows, with its constant at 1, carry on each
  output column the layout's type's output (`evalT`, its callees by digest) on its input columns;
* **`unit_sound`**: the same for the unit's rows as the block holds them (Δ applied, `Flock.DeriveAll.blockRow`), with the
  block's constant at 1.

So both hold of whatever `deriveAll` builds: nothing here is about how it builds. The flat case is `compose_sound`
(`Types/Flat.lean`). Reads are refused by the check for now (S3c-2), so `evalT`'s read case is not yet used.
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
      let i := (List.range idx.size).foldl
        (fun acc b => acc + if formVal (vals.getD · false) (idx.getD b default) then 2 ^ b else 0) 0
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

/-- On column values `z` that satisfy a layout's rows: each tested row's equation, each constant copy at 1, each binding
copy equal to its source once the callee's constant is 1, and the layout's constant at 1. -/
structure Facts (k : Chk) (z : ℕ → Bool) : Prop where
  row : ∀ c r, k.rowIs c r = true → z c = (xorSum r.1 z && xorSum r.2 z)
  one : ∀ c, k.oneAt c = true → z c = true
  bind : ∀ c src cb, k.bindAt c src cb = true → z cb = true → z c = xorSum src z
  const : z k.d.const = true

theorem rowIs_inner {d : D} {c : Nat} {r : List Nat × List Nat} (h : Chk.rowIs ⟨d, false⟩ c r = true) :
    d.rows[c]? = some r := by
  simpa [Chk.rowIs] using h

theorem oneAt_inner {d : D} {c : Nat} (h : Chk.oneAt ⟨d, false⟩ c = true) : d.rows[c]? = some ([d.const], [d.const]) := by
  simpa [Chk.oneAt] using h

theorem bindAt_inner {d : D} {c : Nat} {src : List Nat} {cb : Nat} (h : Chk.bindAt ⟨d, false⟩ c src cb = true) :
    d.rows[c]? = some (src, [cb]) := by
  simpa [Chk.bindAt] using h

theorem rowIs_unit {d : D} {c : Nat} {r : List Nat × List Nat} (h : Chk.rowIs ⟨d, true⟩ c r = true) (one : Nat) :
    blockRow d one c = some r := by
  simp only [Chk.rowIs, Bool.and_eq_true, beq_iff_eq, Bool.not_true, Bool.false_or, Option.isNone_iff_eq_none] at h
  simp [blockRow, h.2, h.1]

theorem oneAt_unit {d : D} {c : Nat} (h : Chk.oneAt ⟨d, true⟩ c = true) (one : Nat) :
    blockRow d one c = some ([one], [one]) := by
  simp only [Chk.oneAt, ↓reduceIte, beq_iff_eq] at h
  simp [blockRow, h]

theorem bindAt_unit {d : D} {c : Nat} {src : List Nat} {cb : Nat} (h : Chk.bindAt ⟨d, true⟩ c src cb = true)
    (one : Nat) : blockRow d one c = some (src, src) := by
  simp only [Chk.bindAt, ↓reduceIte, beq_iff_eq] at h
  simp [blockRow, h]

theorem facts_inner {d : D} {z : ℕ → Bool} (hz : Sat d.rows.toList z) (h1 : z d.const = true) :
    Facts ⟨d, false⟩ z where
  row c r h := hz c r (by rw [Array.getElem?_toList]; exact rowIs_inner h)
  one c h := by
    rw [hz c _ (by rw [Array.getElem?_toList]; exact oneAt_inner h)]
    simp [xorSum, h1]
  bind c src cb h hcb := by
    rw [hz c _ (by rw [Array.getElem?_toList]; exact bindAt_inner h)]
    simp [xorSum, hcb]
  const := h1

theorem facts_unit {d : D} {one : Nat} {z : ℕ → Bool}
    (hz : ∀ c r, blockRow d one c = some r → z c = (xorSum r.1 z && xorSum r.2 z)) (h1 : z one = true)
    (hc : Chk.oneAt ⟨d, true⟩ d.const = true) : Facts ⟨d, true⟩ z where
  row c r h := hz c r (rowIs_unit h one)
  one c h := by
    rw [hz c _ (oneAt_unit h one)]
    simp [xorSum, h1]
  bind c src cb h _ := by
    rw [hz c _ (bindAt_unit h one)]
    simp
  const := by
    rw [hz _ _ (oneAt_unit hc one)]
    simp [xorSum, h1]

/-! ## A layout's check, unfolded -/

theorem getD_of_getElem? {α : Type} {a : Array α} {i : Nat} {x d : α} (h : a[i]? = some x) : a.getD i d = x := by
  obtain ⟨hi, hx⟩ := Array.getElem?_eq_some_iff.mp h
  simp [Array.getD, hi, hx]

theorem checkLayout_spec {types : String → Option CircuitType} {ls : Array Layout} {done : Array D} {unit jj : Nat}
    (h : checkLayout types ls done unit jj = true) :
    ∃ t, (ls.getD jj default).type.bind types = some t ∧ t.outputs.size = t.outBits ∧
      (∀ o (ho : o < t.outputs.size), t.outputs[o].c ≤ 1) ∧
      (done.getD jj default).inCols.size = t.inBits ∧ (done.getD jj default).outCols.size = t.outputs.size ∧
      Chk.oneAt ⟨done.getD jj default, jj == unit⟩ (done.getD jj default).const = true ∧
      (∀ w < t.inBits, Chk.rowIs ⟨done.getD jj default, jj == unit⟩ ((done.getD jj default).inCols.getD w 0)
        ([(done.getD jj default).inCols.getD w 0], [(done.getD jj default).inCols.getD w 0]) = true) ∧
      ∃ wire, walk types ls done unit jj ⟨done.getD jj default, jj == unit⟩ jj true t (ls.getD jj default)
          t.items.toList 0 ((Array.range t.inBits).map fun w => ([(done.getD jj default).inCols.getD w 0], 0))
          (done.getD jj default).items.toList = some (wire, []) ∧
        ∀ o < t.outputs.size, Chk.rowIs ⟨done.getD jj default, jj == unit⟩ ((done.getD jj default).outCols.getD o 0)
          (resolve (done.getD jj default).const (ent (formOf wire (t.outputs.getD o default))),
            [(done.getD jj default).const]) = true := by
  unfold checkLayout at h
  dsimp only at h
  split at h
  · rename_i t _ ht
    simp only [Bool.and_eq_true, beq_iff_eq, List.all_eq_true, List.mem_range] at h
    obtain ⟨⟨⟨⟨⟨⟨hos, hoc⟩, hin⟩, hout⟩, hone⟩, hins⟩, hw⟩ := h
    refine ⟨t, ht, hos, fun o ho => by simpa using (Array.all_eq_true.mp hoc) o ho, hin, hout, hone, hins, ?_⟩
    split at hw
    · rename_i _ wire hwk
      exact ⟨wire, hwk, fun o ho => List.all_eq_true.mp hw o (List.mem_range.mpr ho)⟩
    · cases hw
  · cases h

/-! ## A placed call -/

/-- Layout `jj`'s rows compute its type, at every fuel past its index. -/
def LayoutSound (types : String → Option CircuitType) (words : String → Option (Array Nat)) (ls : Array Layout)
    (done : Array D) (jj : Nat) : Prop :=
  ∀ u, (ls.getD jj default).type.bind types = some u → ∀ z, Sat (done.getD jj default).rows.toList z →
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

theorem placed_sound {types : String → Option CircuitType} {words : String → Option (Array Nat)} {ls : Array Layout}
    {done : Array D} {unit : Nat} {k : Chk} {z : ℕ → Bool} (hf : Facts k z) {jj base : Nat} {ex : List Nat}
    {wire : Array CF} {vals : List Bool} (hinv : Inv z wire vals) {args : Array Form} {u : CircuitType}
    (hcheck : checkLayout types ls done unit jj = true) (hjj : jj ≠ unit)
    (hty : (ls.getD jj default).type.bind types = some u) (hsound : LayoutSound types words ls done jj)
    (hsz : args.size = u.inBits) (hargs : ∀ i (hi : i < args.size), args[i].c ≤ 1)
    (hout : (done.getD jj default).outCols.size = u.outBits)
    (hp : placedOk k (done.getD jj default) base ex wire args = true) (F : Nat) (hF : jj < F) (o : Nat) :
    cfVal z (((done.getD jj default).outCols.map fun col => (([base + col], 0) : CF)).getD o ([], 0)) =
      ((List.range u.outBits).map fun o =>
        (evalT types words F u fun i => formVal (vals.getD · false) (args.getD i default)).getD o false).getD o false := by
  obtain ⟨u', hty', hos', -, -, -, hone', hins', -⟩ := checkLayout_spec hcheck
  rw [hty] at hty'
  cases hty'
  have hjj' : (jj == unit) = false := by simp [hjj]
  rw [hjj'] at hone' hins'
  unfold LayoutSound at hsound
  generalize done.getD jj default = sub at *
  unfold placedOk at hp
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range] at hp
  obtain ⟨⟨hcb, hbind⟩, hemb⟩ := hp
  have hcb1 : z (base + sub.const) = true := hf.one _ hcb
  -- the callee's rows hold at its base
  have hsat : Sat sub.rows.toList fun c => z (c + base) := by
    intro c r hr
    rw [Array.getElem?_toList] at hr
    have hc : c < sub.rows.size := (Array.getElem?_eq_some_iff.mp hr).1
    have hrc := hemb c hc
    simp only [Bool.or_eq_true, beq_iff_eq, List.any_eq_true, List.mem_range, Bool.and_eq_true,
      Bool.not_eq_true'] at hrc
    rcases hrc with (hc0 | ⟨i, hi, -, hic⟩) | hrow
    · subst hc0
      rw [oneAt_inner hone'] at hr
      cases hr
      simp [xorSum]
    · have := rowIs_inner (hins' i (by omega))
      rw [hic, hr] at this
      cases this
      simp [xorSum]
    · rw [getD_of_getElem? hr] at hrow
      show z (c + base) = _
      rw [Nat.add_comm, hf.row _ _ hrow]
      simp only [shiftRow, xorSum_map]
      rfl
  have hS := hsound u hty _ hsat (by rw [Nat.add_comm]; exact hcb1) F hF
  -- its inputs carry the call's input forms
  have hx : ∀ i < u.inBits, z (sub.inCols.getD i 0 + base) = formVal (vals.getD · false) (args.getD i default) := by
    intro i hi
    have hbi := hbind i (by omega)
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
      exact hinv.formOf _ (by rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (by omega)]; exact hargs i (by omega))
  rw [getD_map_col, list_getD_map_range]
  by_cases ho : o < sub.outCols.size
  · rw [ite_eq_left ho, ite_eq_left (by omega), cfVal_col, Nat.add_comm, hS o (by omega)]
    exact congrArg (fun l => List.getD l o false) (evalT_congr types words F u hx)
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
    {done : Array D} {unit j : Nat} {k : Chk} {z : ℕ → Bool} (hf : Facts k z)
    (hcallee : ∀ jj < j, jj ≠ unit → checkLayout types ls done unit jj = true ∧ LayoutSound types words ls done jj) :
    ∀ jl top t l items c wire refs wire' refs',
      walk types ls done unit j k jl top t l items c wire refs = some (wire', refs') →
      ∀ F, jl ≤ F → (top = true → jl = j) → ∀ vals, Inv z wire vals →
        Inv z wire' (runT types words (fun u y => evalT types words F u y) t vals items) := by
  intro jl
  induction jl using Nat.strongRecOn with
  | ind jl ihj =>
  intro top t l items
  induction items with
  | nil =>
    intro c wire refs wire' refs' h F _ _ vals hinv
    unfold walk at h
    cases h
    exact hinv
  | cons it rest ih =>
    intro c wire refs wire' refs' h F hF htop vals hinv
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
          refine ih _ _ _ _ _ h F hF htop _ (hinv.push _ _ ?_)
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
                have hin := ihj _ hlt false u _ _ 0 _ _ w'' refs'' hw F' (by omega) (by simp) _
                  (inv_args hinv hsz hargs)
                simp only [runT, itemOut, hu]
                refine ih _ _ _ _ _ h (F' + 1) hF htop _ (hinv.append (by simp [hos]) fun o => ?_)
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
              refine ih _ _ _ _ _ h F hF htop _
                (hinv.append (by rw [Array.size_map, List.length_map, List.length_range]; exact hout) fun o => ?_)
              exact placed_sound hf hinv hchk hjju hty' hsound hsz hargs hout hp F (by have := htop htop'; omega) o
            · cases h
          · cases h
    | read _ _ => unfold walk at h; cases h
    | bad _ => unfold walk at h; cases h

/-! ## Every layout -/

/-- A checked layout, on column values where its row tests hold: each output column carries the type's output on its
input columns. -/
theorem checked_sound {types : String → Option CircuitType} {words : String → Option (Array Nat)} {ls : Array Layout}
    {done : Array D} {unit jj : Nat} (hcheck : checkLayout types ls done unit jj = true) {t : CircuitType}
    (ht : (ls.getD jj default).type.bind types = some t) {z : ℕ → Bool}
    (hf : Facts ⟨done.getD jj default, jj == unit⟩ z)
    (hcallee : ∀ jj' < jj, jj' ≠ unit → checkLayout types ls done unit jj' = true ∧ LayoutSound types words ls done jj')
    (F : Nat) (hF : jj < F) (o : Nat) (ho : o < t.outputs.size) :
    z ((done.getD jj default).outCols.getD o 0) =
      (evalT types words F t fun i => z ((done.getD jj default).inCols.getD i 0)).getD o false := by
  obtain ⟨t', ht', -, hoc, -, -, -, -, wire, hw, hrows⟩ := checkLayout_spec hcheck
  rw [ht] at ht'
  cases ht'
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
  have hinv := walk_sound hf hcallee jj true t _ _ 0 _ _ wire [] hw F' (by omega) (fun _ => rfl) _ hinv0
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
    {done : Array D} {unit : Nat} (hall : ∀ jj < ls.size, checkLayout types ls done unit jj = true) :
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
    {done : Array D} {unit : Nat} (hall : ∀ jj < ls.size, checkLayout types ls done unit jj = true)
    (hu : unit < ls.size) {t : CircuitType} (ht : (ls.getD unit default).type.bind types = some t) (one : Nat)
    {z : ℕ → Bool}
    (hz : SatAt (blockRow (done.getD unit default) one) z)
    (h1 : z one = true) (F : Nat) (hF : unit < F) (o : Nat) (ho : o < t.outputs.size) :
    z ((done.getD unit default).outCols.getD o 0) =
      (evalT types words F t fun i => z ((done.getD unit default).inCols.getD i 0)).getD o false := by
  obtain ⟨-, -, -, -, -, -, hone, -⟩ := checkLayout_spec (hall unit hu)
  have hk : (unit == unit) = true := by simp
  rw [hk] at hone
  exact checked_sound (hall unit hu) ht (by rw [hk]; exact facts_unit hz h1 hone)
    (fun jj hlt hne => ⟨hall jj (by omega), layouts_sound_of hall jj (by omega) hne⟩) F hF o ho

theorem deriveChecked_all {types : String → Option CircuitType} {ls : Array Layout} {info : String → Option TableInfo}
    {words : String → Option (Array Nat)} {unit : Nat} {done : Array D}
    (h : deriveChecked types ls info words unit = .ok done) : ∀ jj < ls.size, checkLayout types ls done unit jj = true := by
  unfold deriveChecked at h
  cases hd : deriveAll types ls info words unit with
  | error e => simp [hd, bind, Except.bind] at h
  | ok done' =>
    simp only [hd, bind, Except.bind] at h
    split at h
    · cases h
    · rename_i hf
      cases h
      intro jj hjj
      have := List.find?_eq_none.mp hf jj (List.mem_range.mpr hjj)
      simpa using this


/-! ## The statements -/

theorem lt_of_type {ls : Array Layout} {types : String → Option CircuitType} {j : ℕ} {t : CircuitType}
    (ht : (ls.getD j default).type.bind types = some t) : j < ls.size := by
  by_contra hj
  have : ls.getD j default = default := by simp [Array.getD, hj]
  rw [this] at ht
  cases ht

/-- **The unit's derived rows compute its type** (L1 over the type DAG). If `deriveChecked` accepts the layouts, column
values that satisfy the unit's rows as the block holds them (Δ applied), with the block's constant `one` at 1, carry on
each output column the unit type's output on its input columns. -/
theorem unit_sound {types : String → Option CircuitType} {ls : Array Layout} {info : String → Option TableInfo}
    {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D}
    (hd : deriveChecked types ls info words unit = .ok done) {t : CircuitType}
    (ht : (ls.getD unit default).type.bind types = some t) (one : ℕ) (z : ℕ → Bool)
    (hz : SatAt (blockRow (done.getD unit default) one) z) (h1 : z one = true) (F : ℕ) (hF : unit < F) (o : ℕ)
    (ho : o < t.outputs.size) :
    z ((done.getD unit default).outCols.getD o 0) =
      (evalT types words F t fun i => z ((done.getD unit default).inCols.getD i 0)).getD o false :=
  unit_sound_of (deriveChecked_all hd) (lt_of_type ht) ht one hz h1 F hF o ho

/-- **Every other layout's derived rows compute its type.** If `deriveChecked` accepts the layouts, column values that
satisfy layout `j`'s rows (a placed or inlined callee's, in its own coordinates), with its constant at 1, carry on each
output column its type's output on its input columns. -/
theorem layout_sound {types : String → Option CircuitType} {ls : Array Layout} {info : String → Option TableInfo}
    {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D}
    (hd : deriveChecked types ls info words unit = .ok done) {j : ℕ} (hj : j ≠ unit) {t : CircuitType}
    (ht : (ls.getD j default).type.bind types = some t) (z : ℕ → Bool) (hz : Sat (done.getD j default).rows.toList z)
    (h1 : z (done.getD j default).const = true) (F : ℕ) (hF : j < F) (o : ℕ) (ho : o < t.outputs.size) :
    z ((done.getD j default).outCols.getD o 0) =
      (evalT types words F t fun i => z ((done.getD j default).inCols.getD i 0)).getD o false :=
  layouts_sound_of (deriveChecked_all hd) j (lt_of_type ht) hj t ht z hz h1 F hF o ho

end FlockSoundness.Types.Dag
