import Proofs.Flock.Soundness.Types.Dag
import Proofs.Flock.Recursive.UniversalClass

/-!
# Recursion: the universal unit's lowering computes its class's interpretation

`UniversalUnit_v1{G, N_IN, N_OUT}` is `universal.build` over GF(2) forms: gate `g` gathers its two sources from the
`N_IN + g` wires before it (`gather.gather`, a multiplexer tree padded with `0`), computes its op on them, and the outputs
gather from all `W` wires. This file builds that circuit in Lean, generically in the parameters, and proves what it
computes.

* **The gather lemma, once** (`gatherB_eq`, `gatherB_lt`, `gatherB_ge`): the multiplexer tree on values, `gatherB idx cur`,
  is `cur[idx]` when the index is below the words' count and `0` at or past it, whenever the words fit the index
  (`cur.length ≤ 2^|idx|`). `getD_levelB` is one level: word `i` of the level is word `2i + s` before it, `0` past the end.
* **The builder** (`uuType p`): `build`'s ANDs as a flat `Flock.CircuitType`, in `build`'s creation order, over inputs
  `(op, src, out, x)` in `UniversalClass.layout`'s order (`2G`, `2G·K`, `N_OUT·K`, `N_IN` bits). Its forms are the
  ascending wire lists `forms` makes, all with constant `0` (the fill is `0`, and nothing negates a form).
* **What it computes** (`eval_uuType`): on any input bits, the type's outputs are `Desc.eval` of the description those bits
  hold on the data they hold, for every `p`. Each source and each output is the gather lemma at `cur = Desc.wiresTo`.
* **Flat at every width** (`flatOk_uuType`): the type passes `flatOk`, `derive`'s first check, for every `p`: each AND
  reads ascending wires below its own, and each output ascending wires below the last.
* **Its rows** (`uu_sound`, `uu_rows_iff`): whenever `derive` lays the type out (the verifier's own check; `accepted`
  runs it, with the format's rules, at the classes the digests pin), column values
  that satisfy the rows with the constant at 1 carry `Desc.eval` of their description columns on their data columns; and
  outputs `y` are carried by some satisfying assignment with description `d` and data `x` exactly when `y = d.eval x`.
* **On a typed statement's unit rows** (`uu_unit_sound`): when `deriveChecked` accepts the layouts and the unit's type is
  `uuType p`, the unit's rows as the block holds them carry `Desc.eval` the same way (`Dag.unit_sound`, with
  `Dag.evalT_flat` at `flatOk_uuType`).
* **The tie to Python** (the `#guard`s at the end): the type's digest is the one `circuit_types.from_gf2` gives for
  `universal.build` over the same four input ports, at six classes up to `{16, 16, 8}`; `test_circuit_types.py` reads
  these pins and recomputes them in Python. Past them it is one construction: both builders are the same recursion in
  the parameters, Python's `forms` folds no AND of the unit for any parameters (an index bit never occurs in a form, and
  every gathered value and gate output holds a fresh AND wire), and `from_gf2` keeps every AND (each wire feeds a later
  gather or an output's), so both make `universal.and_count` ANDs, in one order, at every width.

The staged statement's unit type is not this one byte for byte: `class_statement` lowers the traced Definition's body, its
gates in level order and its inputs one port a leaf. It is the same circuit up to renaming wires, and `uu_unit_sound`
reaches it once it is staged as `uuType p`.
-/

namespace FlockSoundness.Discharge.Recursion.UniversalLower

open Flock Flock.Derive Audit FlockSoundness.Types FlockSoundness.Discharge.Recursion.UniversalClass

/-! ## The multiplexer tree on values -/

/-- `forms.mux(s, hi, lo)`'s value: `lo xor (s and (hi xor lo))`. -/
def muxB (s hi lo : Bool) : Bool := xor lo (s && xor hi lo)

theorem muxB_eq (s hi lo : Bool) : muxB s hi lo = if s then hi else lo := by
  cases s <;> cases hi <;> cases lo <;> rfl

/-- One level of `gather.gather` on values: words `2i` and `2i + 1` meet in a mux on index bit `s`, and an odd last word
meets the fill `0`. -/
def levelB (s : Bool) : List Bool → List Bool
  | [] => []
  | [a] => [muxB s false a]
  | a :: b :: rest => muxB s b a :: levelB s rest

/-- `gather.gather` on values, the index least significant bit first. -/
def gatherB : List Bool → List Bool → Bool
  | [], cur => cur.headD false
  | s :: idx, cur => gatherB idx (levelB s cur)

/-- An index's value, least significant bit first. -/
def ofBitsL : List Bool → ℕ
  | [] => 0
  | s :: idx => s.toNat + 2 * ofBitsL idx

theorem ofBitsL_ofFn : ∀ (k : ℕ) (b : Fin k → Bool), ofBitsL (List.ofFn b) = ofBits k b
  | 0, _ => rfl
  | k + 1, b => by rw [List.ofFn_succ, ofBitsL, ofBitsL_ofFn k, ofBits]

/-- **One level**: word `i` after the level is word `2i + s` before it, and `0` past the end. -/
theorem getD_levelB (s : Bool) : ∀ (cur : List Bool) (i : ℕ),
    (levelB s cur).getD i false = cur.getD (2 * i + s.toNat) false
  | [], i => by simp [levelB]
  | [a], 0 => by cases s <;> simp [levelB, muxB_eq]
  | [a], i + 1 => by
    rw [show 2 * (i + 1) + s.toNat = (2 * i + 1 + s.toNat) + 1 by omega]
    simp [levelB]
  | a :: b :: rest, 0 => by cases s <;> simp [levelB, muxB_eq]
  | a :: b :: rest, i + 1 => by
    rw [levelB, List.getD_cons_succ, getD_levelB s rest i,
      show 2 * (i + 1) + s.toNat = (2 * i + s.toNat) + 1 + 1 by omega, List.getD_cons_succ, List.getD_cons_succ]

theorem length_levelB (s : Bool) : ∀ cur : List Bool, (levelB s cur).length = (cur.length + 1) / 2
  | [] => rfl
  | [_] => by simp [levelB]
  | a :: b :: rest => by
    simp only [levelB, List.length_cons, length_levelB s rest]
    omega

/-- **The gather lemma.** A multiplexer tree whose words fit its index selects the word at the index, and `0` when the
index is at or past the words' count. -/
theorem gatherB_eq : ∀ (idx cur : List Bool), cur.length ≤ 2 ^ idx.length →
    gatherB idx cur = cur.getD (ofBitsL idx) false
  | [], [], _ => rfl
  | [], [_], _ => rfl
  | [], _ :: _ :: _, h => by simp at h
  | s :: idx, cur, h => by
    have h' : (levelB s cur).length ≤ 2 ^ idx.length := by
      rw [length_levelB]
      rw [List.length_cons, pow_succ] at h
      omega
    rw [gatherB, gatherB_eq idx _ h', getD_levelB, ofBitsL]
    congr 1
    omega

/-- An index below the words' count selects that word. -/
theorem gatherB_lt {idx cur : List Bool} (h : cur.length ≤ 2 ^ idx.length) (hi : ofBitsL idx < cur.length) :
    gatherB idx cur = cur[ofBitsL idx] := by
  rw [gatherB_eq idx cur h, List.getD_eq_getElem _ _ hi]

/-- An index at or past the words' count gives `0`. -/
theorem gatherB_ge {idx cur : List Bool} (h : cur.length ≤ 2 ^ idx.length) (hi : cur.length ≤ ofBitsL idx) :
    gatherB idx cur = false := by
  rw [gatherB_eq idx cur h, List.getD_eq_default _ _ hi]

/-! ## Linear forms with constant 0 -/

/-- The XOR of two linear forms with constant `0`, as ascending wire lists: their symmetric difference, by merging. -/
def symm : List ℕ → List ℕ → List ℕ
  | [], m => m
  | a :: l, [] => a :: l
  | a :: l, b :: m =>
    if a < b then a :: symm l (b :: m) else if b < a then b :: symm (a :: l) m else symm l m
termination_by l m => l.length + m.length

theorem xorSum_nil (v : ℕ → Bool) : xorSum ([] : List ℕ) v = false := rfl

theorem xorSum_cons (a : ℕ) (l : List ℕ) (v : ℕ → Bool) : xorSum (a :: l) v = xor (v a) (xorSum l v) := rfl

/-- `symm` is the XOR of the two forms' values, on any lists. -/
theorem xorSum_symm (v : ℕ → Bool) (l m : List ℕ) : xorSum (symm l m) v = xor (xorSum l v) (xorSum m v) := by
  fun_induction symm l m with
  | case1 m => simp [xorSum_nil]
  | case2 a l => simp [xorSum_nil]
  | case3 a l b m _ ih =>
    rw [xorSum_cons, ih, xorSum_cons, xorSum_cons]
    cases v a <;> cases v b <;> cases xorSum l v <;> cases xorSum m v <;> rfl
  | case4 a l b m _ _ ih =>
    rw [xorSum_cons, ih, xorSum_cons, xorSum_cons]
    cases v a <;> cases v b <;> cases xorSum l v <;> cases xorSum m v <;> rfl
  | case5 a l b m h1 h2 ih =>
    obtain rfl : a = b := by omega
    rw [ih, xorSum_cons, xorSum_cons]
    cases v a <;> cases xorSum l v <;> cases xorSum m v <;> rfl

theorem mem_symm {x : ℕ} {l m : List ℕ} (h : x ∈ symm l m) : x ∈ l ∨ x ∈ m := by
  fun_induction symm l m with
  | case1 m => exact Or.inr h
  | case2 a l => exact Or.inl h
  | case3 a l b m _ ih =>
    rcases List.mem_cons.1 h with rfl | h
    · exact Or.inl (List.mem_cons_self ..)
    · rcases ih h with h | h
      · exact Or.inl (List.mem_cons_of_mem _ h)
      · exact Or.inr h
  | case4 a l b m _ _ ih =>
    rcases List.mem_cons.1 h with rfl | h
    · exact Or.inr (List.mem_cons_self ..)
    · rcases ih h with h | h
      · exact Or.inl h
      · exact Or.inr (List.mem_cons_of_mem _ h)
  | case5 a l b m _ _ ih =>
    rcases ih h with h | h
    · exact Or.inl (List.mem_cons_of_mem _ h)
    · exact Or.inr (List.mem_cons_of_mem _ h)

/-- A wire list as a `Form`, constant `0`. -/
def form (l : List ℕ) : Form := ⟨l.toArray, 0⟩

theorem formVal_form (v : ℕ → Bool) (l : List ℕ) : formVal v (form l) = xorSum l v := by
  simp [formVal, form]

/-! ## The builder

Each step takes the items so far and returns a form with the items after it, in `build`'s creation order. `n` is the
type's input count: wire `n + k` is item `k`'s AND. -/

/-- `C.AND(a, b)`: one AND item; its wire is the next one. -/
def andL (n : ℕ) (its : Array Item) (a b : List ℕ) : List ℕ × Array Item :=
  ([n + its.size], its.push (.and (form a) (form b)))

/-- `forms.mux(C, s, hi, lo)`: `lo ^ C.AND(s, hi ^ lo)`. -/
def muxL (n : ℕ) (its : Array Item) (s hi lo : List ℕ) : List ℕ × Array Item :=
  let r := andL n its s (symm hi lo)
  (symm lo r.1, r.2)

/-- One level of `gather.gather` on index bit `s`: words `2i` and `2i + 1` meet in a mux, an odd last word meets the fill
`0`, left to right. -/
def levelL (n : ℕ) (s : List ℕ) : List (List ℕ) → Array Item → List (List ℕ) × Array Item
  | [], its => ([], its)
  | [a], its => let r := muxL n its s [] a; ([r.1], r.2)
  | a :: b :: rest, its =>
    let r := muxL n its s b a
    let q := levelL n s rest r.2
    (r.1 :: q.1, q.2)

/-- `gather.gather(C, idx, words, 0)` on one-bit words, the index least significant bit first. -/
def gatherL (n : ℕ) : List (List ℕ) → List (List ℕ) → Array Item → List ℕ × Array Item
  | [], cur, its => (cur.headD [], its)
  | s :: idx, cur, its => let q := levelL n s cur its; gatherL n idx q.1 q.2

/-- A gate: `s = a ^ b`, then `s ^ C.AND(t, C.AND(a, b) ^ s) ^ nn`. -/
def gateL (n : ℕ) (its : Array Item) (t nn a b : List ℕ) : List ℕ × Array Item :=
  let ab := andL n its a b
  let s := symm a b
  let r := andL n ab.2 t (symm ab.1 s)
  (symm (symm s r.1) nn, r.2)

/-- `[gather(C, idx, words, 0)[0] for idx in idxs]`, in order. -/
def outsL (n : ℕ) (ws : List (List ℕ)) : List (List (List ℕ)) → Array Item → List (List ℕ) × Array Item
  | [], its => ([], its)
  | idx :: rest, its =>
    let r := gatherL n idx ws its
    let q := outsL n ws rest r.2
    (r.1 :: q.1, q.2)

section Unit

variable (p : Params)

/-- The type's input count: the description's bits, then `x`. -/
abbrev nIns : ℕ := p.width + p.nIn

/-- Gate `g`'s op bit `t`, input wire `2g + t`. -/
def opW (g t : ℕ) : List ℕ := [2 * g + t]

/-- Gate `g`'s source `s`, its `K` input wires least significant first. -/
def srcIdx (g s : ℕ) : List (List ℕ) := List.ofFn fun j : Fin p.k => [2 * p.g + (2 * g + s) * p.k + j.val]

/-- Output `i`'s index, its `K` input wires least significant first. -/
def outIdx (i : ℕ) : List (List ℕ) := List.ofFn fun j : Fin p.k => [2 * p.g + 2 * p.g * p.k + i * p.k + j.val]

/-- The data `x`: input wire `width + i`. -/
def xW : List (List ℕ) := List.ofFn fun i : Fin p.nIn => [p.width + i.val]

/-- `build`'s loop: the wire list (the inputs `x`, then one form a gate) and the items after gates `0 … g−1`. -/
def gatesL : ℕ → List (List ℕ) × Array Item
  | 0 => (xW p, #[])
  | g + 1 =>
    let q := gatesL g
    if g < p.g then
      let a := gatherL (nIns p) (srcIdx p g 0) q.1 q.2
      let b := gatherL (nIns p) (srcIdx p g 1) q.1 a.2
      let r := gateL (nIns p) b.2 (opW g 0) (opW g 1) a.1 b.1
      (q.1 ++ [r.1], r.2)
    else q

/-- The outputs' gathers over all `W` wires, after the gates. -/
def outsAll : List (List ℕ) × Array Item :=
  outsL (nIns p) (gatesL p p.g).1 (List.ofFn fun i : Fin p.nOut => outIdx p i) (gatesL p p.g).2

/-- **`UniversalUnit_v1{G, N_IN, N_OUT}` as a flat circuit type**: `circuit_types.from_gf2` of `universal.build` over the
input ports `(op, src, out, x)`, `2G`, `2G·K`, `N_OUT·K` and `N_IN` bits, one output port of `N_OUT` bits. -/
def uuType : CircuitType where
  label := PROGRAM
  ins := #[(1, 2 * p.g), (1, 2 * p.g * p.k), (1, p.nOut * p.k), (1, p.nIn)]
  outs := #[(1, p.nOut)]
  calls := #[]
  tables := #[]
  items := (outsAll p).2
  outputs := ((outsAll p).1.map form).toArray

theorem uuType_inBits : (uuType p).inBits = (nIns p) := by
  simp only [CircuitType.inBits, CircuitType.portBits, uuType, nIns, Params.width, List.foldl_toArray',
    List.foldl_cons, List.foldl_nil]
  ring

end Unit

/-! ## What the builder computes

`V its` is the wire values once items `its` have run on inputs `x`. A form is good at `its` when its wires are all below
`n + its.size`; a good form keeps its value as items are appended. -/

theorem run_append (v : ℕ → Bool) (w : ℕ) (l m : List Item) :
    run v w (l ++ m) = run (run v w l) (w + l.length) m := by
  induction l generalizing v w with
  | nil => rfl
  | cons it l ih =>
    simp only [List.cons_append, run, ih, List.length_cons]
    rw [Nat.add_assoc, Nat.add_comm 1]

section Builder

variable (x : ℕ → Bool) (n : ℕ)

/-- The wire values once `its` have run. -/
def V (its : Array Item) : ℕ → Bool := run x n its.toList

/-- A form's value once `its` have run. -/
def lv (its : Array Item) (l : List ℕ) : Bool := xorSum l (V x n its)

/-- Every wire of the form is below `n + its.size`. -/
def Good (its : Array Item) (l : List ℕ) : Prop := ∀ j ∈ l, j < n + its.size

/-- `its'` appends items to `its`. -/
def Ext (its its' : Array Item) : Prop := ∃ m, its'.toList = its.toList ++ m

variable {x n}

theorem Ext.refl (its : Array Item) : Ext its its := ⟨[], by simp⟩

theorem Ext.trans {a b c : Array Item} (h1 : Ext a b) (h2 : Ext b c) : Ext a c := by
  obtain ⟨m1, h1⟩ := h1
  obtain ⟨m2, h2⟩ := h2
  exact ⟨m1 ++ m2, by rw [h2, h1, List.append_assoc]⟩

theorem Ext.size_le {a b : Array Item} (h : Ext a b) : a.size ≤ b.size := by
  obtain ⟨m, h⟩ := h
  have := congrArg List.length h
  simp only [Array.length_toList, List.length_append] at this
  omega

theorem Ext.push (its : Array Item) (it : Item) : Ext its (its.push it) := ⟨[it], by simp⟩

theorem Good.ext {its its' : Array Item} {l : List ℕ} (h : Ext its its') (hl : Good n its l) : Good n its' l :=
  fun j hj => by have := hl j hj; have := h.size_le; omega

theorem V_ext {its its' : Array Item} (h : Ext its its') {j : ℕ} (hj : j < n + its.size) :
    V x n its' j = V x n its j := by
  obtain ⟨m, hm⟩ := h
  unfold V
  rw [hm, run_append]
  exact run_lt m _ _ j (by simpa using hj)

theorem lv_ext {its its' : Array Item} (h : Ext its its') {l : List ℕ} (hl : Good n its l) :
    lv x n its' l = lv x n its l :=
  xorSum_congr fun j hj => V_ext h (hl j hj)

theorem lv_symm (its : Array Item) (l m : List ℕ) : lv x n its (symm l m) = xor (lv x n its l) (lv x n its m) :=
  xorSum_symm _ l m

theorem Good.symm {its : Array Item} {l m : List ℕ} (hl : Good n its l) (hm : Good n its m) :
    Good n its (UniversalLower.symm l m) :=
  fun j hj => (mem_symm hj).elim (hl j) (hm j)

theorem good_nil (its : Array Item) : Good n its [] := fun _ h => absurd h List.not_mem_nil

/-- An input wire: its value is the input's. -/
theorem lv_input (its : Array Item) {i : ℕ} (hi : i < n) : lv x n its [i] = x i := by
  simp only [lv, xorSum_cons, xorSum_nil, Bool.xor_false, V]
  exact run_lt _ _ _ _ (by omega)

theorem good_input (its : Array Item) {i : ℕ} (hi : i < n) : Good n its [i] := by
  intro j hj; rw [List.mem_singleton] at hj; omega

/-- **An AND**: its wire holds the AND of its two forms' values. -/
theorem andL_spec (its : Array Item) (a b : List ℕ) :
    Ext its (andL n its a b).2 ∧ Good n (andL n its a b).2 (andL n its a b).1 ∧
      lv x n (andL n its a b).2 (andL n its a b).1 = (lv x n its a && lv x n its b) := by
  refine ⟨Ext.push _ _, fun j hj => ?_, ?_⟩
  · simp only [andL, List.mem_singleton] at hj; simp [andL, hj]
  · simp only [andL, lv, xorSum_cons, xorSum_nil, Bool.xor_false, V, Array.toList_push, run_append, run,
      Array.length_toList, ↓reduceIte, itemVal, formVal_form]

/-- **A mux**: `muxB` of its index bit's and its two words' values. -/
theorem muxL_spec (its : Array Item) {s hi lo : List ℕ} (hl : Good n its lo) :
    Ext its (muxL n its s hi lo).2 ∧ Good n (muxL n its s hi lo).2 (muxL n its s hi lo).1 ∧
      lv x n (muxL n its s hi lo).2 (muxL n its s hi lo).1 = muxB (lv x n its s) (lv x n its hi) (lv x n its lo) := by
  obtain ⟨e, g, v⟩ := andL_spec (x := x) its s (symm hi lo)
  refine ⟨e, (hl.ext e).symm g, ?_⟩
  simp only [muxL]
  rw [lv_symm, v, lv_ext e hl, lv_symm, muxB]

/-- Every form of the list is good. -/
def Goods (n : ℕ) (its : Array Item) (L : List (List ℕ)) : Prop := ∀ l ∈ L, Good n its l

theorem Goods.ext {its its' : Array Item} {L : List (List ℕ)} (h : Ext its its') (hL : Goods n its L) :
    Goods n its' L := fun l hl => (hL l hl).ext h

theorem map_lv_ext {its its' : Array Item} (h : Ext its its') {L : List (List ℕ)} (hL : Goods n its L) :
    L.map (lv x n its') = L.map (lv x n its) :=
  List.map_congr_left fun l hl => lv_ext h (hL l hl)

/-- **A level**, on values: `levelB` of the index bit's value on the words' values. -/
theorem levelL_spec {s : List ℕ} : ∀ (cur : List (List ℕ)) (its : Array Item), Good n its s → Goods n its cur →
    Ext its (levelL n s cur its).2 ∧ Goods n (levelL n s cur its).2 (levelL n s cur its).1 ∧
      (levelL n s cur its).1.map (lv x n (levelL n s cur its).2) = levelB (lv x n its s) (cur.map (lv x n its))
  | [], its, _, _ => ⟨Ext.refl _, fun _ h => absurd h List.not_mem_nil, rfl⟩
  | [a], its, hs, hc => by
    obtain ⟨e, g, v⟩ := muxL_spec (x := x) (s := s) (hi := []) its (hc a (List.mem_singleton_self _))
    refine ⟨e, fun l hl => by rw [List.mem_singleton.1 hl]; exact g, ?_⟩
    simp only [levelL, List.map_cons, List.map_nil, levelB, v]
    rfl
  | a :: b :: rest, its, hs, hc => by
    have ha := hc a (by simp)
    have hb := hc b (by simp)
    have hr : Goods n its rest := fun l hl => hc l (by simp [hl])
    obtain ⟨e, g, v⟩ := muxL_spec (x := x) (s := s) (hi := b) its ha
    obtain ⟨e', g', v'⟩ := levelL_spec rest _ (hs.ext e) (hr.ext e)
    refine ⟨e.trans e', ?_, ?_⟩
    · intro l hl
      rcases List.mem_cons.1 hl with rfl | hl
      · exact g.ext e'
      · exact g' l hl
    · simp only [levelL, List.map_cons, levelB]
      rw [lv_ext e' g, v, v', lv_ext e hs, map_lv_ext e hr]

/-- **A gather**, on values: `gatherB` of the index's values on the words' values. -/
theorem gatherL_spec : ∀ (idx cur : List (List ℕ)) (its : Array Item), Goods n its idx → Goods n its cur →
    Ext its (gatherL n idx cur its).2 ∧ Good n (gatherL n idx cur its).2 (gatherL n idx cur its).1 ∧
      lv x n (gatherL n idx cur its).2 (gatherL n idx cur its).1 = gatherB (idx.map (lv x n its)) (cur.map (lv x n its))
  | [], cur, its, _, hc => by
    refine ⟨Ext.refl _, ?_, ?_⟩
    · cases cur with
      | nil => exact good_nil its
      | cons a _ => exact hc a (List.mem_cons_self ..)
    · cases cur <;> rfl
  | s :: idx, cur, its, hi, hc => by
    have hs := hi s (List.mem_cons_self ..)
    have hi' : Goods n its idx := fun l hl => hi l (List.mem_cons_of_mem _ hl)
    obtain ⟨e, g, v⟩ := levelL_spec (x := x) cur its hs hc
    obtain ⟨e', g', v'⟩ := gatherL_spec idx _ _ (hi'.ext e) g
    refine ⟨e.trans e', g', ?_⟩
    simp only [gatherL, List.map_cons, gatherB]
    rw [v', v, map_lv_ext e hi']

/-- **A gate**: `gateFn` of its op bits' and its sources' values. -/
theorem gateL_spec (its : Array Item) {t nn a b : List ℕ} (ht : Good n its t) (hn : Good n its nn)
    (ha : Good n its a) (hb : Good n its b) :
    Ext its (gateL n its t nn a b).2 ∧ Good n (gateL n its t nn a b).2 (gateL n its t nn a b).1 ∧
      lv x n (gateL n its t nn a b).2 (gateL n its t nn a b).1 =
        gateFn (lv x n its t) (lv x n its nn) (lv x n its a) (lv x n its b) := by
  obtain ⟨e1, g1, v1⟩ := andL_spec (x := x) its a b
  have hs : Good n its (symm a b) := ha.symm hb
  obtain ⟨e2, g2, v2⟩ := andL_spec (x := x) (andL n its a b).2 t (symm (andL n its a b).1 (symm a b))
  refine ⟨e1.trans e2, ((hs.ext (e1.trans e2)).symm g2).symm (hn.ext (e1.trans e2)), ?_⟩
  simp only [gateL]
  rw [lv_symm, lv_symm, v2, lv_ext (e1.trans e2) hs, lv_ext (e1.trans e2) hn,
    lv_symm (andL n its a b).2 (andL n its a b).1 (symm a b), v1, lv_ext e1 ht, lv_ext e1 hs, lv_symm its a b]
  unfold gateFn
  cases lv x n its t <;> cases lv x n its nn <;> cases lv x n its a <;> cases lv x n its b <;> rfl

/-- **The outputs' gathers**, on values. -/
theorem outsL_spec (ws : List (List ℕ)) : ∀ (idxs : List (List (List ℕ))) (its : Array Item), Goods n its ws →
    (∀ idx ∈ idxs, Goods n its idx) →
    Ext its (outsL n ws idxs its).2 ∧ Goods n (outsL n ws idxs its).2 (outsL n ws idxs its).1 ∧
      (outsL n ws idxs its).1.map (lv x n (outsL n ws idxs its).2) =
        idxs.map fun idx => gatherB (idx.map (lv x n its)) (ws.map (lv x n its))
  | [], its, _, _ => ⟨Ext.refl _, fun _ h => absurd h List.not_mem_nil, rfl⟩
  | idx :: rest, its, hw, hi => by
    obtain ⟨e, g, v⟩ := gatherL_spec (x := x) idx ws its (hi idx (List.mem_cons_self ..)) hw
    obtain ⟨e', g', v'⟩ := outsL_spec ws rest _ (hw.ext e)
      (fun l hl => (hi l (List.mem_cons_of_mem _ hl)).ext e)
    refine ⟨e.trans e', ?_, ?_⟩
    · intro l hl
      rcases List.mem_cons.1 hl with rfl | hl
      · exact g.ext e'
      · exact g' l hl
    · simp only [outsL, List.map_cons]
      rw [lv_ext e' g, v, v', map_lv_ext e hw]
      congr 1
      refine List.map_congr_left fun l hl => ?_
      rw [map_lv_ext e (hi l (List.mem_cons_of_mem _ hl))]

end Builder

/-! ## The unit computes `Desc.eval` -/

variable {p : Params}

/-- The description the input bits hold: input wire `j < width`. -/
def descOf (x : ℕ → Bool) : Desc p := fun j => x j.val

/-- The data the input bits hold: input wire `width + i`. -/
def dataOf (x : ℕ → Bool) : Fin p.nIn → Bool := fun i => x (p.width + i.val)

theorem srcIdx_val (x : ℕ → Bool) (its : Array Item) (g : Fin p.g) (s : Fin 2) :
    (srcIdx p g s).map (lv x (nIns p) its) = List.ofFn fun j => descOf x (layout p (.inr (.inl ((g, s), j)))) := by
  rw [srcIdx, List.map_ofFn]
  congr 1
  funext j
  simp only [Function.comp, descOf, layout_src]
  exact lv_input its (by
    have := (layout p (.inr (.inl ((g, s), j)))).isLt
    rw [layout_src] at this
    simp only [nIns]; omega)

theorem outIdx_val (x : ℕ → Bool) (its : Array Item) (i : Fin p.nOut) :
    (outIdx p i).map (lv x (nIns p) its) = List.ofFn fun j => descOf x (layout p (.inr (.inr (i, j)))) := by
  rw [outIdx, List.map_ofFn]
  congr 1
  funext j
  simp only [Function.comp, descOf, layout_out]
  exact lv_input its (by
    have := (layout p (.inr (.inr (i, j)))).isLt
    rw [layout_out] at this
    simp only [nIns]; omega)

theorem goods_srcIdx (its : Array Item) (g : Fin p.g) (s : Fin 2) : Goods (nIns p) its (srcIdx p g s) := by
  intro l hl
  obtain ⟨j, rfl⟩ := List.mem_ofFn.1 hl
  apply good_input
  have := (layout p (.inr (.inl ((g, s), j)))).isLt
  rw [layout_src] at this
  simp only [nIns]; omega

theorem goods_outIdx (its : Array Item) (i : Fin p.nOut) : Goods (nIns p) its (outIdx p i) := by
  intro l hl
  obtain ⟨j, rfl⟩ := List.mem_ofFn.1 hl
  apply good_input
  have := (layout p (.inr (.inr (i, j)))).isLt
  rw [layout_out] at this
  simp only [nIns]; omega

theorem opW_spec (x : ℕ → Bool) (its : Array Item) (g : Fin p.g) (t : Fin 2) :
    Good (nIns p) its (opW g t) ∧ lv x (nIns p) its (opW g t) = descOf x (layout p (.inl (g, t))) := by
  have := (layout p (.inl (g, t))).isLt
  rw [layout_op] at this
  refine ⟨good_input its (by simp only [nIns]; omega), ?_⟩
  rw [opW, lv_input its (by simp only [nIns]; omega), descOf, layout_op]

/-- **The gate loop**: after gates `0 … g−1`, the wire list's values are `Desc.wiresTo`'s. -/
theorem gatesL_spec (x : ℕ → Bool) : ∀ g, g ≤ p.g →
    Goods (nIns p) (gatesL p g).2 (gatesL p g).1 ∧
      (gatesL p g).1.map (lv x (nIns p) (gatesL p g).2) = (descOf (p := p) x).wiresTo (dataOf x) g
  | 0, _ => by
    refine ⟨fun l hl => ?_, ?_⟩
    · obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hl
      exact good_input _ (by simp only [nIns]; omega)
    · simp only [gatesL, xW, List.map_ofFn, Desc.wiresTo]
      congr 1
      funext i
      exact lv_input _ (by simp only [nIns]; omega)
  | g + 1, hg => by
    have hlt : g < p.g := hg
    obtain ⟨hq, vq⟩ := gatesL_spec x g (Nat.le_of_succ_le hg)
    set q := gatesL p g with hqdef
    have hlen : (q.1.map (lv x (nIns p) q.2)).length ≤ 2 ^ p.k := by
      rw [vq, Desc.length_wiresTo _ _ g (Nat.le_of_succ_le hg)]
      have := p.w_le_two_pow_k
      simp only [Params.w] at this; omega
    have hidx : ∀ s : Fin 2, (srcIdx p g s).length = p.k := fun s => by simp [srcIdx]
    obtain ⟨ea, ga, va⟩ := gatherL_spec (x := x) (srcIdx p g 0) q.1 q.2 (goods_srcIdx q.2 ⟨g, hlt⟩ 0) hq
    set a := gatherL (nIns p) (srcIdx p g 0) q.1 q.2 with hadef
    obtain ⟨eb, gb, vb⟩ := gatherL_spec (x := x) (srcIdx p g 1) q.1 a.2 (goods_srcIdx a.2 ⟨g, hlt⟩ 1) (hq.ext ea)
    set b := gatherL (nIns p) (srcIdx p g 1) q.1 a.2 with hbdef
    have ht : Good (nIns p) b.2 (opW g 0) := (opW_spec (p := p) x b.2 ⟨g, hlt⟩ 0).1
    have vt : lv x (nIns p) b.2 (opW g 0) = (descOf (p := p) x).opT ⟨g, hlt⟩ := (opW_spec x b.2 ⟨g, hlt⟩ 0).2
    have hn : Good (nIns p) b.2 (opW g 1) := (opW_spec (p := p) x b.2 ⟨g, hlt⟩ 1).1
    have vn : lv x (nIns p) b.2 (opW g 1) = (descOf (p := p) x).opN ⟨g, hlt⟩ := (opW_spec x b.2 ⟨g, hlt⟩ 1).2
    obtain ⟨er, gr, vr⟩ := gateL_spec (x := x) b.2 ht hn (ga.ext eb) gb
    set r := gateL (nIns p) b.2 (opW g 0) (opW g 1) a.1 b.1 with hrdef
    have hsrc : ∀ (s : Fin 2) (its : Array Item), Ext q.2 its →
        gatherB ((srcIdx p g s).map (lv x (nIns p) its)) (q.1.map (lv x (nIns p) its)) =
          ((descOf (p := p) x).wiresTo (dataOf x) g).getD ((descOf x).src ⟨g, hlt⟩ s) false := by
      intro s its e
      rw [map_lv_ext e hq, vq, gatherB_eq _ _ (by rw [List.length_map, hidx s, ← vq]; exact hlen),
        srcIdx_val x its ⟨g, hlt⟩ s, ofBitsL_ofFn]
      rfl
    have hstep : gatesL p (g + 1) = (q.1 ++ [r.1], r.2) := by
      rw [gatesL, ite_eq_left hlt]
    rw [hstep]
    refine ⟨?_, ?_⟩
    · intro l hl
      rcases List.mem_append.1 hl with hl | hl
      · exact (hq l hl).ext (ea.trans (eb.trans er))
      · rw [List.mem_singleton.1 hl]; exact gr
    · have h0 := hsrc 0 q.2 (Ext.refl _)
      have h1 := hsrc 1 a.2 ea
      simp only [Fin.val_zero, Fin.val_one] at h0 h1
      rw [Desc.wiresTo, dite_eq_left hlt, List.map_append, map_lv_ext (ea.trans (eb.trans er)) hq, vq,
        List.map_singleton, vr, vt, vn, lv_ext eb ga, va, vb, h0, h1]

theorem outsAll_spec (x : ℕ → Bool) :
    (outsAll p).1.map (lv x (nIns p) (outsAll p).2) = List.ofFn ((descOf (p := p) x).eval (dataOf x)) := by
  obtain ⟨hq, vq⟩ := gatesL_spec (p := p) x p.g le_rfl
  have hlen : ((gatesL p p.g).1.map (lv x (nIns p) (gatesL p p.g).2)).length ≤ 2 ^ p.k := by
    rw [vq, Desc.length_wiresTo _ _ p.g le_rfl]
    exact p.w_le_two_pow_k
  obtain ⟨-, -, v⟩ := outsL_spec (x := x) (gatesL p p.g).1 (List.ofFn fun i : Fin p.nOut => outIdx p i)
    (gatesL p p.g).2 hq (fun idx h => by obtain ⟨i, rfl⟩ := List.mem_ofFn.1 h; exact goods_outIdx _ i)
  rw [outsAll, v, List.map_ofFn]
  congr 1
  funext i
  simp only [Function.comp]
  rw [gatherB_eq _ _ (by simpa [outIdx] using hlen), outIdx_val, ofBitsL_ofFn, vq]
  rfl

theorem length_outsAll : (outsAll p).1.length = p.nOut := by
  have := congrArg List.length (outsAll_spec (p := p) (fun _ => false))
  simpa using this

/-- **The unit computes its class's interpretation**, for every `p`: on any input bits, the type's outputs are
`Desc.eval` of the description in its first `width` inputs on the data in its last `N_IN`. -/
theorem eval_uuType (x : ℕ → Bool) : eval (uuType p) x = List.ofFn ((descOf (p := p) x).eval (dataOf x)) := by
  rw [eval, uuType_inBits]
  simp only [uuType, List.map_map]
  rw [← outsAll_spec x]
  refine List.map_congr_left fun l _ => ?_
  simp only [Function.comp, formVal_form]
  rfl

theorem size_outputs : (uuType p).outputs.size = p.nOut := by
  simp [uuType, length_outsAll]

/-! ## Flat at every width

`flatOk`, the type's half of `derive`'s check, holds for every `p`: each AND reads ascending wires below its own, and each
output ascending wires below the last. -/

section Flat

variable {n : ℕ}

/-- A good form whose wires ascend. -/
def Ok (n : ℕ) (its : Array Item) (l : List ℕ) : Prop := Good n its l ∧ l.Pairwise (· < ·)

/-- Every item so far is an AND of two forms whose wires ascend and lie below its own. -/
def ItemsOk (n : ℕ) (its : Array Item) : Prop :=
  ∀ k (hk : k < its.size), ∃ a b, its[k] = .and (form a) (form b) ∧ a.Pairwise (· < ·) ∧ b.Pairwise (· < ·) ∧
    (∀ j ∈ a, j < n + k) ∧ (∀ j ∈ b, j < n + k)

theorem pairwise_symm {l m : List ℕ} (hl : l.Pairwise (· < ·)) (hm : m.Pairwise (· < ·)) :
    (symm l m).Pairwise (· < ·) := by
  fun_induction symm l m with
  | case1 m => exact hm
  | case2 a l => exact hl
  | case3 a l b m hab ih =>
    have hl' := List.pairwise_cons.1 hl
    refine List.pairwise_cons.2 ⟨fun y hy => ?_, ih hl'.2 hm⟩
    rcases mem_symm hy with hy | hy
    · exact hl'.1 y hy
    · rcases List.mem_cons.1 hy with rfl | hy
      · exact hab
      · exact lt_trans hab ((List.pairwise_cons.1 hm).1 y hy)
  | case4 a l b m _ hba ih =>
    have hm' := List.pairwise_cons.1 hm
    refine List.pairwise_cons.2 ⟨fun y hy => ?_, ih hl hm'.2⟩
    rcases mem_symm hy with hy | hy
    · rcases List.mem_cons.1 hy with rfl | hy
      · exact hba
      · exact lt_trans hba ((List.pairwise_cons.1 hl).1 y hy)
    · exact hm'.1 y hy
  | case5 a l b m _ _ ih => exact ih (List.pairwise_cons.1 hl).2 (List.pairwise_cons.1 hm).2

theorem formOk_form {l : List ℕ} (hs : l.Pairwise (· < ·)) (hl : ∀ j ∈ l, j < n) :
    CircuitType.formOk n (form l) = true := by
  simp only [CircuitType.formOk, form, zero_le, decide_true, List.size_toArray, List.all_toArray', Bool.true_and,
    List.getElem!_toArray, List.getElem!_eq_getElem?_getD, Nat.default_eq_zero, Bool.and_eq_true, List.all_eq_true,
    decide_eq_true_eq, List.mem_range]
  refine ⟨hl, fun i hi => ?_⟩
  rw [List.getElem?_eq_getElem (by omega), List.getElem?_eq_getElem (by omega)]
  exact List.pairwise_iff_getElem.1 hs i (i + 1) (by omega) (by omega) (by omega)

theorem Ok.ext {its its' : Array Item} {l : List ℕ} (h : Ext its its') (hl : Ok n its l) : Ok n its' l :=
  ⟨hl.1.ext h, hl.2⟩

theorem Ok.symm {its : Array Item} {l m : List ℕ} (hl : Ok n its l) (hm : Ok n its m) :
    Ok n its (UniversalLower.symm l m) :=
  ⟨hl.1.symm hm.1, pairwise_symm hl.2 hm.2⟩

theorem ok_nil (its : Array Item) : Ok n its [] := ⟨good_nil its, List.Pairwise.nil⟩

theorem andL_ok {its : Array Item} {a b : List ℕ} (hI : ItemsOk n its) (ha : Ok n its a) (hb : Ok n its b) :
    ItemsOk n (andL n its a b).2 ∧ Ok n (andL n its a b).2 (andL n its a b).1 := by
  refine ⟨fun k hk => ?_, (andL_spec (x := fun _ => false) (n := n) its a b).2.1, List.pairwise_singleton _ _⟩
  simp only [andL, Array.size_push] at hk ⊢
  rcases Nat.lt_succ_iff_lt_or_eq.1 hk with hk | rfl
  · rw [Array.getElem_push_lt hk]; exact hI k hk
  · rw [Array.getElem_push_eq]
    exact ⟨a, b, rfl, ha.2, hb.2, ha.1, hb.1⟩

theorem muxL_ok {its : Array Item} {s hi lo : List ℕ} (hI : ItemsOk n its) (hs : Ok n its s) (hh : Ok n its hi)
    (hl : Ok n its lo) : ItemsOk n (muxL n its s hi lo).2 ∧ Ok n (muxL n its s hi lo).2 (muxL n its s hi lo).1 := by
  obtain ⟨hI', hr⟩ := andL_ok hI hs (hh.symm hl)
  have e := (andL_spec (x := fun _ => false) (n := n) its s (symm hi lo)).1
  exact ⟨hI', (hl.ext e).symm hr⟩

theorem levelL_ok {s : List ℕ} : ∀ (cur : List (List ℕ)) (its : Array Item), ItemsOk n its → Ok n its s →
    (∀ l ∈ cur, Ok n its l) →
    ItemsOk n (levelL n s cur its).2 ∧ ∀ l ∈ (levelL n s cur its).1, Ok n (levelL n s cur its).2 l
  | [], its, hI, _, _ => ⟨hI, fun _ h => absurd h List.not_mem_nil⟩
  | [a], its, hI, hs, hc => by
    obtain ⟨hI', hr⟩ := muxL_ok hI hs (ok_nil its) (hc a (List.mem_singleton_self _))
    exact ⟨hI', fun l hl => by rw [List.mem_singleton.1 hl]; exact hr⟩
  | a :: b :: rest, its, hI, hs, hc => by
    have ha := hc a (by simp)
    have hb := hc b (by simp)
    have hr : ∀ l ∈ rest, Ok n its l := fun l hl => hc l (by simp [hl])
    obtain ⟨hI1, hm⟩ := muxL_ok hI hs hb ha
    have e := (muxL_spec (x := fun _ => false) (n := n) (s := s) (hi := b) its ha.1).1
    obtain ⟨hI2, hq⟩ := levelL_ok rest _ hI1 (hs.ext e) (fun l hl => (hr l hl).ext e)
    have e' := (levelL_spec (x := fun _ => false) (n := n) rest _ (hs.1.ext e) (fun l hl => (hr l hl).1.ext e)).1
    refine ⟨hI2, fun l hl => ?_⟩
    rcases List.mem_cons.1 hl with rfl | hl
    · exact hm.ext e'
    · exact hq l hl

theorem gatherL_ok : ∀ (idx cur : List (List ℕ)) (its : Array Item), ItemsOk n its → (∀ l ∈ idx, Ok n its l) →
    (∀ l ∈ cur, Ok n its l) →
    ItemsOk n (gatherL n idx cur its).2 ∧ Ok n (gatherL n idx cur its).2 (gatherL n idx cur its).1
  | [], cur, its, hI, _, hc => by
    refine ⟨hI, ?_⟩
    cases cur with
    | nil => exact ok_nil its
    | cons a _ => exact hc a (List.mem_cons_self ..)
  | s :: idx, cur, its, hI, hi, hc => by
    have hs := hi s (List.mem_cons_self ..)
    obtain ⟨hI1, hq⟩ := levelL_ok cur its hI hs hc
    have e := (levelL_spec (x := fun _ => false) (n := n) cur its hs.1 (fun l hl => (hc l hl).1)).1
    exact gatherL_ok idx _ _ hI1 (fun l hl => (hi l (List.mem_cons_of_mem _ hl)).ext e) hq

theorem gateL_ok {its : Array Item} {t nn a b : List ℕ} (hI : ItemsOk n its) (ht : Ok n its t) (hn : Ok n its nn)
    (ha : Ok n its a) (hb : Ok n its b) :
    ItemsOk n (gateL n its t nn a b).2 ∧ Ok n (gateL n its t nn a b).2 (gateL n its t nn a b).1 := by
  obtain ⟨hI1, hab⟩ := andL_ok hI ha hb
  have e1 := (andL_spec (x := fun _ => false) (n := n) its a b).1
  have hs : Ok n its (symm a b) := ha.symm hb
  obtain ⟨hI2, hr⟩ := andL_ok hI1 (ht.ext e1) (hab.symm (hs.ext e1))
  have e2 := (andL_spec (x := fun _ => false) (n := n) (andL n its a b).2 t (symm (andL n its a b).1 (symm a b))).1
  exact ⟨hI2, ((hs.ext (e1.trans e2)).symm hr).symm (hn.ext (e1.trans e2))⟩

theorem outsL_ok (ws : List (List ℕ)) : ∀ (idxs : List (List (List ℕ))) (its : Array Item), ItemsOk n its →
    (∀ l ∈ ws, Ok n its l) → (∀ idx ∈ idxs, ∀ l ∈ idx, Ok n its l) →
    ItemsOk n (outsL n ws idxs its).2 ∧ ∀ l ∈ (outsL n ws idxs its).1, Ok n (outsL n ws idxs its).2 l
  | [], its, hI, _, _ => ⟨hI, fun _ h => absurd h List.not_mem_nil⟩
  | idx :: rest, its, hI, hw, hi => by
    have hi0 := hi idx (List.mem_cons_self ..)
    have hi' : ∀ i ∈ rest, ∀ l ∈ i, Ok n its l := fun i h => hi i (List.mem_cons_of_mem _ h)
    obtain ⟨hI1, hr⟩ := gatherL_ok idx ws its hI hi0 hw
    have e := (gatherL_spec (x := fun _ => false) (n := n) idx ws its (fun l hl => (hi0 l hl).1) (fun l hl => (hw l hl).1)).1
    obtain ⟨hI2, hq⟩ := outsL_ok ws rest _ hI1 (fun l hl => (hw l hl).ext e) (fun i h l hl => (hi' i h l hl).ext e)
    have e' := (outsL_spec (x := fun _ => false) (n := n) ws rest _ (fun l hl => (hw l hl).1.ext e)
      (fun i h l hl => (hi' i h l hl).1.ext e)).1
    refine ⟨hI2, fun l hl => ?_⟩
    rcases List.mem_cons.1 hl with rfl | hl
    · exact hr.ext e'
    · exact hq l hl

end Flat

theorem gatesL_ok : ∀ g, g ≤ p.g →
    ItemsOk (nIns p) (gatesL p g).2 ∧ ∀ l ∈ (gatesL p g).1, Ok (nIns p) (gatesL p g).2 l
  | 0, _ => by
    refine ⟨fun k hk => absurd hk (by simp [gatesL]), fun l hl => ?_⟩
    obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hl
    exact ⟨good_input _ (by simp only [nIns]; omega), List.pairwise_singleton _ _⟩
  | g + 1, hg => by
    have hlt : g < p.g := hg
    obtain ⟨hI, hq⟩ := gatesL_ok g (Nat.le_of_succ_le hg)
    set q := gatesL p g with hqdef
    have hsrc : ∀ (s : Fin 2) (its : Array Item), ∀ l ∈ srcIdx p g s, Ok (nIns p) its l := fun s its l hl =>
      ⟨goods_srcIdx its ⟨g, hlt⟩ s l hl, by obtain ⟨j, rfl⟩ := List.mem_ofFn.1 hl; exact List.pairwise_singleton _ _⟩
    obtain ⟨hIa, ha⟩ := gatherL_ok (srcIdx p g 0) q.1 q.2 hI (hsrc 0 q.2) hq
    have ea := (gatherL_spec (x := fun _ => false) (n := nIns p) (srcIdx p g 0) q.1 q.2 (goods_srcIdx q.2 ⟨g, hlt⟩ 0)
      (fun l hl => (hq l hl).1)).1
    set a := gatherL (nIns p) (srcIdx p g 0) q.1 q.2 with hadef
    obtain ⟨hIb, hb⟩ := gatherL_ok (srcIdx p g 1) q.1 a.2 hIa (hsrc 1 a.2) (fun l hl => (hq l hl).ext ea)
    have eb := (gatherL_spec (x := fun _ => false) (n := nIns p) (srcIdx p g 1) q.1 a.2 (goods_srcIdx a.2 ⟨g, hlt⟩ 1)
      (fun l hl => (hq l hl).1.ext ea)).1
    set b := gatherL (nIns p) (srcIdx p g 1) q.1 a.2 with hbdef
    have ht : Ok (nIns p) b.2 (opW g 0) := ⟨(opW_spec (p := p) (fun _ => false) b.2 ⟨g, hlt⟩ 0).1, List.pairwise_singleton _ _⟩
    have hn : Ok (nIns p) b.2 (opW g 1) := ⟨(opW_spec (p := p) (fun _ => false) b.2 ⟨g, hlt⟩ 1).1, List.pairwise_singleton _ _⟩
    obtain ⟨hIr, hr⟩ := gateL_ok hIb ht hn (ha.ext eb) hb
    have er := (gateL_spec (x := fun _ => false) (n := nIns p) b.2 ht.1 hn.1 (ha.1.ext eb) hb.1).1
    set r := gateL (nIns p) b.2 (opW g 0) (opW g 1) a.1 b.1 with hrdef
    have hstep : gatesL p (g + 1) = (q.1 ++ [r.1], r.2) := by
      rw [gatesL, ite_eq_left hlt]
    rw [hstep]
    refine ⟨hIr, fun l hl => ?_⟩
    rcases List.mem_append.1 hl with hl | hl
    · exact (hq l hl).ext (ea.trans (eb.trans er))
    · rw [List.mem_singleton.1 hl]; exact hr

theorem outsAll_ok : ItemsOk (nIns p) (outsAll p).2 ∧ ∀ l ∈ (outsAll p).1, Ok (nIns p) (outsAll p).2 l := by
  obtain ⟨hI, hq⟩ := gatesL_ok (p := p) p.g le_rfl
  refine outsL_ok _ _ _ hI hq fun idx h l hl => ?_
  obtain ⟨i, rfl⟩ := List.mem_ofFn.1 h
  exact ⟨goods_outIdx _ i l hl, by obtain ⟨j, rfl⟩ := List.mem_ofFn.1 hl; exact List.pairwise_singleton _ _⟩

/-- **`flatOk` at every width**: every item of `uuType p` is an AND of two forms whose wires ascend below its own, and
its `N_OUT` outputs' wires ascend below the last. -/
theorem flatOk_uuType : flatOk (uuType p) = true := by
  obtain ⟨hI, hO⟩ := outsAll_ok (p := p)
  unfold flatOk
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range, beq_iff_eq, uuType_inBits]
  refine ⟨⟨fun k hk => ?_, ?_⟩, ?_⟩
  · obtain ⟨a, b, he, hsa, hsb, ha, hb⟩ := hI k hk
    have hi : (uuType p).items[k]? = some (.and (form a) (form b)) := by
      simp only [uuType] at hk ⊢
      rw [Array.getElem?_eq_getElem hk, he]
    rw [hi]
    simp only [formOk_form hsa ha, formOk_form hsb hb, Bool.and_self]
  · rw [size_outputs]
    simp [uuType, CircuitType.outBits, CircuitType.portBits]
  · simp only [uuType, List.all_toArray', List.all_map, List.all_eq_true, Function.comp]
    exact fun l hl => formOk_form (hO l hl).2 (hO l hl).1

/-! ## Its rows -/

/-- **Soundness of the unit's rows.** Whenever `derive` lays `uuType p` out, column values that satisfy its rows with the
constant at 1 carry on output `i`'s column `Desc.eval` of the description on the first `width` input columns, on the data
on the last `N_IN`. -/
theorem uu_sound {inSplit outSplit : List ℕ} {dd : Derived} (hd : derive (uuType p) inSplit outSplit = .ok dd)
    (z : ℕ → Bool) (hz : Sat dd.rows z) (h1 : z dd.geo.const = true) (i : Fin p.nOut) :
    z (dd.outCol i) = Desc.eval (fun j : Fin p.width => z (dd.inCol j)) (fun a => z (dd.inCol (p.width + a))) i := by
  rw [compose_sound hd z hz h1 i (by rw [size_outputs]; exact i.isLt), eval_uuType, getD_ofFn, dite_eq_left i.isLt]
  rfl

/-- The input bits of a description `d` and data `xs`. -/
def inputs (d : Desc p) (xs : Fin p.nIn → Bool) (i : ℕ) : Bool :=
  if h : i < p.width then d ⟨i, h⟩ else if h' : i - p.width < p.nIn then xs ⟨i - p.width, h'⟩ else false

theorem descOf_inputs (d : Desc p) (xs : Fin p.nIn → Bool) : descOf (inputs d xs) = d := by
  funext j; simp [descOf, inputs, j.isLt]

theorem dataOf_inputs (d : Desc p) (xs : Fin p.nIn → Bool) : dataOf (p := p) (inputs d xs) = xs := by
  funext i; simp [dataOf, inputs, i.isLt]

/-- **The unit's statement, exactly.** When `derive` lays `uuType p` out, outputs `y` are carried by some assignment that
satisfies the rows with the constant at 1, with description `d` and data `xs` on the input columns, exactly when
`y = d.eval xs`. -/
theorem uu_rows_iff {inSplit outSplit : List ℕ} {dd : Derived} (hd : derive (uuType p) inSplit outSplit = .ok dd)
    (d : Desc p) (xs : Fin p.nIn → Bool) (y : Fin p.nOut → Bool) :
    (∃ z : ℕ → Bool, Sat dd.rows z ∧ z dd.geo.const = true ∧ (∀ j : Fin p.width, z (dd.inCol j) = d j) ∧
      (∀ a : Fin p.nIn, z (dd.inCol (p.width + a)) = xs a) ∧ ∀ i : Fin p.nOut, z (dd.outCol i) = y i) ↔
      y = d.eval xs := by
  constructor
  · rintro ⟨z, hz, h1, hd', hx, hy⟩
    funext i
    rw [← hy i, uu_sound hd z hz h1 i]
    congr 1
    · funext j; exact hd' j
    · funext a; exact hx a
  · rintro rfl
    obtain ⟨hs, h1, hin, hout⟩ := compose_complete hd (inputs d xs)
    refine ⟨_, hs, h1, fun j => ?_, fun a => ?_, fun i => ?_⟩
    · rw [hin j (by rw [uuType_inBits]; simp only [nIns]; omega)]
      simp [inputs, j.isLt]
    · rw [hin _ (by rw [uuType_inBits]; simp only [nIns]; omega)]
      simp [inputs, a.isLt]
    · rw [hout i (by rw [size_outputs]; exact i.isLt), eval_uuType, getD_ofFn, dite_eq_left i.isLt, descOf_inputs,
        dataOf_inputs]

/-- **On a typed statement's unit rows.** When `deriveChecked` accepts a statement's layouts and its unit's type is
`uuType p`, column values that satisfy the unit's rows as the block holds them (Δ applied), with the block's `one` at 1,
carry `Desc.eval` of their description columns on their data columns. -/
theorem uu_unit_sound {types : String → Option CircuitType} {ls : Array Flock.Layout} {info : String → Option TableInfo}
    {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array DeriveAll.D}
    (hd : DeriveAll.deriveChecked types ls info words unit = .ok done)
    (ht : (ls.getD unit default).type.bind types = some (uuType p)) (one : ℕ) (z : ℕ → Bool)
    (hz : Dag.SatAt (DeriveAll.blockRow words (done.getD unit default) one) z) (h1 : z one = true) (i : Fin p.nOut) :
    z ((done.getD unit default).outCols.getD i 0) =
      Desc.eval (fun j : Fin p.width => z ((done.getD unit default).inCols.getD j 0))
        (fun a => z ((done.getD unit default).inCols.getD (p.width + a) 0)) i := by
  rw [Dag.unit_sound hd ht one z hz h1 (unit + 1) (Nat.lt_succ_self unit) i (by rw [size_outputs]; exact i.isLt),
    Dag.evalT_flat types words flatOk_uuType unit, eval_uuType, getD_ofFn, dite_eq_left i.isLt]
  rfl

/-! ## The tie to Python

`circuit_types.from_gf2(C, op + src + out + x, build(C, G, N_IN, N_OUT, …), [(1, 2G), (1, 2G·K), (1, N_OUT·K),
(1, N_IN)], [(1, N_OUT)], "program", {})` over a `gf2.Circuit`, its digest. The item counts are `universal.and_count`. -/

/-- The type's digest is `h`. -/
def digestIs (t : CircuitType) (h : String) : Bool :=
  match t.digest with
  | .ok d => d == h
  | .error _ => false

/-- The format's rules accept the type, and `derive` lays it out (one group a side), so `uu_rows_iff` applies. -/
def accepted (t : CircuitType) : Bool :=
  (match CircuitType.check (fun _ => none) t with | .ok _ => true | .error _ => false) &&
    match derive t [] [] with | .ok _ => true | .error _ => false

#guard ([⟨1, 1, 1⟩, ⟨2, 2, 1⟩, ⟨3, 2, 2⟩, ⟨5, 3, 4⟩, ⟨7, 9, 3⟩, ⟨16, 16, 8⟩] : List Params).all (accepted ∘ uuType)

#guard (uuType ⟨1, 1, 1⟩).items.size == 5
#guard digestIs (uuType ⟨1, 1, 1⟩)
  "7893b854a063f857204b130c83412de9bab8b4343c2a310082a6ebf92a2463e1a47c3b8ef8d89e12e23a0eed1ce55bad57b116d5408ec7ca04f33e25adb94dca"
#guard (uuType ⟨2, 2, 1⟩).items.size == 17
#guard digestIs (uuType ⟨2, 2, 1⟩)
  "328621bfd9f046f61da710617fd3d3c05b5610b264756be3c537d31a06bc3a82f72642f3b1e07841418e4b25ec1c82b57db902d39c47f032b57de0ea9ec9f474"
#guard (uuType ⟨3, 2, 2⟩).items.size == 40
#guard digestIs (uuType ⟨3, 2, 2⟩)
  "af5181c36f9cdf73afa01bad1b1483534c7bc447dc7b95aade613d3e9451f5411853fc2d4439ecb279b0b26d0c0eccb510116dd0eb227b6e03421ae26237679a"
#guard (uuType ⟨5, 3, 4⟩).items.size == 92
#guard digestIs (uuType ⟨5, 3, 4⟩)
  "7a9f3ed9861b21b68c5b8965f0911c01b97427c874628e97f3ca655237f98ed4a7239c8af3917a98fd84221b0505cfc65db031474cd992864fae8c61a010ce97"
#guard (uuType ⟨7, 9, 3⟩).items.size == 237
#guard digestIs (uuType ⟨7, 9, 3⟩)
  "91e4515de7ea597ca9b8693a3ecdd3be4a2da6a94fe1d9f1fa653bb6a450ca0abfe8eda5e612b7f2b94ed393f90d4e8c56838f130c9becea5da7d996053a7f8e"
#guard (uuType ⟨16, 16, 8⟩).items.size == 1066
#guard digestIs (uuType ⟨16, 16, 8⟩)
  "4dc00ba99ad07dd1fd8d4433b9ce86acbe43fc079535fd77bfab60cde2732c1cd52b2a033b355e8f5ff2cae416ef6baed086c72d1dbda118fd9b34591516d0d3"

end FlockSoundness.Discharge.Recursion.UniversalLower
