import Proofs.Flock.Recursive.Universal
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.List.GetD

/-!
# Recursion: the universal unit's class of private circuits

`UniversalUnit_v1{G, N_IN, N_OUT}` (`verity/ml/private_circuits/universal.py`) runs a private circuit given as a value,
its description `(op, src, out)`, on the data `x`. The class (Daniel, 9 Oct, decision 3) is every description of the
unit's width, read in gate order: each gate reads only inputs and earlier gates, so no description has a cycle. This
file states that class and what it reveals.

* `Params`: `G`, `N_IN`, `N_OUT`; `W = N_IN + G` wires, `K = max 1 (bit_length (W − 1))` bits a wire index
  (`universal.index_bits`), and the width `2G + 2G·K + N_OUT·K`. The real class is `{G = 8090, N_IN = 64, N_OUT = 32}`:
  `K = 13`, 226,936 bits (`real_width`).
* `layout`: the description's bit order, `universal.arguments`' leaf order (op, then src, then out; gate-major, the
  sources `a` then `b`, each index least significant bit first), as an equivalence between the places (an op bit, a
  source index's bit, an output index's bit) and the positions of the width (`layout_op`, `layout_src`, `layout_out`).
* `Desc p`: a description, any `Fin p.width → Bool`. `Desc.eval` is `universal.interpret` on its bits: the wire list
  grows gate by gate, a source `a` reads `w[a]` when `a < len w` and `0` otherwise, an output `o` reads wire `o` when
  `o < W` and `0` otherwise. Its defining equations: `wire_input`, `wire_gate` (wire `N_IN + g` is
  `op_g (src a_g) (src b_g)`, `src i` wire `i` when `i < N_IN + g` and `0` otherwise), `eval_eq` (output `j` is wire
  `out_j` when `out_j < W`, else `0`).
* `Circ p`: a private circuit of the class, acyclic by its type (gate `g`'s sources are `Option (Fin (N_IN + g))`).
  `decode` is total, with no well-formedness hypothesis; `eval_decode`: the description runs as its decoded circuit
  does. `decode_surjective`: every circuit of the class is some description's (when its outputs are wires, or
  `W < 2^K` leaves an index for the constant `0`).
* `uclass p`, `mem_uclass`, `mem_uclass_iff`, `card_desc`: the class is every description of the width, so it reveals
  `p` and nothing else: membership is the same for every private circuit.
* `k_vectors`, `real_width`, `eval_vectors_111` (all 32 descriptions of `{1, 1, 1}`), `eval_vectors_322` (six random
  ones of `{3, 2, 2}`, with forward reads and outputs past `W`): Python's `index_bits` and `interpret`, by `decide`.
* `Hidden.side_of_nnz_zero`, `HClass.ofStmt_toStatement`, `Hidden.exists_side_ne`: how `HClass.ofStmt` relates. At
  `nnz = 0` its class is one statement, the executable's own (`Refine.stmtOf st`): the universal unit hides nothing in
  its matrices, and a description lives in the witness rows and the regions' commit strings. With a hidden entry and a
  unit slot inside the block, the class holds a statement whose `A₀` is not the public one, which is no universal unit.
-/

namespace FlockSoundness.Discharge.Recursion.UniversalClass

/-- Python's `int.bit_length`: 0 for 0, else `⌊log₂ n⌋ + 1`. -/
def bitLength (n : ℕ) : ℕ := if n = 0 then 0 else n.log2 + 1

theorem lt_two_pow_bitLength (n : ℕ) : n < 2 ^ bitLength n := by
  unfold bitLength
  split_ifs with h
  · simp [h]
  · exact Nat.lt_log2_self

/-- `UniversalUnit_v1{G, N_IN, N_OUT}`'s parameters. -/
structure Params where
  g : ℕ
  nIn : ℕ
  nOut : ℕ
  deriving DecidableEq

namespace Params

variable (p : Params)

/-- `W = N_IN + G`: the inputs, then one wire a gate. -/
def w : ℕ := p.nIn + p.g

/-- `K`, the bits of a wire index: `universal.index_bits`, `max(1, (N_IN + G - 1).bit_length())`. -/
def k : ℕ := max 1 (bitLength (p.w - 1))

/-- The description's width: `2G` op bits, `2G·K` source bits, `N_OUT·K` output bits. -/
abbrev width : ℕ := p.g * 2 + (p.g * 2 * p.k + p.nOut * p.k)

theorem one_le_k : 1 ≤ p.k := le_max_left _ _

/-- Every wire has a `K`-bit index: `W ≤ 2^K`. -/
theorem w_le_two_pow_k : p.w ≤ 2 ^ p.k := by
  have h1 := lt_two_pow_bitLength (p.w - 1)
  have h2 : 2 ^ bitLength (p.w - 1) ≤ 2 ^ p.k := Nat.pow_le_pow_right (by decide) (le_max_right _ _)
  have h3 : 1 ≤ 2 ^ p.k := Nat.one_le_two_pow
  omega

end Params

/-- A position of the description: op bit `t` of gate `g` (`inl (g, t)`); bit `j` of gate `g`'s source `s`, `0` for `a`
and `1` for `b` (`inr (inl ((g, s), j))`); bit `j` of output `i` (`inr (inr (i, j))`). -/
abbrev Place (p : Params) := (Fin p.g × Fin 2) ⊕ ((Fin p.g × Fin 2) × Fin p.k) ⊕ (Fin p.nOut × Fin p.k)

/-- **The description's bit order**, `universal.arguments`' leaf order: the op bits, gate by gate (bit 0, then bit 1);
then the sources, gate by gate, `a` then `b`, each index's `K` bits least significant first; then the outputs, each
index's `K` bits least significant first. An equivalence: every position of the width is exactly one place. -/
def layout (p : Params) : Place p ≃ Fin p.width :=
  (Equiv.sumCongr finProdFinEquiv
      ((Equiv.sumCongr ((Equiv.prodCongr finProdFinEquiv (Equiv.refl _)).trans finProdFinEquiv)
        finProdFinEquiv).trans finSumFinEquiv)).trans finSumFinEquiv

variable {p : Params}

theorem layout_op (g : Fin p.g) (t : Fin 2) : (layout p (.inl (g, t))).val = 2 * g.val + t.val := by
  simp only [layout, Equiv.trans_apply, Equiv.sumCongr_apply, Sum.map_inl, finSumFinEquiv_apply_left, Fin.val_castAdd,
    finProdFinEquiv_apply_val]
  omega

theorem layout_src (g : Fin p.g) (s : Fin 2) (j : Fin p.k) :
    (layout p (.inr (.inl ((g, s), j)))).val = 2 * p.g + (2 * g.val + s.val) * p.k + j.val := by
  simp only [layout, Equiv.trans_apply, Equiv.sumCongr_apply, Sum.map_inr, Sum.map_inl, finSumFinEquiv_apply_left,
    finSumFinEquiv_apply_right, Fin.val_castAdd, Fin.val_natAdd, finProdFinEquiv_apply_val, Equiv.prodCongr_apply,
    Prod.map, Equiv.refl_apply]
  ring

theorem layout_out (i : Fin p.nOut) (j : Fin p.k) :
    (layout p (.inr (.inr (i, j)))).val = 2 * p.g + 2 * p.g * p.k + i.val * p.k + j.val := by
  simp only [layout, Equiv.trans_apply, Equiv.sumCongr_apply, Sum.map_inr, finSumFinEquiv_apply_right,
    Fin.val_natAdd, finProdFinEquiv_apply_val]
  ring

/-- **A description** of the class: any bit string of the width. -/
abbrev Desc (p : Params) := Fin p.width → Bool

/-- The value of `k` bits, least significant first. -/
def ofBits : (k : ℕ) → (Fin k → Bool) → ℕ
  | 0, _ => 0
  | k + 1, b => (b 0).toNat + 2 * ofBits k fun j => b j.succ

/-- **The gate's function** on its sources' values: op bit 0 picks `a ∧ b` over `a xor b`, op bit 1 negates. Op 0, 1, 2,
3 are XOR, AND, XNOR, NAND. -/
def gateFn (t n a b : Bool) : Bool := xor (if t then a && b else xor a b) n

namespace Desc

variable (d : Desc p)

/-- Gate `g`'s op bit 0: AND over XOR. -/
def opT (g : Fin p.g) : Bool := d (layout p (.inl (g, 0)))

/-- Gate `g`'s op bit 1: negate. -/
def opN (g : Fin p.g) : Bool := d (layout p (.inl (g, 1)))

/-- Gate `g`'s source `s` (`0` is `a`, `1` is `b`), a `K`-bit index. -/
def src (g : Fin p.g) (s : Fin 2) : ℕ := ofBits p.k fun j => d (layout p (.inr (.inl ((g, s), j))))

/-- Output `i`, a `K`-bit index. -/
def out (i : Fin p.nOut) : ℕ := ofBits p.k fun j => d (layout p (.inr (.inr (i, j))))

/-- **The wire list before gate `g`**, as `universal.interpret` builds it: the inputs `x`, then gates `0 … g−1`, each
the gate's function of `w[a]` and `w[b]`, where `w[i]` is `0` when `i` is not below the list's length. -/
def wiresTo (x : Fin p.nIn → Bool) : ℕ → List Bool
  | 0 => List.ofFn x
  | g + 1 =>
    if h : g < p.g then
      wiresTo x g ++ [gateFn (d.opT ⟨g, h⟩) (d.opN ⟨g, h⟩) ((wiresTo x g).getD (d.src ⟨g, h⟩ 0) false)
        ((wiresTo x g).getD (d.src ⟨g, h⟩ 1) false)]
    else wiresTo x g

/-- Wire `i`: the inputs, then the gates; `0` past the last of the `W` wires. -/
def wire (x : Fin p.nIn → Bool) (i : ℕ) : Bool := (d.wiresTo x p.g).getD i false

/-- **The description run on `x`**, `universal.interpret`: output `i` is wire `out_i`, or `0` when `out_i ≥ W`. -/
def eval (x : Fin p.nIn → Bool) (i : Fin p.nOut) : Bool := (d.wiresTo x p.g).getD (d.out i) false

end Desc


theorem getD_ofFn {n : ℕ} (v : Fin n → Bool) (a : ℕ) :
    (List.ofFn v).getD a false = if h : a < n then v ⟨a, h⟩ else false := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_ofFn]
  split_ifs <;> rfl

namespace Desc

variable (d : Desc p) (x : Fin p.nIn → Bool)

theorem length_wiresTo : ∀ g, g ≤ p.g → (d.wiresTo x g).length = p.nIn + g
  | 0, _ => by simp [wiresTo]
  | g + 1, h => by
    rw [wiresTo, dite_eq_left (by omega), List.length_append, length_wiresTo g (by omega), List.length_singleton]
    rfl

theorem getD_wiresTo_succ {g : ℕ} (hg : g < p.g) {i : ℕ} (hi : i < p.nIn + g) :
    (d.wiresTo x (g + 1)).getD i false = (d.wiresTo x g).getD i false := by
  rw [wiresTo, dite_eq_left hg, List.getD_append _ _ _ _ (by rw [d.length_wiresTo x g hg.le]; exact hi)]

theorem getD_wiresTo_mono {g g' : ℕ} (hg : g ≤ g') (hg' : g' ≤ p.g) {i : ℕ} (hi : i < p.nIn + g) :
    (d.wiresTo x g').getD i false = (d.wiresTo x g).getD i false := by
  induction g', hg using Nat.le_induction with
  | base => rfl
  | succ g' hle ih => rw [d.getD_wiresTo_succ x (by omega) (by omega), ih (by omega)]

/-- Wire `i`, below gate `g`'s own wire, is the wire list's before gate `g`. -/
theorem wire_eq {g : ℕ} (hg : g ≤ p.g) {i : ℕ} (hi : i < p.nIn + g) :
    d.wire x i = (d.wiresTo x g).getD i false :=
  d.getD_wiresTo_mono x hg le_rfl hi

/-- **What gate `g` reads at source index `a`**: wire `a` when `a < N_IN + g` (an input or an earlier gate), and `0`
otherwise, which is gate `g`'s own wire, every later one, and every index past `W`. -/
def srcVal (g a : ℕ) : Bool := if a < p.nIn + g then d.wire x a else false

theorem getD_wiresTo_eq_srcVal {g : ℕ} (hg : g ≤ p.g) (a : ℕ) :
    (d.wiresTo x g).getD a false = d.srcVal x g a := by
  unfold srcVal
  split_ifs with ha
  · exact (d.wire_eq x hg ha).symm
  · exact List.getD_eq_default _ _ (by rw [d.length_wiresTo x g hg]; omega)

/-- **The inputs**: wire `i < N_IN` is `x_i`. -/
theorem wire_input (i : Fin p.nIn) : d.wire x i = x i := by
  rw [d.wire_eq x (Nat.zero_le _) (show i.val < p.nIn + 0 from i.isLt), wiresTo, getD_ofFn, dite_eq_left i.isLt]

/-- **Gate `g`'s wire**: wire `N_IN + g` is the gate's function (op bits `t`, `n`) of what it reads at its two source
indices, `srcVal`. -/
theorem wire_gate (g : Fin p.g) :
    d.wire x (p.nIn + g) = gateFn (d.opT g) (d.opN g) (d.srcVal x g (d.src g 0)) (d.srcVal x g (d.src g 1)) := by
  rw [d.wire_eq x (g := g + 1) g.isLt (by omega), wiresTo, dite_eq_left g.isLt,
    List.getD_append_right _ _ _ _ (by rw [d.length_wiresTo x g g.isLt.le]),
    d.length_wiresTo x g g.isLt.le, Nat.sub_self, List.getD_cons_zero,
    d.getD_wiresTo_eq_srcVal x g.isLt.le, d.getD_wiresTo_eq_srcVal x g.isLt.le]

/-- Past the last of the `W` wires, a wire is `0`. -/
theorem wire_of_le {i : ℕ} (h : p.w ≤ i) : d.wire x i = false :=
  List.getD_eq_default _ _ (by rw [d.length_wiresTo x p.g le_rfl]; exact h)

/-- **The outputs**: output `i` is wire `out_i` when `out_i < W`, and `0` otherwise. -/
theorem eval_eq (i : Fin p.nOut) : d.eval x i = if d.out i < p.w then d.wire x (d.out i) else false := by
  split_ifs with h
  · rfl
  · exact d.wire_of_le x (not_lt.1 h)

end Desc

/-- **A private circuit of the class, acyclic by its type.** Gate `g` is its op bits `(t, n)` on two sources, each one
of the `N_IN + g` wires before it (the inputs, then gates `0 … g−1`) or the constant `0` (`none`), and output `i` is one
of the `W` wires or the constant `0`. No source can name its own gate or a later one. -/
structure Circ (p : Params) where
  op : Fin p.g → Bool × Bool
  src : (g : Fin p.g) → Fin 2 → Option (Fin (p.nIn + g))
  out : Fin p.nOut → Option (Fin p.w)

/-- A source's or an output's value: its wire's, or `0`. -/
def rd {n : ℕ} (v : Fin n → Bool) : Option (Fin n) → Bool
  | some i => v i
  | none => false

namespace Circ

variable (c : Circ p) (x : Fin p.nIn → Bool)

/-- The values of the first `N_IN + g` wires, gate by gate: gate `g` reads only `vals g`. -/
def vals : (g : ℕ) → g ≤ p.g → Fin (p.nIn + g) → Bool
  | 0, _ => x
  | g + 1, h => Fin.snoc (α := fun _ => Bool) (vals g (Nat.le_of_succ_le h))
      (gateFn (c.op ⟨g, h⟩).1 (c.op ⟨g, h⟩).2 (rd (vals g (Nat.le_of_succ_le h)) (c.src ⟨g, h⟩ 0))
        (rd (vals g (Nat.le_of_succ_le h)) (c.src ⟨g, h⟩ 1)))

/-- The circuit run on `x`. -/
def eval (i : Fin p.nOut) : Bool := rd (c.vals x p.g le_rfl) (c.out i)

end Circ

/-- **The decoding, total**: every description is a circuit of the class, with no hypothesis. A source index at or past
the reading gate's own wire is the constant `0`, and so is an output index past `W`. -/
def decode (d : Desc p) : Circ p where
  op g := (d.opT g, d.opN g)
  src g s := if h : d.src g s < p.nIn + g then some ⟨_, h⟩ else none
  out i := if h : d.out i < p.w then some ⟨_, h⟩ else none

theorem rd_dite {n a : ℕ} (v : Fin n → Bool) :
    rd v (if h : a < n then some ⟨a, h⟩ else none) = (List.ofFn v).getD a false := by
  rw [getD_ofFn]
  split_ifs <;> rfl

/-- The wire list before gate `g` is the decoded circuit's first `N_IN + g` values. -/
theorem wiresTo_eq_ofFn (d : Desc p) (x : Fin p.nIn → Bool) :
    ∀ g (h : g ≤ p.g), d.wiresTo x g = List.ofFn ((decode d).vals x g h)
  | 0, _ => rfl
  | g + 1, h => by
    rw [Desc.wiresTo, dite_eq_left (show g < p.g from h), wiresTo_eq_ofFn d x g (Nat.le_of_succ_le h)]
    show _ = List.ofFn (n := p.nIn + g + 1) _
    rw [List.ofFn_succ', List.concat_eq_append]
    simp only [Circ.vals, Fin.snoc_castSucc, Fin.snoc_last, decode, rd_dite]
    rfl

/-- **The description runs as its decoded circuit does**: `universal.interpret`'s reading of the bits (a forward read
or an output past `W` reads `0`) is the evaluation of an acyclic circuit of the class. -/
theorem eval_decode (d : Desc p) (x : Fin p.nIn → Bool) : (decode d).eval x = d.eval x := by
  funext i
  rw [Circ.eval, Desc.eval, wiresTo_eq_ofFn d x p.g le_rfl]
  exact rd_dite _

/-! ### Every circuit of the class is a description's -/

/-- The `k` bits of `n`, least significant first. -/
def toBits (k n : ℕ) : Fin k → Bool := fun j => n.testBit j

theorem ofBits_toBits : ∀ k n, n < 2 ^ k → ofBits k (toBits k n) = n
  | 0, n, h => by simp at h; simp [ofBits, h]
  | k + 1, n, h => by
    have hb : (fun j : Fin k => toBits (k + 1) n j.succ) = toBits k (n / 2) := by
      funext j
      simp [toBits, Fin.val_succ, Nat.testBit_succ]
    rw [ofBits, hb, ofBits_toBits k (n / 2) (by rw [pow_succ] at h; omega)]
    simp only [toBits, Fin.val_zero, Nat.testBit_zero]
    rcases Nat.mod_two_eq_zero_or_one n with h2 | h2 <;> simp [h2] <;> omega

/-- A circuit's source index: its wire's, or the reading gate's own wire (which reads `0`). -/
def Circ.srcIdx (c : Circ p) (g : Fin p.g) (s : Fin 2) : ℕ :=
  match c.src g s with
  | some a => a
  | none => p.nIn + g

/-- A circuit's output index: its wire's, or `W` (which reads `0`). -/
def Circ.outIdx (c : Circ p) (i : Fin p.nOut) : ℕ :=
  match c.out i with
  | some o => o
  | none => p.w

/-- **A circuit's description**: its op bits, and each index's `K` bits at its place in the layout. -/
def encode (c : Circ p) : Desc p := fun q =>
  match (layout p).symm q with
  | .inl (g, t) => if t = 0 then (c.op g).1 else (c.op g).2
  | .inr (.inl ((g, s), j)) => toBits p.k (c.srcIdx g s) j
  | .inr (.inr (i, j)) => toBits p.k (c.outIdx i) j

theorem srcIdx_lt (c : Circ p) (g : Fin p.g) (s : Fin 2) : c.srcIdx g s < 2 ^ p.k := by
  have h := p.w_le_two_pow_k
  unfold Circ.srcIdx Params.w at *
  split
  · rename_i a _
    have := a.isLt
    omega
  · omega

/-- **Every circuit of the class is some description's**, when each of its outputs is a wire or `W < 2^K` leaves an
index for the constant `0`. -/
theorem decode_encode (c : Circ p) (hout : p.w < 2 ^ p.k ∨ ∀ i, (c.out i).isSome) : decode (encode c) = c := by
  obtain ⟨op, src, out⟩ := c
  have hsrc : ∀ g s, (encode ⟨op, src, out⟩).src g s = Circ.srcIdx ⟨op, src, out⟩ g s := fun g s => by
    simp only [Desc.src, encode, Equiv.symm_apply_apply]
    exact ofBits_toBits _ _ (srcIdx_lt _ g s)
  have hout' : ∀ i, (encode ⟨op, src, out⟩).out i = Circ.outIdx ⟨op, src, out⟩ i := fun i => by
    simp only [Desc.out, encode, Equiv.symm_apply_apply]
    refine ofBits_toBits _ _ ?_
    unfold Circ.outIdx
    split
    · rename_i o _
      exact lt_of_lt_of_le o.isLt p.w_le_two_pow_k
    · rename_i hn
      rcases hout with h | h
      · exact h
      · exact absurd (h i) (by rw [hn]; simp)
  simp only [decode, Circ.mk.injEq]
  refine ⟨?_, ?_, ?_⟩
  · funext g
    simp [Desc.opT, Desc.opN, encode]
  · funext g s
    rw [hsrc]
    unfold Circ.srcIdx
    split
    · rename_i a ha
      simp only [a.isLt, dite_eq_left, Fin.eta]
      exact ha.symm
    · rename_i ha
      simp only [lt_irrefl, dite_false]
      exact ha.symm
  · funext i
    rw [hout']
    unfold Circ.outIdx
    split
    · rename_i o ho
      simp only [o.isLt, dite_eq_left, Fin.eta]
      exact ho.symm
    · rename_i ho
      simp only [lt_irrefl, dite_false]
      exact ho.symm

theorem decode_surjective (c : Circ p) (hout : p.w < 2 ^ p.k ∨ ∀ i, (c.out i).isSome) : ∃ d, decode d = c :=
  ⟨encode c, decode_encode c hout⟩

/-! ### The class -/

/-- **The class `UniversalUnit_v1{G, N_IN, N_OUT}` runs**: every description of the width, read in gate order. -/
def uclass (p : Params) : Set (Desc p) := Set.univ

/-- Every description is a member: no well-formedness condition exists to fail. -/
theorem mem_uclass (d : Desc p) : d ∈ uclass p := Set.mem_univ d

/-- **What the class reveals is its parameters**: a function of `p` alone, it holds every description of the width, so
membership is the same for every private circuit of the class. -/
theorem mem_uclass_iff (d d' : Desc p) : d ∈ uclass p ↔ d' ∈ uclass p := iff_of_true (mem_uclass d) (mem_uclass d')

/-- The class has `2^width` members. -/
theorem card_desc : Fintype.card (Desc p) = 2 ^ p.width := by simp

/-! ### Python's vectors

`universal.index_bits` and `universal.interpret` (with the layout read as `universal.arguments` writes it) on small
classes, checked by the kernel. -/

/-- The description whose bit `i` is bit `i` of `n`. -/
def Desc.ofNat (p : Params) (n : ℕ) : Desc p := fun i => n.testBit i

/-- The input whose bit `j` is bit `j` of `n`. -/
def inputOf (p : Params) (n : ℕ) : Fin p.nIn → Bool := fun j => n.testBit j

/-- `K` at each `(G, N_IN)`, as `universal.index_bits` gives it. -/
theorem k_vectors : ([(1, 1, 1), (1, 2, 2), (2, 2, 2), (3, 2, 3), (4, 4, 3), (5, 3, 3), (1, 7, 3), (8, 8, 4), (9, 8, 5),
    (2, 0, 1), (1, 0, 1), (8090, 64, 13), (64, 8090, 13)] : List (ℕ × ℕ × ℕ)).all
      (fun v => (Params.k ⟨v.1, v.2.1, 1⟩) == v.2.2) = true := by
  decide

/-- The real class, `UniversalUnit_v1{G = 8090, N_IN = 64, N_OUT = 32}`: `K = 13` and a description of 226,936 bits. -/
theorem real_width : (⟨8090, 64, 32⟩ : Params).k = 13 ∧ (⟨8090, 64, 32⟩ : Params).width = 226936 := by
  decide

/-- Every description of `{1, 1, 1}` (width 5) on both inputs: bit `2n + x` of the number is `interpret`'s output on
description `n` and input `x`. -/
theorem eval_vectors_111 : (List.range 32).all (fun n => (List.range 2).all fun x =>
    (Desc.ofNat ⟨1, 1, 1⟩ n).eval (inputOf _ x) ⟨0, by decide⟩ == (17353163729913162410 : ℕ).testBit (2 * n + x))
      = true := by
  decide

/-- Six random descriptions of `{3, 2, 2}` (`W = 5`, `K = 3`, width 30, so forward reads and outputs past `W` both
occur), each on its four inputs: `interpret`'s outputs. -/
theorem eval_vectors_322 : ([(716328577, [[0, 0], [0, 0], [0, 0], [0, 0]]),
    (999924601, [[1, 0], [1, 0], [1, 0], [1, 0]]), (839344450, [[1, 0], [0, 0], [1, 0], [0, 0]]),
    (214459031, [[0, 0], [0, 0], [0, 1], [0, 1]]), (899386510, [[0, 0], [0, 0], [0, 0], [0, 0]]),
    (495205171, [[0, 0], [0, 1], [0, 0], [0, 1]])] :
      List (ℕ × List (List ℕ))).all (fun v => (List.range 4).all fun x =>
        List.ofFn ((Desc.ofNat ⟨3, 2, 2⟩ v.1).eval (inputOf _ x)) == (v.2.getD x []).map (· == 1)) = true := by
  decide

end FlockSoundness.Discharge.Recursion.UniversalClass

namespace FlockSoundness.Discharge.Recursion

namespace Hidden

theorem tmpl_of_nnz_zero {cls : HClass} (h : cls.nnz = 0) (M : Hidden cls) (s : Bool) (i c : ℕ) :
    M.tmpl s i c = 0 := by
  have : IsEmpty (Fin cls.nnz) := by rw [h]; infer_instance
  simp [tmpl]

/-- **A class of no hidden entries hides nothing**: both sides of every hidden statement are the public part. -/
theorem side_of_nnz_zero {cls : HClass} (h : cls.nnz = 0) (M : Hidden cls) :
    M.side false = cls.A₀pub ∧ M.side true = cls.B₀pub := by
  constructor <;> funext i c <;> simp [side, tmpl_of_nnz_zero h]

/-- **A class with a hidden entry and a unit slot inside the block has a statement off its public part**: one value-1
entry at the slot's corner changes `A₀` there. -/
theorem exists_side_ne {cls : HClass} (hn : 0 < cls.nnz) {u : ℕ} (hu : u ∈ cls.units)
    (hlt : u * 2 ^ cls.slotLog < 2 ^ cls.kLog) : ∃ M : Hidden cls, M.side false ≠ cls.A₀pub := by
  let e₀ : HEntry cls.slotLog := ⟨false, ⟨0, Nat.two_pow_pos _⟩, ⟨0, Nat.two_pow_pos _⟩, 1⟩
  let M : Hidden cls := fun e => if e.val = 0 then e₀ else HEntry.pad
  refine ⟨M, fun hM => ?_⟩
  have h := congrFun (congrFun hM ⟨u * 2 ^ cls.slotLog, hlt⟩) ⟨u * 2 ^ cls.slotLog, hlt⟩
  have ht : M.tmpl false 0 0 = 1 := by
    unfold tmpl
    rw [Finset.sum_eq_single ⟨0, hn⟩]
    · simp [M, e₀]
    · intro e _ he
      have : e.val ≠ 0 := fun h0 => he (Fin.ext h0)
      simp [M, this, HEntry.pad]
    · exact fun h0 => absurd (Finset.mem_univ _) h0
  simp only [side, Bool.false_eq_true, ite_false, Nat.mul_div_cancel _ (Nat.two_pow_pos cls.slotLog),
    Nat.mul_mod_left, true_and, hu, ite_true, ht] at h
  exact one_ne_zero (add_eq_left.1 h)

end Hidden

/-- **`HClass.ofStmt` at no hidden entries is the executable statement itself**: every hidden statement of the class is
`Refine.stmtOf st` at its regions. The universal unit's class is this one statement: its matrices are the unit's, a
function of `{G, N_IN, N_OUT}` and the instances, and the private circuit is in the witness rows and the regions. -/
theorem HClass.ofStmt_toStatement (st : Flock.Stmt) (hwf : Refine.StmtWF st) (hm : 13 ≤ st.m) (slotLog : ℕ)
    (units : Finset ℕ) (M : Hidden (HClass.ofStmt st hwf hm slotLog units 0))
    (regs : List (FlockSoundness.Model.Region st.c.kLog (st.m - st.c.kLog))) :
    M.toStatement regs = Refine.stmtOf st hwf regs := by
  obtain ⟨hA, hB⟩ := Hidden.side_of_nnz_zero rfl M
  show (⟨st.m, st.c.kLog, hwf.hk, hwf.hk6, ⟨st.pin, hwf.pin⟩, M.side false, M.side true, regs⟩ :
    FlockSoundness.Model.Statement) = _
  rw [hA, hB]
  rfl

end FlockSoundness.Discharge.Recursion
