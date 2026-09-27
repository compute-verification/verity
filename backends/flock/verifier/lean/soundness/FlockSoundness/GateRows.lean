import Mathlib.Data.Nat.Bitwise
import FlockSoundness.Audit.Circuit

/-!
# Rows from a gate circuit (`flock-ir-unit/v2`)

A unit's rows are derived from its gate circuit (AND, XOR and NOT of earlier wires and constants,
the IR's Boolean export): each wire is a linear form over the committed columns, an XOR is the sum
of two forms, a NOT flips the constant, and an AND is a new computed row reading its operands'
forms. The layout is `flock-ir-unit/v2`: the input groups first, the ANDs in order, the output
groups as copy rows, and the constant last (`verity_flock.ir_lower.layout`).

A row's columns are a bitmask, so that the derivation computes with `Nat.xor`, which the kernel
evaluates natively. `psum z m n` is the parity of `z` over the bits of `m` below `n`, and
`xorSum_eq_psum` relates it to the audit's `xorSum` over a list of columns.
-/

namespace FlockSoundness.GateRows

open Audit

/-! ## Parity over the bits of a mask -/

/-- The XOR of `z c` over the bits `c < n` of `m`. -/
def psum (z : ℕ → Bool) (m : ℕ) : ℕ → Bool
  | 0 => false
  | n + 1 => Bool.xor (psum z m n) (m.testBit n && z n)

theorem psum_zero (z : ℕ → Bool) (n : ℕ) : psum z 0 n = false := by
  induction n with
  | zero => rfl
  | succ n ih => simp [psum, ih]

theorem psum_xor (z : ℕ → Bool) (a b n : ℕ) :
    psum z (a ^^^ b) n = Bool.xor (psum z a n) (psum z b n) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [psum, ih, Nat.testBit_xor]
    cases a.testBit n <;> cases b.testBit n <;> cases z n <;> cases psum z a n <;> cases psum z b n <;> rfl

theorem psum_two_pow (z : ℕ → Bool) (c n : ℕ) : psum z (2 ^ c) n = (decide (c < n) && z c) := by
  induction n with
  | zero => simp [psum]
  | succ n ih =>
    simp only [psum, ih, Nat.testBit_two_pow]
    by_cases h : c = n
    · subst h; simp
    · by_cases hc : c < n
      · simp [h, hc, show c < n + 1 by omega]
      · simp [h, hc, show ¬ c < n + 1 by omega]

/-- A list of columns as a mask: the XOR of their powers of two, so a repeated column cancels. -/
def listMask : List ℕ → ℕ
  | [] => 0
  | c :: l => 2 ^ c ^^^ listMask l

theorem xorSum_eq_psum (z : ℕ → Bool) (n : ℕ) (l : List ℕ) (hl : ∀ c ∈ l, c < n) :
    xorSum l z = psum z (listMask l) n := by
  induction l with
  | nil => simp [xorSum, listMask, psum_zero]
  | cons c l ih =>
    have hc : c < n := hl c (by simp)
    have ih' := ih (fun d hd => hl d (by simp [hd]))
    simp only [xorSum, List.map_cons, List.foldr_cons] at ih' ⊢
    rw [ih', listMask, psum_xor, psum_two_pow]
    simp [hc]

/-! ## A map from wires to forms, for the kernel

A complete binary tree of depth `d`, indexed by the low bits of the key: structural recursion, so the
kernel evaluates it. Keys below `2 ^ d`. -/

inductive PM (α : Type) where
  | leaf
  | node (v : Option α) (l r : PM α)

namespace PM

variable {α : Type}

def get : ℕ → PM α → ℕ → Option α
  | _, .leaf, _ => none
  | 0, .node v _ _, _ => v
  | d + 1, .node _ l r, k => if k % 2 = 0 then get d l (k / 2) else get d r (k / 2)

def set : ℕ → PM α → ℕ → α → PM α
  | 0, .leaf, _, a => .node (some a) .leaf .leaf
  | 0, .node _ l r, _, a => .node (some a) l r
  | d + 1, .leaf, k, a =>
    if k % 2 = 0 then .node none (set d .leaf (k / 2) a) .leaf else .node none .leaf (set d .leaf (k / 2) a)
  | d + 1, .node v l r, k, a =>
    if k % 2 = 0 then .node v (set d l (k / 2) a) r else .node v l (set d r (k / 2) a)

theorem get_leaf (d k : ℕ) : get d (.leaf : PM α) k = none := by
  cases d <;> rfl

theorem get_set (d : ℕ) (t : PM α) (k j : ℕ) (a : α) (hk : k < 2 ^ d) (hj : j < 2 ^ d) :
    get d (set d t k a) j = if k = j then some a else get d t j := by
  induction d generalizing t k j with
  | zero =>
    have : k = 0 := by simpa using hk
    have : j = 0 := by simpa using hj
    subst k; subst j
    cases t <;> simp [get, set]
  | succ d ih =>
    have hk' : k / 2 < 2 ^ d := by rw [pow_succ] at hk; omega
    have hj' : j / 2 < 2 ^ d := by rw [pow_succ] at hj; omega
    have e : ∀ t' : PM α, get d (set d t' (k / 2) a) (j / 2) =
        if k / 2 = j / 2 then some a else get d t' (j / 2) := fun t' => ih t' _ _ hk' hj'
    by_cases hkj : k = j
    · subst hkj
      cases t <;> by_cases h0 : k % 2 = 0 <;> simp [set, get, h0, e]
    · have hne : k % 2 = j % 2 → k / 2 ≠ j / 2 := by omega
      cases t <;> by_cases h0 : k % 2 = 0 <;> by_cases h1 : j % 2 = 0 <;>
        simp [set, get, h0, h1, e, hkj, get_leaf] <;> omega

end PM

/-! ## Gate circuits -/

/-- An operand: a constant, or a wire. Wires `0 … nIn − 1` are the inputs, and gate `k` makes wire
`nIn + k`. -/
inductive Src where
  | zero
  | one
  | wire (i : ℕ)

inductive Gate where
  | and (a b : Src)
  | xor (a b : Src)
  | not (a : Src)

/-- A gate circuit: its inputs, its gates in order, and its outputs. -/
structure Circ where
  nIn : ℕ
  gates : List Gate
  outs : List Src

def Src.val (v : ℕ → Bool) : Src → Bool
  | .zero => false
  | .one => true
  | .wire i => v i

def Gate.val (v : ℕ → Bool) : Gate → Bool
  | .and a b => a.val v && b.val v
  | .xor a b => Bool.xor (a.val v) (b.val v)
  | .not a => !(a.val v)

/-- The wire values once the gates from wire `w` on have run, each on the values before it. -/
def run (v : ℕ → Bool) (w : ℕ) : List Gate → ℕ → Bool
  | [] => v
  | g :: gs => run (fun j => if j = w then g.val v else v j) (w + 1) gs

/-- The circuit's outputs on inputs `x`. -/
def Circ.eval (C : Circ) (x : ℕ → Bool) : List Bool := C.outs.map (Src.val (run x C.nIn C.gates))

/-- Each gate reads only earlier wires. -/
def Src.below (n : ℕ) : Src → Bool
  | .wire i => decide (i < n)
  | _ => true

def Gate.below (n : ℕ) : Gate → Bool
  | .and a b => a.below n && b.below n
  | .xor a b => a.below n && b.below n
  | .not a => a.below n

def gatesWF : ℕ → List Gate → Bool
  | _, [] => true
  | w, g :: gs => g.below w && gatesWF (w + 1) gs

/-! ## The derivation -/

/-- A linear form: a mask of committed columns, and the constant. -/
abbrev Form := ℕ × Bool

/-- The depth of the wire map: wires below `2 ^ 20`. -/
def depth : ℕ := 20

/-- The derivation's state: each wire's form, the next wire, the next AND's column, and the AND
rows so far, latest first. -/
structure St where
  forms : PM Form
  w : ℕ
  col : ℕ
  rows : List (Form × Form)

def srcForm (t : PM Form) : Src → Form
  | .zero => (0, false)
  | .one => (0, true)
  | .wire i => (PM.get depth t i).getD (0, false)

def St.step (s : St) : Gate → St
  | .and a b => ⟨PM.set depth s.forms s.w (2 ^ s.col, false), s.w + 1, s.col + 1,
      (srcForm s.forms a, srcForm s.forms b) :: s.rows⟩
  | .xor a b => ⟨PM.set depth s.forms s.w ((srcForm s.forms a).1 ^^^ (srcForm s.forms b).1,
      Bool.xor (srcForm s.forms a).2 (srcForm s.forms b).2), s.w + 1, s.col, s.rows⟩
  | .not a => ⟨PM.set depth s.forms s.w ((srcForm s.forms a).1, !(srcForm s.forms a).2), s.w + 1,
      s.col, s.rows⟩

def St.steps (s : St) : List Gate → St
  | [] => s
  | g :: gs => (s.step g).steps gs

/-- A layout: each input's column, where the ANDs start, and the output region's rows by
column (`outAt r`: the output a copy row at column `r` carries, if any), `outLen` columns long. -/
structure Lay where
  inCol : ℕ → ℕ
  andBase : ℕ
  outLen : ℕ
  outAt : ℕ → Option ℕ

/-- The input forms: input `k` is column `inCol k`. -/
def inputs (L : Lay) : ℕ → PM Form
  | 0 => .leaf
  | k + 1 => PM.set depth (inputs L k) k (2 ^ L.inCol k, false)

def start (C : Circ) (L : Lay) : St := ⟨inputs L C.nIn, C.nIn, L.andBase, []⟩

def final (C : Circ) (L : Lay) : St := (start C L).steps C.gates

def nAnd (C : Circ) : ℕ := (C.gates.filter fun g => match g with | .and _ _ => true | _ => false).length

/-- The constant's column: after the ANDs and the output region. -/
def constCol (C : Circ) (L : Lay) : ℕ := L.andBase + nAnd C + L.outLen

/-- A form as a row's mask: the constant is the constant's column. -/
def mask (k : ℕ) (f : Form) : ℕ := f.1 ^^^ (if f.2 then 2 ^ k else 0)

/-- The input region's row at column `r`: an input's self row, or a forced-zero row. -/
def inRow (C : Circ) (L : Lay) (r : ℕ) : ℕ × ℕ :=
  if (List.range C.nIn).any (fun k => L.inCol k == r) then (2 ^ r, 2 ^ r) else (0, 0)

/-- The output region's row at column `r`: a copy row `form · 1`, or a forced-zero row. -/
def outRow (C : Circ) (L : Lay) (S : St) (r : ℕ) : ℕ × ℕ :=
  match L.outAt r with
  | some o => (mask (constCol C L) (srcForm S.forms (C.outs.getD o .zero)), 2 ^ constCol C L)
  | none => (0, 0)

/-- **The rows**, as masks: the input region, the ANDs in order, the output region, the constant. -/
def rows (C : Circ) (L : Lay) : List (ℕ × ℕ) :=
  let S := final C L
  let k := constCol C L
  (List.range L.andBase).map (inRow C L) ++
    S.rows.reverse.map (fun p => (mask k p.1, mask k p.2)) ++
    (List.range L.outLen).map (fun i => outRow C L S (L.andBase + nAnd C + i)) ++
    [(2 ^ k, 2 ^ k)]

/-- Rows are satisfied by column values `z`: row `r` is `z r = (A·z)(B·z)`. -/
def Sat (R : List (ℕ × ℕ)) (z : ℕ → Bool) : Prop :=
  ∀ r p, R[r]? = some p → z r = (psum z p.1 R.length && psum z p.2 R.length)

/-! ## The circuit's values -/

theorem run_lt (v : ℕ → Bool) (w : ℕ) (gs : List Gate) (j : ℕ) (hj : j < w) : run v w gs j = v j := by
  induction gs generalizing v w with
  | nil => rfl
  | cons g gs ih =>
    simp only [run]
    rw [ih _ _ (by omega)]
    simp [show j ≠ w by omega]

theorem Src.val_congr {v v' : ℕ → Bool} {n : ℕ} (s : Src) (hs : s.below n = true)
    (h : ∀ j < n, v j = v' j) : s.val v = s.val v' := by
  cases s with
  | zero => rfl
  | one => rfl
  | wire i => simp only [below, decide_eq_true_eq] at hs; exact h i hs

theorem Gate.val_congr {v v' : ℕ → Bool} {n : ℕ} (g : Gate) (hg : g.below n = true)
    (h : ∀ j < n, v j = v' j) : g.val v = g.val v' := by
  cases g with
  | and a b =>
    simp only [below, Bool.and_eq_true] at hg
    simp only [val, Src.val_congr a hg.1 h, Src.val_congr b hg.2 h]
  | xor a b =>
    simp only [below, Bool.and_eq_true] at hg
    simp only [val, Src.val_congr a hg.1 h, Src.val_congr b hg.2 h]
  | not a => simp only [below] at hg; simp only [val, Src.val_congr a hg h]

/-- With every gate reading earlier wires, wire `w + k` holds gate `k` on the final values. -/
theorem run_gate (v : ℕ → Bool) (w : ℕ) (gs : List Gate) (hwf : gatesWF w gs = true) (k : ℕ)
    (hk : k < gs.length) : run v w gs (w + k) = gs[k].val (run v w gs) := by
  induction gs generalizing v w k with
  | nil => simp at hk
  | cons g gs ih =>
    simp only [gatesWF, Bool.and_eq_true] at hwf
    cases k with
    | zero =>
      simp only [run, Nat.add_zero, List.getElem_cons_zero]
      rw [run_lt _ _ _ _ (by omega)]
      simp only [↓reduceIte]
      exact Gate.val_congr g hwf.1 fun j hj => by
        rw [run_lt _ _ _ _ (by omega)]; simp [show j ≠ w by omega]
    | succ k =>
      simp only [run, List.getElem_cons_succ]
      rw [show w + (k + 1) = (w + 1) + k by omega]
      exact ih _ _ hwf.2 k (by simpa using hk)

/-! ## Masks and forms -/

/-- A form's value: the parity of its columns, plus its constant. -/
def fval (z : ℕ → Bool) (n : ℕ) (f : Form) : Bool := Bool.xor (psum z f.1 n) f.2

theorem psum_mask (z : ℕ → Bool) (n k : ℕ) (f : Form) (hk : k < n) (h1 : z k = true) :
    psum z (mask k f) n = fval z n f := by
  unfold mask fval
  rw [psum_xor]
  cases f.2
  · simp [psum_zero]
  · simp [psum_two_pow, hk, h1]

theorem fval_srcForm_const (z : ℕ → Bool) (n : ℕ) (t : PM Form) :
    fval z n (srcForm t .zero) = false ∧ fval z n (srcForm t .one) = true := by
  simp [fval, srcForm, psum_zero]

/-! ## The derivation's state -/

def isAnd : Gate → Bool
  | .and _ _ => true
  | _ => false

theorem nAnd_eq (C : Circ) : nAnd C = (C.gates.filter isAnd).length := by
  unfold nAnd; congr 1

theorem steps_w (s : St) (gs : List Gate) : (s.steps gs).w = s.w + gs.length := by
  induction gs generalizing s with
  | nil => simp [St.steps]
  | cons g gs ih =>
    simp only [St.steps, ih, List.length_cons]
    cases g <;> (simp only [St.step]; try omega)

theorem steps_rows (s : St) (gs : List Gate) :
    ∃ l, (s.steps gs).rows = l ++ s.rows ∧ l.length = (gs.filter isAnd).length := by
  induction gs generalizing s with
  | nil => exact ⟨[], by simp [St.steps]⟩
  | cons g gs ih =>
    obtain ⟨l, hl, hn⟩ := ih (s.step g)
    cases g with
    | and a b =>
      refine ⟨l ++ [(srcForm s.forms a, srcForm s.forms b)], ?_, ?_⟩
      · show ((s.step (.and a b)).steps gs).rows = _
        rw [hl]; simp [St.step]
      · simp [hn, List.filter_cons, isAnd]
    | xor a b => exact ⟨l, by simpa [St.steps, St.step] using hl, by simpa [List.filter_cons, isAnd] using hn⟩
    | not a => exact ⟨l, by simpa [St.steps, St.step] using hl, by simpa [List.filter_cons, isAnd] using hn⟩

theorem final_rows_length (C : Circ) (L : Lay) : (final C L).rows.length = nAnd C := by
  obtain ⟨l, hl, hn⟩ := steps_rows (start C L) C.gates
  rw [final, hl, nAnd_eq]; simp [start, hn]

theorem rows_length (C : Circ) (L : Lay) : (rows C L).length = constCol C L + 1 := by
  simp [rows, final_rows_length, constCol]; omega

theorem rows_and (C : Circ) (L : Lay) (j : ℕ) (hj : j < (final C L).rows.length) :
    (rows C L)[L.andBase + j]? = some (mask (constCol C L) ((final C L).rows.reverse[j]'(by simpa using hj)).1,
      mask (constCol C L) ((final C L).rows.reverse[j]'(by simpa using hj)).2) := by
  have h1 : ((List.range L.andBase).map (inRow C L)).length = L.andBase := by simp
  have h2 : ((final C L).rows.reverse.map
      (fun p => (mask (constCol C L) p.1, mask (constCol C L) p.2))).length = (final C L).rows.length := by simp
  have i1 : L.andBase + j < ((List.range L.andBase).map (inRow C L) ++ (final C L).rows.reverse.map
      (fun p => (mask (constCol C L) p.1, mask (constCol C L) p.2))).length := by
    rw [List.length_append, h1, h2]; omega
  have i2 : L.andBase + j < ((List.range L.andBase).map (inRow C L) ++ (final C L).rows.reverse.map
      (fun p => (mask (constCol C L) p.1, mask (constCol C L) p.2)) ++
      (List.range L.outLen).map (fun i => outRow C L (final C L) (L.andBase + nAnd C + i))).length := by
    rw [List.length_append]; omega
  have i3 : ((List.range L.andBase).map (inRow C L)).length ≤ L.andBase + j := by rw [h1]; omega
  unfold rows
  simp only
  rw [List.getElem?_append_left i2, List.getElem?_append_left i1, List.getElem?_append_right i3, h1,
    Nat.add_sub_cancel_left, List.getElem?_map, List.getElem?_eq_getElem (by simpa using hj)]
  rfl

theorem rows_out (C : Circ) (L : Lay) (i : ℕ) (hi : i < L.outLen) :
    (rows C L)[L.andBase + nAnd C + i]? = some (outRow C L (final C L) (L.andBase + nAnd C + i)) := by
  have h1 : ((List.range L.andBase).map (inRow C L)).length = L.andBase := by simp
  have h2 : ((final C L).rows.reverse.map
      (fun p => (mask (constCol C L) p.1, mask (constCol C L) p.2))).length = nAnd C := by
    simp [final_rows_length]
  have h3 : ((List.range L.outLen).map (fun i => outRow C L (final C L) (L.andBase + nAnd C + i))).length =
      L.outLen := by simp
  have hAB : ((List.range L.andBase).map (inRow C L) ++ (final C L).rows.reverse.map
      (fun p => (mask (constCol C L) p.1, mask (constCol C L) p.2))).length = L.andBase + nAnd C := by
    rw [List.length_append, h1, h2]
  have i1 : L.andBase + nAnd C + i < ((List.range L.andBase).map (inRow C L) ++ (final C L).rows.reverse.map
      (fun p => (mask (constCol C L) p.1, mask (constCol C L) p.2)) ++
      (List.range L.outLen).map (fun i => outRow C L (final C L) (L.andBase + nAnd C + i))).length := by
    rw [List.length_append, hAB, h3]; omega
  have i2 : ((List.range L.andBase).map (inRow C L) ++ (final C L).rows.reverse.map
      (fun p => (mask (constCol C L) p.1, mask (constCol C L) p.2))).length ≤ L.andBase + nAnd C + i := by
    rw [hAB]; omega
  unfold rows
  simp only
  rw [List.getElem?_append_left i1, List.getElem?_append_right i2, hAB,
    show L.andBase + nAnd C + i - (L.andBase + nAnd C) = i by omega, List.getElem?_map,
    List.getElem?_range hi, Option.map_some]

theorem rows_const (C : Circ) (L : Lay) :
    (rows C L)[constCol C L]? = some (2 ^ constCol C L, 2 ^ constCol C L) := by
  have h1 : ((List.range L.andBase).map (inRow C L)).length = L.andBase := by simp
  have h2 : ((final C L).rows.reverse.map
      (fun p => (mask (constCol C L) p.1, mask (constCol C L) p.2))).length = nAnd C := by
    simp [final_rows_length]
  have h3 : ((List.range L.outLen).map (fun i => outRow C L (final C L) (L.andBase + nAnd C + i))).length =
      L.outLen := by simp
  have hABC : ((List.range L.andBase).map (inRow C L) ++ (final C L).rows.reverse.map
      (fun p => (mask (constCol C L) p.1, mask (constCol C L) p.2)) ++
      (List.range L.outLen).map (fun i => outRow C L (final C L) (L.andBase + nAnd C + i))).length =
      constCol C L := by
    rw [List.length_append, List.length_append, h1, h2, h3]; rfl
  unfold rows
  simp only
  rw [List.getElem?_append_right (by rw [hABC]), hABC, Nat.sub_self]
  rfl

theorem rows_and' (C : Circ) (L : Lay) (j : ℕ) (p : Form × Form) (hp : (final C L).rows.reverse[j]? = some p) :
    (rows C L)[L.andBase + j]? = some (mask (constCol C L) p.1, mask (constCol C L) p.2) := by
  obtain ⟨hj, hjp⟩ := List.getElem?_eq_some_iff.mp hp
  rw [rows_and C L j (by simpa using hj)]
  simp only [hjp]

theorem rev_index {α : Type} (l r : List α) (x : α) : (l ++ x :: r).reverse[r.length]? = some x := by
  rw [List.reverse_append, List.reverse_cons, List.append_assoc, List.getElem?_append_right (by simp)]
  simp

/-! ## Soundness: satisfied rows compute the circuit -/

theorem get_inputs (L : Lay) (n k : ℕ) (hk : k < n) (hn : n ≤ 2 ^ depth) :
    PM.get depth (inputs L n) k = some (2 ^ L.inCol k, false) := by
  induction n with
  | zero => omega
  | succ n ih =>
    simp only [inputs]
    rw [PM.get_set _ _ _ _ _ (by omega) (by omega)]
    by_cases h : n = k
    · subst h; simp
    · rw [ite_eq_right h]; exact ih (by omega) (by omega)

theorem fval_srcForm (z : ℕ → Bool) {t : PM Form} {n W : ℕ} {V : ℕ → Bool}
    (ht : ∀ i < n, ∃ f, PM.get depth t i = some f ∧ fval z W f = V i)
    (a : Src) (ha : a.below n = true) : fval z W (srcForm t a) = a.val V := by
  cases a with
  | zero => simp [srcForm, fval, psum_zero, Src.val]
  | one => simp [srcForm, fval, psum_zero, Src.val]
  | wire i =>
    simp only [Src.below, decide_eq_true_eq] at ha
    obtain ⟨f, hf, hv⟩ := ht i ha
    simp [srcForm, hf, hv, Src.val]

theorem inv_steps (C : Circ) (L : Lay) (z : ℕ → Bool) (hz : Sat (rows C L) z)
    (h1 : z (constCol C L) = true) (V : ℕ → Bool) (gs : List Gate) :
    ∀ (s : St), s.steps gs = final C L → gatesWF s.w gs = true → s.w + gs.length ≤ 2 ^ depth →
      s.col = L.andBase + s.rows.length →
      (∀ j (hj : j < gs.length), V (s.w + j) = gs[j].val V) →
      (∀ i < s.w, ∃ f, PM.get depth s.forms i = some f ∧ fval z (rows C L).length f = V i) →
      ∀ i < s.w + gs.length, ∃ f, PM.get depth (final C L).forms i = some f ∧
        fval z (rows C L).length f = V i := by
  have hW : constCol C L < (rows C L).length := by rw [rows_length]; omega
  induction gs with
  | nil => intro s hs _ _ _ _ hf; rw [← hs]; simpa [St.steps] using hf
  | cons g gs ih =>
    intro s hs hwf hb hcol hV hf
    simp only [gatesWF, Bool.and_eq_true] at hwf
    have hs' : (s.step g).steps gs = final C L := hs
    have hw' : (s.step g).w = s.w + 1 := by cases g <;> rfl
    have hsrc := fun (a : Src) (ha : a.below s.w = true) => fval_srcForm z hf a ha
    -- the new wire's form holds its value
    have hnew : ∀ f, PM.get depth (s.step g).forms s.w = some f → fval z (rows C L).length f = V s.w := by
      have hv0 := hV 0 (by simp)
      simp only [Nat.add_zero, List.getElem_cons_zero] at hv0
      intro f hfg
      cases g with
      | and a b =>
        simp only [St.step] at hfg
        rw [PM.get_set _ _ _ _ _ (by simp at hb; omega) (by simp at hb; omega), ite_eq_left rfl] at hfg
        cases hfg
        simp only [Gate.below, Bool.and_eq_true] at hwf
        obtain ⟨l, hl, _⟩ := steps_rows (s.step (.and a b)) gs
        rw [hs'] at hl
        have hrev : (final C L).rows.reverse[s.rows.length]? =
            some (srcForm s.forms a, srcForm s.forms b) := by
          rw [hl]; exact rev_index l s.rows (srcForm s.forms a, srcForm s.forms b)
        have hrow := rows_and' C L _ _ hrev
        have hsat := hz _ _ hrow
        have hlt : L.andBase + s.rows.length < (rows C L).length := by
          have := List.getElem?_eq_some_iff.mp hrow; exact this.1
        have ea := psum_mask z _ _ (srcForm s.forms a) hW h1
        have eb := psum_mask z _ _ (srcForm s.forms b) hW h1
        rw [hv0, Gate.val, ← hsrc a hwf.1.1, ← hsrc b hwf.1.2, ← ea, ← eb]
        simp only at hsat
        rw [← hsat, ← hcol]
        simp [fval, psum_two_pow, hcol, hlt]
      | xor a b =>
        simp only [St.step] at hfg
        rw [PM.get_set _ _ _ _ _ (by simp at hb; omega) (by simp at hb; omega), ite_eq_left rfl] at hfg
        cases hfg
        simp only [Gate.below, Bool.and_eq_true] at hwf
        rw [hv0, Gate.val, ← hsrc a hwf.1.1, ← hsrc b hwf.1.2]
        simp only [fval, psum_xor]
        cases psum z (srcForm s.forms a).1 (rows C L).length <;>
          cases psum z (srcForm s.forms b).1 (rows C L).length <;>
          cases (srcForm s.forms a).2 <;> cases (srcForm s.forms b).2 <;> rfl
      | not a =>
        simp only [St.step] at hfg
        rw [PM.get_set _ _ _ _ _ (by simp at hb; omega) (by simp at hb; omega), ite_eq_left rfl] at hfg
        cases hfg
        simp only [Gate.below] at hwf
        rw [hv0, Gate.val, ← hsrc a hwf.1]
        simp only [fval]
        cases psum z (srcForm s.forms a).1 (rows C L).length <;> cases (srcForm s.forms a).2 <;> rfl
    have hf' : ∀ i < (s.step g).w, ∃ f, PM.get depth (s.step g).forms i = some f ∧
        fval z (rows C L).length f = V i := by
      intro i hi
      rw [hw'] at hi
      by_cases hie : i = s.w
      · subst hie
        have hget : ∃ f, PM.get depth (s.step g).forms s.w = some f := by
          cases g <;> simp only [St.step] <;>
            rw [PM.get_set _ _ _ _ _ (by simp at hb; omega) (by simp at hb; omega), ite_eq_left rfl] <;> simp
        obtain ⟨f, hfg⟩ := hget
        exact ⟨f, hfg, hnew f hfg⟩
      · obtain ⟨f, hfg, hv⟩ := hf i (by omega)
        refine ⟨f, ?_, hv⟩
        have hg : PM.get depth (s.step g).forms i = PM.get depth s.forms i := by
          cases g <;> simp only [St.step] <;>
            rw [PM.get_set _ _ _ _ _ (by simp at hb; omega) (by simp at hb; omega), ite_eq_right (Ne.symm hie)]
        rw [hg, hfg]
    have hcol' : (s.step g).col = L.andBase + (s.step g).rows.length := by
      cases g <;> (simp [St.step, hcol]; try omega)
    have hV' : ∀ j (hj : j < gs.length), V ((s.step g).w + j) = gs[j].val V := by
      intro j hj
      have := hV (j + 1) (by simp; omega)
      rw [hw', show s.w + 1 + j = s.w + (j + 1) by omega]
      simpa using this
    intro i hi
    exact ih (s.step g) hs' (by rw [hw']; exact hwf.2) (by rw [hw']; simp at hb; omega) hcol' hV' hf' i
      (by rw [hw']; simp at hi; omega)

/-- **Satisfied rows compute the circuit.** If `z` satisfies the derived rows with the constant 1,
each copy row in the output region holds the circuit's output on the input columns. -/
theorem rows_sound (C : Circ) (L : Lay) (hwf : gatesWF C.nIn C.gates = true)
    (hb : C.nIn + C.gates.length ≤ 2 ^ depth) (hin : ∀ k < C.nIn, L.inCol k < L.andBase)
    (houts : ∀ o ∈ C.outs, o.below (C.nIn + C.gates.length) = true)
    (z : ℕ → Bool) (hz : Sat (rows C L) z) (h1 : z (constCol C L) = true)
    (i : ℕ) (hi : i < L.outLen) (o : ℕ) (ho : L.outAt (L.andBase + nAnd C + i) = some o)
    (hol : o < C.outs.length) :
    z (L.andBase + nAnd C + i) = (C.eval fun k => z (L.inCol k))[o]'(by simpa [Circ.eval] using hol) := by
  set x : ℕ → Bool := fun k => z (L.inCol k)
  set V := run x C.nIn C.gates
  have hW : constCol C L < (rows C L).length := by rw [rows_length]; omega
  have hV : ∀ j (hj : j < C.gates.length), V (C.nIn + j) = C.gates[j].val V :=
    fun j hj => run_gate x C.nIn C.gates hwf j hj
  have hf0 : ∀ i < C.nIn, ∃ f, PM.get depth (inputs L C.nIn) i = some f ∧
      fval z (rows C L).length f = V i := by
    intro k hk
    refine ⟨_, get_inputs L C.nIn k hk (by omega), ?_⟩
    have hlt : L.inCol k < (rows C L).length := by
      have := hin k hk; rw [rows_length]; unfold constCol; omega
    simp only [fval, psum_two_pow, hlt, decide_true, Bool.true_and, Bool.xor_false]
    show z (L.inCol k) = V k
    rw [show V k = x k from run_lt x C.nIn C.gates k hk]
  have hall := inv_steps C L z hz h1 V C.gates (start C L) rfl hwf hb (by simp [start]) hV hf0
  have hrow := rows_out C L i hi
  rw [outRow, ho] at hrow
  have hsat := hz _ _ hrow
  have hmem : C.outs[o] ∈ C.outs := List.getElem_mem hol
  rw [hsat, psum_mask z _ _ _ hW h1, psum_two_pow]
  simp only [hW, decide_true, h1, Bool.and_true]
  rw [List.getD_eq_getElem _ _ hol, fval_srcForm z hall _ (houts _ hmem)]
  simp [Circ.eval, V, x]

end FlockSoundness.GateRows
