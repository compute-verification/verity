/-
`Flock.Partition`, the partition of record of a Boolean Program: the word Program's partition lifted by σ, the map from each
word Definition to its Boolean version. This module checks the lift's structure on the two descriptors (`Program.decode` of
each), as `verity_vllm.program.boolean.sigma_check` checks it on Python objects, which stays the mirror (the
`verity/core/service/sigma` difftest holds the two together).

The Boolean descriptor carries the lift as data, a top-level field under its program SHA-512:

    "word": {"program_sha512": "<the word descriptor's SHA-512, hex>",
             "sigma": {"<word definition id>": "<Boolean definition id>", ...}}

The check (`check`), each refusal named by its code (`codes`) with the fix in its message:

* `word-link`: the field is that object, and its digest is the word descriptor's.
* `sigma-table`, `sigma-signature`: each entry maps a word Definition to a Boolean one that declares the word one's signature on
  bits (`bitTy` of each parameter and of the return; names aside).
* `inputs`: the Boolean root's input nodes hold bits (`Input1_v1`, `RegisteredInput1_v1`), and the word root's input leaves,
  each its width in bits, are those bits in order, bound alike (per run or at registration).
* `signature`: the Boolean root returns the word root's return type on bits.
* `node-count`: the roots have as many Calls (`Prog.calls`).
* per Call, in order: `sigma-table` (its Boolean Definition is σ of its word Definition), `sigma-not-function` (an earlier Call
  of the same word Definition called σ's image, this one another), `node-form` (form, a word primitive's Call being a Call of
  its Boolean composite; members, axes, scan inputs, arity and output type on bits), `operand-type` and `operand-runs` (each
  operand, in its type on bits, is the word operand's runs moved onto the bits they occupy).
* `return-runs`: the Boolean root's return is the word root's, moved.

"Moved": a word reference's leaf `j` of a node's output is bit `bitOff out j` of that output, the prefix sums of the output's
leaf widths (`Program.Ty.widths`; `bitOff_eq`), and the root's input nodes are one stream of bits in node order. A sequence's runs (`partRuns`)
are compared once joined (`merged`), so a run the Boolean side splits differently is the same run.

Cost: linear in the two descriptors' parts and in the runs a strided part spans (its rows, or its leaves when its inner stride
is not 1, as `refs.runs` walks it), each leaf's bit found in its type's depth and fields. Never over the unrolled circuit: no
gate, and no callee's body.

What carries across (`reads_lift`, `edges_lift`, `acyclic_lift`): a Call reads another in the Boolean root exactly when the
word Call reads the word one, so the units of any partition of the root's Calls read each other in the same graph, and the
partition invariant's `unit-cycle` refusal on the word partition is the one on its lift. What does not: a `Q_call` cut inside a
leaf word Call (its gates are word primitives, those of σ(f) are bits) says nothing about the Boolean Call; inside a leaf
Boolean Definition its own cut applies.

What this does not check: that σ(f) computes f. That stays a Python conformance fact (each Boolean version is tested against
its word Definition, completeness under Boolean-first), because credited work reads Boolean gate counts only.
-/
import Lean.Data.Json
import Std.Data.HashMap
import Std.Data.HashSet
import Flock.Program

namespace Flock.Sigma

open Lean (Json)
open Flock.Program

/-! ### Types on bits -/

private theorem sizeOf_lt_of_mem {fs : Array Ty} {t : Ty} (h : t ∈ fs) : sizeOf t < 1 + sizeOf fs := by
  have := Array.sizeOf_lt_of_mem h
  omega

private theorem sizeOf_snd_lt_of_mem {fs : Array (String × Ty)} {f : String × Ty} (h : f ∈ fs) :
    sizeOf f.2 < 1 + sizeOf fs := by
  have := Array.sizeOf_lt_of_mem h
  obtain ⟨k, t⟩ := f
  simp only [Prod.mk.sizeOf_spec] at this ⊢
  omega

/-- `boolean.bit_type`: each `Value<w>` becomes `Array<w, Value<1>>`, and `Value<1>` stays a bit. -/
def bitTy : Ty → Ty
  | .value w => if w = 1 then .value 1 else .array w (.value 1)
  | .array n e => .array n (bitTy e)
  | .tuple fs => .tuple (fs.attach.map fun f => bitTy f.1)
  | .record fs => .record (fs.attach.map fun f => (f.1.1, bitTy f.1.2))
termination_by t => t
decreasing_by
  all_goals simp only [Ty.array.sizeOf_spec, Ty.tuple.sizeOf_spec, Ty.record.sizeOf_spec]
  · omega
  · exact sizeOf_lt_of_mem f.2
  · exact sizeOf_snd_lt_of_mem f.2

/-- A type's bits: its leaf widths' sum. -/
def bits : Ty → Nat
  | .value w => w
  | .array n e => n * bits e
  | .tuple fs => fs.attach.foldl (fun a f => a + bits f.1) 0
  | .record fs => fs.attach.foldl (fun a f => a + bits f.1.2) 0
termination_by t => t
decreasing_by
  all_goals simp only [Ty.array.sizeOf_spec, Ty.tuple.sizeOf_spec, Ty.record.sizeOf_spec]
  · omega
  · exact sizeOf_lt_of_mem f.2
  · exact sizeOf_snd_lt_of_mem f.2

/-- One field's step of `bitOff` over a Tuple's or Record's fields: `(found, first leaf, first bit)` so far. -/
@[inline] private def fieldStep (j : Nat) (leaves bits : Nat) (off : Nat → Nat) :
    Option Nat × Nat × Nat → Option Nat × Nat × Nat
  | (some r, a, b) => (some r, a, b)
  | (none, a, b) => (if j < a + leaves then some (b + off (j - a)) else none, a + leaves, b + bits)

/-- `boolean.bit_offsets`: word leaf `j` of `t` (`0 ≤ j ≤ leaves`) is bit `bitOff t j` of `bitTy t`, the sum of the first `j`
leaf widths (`Ty.widths`), found through the type: an Array's element by division, a field by its leaf range. -/
def bitOff : Ty → Nat → Nat
  | .value w, j => if j = 0 then 0 else w
  | .array _ e, j => if e.leaves = 0 then 0 else j / e.leaves * bits e + bitOff e (j % e.leaves)
  | .tuple fs, j =>
    let (r, _, b) := fs.attach.foldl (fun acc f => fieldStep j f.1.leaves (bits f.1) (bitOff f.1) acc) (none, 0, 0)
    r.getD b
  | .record fs, j =>
    let (r, _, b) := fs.attach.foldl (fun acc f => fieldStep j f.1.2.leaves (bits f.1.2) (bitOff f.1.2) acc) (none, 0, 0)
    r.getD b
termination_by t => t
decreasing_by
  all_goals simp only [Ty.array.sizeOf_spec, Ty.tuple.sizeOf_spec, Ty.record.sizeOf_spec]
  · omega
  · exact sizeOf_lt_of_mem f.2
  · exact sizeOf_snd_lt_of_mem f.2

/-! ### `bitOff` is the prefix sums of the leaf widths -/

private theorem foldl_append_toList {β : Type _} (l : List β) (g : β → Array Nat) (acc : Array Nat) :
    (l.foldl (fun a t => a ++ g t) acc).toList = acc.toList ++ l.flatMap fun t => (g t).toList := by
  induction l generalizing acc with
  | nil => simp
  | cons x l ih => simp [List.foldl, ih, Array.toList_append, List.flatMap_cons, List.append_assoc]

private theorem foldl_add {β : Type _} (l : List β) (g : β → Nat) (acc : Nat) :
    l.foldl (fun a t => a + g t) acc = acc + (l.map g).sum := by
  induction l generalizing acc with
  | nil => simp
  | cons x l ih => simp [List.foldl, ih]; omega

private theorem sum_flatMap {β : Type _} (l : List β) (g : β → List Nat) :
    (l.flatMap g).sum = (l.map fun x => (g x).sum).sum := by
  induction l with
  | nil => simp
  | cons x l ih => simp [List.flatMap_cons, List.sum_append, ih]

private theorem length_flatMap' {β : Type _} (l : List β) (g : β → List Nat) :
    (l.flatMap g).length = (l.map fun x => (g x).length).sum := by
  induction l with
  | nil => simp
  | cons x l ih => simp [List.flatMap_cons, ih]

private theorem sum_map_const {α : Type _} (l : List α) (c : Nat) : (l.map fun _ => c).sum = l.length * c := by
  induction l with
  | nil => simp
  | cons x l ih => simp [ih, Nat.succ_mul]; omega

private theorem copies_take {α : Type _} (l : List α) (W : List Nat) (hW : 0 < W.length) (j : Nat)
    (hj : j ≤ l.length * W.length) :
    ((l.flatMap fun _ => W).take j).sum = j / W.length * W.sum + (W.take (j % W.length)).sum := by
  induction l generalizing j with
  | nil => simp at hj; subst hj; simp
  | cons x l ih =>
    rw [List.flatMap_cons, List.take_append, List.sum_append]
    by_cases h : j < W.length
    · have : j - W.length = 0 := by omega
      rw [this, Nat.div_eq_of_lt h, Nat.mod_eq_of_lt h]; simp
    · have hle : W.length ≤ j := by omega
      rw [List.take_of_length_le hle, ih (j - W.length) (by simp [Nat.succ_mul] at hj; omega),
        Nat.div_eq_sub_div hW hle, Nat.mod_eq_sub_mod hle, Nat.add_mul, Nat.one_mul]
      omega

private theorem arrayWidths (n : Nat) (e : Ty) :
    (Ty.widths (.array n e)).toList = (List.range n).flatMap fun _ => (Ty.widths e).toList := by
  simp only [Ty.widths]
  rw [← Array.foldl_toList, foldl_append_toList (g := fun _ => Ty.widths e), Array.toList_range]
  simp

private theorem tupleWidths (fs : Array Ty) :
    (Ty.widths (.tuple fs)).toList = fs.toList.flatMap fun t => (Ty.widths t).toList := by
  simp only [Ty.widths]
  rw [← Array.foldl_toList, foldl_append_toList (g := Ty.widths)]
  simp

private theorem recordWidths (fs : Array (String × Ty)) :
    (Ty.widths (.record fs)).toList = fs.toList.flatMap fun t => (Ty.widths t.2).toList := by
  simp only [Ty.widths]
  rw [← Array.foldl_toList, foldl_append_toList fs.toList (fun t : String × Ty => Ty.widths t.2) #[]]
  simp

theorem widths_length : (t : Ty) → (Ty.widths t).toList.length = t.leaves
  | .value w => by simp [Ty.widths, Ty.leaves]
  | .array n e => by
    rw [arrayWidths, length_flatMap']
    have ih := widths_length e
    simp only [ih, sum_map_const, List.length_range, Ty.leaves]
  | .tuple fs => by
    have ih : ∀ f ∈ fs.toList, (Ty.widths f).toList.length = f.leaves := fun f hf => widths_length f
    rw [tupleWidths, length_flatMap', List.map_congr_left ih]
    simp only [Ty.leaves]
    rw [← Array.foldl_toList, foldl_add]; simp
  | .record fs => by
    have ih : ∀ f ∈ fs.toList, (Ty.widths f.2).toList.length = f.2.leaves := fun f hf => widths_length f.2
    rw [recordWidths, length_flatMap', List.map_congr_left ih]
    simp only [Ty.leaves]
    rw [← Array.foldl_toList, foldl_add]; simp
termination_by t => t
decreasing_by
  all_goals simp only [Ty.array.sizeOf_spec, Ty.tuple.sizeOf_spec, Ty.record.sizeOf_spec]
  · omega
  · exact sizeOf_lt_of_mem (Array.mem_def.mpr hf)
  · exact sizeOf_snd_lt_of_mem (Array.mem_def.mpr hf)


theorem widths_sum : (t : Ty) → (Ty.widths t).toList.sum = bits t
  | .value w => by simp [Ty.widths, bits]
  | .array n e => by
    rw [arrayWidths, sum_flatMap, sum_map_const, List.length_range, widths_sum e]
    simp [bits]
  | .tuple fs => by
    have ih : ∀ f ∈ fs.toList, (Ty.widths f).toList.sum = bits f := fun f hf => widths_sum f
    rw [tupleWidths, sum_flatMap, List.map_congr_left ih]
    have e : fs.attach.foldl (fun a f => a + bits f.val) 0 = fs.toList.foldl (fun a f => a + bits f) 0 :=
      (Array.foldl_attach (f := fun a (f : Ty) => a + bits f)).trans (Array.foldl_toList _).symm
    simp only [bits]
    rw [e, foldl_add]; simp
  | .record fs => by
    have ih : ∀ f ∈ fs.toList, (Ty.widths f.2).toList.sum = bits f.2 := fun f hf => widths_sum f.2
    rw [recordWidths, sum_flatMap, List.map_congr_left ih]
    have e : fs.attach.foldl (fun a f => a + bits f.val.2) 0 = fs.toList.foldl (fun a f => a + bits f.2) 0 :=
      (Array.foldl_attach (f := fun a (f : String × Ty) => a + bits f.2)).trans (Array.foldl_toList _).symm
    simp only [bits]
    rw [e, foldl_add]; simp
termination_by t => t
decreasing_by
  all_goals simp only [Ty.array.sizeOf_spec, Ty.tuple.sizeOf_spec, Ty.record.sizeOf_spec]
  · omega
  · exact sizeOf_lt_of_mem (Array.mem_def.mpr hf)
  · exact sizeOf_snd_lt_of_mem (Array.mem_def.mpr hf)

private theorem fieldStep_none (j L B : Nat) (O : Nat → Nat) (a b : Nat) :
    fieldStep j L B O (none, a, b) = (if j < a + L then some (b + O (j - a)) else none, a + L, b + B) := rfl

private theorem flatMap_nil' {α : Type _} (l : List α) : (l.flatMap fun _ => ([] : List Nat)) = [] := by
  induction l with
  | nil => rfl
  | cons x l ih => simp [List.flatMap_cons, ih]

private theorem fieldStep_some (j : Nat) {β : Type _} (p : β → Ty) (l : List β) (r a b : Nat) :
    l.foldl (fun acc x => fieldStep j (p x).leaves (bits (p x)) (bitOff (p x)) acc) (some r, a, b) = (some r, a, b) := by
  induction l with
  | nil => rfl
  | cons x l ih => simp only [List.foldl, fieldStep]; exact ih

private theorem fieldFold (j : Nat) {β : Type _} (p : β → Ty) (l : List β)
    (ih : ∀ x ∈ l, ∀ k ≤ (p x).leaves, bitOff (p x) k = ((Ty.widths (p x)).toList.take k).sum)
    (a b : Nat) (h1 : a ≤ j) (h2 : j ≤ a + (l.map fun x => (p x).leaves).sum) (q : Option Nat × Nat × Nat)
    (hq : l.foldl (fun acc x => fieldStep j (p x).leaves (bits (p x)) (bitOff (p x)) acc) (none, a, b) = q) :
    q.1.getD q.2.2 = b + ((l.flatMap fun x => (Ty.widths (p x)).toList).take (j - a)).sum := by
  induction l generalizing a b with
  | nil =>
    subst hq
    have : j - a = 0 := by simp at h2; omega
    simp [this]
  | cons x l ihl =>
    have hx := widths_length (p x)
    rw [List.foldl_cons, fieldStep_none] at hq
    rw [List.flatMap_cons, List.take_append, List.sum_append]
    by_cases hj : j < a + (p x).leaves
    · simp only [hj, ↓reduceIte, fieldStep_some] at hq
      subst hq
      have : j - a - (Ty.widths (p x)).toList.length = 0 := by omega
      rw [this, ih x (List.mem_cons_self ..) (j - a) (by omega)]
      simp
    · simp only [hj, ↓reduceIte] at hq
      have hb := widths_sum (p x)
      rw [ihl (fun y hy => ih y (List.mem_cons_of_mem _ hy)) (a + (p x).leaves) (b + bits (p x)) (by omega)
        (by simp at h2 ⊢; omega) hq, List.take_of_length_le (l := (Ty.widths (p x)).toList) (i := j - a) (by omega), hb, hx]
      have : j - (a + (p x).leaves) = j - a - (p x).leaves := by omega
      rw [this]; omega

/-- `bitOff` is the prefix sums of the leaf widths: word leaf `j` of `t` is bit `Σ widths[0..j)` of `bitTy t`. -/
theorem bitOff_eq : (t : Ty) → (j : Nat) → j ≤ t.leaves → bitOff t j = ((Ty.widths t).toList.take j).sum
  | .value w, j, h => by
    simp only [Ty.leaves] at h
    rcases (by omega : j = 0 ∨ j = 1) with rfl | rfl <;> simp [bitOff, Ty.widths]
  | .array n e, j, h => by
    have hl := widths_length e
    rw [arrayWidths]
    simp only [bitOff]
    by_cases hL : e.leaves = 0
    · have : (Ty.widths e).toList = [] := List.eq_nil_of_length_eq_zero (by omega)
      simp [hL, this, flatMap_nil']
    · simp only [hL, ↓reduceIte]
      rw [copies_take _ _ (by omega) j (by simp only [Ty.leaves] at h; rw [List.length_range, hl]; exact h),
        hl, widths_sum e, bitOff_eq e (j % e.leaves) (Nat.le_of_lt (Nat.mod_lt _ (by omega)))]
  | .tuple fs, j, h => by
    have ih : ∀ f ∈ fs.toList, ∀ k ≤ f.leaves, bitOff f k = ((Ty.widths f).toList.take k).sum :=
      fun f hf k hk => bitOff_eq f k hk
    have e : fs.attach.foldl (fun acc f => fieldStep j f.val.leaves (bits f.val) (bitOff f.val) acc) (none, 0, 0)
        = fs.toList.foldl (fun acc f => fieldStep j f.leaves (bits f) (bitOff f) acc) (none, 0, 0) :=
      (Array.foldl_attach (f := fun acc (f : Ty) => fieldStep j f.leaves (bits f) (bitOff f) acc)).trans
        (Array.foldl_toList _).symm
    have hj : j ≤ 0 + (fs.toList.map fun x => x.leaves).sum := by
      simp only [Ty.leaves] at h; rw [← Array.foldl_toList, foldl_add] at h; simpa using h
    have := fieldFold j id fs.toList ih 0 0 (Nat.zero_le _) hj _ rfl
    simp only [bitOff, e]
    rw [tupleWidths]
    simpa using this
  | .record fs, j, h => by
    have ih : ∀ f ∈ fs.toList, ∀ k ≤ f.2.leaves, bitOff f.2 k = ((Ty.widths f.2).toList.take k).sum :=
      fun f hf k hk => bitOff_eq f.2 k hk
    have e : fs.attach.foldl (fun acc f => fieldStep j f.val.2.leaves (bits f.val.2) (bitOff f.val.2) acc) (none, 0, 0)
        = fs.toList.foldl (fun acc f => fieldStep j f.2.leaves (bits f.2) (bitOff f.2) acc) (none, 0, 0) :=
      (Array.foldl_attach (f := fun acc (f : String × Ty) => fieldStep j f.2.leaves (bits f.2) (bitOff f.2) acc)).trans
        (Array.foldl_toList _).symm
    have hj : j ≤ 0 + (fs.toList.map fun x => x.2.leaves).sum := by
      simp only [Ty.leaves] at h; rw [← Array.foldl_toList, foldl_add] at h; simpa using h
    have := fieldFold j Prod.snd fs.toList ih 0 0 (Nat.zero_le _) hj _ rfl
    simp only [bitOff, e]
    rw [recordWidths]
    simpa using this
termination_by t => t
decreasing_by
  all_goals simp only [Ty.array.sizeOf_spec, Ty.tuple.sizeOf_spec, Ty.record.sizeOf_spec]
  · omega
  · exact sizeOf_lt_of_mem (Array.mem_def.mpr hf)
  · exact sizeOf_snd_lt_of_mem (Array.mem_def.mpr hf)

/-! ### The root as runs of bits -/

/-- Where a reference's leaves live: the root's one stream of input bits, Call `c`'s output, or (never in a decoded root,
which has no parameters and whose references name its nodes) a parameter or no node. -/
inductive Src where
  | input
  | call (c : Nat)
  | param (i : Nat)
  | dangling
  deriving DecidableEq, Inhabited

/-- Bits `[lo, hi)` of a source. -/
structure Run where
  src : Src
  lo : Nat
  hi : Nat
  deriving DecidableEq, Inhabited

/-- A root body as the check reads it. -/
structure Side where
  /-- the root body's nodes -/
  nodes : Array Node
  /-- per root node: its source and the first bit of its output there -/
  place : Array (Src × Nat)
  /-- the root's Calls (`Prog.calls`), in node order -/
  calls : Array Nat
  /-- the input stream's runs `(bits, binding)`, adjacent runs of one binding joined -/
  inRuns : Array (Nat × String)
  inBits : Nat
  /-- the first input node whose leaves are not bits -/
  wide : Option Nat
  ret : Seq
  retType : Ty

private structure Walk where
  place : Array (Src × Nat) := #[]
  calls : Nat := 0
  inRuns : Array (Nat × String) := #[]
  inBits : Nat := 0
  wide : Option Nat := none

def Side.of (P : Prog) : Side :=
  let root := (P.defs[P.root]!).body
  let input := fun (x : Node) => let f := P.defs[x.fn]!; f.prim && isInputId f.key
  let w := root.nodes.foldl (fun (w : Walk) x =>
    if input x then
      let b := bits x.out
      let how := (P.defs[x.fn]!).binding.getD RUN
      { w with
        place := w.place.push (.input, w.inBits), inBits := w.inBits + b,
        inRuns := if b = 0 then w.inRuns else match w.inRuns.back? with
          | some (n, h) => if h == how then w.inRuns.pop.push (n + b, h) else w.inRuns.push (b, how)
          | none => w.inRuns.push (b, how),
        wide := if w.wide.isNone && !(bitTy x.out == x.out) then some w.place.size else w.wide }
    else { w with place := w.place.push (.call w.calls, 0), calls := w.calls + 1 }) {}
  { nodes := root.nodes, place := w.place, calls := (Array.range root.nodes.size).filter (fun k => !input root.nodes[k]!),
    inRuns := w.inRuns, inBits := w.inBits, wide := w.wide, ret := root.ret, retType := root.retType }

/-- The source of a reference into space `p`, collection `i`. -/
def Side.srcOf (S : Side) (p : Bool) (i : Nat) : Src :=
  if p then .param i else match S.place[i]? with
    | some (s, _) => s
    | none => .dangling

/-- The bit of leaf `j` of collection `i` in its source: on the word side (`word`) through the leaf widths of the node's output,
on the Boolean side, whose leaves are bits, the leaf itself. -/
def Side.bit (S : Side) (word : Bool) (p : Bool) (i j : Nat) : Nat :=
  if p then j else match S.place[i]?, S.nodes[i]? with
    | some (_, base), some x => base + (if word then bitOff x.out j else j)
    | _, _ => j

/-- A part's runs in sequence order (`refs._walk_runs`, a strided part's last row whole when it is short). -/
def partRuns (S : Side) (word : Bool) : Part → List Run
  | .affine p i b c => if c = 0 then [] else [⟨S.srcOf p i, S.bit word p i b, S.bit word p i (b + c)⟩]
  | .strided p i b inner os is c =>
    if c = 0 then []
    else if is = 1 ∨ inner = 1 then
      (List.range ((c + inner - 1) / inner)).map fun (r : Nat) =>
        let lo := (b + ((r : Nat) : Int) * os).toNat
        ⟨S.srcOf p i, S.bit word p i lo, S.bit word p i (lo + min inner (c - r * inner : Nat))⟩
    else
      (List.range c).map fun t =>
        let l := ((Part.strided p i b inner os is c).get t).j
        ⟨S.srcOf p i, S.bit word p i l, S.bit word p i (l + 1)⟩
  | .explicit items => items.toList.map fun r => ⟨S.srcOf r.p r.i, S.bit word r.p r.i r.j, S.bit word r.p r.i (r.j + 1)⟩

/-- A part's runs are as it reads: none when it is empty, else at least one and none empty. -/
def partGood (q : Part) (rs : List Run) : Bool :=
  if q.count = 0 then rs.isEmpty else !rs.isEmpty && rs.all fun r => decide (r.lo < r.hi)

/-- One run onto runs kept newest first: an empty run dropped, one that continues the newest run of its source joined to it. -/
def push (acc : List Run) (r : Run) : List Run :=
  if r.lo < r.hi then
    match acc with
    | x :: rest => if x.src = r.src ∧ x.hi = r.lo then { x with hi := r.hi } :: rest else r :: acc
    | [] => [r]
  else acc

/-- `boolean._merged`: runs with the empty dropped and adjacent ones of one source joined. -/
def merged (rs : List Run) : List Run := (rs.foldl push []).reverse

/-- A sequence's parts, with their runs on one side. -/
def seqRuns (S : Side) (word : Bool) (s : Seq) : List (Part × List Run) :=
  s.parts.toList.map fun q => (q, partRuns S word q)

/-! ### The check -/

/-- The refusal codes, in the order the check reaches them. -/
def codes : List String :=
  ["word-link", "sigma-table", "sigma-signature", "inputs", "signature", "node-count", "sigma-not-function", "node-form",
   "operand-type", "operand-runs", "return-runs"]

structure Refusal where
  code : String
  call : Option Nat := none
  operand : Option Nat := none
  definition : Option String := none
  why : String

def Refusal.toJson (e : Refusal) : Json :=
  Json.mkObj ([("ok", Json.bool false), ("code", Json.str e.code)] ++ (e.call.map fun c => ("call", Lean.toJson c)).toList ++
    (e.operand.map fun g => ("operand", Lean.toJson g)).toList ++
    (e.definition.map fun d => ("definition", Json.str d)).toList ++ [("why", Json.str e.why)])

structure Report where
  calls : Nat
  /-- the Boolean Definitions the Calls call -/
  definitions : Nat
  inputBits : Nat

def Report.toJson (r : Report) : Json :=
  Json.mkObj [("ok", Json.bool true), ("calls", Lean.toJson r.calls), ("definitions", Lean.toJson r.definitions),
    ("input_bits", Lean.toJson r.inputBits)]

private def sigTypes (d : Defn) : Array Ty := (d.params.map (·.2)).push d.ret

/-- The `word` field: the word descriptor's digest and σ's table, each entry a word Definition to a Boolean Definition that
declares its signature on bits. Entries in key order. -/
def linkTable (PW PB : Prog) (link : Json) : Except Refusal (Std.HashMap String String) := do
  let fix := "the Boolean descriptor carries \"word\": {\"program_sha512\": <the word descriptor's SHA-512>, \"sigma\": {<word id>: <Boolean id>}}"
  let .obj kvs := link | throw { code := "word-link", why := s!"no word link: {fix}" }
  let keys := kvs.toArray.map (·.1)
  if keys != #["program_sha512", "sigma"] then
    throw { code := "word-link", why := s!"the word link has the fields {keys}: {fix}" }
  let digest := hexOf PW.sha512
  match link.getObjVal? "program_sha512" with
  | .ok (.str d) =>
    if d != digest then
      throw { code := "word-link", why := s!"the word link names program {d}, the word descriptor is {digest}: link the word descriptor this Program was lifted from" }
  | _ => throw { code := "word-link", why := s!"the word link's program_sha512 is not a string: {fix}" }
  let .ok (.obj table) := link.getObjVal? "sigma" | throw { code := "word-link", why := s!"the word link's sigma is not an object: {fix}" }
  let mut out : Std.HashMap String String := {}
  for (w, b) in table.toArray do
    let .str b := b | throw { code := "sigma-table", definition := w, why := s!"σ({w}) is {b.compress}, not a Boolean definition id: map {w} to its Boolean version's definition id, a string" }
    let some iw := PW.index[w]? | throw { code := "sigma-table", definition := w, why := s!"σ has an entry for {w}, which is no definition of the word descriptor: key σ by the word descriptor's definition ids" }
    let some ib := PB.index[b]? | throw { code := "sigma-table", definition := w, why := s!"σ({w}) = {b}, which is no definition of the Boolean descriptor: include σ's image in the Boolean descriptor" }
    let dw := PW.defs[iw]!
    let db := PB.defs[ib]!
    if !(dw.params.size == db.params.size && ((sigTypes dw).map bitTy) == sigTypes db) then
      throw { code := "sigma-signature", definition := w, why := s!"σ({w}) = {b} declares {(sigTypes db).map Ty.render}, {w}'s signature on bits is {(sigTypes dw).map (Ty.render ∘ bitTy)}: σ(f) declares f's signature on bits" }
    out := out.insert w b
  return out

/-- The input streams: the Boolean root's inputs hold bits, and both streams are as long and bound alike, run by run. -/
def inputsRefusal (SW SB : Side) : Option Refusal :=
  match SB.wide with
  | some k => some { code := "inputs", why := s!"Boolean root node {k} is an input of {(SB.nodes[k]!).out.render}: a Boolean Program's inputs are bits (Input1_v1, RegisteredInput1_v1)" }
  | none =>
    if SW.inRuns != SB.inRuns then
      some { code := "inputs", why := s!"the Boolean root's input bits, as (bits, binding) runs, are {SB.inRuns.toList}, the word root's {SW.inRuns.toList}: each word input leaf is its width in bits, in order and bound alike" }
    else none

private def formOn : Form → Form
  | .prim => .call
  | f => f

/-- Operand `g` of Call `c`: its type on bits, and its runs moved. -/
def operandRefusal (SW SB : Side) (c : Nat) (xw xb : Node) (g : Nat) : Option Refusal :=
  match xw.args[g]?, xb.args[g]? with
  | some aw, some ab =>
    if !(bitTy aw.ty == ab.ty) then
      some { code := "operand-type", call := c, operand := g, why := s!"Call {c} operand {g} has type {ab.ty.render}, the word operand's on bits is {(bitTy aw.ty).render}: σ keeps each operand's type, on bits" }
    else
      let rw := seqRuns SW true aw.seq
      let rb := seqRuns SB false ab.seq
      if !(rw.all fun (q, rs) => partGood q rs) || !(rb.all fun (q, rs) => partGood q rs) then
        some { code := "operand-runs", call := c, operand := g, why := s!"Call {c} operand {g}: a reference part spans no bits: each part of n leaves reads n leaves of nonzero width" }
      else if merged (rb.flatMap (·.2)) != merged (rw.flatMap (·.2)) then
        some { code := "operand-runs", call := c, operand := g, why := s!"Call {c} operand {g} reads other bits than the word operand occupies: move each word run onto its bits (leaf j of a node's output is bit Σ widths[0..j))" }
      else none
  | _, _ => some { code := "node-form", call := c, operand := g, why := s!"Call {c} has no operand {g} on one side: σ keeps each Call's arity" }

/-- Call `c`: σ of its word Definition, its form, and its operands. `earlier`: whether an earlier Call has its word Definition. -/
def callRefusal (table : Std.HashMap String String) (PW PB : Prog) (SW SB : Side) (c : Nat) : Option Refusal :=
  match SW.calls[c]?, SB.calls[c]? with
  | some kw, some kb =>
    let xw := SW.nodes[kw]!
    let xb := SB.nodes[kb]!
    let fw := (PW.defs[xw.fn]!).key
    let fb := (PB.defs[xb.fn]!).key
    match table[fw]? with
    | none => some { code := "sigma-table", call := c, definition := fw, why := s!"Call {c} calls {fw}, which σ's table has no entry for: add σ({fw})" }
    | some g =>
      if fb != g then
        if (List.range c).any fun c0 => (PW.defs[(SW.nodes[SW.calls[c0]!]!).fn]!).key == fw then
          some { code := "sigma-not-function", call := c, definition := fw, why := s!"Call {c} calls {fb} for {fw}, which an earlier Call lifted to {g}: σ is a function, each Call of {fw} calls {g}" }
        else
          some { code := "sigma-table", call := c, definition := fw, why := s!"Call {c} calls {fb}, σ({fw}) = {g}: the Boolean Call calls σ of the word Call's Definition" }
      else if !(formOn xw.form == formOn xb.form && xw.n == xb.n && xw.axes == xb.axes && xw.nxs == xb.nxs
          && xw.args.size == xb.args.size && bitTy xw.out == xb.out) then
        some { code := "node-form", call := c, why := s!"Call {c} is a {xb.form.name} of {xb.n} over axes {xb.axes} with {xb.args.size} operands out {xb.out.render}, the word Call a {xw.form.name} of {xw.n} over axes {xw.axes} with {xw.args.size} out {xw.out.render} (on bits {(bitTy xw.out).render}): σ keeps each node's form" }
      else (List.range xw.args.size).findSome? (operandRefusal SW SB c xw xb)
  | _, _ => some { code := "node-count", call := c, why := s!"Call {c} is on one side only: σ maps each Call to one" }

def callsRefusal (table : Std.HashMap String String) (PW PB : Prog) (SW SB : Side) : Option Refusal :=
  (List.range SW.calls.size).findSome? (callRefusal table PW PB SW SB)

def returnRefusal (SW SB : Side) : Option Refusal :=
  let rw := seqRuns SW true SW.ret
  let rb := seqRuns SB false SB.ret
  if !(rw.all fun (q, rs) => partGood q rs) || !(rb.all fun (q, rs) => partGood q rs) then
    some { code := "return-runs", why := "a part of the return spans no bits: each part of n leaves reads n leaves of nonzero width" }
  else if merged (rb.flatMap (·.2)) != merged (rw.flatMap (·.2)) then
    some { code := "return-runs", why := "the Boolean root returns other bits than the word root's return occupies: move each word run onto its bits" }
  else none

/-- Everything after σ's table, in order. -/
def refusal (table : Std.HashMap String String) (PW PB : Prog) (SW SB : Side) : Option Refusal :=
  match inputsRefusal SW SB with
  | some e => some e
  | none =>
    if !(bitTy SW.retType == SB.retType) then
      some { code := "signature", why := s!"the Boolean root returns {SB.retType.render}, the word root's return on bits is {(bitTy SW.retType).render}: the Boolean Program returns the word one's return, lifted" }
    else if SW.calls.size ≠ SB.calls.size then
      some { code := "node-count", why := s!"the Boolean root has {SB.calls.size} Calls, the word root {SW.calls.size}: σ maps each Call to one" }
    else match callsRefusal table PW PB SW SB with
      | some e => some e
      | none => returnRefusal SW SB

def report (SB : Side) : Report :=
  let fs : Std.HashSet Nat := SB.calls.foldl (fun s k => s.insert (SB.nodes[k]!).fn) {}
  { calls := SB.calls.size, definitions := fs.size, inputBits := SB.inBits }

/-- The lift of the word Program `PW` onto the Boolean Program `PB`, whose descriptor's `word` field is `link`. -/
def check (PW PB : Prog) (link : Json) : Except Refusal Report :=
  match linkTable PW PB link with
  | .error e => .error e
  | .ok table =>
    match refusal table PW PB (Side.of PW) (Side.of PB) with
    | some e => .error e
    | none => .ok (report (Side.of PB))

/-- `check` on two descriptors' bytes, the Boolean one's `word` field read from its JSON. -/
def checkBytes (word bits : ByteArray) : Except Refusal Report := do
  let PW ← match Program.decode word with
    | .ok p => pure p
    | .error e => throw { code := "decode", why := s!"word descriptor: {e}" }
  let PB ← match Program.decode bits with
    | .ok p => pure p
    | .error e => throw { code := "decode", why := s!"Boolean descriptor: {e}" }
  let link := match String.fromUTF8? bits with
    | some t => match Json.parse t with
      | .ok j => (j.getObjVal? "word").toOption.getD Json.null
      | .error _ => Json.null
    | none => Json.null
  check PW PB link

/-- `flock-verify sigma-check WORD BITS`: one `SIGMACHECK` line, the report or the refusal; exit 0 when the lift holds. -/
def cmd (args : List String) : IO UInt32 := do
  let [wp, bp] := args | IO.eprintln "usage: flock-verify sigma-check WORD.json BITS.json"; return 2
  let word ← try IO.FS.readBinFile wp catch _ => pure ByteArray.empty
  let bits ← try IO.FS.readBinFile bp catch _ => pure ByteArray.empty
  match checkBytes word bits with
  | .ok r => IO.println s!"SIGMACHECK {r.toJson.compress}"; return 0
  | .error e => IO.println s!"SIGMACHECK {e.toJson.compress}"; return 1

/-! ### What carries across: the Calls' reads -/

/-- Call `c` of a root reads Call `c'`: some reference of one of its operands names a leaf of `c'`'s output. -/
def Reads (S : Side) (c c' : Nat) : Prop :=
  ∃ k x, S.calls[c]? = some k ∧ S.nodes[k]? = some x ∧ ∃ a ∈ x.args, ∃ q ∈ a.seq.parts, ∃ t < q.count,
    S.srcOf (q.get t).p (q.get t).i = .call c'

/-- The read graph of a partition `π` of the root's Calls into units: `u → v` when a Call of `v` reads a Call of `u`. -/
def Edge (S : Side) (π : Nat → Nat) (u v : Nat) : Prop :=
  ∃ c c', Reads S c c' ∧ π c' = u ∧ π c = v ∧ u ≠ v

theorem mem_push (acc : List Run) (r : Run) (s : Src) :
    s ∈ (push acc r).map Run.src ↔ s ∈ acc.map Run.src ∨ (r.lo < r.hi ∧ r.src = s) := by
  unfold push
  by_cases hr : r.lo < r.hi
  · simp only [hr, ↓reduceIte]
    cases acc with
    | nil => simp [eq_comm]
    | cons x rest =>
      dsimp only
      by_cases hx : x.src = r.src ∧ x.hi = r.lo
      · simp only [hx, and_self, ↓reduceIte]
        simp only [List.map_cons, List.mem_cons, true_and]
        grind
      · simp only [hx, ↓reduceIte]
        simp only [List.map_cons, List.mem_cons, true_and]
        grind
  · simp [hr]

theorem mem_foldl_push (s : Src) : ∀ (rs acc : List Run),
    s ∈ (rs.foldl push acc).map Run.src ↔ s ∈ acc.map Run.src ∨ ∃ r ∈ rs, r.lo < r.hi ∧ r.src = s
  | [], acc => by simp
  | r :: rs, acc => by
    rw [List.foldl_cons, mem_foldl_push s rs (push acc r), mem_push]
    simp only [List.mem_cons, exists_eq_or_imp]
    grind

/-- Joining runs keeps their sources: a source is in the merged runs exactly when a nonempty run of it is in the runs. -/
theorem mem_merged (rs : List Run) (s : Src) :
    s ∈ (merged rs).map Run.src ↔ ∃ r ∈ rs, r.lo < r.hi ∧ r.src = s := by
  unfold merged
  rw [List.map_reverse, List.mem_reverse, mem_foldl_push]
  simp

/-- A part whose runs are as it reads has, among its nonempty runs, the sources of its references. -/
theorem part_srcs (S : Side) (word : Bool) (q : Part) (h : partGood q (partRuns S word q) = true) (s : Src) :
    (∃ r ∈ partRuns S word q, r.lo < r.hi ∧ r.src = s) ↔ ∃ t < q.count, S.srcOf (q.get t).p (q.get t).i = s := by
  have hall : ∀ r ∈ partRuns S word q, r.lo < r.hi := by
    intro r hr
    unfold partGood at h
    split at h
    · simp at h; simp [h] at hr
    · simp only [Bool.and_eq_true, Bool.not_eq_true', List.isEmpty_eq_false_iff, List.all_eq_true, decide_eq_true_eq] at h
      exact h.2 r hr
  have hne : q.count ≠ 0 → partRuns S word q ≠ [] := by
    intro hc
    unfold partGood at h
    simp only [hc, ite_false, Bool.and_eq_true, Bool.not_eq_true', List.isEmpty_eq_false_iff] at h
    exact h.1
  have hnil : q.count = 0 → partRuns S word q = [] := by
    intro hc
    unfold partGood at h
    simp only [hc, ite_true, List.isEmpty_iff] at h
    exact h
  constructor
  · rintro ⟨r, hr, -, rfl⟩
    cases q with
    | affine p i b c =>
      simp only [partRuns] at hr
      split at hr
      · simp at hr
      · simp only [List.mem_singleton] at hr
        subst hr
        exact ⟨0, by simp [Part.count]; omega, by simp [Part.get]⟩
    | strided p i b inner os is c =>
      have hc : c ≠ 0 := by
        intro hc; have := hnil (by simp [Part.count, hc]); rw [this] at hr; simp at hr
      refine ⟨0, by simp [Part.count]; omega, ?_⟩
      simp only [partRuns, hc, ite_false] at hr
      split at hr <;> (simp only [List.mem_map] at hr; obtain ⟨_, _, rfl⟩ := hr; simp [Part.get])
    | explicit items =>
      simp only [partRuns, List.mem_map] at hr
      obtain ⟨x, hx, rfl⟩ := hr
      obtain ⟨t, ht, rfl⟩ := List.getElem_of_mem hx
      refine ⟨t, by simpa [Part.count] using ht, ?_⟩
      simp [Part.get, getElem!_pos items t (by simpa using ht)]
  · rintro ⟨t, ht, rfl⟩
    have hc : q.count ≠ 0 := by omega
    cases q with
    | affine p i b c =>
      have hc' : c ≠ 0 := by simpa [Part.count] using hc
      refine ⟨⟨S.srcOf p i, S.bit word p i b, S.bit word p i (b + c)⟩, by simp [partRuns, hc'], ?_, by simp [Part.get]⟩
      exact hall _ (by simp [partRuns, hc'])
    | strided p i b inner os is c =>
      obtain ⟨r, hr⟩ := List.exists_mem_of_ne_nil _ (hne hc)
      refine ⟨r, hr, hall r hr, ?_⟩
      have hc' : c ≠ 0 := by simpa [Part.count] using hc
      simp only [partRuns, hc', ite_false] at hr
      split at hr <;> (simp only [List.mem_map] at hr; obtain ⟨_, _, rfl⟩ := hr; simp [Part.get])
    | explicit items =>
      have ht' : t < items.size := by simpa [Part.count] using ht
      let x := items[t]
      refine ⟨⟨S.srcOf x.p x.i, S.bit word x.p x.i x.j, S.bit word x.p x.i (x.j + 1)⟩, ?_, ?_, ?_⟩
      · simp only [partRuns, List.mem_map]
        exact ⟨x, Array.getElem_mem_toList ht', rfl⟩
      · apply hall
        simp only [partRuns, List.mem_map]
        exact ⟨x, Array.getElem_mem_toList ht', rfl⟩
      · simp [x, Part.get, getElem!_pos items t ht']

/-- A sequence whose parts' runs are as they read: the sources of its merged runs are the sources of its references. -/
theorem seq_srcs (S : Side) (word : Bool) (sq : Seq) (h : (seqRuns S word sq).all (fun (q, rs) => partGood q rs) = true)
    (s : Src) : s ∈ (merged ((seqRuns S word sq).flatMap (·.2))).map Run.src ↔
      ∃ q ∈ sq.parts, ∃ t < q.count, S.srcOf (q.get t).p (q.get t).i = s := by
  rw [mem_merged]
  simp only [seqRuns, List.all_eq_true, List.mem_map, forall_exists_index, and_imp, forall_apply_eq_imp_iff₂,
    List.flatMap_map, List.mem_flatMap] at h ⊢
  constructor
  · rintro ⟨r, ⟨q, hq, hr⟩, hlt, hs⟩
    obtain ⟨t, ht, e⟩ := (part_srcs S word q (h q hq) s).1 ⟨r, hr, hlt, hs⟩
    exact ⟨q, Array.mem_def.2 hq, t, ht, e⟩
  · rintro ⟨q, hq, t, ht, e⟩
    have hq' : q ∈ sq.parts.toList := Array.mem_def.1 hq
    obtain ⟨r, hr, hlt, hs⟩ := (part_srcs S word q (h q hq') s).2 ⟨t, ht, e⟩
    exact ⟨r, ⟨q, hq', hr⟩, hlt, hs⟩

private theorem operand_srcs {SW SB : Side} {c : Nat} {xw xb : Node} {g : Nat} {aw ab : Arg}
    (hw : xw.args[g]? = some aw) (hb : xb.args[g]? = some ab) (h : operandRefusal SW SB c xw xb g = none) (s : Src) :
    (∃ q ∈ ab.seq.parts, ∃ t < q.count, SB.srcOf (q.get t).p (q.get t).i = s) ↔
      ∃ q ∈ aw.seq.parts, ∃ t < q.count, SW.srcOf (q.get t).p (q.get t).i = s := by
  unfold operandRefusal at h
  rw [hw, hb] at h
  simp only at h
  split at h
  · contradiction
  · split at h
    · contradiction
    · rename_i hgood
      split at h
      · contradiction
      · rename_i heq
        simp only [Bool.or_eq_true, Bool.not_eq_true', not_or, Bool.not_eq_false] at hgood
        have heq' : merged ((seqRuns SB false ab.seq).flatMap (·.2)) = merged ((seqRuns SW true aw.seq).flatMap (·.2)) := by
          simpa using heq
        rw [← seq_srcs SB false ab.seq hgood.2 s, ← seq_srcs SW true aw.seq hgood.1 s, heq']

private theorem call_srcs {table : Std.HashMap String String} {PW PB : Prog} {SW SB : Side} {c kw kb : Nat}
    (hw : SW.calls[c]? = some kw) (hb : SB.calls[c]? = some kb) (h : callRefusal table PW PB SW SB c = none) (s : Src) :
    (∃ a ∈ (SB.nodes[kb]!).args, ∃ q ∈ a.seq.parts, ∃ t < q.count, SB.srcOf (q.get t).p (q.get t).i = s) ↔
      ∃ a ∈ (SW.nodes[kw]!).args, ∃ q ∈ a.seq.parts, ∃ t < q.count, SW.srcOf (q.get t).p (q.get t).i = s := by
  unfold callRefusal at h
  rw [hw, hb] at h
  simp only at h
  split at h
  · contradiction
  · split at h
    · split at h <;> contradiction
    · split at h
      · contradiction
      · rename_i hform
        simp only [Bool.not_eq_true', Bool.not_eq_false, Bool.and_eq_true, beq_iff_eq] at hform
        have hsize : (SW.nodes[kw]!).args.size = (SB.nodes[kb]!).args.size := hform.1.2
        rw [List.findSome?_eq_none_iff] at h
        have hop : ∀ (g : Nat) (aw ab : Arg), (SW.nodes[kw]!).args[g]? = some aw → (SB.nodes[kb]!).args[g]? = some ab →
            ((∃ q ∈ ab.seq.parts, ∃ t < q.count, SB.srcOf (q.get t).p (q.get t).i = s) ↔
              ∃ q ∈ aw.seq.parts, ∃ t < q.count, SW.srcOf (q.get t).p (q.get t).i = s) := by
          intro g aw ab hgw hgb
          have hg : g < (SW.nodes[kw]!).args.size := by
            rcases Nat.lt_or_ge g (SW.nodes[kw]!).args.size with hg | hg
            · exact hg
            · rw [Array.getElem?_eq_none hg] at hgw; contradiction
          exact operand_srcs hgw hgb (h g (List.mem_range.2 hg)) s
        constructor
        · rintro ⟨ab, hab, rest⟩
          obtain ⟨g, hg, rfl⟩ := Array.getElem_of_mem hab
          have hg' : g < (SW.nodes[kw]!).args.size := hsize ▸ hg
          exact ⟨_, Array.getElem_mem hg',
            (hop g _ _ (Array.getElem?_eq_getElem hg') (Array.getElem?_eq_getElem hg)).1 rest⟩
        · rintro ⟨aw, haw, rest⟩
          obtain ⟨g, hg, rfl⟩ := Array.getElem_of_mem haw
          have hg' : g < (SB.nodes[kb]!).args.size := hsize ▸ hg
          exact ⟨_, Array.getElem_mem hg',
            (hop g _ _ (Array.getElem?_eq_getElem hg) (Array.getElem?_eq_getElem hg')).2 rest⟩

/-- Reads with the node fetched as the check fetches it: the same as `Reads` when the Calls name nodes of the root. -/
private theorem reads_iff (S : Side) (hk : ∀ k ∈ S.calls, k < S.nodes.size) (c c' : Nat) :
    Reads S c c' ↔ ∃ k, S.calls[c]? = some k ∧ ∃ a ∈ (S.nodes[k]!).args, ∃ q ∈ a.seq.parts, ∃ t < q.count,
      S.srcOf (q.get t).p (q.get t).i = .call c' := by
  constructor
  · rintro ⟨k, x, hc, hx, rest⟩
    refine ⟨k, hc, ?_⟩
    have hlt : k < S.nodes.size := hk k (Array.mem_of_getElem? hc)
    rw [getElem!_pos S.nodes k hlt]
    rw [Array.getElem?_eq_getElem hlt] at hx
    cases hx
    exact rest
  · rintro ⟨k, hc, rest⟩
    have hlt : k < S.nodes.size := hk k (Array.mem_of_getElem? hc)
    rw [getElem!_pos S.nodes k hlt] at rest
    exact ⟨k, _, hc, Array.getElem?_eq_getElem hlt, rest⟩

/-- `Side.of` numbers as Calls only nodes of the root. -/
theorem calls_lt (P : Prog) : ∀ k ∈ (Side.of P).calls, k < (Side.of P).nodes.size := by
  intro k hk
  simp only [Side.of, Array.mem_filter, Array.mem_range] at hk ⊢
  exact hk.1

private theorem check_ok {PW PB : Prog} {link : Json} {r : Report} (h : check PW PB link = .ok r) :
    ∃ table, refusal table PW PB (Side.of PW) (Side.of PB) = none := by
  unfold check at h
  split at h
  · contradiction
  · rename_i table _
    refine ⟨table, ?_⟩
    cases hr : refusal table PW PB (Side.of PW) (Side.of PB) with
    | none => rfl
    | some e => simp [hr] at h

private theorem refusal_none {table : Std.HashMap String String} {PW PB : Prog} {SW SB : Side}
    (h : refusal table PW PB SW SB = none) :
    SW.calls.size = SB.calls.size ∧ callsRefusal table PW PB SW SB = none := by
  unfold refusal at h
  split at h
  · contradiction
  · split at h
    · contradiction
    · split at h
      · contradiction
      · rename_i hs
        refine ⟨by simpa using hs, ?_⟩
        cases hc : callsRefusal table PW PB SW SB with
        | none => rfl
        | some e => simp [hc] at h

private theorem lt_of_getElem? {α : Type} {xs : Array α} {i : Nat} {x : α} (h : xs[i]? = some x) : i < xs.size := by
  rcases Nat.lt_or_ge i xs.size with hl | hl
  · exact hl
  · rw [Array.getElem?_eq_none hl] at h; contradiction

/-- The lift adds no read: a Call of the Boolean root reads another exactly when the word Call reads the word one. -/
theorem reads_lift {PW PB : Prog} {link : Json} {r : Report} (h : check PW PB link = .ok r) (c c' : Nat) :
    Reads (Side.of PB) c c' ↔ Reads (Side.of PW) c c' := by
  obtain ⟨table, hn⟩ := check_ok h
  obtain ⟨hsize, hcalls⟩ := refusal_none hn
  rw [reads_iff _ (calls_lt PB), reads_iff _ (calls_lt PW)]
  unfold callsRefusal at hcalls
  rw [List.findSome?_eq_none_iff] at hcalls
  constructor
  · rintro ⟨kb, hb, rest⟩
    have hlt : c < (Side.of PW).calls.size := hsize ▸ lt_of_getElem? hb
    exact ⟨_, Array.getElem?_eq_getElem hlt,
      (call_srcs (Array.getElem?_eq_getElem hlt) hb (hcalls c (List.mem_range.2 hlt)) _).1 rest⟩
  · rintro ⟨kw, hw, rest⟩
    have hlt : c < (Side.of PB).calls.size := hsize ▸ lt_of_getElem? hw
    have hlt' : c < (Side.of PW).calls.size := lt_of_getElem? hw
    exact ⟨_, Array.getElem?_eq_getElem hlt,
      (call_srcs hw (Array.getElem?_eq_getElem hlt) (hcalls c (List.mem_range.2 hlt')) _).2 rest⟩

/-- Any partition of the Calls into units has the same read graph on the lift. -/
theorem edges_lift {PW PB : Prog} {link : Json} {r : Report} (h : check PW PB link = .ok r) (π : Nat → Nat) :
    Edge (Side.of PB) π = Edge (Side.of PW) π := by
  funext u v
  apply propext
  unfold Edge
  simp only [reads_lift h]

/-- So the units of the lifted partition read each other in a cycle exactly when the word partition's do: the `unit-cycle`
refusal on the word partition is the one on its lift. -/
theorem acyclic_lift {PW PB : Prog} {link : Json} {r : Report} (h : check PW PB link = .ok r) (π : Nat → Nat) :
    (∀ u, ¬ Relation.TransGen (Edge (Side.of PB) π) u u) ↔ ∀ u, ¬ Relation.TransGen (Edge (Side.of PW) π) u u := by
  rw [edges_lift h]

end Flock.Sigma
