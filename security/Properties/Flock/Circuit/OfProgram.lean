module

public import Definitions.Core.Expand
public import Definitions.Core.Partition
public import Flock.Program
public import Properties.Flock.EvalBool

/-!
# The registered description as core's circuit (`ofProgram`)

C is the decoded description `P` (`Flock.Program.Prog`, PROTOCOL.md) read as core's `Verity.Circuit`: one gate per gate of
§6's canonical layout, in its order, each located by the decoder's own `Prog.gate`. A gate's width is its primitive's
`Value<w>`, it is an input gate exactly when its primitive is `Input<w>_v1` or `RegisteredInput<w>_v1`, it reads its
operands, and its operation is the meaning `sem` gives its primitive, on its operands' bits in order, each value low bit
first. The circuit's outputs are the gates the root's return resolves to. `ofProgram` is C: its meaning is the one
evaluator, `Prog.evalBool`, so a primitive Definition's gate computes that Definition's value on its operands' bits.

`ofProgramWith` is total by construction: a gate reads only its operands below it, so `reads_lt` holds by definition and
the operation reads nothing else, and a gate `Prog.gate` refuses is an input of width 0. On a description `decode` accepts
no operand is dropped and no gate is refused.

`partOf` reads the proof blocks off their gate ranges (`[lo, hi)`, several a block: a unit's, a template query's ranges and
`Spot.BlockFront.Front.ranges`, or a root node's, the inference-only policy's owners): a computed gate in one of block `i`'s
ranges is in block `i`, one in none in the last block, empty on an accepted partition. `layoutOf` is core's
`Layout`: a `RegisteredInput` gate's entry is its root node and its leaf its offset in that node's gates, and every gate is
in the one execution phase.

A definition for the proofs: no verifier runs it, and the verifiers' descriptor-sized summaries are held to it.
-/

@[expose] public section

namespace Flock.OfProgram

open Flock.Program

/-- Gate `g`'s primitive (its index in `P.defs`) and operands, as `Prog.gate` locates them; none where it refuses. -/
def gateAt (P : Prog) (g : ℕ) : Option (ℕ × Array ℕ) :=
  match P.gate g with
  | .ok (p, ops, _) => some (p, ops)
  | .error _ => none

/-- The number of gates: the root's. -/
def size (P : Prog) : ℕ := (P.defs[P.root]!).gates

/-- A gate's primitive's key, `""` where it can't be located. -/
def keyAt (P : Prog) (g : ℕ) : String :=
  match gateAt P g with
  | some (p, _) => (P.defs[p]!).key
  | none => ""

/-- A gate's width: its primitive's `Value<w>`, 0 where it can't be located or returns more than one value. -/
def widthAt (P : Prog) (g : ℕ) : ℕ :=
  match gateAt P g with
  | some (p, _) => match (P.defs[p]!).ret with
    | .value w => w
    | _ => 0
  | none => 0

/-- Whether gate `g` is an input gate: its primitive is an input's, or it can't be located. -/
def inputAt (P : Prog) (g : ℕ) : Bool :=
  match gateAt P g with
  | some (p, _) => isInputId (P.defs[p]!).key
  | none => true

/-- Whether gate `g` is an input bound at registration (`RegisteredInput<w>_v1`, §5.2). -/
def registeredAt (P : Prog) (g : ℕ) : Bool :=
  (inputId? (keyAt P g)).map (·.1) == some REGISTRATION

/-- Gate `g`'s operands below it, in order, repeats kept: the gates its operation reads. -/
def opsBelow (P : Prog) (g : Fin (size P)) : List (Fin (size P)) :=
  if inputAt P g then [] else
  match gateAt P g with
  | some (_, ops) => ops.toList.filterMap fun a => if h : a < g.val then some ⟨a, lt_trans h g.isLt⟩ else none
  | none => []

theorem opsBelow_lt {P : Prog} {g a : Fin (size P)} (h : a ∈ opsBelow P g) : a < g := by
  unfold opsBelow at h
  split at h
  · simp at h
  · split at h
    · obtain ⟨x, -, hx⟩ := List.mem_filterMap.1 h
      split at hx
      · cases hx; assumption
      · cases hx
    · simp at h

/-- The bits an operation reads: each operand's value, low bit first, operands in order. -/
def opBits {P : Prog} (ops : List (Fin (size P))) (v : (a : Fin (size P)) → BitVec (widthAt P a)) : List Bool :=
  List.flatMap (fun a : Fin (size P) => List.ofFn fun i : Fin (widthAt P a) => (v a).getLsbD i) ops

theorem opBits_congr {P : Prog} (ops : List (Fin (size P))) (v v' : (a : Fin (size P)) → BitVec (widthAt P a))
    (h : ∀ a ∈ ops, v a = v' a) : opBits ops v = opBits ops v' := by
  unfold opBits
  exact List.flatMap_congr fun (a : Fin (size P)) ha => by rw [h a ha]

/-- The gates the root's return resolves to (§7). -/
def outputGates (P : Prog) : List ℕ :=
  let root := P.defs[P.root]!
  (List.range root.ret.leaves).filterMap fun j => (resolve P P.rootScope (root.body.ret.get j)).toOption

/-- **C, the registered description as core's circuit**, with `sem p bits` the meaning of primitive `p` (its index in
`P.defs`) on its operands' bits. -/
def ofProgramWith (P : Prog) (sem : ℕ → List Bool → List Bool) : Verity.Circuit where
  N := size P
  w g := widthAt P g
  isInput g := inputAt P g
  reads g := (opsBelow P g).toFinset
  reads_lt g a ha := opsBelow_lt (List.mem_toFinset.1 ha)
  input_reads g h := by
    show (opsBelow P g).toFinset = ∅
    simp [opsBelow, show inputAt P g = true from h]
  op g v :=
    match gateAt P g with
    | some (p, _) => Verity.ofBits fun i => (sem p (opBits (opsBelow P g) v)).getD i false
    | none => 0
  op_congr g v v' h := by
    split
    · rw [opBits_congr _ v v' fun a ha => h a (List.mem_toFinset.2 ha)]
    · rfl
  outputs := ((outputGates P).filterMap fun g => if h : g < size P then some ⟨g, h⟩ else none).toFinset

/-- The block of gate `g`: the first block one of whose ranges `[lo, hi)` holds it, else the last block. -/
def blockOf (blocks : Array (Array (ℕ × ℕ))) (g : ℕ) : Fin (blocks.size + 1) :=
  match blocks.findFinIdx? fun rs => rs.any fun r => decide (r.1 ≤ g ∧ g < r.2) with
  | some i => i.castSucc
  | none => Fin.last _

/-- **C**, the registered description as core's circuit: each gate's operation is `Prog.evalBool` of its primitive. -/
def ofProgram (P : Prog) : Verity.Circuit := ofProgramWith P (Prog.evalBool P)

/-- **The proof blocks**: a computed gate in one of block `i`'s gate ranges `[lo, hi)` is in block `i`, one in no block's
in the last block; an input gate is in none. -/
def partOf (P : Prog) (blocks : Array (Array (ℕ × ℕ))) : Verity.Partition (ofProgram P) (blocks.size + 1) where
  unit g := if inputAt P g then none else some (blockOf blocks g)
  input_iff g := by
    show (if inputAt P g then none else some (blockOf blocks g)) = none ↔ inputAt P g = true
    cases inputAt P g <;> simp

/-- Root node `k` holding gate `g`: `offsets[k] ≤ g < offsets[k + 1]`. -/
def rootNode (P : Prog) (g : ℕ) : Option ℕ :=
  let offs := (P.defs[P.root]!).body.offsets
  (List.range (offs.size - 1)).find? fun k => decide (offs[k]! ≤ g ∧ g < offs[k + 1]!)

/-- A registered gate's entry and leaf: its root node, and its offset in that node's gates. -/
def regAt (P : Prog) (g : ℕ) : Option (ℕ × ℕ) :=
  if registeredAt P g then (rootNode P g).map fun k => (k, g - (P.defs[P.root]!).body.offsets[k]!) else none

theorem rootNode_le {P : Prog} {g k : ℕ} (h : rootNode P g = some k) :
    (P.defs[P.root]!).body.offsets[k]! ≤ g := by
  unfold rootNode at h
  have := List.find?_some h
  simp only [decide_eq_true_eq] at this
  exact this.1

theorem regAt_inj {P : Prog} {g h : ℕ} (hs : (regAt P g).isSome) (he : regAt P g = regAt P h) : g = h := by
  unfold regAt at hs he
  by_cases hg : registeredAt P g = true
  · by_cases hh : registeredAt P h = true
    · simp only [hg, hh, ↓reduceIte] at hs he
      cases hk : rootNode P g with
      | none => simp [hk] at hs
      | some k =>
        cases hk' : rootNode P h with
        | none => simp [hk, hk'] at he
        | some k' =>
          simp only [hk, hk', Option.map_some, Option.some.injEq, Prod.mk.injEq] at he
          obtain ⟨rfl, e⟩ := he
          have := rootNode_le hk
          have := rootNode_le hk'
          omega
    · simp only [hg, hh, ↓reduceIte, Bool.false_eq_true] at hs he
      cases hk : rootNode P g <;> simp_all
  · simp [hg] at hs

theorem inputAt_of_registered {P : Prog} {g : ℕ} (h : registeredAt P g = true) : inputAt P g = true := by
  unfold registeredAt keyAt at h
  unfold inputAt
  cases hg : gateAt P g with
  | none => rfl
  | some x =>
    obtain ⟨p, ops⟩ := x
    rw [hg] at h
    unfold isInputId
    cases hi : inputId? (P.defs[p]!).key <;> simp_all

/-- **Where C's values are committed** (core's `Layout`): a registered gate at its root node's entry, at its offset in
that node; every gate in the one execution phase. -/
def layoutOf (P : Prog) : Verity.Layout (ofProgram P) where
  reg g := regAt P g
  reg_input g h := by
    show inputAt P g = true
    apply inputAt_of_registered
    unfold regAt at h
    by_cases hr : registeredAt P g = true
    · exact hr
    · simp [hr] at h
  reg_inj g h hs he := Fin.ext (regAt_inj hs he)
  phase _ := 0
  phase_reads _ _ _ := le_refl 0

end Flock.OfProgram
