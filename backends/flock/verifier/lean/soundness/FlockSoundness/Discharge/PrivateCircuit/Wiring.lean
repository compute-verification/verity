import FlockSoundness.Discharge.PrivateCircuit.InterpPad

/-!
# Protocol 2, the private circuit: the public wiring between units

A class publishes `(N, G, nIn, nOut)` and a **topology** `T` (`Topo`): for each unit `j` and input `k < nIn`, the source
of that input, either the circuit's input `m` or output `l` of a unit `i < j` (`Src j`), so `T` is acyclic by
construction. Only each unit's gates and its internal wiring are hidden. This is protocol 1's structure at the universal
unit: unit `j`'s input `k` is a cut wire, the gate of `UniversalUnit_v1`'s Call `i` that computes output `l` (or the
circuit's input gate `m`), read across units, so it is committed (`Audit.Partition.mem_committed_of_read`) and has one
commit string (`Binding.Layout.pos`) that every unit reading it decodes (`Binding.Layout.decode_row`).

* `evalT T C x`: `C` end to end on the circuit's inputs `x`, unit by unit (`eval`), each unit's inputs routed by `T` from
  `x` and the earlier units' outputs (`route`). A source past a unit's outputs, or a unit past `C`'s, reads `0`, as a
  dummy's output does.
* `Wired T c xOf yOf x`: the claimed inputs `xOf j` of every unit `j < N` are what `T` routes to them from the claimed
  outputs `yOf` and the inputs `x`. In the universal circuit `xOf v j k` and the producer's `yOf v i` read one wire of
  the one assignment `v`, so `Wired` holds at every `v` by the layout: a fact about the circuit the topology compiles to,
  discharged where it is built, as `UnitWrong` is.
* `Reaches T c j i`: unit `j` feeds unit `i` along `T` (`j = i`, or `j` feeds a unit routed to one of `i`'s inputs).

The theorems: `wired_cone` (the units feeding unit `i` correct give `i`'s owed outputs), `wired_eval` (every unit correct
gives `evalT` at every unit) and `wired_wrong` (an owed output other than `evalT`'s is at least one wrong unit). Only the
0/1 count holds, not "as many differing units as wrong ones": one wrong unit feeding all the others can move every
output while every other unit is correct on its claimed inputs.
-/

namespace FlockSoundness.Discharge.PrivateCircuit

namespace Universal

/-- **The source of a unit's input** in unit `j`: the circuit's input `m`, or output `l` of an earlier unit `i < j`. -/
inductive Src (j : Nat) where
  | input (m : Nat)
  | unit (i : Fin j) (l : Nat)

/-- **A public topology**: the source of every unit's every input, acyclic by construction (`Src j` names only units
before `j`). The class's topology is `src j k` at `j < N`, `k < nIn`. -/
structure Topo where
  src : (j : Nat) → Nat → Src j

/-- A source's bit, from the circuit's inputs `x` and the units' outputs `y`: `x m`, or bit `l` of unit `i`'s outputs
(`0` past them). -/
def route {j : Nat} (x : Nat → Bool) (y : Nat → List Bool) : Src j → Bool
  | .input m => x m
  | .unit i l => read (y i.val) l

/-- **`C` end to end** on the circuit's inputs `x` under the topology `T`: unit `j`'s outputs are `eval C j` on its inputs,
each routed by `T` from `x` and the outputs of the units before `j`. -/
def evalT (T : Topo) (C : Circ) (x : Nat → Bool) (j : Nat) : List Bool :=
  eval C j fun k => match T.src j k with
    | .input m => x m
    | .unit i l => read (evalT T C x i.val) l
termination_by j
decreasing_by exact i.isLt

/-- **The claimed values agree along `T`**: every unit `j < N`'s claimed input `k < nIn` is the circuit's input or the
claimed output that `T` routes to it. -/
def Wired (T : Topo) (c : Cls) (xOf : Nat → Nat → Bool) (yOf : Nat → List Bool) (x : Nat → Bool) : Prop :=
  ∀ j < c.N, ∀ k < c.nIn, xOf j k = route x yOf (T.src j k)

/-- **Unit `j` feeds unit `i`** along `T`: `j = i`, or `j` feeds a unit `i'` that `T` routes to an input `k < nIn` of `i`. -/
inductive Reaches (T : Topo) (c : Cls) : Nat → Nat → Prop
  | refl (i : Nat) : Reaches T c i i
  | step {j i : Nat} (k : Nat) (hk : k < c.nIn) (i' : Fin i) (l : Nat) (h : T.src i k = .unit i' l)
      (hr : Reaches T c j i'.val) : Reaches T c j i

theorem Reaches.le {T : Topo} {c : Cls} {j i : Nat} (h : Reaches T c j i) : j ≤ i := by
  induction h with
  | refl => exact Nat.le_refl _
  | step _ _ i' _ _ _ ih => exact Nat.le_trans ih (Nat.le_of_lt i'.isLt)

theorem evalT_eq (T : Topo) (C : Circ) (x : Nat → Bool) (j : Nat) :
    evalT T C x j = eval C j fun k => route x (evalT T C x) (T.src j k) := by
  rw [evalT]
  congr 1

theorem eval_congr {C : Circ} {c : Cls} (hC : fits C c) (j : Nat) {f g : Nat → Bool}
    (h : ∀ k < c.nIn, f k = g k) : eval C j f = eval C j g := by
  unfold eval
  cases hj : C[j]? with
  | none => rfl
  | some u =>
    have hu := (hC.2.2 u (List.mem_of_getElem? hj)).2.1
    have : (List.ofFn fun k : Fin u.nIn => f k) = List.ofFn fun k : Fin u.nIn => g k :=
      congrArg List.ofFn (funext fun k => h k (by omega))
    simp only [interpret, this]

theorem eval_past {C : Circ} {j : Nat} (h : C.length ≤ j) (f : Nat → Bool) : eval C j f = [] := by
  simp [eval, List.getElem?_eq_none h]

theorem read_dummy (c : Cls) (l : Nat) : read (dummy c) l = false := by
  unfold read dummy
  rw [List.getD_eq_getElem?_getD, List.getElem?_replicate]
  split <;> rfl

/-- The induction behind `wired_cone`: at a unit `i < N` whose feeding units are all correct, the claimed outputs are
`evalT`'s, bit for bit, and as lists at a unit of `C`. -/
theorem cone_aux (T : Topo) (C : Circ) (c : Cls) (hC : fits C c) (xOf : Nat → Nat → Bool) (yOf : Nat → List Bool)
    (x : Nat → Bool) (hw : Wired T c xOf yOf x) :
    ∀ i, i < c.N → (∀ j, Reaches T c j i → run c (pad C c) j (xOf j) = yOf j) →
      (i < size C → yOf i = evalT T C x i) ∧ ∀ l, read (yOf i) l = read (evalT T C x i) l := by
  obtain ⟨-, hreal, hdum⟩ := pad_correct C c hC
  intro i
  induction i using Nat.strong_induction_on with
  | _ i ih =>
  intro hiN hcone
  have hrun := hcone i (.refl i)
  have heq : i < size C → yOf i = evalT T C x i := by
    intro hi
    rw [← hrun, hreal i hi (xOf i), evalT_eq]
    refine eval_congr hC i fun k hk => ?_
    rw [hw i hiN k hk]
    cases hs : T.src i k with
    | input m => rfl
    | unit i' l =>
      exact (ih i'.val i'.isLt (Nat.lt_trans i'.isLt hiN) fun j hj => hcone j (.step k hk i' l hs hj)).2 l
  refine ⟨heq, fun l => ?_⟩
  by_cases hi : i < size C
  · rw [heq hi]
  · have hle : C.length ≤ i := by unfold size at hi; omega
    rw [← hrun, hdum i hle hiN (xOf i), evalT_eq, eval_past hle, read_dummy]
    rfl

end Universal

open Universal in
/-- **The units feeding unit `i` give its owed outputs.** `C` fits the class `c`, and the claimed inputs and outputs agree
along the public topology `T` (`Wired`, at the circuit's inputs `x`). If every unit feeding unit `i < size C` along `T`
(`Reaches`), `i` included, is correct, the universal unit running `pad C c` on its claimed inputs giving its claimed
outputs, then unit `i`'s claimed outputs are `C`'s end to end on `x` (`evalT`). Protocol 1's `compose_cone` at the
universal unit; `interp` is tested equal to `UniversalUnit_v1`, not proved. -/
theorem wired_cone (T : Universal.Topo) (C : Universal.Circ) (c : Universal.Cls) (hC : Universal.fits C c)
    (xOf : ℕ → ℕ → Bool) (yOf : ℕ → List Bool) (x : ℕ → Bool) (hw : Universal.Wired T c xOf yOf x)
    (i : ℕ) (hi : i < Universal.size C)
    (hcone : ∀ j, Universal.Reaches T c j i → Universal.run c (Universal.pad C c) j (xOf j) = yOf j) :
    yOf i = Universal.evalT T C x i :=
  (cone_aux T C c hC xOf yOf x hw i (Nat.lt_of_lt_of_le hi (pad_correct C c hC).1) hcone).1 hi

open Universal in
/-- **Composition at the universal unit.** `C` fits `c`; every unit of the padded program is correct on its claimed
inputs (`run c (pad C c) i (xOf i) = yOf i` for `i < units c`, dummies included), and the claimed values agree along the
public topology `T` (`Wired`). Then every unit of `C` outputs what `C` end to end on the circuit's inputs `x` gives it
(`evalT`), so the owed outputs are `evalT`'s. -/
theorem wired_eval (T : Universal.Topo) (C : Universal.Circ) (c : Universal.Cls) (hC : Universal.fits C c)
    (xOf : ℕ → ℕ → Bool) (yOf : ℕ → List Bool) (x : ℕ → Bool) (hw : Universal.Wired T c xOf yOf x)
    (hu : ∀ i < Universal.units c, Universal.run c (Universal.pad C c) i (xOf i) = yOf i) :
    ∀ i < Universal.size C, yOf i = Universal.evalT T C x i := fun i hi =>
  wired_cone T C c hC xOf yOf x hw i hi fun j hj =>
    hu j (Nat.lt_of_le_of_lt hj.le (Nat.lt_of_lt_of_le hi (pad_correct C c hC).1))

open Universal in
/-- **An owed output off `evalT` is a wrong unit**: under `Wired`, if some unit of `C` claims outputs other than `C`'s end
to end on `x`, the padded program has at least one wrong unit, of `units c` (`wrong`, the count `pad_wrong` and
`UnitWrong.card` read). Only this 0/1 form holds: one wrong unit can move every unit it feeds. -/
theorem wired_wrong (T : Universal.Topo) (C : Universal.Circ) (c : Universal.Cls) (hC : Universal.fits C c)
    (xOf : ℕ → ℕ → Bool) (yOf : ℕ → List Bool) (x : ℕ → Bool) (hw : Universal.Wired T c xOf yOf x)
    (h : ∃ i < Universal.size C, yOf i ≠ Universal.evalT T C x i) :
    1 ≤ wrong (Universal.run c (Universal.pad C c)) (Universal.units c) xOf yOf := by
  obtain ⟨i, hi, hne⟩ := h
  refine Nat.one_le_iff_ne_zero.2 fun h0 => hne (wired_eval T C c hC xOf yOf x hw (fun j hj => ?_) i hi)
  by_contra hj'
  unfold wrong at h0
  rw [Finset.card_eq_zero, Finset.filter_eq_empty_iff] at h0
  exact h0 (Finset.mem_range.2 hj) hj'

end FlockSoundness.Discharge.PrivateCircuit
