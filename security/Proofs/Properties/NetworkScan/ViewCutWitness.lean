module

public import Proofs.Properties.NetworkScan.ViewCut
meta import Circuits.Program

/-!
# The scan's cut is not trivial

`scanViewCut_nontrivial` (the two-lemma rule's `_nontrivial` for `scanViewCut`): at a Program built by hand and a
circuit, which a scan check accepting everything would pass, the free inputs other than the view gates are not checked
inputs. The Program `oneInput` is a root of one node, an input primitive of one gate, which the root returns. It has no
template, so its scan layout reads step 0's presence gate at that input gate (`oneInput_egress`), and its egress node
has no view gate (`oneInput_viewSet`). The circuit `oneInputC` is one input gate, returned: two runs that differ at that
input give step 0 two egress rows (`not_checked`).

The instance is the model's. `oneInputC` is `oneInput`'s flattening given that `Flock.Program.isInputId` holds of the
primitive's key `Input1_v1` (`oneInput_flattens`): the walk locates gate 0 at the primitive with no operand
(`oneInput_gate`) and the return resolves to it (`oneInput_resolve`). The kernel does not evaluate `isInputId` at a
literal key, so that one fact is held by a `#guard`, which the compiled code evaluates, and the witness doesn't carry
`Flattens`. No flattening with no input gate would do: a circuit with no input gate has one run. Nor can a file
decoding to the Program be exhibited, since `Flock.Program.decode` reads through `Lean.Json.parse`, a `partial def` the
kernel can't unfold.
-/

@[expose] public section

namespace Properties.NetworkScan.ViewCutWitness

open Flock.Program
open NciCheck.Properties.SecurityProofs
open NetworkCheck.Period
open Properties.NetworkScan.WalkProof

/-- An input primitive of one gate, whose one parameter has no leaf. -/
def inDef : Defn :=
  { key := "Input1_v1", id := "Input1_v1", prim := true, params := #[("x", .array 0 (.value 1))], ret := .value 1,
    gates := 1, paramLeaves := 0, outLeaves := 1 }

/-- A root of one node, the input primitive, which it returns. -/
def rootDef : Defn :=
  { key := "root", id := "Root_v1", prim := false, params := #[], ret := .value 1,
    body := { params := #[], retType := .value 1,
              nodes := #[{ form := .prim, fn := 0, args := #[], out := .value 1 }],
              ret := { parts := #[.affine false 0 0 1], starts := #[0], count := 1 },
              offsets := #[0, 1] },
    gates := 1, paramLeaves := 0, outLeaves := 1 }

/-- The Program of one input gate, returned. -/
def oneInput : Prog :=
  { defs := #[inDef, rootDef], index := {}, root := 1, programFn := "", sha512 := ByteArray.empty }

/-- One input gate, returned. -/
def oneInputC : Claims.Sampling.Prog := ⟨[.input], [0]⟩

/-- One slot of one lane, cells of no byte, and a window of one step. -/
def oneStep : Schedule := ⟨0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, none⟩

-- The input primitive's key is an input id, as the compiled code evaluates it.
#guard Flock.Program.isInputId "Input1_v1"

/-- A node with no argument has no operand. -/
theorem operands_of_nil (P : Prog) (sc : Scope) (node : Node) (m : ℕ) (hargs : node.args = #[]) :
    operands P sc node m = .ok #[] := by
  rcases h : operands P sc node m with e | ops
  · exfalso
    unfold operands at h
    simp [hargs, pure, Except.pure, bind, Except.bind] at h
  · rw [Walk.operands_nil P sc node m ops hargs h]

theorem oneInput_resolve : resolve oneInput oneInput.rootScope ⟨false, 0, 0⟩ = .ok 0 := by
  have h1 : Flock.Steps.run "scope resolution" (resolveStep oneInput) 1 (oneInput.rootScope, ⟨false, 0, 0⟩) =
      .ok 0 := rfl
  have h2 : 1 ≤ Flock.Steps.FUEL := by unfold Flock.Steps.FUEL; omega
  exact Flock.Steps.run_mono h2 h1

theorem oneInput_gateAt : gateAt oneInput oneInput.rootScope ⟨false, 0, 0⟩ = 0 :=
  congrArg (fun r : Flock.V ℕ => r.toOption.getD 0) oneInput_resolve

theorem oneInput_gateStep :
    gateStep oneInput 0 oneInput.rootScope =
      .ok (.inl (((oneInput.defs[oneInput.root]!).body.nodes[0]!).fn, #[], oneInput.rootScope)) := by
  have hsize : (oneInput.defs[oneInput.root]!).body.nodes.size = 1 := rfl
  have hargs : ((oneInput.defs[oneInput.root]!).body.nodes[0]!).args = #[] := rfl
  have hc : (oneInput.defs[((oneInput.defs[oneInput.root]!).body.nodes[0]!).fn]!).prim = true := rfl
  have h1 : (oneInput.defs[oneInput.root]!).body.offsets[0]! ≤ 0 := Nat.le_refl 0
  have h2 : 0 < (oneInput.defs[oneInput.root]!).body.offsets[0 + 1]! := Nat.zero_lt_one
  unfold gateStep
  simp only [bind, Except.bind, rootScope_d, rootScope_off, Nat.sub_zero]
  split
  · rename_i k' hk'
    obtain ⟨hk'K, -⟩ := DescAgree.find?_range_some hk'
    have hkk : k' = 0 := by omega
    subst hkk
    simp only [hc, ite_true]
    rw [operands_of_nil oneInput _ _ _ hargs]
    rfl
  · exfalso
    rename_i hne
    have hnone := Option.eq_none_iff_forall_ne_some.2 hne
    rw [Array.find?_eq_none] at hnone
    exact hnone 0 (Array.mem_range.2 (by omega))
      (by simp only [Bool.and_eq_true]; exact ⟨decide_eq_true h1, decide_eq_true h2⟩)

/-- **The walk locates `oneInput`'s gate** at the input primitive, with no operand. -/
theorem oneInput_gate : oneInput.gate 0 = .ok (0, #[], oneInput.rootScope) := by
  have hg1 : (oneInput.defs[oneInput.root]!).gates = 1 := rfl
  have hng : ¬ (0 ≥ (oneInput.defs[oneInput.root]!).gates) := by omega
  unfold Prog.gate
  simp only [hng, ite_false, bind, Except.bind]
  rw [CausalProof.fuel_succ, Flock.Steps.run.eq_2]
  simp only [oneInput_gateStep, bind, Except.bind, pure, Except.pure]
  rfl

theorem oneInput_ret_get (j : ℕ) : (oneInput.defs[oneInput.root]!).body.ret.get j = ⟨false, 0, 0 + j⟩ :=
  CausalProof.seq_get_one (q := .affine false 0 0 1) rfl j

/-- **`oneInput` flattens to `oneInputC`**, given that its primitive's key is an input id. -/
theorem oneInput_flattens (hin : isInputId (oneInput.defs[0]!).key = true) : Flattens oneInput oneInputC := by
  refine ⟨rfl, fun g hg => ?_, fun g => ⟨fun h => ?_, fun h => ?_⟩⟩
  · have hl : oneInputC.gates.length = 1 := rfl
    have hg0 : g = 0 := by omega
    subst hg0
    refine ⟨0, #[], oneInput.rootScope, oneInput_gate, ⟨fun _ => hin, fun _ => rfl⟩, fun a => ?_⟩
    have hr : (oneInputC.gates[0]?.map Claims.Sampling.BGate.reads).getD [] = [] := rfl
    rw [hr]
    simp
  · have hg0 : g = 0 := List.mem_singleton.1 h
    subst hg0
    refine ⟨0, Nat.zero_lt_one, ?_⟩
    rw [oneInput_ret_get]
    exact oneInput_resolve
  · obtain ⟨j, hj, hres⟩ := h
    have hc : (oneInput.defs[oneInput.root]!).body.ret.count = 1 := rfl
    have hj0 : j = 0 := by omega
    subst hj0
    rw [oneInput_ret_get] at hres
    have h0 : resolve oneInput oneInput.rootScope ⟨false, 0, 0 + 0⟩ = .ok 0 := oneInput_resolve
    rw [h0] at hres
    cases hres
    exact List.mem_singleton.2 rfl

theorem oneInput_ingress (i : ℕ) : (scanLayout oneInput oneStep).ingress i = [] := rfl

/-- Step 0's slot reads its presence at the input gate and no frame bit. -/
theorem oneInput_egress (σ : Fin oneStep.allocation) : (scanLayout oneInput oneStep).egress 0 0 σ = (0, []) := by
  have h : (scanLayout oneInput oneStep).egress 0 0 σ = (gateAt oneInput oneInput.rootScope ⟨false, 0, 0⟩, []) :=
    rfl
  rw [h, oneInput_gateAt]

theorem oneInput_viewGates (t : ℕ) : viewGates oneInput t = [] := by
  have hl : ((calleeAt oneInput ((templateNodes oneInput).getLast?.getD 0)).params[0]!).2.leaves = 0 := by
    have h : ((calleeAt oneInput ((templateNodes oneInput).getLast?.getD 0)).params[0]!).2 =
        Ty.array 0 (.value 1) := rfl
    rw [h]
    simp [Ty.leaves]
  show paramGates oneInput _ t 0
    (((calleeAt oneInput ((templateNodes oneInput).getLast?.getD 0)).params[0]!).2.leaves) = []
  rw [hl]
  rfl

/-- `oneInput` has no view gate. -/
theorem oneInput_viewSet (s : Schedule) : viewSet oneInput s = ∅ := by
  have h : (List.range s.window).flatMap (viewGates oneInput) = [] := by
    rw [List.flatMap_eq_nil_iff]
    intro t _
    exact oneInput_viewGates t
  unfold viewSet
  rw [h, List.toFinset_nil]

/-- A layout with no ingress gate reads every input from the advice. -/
theorem inputOf_of_no_ingress {L S : ℕ} (lay : NetTiming.BoundaryGates L S) (h : ∀ i, lay.ingress i = []) (k : ℕ)
    (ins : List (List Bool)) (a : ℕ → Bool) (g : ℕ) : lay.inputOf k ins a g = a g := by
  unfold NetTiming.BoundaryGates.inputOf
  have h1 : (List.range ins.length).findSome? (fun i => ((lay.ingress i).findIdx? (· == g)).map (i, ·)) = none := by
    rw [List.findSome?_eq_none_iff]
    intro i _
    simp [h i]
  have h2 : (List.range k).any (fun i => (lay.ingress i).contains g) = false := by
    rw [List.any_eq_false]
    intro i _
    simp [h i]
  rw [h1, h2]
  rfl

/-- An input gate that nothing holds has its input's value. -/
theorem valOver_input (C : Claims.Sampling.Prog) (inp : ℕ → Bool) (g : ℕ) (hx : C.gates[g]? = some .input) :
    NetTiming.valOver C inp (fun _ => none) g = inp g := by
  rw [NetTiming.valOver_of_none_some C inp (free := fun _ => none) rfl hx]
  rfl

/-- The row at a slot whose gates are `(g, gs)`. -/
theorem rowOf_at {S : ℕ} (val : ℕ → Bool) (p : Fin S → ℕ × List ℕ) (σ : Fin S) {g : ℕ} {gs : List ℕ}
    (h : p σ = (g, gs)) : NetTiming.rowOf val p σ = if val g then some (gs.map val) else none := by
  unfold NetTiming.rowOf
  simp only [h]

/-- The input gate is a free input. -/
theorem zero_free : (0 : ℕ) ∈ NetTiming.freeInputs oneInputC (scanLayout oneInput oneStep) oneStep.window := by
  unfold NetTiming.freeInputs
  rw [Finset.mem_filter]
  refine ⟨Finset.mem_range.2 Nat.zero_lt_one, rfl, ?_⟩
  rw [List.any_eq_false]
  intro i _
  simp [oneInput_ingress i]

/-- **At `oneInput` the cut is not checked**: the input gate is step 0's presence gate, a free input and no view
gate, and two runs that differ there give step 0 two rows. -/
theorem not_checked (Pt : Claims.Sampling.Partition) (d : Claims.Sampling.Design) :
    ¬ NetTiming.CheckedUnique oneInputC Pt d (scanLayout oneInput oneStep) oneStep.window
      (NetTiming.freeInputs oneInputC (scanLayout oneInput oneStep) oneStep.window \ viewSet oneInput oneStep)
      (fun _ => True) := by
  intro h
  have hchk : (0 : ℕ) ∈
      NetTiming.freeInputs oneInputC (scanLayout oneInput oneStep) oneStep.window \ viewSet oneInput oneStep := by
    rw [oneInput_viewSet, Finset.sdiff_empty]
    exact zero_free
  have hoff : ∀ g ∉
      NetTiming.freeInputs oneInputC (scanLayout oneInput oneStep) oneStep.window \ viewSet oneInput oneStep,
      (fun _ : ℕ => false) g = (fun g : ℕ => decide (g = 0)) g := by
    intro g hg
    have hne : g ≠ 0 := fun e => hg (by rw [e]; exact hchk)
    simp [hne]
  have hrow := h 0 Nat.zero_lt_one [[]] [[]] (fun _ => false) (fun g => decide (g = 0)) (fun _ => none) rfl rfl
    (fun _ _ => rfl) hoff (fun g b hb => by simp at hb) trivial trivial 0
  have hval : ∀ a : ℕ → Bool,
      NetTiming.valOver oneInputC ((scanLayout oneInput oneStep).inputOf oneStep.window [[]] a) (fun _ => none) 0 =
        a 0 := by
    intro a
    rw [valOver_input oneInputC _ 0 rfl, inputOf_of_no_ingress _ oneInput_ingress]
  have hr := congrFun hrow ⟨0, Nat.zero_lt_one⟩
  rw [rowOf_at _ ((scanLayout oneInput oneStep).egress 0 0) ⟨0, Nat.zero_lt_one⟩ (oneInput_egress _),
    rowOf_at _ ((scanLayout oneInput oneStep).egress 0 0) ⟨0, Nat.zero_lt_one⟩ (oneInput_egress _), hval, hval] at hr
  simp at hr

end Properties.NetworkScan.ViewCutWitness

namespace Properties.NetworkScan

/-- **`ScanViewCut`'s conclusion is not trivial** (the two-lemma rule's `_nontrivial`): at a Program and a circuit,
built by hand, which a scan check accepting everything would pass, the free inputs other than the view gates are not
checked inputs under no check. The Program is one input gate, returned, and the circuit its flattening given that the
primitive's key is an input id (`ViewCutWitness.oneInput_flattens`): the input is step 0's presence gate and no view
gate (`ViewCutWitness.not_checked`). The instance is the model's: the kernel does not evaluate
`Flock.Program.isInputId` at a literal key, so the witness doesn't carry `Flattens`. -/
theorem scanViewCut_nontrivial :
    ∃ (P : Flock.Program.Prog) (C : Claims.Sampling.Prog) (s : NetworkCheck.Period.Schedule)
      (Pt : Claims.Sampling.Partition) (d : Claims.Sampling.Design),
      ¬ NetTiming.CheckedUnique C Pt d (scanLayout P s) s.window
        (NetTiming.freeInputs C (scanLayout P s) s.window \ viewSet P s) (fun _ => True) :=
  ⟨ViewCutWitness.oneInput, ViewCutWitness.oneInputC, ViewCutWitness.oneStep, ⟨[]⟩, ⟨[]⟩,
    ViewCutWitness.not_checked _ _⟩

end Properties.NetworkScan
