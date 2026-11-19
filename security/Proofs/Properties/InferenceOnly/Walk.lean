module

public import Proofs.Properties.InferenceOnly.DescAgree
public import Proofs.Properties.Flock.Decoder.Walk
import Mathlib.Analysis.SpecialFunctions.Pow.NNReal
import Mathlib.Data.NNRat.Floor
import Mathlib.Tactic.ENatToNat
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.Measurability.Init
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Tactic.ReduceModChar

/-!
# `Prog.gate`'s walk, within `Desc.walkable`

Toward `DescAgree.GatesLocated`: what `Desc.walkable` gives a walk of `Flock.Steps.run`. A walk whose steps each lower a
potential finishes within any fuel above the potential it starts at (`run_of_potential`), and `Desc.activations` and
`Desc.depths`, folded callees first, are their recurrences (`activations_get`, `depths_get`). A reference of a sequence
is its part's element, found by bisection over the parts' starts (`seq_get`, `bisect_spec`), and a strided part's
elements lie between the corners `Flock.Program.parsePartsIn` checks (`strided_corners`), so every reference a parsed
sequence holds names a leaf (`parsePartsIn_valid`). A composite's nodes' members hold fewer activations than it and nest
less deep (`activations_child`, `depths_child`), which is what the walk's potential lowers. On bodies laid out as the
decoder lays them (`LaidOut`), `Prog.gate`'s descent goes one level down a step to the activation holding a primitive
(`gateStep_holds`), so it locates every gate of the root's within `Desc.walkable`'s fuel wherever the operands it reaches
resolve in the activations it visits (`Reach`; `gate_ok_of`, `depth_lt_fuel`).

`Desc.walkable` checks every body laid out so, the root a composite, every primitive returning one value and every
input node called with no arguments (`Desc.bodiesLaidOut`, `bodiesLaidOut_spec`), so the primitive the walk locates
returns one value and an input's gate reads nothing (`gate_reads_of`).

So `GatesLocated` holds wherever two smaller statements do, each a named `Prop` here (`gatesLocated_of`):
`OperandsResolve` (operands resolve in every activation the walk visits; `OperandsResolve.operandsResolve_of` proves it
from `Resolve.BodiesOk`) and `OperandsEarlier` (every operand the walk locates is an earlier gate;
`Resolve.operandsEarlier_of` proves it from `Resolve.BodiesOk`), and `Resolve.bodiesOk` proves `Resolve.BodiesOk`.
`descAgrees_of_gaps` gives `DescAgrees` from those two and `FreshExact`.
-/

@[expose] public section

namespace InferenceOnlyVerifier.Properties.SecurityProofs.Walk

open Flock.Program (Prog Seq Part parsePartsIn Lane gridChecks parseGrid)

theorem calleesFirst_spec (P : Prog) (h : Desc.calleesFirst P = true) :
    ∀ d < P.defs.size, ∀ n ∈ (P.defs[d]!).body.nodes, n.fn < d := by
  intro d hd n hn
  unfold Desc.calleesFirst at h
  have := List.all_eq_true.1 h d (List.mem_range.2 hd)
  exact of_decide_eq_true (Array.all_eq_true_iff_forall_mem.1 this n hn)

theorem activations_size (P : Prog) : (Desc.activations P).size = P.defs.size :=
  (pushFold_spec _ _).1

/-- **The activations of one activation of Definition `d`**: none for a primitive, else its own and its composite
nodes' members'. -/
theorem activations_get (P : Prog) (hc : Desc.calleesFirst P = true) {d : ℕ} (hd : d < P.defs.size) :
    (Desc.activations P)[d]! = if (P.defs[d]!).prim then 0 else
      1 + (P.defs[d]!).body.nodes.foldl (fun s n => s + n.members * (Desc.activations P)[n.fn]!) 0 := by
  unfold Desc.activations
  rw [(pushFold_spec _ _).2 d hd]
  simp only
  split
  · rfl
  · congr 1
    apply foldl_nodes_congr
    intro n hn s
    rw [getElem!_extract _ (calleesFirst_spec P hc d hd n hn) (by rw [(pushFold_spec _ _).1]; omega)]

theorem depths_size (P : Prog) : (Desc.depths P).size = P.defs.size :=
  (pushFold_spec _ _).1

/-- **How deep Definition `d`'s activations nest**: none for a primitive, else one more than its callees'. -/
theorem depths_get (P : Prog) (hc : Desc.calleesFirst P = true) {d : ℕ} (hd : d < P.defs.size) :
    (Desc.depths P)[d]! = if (P.defs[d]!).prim then 0 else
      1 + (P.defs[d]!).body.nodes.foldl (fun s n => max s (Desc.depths P)[n.fn]!) 0 := by
  unfold Desc.depths
  rw [(pushFold_spec _ _).2 d hd]
  simp only
  split
  · rfl
  · congr 1
    apply foldl_nodes_congr
    intro n hn s
    rw [getElem!_extract _ (calleesFirst_spec P hc d hd n hn) (by rw [(pushFold_spec _ _).1]; omega)]


/-- **A composite's node's members are fewer activations than the composite's own**: `Desc.activations` counts one for
the activation itself, so a node's `members` copies of its callee's activations stay strictly below. -/
theorem activations_child (P : Prog) (hc : Desc.calleesFirst P = true) {d : ℕ} (hd : d < P.defs.size)
    (hp : (P.defs[d]!).prim = false) {n : Flock.Program.Node} (hn : n ∈ (P.defs[d]!).body.nodes) :
    n.members * (Desc.activations P)[n.fn]! < (Desc.activations P)[d]! := by
  rw [activations_get P hc hd, hp, if_neg (by simp)]
  have := le_foldl_add (fun n : Flock.Program.Node => n.members * (Desc.activations P)[n.fn]!)
    (P.defs[d]!).body.nodes.toList 0 n (Array.mem_toList_iff.2 hn)
  rw [Array.foldl_toList] at this
  omega

/-- **A composite's callees nest less deep than it**. -/
theorem depths_child (P : Prog) (hc : Desc.calleesFirst P = true) {d : ℕ} (hd : d < P.defs.size)
    (hp : (P.defs[d]!).prim = false) {n : Flock.Program.Node} (hn : n ∈ (P.defs[d]!).body.nodes) :
    (Desc.depths P)[n.fn]! < (Desc.depths P)[d]! := by
  rw [depths_get P hc hd, hp, if_neg (by simp)]
  have := le_foldl_max_of_mem (fun n : Flock.Program.Node => (Desc.depths P)[n.fn]!)
    (P.defs[d]!).body.nodes.toList 0 n (Array.mem_toList_iff.2 hn)
  rw [Array.foldl_toList] at this
  omega

/-- What `Prog.gate P g`'s walk keeps: an activation of a composite Definition, `g` within its gates. -/
def Holds (P : Prog) (g : ℕ) (sc : Flock.Program.Scope) : Prop :=
  sc.d < P.defs.size ∧ (P.defs[sc.d]!).prim = false ∧ sc.off ≤ g ∧ g - sc.off < (P.defs[sc.d]!).gates

/-- **The member `gateStep` goes into**: `x` gates into a node, member `m` holds it, `x - m · gates` into that member. -/
theorem member_of (nd : Flock.Program.Node) (cg x : ℕ) (hx : x < nd.members * cg) :
    (if (nd.form == .prim || nd.form == .call) = true then 0 else x / cg) < nd.members ∧
      (if (nd.form == .prim || nd.form == .call) = true then 0 else x / cg) * cg ≤ x ∧
      x - (if (nd.form == .prim || nd.form == .call) = true then 0 else x / cg) * cg < cg := by
  have hpos : 0 < cg := Nat.pos_of_ne_zero (by rintro rfl; simp at hx)
  unfold Flock.Program.Node.members at *
  rcases nd with ⟨f, fn, args, out, axes, n, nxs⟩
  cases f
  · change x < 1 * cg at hx
    change 0 < 1 ∧ 0 * cg ≤ x ∧ x - 0 * cg < cg
    omega
  · change x < 1 * cg at hx
    change 0 < 1 ∧ 0 * cg ≤ x ∧ x - 0 * cg < cg
    omega
  · change x < n * cg at hx
    change x / cg < n ∧ x / cg * cg ≤ x ∧ x - x / cg * cg < cg
    refine ⟨(Nat.div_lt_iff_lt_mul hpos).2 hx, Nat.div_mul_le_self x cg, ?_⟩
    have h1 := Nat.div_add_mod x cg
    rw [Nat.mul_comm] at h1
    have h2 := Nat.mod_lt x hpos
    omega
  · change x < n * cg at hx
    change x / cg < n ∧ x / cg * cg ≤ x ∧ x - x / cg * cg < cg
    refine ⟨(Nat.div_lt_iff_lt_mul hpos).2 hx, Nat.div_mul_le_self x cg, ?_⟩
    have h1 := Nat.div_add_mod x cg
    rw [Nat.mul_comm] at h1
    have h2 := Nat.mod_lt x hpos
    omega

/-- **One step of `Prog.gate`'s walk goes down**: from an activation that `Holds`, where the operands of a primitive node's
member resolve, `gateStep` ends at a primitive or moves to an activation that `Holds`, of a Definition nesting less deep. -/
theorem gateStep_holds (P : Prog) (hc : Desc.calleesFirst P = true) (hlo : ∀ d < P.defs.size, LaidOut P d) {g : ℕ}
    {sc : Flock.Program.Scope} (hs : Holds P g sc)
    (hops : ∀ k < (P.defs[sc.d]!).body.nodes.size, (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true →
      ∀ m < ((P.defs[sc.d]!).body.nodes[k]!).members, ∀ e,
        Flock.Program.operands P sc ((P.defs[sc.d]!).body.nodes[k]!) m ≠ .error e) :
    (∃ x, Flock.Program.gateStep P g sc = .ok (.inl x)) ∨
      ∃ sc', Flock.Program.gateStep P g sc = .ok (.inr sc') ∧ Holds P g sc' ∧
        (Desc.depths P)[sc'.d]! < (Desc.depths P)[sc.d]! := by
  suffices H : ∀ r, Flock.Program.gateStep P g sc = r → (∀ e, r ≠ .error e) ∧ ∀ sc', r = .ok (.inr sc') →
      Holds P g sc' ∧ (Desc.depths P)[sc'.d]! < (Desc.depths P)[sc.d]! by
    rcases h : Flock.Program.gateStep P g sc with e | (x | sc')
    · exact absurd rfl ((H _ h).1 e)
    · exact Or.inl ⟨x, rfl⟩
    · exact Or.inr ⟨sc', rfl, (H _ h).2 sc' rfl⟩
  intro r h
  obtain ⟨hd, hp, hoff, hlt⟩ := hs
  obtain ⟨h0, hK, hnodes⟩ := hlo sc.d hd hp
  obtain ⟨k0, hk0, hb0⟩ := DescAgree.node_exists _ (P.defs[sc.d]!).body.nodes.size h0 (g - sc.off)
    (by rw [hK]; exact hlt)
  unfold Flock.Program.gateStep at h
  simp only [bind, Except.bind] at h
  split at h
  · rename_i k hk
    obtain ⟨hkK, hpk⟩ := DescAgree.find?_range_some hk
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hpk
    have hx : g - sc.off - (P.defs[sc.d]!).body.offsets[k]! <
        ((P.defs[sc.d]!).body.nodes[k]!).members * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates := by
      have := hnodes k hkK
      omega
    obtain ⟨hm1, hm2, hm3⟩ := member_of ((P.defs[sc.d]!).body.nodes[k]!) _ _ hx
    by_cases hpc : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true
    · simp only [hpc, ite_true] at h
      split at h
      · rename_i e he
        exact absurd he (hops k hkK hpc _ hm1 e)
      · simp only [pure, Except.pure] at h
        subst h
        exact ⟨fun e he => (by cases he), fun sc' he => (by cases he)⟩
    · simp only [hpc, Bool.false_eq_true, ite_false, pure, Except.pure] at h
      subst h
      refine ⟨fun e he => (by cases he), fun sc' he => ?_⟩
      simp only [Except.ok.injEq, Sum.inr.injEq] at he
      subst he
      have hmem : (P.defs[sc.d]!).body.nodes[k]! ∈ (P.defs[sc.d]!).body.nodes := by
        rw [getElem!_pos _ k hkK]
        exact Array.getElem_mem hkK
      have hfn := calleesFirst_spec P hc sc.d hd _ hmem
      simp only [Holds, Flock.Program.Prog.child, Flock.Program.Scope.d.eq_2, Flock.Program.Scope.off.eq_2]
      refine ⟨⟨by omega, by simpa using hpc, by omega, by omega⟩, depths_child P hc hd hp hmem⟩
  · exfalso
    first
    | (rename_i hne
       have hnone := Option.eq_none_iff_forall_ne_some.2 hne
       rw [Array.find?_eq_none] at hnone
       exact hnone k0 (Array.mem_range.2 hk0) (by simp only [Bool.and_eq_true, decide_eq_true_eq]; exact hb0))
    | (rename_i hne _
       have hnone := Option.eq_none_iff_forall_ne_some.2 hne
       rw [Array.find?_eq_none] at hnone
       exact hnone k0 (Array.mem_range.2 hk0) (by simp only [Bool.and_eq_true, decide_eq_true_eq]; exact hb0))

/-- The activations `Prog.gate P g`'s walk visits: the root's, and each one a step of it moves to. -/
inductive Reach (P : Prog) (g : ℕ) : Flock.Program.Scope → Prop
  | root : Reach P g P.rootScope
  | step {sc sc' : Flock.Program.Scope} : Reach P g sc → Flock.Program.gateStep P g sc = .ok (.inr sc') → Reach P g sc'

theorem walkable_recut (P : Prog) (h : Desc.walkable P = true) : (P.defs[P.root]!).body.recut = #[] := by
  unfold Desc.walkable Desc.resolvable at h
  simp only [Bool.and_eq_true] at h
  exact Array.isEmpty_iff.mp h.2.1.1.2

/-- **`Prog.gate`'s walk goes down to a primitive**: on a Program its decoder laid out, every gate of the root's is located
within the root's depth of steps, where the operands of each primitive node's member resolve in the activations the walk
visits (`Reach`). -/
theorem gate_ok_of (P : Prog) (hc : Desc.calleesFirst P = true) (hlo : ∀ d < P.defs.size, LaidOut P d)
    (hr : P.root < P.defs.size) (hrp : (P.defs[P.root]!).prim = false)
    (hrc : (P.defs[P.root]!).body.recut = #[])
    (hfuel : (Desc.depths P)[P.root]! < Flock.Steps.FUEL) {g : ℕ} (hg : g < (P.defs[P.root]!).gates)
    (hops : ∀ sc, Reach P g sc → Holds P g sc → ∀ k < (P.defs[sc.d]!).body.nodes.size,
      (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true → ∀ m < ((P.defs[sc.d]!).body.nodes[k]!).members, ∀ e,
        Flock.Program.operands P sc ((P.defs[sc.d]!).body.nodes[k]!) m ≠ .error e) :
    ∃ x, P.gate g = .ok x := by
  obtain ⟨b, hb, -⟩ := run_of_potential (what := "gate location") (step := Flock.Program.gateStep P g)
    (fun sc => Reach P g sc ∧ Holds P g sc) (fun sc => (Desc.depths P)[sc.d]!) (fun _ => True)
    (fun sc hsc => by
      obtain ⟨hR, hs⟩ := hsc
      rcases gateStep_holds P hc hlo hs (hops sc hR hs) with ⟨x, hx⟩ | ⟨sc', h1, h2, h3⟩
      · exact Or.inl ⟨x, hx, trivial⟩
      · exact Or.inr ⟨sc', h1, ⟨Reach.step hR h1, h2⟩, h3⟩)
    Flock.Steps.FUEL P.rootScope
    ⟨Reach.root, ⟨hr, hrp, Nat.zero_le _, (by show g - 0 < (P.defs[P.root]!).gates; omega)⟩⟩ hfuel
  have hng : ¬ (g ≥ (P.defs[P.root]!).gates) := by omega
  refine ⟨b, ?_⟩
  rw [Flock.Program.Prog.gate_of hrc]
  simp only [hng, ite_false, bind, Except.bind]
  exact hb

/-- **`Desc.walkable` leaves the descent its fuel.** -/
theorem depth_lt_fuel (P : Prog) (h : Desc.walkable P = true) : (Desc.depths P)[P.root]! < Flock.Steps.FUEL := by
  unfold Desc.walkable Desc.walkBound at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  have h2 := h.2.2
  have : (Desc.depths P)[P.root]! + 2 ≤
      ((Desc.activations P)[P.root]! + 1) * ((Desc.depths P)[P.root]! + 2) := by
    exact Nat.le_mul_of_pos_left _ (by omega)
  omega

/-- What `Desc.bodiesLaidOut` checks: the root a composite, every composite's body laid out (`LaidOut`), every primitive
returning one `Value<w>`, and every input node of a composite called with no arguments. -/
theorem bodiesLaidOut_spec (P : Prog) (h : Desc.bodiesLaidOut P = true) :
    (P.defs[P.root]!).prim = false ∧ (∀ d < P.defs.size, LaidOut P d) ∧
      (∀ d < P.defs.size, (P.defs[d]!).prim = true → ∃ w, (P.defs[d]!).ret = .value w) ∧
      ∀ d < P.defs.size, (P.defs[d]!).prim = false → ∀ k < (P.defs[d]!).body.nodes.size,
        Desc.isInputNode P ((P.defs[d]!).body.nodes[k]!) = true → ((P.defs[d]!).body.nodes[k]!).args = #[] := by
  unfold Desc.bodiesLaidOut at h
  simp only [Bool.and_eq_true, Bool.not_eq_true', List.all_eq_true, List.mem_range] at h
  obtain ⟨hr, hall⟩ := h
  refine ⟨hr, fun d hd hp => ?_, fun d hd hp => ?_, fun d hd hp k hk hin => ?_⟩
  · have := hall d hd
    simp only [hp, Bool.false_eq_true, ite_false, Bool.and_eq_true, beq_iff_eq, List.all_eq_true,
      List.mem_range] at this
    exact ⟨this.1.1, this.1.2, fun k hk => (this.2 k hk).1⟩
  · have := hall d hd
    simp only [hp, ite_true] at this
    revert this
    cases (P.defs[d]!).ret <;> simp
  · have := hall d hd
    simp only [hp, Bool.false_eq_true, ite_false, Bool.and_eq_true, beq_iff_eq, List.all_eq_true,
      List.mem_range] at this
    have := (this.2 k hk).2
    simp only [hin, Bool.not_true, Bool.false_or, Array.isEmpty_iff] at this
    exact this

/-- A step of the walk that ends: at a primitive node `k` of the activation, with its member's operands. -/
theorem gateStep_inl (P : Prog) (g : ℕ) (sc : Flock.Program.Scope) (x)
    (h : Flock.Program.gateStep P g sc = .ok (.inl x)) :
    ∃ k < (P.defs[sc.d]!).body.nodes.size, ∃ m, (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true ∧
      x.1 = ((P.defs[sc.d]!).body.nodes[k]!).fn ∧ x.2.2 = sc ∧
      Flock.Program.operands P sc ((P.defs[sc.d]!).body.nodes[k]!) m = .ok x.2.1 := by
  unfold Flock.Program.gateStep at h
  simp only [bind, Except.bind] at h
  split at h
  · rename_i k hk
    obtain ⟨hkK, -⟩ := DescAgree.find?_range_some hk
    by_cases hp : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true
    · simp only [hp, ite_true] at h
      split at h
      · cases h
      · rename_i ops hops
        simp only [pure, Except.pure, Except.ok.injEq, Sum.inl.injEq] at h
        subst h
        exact ⟨k, hkK, _, hp, rfl, rfl, hops⟩
    · simp only [hp, Bool.false_eq_true, ite_false, pure, Except.pure, Except.ok.injEq] at h
      cases h
  · simp [throw, throwThe, MonadExceptOf.throw] at h

/-- A step of the walk that goes on: into a member of a composite node `k` of the activation. -/
theorem gateStep_inr (P : Prog) (g : ℕ) (sc sc' : Flock.Program.Scope)
    (h : Flock.Program.gateStep P g sc = .ok (.inr sc')) :
    ∃ k < (P.defs[sc.d]!).body.nodes.size, (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = false ∧
      sc'.d = ((P.defs[sc.d]!).body.nodes[k]!).fn := by
  unfold Flock.Program.gateStep at h
  simp only [bind, Except.bind] at h
  split at h
  · rename_i k hk
    obtain ⟨hkK, -⟩ := DescAgree.find?_range_some hk
    by_cases hp : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true
    · simp only [hp, ite_true] at h
      split at h
      · cases h
      · simp only [pure, Except.pure, Except.ok.injEq] at h
        cases h
    · simp only [hp, Bool.false_eq_true, ite_false, pure, Except.pure, Except.ok.injEq, Sum.inr.injEq] at h
      subst h
      exact ⟨k, hkK, by simpa using hp, rfl⟩
  · simp [throw, throwThe, MonadExceptOf.throw] at h

theorem operands_nil (P : Prog) (sc : Flock.Program.Scope) (node : Flock.Program.Node) (m : ℕ) (ops : Array ℕ)
    (hargs : node.args = #[]) (h : Flock.Program.operands P sc node m = .ok ops) : ops = #[] := by
  rw [Flock.Program.operands_of (Flock.Program.operands_cond h)] at h
  simp [hargs, pure, Except.pure, bind, Except.bind] at h
  first | exact h | exact h.symm

/-- **What the walk locates, on bodies the decoder's way**: the primitive returns one `Value<w>`, and an input primitive's
gate reads nothing (it is an input node, so `Desc.bodiesLaidOut` gives it no arguments). -/
theorem gate_reads_of (P : Prog) (hb : Desc.bodiesLaidOut P = true) {g p : ℕ} {ops : Array ℕ}
    {sc : Flock.Program.Scope} (h : P.gate g = .ok (p, ops, sc)) :
    (∃ w, (P.defs[p]!).ret = .value w) ∧ (Flock.Program.isInputId (P.defs[p]!).key = true → ops = #[]) := by
  obtain ⟨hr, -, hret, hinp⟩ := bodiesLaidOut_spec P hb
  rw [Flock.Program.Prog.gate_of (Flock.Program.Prog.gate_recut h)] at h
  by_cases hge : g ≥ (P.defs[P.root]!).gates
  · simp [hge, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  simp only [hge, ite_false, bind, Except.bind] at h
  refine DescAgree.run_inv "gate location" (Flock.Program.gateStep P g) (fun sc => (P.defs[sc.d]!).prim = false)
    (fun b => (∃ w, (P.defs[b.1]!).ret = .value w) ∧ (Flock.Program.isInputId (P.defs[b.1]!).key = true → b.2.1 = #[]))
    ?_ ?_ Flock.Steps.FUEL P.rootScope (p, ops, sc) hr h
  · intro a b ha hb'
    obtain ⟨k, hk, m, hp, h1, -, hops⟩ := gateStep_inl P g a b hb'
    have hd : a.d < P.defs.size := by
      refine (Nat.lt_or_ge a.d P.defs.size).elim id fun hn => ?_
      rw [getElem!_neg P.defs a.d (by omega)] at hk
      have h0 : (default : Flock.Program.Defn).body.nodes.size = 0 := rfl
      omega
    have hfn : ((P.defs[a.d]!).body.nodes[k]!).fn < P.defs.size := by
      refine (Nat.lt_or_ge _ P.defs.size).elim id fun hn => ?_
      rw [getElem!_neg P.defs _ (by omega)] at hp
      cases hp
    rw [h1]
    refine ⟨hret _ hfn hp, fun hkey => operands_nil P a _ m _ (hinp a.d hd ha k hk ?_) hops⟩
    simp [Desc.isInputNode, hp, hkey]
  · intro a a' _ ha'
    obtain ⟨k, -, hp, hd'⟩ := gateStep_inr P g a a' ha'
    show (P.defs[a'.d]!).prim = false
    rw [hd']; exact hp

/-- **Operands resolve on the walk.** On a Program `Desc.evaluate` reads, in every activation `Prog.gate`'s walk
visits, every primitive node's members' operands resolve (`Flock.Program.resolve` within its fuel, past no scan of no
iterations). Only a primitive node's: `Prog.gate` resolves no other node's operands, and a composite node's arguments
may name a scan of no iterations that no gate reads. -/
def OperandsResolve : Prop :=
  ∀ (bytes : ByteArray) (P : Prog), Flock.Program.decode bytes = .ok P → P.orderViolation = none →
    Desc.rootLaidOut P = true → Desc.inputsAtRoot P = true → Desc.walkable P = true →
    (∃ reads, Desc.analyze P = .ok reads) → ∀ g < gates P, ∀ sc, Reach P g sc → Holds P g sc →
      ∀ k < (P.defs[sc.d]!).body.nodes.size, (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true →
        ∀ m < ((P.defs[sc.d]!).body.nodes[k]!).members, ∀ e,
          Flock.Program.operands P sc ((P.defs[sc.d]!).body.nodes[k]!) m ≠ .error e

/-- **Every operand is an earlier gate.** On such a Program, each operand `Prog.gate` locates for a gate is a gate
before it. -/
def OperandsEarlier : Prop :=
  ∀ (bytes : ByteArray) (P : Prog), Flock.Program.decode bytes = .ok P → P.orderViolation = none →
    Desc.rootLaidOut P = true → Desc.inputsAtRoot P = true → Desc.walkable P = true →
    (∃ reads, Desc.analyze P = .ok reads) → ∀ g < gates P, ∀ p ops sc, P.gate g = .ok (p, ops, sc) → ∀ a ∈ ops, a < g

/-- **`GatesLocated`, from resolution on the walk and the order of its operands**: `Desc.walkable` gives the walk its fuel
(`depth_lt_fuel`) and bodies laid out (`bodiesLaidOut_spec`), so with operands resolving it ends at a primitive
(`gate_ok_of`), which returns one value and, for an input, reads nothing (`gate_reads_of`). -/
theorem gatesLocated_of (hO : OperandsResolve) (hE : OperandsEarlier) : GatesLocated := by
  intro bytes P hdec hov hlo hin hwk han g hg
  have hwk' := hwk
  unfold Desc.walkable at hwk'
  simp only [Bool.and_eq_true] at hwk'
  have hc : Desc.calleesFirst P = true := hwk'.1.1
  have hb : Desc.bodiesLaidOut P = true := hwk'.1.2
  obtain ⟨hrp, hlay, -, -⟩ := bodiesLaidOut_spec P hb
  have hr : P.root < P.defs.size := by
    refine (Nat.lt_or_ge P.root P.defs.size).elim id fun hn => ?_
    have hg' : g < (P.defs[P.root]!).gates := hg
    rw [getElem!_neg P.defs P.root (by omega)] at hg'
    have h0 : (default : Flock.Program.Defn).gates = 0 := rfl
    omega
  obtain ⟨⟨p, ops, sc⟩, hx⟩ := gate_ok_of P hc hlay hr hrp (walkable_recut P hwk) (depth_lt_fuel P hwk) hg
    (hO bytes P hdec hov hlo hin hwk han g hg)
  obtain ⟨⟨w, hw⟩, hinp⟩ := gate_reads_of P hb hx
  exact ⟨p, ops, w, ⟨⟨sc, hx⟩, hw, hinp⟩, hE bytes P hdec hov hlo hin hwk han g hg p ops sc hx⟩

/-- **`DescAgrees`, from three statements**: `GatesLocated` through `gatesLocated_of`, and `FreshExact`. -/
theorem descAgrees_of_gaps (hO : OperandsResolve) (hE : OperandsEarlier) (hN : FreshExact) :
    Properties.DescAgrees :=
  descAgrees_of (gatesLocated_of hO hE) hN

end InferenceOnlyVerifier.Properties.SecurityProofs.Walk
