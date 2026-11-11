module

public import Proofs.Properties.Nci.RecordGates
public import Proofs.Properties.Nci.ResolveRows

/-!
# A unit of a primitive

A unit is a member of a root node that is no input node, and its Definition may be a primitive: then the node is a
batch of that primitive (`prim_unit`), the unit is one gate, and its Definition has the body the decoder leaves a
primitive, `default`. Resolution reads that body as it reads any other. A reference of a default body names its
node 0, which is no node: the default node, whose form is a primitive's, at offset 0. So resolution ends at once,
at the activation's own gate (`resolve_default`), and every leaf such a unit returns is that gate (`ret_default`,
`returnsOwn_prim`). The gate is located at the root, at the unit's node, and its operands are the node's member's
(`gate_member_prim`, `prim_unit_gate`).

`run_one` and `run_two_ok` give a walk its result from its first step, or its first two, at any fuel that covers
them: where `run_two` reads a result off a walk that ended, these say that the walk ends.
-/

@[expose] public section

namespace NciCheck.Properties.SecurityProofs.Resolve

open Flock.Program

/-- A walk whose first step ends has that step's result, at any fuel but none. -/
theorem run_one {α β : Type} {what : String} {step : α → Flock.V (β ⊕ α)} {a : α} {b : β}
    (h : step a = .ok (.inl b)) (n : ℕ) : Flock.Steps.run what step (n + 1) a = .ok b := by
  rw [Flock.Steps.run.eq_2]
  simp only [h, bind, Except.bind]
  rfl

/-- A walk whose first step moves on and whose second ends has the second's result, at any fuel of two steps or
more. -/
theorem run_two_ok {α β : Type} {what : String} {step : α → Flock.V (β ⊕ α)} {a a' : α} {b : β}
    (h1 : step a = .ok (.inr a')) (h2 : step a' = .ok (.inl b)) (n : ℕ) :
    Flock.Steps.run what step (n + 2) a = .ok b := by
  rw [Flock.Steps.run.eq_2]
  simp only [h1, bind, Except.bind]
  exact run_one h2 n

/-- Reference `j` of the default sequence is leaf `j` of node 0: a sequence of no parts reads the default part. -/
theorem default_get (j : ℕ) : (default : Seq).get j = ⟨false, 0, j⟩ := by
  have hfor : ∀ {β : Type} (g : ℕ → β → Id (ForInStep β)) (init : β),
      (forIn [:(default : Seq).parts.size] init g : Id β) = pure init := by
    intro β g init
    rw [Std.Legacy.Range.forIn_eq_forIn_range']
    rfl
  unfold Flock.Program.Seq.get
  have hc : ((default : Seq).parts.size == 1) = false := rfl
  simp only [hc, Bool.false_eq_true, ↓reduceIte, hfor]
  show (⟨false, 0, 0 + (j - 0)⟩ : Ref) = ⟨false, 0, j⟩
  rw [Nat.zero_add]
  rfl

/-- One step of resolution from a node's reference in an activation whose body is the default ends at the
activation's own gate: the body has no node, so the reference reads the default node, a primitive's, at offset 0. -/
theorem resolveStep_default (P : Prog) (sc : Scope) (k j : ℕ) (hb : (P.defs[sc.d]!).body = default) :
    resolveStep P (sc, ⟨false, k, j⟩) = .ok (.inl sc.off) := by
  have hf : ((P.defs[sc.d]!).body.nodes[k]!).form = .prim := by rw [hb]; rfl
  have ho : (P.defs[sc.d]!).body.offsets[k]! = 0 := by rw [hb]; rfl
  rw [rs_prim P sc k j hf, ho]
  rfl

/-- **A node's reference in an activation whose body is the default resolves to the activation's own gate.** A
primitive's Definition has the default body (`BodiesOk.CalleeOk`), so in the activation of a unit of a primitive
`⟨false, 0, j⟩` resolves to the unit's one gate. It holds by the decoder's defaults, not by a rule of the format:
the default body has no node, the default node's form is a primitive's, and an offset out of range reads 0. -/
theorem resolve_default (P : Prog) (sc : Scope) (k j : ℕ) (hb : (P.defs[sc.d]!).body = default) :
    resolve P sc ⟨false, k, j⟩ = .ok sc.off := by
  obtain ⟨n, hn⟩ : ∃ n, Flock.Steps.FUEL = n + 1 :=
    ⟨Flock.Steps.FUEL - 1, by unfold Flock.Steps.FUEL; omega⟩
  unfold resolve
  rw [hn]
  exact run_one (resolveStep_default P sc k j hb) n

/-- **Every leaf an activation whose body is the default returns is the activation's own gate**: reference `t` of
the default body's return is `⟨false, 0, t⟩` (`default_get`), which `resolve_default` resolves. As
`resolve_default`, it holds by the decoder's defaults, not by a rule of the format. -/
theorem ret_default (P : Prog) (sc : Scope) (t : ℕ) (hb : (P.defs[sc.d]!).body = default) :
    resolve P sc ((P.defs[sc.d]!).body.ret.get t) = .ok sc.off := by
  rw [hb]
  show resolve P sc ((default : Seq).get t) = _
  rw [default_get]
  exact resolve_default P sc 0 t hb

section
variable (P : Prog) (hL : ∀ d < P.defs.size, Walk.LaidOut P d)
include hL

/-- **The step of `Prog.gate`'s walk at a member of a primitive's node**: for the gate of member `m` of node `k` of
an activation, the node calling a primitive, the step from the activation ends there, with the member's operands. -/
theorem gateStep_at_prim (sc : Scope) (hd : sc.d < P.defs.size) (hp : (P.defs[sc.d]!).prim = false) {k m g : ℕ}
    (hk : k < (P.defs[sc.d]!).body.nodes.size) (hm : m < ((P.defs[sc.d]!).body.nodes[k]!).members)
    (hc : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true)
    (hlo : sc.off + (P.defs[sc.d]!).body.offsets[k]! +
      m * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates ≤ g)
    (hhi : g < sc.off + (P.defs[sc.d]!).body.offsets[k]! +
      m * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates +
      (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates) {x : (ℕ × Array ℕ × Scope) ⊕ Scope}
    (h : gateStep P g sc = .ok x) :
    ∃ ops, x = .inl (((P.defs[sc.d]!).body.nodes[k]!).fn, ops, sc) ∧
      operands P sc ((P.defs[sc.d]!).body.nodes[k]!) m = .ok ops := by
  have hmono := offsets_mono P sc.d (hL sc.d hd) hp
  obtain ⟨-, -, hnodes⟩ := hL sc.d hd hp
  have hstep := hnodes k hk
  have hle : (m + 1) * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates ≤
      ((P.defs[sc.d]!).body.nodes[k]!).members * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates :=
    Nat.mul_le_mul_right _ hm
  rw [Nat.add_mul, Nat.one_mul] at hle
  unfold gateStep at h
  simp only [bind, Except.bind] at h
  split at h
  · rename_i k' hk'
    obtain ⟨hkK, hpk⟩ := DescAgree.find?_range_some hk'
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hpk
    have hkk : k' = k := by
      rcases Nat.lt_trichotomy k' k with hlt | heq | hgt
      · have := hmono (k' + 1) k (by omega) (by omega)
        omega
      · exact heq
      · have := hmono (k + 1) k' (by omega) (by omega)
        omega
    rw [hkk] at h
    simp only [hc, ite_true] at h
    split at h
    · cases h
    · rename_i ops hops
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      have hx : g - sc.off - (P.defs[sc.d]!).body.offsets[k]! <
          ((P.defs[sc.d]!).body.nodes[k]!).members * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates := by
        omega
      obtain ⟨-, hm2, hm3⟩ := Walk.member_of ((P.defs[sc.d]!).body.nodes[k]!) _ _ hx
      have e1 : (g - sc.off - (P.defs[sc.d]!).body.offsets[k]!) /
          (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates = m :=
        Nat.div_eq_of_lt_le (by omega) (by rw [Nat.add_mul, Nat.one_mul]; omega)
      have hM := (Nat.div_eq_of_lt_le hm2 (by rw [Nat.add_mul, Nat.one_mul]; omega)).symm.trans e1
      exact ⟨ops, rfl,
        (congrArg (fun M => operands P sc ((P.defs[sc.d]!).body.nodes[k]!) M = Except.ok ops) hM).mp hops⟩
  · simp [throw, throwThe, MonadExceptOf.throw] at h

variable (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
include hroot

/-- **The gate of a member of a root node of a primitive is located at the root**, at that node: its primitive is the
node's callee, and its operands are the member's. The walk's first step, from the root, ends there
(`gateStep_at_prim`). -/
theorem gate_member_prim {k m : ℕ} (hk : k < (P.defs[P.root]!).body.nodes.size)
    (hm : m < ((P.defs[P.root]!).body.nodes[k]!).members)
    (hc : (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).prim = true)
    (hG : 0 < (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).gates) {p : ℕ} {ops : Array ℕ} {sc' : Scope}
    (h : P.gate (P.child P.rootScope k m).off = .ok (p, ops, sc')) :
    p = ((P.defs[P.root]!).body.nodes[k]!).fn ∧ sc' = P.rootScope ∧
      operands P P.rootScope ((P.defs[P.root]!).body.nodes[k]!) m = .ok ops := by
  unfold Prog.gate at h
  by_cases hge : (P.child P.rootScope k m).off ≥ (P.defs[P.root]!).gates
  · simp [hge, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  simp only [hge, ite_false, bind, Except.bind] at h
  obtain ⟨n, hn⟩ : ∃ n, Flock.Steps.FUEL = n + 1 :=
    ⟨Flock.Steps.FUEL - 1, by unfold Flock.Steps.FUEL; omega⟩
  rw [hn, Flock.Steps.run.eq_2] at h
  rcases hs : gateStep P (P.child P.rootScope k m).off P.rootScope with e | x
  · simp [hs, bind, Except.bind] at h
  obtain ⟨ops', rfl, hops⟩ := gateStep_at_prim P hL P.rootScope hroot.1 hroot.2 hk hm hc (Nat.le_refl _)
    (Nat.lt_add_of_pos_right hG) hs
  simp only [hs, bind, Except.bind, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
  obtain ⟨rfl, rfl, rfl⟩ := h
  exact ⟨rfl, rfl, hops⟩

end

end NciCheck.Properties.SecurityProofs.Resolve

namespace NciCheck.Properties.Record.SecurityProofs

open Flock.Program (Prog Scope)
open Flock.TemplateQuery (TRange)
open NciCheck.Record.SecurityProofs (ranges_spec)
open NciCheck.Properties.SecurityProofs

/-- **A unit of a primitive is a member of a root batch of it**, of one gate that holds one value, and its
Definition has the default body. The root's ranges make a unit's node a call or a batch, and the decoder gives a
call a composite to call. -/
theorem prim_unit {P : Prog} {ts : Array String} {rs : Array TRange} (hP : ReadProg P)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) {tid : String} {u : ℕ}
    {sc : Scope} (hu : UnitAt P rs tid u sc) (hp : (P.defs[sc.d]!).prim = true) :
    ∃ k m, k < (P.defs[P.root]!).body.nodes.size ∧ ((P.defs[P.root]!).body.nodes[k]!).form = .batch ∧
      m < ((P.defs[P.root]!).body.nodes[k]!).members ∧ sc = P.child P.rootScope k m ∧
      (P.defs[sc.d]!).body = default ∧ (P.defs[sc.d]!).gates = 1 ∧ ∃ w, (P.defs[sc.d]!).ret = .value w := by
  obtain ⟨hr, -, -, -⟩ := read_bodies hP
  obtain ⟨⟨bytes, hdec⟩, -, -, -, -, -, hfirst⟩ := hP
  obtain ⟨-, hall⟩ := ranges_spec hr hfirst hrs
  obtain ⟨i, m, hi, -, hm, -, rfl⟩ := hu
  have hmem : rs[i]! ∈ rs.toList := by
    rw [getElem!_pos rs i hi]
    simp
  obtain ⟨hlt, -, hform, hcount⟩ := hall _ hmem
  obtain ⟨-, hcallee, hnode⟩ := Resolve.decoded_root bytes P hdec _ hlt
  have hp' : (P.defs[((P.defs[P.root]!).body.nodes[rs[i]!.node]!).fn]!).prim = true := hp
  obtain ⟨hbody, hg1, -, hw⟩ := hcallee.1 hp'
  have hbatch : ((P.defs[P.root]!).body.nodes[rs[i]!.node]!).form = .batch := by
    rcases hform with hcall | hbatch
    · have := hnode.2.2.2.1 (Or.inl hcall)
      rw [hp'] at this
      cases this
    · exact hbatch
  exact ⟨rs[i]!.node, m, hlt, hbatch, by rw [← hcount]; exact hm, rfl, hbody, hg1, hw⟩

/-- **The gate of a unit of a primitive**: the unit's one gate is located at the root, its primitive is the unit's
Definition, and its operands are those of the unit's member of its root node. -/
theorem prim_unit_gate {P : Prog} {ts : Array String} {rs : Array TRange} (hP : ReadProg P)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) {tid : String} {u : ℕ}
    {sc : Scope} (hu : UnitAt P rs tid u sc) (hp : (P.defs[sc.d]!).prim = true) {p : ℕ} {ops : Array ℕ}
    {sc' : Scope} (h : P.gate sc.off = .ok (p, ops, sc')) :
    p = sc.d ∧ sc' = P.rootScope ∧ ∃ k m, k < (P.defs[P.root]!).body.nodes.size ∧
      ((P.defs[P.root]!).body.nodes[k]!).form = .batch ∧ m < ((P.defs[P.root]!).body.nodes[k]!).members ∧
      sc = P.child P.rootScope k m ∧
      Flock.Program.operands P P.rootScope ((P.defs[P.root]!).body.nodes[k]!) m = .ok ops := by
  obtain ⟨k, m, hk, hf, hm, rfl, -, hg1, -⟩ := prim_unit hP hrs hu hp
  obtain ⟨hr, hrp, hlay, -⟩ := read_bodies hP
  have hc : (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).prim = true := hp
  have hG : 0 < (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).gates := by
    have h1 : (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).gates = 1 := hg1
    omega
  obtain ⟨h1, h2, h3⟩ := Resolve.gate_member_prim P hlay ⟨hr, hrp⟩ hk hm hc hG h
  exact ⟨h1, h2, k, m, hk, hf, hm, rfl, h3⟩

/-- **What a unit of a primitive returns is its own gate** (`ReturnsOwn`, of such a unit): every leaf of an output
row of the unit is the unit's one gate. -/
theorem returnsOwn_prim {P : Prog} {ts : Array String} {rs : Array TRange} (hP : ReadProg P)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) {tid : String} {u : ℕ}
    {sc : Scope} {out : String} {l g lo hi : ℕ} (hu : UnitAt P rs tid u sc) (hp : (P.defs[sc.d]!).prim = true)
    (ho : OutLeaf P sc out l g lo hi) : g = sc.off ∧ sc.off ≤ g ∧ g < sc.off + (P.defs[sc.d]!).gates := by
  obtain ⟨k, m, -, -, -, -, hbody, hg1, -⟩ := prim_unit hP hrs hu hp
  obtain ⟨f, -, -, -, hres, -, -⟩ := ho
  rw [Resolve.ret_default P sc _ hbody] at hres
  have hg : sc.off = g := Except.ok.inj hres
  subst hg
  exact ⟨rfl, Nat.le_refl _, by rw [hg1]; omega⟩

end NciCheck.Properties.Record.SecurityProofs
