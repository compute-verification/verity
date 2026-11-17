module

public import Proofs.Properties.InferenceOnly.RecordGates
public import Proofs.Properties.InferenceOnly.ResolveRows

/-! A unit of a primitive is one gate of a root batch of it, whose default body resolution reads as any other. -/

@[expose] public section

namespace InferenceOnlyCheck.Properties.SecurityProofs.Resolve

open Flock.Program
open InferenceOnlyCheck.Record.SecurityProofs (batch_dims)

theorem run_one {α β : Type} {what : String} {step : α → Flock.V (β ⊕ α)} {a : α} {b : β}
    (h : step a = .ok (.inl b)) (n : ℕ) : Flock.Steps.run what step (n + 1) a = .ok b := by
  rw [Flock.Steps.run.eq_2]
  simp only [h, bind, Except.bind]
  rfl

theorem run_two_ok {α β : Type} {what : String} {step : α → Flock.V (β ⊕ α)} {a a' : α} {b : β}
    (h1 : step a = .ok (.inr a')) (h2 : step a' = .ok (.inl b)) (n : ℕ) :
    Flock.Steps.run what step (n + 2) a = .ok b := by
  rw [Flock.Steps.run.eq_2]
  simp only [h1, bind, Except.bind]
  exact run_one h2 n

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

theorem resolveStep_default (P : Prog) (sc : Scope) (k j : ℕ) (hb : (P.defs[sc.d]!).body = default) :
    resolveStep P (sc, ⟨false, k, j⟩) = .ok (.inl sc.off) := by
  have hf : ((P.defs[sc.d]!).body.nodes[k]!).form = .prim := by rw [hb]; rfl
  have ho : (P.defs[sc.d]!).body.offsets[k]! = 0 := by rw [hb]; rfl
  rw [rs_prim P sc k j hf, ho]
  rfl

theorem resolve_default (P : Prog) (sc : Scope) (k j : ℕ) (hb : (P.defs[sc.d]!).body = default) :
    resolve P sc ⟨false, k, j⟩ = .ok sc.off := by
  obtain ⟨n, hn⟩ : ∃ n, Flock.Steps.FUEL = n + 1 :=
    ⟨Flock.Steps.FUEL - 1, by unfold Flock.Steps.FUEL; omega⟩
  unfold resolve
  rw [hn]
  exact run_one (resolveStep_default P sc k j hb) n

theorem ret_default (P : Prog) (sc : Scope) (t : ℕ) (hb : (P.defs[sc.d]!).body = default) :
    resolve P sc ((P.defs[sc.d]!).body.ret.get t) = .ok sc.off := by
  rw [hb]
  show resolve P sc ((default : Seq).get t) = _
  rw [default_get]
  exact resolve_default P sc 0 t hb

section
variable (P : Prog) (hL : ∀ d < P.defs.size, Walk.LaidOut P d)
include hL

variable (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
include hroot

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

section
variable (bytes : ByteArray) (P : Prog) (hdec : Flock.Program.decode bytes = .ok P)
  (hov : P.orderViolation = none) (hlay : Desc.bodiesLaidOut P = true) (hown : Record.ownRows P = true)
include hdec hov hlay hown

theorem unit_leaf_ok {k : ℕ} (hk : k < (P.defs[P.root]!).body.nodes.size)
    (hu : Desc.isInputNode P ((P.defs[P.root]!).body.nodes[k]!) = false)
    (hf : ((P.defs[P.root]!).body.nodes[k]!).form = .call ∨ ((P.defs[P.root]!).body.nodes[k]!).form = .batch)
    {m : ℕ} (hm : m < ((P.defs[P.root]!).body.nodes[k]!).members) {i : ℕ}
    (hi : i < ((P.defs[P.root]!).body.nodes[k]!).args.size) {t : ℕ}
    (ht : t < ((P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).params[i]!).2.leaves) (d off : ℕ) :
    ∃ g, resolveStep P (P.rootScope, (((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.get
        (argIdx ((P.defs[P.root]!).body.nodes[k]!) i m t)) = .ok (.inl g) ∧
      resolve P (.child P.rootScope d off k m) ⟨true, i, t⟩ = .ok g := by
  have hx := argIdx_lt bytes P hdec hown hk hu hf hm hi ht
  obtain ⟨n, j, -, -, -, -, hstep⟩ := arg_leaf bytes P hdec hov hlay hown hk hu hi hx
  obtain ⟨fuel, hfuel⟩ : ∃ fuel, Flock.Steps.FUEL = fuel + 2 :=
    ⟨Flock.Steps.FUEL - 2, by unfold Flock.Steps.FUEL; omega⟩
  refine ⟨_, hstep, ?_⟩
  unfold resolve
  rw [hfuel]
  exact run_two_ok (rs_unit P d off k m i t hf) hstep fuel

end

end InferenceOnlyCheck.Properties.SecurityProofs.Resolve

namespace InferenceOnlyCheck.Properties.Record.SecurityProofs

open Flock.Program (Prog Scope Ref)
open Flock.TemplateQuery (TRange)
open InferenceOnlyCheck.Record.SecurityProofs (ranges_spec batch_dims)
open InferenceOnlyCheck.Properties.SecurityProofs

theorem prim_unit {P : Prog} {ts : Array String} {rs : Array TRange} (hP : ReadProg P)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) {tid : String} {u : ℕ}
    {sc : Scope} (hu : UnitAt P rs tid u sc) (hp : (P.defs[sc.d]!).prim = true) :
    ∃ k m, k < (P.defs[P.root]!).body.nodes.size ∧ ((P.defs[P.root]!).body.nodes[k]!).form = .batch ∧
      m < ((P.defs[P.root]!).body.nodes[k]!).members ∧ sc = P.child P.rootScope k m ∧
      (P.defs[sc.d]!).body = default ∧ (P.defs[sc.d]!).gates = 1 ∧ ∃ w, (P.defs[sc.d]!).ret = .value w := by
  obtain ⟨hr, -, -, -⟩ := read_bodies hP
  have hir := readProg_inputs hP
  obtain ⟨⟨bytes, hdec⟩, -, -, -, -, -, hfirst⟩ := hP
  obtain ⟨-, hall⟩ := ranges_spec hr hfirst hir hrs
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

theorem read_walk {P : Prog} (hP : ReadProg P) : P.orderViolation = none ∧ Desc.bodiesLaidOut P = true := by
  obtain ⟨-, hov, -, -, hwk, -, -⟩ := hP
  unfold Desc.walkable at hwk
  simp only [Bool.and_eq_true] at hwk
  exact ⟨hov, hwk.1.2⟩

theorem unit_place {P : Prog} {ts : Array String} {rs : Array TRange} (hP : ReadProg P)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) {tid : String} {u : ℕ}
    {sc : Scope} (hu : UnitAt P rs tid u sc) :
    ∃ j m, j < rs.size ∧ rs[j]!.template = tid ∧
      u = ((rs.extract 0 j).filter fun r => r.template == tid).foldl (fun a r => a + r.count) 0 + m ∧
      sc = P.child P.rootScope rs[j]!.node m ∧ rs[j]!.node < (P.defs[P.root]!).body.nodes.size ∧
      Desc.isInputNode P ((P.defs[P.root]!).body.nodes[rs[j]!.node]!) = false ∧
      (((P.defs[P.root]!).body.nodes[rs[j]!.node]!).form = .call ∨
        ((P.defs[P.root]!).body.nodes[rs[j]!.node]!).form = .batch) ∧
      m < ((P.defs[P.root]!).body.nodes[rs[j]!.node]!).members ∧
      ((P.defs[P.root]!).body.nodes[rs[j]!.node]!).args.size =
        (P.defs[((P.defs[P.root]!).body.nodes[rs[j]!.node]!).fn]!).params.size := by
  obtain ⟨hr, -, -, -⟩ := read_bodies hP
  have hir := readProg_inputs hP
  obtain ⟨⟨bytes, hdec⟩, -, -, -, -, -, hfirst⟩ := hP
  obtain ⟨hnodes, hall⟩ := ranges_spec hr hfirst hir hrs
  obtain ⟨j, m, hj, htj, hm, huj, hsc⟩ := hu
  have hmem : rs[j]! ∈ rs.toList := by
    rw [getElem!_pos rs j hj]
    simp
  obtain ⟨hlt, -, hform, hcount⟩ := hall _ hmem
  have hno : rs[j]!.node ∈ rs.toList.map (·.node) := List.mem_map.2 ⟨_, hmem, rfl⟩
  rw [hnodes, List.mem_filter] at hno
  have hkey : Flock.Program.isInputId (P.defs[((P.defs[P.root]!).body.nodes[rs[j]!.node]!).fn]!).key = false := by
    simpa using hno.2
  obtain ⟨-, -, hnode⟩ := Resolve.decoded_root bytes P hdec _ hlt
  refine ⟨j, m, hj, htj, huj, hsc, hlt, ?_, hform, ?_, hnode.1⟩
  · simp [Desc.isInputNode, hkey]
  · rw [← hcount]
    exact hm

theorem unit_member {P : Prog} {ts : Array String} {rs : Array TRange} (hP : ReadProg P)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) {tid : String} {u : ℕ}
    {sc : Scope} (hu : UnitAt P rs tid u sc) :
    ∃ k m, k < (P.defs[P.root]!).body.nodes.size ∧
      Desc.isInputNode P ((P.defs[P.root]!).body.nodes[k]!) = false ∧
      (((P.defs[P.root]!).body.nodes[k]!).form = .call ∨ ((P.defs[P.root]!).body.nodes[k]!).form = .batch) ∧
      m < ((P.defs[P.root]!).body.nodes[k]!).members ∧
      (∀ i, i < (P.defs[sc.d]!).params.size → i < ((P.defs[P.root]!).body.nodes[k]!).args.size) ∧
      sc = P.child P.rootScope k m := by
  obtain ⟨j, m, -, -, -, rfl, hk, hnode, hform, hm, hargs⟩ := unit_place hP hrs hu
  exact ⟨rs[j]!.node, m, hk, hnode, hform, hm, fun i hi => Nat.lt_of_lt_of_eq hi hargs.symm, rfl⟩

theorem unit_param_ok {P : Prog} {ts : Array String} {rs : Array TRange} (hP : ReadProg P)
    (hO : InferenceOnlyCheck.Record.ownRows P = true)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) {tid : String} {u : ℕ}
    {sc : Scope} (hu : UnitAt P rs tid u sc) {i t : ℕ} (hi : i < (P.defs[sc.d]!).params.size)
    (ht : t < ((P.defs[sc.d]!).params[i]!).2.leaves) :
    ∃ g, Flock.Program.resolve P sc (⟨true, i, t⟩ : Ref) = .ok g := by
  obtain ⟨hov, hlay⟩ := read_walk hP
  obtain ⟨k, m, hk, hnode, hform, hm, hargs, rfl⟩ := unit_member hP hrs hu
  obtain ⟨⟨bytes, hdec⟩, -⟩ := hP
  obtain ⟨g, -, hg⟩ := Resolve.unit_leaf_ok bytes P hdec hov hlay hO hk hnode hform hm (hargs i hi) ht
    ((P.defs[P.root]!).body.nodes[k]!).fn (P.child P.rootScope k m).off
  exact ⟨g, hg⟩

theorem prim_unit_operands {P : Prog} {ts : Array String} {rs : Array TRange} (hP : ReadProg P)
    (hO : InferenceOnlyCheck.Record.ownRows P = true)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) {tid : String} {u : ℕ}
    {sc : Scope} (hu : UnitAt P rs tid u sc) (hp : (P.defs[sc.d]!).prim = true) {p : ℕ} {ops : Array ℕ}
    {sc' : Scope} (h : P.gate sc.off = .ok (p, ops, sc')) :
    ∀ o ∈ ops, ∃ i t, i < (P.defs[sc.d]!).params.size ∧ t < ((P.defs[sc.d]!).params[i]!).2.leaves ∧
      Flock.Program.resolve P sc (⟨true, i, t⟩ : Ref) = .ok o := by
  obtain ⟨hpd, -, k, m, hk, hf, hm, hsc, hops⟩ := prim_unit_gate hP hrs hu hp h
  obtain ⟨-, -, -, -, -, -, -, hg1, -⟩ := prim_unit hP hrs hu hp
  have hkey : Flock.Program.isInputId (P.defs[p]!).key = false :=
    unitsComputed hP hrs tid u sc sc.off p ops sc' hu (Nat.le_refl _) (by rw [hg1]; omega) h
  obtain ⟨hov, hlay⟩ := read_walk hP
  obtain ⟨⟨bytes, hdec⟩, -, -, -, -, -, -⟩ := hP
  subst hsc
  subst hpd
  have hkey' : Flock.Program.isInputId (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).key = false := hkey
  have hnode : Desc.isInputNode P ((P.defs[P.root]!).body.nodes[k]!) = false := by
    simp [Desc.isInputNode, hkey']
  obtain ⟨-, -, hno⟩ := Resolve.decoded_root bytes P hdec k hk
  intro o ho
  obtain ⟨i, hi, t, ht, hres⟩ := Resolve.operands_spec P P.rootScope _ m ops hops o ho
  have hd : (((P.defs[P.root]!).body.nodes[k]!).args[i]!).dims.isSome = true := by
    refine batch_dims hO (n := (P.defs[P.root]!).body.nodes[k]!) ?_ hf ?_
    · refine List.mem_filter.2 ⟨?_, by simp [hnode]⟩
      rw [getElem!_pos _ k hk]
      exact Array.mem_toList_iff.2 (Array.getElem_mem hk)
    · rw [getElem!_pos _ i hi]
      exact Array.getElem_mem hi
  rw [hf, hd] at ht
  rw [hf] at hres
  have ht' : t < ((P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).params[i]!).2.leaves := ht
  have hres' : Flock.Program.resolve P P.rootScope
      ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.get
        (Resolve.memberIdx (((P.defs[P.root]!).body.nodes[k]!).args[i]!) m t)) = .ok o := by
    rw [← Resolve.memberRef_eq]
    exact hres
  obtain ⟨g, hstep, hg⟩ := Resolve.unit_leaf_ok bytes P hdec hov hlay hO hk hnode (Or.inr hf) hm hi ht'
    ((P.defs[P.root]!).body.nodes[k]!).fn (P.child P.rootScope k m).off
  rw [Resolve.argIdx_batch hf] at hstep
  have hog : o = g := by
    obtain ⟨fuel, hfuel⟩ : ∃ fuel, Flock.Steps.FUEL = fuel + 1 :=
      ⟨Flock.Steps.FUEL - 1, by unfold Flock.Steps.FUEL; omega⟩
    unfold Flock.Program.resolve at hres'
    rw [hfuel, Resolve.run_one hstep fuel] at hres'
    exact (Except.ok.inj hres').symm
  refine ⟨i, t, Nat.lt_of_lt_of_eq hi hno.1, ht', ?_⟩
  rw [hog]
  exact hg

theorem outLeaf_prim {P : Prog} {ts : Array String} {rs : Array TRange} (hP : ReadProg P)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) {tid : String} {u : ℕ}
    {sc : Scope} (hu : UnitAt P rs tid u sc) (hp : (P.defs[sc.d]!).prim = true) :
    ∃ w, (P.defs[sc.d]!).ret = .value w ∧ OutLeaf P sc ("o" ++ toString 0) 0 sc.off 0 w ∧
      ∀ out l g lo hi, OutLeaf P sc out l g lo hi →
        out = "o" ++ toString 0 ∧ l = 0 ∧ g = sc.off ∧ lo = 0 ∧ hi = w := by
  obtain ⟨k, m, -, -, -, -, hbody, -, w, hw⟩ := prim_unit hP hrs hu hp
  have hfields : Flock.Program.fieldsOf (P.defs[sc.d]!).ret = [.value w] :=
    (congrArg Flock.Program.fieldsOf hw).trans rfl
  have hwidths : (Flock.Program.Ty.value w).widths = #[w] := by rw [Flock.Program.Ty.widths]
  have hrow : outRow (P.defs[sc.d]!) 0 = (0, #[w]) := by
    unfold outRow
    rw [hfields]
    exact Prod.ext rfl hwidths
  have hres : Flock.Program.resolve P sc ((P.defs[sc.d]!).body.ret.get 0) = .ok sc.off :=
    Resolve.ret_default P sc 0 hbody
  refine ⟨w, hw, ⟨0, rfl, ?_, ?_, ?_, ?_, ?_⟩, ?_⟩
  · rw [hfields]
    exact Nat.zero_lt_one
  · rw [hrow]
    exact Nat.zero_lt_one
  · rw [hrow]
    exact hres
  · rw [hrow]
    rfl
  · rw [hrow]
    exact (Nat.zero_add w).symm
  · rintro out l g lo hi ⟨f, rfl, hf, hl, hg, rfl, rfl⟩
    rw [hfields] at hf
    have hf0 : f = 0 := Nat.lt_one_iff.1 hf
    subst hf0
    rw [hrow] at hl hg ⊢
    have hl0 : l = 0 := Nat.lt_one_iff.1 hl
    subst hl0
    rw [Resolve.ret_default P sc _ hbody] at hg
    exact ⟨rfl, rfl, (Except.ok.inj hg).symm, rfl, Nat.zero_add w⟩

end InferenceOnlyCheck.Properties.Record.SecurityProofs
