module

public import Proofs.Properties.Nci.ResolvePassed
public import Proofs.Properties.Nci.RecordOwn

/-!
# A unit's row is input gates of its own

A unit is a member of a root node that is no input node, a call or a batch. Where `Record.ownRows` holds, a leaf of
one of its parameters resolves in two steps: up to the root, at a reference of the node's argument (`rs_unit`), and
there to a gate of an input node, since the reference lies in a part that is a run of an input node's leaves
(`arg_leaf`). So the leaf is gate `j` of input node `n` (`unit_leaf`), and no two leaves of units are one gate
(`unit_leaf_inj`): a gate of the root names its node and the leaf (`node_of_gate`), a leaf of an input node lies in
one run at one place (`leafAt_inj`, from `Record.SecurityProofs.runs_apart`), and a batch's members read an argument
at references of their own (`memberIdx_inj`).
-/

@[expose] public section

namespace NciCheck.Properties.SecurityProofs.Resolve

open Flock.Program
open NciCheck.Record.SecurityProofs (runOf unitsOf partsIn Apart runs_apart part_run batch_dims runOf_some
  axisDims_of_removeAxis)

/-- A walk whose first step moves on and whose second ends has the second's result. -/
theorem run_two {α β : Type} {what : String} {step : α → Flock.V (β ⊕ α)} {a a' : α} {b b' : β}
    (h1 : step a = .ok (.inr a')) (h2 : step a' = .ok (.inl b')) :
    ∀ n, Flock.Steps.run what step n a = .ok b → b = b'
  | 0, h => by simp [Flock.Steps.run, throw, throwThe, MonadExceptOf.throw] at h
  | n + 1, h => by
    rw [Flock.Steps.run.eq_2] at h
    simp only [h1, bind, Except.bind] at h
    cases n with
    | zero => simp [Flock.Steps.run, throw, throwThe, MonadExceptOf.throw] at h
    | succ n =>
      rw [Flock.Steps.run.eq_2] at h
      simp only [h2, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
      exact h.symm

/-- `a · L + r` with `r < L` names `a` and `r`. -/
theorem mul_add_inj {a b r s L : ℕ} (hr : r < L) (hs : s < L) (h : a * L + r = b * L + s) : a = b ∧ r = s := by
  have hL : 0 < L := by omega
  have h1 : (a * L + r) / L = a := by
    rw [Nat.mul_comm, Nat.mul_add_div hL, Nat.div_eq_of_lt hr, Nat.add_zero]
  have h2 : (b * L + s) / L = b := by
    rw [Nat.mul_comm, Nat.mul_add_div hL, Nat.div_eq_of_lt hs, Nat.add_zero]
  have hab : a = b := by rw [← h1, ← h2, h]
  subst hab
  exact ⟨rfl, by omega⟩

/-- **Two members of a batch read an argument it maps at references of their own**: a reference names the member
and the leaf. -/
theorem memberIdx_inj (a : Arg) {d L : ℕ} (hd : a.dims = some (d, L)) (hL : 0 < L) {m m' t t' : ℕ} (hm : m < d)
    (hm' : m' < d) (h : memberIdx a m t = memberIdx a m' t') : m = m' ∧ t = t' := by
  unfold memberIdx at h
  rw [hd] at h
  obtain ⟨h1, h2⟩ := mul_add_inj (Nat.mod_lt t hL) (Nat.mod_lt t' hL) h
  obtain ⟨h3, h4⟩ := mul_add_inj hm hm' h1
  refine ⟨h4, ?_⟩
  rw [← Nat.div_add_mod t L, ← Nat.div_add_mod t' L, h3, h2]

/-- The reference of argument `i` of node `nd` that leaf `t` of member `m`'s parameter `i` is. -/
def argIdx (nd : Node) (i m t : ℕ) : ℕ := match nd.form with
  | .batch => memberIdx (nd.args[i]!) m t
  | _ => t

theorem argIdx_batch {nd : Node} (h : nd.form = .batch) (i m t : ℕ) :
    argIdx nd i m t = memberIdx (nd.args[i]!) m t := by
  unfold argIdx; rw [h]

theorem argIdx_call {nd : Node} (h : nd.form = .call) (i m t : ℕ) : argIdx nd i m t = t := by
  unfold argIdx; rw [h]

/-- **A unit's parameter leaf, one step up**: the root, at a reference of the node's argument. -/
theorem rs_unit (P : Prog) (d off k m i t : ℕ)
    (hf : ((P.defs[P.root]!).body.nodes[k]!).form = .call ∨ ((P.defs[P.root]!).body.nodes[k]!).form = .batch) :
    resolveStep P (.child P.rootScope d off k m, ⟨true, i, t⟩) =
      .ok (.inr (P.rootScope, (((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.get
        (argIdx ((P.defs[P.root]!).body.nodes[k]!) i m t))) := by
  rcases hf with hf | hf
  · rw [argIdx_call hf]
    exact rs_up P P.rootScope d off k m i t (Or.inr hf)
  · rw [argIdx_batch hf, ← memberRef_eq]
    exact rs_up_batch P P.rootScope d off k m i t hf

/-- A leaf of a root node that is a primitive is the node's gate. -/
theorem rs_root_prim (P : Prog) (n j : ℕ) (h : ((P.defs[P.root]!).body.nodes[n]!).form = .prim) :
    resolveStep P (P.rootScope, ⟨false, n, j⟩) = .ok (.inl ((P.defs[P.root]!).body.offsets[n]!)) :=
  (rs_prim P P.rootScope n j h).trans
    (congrArg (fun v => (Except.ok (Sum.inl v) : Flock.V (ℕ ⊕ Scope × Ref))) (Nat.zero_add _))

/-- Leaf `j` of a root node that is a batch of a primitive of one output leaf is the node's gate `j`. -/
theorem rs_root_batch (P : Prog) (n j : ℕ) (h : ((P.defs[P.root]!).body.nodes[n]!).form = .batch)
    (hp : (P.defs[((P.defs[P.root]!).body.nodes[n]!).fn]!).prim = true)
    (hol : (P.defs[((P.defs[P.root]!).body.nodes[n]!).fn]!).outLeaves = 1) :
    resolveStep P (P.rootScope, ⟨false, n, j⟩) = .ok (.inl ((P.defs[P.root]!).body.offsets[n]! + j)) := by
  have h1 := rs_batch P P.rootScope n j h
  have h2 : resolveStep P (P.rootScope, ⟨false, n, j⟩) =
      .ok (.inl (0 + (P.defs[P.root]!).body.offsets[n]! +
        j / (P.defs[((P.defs[P.root]!).body.nodes[n]!).fn]!).outLeaves)) := by
    rw [h1]
    show (if (P.defs[((P.defs[P.root]!).body.nodes[n]!).fn]!).prim = true then _ else _) = _
    rw [if_pos hp]
    rfl
  rw [h2, hol, Nat.div_one, Nat.zero_add]

/-- **An input node of the root, a primitive or a batch, has one gate a leaf of its output.** -/
theorem input_node (c : Defn) (nd : Node) (hc : BodiesOk.CalleeOk c) (hn : BodiesOk.NodeOk c nd)
    (hp : c.prim = true) (hf : nd.form = .prim ∨ nd.form = .batch) :
    nd.out.leaves = nd.members ∧ c.gates = 1 ∧ c.outLeaves = 1 := by
  obtain ⟨-, -, -, -, hsh, -⟩ := hn
  obtain ⟨-, hg, hol, w, hw⟩ := hc.1 hp
  refine ⟨?_, hg, hol⟩
  rcases hf with hf | hf
  · rw [hf] at hsh
    rw [prim_members nd (Or.inl hf), hsh.1, hw, BodiesOk.leaves_value]
  · rw [hf] at hsh
    rw [bs_members nd (Or.inl hf), hsh.1, BodiesOk.leaves_array, hw, BodiesOk.leaves_value, Nat.mul_one]

/-- What the decoder establishes of each node of the root and of the Definition it calls. -/
theorem decoded_root (bytes : ByteArray) (P : Prog) (hdec : Flock.Program.decode bytes = .ok P) :
    ∀ k < (P.defs[P.root]!).body.nodes.size,
      ((P.defs[P.root]!).body.nodes[k]!).fn < P.defs.size ∧
        BodiesOk.CalleeOk (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!) ∧
        BodiesOk.NodeOk (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!) ((P.defs[P.root]!).body.nodes[k]!) := by
  obtain ⟨hr, -, -⟩ := BodiesOk.decode_root bytes P hdec
  obtain ⟨st, hP, hinv, -⟩ := BodiesOk.decode_spec bytes P hdec
  have hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  intro k hk
  obtain ⟨-, hnodes, -⟩ := hok P.root hr
  obtain ⟨hfn, hnk, -⟩ := hnodes k hk
  have hlt : ((P.defs[P.root]!).body.nodes[k]!).fn < P.defs.size := Nat.lt_trans hfn hr
  exact ⟨hlt, (hok _ hlt).1, hnk⟩

/-- Reference `x` of argument `i` of root node `k` is leaf `j` of the root's node `n`: it lies in a part of the
argument that is a run of `n`'s leaves, `j` as far into the run as `x` into the part. -/
def LeafAt (P : Prog) (k i x n j : ℕ) : Prop :=
  ∃ q base count, q < ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts).size ∧
    runOf P ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts[q]!) = some (n, base, count) ∧
    ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq).starts[q]! ≤ x ∧
    x < ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq).starts[q]! + count ∧
    j = base + (x - ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq).starts[q]!)

/-- **A leaf of an input node is a reference of one argument, at one place.** -/
theorem leafAt_inj {P : Prog} (hown : Record.ownRows P = true) {k i x k' i' x' n j : ℕ}
    (hk : k < (P.defs[P.root]!).body.nodes.size)
    (hu : Desc.isInputNode P ((P.defs[P.root]!).body.nodes[k]!) = false)
    (hi : i < ((P.defs[P.root]!).body.nodes[k]!).args.size)
    (hk' : k' < (P.defs[P.root]!).body.nodes.size)
    (hu' : Desc.isInputNode P ((P.defs[P.root]!).body.nodes[k']!) = false)
    (hi' : i' < ((P.defs[P.root]!).body.nodes[k']!).args.size)
    (h : LeafAt P k i x n j) (h' : LeafAt P k' i' x' n j) : k = k' ∧ i = i' ∧ x = x' := by
  obtain ⟨q, base, count, hq, hr, h1, h2, hj⟩ := h
  obtain ⟨q', base', count', hq', hr', h1', h2', hj'⟩ := h'
  by_cases hpos : k = k' ∧ i = i' ∧ q = q'
  · obtain ⟨rfl, rfl, rfl⟩ := hpos
    rw [hr] at hr'
    have hb : base = base' := by
      have := Option.some.inj hr'
      exact (Prod.mk.inj (Prod.mk.inj this).2).1
    exact ⟨rfl, rfl, by omega⟩
  · have hap := runs_apart hown hk hu hi hq hk' hu' hi' hq' hr hr' (by show 0 < count; omega)
      (by show 0 < count'; omega) hpos
    unfold Apart at hap
    have hap' : n ≠ n ∨ base + count ≤ base' ∨ base' + count' ≤ base := hap
    omega

/-- **A gate of the root names its node and the leaf**, where each node's gates are its output's leaves. -/
theorem node_of_gate (P : Prog) (hl : Walk.LaidOut P P.root) (hp : (P.defs[P.root]!).prim = false) {n n' j j' : ℕ}
    (hn : n < (P.defs[P.root]!).body.nodes.size) (hn' : n' < (P.defs[P.root]!).body.nodes.size)
    (hj : j < ((P.defs[P.root]!).body.nodes[n]!).out.leaves)
    (hj' : j' < ((P.defs[P.root]!).body.nodes[n']!).out.leaves)
    (ho : (P.defs[P.root]!).body.offsets[n + 1]! =
      (P.defs[P.root]!).body.offsets[n]! + ((P.defs[P.root]!).body.nodes[n]!).out.leaves)
    (ho' : (P.defs[P.root]!).body.offsets[n' + 1]! =
      (P.defs[P.root]!).body.offsets[n']! + ((P.defs[P.root]!).body.nodes[n']!).out.leaves)
    (h : (P.defs[P.root]!).body.offsets[n]! + j = (P.defs[P.root]!).body.offsets[n']! + j') : n = n' ∧ j = j' := by
  have hmono := offsets_mono P P.root hl hp
  rcases Nat.lt_trichotomy n n' with hlt | heq | hgt
  · have := hmono (n + 1) n' (by omega) (by omega)
    omega
  · subst heq
    exact ⟨rfl, by omega⟩
  · have := hmono (n' + 1) n (by omega) (by omega)
    omega

section
variable (bytes : ByteArray) (P : Prog) (hdec : Flock.Program.decode bytes = .ok P)
  (hov : P.orderViolation = none) (hlay : Desc.bodiesLaidOut P = true) (hown : Record.ownRows P = true)
include hdec hov hlay hown

/-- **A reference of an argument of a root node that is no input node is a leaf of an input node, and resolves
from the root to that node's gate of the leaf.** -/
theorem arg_leaf {k : ℕ} (hk : k < (P.defs[P.root]!).body.nodes.size)
    (hu : Desc.isInputNode P ((P.defs[P.root]!).body.nodes[k]!) = false) {i : ℕ}
    (hi : i < ((P.defs[P.root]!).body.nodes[k]!).args.size) {x : ℕ}
    (hx : x < ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq).count) :
    ∃ n j, LeafAt P k i x n j ∧ n < (P.defs[P.root]!).body.nodes.size ∧
      j < ((P.defs[P.root]!).body.nodes[n]!).out.leaves ∧
      (P.defs[P.root]!).body.offsets[n + 1]! =
        (P.defs[P.root]!).body.offsets[n]! + ((P.defs[P.root]!).body.nodes[n]!).out.leaves ∧
      resolveStep P (P.rootScope, (((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.get x) =
        .ok (.inl ((P.defs[P.root]!).body.offsets[n]! + j)) := by
  obtain ⟨hroot, hB, hL, -, -, harg⟩ := decoded_walk bytes P hdec hov hlay
  have hs := harg P.root hroot.1 k hk i hi
  have hx' : x < (Seq.ofParts (((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts).count := by
    rw [← hs]
    exact hx
  obtain ⟨q, hq, h1, h2, hget⟩ := Walk.seq_get _ hx'
  rw [← hs] at h1 h2 hget
  have hunit : (P.defs[P.root]!).body.nodes[k]! ∈ unitsOf P := by
    refine List.mem_filter.2 ⟨?_, by simp [hu]⟩
    rw [getElem!_pos _ k hk]
    exact Array.mem_toList_iff.2 (Array.getElem_mem hk)
  have hpart : (((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts[q]! ∈
      partsIn ((P.defs[P.root]!).body.nodes[k]!) := by
    refine List.mem_flatMap.2 ⟨((P.defs[P.root]!).body.nodes[k]!).args[i]!, ?_, ?_⟩
    · rw [getElem!_pos _ i hi]
      exact Array.mem_toList_iff.2 (Array.getElem_mem hi)
    · rw [getElem!_pos _ q hq]
      exact Array.mem_toList_iff.2 (Array.getElem_mem hq)
  obtain ⟨⟨n, base, count⟩, hr⟩ := part_run hown hunit hpart
  obtain ⟨hqe, hn, hin, hform⟩ := runOf_some hr
  have hqe' : (((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts[q]! = .affine false n base count := hqe
  have hn' : n < (P.defs[P.root]!).body.nodes.size := hn
  have hin' : Desc.isInputNode P ((P.defs[P.root]!).body.nodes[n]!) = true := hin
  have hform' : ((P.defs[P.root]!).body.nodes[n]!).form = .prim ∨
      ((P.defs[P.root]!).body.nodes[n]!).form = .batch := hform
  rw [hqe'] at h2 hget
  have h2' : x < ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq).starts[q]! + count := h2
  have hget' : (((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.get x =
      ⟨false, n, base + (x - ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq).starts[q]!)⟩ := hget
  have hbody : BodyOk P P.root := hB P.rootScope ⟨rfl, rfl, rfl⟩
  obtain ⟨-, hargs, -, -, -⟩ := hbody.1 k hk
  have href := (hargs i x hx).2 (by rw [hget'])
  rw [hget'] at href
  have hjlt : base + (x - ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq).starts[q]!) <
      ((P.defs[P.root]!).body.nodes[n]!).out.leaves := href.2
  obtain ⟨-, hc, hno⟩ := decoded_root bytes P hdec n hn'
  have hprim : (P.defs[((P.defs[P.root]!).body.nodes[n]!).fn]!).prim = true := by
    have h0 := hin'
    unfold Desc.isInputNode at h0
    exact (Bool.and_eq_true_iff.1 h0).1
  obtain ⟨hleaves, hg1, hol⟩ := input_node _ _ hc hno hprim hform'
  have hlaid := (hL P.root hroot.1 hroot.2).2.2 n hn'
  refine ⟨n, base + (x - ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq).starts[q]!),
    ⟨q, base, count, hq, hr, h1, h2', rfl⟩, hn', hjlt, by rw [hlaid, hg1, hleaves, Nat.mul_one], ?_⟩
  rw [hget']
  rcases hform' with hf | hf
  · have hj0 : base + (x - ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq).starts[q]!) = 0 := by
      rw [hleaves, prim_members _ (Or.inl hf)] at hjlt
      omega
    rw [hj0, Nat.add_zero]
    exact rs_root_prim P n 0 hf
  · exact rs_root_batch P n _ hf hprim hol

/-- **A leaf of a unit's parameter is a leaf of an input node, and resolves to that node's gate of the leaf.** The
unit is member `m` of root node `k`, a call or a batch that is no input node; the leaf is leaf `t` of its parameter
`i`. -/
theorem unit_leaf {k : ℕ} (hk : k < (P.defs[P.root]!).body.nodes.size)
    (hu : Desc.isInputNode P ((P.defs[P.root]!).body.nodes[k]!) = false)
    (hf : ((P.defs[P.root]!).body.nodes[k]!).form = .call ∨ ((P.defs[P.root]!).body.nodes[k]!).form = .batch)
    {m : ℕ} (hm : m < ((P.defs[P.root]!).body.nodes[k]!).members) {i : ℕ}
    (hi : i < ((P.defs[P.root]!).body.nodes[k]!).args.size) {t : ℕ}
    (ht : t < ((P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).params[i]!).2.leaves) (d off : ℕ) :
    ∃ n j, LeafAt P k i (argIdx ((P.defs[P.root]!).body.nodes[k]!) i m t) n j ∧
      n < (P.defs[P.root]!).body.nodes.size ∧ j < ((P.defs[P.root]!).body.nodes[n]!).out.leaves ∧
      (P.defs[P.root]!).body.offsets[n + 1]! =
        (P.defs[P.root]!).body.offsets[n]! + ((P.defs[P.root]!).body.nodes[n]!).out.leaves ∧
      ∀ g, resolve P (.child P.rootScope d off k m) ⟨true, i, t⟩ = .ok g →
        g = (P.defs[P.root]!).body.offsets[n]! + j := by
  obtain ⟨-, hc, hno⟩ := decoded_root bytes P hdec k hk
  have hx : argIdx ((P.defs[P.root]!).body.nodes[k]!) i m t <
      ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq).count := by
    rcases hf with hf | hf
    · rw [argIdx_call hf]
      have hcp : (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).prim = false := hno.2.2.2.1 (Or.inl hf)
      have hbp := (hc.2 hcp).1
      have hup := BodiesOk.up_ok _ _ hc hno i (by rw [hbp, ← hno.1]; exact hi) t (by rw [hbp]; exact ht) m hm
      rw [hf] at hup
      exact hup
    · rw [argIdx_batch hf]
      have hd : (((P.defs[P.root]!).body.nodes[k]!).args[i]!).dims.isSome = true := by
        refine batch_dims hown (n := (P.defs[P.root]!).body.nodes[k]!) ?_ hf ?_
        · refine List.mem_filter.2 ⟨?_, by simp [hu]⟩
          rw [getElem!_pos _ k hk]
          exact Array.mem_toList_iff.2 (Array.getElem_mem hk)
        · rw [getElem!_pos _ i hi]
          exact Array.getElem_mem hi
      have hop := BodiesOk.op_ok _ _ hno i hi m hm t (by rw [hf, hd]; exact ht)
      rw [hf] at hop
      exact hop
  obtain ⟨n, j, hleaf, hn, hj, ho, hstep⟩ := arg_leaf bytes P hdec hov hlay hown hk hu hi hx
  exact ⟨n, j, hleaf, hn, hj, ho, fun g hg => run_two (rs_unit P d off k m i t hf) hstep _ hg⟩

/-- **No two leaves of units' parameters are one gate.** -/
theorem unit_leaf_inj {k k' : ℕ} (hk : k < (P.defs[P.root]!).body.nodes.size)
    (hk' : k' < (P.defs[P.root]!).body.nodes.size)
    (hu : Desc.isInputNode P ((P.defs[P.root]!).body.nodes[k]!) = false)
    (hu' : Desc.isInputNode P ((P.defs[P.root]!).body.nodes[k']!) = false)
    (hf : ((P.defs[P.root]!).body.nodes[k]!).form = .call ∨ ((P.defs[P.root]!).body.nodes[k]!).form = .batch)
    (hf' : ((P.defs[P.root]!).body.nodes[k']!).form = .call ∨ ((P.defs[P.root]!).body.nodes[k']!).form = .batch)
    {m m' : ℕ} (hm : m < ((P.defs[P.root]!).body.nodes[k]!).members)
    (hm' : m' < ((P.defs[P.root]!).body.nodes[k']!).members) {i i' : ℕ}
    (hi : i < ((P.defs[P.root]!).body.nodes[k]!).args.size)
    (hi' : i' < ((P.defs[P.root]!).body.nodes[k']!).args.size) {t t' : ℕ}
    (ht : t < ((P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).params[i]!).2.leaves)
    (ht' : t' < ((P.defs[((P.defs[P.root]!).body.nodes[k']!).fn]!).params[i']!).2.leaves) (d off d' off' : ℕ)
    {g : ℕ} (h : resolve P (.child P.rootScope d off k m) ⟨true, i, t⟩ = .ok g)
    (h' : resolve P (.child P.rootScope d' off' k' m') ⟨true, i', t'⟩ = .ok g) :
    k = k' ∧ m = m' ∧ i = i' ∧ t = t' := by
  obtain ⟨hroot, -, hL, -, -, -⟩ := decoded_walk bytes P hdec hov hlay
  obtain ⟨n, j, hleaf, hn, hj, ho, hg⟩ := unit_leaf bytes P hdec hov hlay hown hk hu hf hm hi ht d off
  obtain ⟨n', j', hleaf', hn', hj', ho', hg'⟩ := unit_leaf bytes P hdec hov hlay hown hk' hu' hf' hm' hi' ht' d' off'
  have hgg := (hg g h).symm.trans (hg' g h')
  obtain ⟨rfl, rfl⟩ := node_of_gate P (hL P.root hroot.1) hroot.2 hn hn' hj hj' ho ho' hgg
  obtain ⟨rfl, rfl, hxx⟩ := leafAt_inj hown hk hu hi hk' hu' hi' hleaf hleaf'
  obtain ⟨-, -, hno⟩ := decoded_root bytes P hdec k hk
  rcases hf with hf | hf
  · rw [argIdx_call hf, argIdx_call hf] at hxx
    rw [prim_members _ (Or.inr hf)] at hm hm'
    exact ⟨rfl, by omega, rfl, hxx⟩
  · rw [argIdx_batch hf, argIdx_batch hf] at hxx
    rw [bs_members _ (Or.inl hf)] at hm hm'
    have hd : (((P.defs[P.root]!).body.nodes[k]!).args[i]!).dims.isSome = true := by
      refine batch_dims hown (n := (P.defs[P.root]!).body.nodes[k]!) ?_ hf ?_
      · refine List.mem_filter.2 ⟨?_, by simp [hu]⟩
        rw [getElem!_pos _ k hk]
        exact Array.mem_toList_iff.2 (Array.getElem_mem hk)
      · rw [getElem!_pos _ i hi]
        exact Array.getElem_mem hi
    obtain ⟨-, -, -, -, hsh, hdims⟩ := hno
    rw [hf] at hsh
    rcases hsh.2 i hi with ⟨hnone, -⟩ | ⟨ax, mt, hrem, -, hdax⟩
    · rw [hnone] at hd
      simp at hd
    · obtain ⟨L, hLax⟩ := axisDims_of_removeAxis ax _ hrem
      rw [hLax] at hdax
      have hLpos : 0 < L := by
        rcases Nat.eq_zero_or_pos L with h0 | h0
        · have := hdims i hi _ L hdax h0
          omega
        · exact h0
      obtain ⟨hmm, htt⟩ := memberIdx_inj _ hdax hLpos hm hm' hxx
      exact ⟨rfl, hmm, rfl, htt⟩

end

end NciCheck.Properties.SecurityProofs.Resolve
