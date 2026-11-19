module

public import Proofs.Properties.InferenceOnly.ResolveRows
public import Proofs.Properties.InferenceOnly.RecordOwnApart
public import Proofs.Properties.InferenceOnly.RecordRows
public import Proofs.Properties.InferenceOnly.RecordAligned

/-!
# A leaf of a row and the gate it is, where its argument reads no declared node

A unit's parameter leaf resolves, through the root's reference of the node's argument, to a leaf of an input node
(`unit_own_leaf`). Where the argument has no part over a declared node, the leaf is a leaf of one run `own-rows` reads,
and a run's leaf is read at one place: no two such leaves of units' parameters are one gate (`unit_own_inj`).

* `RowPos`, `row_pos`: where a row's leaf is, as `ownRows_holds` reads it: the range, member, parameter and leaf of the
  unit's parameter it is.
* `rows_eq_of_pos`: two leaves at one member and one parameter leaf of one node are one leaf of one row.
-/

@[expose] public section

namespace InferenceOnlyVerifier.Properties.SecurityProofs.Resolve

open Flock.Program
open InferenceOnlyVerifier.Record (overDeclared)
open InferenceOnlyVerifier.Record.SecurityProofs (runOf unitsOf partsIn Apart part_run runOf_some NoEdge OwnRunsOf
  ownPart_run ownBatch_dims ReadAsRun runs_apart_of axisDims_of_removeAxis)

/-- Reference `x` of argument `i` of root node `k` is leaf `j` of the root's node `n`: it lies in a part `own-rows` reads
as a run, that run's leaf `j` as far into the run as `x` into the part. -/
def LeafAtOwn (P : Prog) (D : List ℕ) (k i x n j : ℕ) : Prop :=
  ∃ q base count, q < ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts).size ∧
    runOf P ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts[q]!) = some (n, base, count) ∧
    ReadAsRun P D ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts[q]!) = true ∧
    ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq).starts[q]! ≤ x ∧
    x < ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq).starts[q]! + count ∧
    j = base + (x - ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq).starts[q]!)

/-- **A leaf of an input node is a reference of one argument, at one place**, where its parts are read as runs. -/
theorem leafAtOwn_inj {P : Prog} {D : List ℕ} (hown : OwnRunsOf P D) {k i x k' i' x' n j : ℕ}
    (hk : k < (P.defs[P.root]!).body.nodes.size)
    (hu : Desc.isInputNode P ((P.defs[P.root]!).body.nodes[k]!) = false)
    (hi : i < ((P.defs[P.root]!).body.nodes[k]!).args.size)
    (hk' : k' < (P.defs[P.root]!).body.nodes.size)
    (hu' : Desc.isInputNode P ((P.defs[P.root]!).body.nodes[k']!) = false)
    (hi' : i' < ((P.defs[P.root]!).body.nodes[k']!).args.size)
    (h : LeafAtOwn P D k i x n j) (h' : LeafAtOwn P D k' i' x' n j) : k = k' ∧ i = i' ∧ x = x' := by
  obtain ⟨q, base, count, hq, hr, hG, h1, h2, hj⟩ := h
  obtain ⟨q', base', count', hq', hr', hG', h1', h2', hj'⟩ := h'
  by_cases hpos : k = k' ∧ i = i' ∧ q = q'
  · obtain ⟨rfl, rfl, rfl⟩ := hpos
    rw [hr] at hr'
    have hb : base = base' := by
      have := Option.some.inj hr'
      exact (Prod.mk.inj (Prod.mk.inj this).2).1
    exact ⟨rfl, rfl, by omega⟩
  · have hap := runs_apart_of hown hk hu hi hq hk' hu' hi' hq' hG hG' hr hr' (by show 0 < count; omega)
      (by show 0 < count'; omega) hpos
    unfold Apart at hap
    have hap' : n ≠ n ∨ base + count ≤ base' ∨ base' + count' ≤ base := hap
    omega

/-- **A reference of an argument none of whose parts is over a declared node is a leaf of an input node, and
resolves from the root to that node's gate of the leaf.** -/
theorem arg_own_leaf (bytes : ByteArray) (P : Prog) (hdec : Flock.Program.decode bytes = .ok P)
    (hov : P.orderViolation = none) (hlay : Desc.bodiesLaidOut P = true) {D : List ℕ} (hown : OwnRunsOf P D)
    (hNE : NoEdge P) {k : ℕ} (hk : k < (P.defs[P.root]!).body.nodes.size)
    (hu : Desc.isInputNode P ((P.defs[P.root]!).body.nodes[k]!) = false) {i : ℕ}
    (hi : i < ((P.defs[P.root]!).body.nodes[k]!).args.size) {x : ℕ}
    (hx : x < ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq).count)
    (hnd : ∀ q ∈ ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts).toList, overDeclared D q = false)
    (hrc : (P.defs[P.root]!).body.recut = #[] := by assumption) :
    ∃ n j, LeafAtOwn P D k i x n j ∧ n < (P.defs[P.root]!).body.nodes.size ∧ D.contains n = false ∧
      j < ((P.defs[P.root]!).body.nodes[n]!).out.leaves ∧
      (P.defs[P.root]!).body.offsets[n + 1]! =
        (P.defs[P.root]!).body.offsets[n]! + ((P.defs[P.root]!).body.nodes[n]!).out.leaves ∧
      resolveStep P (P.rootScope, (((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.get x) =
        .ok (.inl ((P.defs[P.root]!).body.offsets[n]! + j)) := by
  obtain ⟨q, hq, h1, h2, hget⟩ := arg_part bytes P hdec hov hlay hk hi hx
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
  have hdq : overDeclared D ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts[q]!) = false :=
    hnd _ (by
      rw [getElem!_pos _ q hq]
      exact Array.mem_toList_iff.2 (Array.getElem_mem hq))
  have hedge := hNE _ hunit _ hpart
  obtain ⟨⟨n, base, count⟩, hr⟩ := ownPart_run hown hunit hpart hedge hdq
  obtain ⟨hqe, hn, hin, hform⟩ := runOf_some hr
  have hqe' : (((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts[q]! = .affine false n base count := hqe
  have hdn : D.contains n = false := by
    rw [hqe'] at hdq
    simpa [overDeclared] using hdq
  have hrun : ReadAsRun P D ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts[q]!) = true := by
    simp [ReadAsRun, hedge, hdq]
  rw [hqe'] at h2 hget
  have h2' : x < ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq).starts[q]! + count := h2
  have hget' : (((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.get x =
      ⟨false, n, base + (x - ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq).starts[q]!)⟩ := hget
  obtain ⟨hjlt, hol, hres⟩ := ref_leaf bytes P hdec hov hlay hk hi hx hget' hn hin hform
  exact ⟨n, _, ⟨q, base, count, hq, hr, hrun, h1, h2', rfl⟩, hn, hdn, hjlt, hol, hres⟩

/-- **A leaf of a unit's parameter, whose argument reads no declared node, is a leaf of an input node, and resolves to
that node's gate of the leaf.** -/
theorem unit_own_leaf (bytes : ByteArray) (P : Prog) (hdec : Flock.Program.decode bytes = .ok P)
    (hov : P.orderViolation = none) (hlay : Desc.bodiesLaidOut P = true) {D : List ℕ} (hown : OwnRunsOf P D)
    (hNE : NoEdge P) {k : ℕ} (hk : k < (P.defs[P.root]!).body.nodes.size)
    (hu : Desc.isInputNode P ((P.defs[P.root]!).body.nodes[k]!) = false)
    (hf : ((P.defs[P.root]!).body.nodes[k]!).form = .call ∨ ((P.defs[P.root]!).body.nodes[k]!).form = .batch)
    {m : ℕ} (hm : m < ((P.defs[P.root]!).body.nodes[k]!).members) {i : ℕ}
    (hi : i < ((P.defs[P.root]!).body.nodes[k]!).args.size) {t : ℕ}
    (ht : t < ((P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).params[i]!).2.leaves) (d off : ℕ)
    (hnd : ∀ q ∈ ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts).toList, overDeclared D q = false)
    (hrc : (P.defs[P.root]!).body.recut = #[] := by assumption) :
    ∃ n j, LeafAtOwn P D k i (argIdx ((P.defs[P.root]!).body.nodes[k]!) i m t) n j ∧
      n < (P.defs[P.root]!).body.nodes.size ∧ D.contains n = false ∧
      j < ((P.defs[P.root]!).body.nodes[n]!).out.leaves ∧
      (P.defs[P.root]!).body.offsets[n + 1]! =
        (P.defs[P.root]!).body.offsets[n]! + ((P.defs[P.root]!).body.nodes[n]!).out.leaves ∧
      ∀ g, resolve P (.child P.rootScope d off k m) ⟨true, i, t⟩ = .ok g →
        g = (P.defs[P.root]!).body.offsets[n]! + j := by
  have hx := argIdx_lt_of bytes P hdec hk hf hm hi ht
  obtain ⟨n, j, hleaf, hn, hdn, hj, ho, hstep⟩ := arg_own_leaf bytes P hdec hov hlay hown hNE hk hu hi hx hnd
  exact ⟨n, j, hleaf, hn, hdn, hj, ho,
    fun g hg => run_two (rs_unit P d off k m i t hf) hstep _ (Flock.Program.resolve_run hg)⟩

/-- **No two leaves of units' parameters, whose arguments read no declared node, are one gate.** -/
theorem unit_own_inj (bytes : ByteArray) (P : Prog) (hdec : Flock.Program.decode bytes = .ok P)
    (hov : P.orderViolation = none) (hlay : Desc.bodiesLaidOut P = true) {D : List ℕ} (hown : OwnRunsOf P D)
    (hNE : NoEdge P) {k k' : ℕ} (hk : k < (P.defs[P.root]!).body.nodes.size)
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
    (hnd : ∀ q ∈ ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts).toList, overDeclared D q = false)
    (hnd' : ∀ q ∈ ((((P.defs[P.root]!).body.nodes[k']!).args[i']!).seq.parts).toList, overDeclared D q = false)
    {g : ℕ} (h : resolve P (.child P.rootScope d off k m) ⟨true, i, t⟩ = .ok g)
    (h' : resolve P (.child P.rootScope d' off' k' m') ⟨true, i', t'⟩ = .ok g) :
    k = k' ∧ m = m' ∧ i = i' ∧ t = t' := by
  have hrc := Flock.Program.resolve_recut rfl h
  obtain ⟨hroot, -, hL, -, -, -⟩ := decoded_walk bytes P hdec hov hlay
  obtain ⟨n, j, hleaf, hn, hdn, hj, ho, hg⟩ := unit_own_leaf bytes P hdec hov hlay hown hNE hk hu hf hm hi ht d off hnd
  obtain ⟨n', j', hleaf', hn', hdn', hj', ho', hg'⟩ :=
    unit_own_leaf bytes P hdec hov hlay hown hNE hk' hu' hf' hm' hi' ht' d' off' hnd'
  have hgg := (hg g h).symm.trans (hg' g h')
  obtain ⟨rfl, rfl⟩ := node_of_gate P (hL P.root hroot.1) hroot.2 hn hn' hj hj' ho ho' hgg
  obtain ⟨rfl, rfl, hxx⟩ := leafAtOwn_inj hown hk hu hi hk' hu' hi' hleaf hleaf'
  obtain ⟨-, -, hno⟩ := decoded_root bytes P hdec hrc k hk
  rcases hf with hf | hf
  · rw [argIdx_call hf, argIdx_call hf] at hxx
    rw [prim_members _ (Or.inr hf)] at hm hm'
    exact ⟨rfl, by omega, rfl, hxx⟩
  · rw [argIdx_batch hf, argIdx_batch hf] at hxx
    rw [bs_members _ (Or.inl hf)] at hm hm'
    have hany : ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts).any (overDeclared D) = false := by
      cases hc : ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts).any (overDeclared D)
      · rfl
      · rw [Array.any_eq_true'] at hc
        obtain ⟨q, hq, hd⟩ := hc
        rw [hnd q (Array.mem_toList_iff.2 hq)] at hd
        cases hd
    have hd : (((P.defs[P.root]!).body.nodes[k]!).args[i]!).dims.isSome = true := by
      refine ownBatch_dims hown (n := (P.defs[P.root]!).body.nodes[k]!) ?_ hf ?_ hany
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

end InferenceOnlyVerifier.Properties.SecurityProofs.Resolve

namespace InferenceOnlyVerifier.Properties.Record.SecurityProofs

open Flock.Program (Prog Scope Ref Ty Defn)
open Flock.TemplateQuery (TRange)
open Spot.RecordText (Row)
open InferenceOnlyVerifier.Properties.SecurityProofs

/-- **Where a row's leaf is**, as `ownRows_holds` reads it: the leaf is leaf `t` of parameter `i` of member `m` of the
root node of range `j`, a call or a batch that is no input node, at row `r.row` of the parameter's cut. -/
structure RowPos (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (r : Row) (l g lo j m i t : ℕ) : Prop where
  hj : j < rs.size
  htj : rs[j]!.template = r.tid
  huj : r.unit = ((rs.extract 0 j).filter fun r' => r'.template == r.tid).foldl (fun a r' => a + r'.count) 0 + m
  hk : rs[j]!.node < (P.defs[P.root]!).body.nodes.size
  hnode : Desc.isInputNode P ((P.defs[P.root]!).body.nodes[rs[j]!.node]!) = false
  hform : ((P.defs[P.root]!).body.nodes[rs[j]!.node]!).form = .call ∨
    ((P.defs[P.root]!).body.nodes[rs[j]!.node]!).form = .batch
  hm : m < ((P.defs[P.root]!).body.nodes[rs[j]!.node]!).members
  hi : i < ((P.defs[P.root]!).body.nodes[rs[j]!.node]!).args.size
  hip : i < (P.defs[((P.defs[P.root]!).body.nodes[rs[j]!.node]!).fn]!).params.size
  hu : UnitAt P rs r.tid r.unit (P.child P.rootScope rs[j]!.node m)
  hname : ((P.defs[((P.defs[P.root]!).body.nodes[rs[j]!.node]!).fn]!).params[i]!).1 = r.port
  hrow : r.row < W.cutsOf r.tid r.port
  hl : l < ((P.defs[((P.defs[P.root]!).body.nodes[rs[j]!.node]!).fn]!).params[i]!).2.leaves / W.cutsOf r.tid r.port
  ht_eq : t = r.row * (((P.defs[((P.defs[P.root]!).body.nodes[rs[j]!.node]!).fn]!).params[i]!).2.leaves /
    W.cutsOf r.tid r.port) + l
  ht : t < ((P.defs[((P.defs[P.root]!).body.nodes[rs[j]!.node]!).fn]!).params[i]!).2.leaves
  hres : Flock.Program.resolve P (P.child P.rootScope rs[j]!.node m) ⟨true, i, t⟩ = .ok g
  hlo : lo = bitAt (inRow (P.defs[((P.defs[P.root]!).body.nodes[rs[j]!.node]!).fn]!) i (W.cutsOf r.tid r.port) r.row).2 l

/-- **Every leaf of a row has a place** (`RowPos`). -/
theorem row_pos {P : Prog} {ts : Array String} {rs : Array TRange} {W : Spot.Wires.Wiring} (hP : ReadProg P)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) {r : Row} {l g lo hi : ℕ}
    (h : RowLeaf P rs W r l g lo hi) : ∃ j m i t, RowPos P rs W r l g lo j m i t := by
  obtain ⟨sc, hu, i, hip, hname, hrow, hl, hres, hlo, -⟩ := h
  obtain ⟨j, m, hj, htj, huj, rfl, hk, hnode, hform, hm, hargs⟩ := unit_place hP hrs hu
  have hsz := inRow_size ((P.defs[((P.defs[P.root]!).body.nodes[rs[j]!.node]!).fn]!)) i (W.cutsOf r.tid r.port) r.row hrow
  have hlt := inRow_lt ((P.defs[((P.defs[P.root]!).body.nodes[rs[j]!.node]!).fn]!)) i (W.cutsOf r.tid r.port) r.row l hrow hl
  exact ⟨j, m, i, (inRow ((P.defs[((P.defs[P.root]!).body.nodes[rs[j]!.node]!).fn]!)) i (W.cutsOf r.tid r.port) r.row).1 + l,
    ⟨hj, htj, huj, hk, hnode, hform, hm, Nat.lt_of_lt_of_eq hip hargs.symm, hip, hu, hname, hrow,
      Nat.lt_of_lt_of_eq hl hsz, rfl, hlt, hres, hlo⟩⟩

/-- **Two leaves at one member and one parameter leaf of one root node are one leaf of one row.** -/
theorem rows_eq_of_pos {P : Prog} {ts : Array String} {rs : Array TRange} {W : Spot.Wires.Wiring} (hP : ReadProg P)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) {r r' : Row}
    {l l' g g' lo lo' j j' m i t : ℕ} (p : RowPos P rs W r l g lo j m i t) (p' : RowPos P rs W r' l' g' lo' j' m i t)
    (hk : rs[j]!.node = rs[j']!.node) : r = r' ∧ l = l' := by
  obtain ⟨hr, -, -, -⟩ := read_bodies hP
  have hir := readProg_inputs hP
  obtain ⟨-, -, -, -, -, -, hfirst⟩ := hP
  have hjj : j = j' := by
    rcases Nat.lt_trichotomy j j' with h | h | h
    · have := ranges_sorted hr hfirst hir hrs p.hj p'.hj h
      omega
    · exact h
    · have := ranges_sorted hr hfirst hir hrs p'.hj p.hj h
      omega
  subst hjj
  have htid : r.tid = r'.tid := p.htj.symm.trans p'.htj
  have hport : r.port = r'.port := p.hname.symm.trans p'.hname
  have hunit : r.unit = r'.unit := by rw [p.huj, p'.huj, htid]
  have hcut : W.cutsOf r'.tid r'.port = W.cutsOf r.tid r.port := by rw [htid, hport]
  have hl' := p'.hl
  have he' := p'.ht_eq
  rw [hcut] at hl' he'
  obtain ⟨hrr, hll⟩ := Resolve.mul_add_inj p.hl hl' (p.ht_eq.symm.trans he')
  rcases r with ⟨a, b, c, d⟩
  rcases r' with ⟨a', b', c', d'⟩
  simp only [Row.mk.injEq]
  exact ⟨⟨htid, hunit, hport, hrr⟩, hll⟩

end InferenceOnlyVerifier.Properties.Record.SecurityProofs
