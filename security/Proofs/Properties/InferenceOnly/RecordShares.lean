module

public import Proofs.Properties.InferenceOnly.RecordShareDecl

/-!
# Two rows that share a gate are one row, or declared at one bit

`declaredShares_of_read`: on a record `read` accepts of a root no unit of which reads an edge, two reader rows that
have no writer and share a gate are one row and one leaf, or the record declares both rows at the same bit of one
registered read or table read (`Properties.Record.OwnOrDeclaredRows`' second half).

The gate a leaf of a unit's parameter is, is a leaf of one input node of the root. Where the argument has no part over
a declared node, it is a leaf of one run `own-rows` reads, at one place (`Resolve.unit_own_inj`). Where it has one, the
argument is that one part (`stretchOf`), `nodes` has checked the declarations that fill its rows (`nodesHeld`), and the
bit of the registered read or table the row is declared at is the node's leaf's (`decl_row`).
-/

@[expose] public section

namespace InferenceOnlyVerifier.Properties.Record.SecurityProofs

open Lean (Json)
open Flock.Program (Prog)
open Flock.TemplateQuery (TRange)
open Spot.RecordText (Row Tmpl NodeFill templatesOf)
open InferenceOnlyVerifier.Record (overDeclared)
open InferenceOnlyVerifier.Record.SecurityProofs (NoEdge OwnRunsOf FinishTemplates read_front finish_text front_text
  program_of_read nodes_of_read ownRunsOf_iff)
open InferenceOnlyVerifier.Properties.SecurityProofs

/-- **Where two rows that have no writer share a gate, they are one row and one leaf, or the record declares both at
one bit of one read or table**: `OwnOrDeclaredRows`' second half. -/
def DeclaredShares (P : Prog) (doc : Json) (rs : Array TRange) (W : Spot.Wires.Wiring) (S : Spot.Wires.Summary) :
    Prop :=
  ∀ r l lo hi r' l' lo' hi' g, RowLeaf P rs W r l g lo hi → RowLeaf P rs W r' l' g lo' hi' →
    S.writerOf r.tid r.unit r.port r.row = none → S.writerOf r'.tid r'.unit r'.port r'.row = none →
      (r = r' ∧ l = l') ∨ ∃ d b b', Spot.RecordText.declaredAt P doc r = some (d, b) ∧
        Spot.RecordText.declaredAt P doc r' = some (d, b') ∧ b + lo = b' + lo'

/-- Two gates of the root's input nodes that are one gate are one leaf of one node. -/
theorem gate_node_eq {P : Prog} (hP : ReadProg P) {n n' j j' : Nat} (hn : n < (P.defs[P.root]!).body.nodes.size)
    (hn' : n' < (P.defs[P.root]!).body.nodes.size) (hj : j < ((P.defs[P.root]!).body.nodes[n]!).out.leaves)
    (hj' : j' < ((P.defs[P.root]!).body.nodes[n']!).out.leaves)
    (ho : (P.defs[P.root]!).body.offsets[n + 1]! =
      (P.defs[P.root]!).body.offsets[n]! + ((P.defs[P.root]!).body.nodes[n]!).out.leaves)
    (ho' : (P.defs[P.root]!).body.offsets[n' + 1]! =
      (P.defs[P.root]!).body.offsets[n']! + ((P.defs[P.root]!).body.nodes[n']!).out.leaves)
    (h : (P.defs[P.root]!).body.offsets[n]! + j = (P.defs[P.root]!).body.offsets[n']! + j') : n = n' ∧ j = j' := by
  have hrc := Walk.walkable_recut P hP.2.2.2.2.1
  obtain ⟨hov, hlay⟩ := read_walk hP
  obtain ⟨⟨bytes, hdec⟩, -⟩ := hP
  obtain ⟨hroot, -, hL, -, -, -⟩ := Resolve.decoded_walk bytes P hdec hov hlay
  exact Resolve.node_of_gate P (hL P.root hroot.1) hroot.2 hn hn' hj hj' ho ho' h

/-- **The template query accepts the front's template ids with the ranges it gives the wiring's.** -/
theorem evaluate_front {f : InferenceOnlyVerifier.Record.Front} {pin : String → Option String}
    {c : InferenceOnlyVerifier.Record.Case} (hFT : FinishTemplates) (hc : InferenceOnlyVerifier.Record.finish f pin = .ok c)
    {rs : Array TRange}
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg f.P) (c.W.templates.map (·.1)).toArray = .ok rs) :
    Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg f.P) (f.ts.map (·.tid)).toArray = .ok rs := by
  obtain ⟨hp1, hp2⟩ := hFT f pin c hc
  refine evaluate_congr (fun t => ?_) hrs
  apply Bool.eq_iff_iff.2
  rw [Array.contains_iff_mem, Array.contains_iff_mem]
  constructor
  · intro ht
    obtain ⟨⟨tid, t'⟩, hm, he0⟩ := List.mem_map.1 (List.mem_toArray.1 ht)
    have he0' : tid = t := he0
    subst he0'
    obtain ⟨tm, htm, he, -⟩ := hp1 tid t' hm
    exact List.mem_toArray.2 (List.mem_map.2 ⟨tm, htm, he⟩)
  · intro ht
    obtain ⟨tm, htm, rfl⟩ := List.mem_map.1 (List.mem_toArray.1 ht)
    obtain ⟨t', ht'⟩ := hp2 tm htm
    exact List.mem_toArray.2 (List.mem_map.2 ⟨(tm.tid, t'), ht', rfl⟩)

/-- **Two rows that have no writer and share a gate are one row and one leaf, or are declared at one bit**, of a Program
whose rows' arguments the record's `nodes` and `own-rows` hold. -/
theorem declaredShares_core {T : InferenceOnlyVerifier.Theta.Trusted} {P : Prog} {doc : Json} {ts : List Tmpl}
    {W : Spot.Wires.Wiring} {x : Spot.RecordText.Inputs} {ns : List Spot.RecordText.NodeDecl} {rs : Array TRange}
    (hP : ReadProg P) (hts : templatesOf P doc = .ok ts) (hnodes : InferenceOnlyVerifier.Record.nodesHeld T P doc ts W x ns = true)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) (ts.map (·.tid)).toArray = .ok rs)
    (hown : OwnRunsOf P (ns.map (·.node))) (hNE : NoEdge P)
    (hdis : Spot.Wires.pairwiseDisjoint (x.decls.map (·.w)) = true)
    (hshape : ∀ d ∈ x.decls, ∀ p, d.fill = Spot.RecordText.Fill.read p → ReadShaped d.w)
    {r r' : Row} {l l' lo lo' hi hi' g : Nat} (h1 : RowLeaf P rs W r l g lo hi) (h2 : RowLeaf P rs W r' l' g lo' hi') :
    (r = r' ∧ l = l') ∨ ∃ d b b', x.declaredAt r = some (d, b) ∧ x.declaredAt r' = some (d, b') ∧ b + lo = b' + lo' := by
  have hrc := Walk.walkable_recut P hP.2.2.2.2.1
  obtain ⟨hov, hlay⟩ := read_walk hP
  have hP' := hP
  obtain ⟨⟨bytes, hdec⟩, -⟩ := hP'
  obtain ⟨j1, m1, i1, t1, p1⟩ := row_pos hP hrs h1
  obtain ⟨j2, m2, i2, t2, p2⟩ := row_pos hP hrs h2
  have own1 : (((((P.defs[P.root]!).body.nodes[rs[j1]!.node]!).args[i1]!).seq.parts).toList.find?
      (overDeclared (ns.map (·.node))) = none) → ∀ q ∈ ((((P.defs[P.root]!).body.nodes[rs[j1]!.node]!).args[i1]!).seq.parts).toList,
        overDeclared (ns.map (·.node)) q = false := fun hf q hq => by
    have := List.find?_eq_none.1 hf q hq
    simpa using this
  have own2 : (((((P.defs[P.root]!).body.nodes[rs[j2]!.node]!).args[i2]!).seq.parts).toList.find?
      (overDeclared (ns.map (·.node))) = none) → ∀ q ∈ ((((P.defs[P.root]!).body.nodes[rs[j2]!.node]!).args[i2]!).seq.parts).toList,
        overDeclared (ns.map (·.node)) q = false := fun hf q hq => by
    have := List.find?_eq_none.1 hf q hq
    simpa using this
  rcases hf1 : ((((P.defs[P.root]!).body.nodes[rs[j1]!.node]!).args[i1]!).seq.parts).toList.find?
      (overDeclared (ns.map (·.node))) with _ | part1
  · rcases hf2 : ((((P.defs[P.root]!).body.nodes[rs[j2]!.node]!).args[i2]!).seq.parts).toList.find?
        (overDeclared (ns.map (·.node))) with _ | part2
    · have hnd1 := own1 hf1
      have hnd2 := own2 hf2
      obtain ⟨hk, hmm, hii, htt⟩ := Resolve.unit_own_inj bytes P hdec hov hlay hown hNE p1.hk p2.hk p1.hnode p2.hnode
        p1.hform p2.hform p1.hm p2.hm p1.hi p2.hi p1.ht p2.ht
        ((P.defs[P.root]!).body.nodes[rs[j1]!.node]!).fn (P.child P.rootScope rs[j1]!.node m1).off
        ((P.defs[P.root]!).body.nodes[rs[j2]!.node]!).fn (P.child P.rootScope rs[j2]!.node m2).off hnd1 hnd2
        p1.hres p2.hres
      subst hmm
      subst hii
      subst htt
      exact Or.inl (rows_eq_of_pos hP hrs p1 p2 hk)
    · exfalso
      have hnd1 := own1 hf1
      obtain ⟨n, z1, -, hn1, hdn1, hz1, ho1, hg1⟩ := Resolve.unit_own_leaf bytes P hdec hov hlay hown hNE p1.hk p1.hnode
        p1.hform p1.hm p1.hi p1.ht ((P.defs[P.root]!).body.nodes[rs[j1]!.node]!).fn
        (P.child P.rootScope rs[j1]!.node m1).off hnd1
      have hgn := hg1 g p1.hres
      obtain ⟨n', z2, w, hn2, hdn2, hz2, ho2, hgn2, -⟩ := decl_row hP hts hnodes hrs hdis hshape p2 hf2
      obtain ⟨hnn, -⟩ := gate_node_eq hP hn1 hn2 hz1 hz2 ho1 ho2 (hgn.symm.trans hgn2)
      subst hnn
      rw [hdn1] at hdn2
      cases hdn2
  · rcases hf2 : ((((P.defs[P.root]!).body.nodes[rs[j2]!.node]!).args[i2]!).seq.parts).toList.find?
        (overDeclared (ns.map (·.node))) with _ | part2
    · exfalso
      have hnd2 := own2 hf2
      obtain ⟨n, z2, -, hn2, hdn2, hz2, ho2, hg2⟩ := Resolve.unit_own_leaf bytes P hdec hov hlay hown hNE p2.hk p2.hnode
        p2.hform p2.hm p2.hi p2.ht ((P.defs[P.root]!).body.nodes[rs[j2]!.node]!).fn
        (P.child P.rootScope rs[j2]!.node m2).off hnd2
      have hgn := hg2 g p2.hres
      obtain ⟨n', z1, w, hn1, hdn1, hz1, ho1, hgn1, -⟩ := decl_row hP hts hnodes hrs hdis hshape p1 hf1
      obtain ⟨hnn, -⟩ := gate_node_eq hP hn1 hn2 hz1 hz2 ho1 ho2 (hgn1.symm.trans hgn)
      subst hnn
      rw [hdn2] at hdn1
      cases hdn1
    · obtain ⟨n, z, w, hn1, hd1, hz1, ho1, hg1, hh1, hcase1⟩ := decl_row hP hts hnodes hrs hdis hshape p1 hf1
      obtain ⟨n', z', w', hn2, hd2, hz2, ho2, hg2, hh2, hcase2⟩ := decl_row hP hts hnodes hrs hdis hshape p2 hf2
      obtain ⟨hnn, hzz⟩ := gate_node_eq hP hn1 hn2 hz1 hz2 ho1 ho2 (hg1.symm.trans hg2)
      subst hnn
      subst hzz
      have hww : w = w' := by
        rw [hh1] at hh2
        exact Option.some.inj hh2
      subst hww
      right
      rcases hcase1 with ⟨pp, o, bits, hfl, hdecl, heq⟩ | ⟨name, root, tb, hfl, htb, hdecl, hlo⟩
      · rcases hcase2 with ⟨pp', o', bits', hfl', hdecl', heq'⟩ | ⟨name', root', tb', hfl', htb', hdecl', hlo'⟩
        · obtain ⟨rfl, rfl⟩ := NodeFill.param.inj (Option.some.inj (hfl.symm.trans hfl'))
          refine ⟨.read pp, bits, bits', hdecl, hdecl', ?_⟩
          have : (bits : Int) + lo = bits' + lo' := by rw [heq, heq']
          exact_mod_cast this
        · exact absurd (Option.some.inj (hfl.symm.trans hfl')) (by intro h; cases h)
      · rcases hcase2 with ⟨pp', o', bits', hfl', hdecl', heq'⟩ | ⟨name', root', tb', hfl', htb', hdecl', hlo'⟩
        · exact absurd (Option.some.inj (hfl.symm.trans hfl')) (by intro h; cases h)
        · obtain ⟨rfl, rfl⟩ := NodeFill.table.inj (Option.some.inj (hfl.symm.trans hfl'))
          obtain rfl : tb = tb' := Option.some.inj (htb.symm.trans htb')
          exact ⟨_, 0, 0, hdecl, hdecl', by rw [hlo, hlo']⟩

/-- **On a record `read` accepts of a root no unit of which reads an edge, two rows that have no writer and share a
gate are one row and one leaf, or are declared at one bit.** -/
theorem declaredShares_of_read (hFT : FinishTemplates) {j tr : Json} {c : InferenceOnlyVerifier.Record.Case}
    (h : InferenceOnlyVerifier.Record.read j tr = .ok c) (hNE : NoEdge c.f.P) {rs : Array TRange}
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg c.f.P)
      (c.W.templates.map (·.1)).toArray = .ok rs) :
    DeclaredShares c.f.P c.f.doc rs c.W c.wired := by
  obtain ⟨f, pin, hf, hc⟩ := read_front h
  obtain ⟨hfc, -, text, htext, hx, -, -⟩ := finish_text hc
  have hrs' := evaluate_front hFT hc (hfc ▸ hrs)
  subst hfc
  have hts : templatesOf c.f.P c.f.doc = .ok c.f.ts := front_text hf
  have hP := readProg_of_read h
  have hnodes := nodes_of_read h
  have hown : OwnRunsOf c.f.P (c.f.nodes.map (·.node)) := ownRunsOf_iff.1 (program_of_read h).2.2.2.2.1
  have hdis := (inputsOf_decls hx).1
  have hshape := inputsOf_reads hx
  have hda : ∀ r, Spot.RecordText.declaredAt c.f.P c.f.doc r = c.x.declaredAt r := by
    intro r
    simp [Spot.RecordText.declaredAt, hts, htext, hx]
  intro r l lo hi r' l' lo' hi' g h1 h2 _ _
  rw [hda r, hda r']
  exact declaredShares_core hP hts hnodes hrs' hown hNE hdis hshape h1 h2

end InferenceOnlyVerifier.Properties.Record.SecurityProofs

namespace InferenceOnlyVerifier.Record.SecurityProofs

/-- **Every record `read` accepts is of a root no unit of which reads an edge.** False of any root with an edge; it
holds of a root whose units read the input nodes the record declares. Where it holds, so does the record's wiring have
no wire. -/
def NoEdgeReads : Prop :=
  ∀ (j tr : Lean.Json) (c : Case), read j tr = .ok c → NoEdge c.f.P

theorem noEdgeReads_of_edgeFreeReads (h : EdgeFreeReads) : NoEdgeReads :=
  fun j tr c hr => noEdge_of_edgeFree (h j tr c hr)

end InferenceOnlyVerifier.Record.SecurityProofs
