import Proofs.Properties.NetworkScan.Query
import Proofs.Properties.NetworkScan.Accept

/-!
# The template query's reading of a decoded Program, at its template nodes

The sampled verifier's template query reads a decoded Program `P` as `Flock.TemplateQuery.Desc.ofProg P`: its
Definitions by key, and each root node its form's name, its callee by the decoder's order and its count. Where no two of
`P`'s Definitions share a key, the root it finds by key is `P`'s, and at a template node
(`NetworkCheck.Period.templateNodes`: a root node whose callee is a composite) it reads the callee's key as the node's
template and a batch's members as its count.

* `ddefOf`, `def?_ofProg_root`: the root the query reads.
* `templateNode_lt`, `tidAt_ofProg`, `countAt_ofProg`, `isInput_ofProg`: a template node is a root node, and the
  query's template there is its callee's key, its count a batch's members.
-/

namespace Properties.NetworkScan.Query

open Flock.TemplateQuery (Desc DDef)
open NetworkCheck.Period (templateNodes)
open Properties.NetworkScan.Fits (node callee)

/-- A decoded Definition as the template query reads it: whether it is a primitive, and each body node's form's name,
callee and count. -/
def ddefOf (d : Flock.Program.Defn) : DDef :=
  { prim := d.prim, nodes := d.body.nodes.map fun x => { form := x.form.name, fn := x.fn, n := x.n } }

theorem getElem!_map_of_lt {α β : Type} [Inhabited α] [Inhabited β] {xs : Array α} {f : α → β} {i : ℕ}
    (h : i < xs.size) : (xs.map f)[i]! = f xs[i]! := by
  rw [getElem!_pos (xs.map f) i (by simpa using h), Array.getElem_map, getElem!_pos xs i h]

/-- **The root the query reads is `P`'s.** Where no two of `P`'s Definitions share a key, the one the query finds at its
root key is `P`'s root. -/
theorem def?_ofProg_root {P : Flock.Program.Prog} (hk : (P.defs.toList.map (·.key)).Nodup)
    (hr : P.root < P.defs.size) :
    (Desc.ofProg P).def? (Desc.ofProg P).root = some (ddefOf (P.defs[P.root]!)) := by
  rw [getElem!_pos P.defs P.root hr]
  have hkey : (Desc.ofProg P).root = P.defs[P.root].key := by
    show (P.defs.getD P.root default).key = _
    rw [array_getD_of_lt _ hr]
  have hdefs : (Desc.ofProg P).defs = P.defs.map fun d => (d.key, ddefOf d) := rfl
  unfold Desc.def?
  rw [hkey, hdefs]
  have hmem : P.defs[P.root] ∈ P.defs := Array.getElem_mem hr
  cases hf : (P.defs.map fun d => (d.key, ddefOf d)).find? (·.1 == P.defs[P.root].key) with
  | none =>
    have := Array.find?_eq_none.1 hf _ (Array.mem_map.2 ⟨_, hmem, rfl⟩)
    simp at this
  | some r =>
    have hp := Array.find?_some hf
    obtain ⟨d, hd, rfl⟩ := Array.mem_map.1 (Array.mem_of_find?_eq_some hf)
    simp only [beq_iff_eq] at hp
    obtain rfl : d = P.defs[P.root] :=
      List.inj_on_of_nodup_map hk (Array.mem_def.1 hd) (Array.mem_def.1 hmem) hp
    rfl

/-- A template node is a node of `P`'s root. -/
theorem templateNode_lt {P : Flock.Program.Prog} {k : ℕ} (hk : k ∈ templateNodes P) :
    k < (P.defs[P.root]!).body.nodes.size := by
  simp only [templateNodes, List.mem_filter, List.mem_range] at hk
  exact hk.1

/-- A template node's callee is one of `P`'s Definitions. -/
theorem templateNode_fn {P : Flock.Program.Prog} {k : ℕ} (hk : k ∈ templateNodes P) :
    (node P k).fn < P.defs.size := by
  simp only [templateNodes, List.mem_filter, List.mem_range] at hk
  obtain ⟨-, h2⟩ := hk
  split at h2
  · rename_i d hd
    exact (Array.getElem?_eq_some_iff.1 hd).1
  · cases h2

/-- **The query's template at a template node** is its callee's key. -/
theorem tidAt_ofProg {P : Flock.Program.Prog} {k : ℕ} (hk : k ∈ templateNodes P) :
    tidAt (Desc.ofProg P) (ddefOf (P.defs[P.root]!)) k = (callee P k).key := by
  have hfn := templateNode_fn hk
  dsimp only [tidAt, ddefOf]
  rw [getElem!_map_of_lt (templateNode_lt hk)]
  show (P.defs.map (·.key)).getD (node P k).fn "" = (P.defs[(node P k).fn]!).key
  have hfn' : (node P k).fn < (P.defs.map (·.key)).size := by simpa using hfn
  rw [array_getD_of_lt _ hfn', Array.getElem_map, getElem!_pos P.defs _ hfn]

/-- The query reads no template node as an input's, where its callee isn't keyed as one. -/
theorem isInput_ofProg {P : Flock.Program.Prog} {k : ℕ} (hk : k ∈ templateNodes P)
    (hin : Flock.Program.isInputId (callee P k).key = false) :
    Flock.TemplateQuery.isInput (tidAt (Desc.ofProg P) (ddefOf (P.defs[P.root]!)) k) = false := by
  rw [tidAt_ofProg hk]
  exact hin

/-- **The query's count at a batch node** is the batch's members. -/
theorem countAt_ofProg {P : Flock.Program.Prog} {k : ℕ} (hk : k < (P.defs[P.root]!).body.nodes.size)
    (hb : (node P k).form = .batch) : countAt (ddefOf (P.defs[P.root]!)) k = (node P k).n := by
  dsimp only [countAt, ddefOf]
  rw [getElem!_map_of_lt hk]
  show (if (node P k).form.name == "batch" then (node P k).n else 1) = (node P k).n
  rw [hb]
  simp [Flock.Program.Form.name]

end Properties.NetworkScan.Query
