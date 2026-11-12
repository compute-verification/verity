module

public import Proofs.Properties.Nci.LeafWidthsDecoded
public import Proofs.Properties.Nci.LeafWidthsResolve
public import Proofs.Properties.Nci.RecordFields
public import Proofs.Properties.Nci.RecordPrim
public import Proofs.Properties.Nci.RecordRowBits
public import Proofs.Properties.Nci.RecordTakes

/-!
# A gate is as wide as the leaf that names it

`LeafWide` of a Program the record check reads: a leaf of a reader row, and a leaf of a unit's output row, is a gate
whose primitive returns one value of the leaf's width.

* A leaf of a unit's parameter (`param_wide`). The unit is a member of a root node; the leaf resolves, one step up,
  through a reference of the node's argument, which the root resolves to a gate of an input node
  (`Resolve.unit_leaf`, `Resolve.unit_leaf_ok`). That gate is as wide as the leaf the reference names in the root's
  body (`LeafWidths.resolve_wide`), which is as wide as the parameter's leaf (`LeafWidths.arg_wide`).
* A leaf of a unit's output row (`outLeaf_wide`). A unit of a primitive has one, its own gate, of the primitive's one
  value (`outLeaf_prim`). A composite unit returns the leaf at a reference of its body's return, which names a leaf
  as wide as the return type's (`LeafWidths.BodyWide`), and the return's leaf is the field's (`field_leaf`).
* `leafWide`: both.
-/

@[expose] public section

namespace NciCheck.Properties.SecurityProofs.Resolve

open Flock.Program
open NciCheck.Record.SecurityProofs (batch_dims)

/-- **The reference a member reads a leaf of its parameter at is a reference of the node's argument**: member `m`
of root node `k`, a call or a batch that is no input node, reads leaf `t` of its parameter `i` at a reference within
the node's argument `i`. -/
theorem argIdx_lt (bytes : ByteArray) (P : Prog) (hdec : Flock.Program.decode bytes = .ok P)
    (hown : Record.ownRows P = true) {k : ℕ} (hk : k < (P.defs[P.root]!).body.nodes.size)
    (hu : Desc.isInputNode P ((P.defs[P.root]!).body.nodes[k]!) = false)
    (hf : ((P.defs[P.root]!).body.nodes[k]!).form = .call ∨ ((P.defs[P.root]!).body.nodes[k]!).form = .batch)
    {m : ℕ} (hm : m < ((P.defs[P.root]!).body.nodes[k]!).members) {i : ℕ}
    (hi : i < ((P.defs[P.root]!).body.nodes[k]!).args.size) {t : ℕ}
    (ht : t < ((P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).params[i]!).2.leaves) :
    argIdx ((P.defs[P.root]!).body.nodes[k]!) i m t <
      ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq).count := by
  obtain ⟨-, hc, hno⟩ := decoded_root bytes P hdec k hk
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

/-- A reference whose first step of resolution ends resolves to where the step ends. -/
theorem resolve_one {P : Prog} {sc : Scope} {r : Ref} {g : ℕ} (h : resolveStep P (sc, r) = .ok (.inl g)) :
    resolve P sc r = .ok g := by
  obtain ⟨n, hn⟩ : ∃ n, Flock.Steps.FUEL = n + 1 :=
    ⟨Flock.Steps.FUEL - 1, by unfold Flock.Steps.FUEL; omega⟩
  unfold resolve
  rw [hn]
  exact run_one h n

end NciCheck.Properties.SecurityProofs.Resolve

namespace NciCheck.Properties.SecurityProofs.LeafWidths

open Flock.Program

/-- The type a body's sources give a reference in range is the type the reference names. -/
theorem srcOf_ty (B : Body) (below : ℕ) (r : Ref) (hp : r.p = true → r.i < B.params.size)
    (hn : r.p = false → r.i < below ∧ r.i < B.nodes.size) :
    (srcOf B below).ty? r.p r.i = some (if r.p then (B.params[r.i]!).2 else (B.nodes[r.i]!).out) := by
  unfold Widths.Src.ty? srcOf
  cases hrp : r.p
  · obtain ⟨h1, h2⟩ := hn hrp
    simp only [Bool.false_eq_true, ↓reduceIte, h1]
    rw [Array.getElem?_map, getElem?_pos B.nodes r.i h2, getElem!_pos B.nodes r.i h2]
    rfl
  · have h1 := hp hrp
    simp only [↓reduceIte]
    rw [Array.getElem?_map, getElem?_pos B.params r.i h1, getElem!_pos B.params r.i h1]
    rfl

end NciCheck.Properties.SecurityProofs.LeafWidths

namespace NciCheck.Properties.Record.SecurityProofs

open Flock.Program (Prog Scope Ref Ty)
open Flock.TemplateQuery (TRange)
open NciCheck.RecordText (Row fieldsOf)
open NciCheck.Record.SecurityProofs (ranges_spec)
open NciCheck.Properties.SecurityProofs

/-- What the width lemmas read of a Program the record check reads: its root a composite Definition of its own, every
body that resolution reaches in range and laid out, what the decoder establishes of each Definition, and every
composite body reading every leaf at its width. -/
theorem read_wide {P : Prog} (hP : ReadProg P) :
    (P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false) ∧
      (∀ sc, Resolve.Good P sc → Resolve.BodyOk P sc.d) ∧ (∀ d < P.defs.size, Walk.LaidOut P d) ∧
      (∀ d < P.defs.size, BodiesOk.DefOk P.defs d P.defs[d]!) ∧
      ∀ d < P.defs.size, (P.defs[d]!).prim = false →
        (P.defs[d]!).body.retType = (P.defs[d]!).ret ∧ LeafWidths.BodyWide (P.defs[d]!).body := by
  obtain ⟨hr, hrp, hlay, hB⟩ := read_bodies hP
  obtain ⟨⟨bytes, hdec⟩, -⟩ := hP
  obtain ⟨st, hPd, hinv, -⟩ := BodiesOk.decode_spec bytes P hdec
  have hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hPd]; exact hinv.1
  exact ⟨⟨hr, hrp⟩, hB, hlay, hok, LeafWidths.decoded_wide bytes P hdec⟩

/-- **A unit is a member of a root node that is a call or a batch and no input node**, with an argument for each
of the unit's Definition's parameters. -/
theorem unit_member {P : Prog} {ts : Array String} {rs : Array TRange} (hP : ReadProg P)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) {tid : String} {u : ℕ}
    {sc : Scope} (hu : UnitAt P rs tid u sc) :
    ∃ k m, k < (P.defs[P.root]!).body.nodes.size ∧
      Desc.isInputNode P ((P.defs[P.root]!).body.nodes[k]!) = false ∧
      (((P.defs[P.root]!).body.nodes[k]!).form = .call ∨ ((P.defs[P.root]!).body.nodes[k]!).form = .batch) ∧
      m < ((P.defs[P.root]!).body.nodes[k]!).members ∧
      (∀ i, i < (P.defs[sc.d]!).params.size → i < ((P.defs[P.root]!).body.nodes[k]!).args.size) ∧
      sc = P.child P.rootScope k m := by
  obtain ⟨hr, -, -, -⟩ := read_bodies hP
  have hir := readProg_inputs hP
  obtain ⟨⟨bytes, hdec⟩, -, -, -, -, -, hfirst⟩ := hP
  obtain ⟨-, hall⟩ := ranges_spec hr hfirst hir hrs
  obtain ⟨k, m, hk, hkey, -, hm, rfl⟩ := unit_node hr hfirst hir hrs hu
  obtain ⟨-, -, hno⟩ := Resolve.decoded_root bytes P hdec k hk
  have hnode : Desc.isInputNode P ((P.defs[P.root]!).body.nodes[k]!) = false := by
    simp [Desc.isInputNode, hkey]
  have hform : ((P.defs[P.root]!).body.nodes[k]!).form = .call ∨
      ((P.defs[P.root]!).body.nodes[k]!).form = .batch := by
    obtain ⟨j, m', hj, -, -, -, hsc⟩ := hu
    have hmem : rs[j]! ∈ rs.toList := by
      rw [getElem!_pos rs j hj]
      simp
    obtain ⟨-, -, hfm, -⟩ := hall _ hmem
    have hkj : k = rs[j]!.node := by
      simp only [Prog.child, Scope.child.injEq] at hsc
      exact hsc.2.2.2.1
    rw [hkj]
    exact hfm
  exact ⟨k, m, hk, hnode, hform, hm, fun i hi => Nat.lt_of_lt_of_eq hi hno.1.symm, rfl⟩

/-- A leaf of a root node's output is a gate of the Program. -/
theorem root_leaf_lt {P : Prog} (hP : ReadProg P) {n j : ℕ} (hn : n < (P.defs[P.root]!).body.nodes.size)
    (hj : j < ((P.defs[P.root]!).body.nodes[n]!).out.leaves)
    (ho : (P.defs[P.root]!).body.offsets[n + 1]! =
      (P.defs[P.root]!).body.offsets[n]! + ((P.defs[P.root]!).body.nodes[n]!).out.leaves) :
    (P.defs[P.root]!).body.offsets[n]! + j < (P.defs[P.root]!).gates := by
  obtain ⟨hr, hrp, hlay, -⟩ := read_bodies hP
  have hmono := Resolve.offsets_mono P P.root (hlay P.root hr) hrp (n + 1) (P.defs[P.root]!).body.nodes.size
    (by omega) (Nat.le_refl _)
  obtain ⟨-, hend, -⟩ := hlay P.root hr hrp
  omega

/-- A gate inside a member of a root node is a gate of the Program. -/
theorem unit_gate_lt {P : Prog} (hP : ReadProg P) {k m : ℕ} (hk : k < (P.defs[P.root]!).body.nodes.size)
    (hm : m < ((P.defs[P.root]!).body.nodes[k]!).members) {g : ℕ}
    (hg : g < (P.child P.rootScope k m).off + (P.defs[(P.child P.rootScope k m).d]!).gates) :
    g < (P.defs[P.root]!).gates := by
  obtain ⟨hr, hrp, hlay, -⟩ := read_bodies hP
  obtain ⟨-, h2⟩ := unit_within P (hlay P.root hr) hrp hk hm
  have hmono := Resolve.offsets_mono P P.root (hlay P.root hr) hrp (k + 1) (P.defs[P.root]!).body.nodes.size
    (by omega) (Nat.le_refl _)
  obtain ⟨-, hend, -⟩ := hlay P.root hr hrp
  omega

/-- A leaf of a slice of an array is the array's leaf that far past the slice's start. -/
theorem extract_get (ws : Array ℕ) (a b l : ℕ) (hl : l < (ws.extract a b).size) :
    (ws.extract a b)[l]! = ws[a + l]! := by
  have h2 : a + l < ws.size := by
    rw [Array.size_extract] at hl
    omega
  rw [getElem!_pos (ws.extract a b) l hl, getElem!_pos ws (a + l) h2, Array.getElem_extract]

/-- **A leaf of a unit's parameter is a gate as wide as the leaf.** The leaf resolves through a reference of the
unit's root node's argument to a gate of an input node; the gate is as wide as the leaf the reference names in the
root's body, and that leaf as wide as the parameter's. -/
theorem param_wide {P : Prog} {ts : Array String} {rs : Array TRange} (hP : ReadProg P)
    (hO : NciCheck.Record.ownRows P = true)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) {tid : String} {u : ℕ}
    {sc : Scope} (hu : UnitAt P rs tid u sc) {i t g : ℕ} (hi : i < (P.defs[sc.d]!).params.size)
    (ht : t < ((P.defs[sc.d]!).params[i]!).2.leaves)
    (hres : Flock.Program.resolve P sc (⟨true, i, t⟩ : Ref) = .ok g) :
    ∃ p ops sc', P.gate g = .ok (p, ops, sc') ∧
      (P.defs[p]!).ret = .value (((P.defs[sc.d]!).params[i]!).2.widths[t]!) := by
  obtain ⟨hroot, hB, hL, hD, hW⟩ := read_wide hP
  obtain ⟨hov, hlayb⟩ := read_walk hP
  obtain ⟨k, m, hk, hnode, hform, hm, hargs, rfl⟩ := unit_member hP hrs hu
  have hP' := hP
  obtain ⟨⟨bytes, hdec⟩, -⟩ := hP'
  have hi' := hargs i hi
  -- the leaf is a leaf of an input node, at a gate of the root
  obtain ⟨n, j, -, hn, hj, ho, hg⟩ := Resolve.unit_leaf bytes P hdec hov hlayb hO hk hnode hform hm hi' ht
    ((P.defs[P.root]!).body.nodes[k]!).fn (P.child P.rootScope k m).off
  have hlt : g < (P.defs[P.root]!).gates := by
    rw [hg g hres]
    exact root_leaf_lt hP hn hj ho
  obtain ⟨p, ops, sc', w, hgate, -, -⟩ := gate_located hP hlt
  -- the root resolves the reference of the node's argument to the same gate
  obtain ⟨g', hstep, hg'⟩ := Resolve.unit_leaf_ok bytes P hdec hov hlayb hO hk hnode hform hm hi' ht
    ((P.defs[P.root]!).body.nodes[k]!).fn (P.child P.rootScope k m).off
  have hgg : g' = g := Except.ok.inj (hg'.symm.trans hres)
  rw [hgg] at hstep
  have hgood : Resolve.Good P P.rootScope := ⟨rfl, rfl, rfl⟩
  have hx := Resolve.argIdx_lt bytes P hdec hO hk hnode hform hm hi' ht
  have href0 := ((hB P.rootScope hgood).1 k hk).2.1 i _ hx
  have href : Resolve.RefOk P P.rootScope.d (P.defs[P.rootScope.d]!).body.nodes.size
      ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.get
        (Resolve.argIdx ((P.defs[P.root]!).body.nodes[k]!) i m t)) :=
    ⟨href0.1, fun h => ⟨Nat.lt_trans (href0.2 h).1 hk, (href0.2 h).2⟩⟩
  have hret := LeafWidths.resolve_wide P hroot hB hL hD hW P.rootScope hgood _ href g
    (Resolve.resolve_one hstep) p ops sc' hgate
  -- the leaf the reference names is as wide as the parameter's
  have hwid : Flock.Program.Widths.wAt
      (LeafWidths.tyOf P P.rootScope ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.get
        (Resolve.argIdx ((P.defs[P.root]!).body.nodes[k]!) i m t)))
      ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.get
        (Resolve.argIdx ((P.defs[P.root]!).body.nodes[k]!) i m t)).j =
      Flock.Program.Widths.wAt ((P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).params[i]!).2 t := by
    have hw := LeafWidths.arg_wide P hroot hB hD hW P.rootScope hgood k hk m hm i hi t ht
    rcases hform with hf | hf
    · rw [Resolve.argIdx_call hf]
      exact hw.1 hf
    · rw [Resolve.argIdx_batch hf]
      exact hw.2 hf
  exact ⟨p, ops, sc', hgate, hret.trans (congrArg Ty.value (hwid.trans (LeafWidths.wAt_widths _ _ ht)))⟩

variable {P : Prog} {ts : Array String} {rs : Array TRange} {W : Spot.Wires.Wiring}

/-- **A leaf of a reader row is a gate as wide as the leaf** (`LeafWide`, its first clause). -/
theorem rowLeaf_wide (hP : ReadProg P) (hO : NciCheck.Record.ownRows P = true)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) {r : Row} {l g lo hi : ℕ}
    (h : RowLeaf P rs W r l g lo hi) :
    ∃ p ops sc, P.gate g = .ok (p, ops, sc) ∧ (P.defs[p]!).ret = .value (hi - lo) := by
  obtain ⟨sc, hu, i, hip, -, hrow, hl, hres, -, hhi⟩ := h
  have hsz := inRow_size (P.defs[sc.d]!) i (W.cutsOf r.tid r.port) r.row hrow
  have hl' : l < (P.defs[sc.d]!).params[i]!.2.leaves / W.cutsOf r.tid r.port := by
    rw [← hsz]
    exact hl
  have ht : r.row * ((P.defs[sc.d]!).params[i]!.2.leaves / W.cutsOf r.tid r.port) + l <
      ((P.defs[sc.d]!).params[i]!).2.leaves := by
    have h1 := Nat.mul_div_le (P.defs[sc.d]!).params[i]!.2.leaves (W.cutsOf r.tid r.port)
    have h2 := Nat.mul_le_mul_right ((P.defs[sc.d]!).params[i]!.2.leaves / W.cutsOf r.tid r.port)
      (show r.row + 1 ≤ W.cutsOf r.tid r.port from hrow)
    rw [Nat.add_one_mul] at h2
    omega
  obtain ⟨p, ops, sc', hgate, hret⟩ := param_wide hP hO hrs hu hip ht hres
  have he : (inRow (P.defs[sc.d]!) i (W.cutsOf r.tid r.port) r.row).2[l]! =
      (P.defs[sc.d]!).params[i]!.2.widths[
        r.row * ((P.defs[sc.d]!).params[i]!.2.leaves / W.cutsOf r.tid r.port) + l]! :=
    extract_get _ _ _ _ hl
  refine ⟨p, ops, sc', hgate, ?_⟩
  rw [hhi, Nat.add_sub_cancel_left, he]
  exact hret

/-- **A leaf of a unit's output row is a gate as wide as the leaf** (`LeafWide`, its second clause). -/
theorem outLeaf_wide (hP : ReadProg P) (hO : NciCheck.Record.ownRows P = true)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) {tid : String} {u : ℕ}
    {sc : Scope} {out : String} {l g lo hi : ℕ} (hu : UnitAt P rs tid u sc) (ho : OutLeaf P sc out l g lo hi) :
    ∃ p ops sc', P.gate g = .ok (p, ops, sc') ∧ (P.defs[p]!).ret = .value (hi - lo) := by
  by_cases hp : (P.defs[sc.d]!).prim = true
  · -- a unit of a primitive: its one output leaf is its own gate
    obtain ⟨w, hw, -, hall⟩ := outLeaf_prim hP hrs hu hp
    obtain ⟨-, -, hg0, hlo0, hhi0⟩ := hall out l g lo hi ho
    obtain ⟨k, m, hk, -, hm, hsc, -, hg1, -⟩ := prim_unit hP hrs hu hp
    have hlt : g < (P.defs[P.root]!).gates := by
      rw [hg0, hsc]
      refine unit_gate_lt hP hk hm ?_
      rw [← hsc, hg1]
      omega
    obtain ⟨p, ops, sc', w', hgate, -, -⟩ := gate_located hP hlt
    rw [hg0] at hgate
    obtain ⟨hpd, -, -⟩ := prim_unit_gate hP hrs hu hp hgate
    refine ⟨p, ops, sc', by rw [hg0]; exact hgate, ?_⟩
    rw [hpd, hw, hlo0, hhi0, Nat.sub_zero]
  · -- a composite unit: the leaf is a reference of its body's return
    have hpf : (P.defs[sc.d]!).prim = false := by simpa using hp
    obtain ⟨hroot, hB, hL, hD, hW⟩ := read_wide hP
    obtain ⟨hov, hlayb⟩ := read_walk hP
    obtain ⟨k, m, hk, hnode, hform, hm, hargs, rfl⟩ := unit_member hP hrs hu
    have hP' := hP
    obtain ⟨⟨bytes, hdec⟩, -⟩ := hP'
    have hgood : Resolve.Good P (P.child P.rootScope k m) :=
      Resolve.good_child P (parent := P.rootScope)
        (by show Resolve.Good P (.top P.root 0 none); unfold Resolve.Good; exact ⟨rfl, rfl, rfl⟩) hk rfl hpf hm rfl
    have hdlt : (P.child P.rootScope k m).d < P.defs.size := (Resolve.good_d P hroot hB _ hgood).1
    obtain ⟨hrt, hbw⟩ := hW _ hdlt hpf
    obtain ⟨hbp, hcount, -⟩ := (hD _ hdlt).1.2 hpf
    obtain ⟨f, -, hf, hl, hres, -, hhi⟩ := ho
    have hl' : l < (fieldsOf (P.defs[(P.child P.rootScope k m).d]!).ret)[f]!.leaves := by
      rw [← widths_size]
      exact hl
    obtain ⟨hfl1, hfl2⟩ := field_leaf (P.defs[(P.child P.rootScope k m).d]!).ret f l hf hl'
    have htc : (outRow (P.defs[(P.child P.rootScope k m).d]!) f).1 + l <
        (P.defs[(P.child P.rootScope k m).d]!).body.ret.count := by
      rw [hcount]
      exact hfl1
    have href := (hB _ hgood).2 _ htc
    -- the gate is one of the unit's, or a leaf of one of its parameters: a gate of the Program either way
    have hgate : ∃ p ops sc', P.gate g = .ok (p, ops, sc') := by
      rcases Resolve.ret_within_of bytes P hdec hov hlayb _ hgood _ htc hres with hin | ⟨i, j, hi', hj, hpar⟩
      · obtain ⟨p, ops, sc', w, hgate, -, -⟩ := gate_located hP (unit_gate_lt hP hk hm hin.2)
        exact ⟨p, ops, sc', hgate⟩
      · rw [hbp] at hi' hj
        obtain ⟨p, ops, sc', hgate, -⟩ := param_wide hP hO hrs hu hi' hj hpar
        exact ⟨p, ops, sc', hgate⟩
    obtain ⟨p, ops, sc', hgate⟩ := hgate
    have hret := LeafWidths.resolve_wide P hroot hB hL hD hW _ hgood _ href g hres p ops sc' hgate
    have hsw := hbw.2 _ htc _ (LeafWidths.srcOf_ty (P.defs[(P.child P.rootScope k m).d]!).body
      (P.defs[(P.child P.rootScope k m).d]!).body.nodes.size _ (fun h => (href.1 h).1)
      (fun h => ⟨(href.2 h).1, (href.2 h).1⟩))
    rw [hrt] at hsw
    refine ⟨p, ops, sc', hgate, ?_⟩
    rw [hhi, Nat.add_sub_cancel_left]
    exact hret.trans (congrArg Ty.value (hsw.trans (hfl2.trans (LeafWidths.wAt_widths _ _ hl'))))

/-- **A gate is as wide as the leaf that names it** (`LeafWide`), of a Program the record check reads whose units'
rows are their own. -/
theorem leafWide (hP : ReadProg P) (hO : NciCheck.Record.ownRows P = true)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) : LeafWide P rs W :=
  ⟨fun _ _ _ _ _ h => rowLeaf_wide hP hO hrs h, fun _ _ _ _ _ _ _ _ hu ho => outLeaf_wide hP hO hrs hu ho⟩

end NciCheck.Properties.Record.SecurityProofs
