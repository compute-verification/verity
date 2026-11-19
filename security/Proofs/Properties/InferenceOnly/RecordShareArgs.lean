module

public import Proofs.Properties.InferenceOnly.RecordTiling
public import Proofs.Properties.InferenceOnly.RecordDeclared
public import Proofs.Properties.InferenceOnly.RecordOwn

/-!
# What `nodesHeld` holds of one argument of a range, and what the text declares of a row

* `ArgOk`, `nodesHeld_arg`: the test `nodesHeld` makes of each argument of each range of a template.
* `argOk_param`, `argOk_table`: what it says of an argument that reads a declared node: the node's width, the
  range's units, the cut, and the `covers` of the declarations that fill the argument's port, a registered read of the
  node's parameter (`ReadOk`) or a table read of the node's table.
* `nodesHeld_table`: a declared node filled by a table is of a table the record lists.
* `declaredAt_read`, `declaredAt_table`: what the text declares of a row some declaration covers.
-/

@[expose] public section

namespace InferenceOnlyVerifier.Properties.Record.SecurityProofs

open InferenceOnlyVerifier.Record (covers withStarts stretchOf overDeclared nodesHeld)
open Spot.RecordText (Decl Row Fill Inputs NodeDecl NodeFill Tmpl)
open Flock.Program (Prog Arg Ty Part)
open Flock.TemplateQuery (TRange)

/-- `nodesHeld`'s test of argument `a` (port `port`, of type `pty`) of the root node of range `r`, whose template's
units start at `u0` among its template's `t`. -/
def ArgOk (P : Prog) (W : Spot.Wires.Wiring) (x : Inputs) (ns : List NodeDecl) (r : TRange) (u0 : Nat) (t : Tmpl)
    (a : Arg) (port : String) (pty : Ty) : Bool :=
  let nodes := (P.defs[P.root]!).body.nodes
  match a.seq.parts.toList.find? (overDeclared (ns.map (·.node))) with
  | none => true
  | some _ => match stretchOf (nodes[r.node]!) a with
    | none => false
    | some (i, b, q, os, e) =>
      let nd := nodes[i]!
      let M := r.count
      let kc := W.cutsOf t.tid port
      let Q : Int := ((M - 1) / q : Nat)
      let lo := b + min 0 (Q * os)
      let hi := b + max 0 (Q * os) + (((min q M - 1) * e + e : Nat) : Int)
      match Flock.Program.Widths.homo nd.out, (ns.find? (·.node == i)).map (·.fill) with
      | some w, some (NodeFill.param p o) =>
        let L := e / kc
        let A : Int := 16 * (o : Int) + (w : Int) * b
        let so : Int := (w : Int) * os / 16
        let si : Int := ((w * e / 16 : Nat) : Int)
        Flock.Program.Widths.homo pty == some w && (nodes[r.node]!).members == M && kc ≥ 1 && e % kc == 0 &&
        0 ≤ lo && hi ≤ (nd.out.leaves : Int) && A % 16 == 0 && ((w : Int) * os) % 16 == 0 && (w * L) % 16 == 0 &&
        covers x.decls t.tid port u0 M kc fun d ua ub ra rb => ReadOk p q u0 so si A (w * L / 16) d ua ub ra rb
      | some _, some (NodeFill.table name root) =>
        (nodes[r.node]!).members == M && b == 0 && e == nd.out.leaves && (M == 1 || os == 0) && q == 1 && kc == 1 &&
        covers x.decls t.tid port u0 M kc fun d _ _ _ _ => match d.fill with
          | .table name' root' => name' == name && root' == root
          | _ => false
      | _, _ => false

/-- **`nodesHeld` holds every argument of every range of its template to `ArgOk`.** -/
theorem nodesHeld_arg {T : InferenceOnlyVerifier.Theta.Trusted} {P : Prog} {doc : Lean.Json} {ts : List Tmpl}
    {W : Spot.Wires.Wiring} {x : Inputs} {ns : List NodeDecl}
    (h : nodesHeld T P doc ts W x ns = true) {rs : Array TRange}
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) (ts.map (·.tid)).toArray = .ok rs)
    {r : TRange} {u0 : Nat} (hr : (r, u0) ∈ withStarts rs) {t : Tmpl} (ht : ts.find? (·.tid == r.template) = some t)
    {a : Arg} {port : String} {pty : Ty}
    (ha : (a, (port, pty)) ∈ ((P.defs[P.root]!).body.nodes[r.node]!).args.toList.zip (P.defs[t.defn]!).params.toList) :
    ArgOk P W x ns r u0 t a port pty = true := by
  unfold nodesHeld at h
  simp only [Bool.and_eq_true] at h
  obtain ⟨-, hm⟩ := h
  rw [hrs] at hm
  have h1 := List.all_eq_true.1 hm (r, u0) hr
  simp only [ht] at h1
  exact List.all_eq_true.1 h1 (a, (port, pty)) ha

/-- **An argument that reads a declared node filled by a registered parameter**: the node's width is the parameter's,
the range's units are the node's members, the port's cut is positive and divides the stretch, the leaves a member's
stretch reads lie inside the node, and the declarations that fill the port's rows of the range's units are each a
registered read of the parameter whose source at every unit and row is the node's leaf there (`ReadOk`). -/
theorem argOk_param {P : Prog} {W : Spot.Wires.Wiring} {x : Inputs} {ns : List NodeDecl} {r : TRange} {u0 : Nat}
    {t : Tmpl} {a : Arg} {port : String} {pty : Ty} (hok : ArgOk P W x ns r u0 t a port pty = true) {part : Part}
    (hfind : a.seq.parts.toList.find? (overDeclared (ns.map (·.node))) = some part) {n : Nat} {b : Int} {q : Nat}
    {os : Int} {e : Nat} (hs : stretchOf ((P.defs[P.root]!).body.nodes[r.node]!) a = some (n, b, q, os, e))
    {w : Nat} {p : String} {o : Nat} (hh : Flock.Program.Widths.homo ((P.defs[P.root]!).body.nodes[n]!).out = some w)
    (hfill : (ns.find? (·.node == n)).map (·.fill) = some (NodeFill.param p o)) :
    Flock.Program.Widths.homo pty = some w ∧ ((P.defs[P.root]!).body.nodes[r.node]!).members = r.count ∧
      1 ≤ W.cutsOf t.tid port ∧ e % W.cutsOf t.tid port = 0 ∧
      0 ≤ b + min 0 ((((r.count - 1) / q : Nat) : Int) * os) ∧
      (16 * (o : Int) + (w : Int) * b) % 16 = 0 ∧ ((w : Int) * os) % 16 = 0 ∧
      (w * (e / W.cutsOf t.tid port)) % 16 = 0 ∧
      covers x.decls t.tid port u0 r.count (W.cutsOf t.tid port) (fun d ua ub ra rb =>
        ReadOk p q u0 ((w : Int) * os / 16) ((w * e / 16 : Nat) : Int) (16 * (o : Int) + (w : Int) * b)
          (w * (e / W.cutsOf t.tid port) / 16) d ua ub ra rb) = true := by
  unfold ArgOk at hok
  simp only [hfind, hs, hh, hfill] at hok
  simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hok
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨hA, hB⟩, hC⟩, hD⟩, hE⟩, -⟩, hG⟩, hH⟩, hI⟩, hJ⟩ := hok
  exact ⟨hA, hB, hC, hD, hE, hG, hH, hI, hJ⟩

/-- **An argument that reads a declared node filled by a table's root or domain id**: the range's units are the node's
members, one unit reads the whole node in one row of an uncut port, and the declarations that fill it are each that
table read. -/
theorem argOk_table {P : Prog} {W : Spot.Wires.Wiring} {x : Inputs} {ns : List NodeDecl} {r : TRange} {u0 : Nat}
    {t : Tmpl} {a : Arg} {port : String} {pty : Ty} (hok : ArgOk P W x ns r u0 t a port pty = true) {part : Part}
    (hfind : a.seq.parts.toList.find? (overDeclared (ns.map (·.node))) = some part) {n : Nat} {b : Int} {q : Nat}
    {os : Int} {e : Nat} (hs : stretchOf ((P.defs[P.root]!).body.nodes[r.node]!) a = some (n, b, q, os, e))
    {w : Nat} {name : String} {root : Bool}
    (hh : Flock.Program.Widths.homo ((P.defs[P.root]!).body.nodes[n]!).out = some w)
    (hfill : (ns.find? (·.node == n)).map (·.fill) = some (NodeFill.table name root)) :
    ((P.defs[P.root]!).body.nodes[r.node]!).members = r.count ∧ b = 0 ∧
      e = ((P.defs[P.root]!).body.nodes[n]!).out.leaves ∧ (r.count = 1 ∨ os = 0) ∧ q = 1 ∧
      W.cutsOf t.tid port = 1 ∧
      covers x.decls t.tid port u0 r.count (W.cutsOf t.tid port) (fun d _ _ _ _ => match d.fill with
        | .table name' root' => name' == name && root' == root
        | _ => false) = true := by
  unfold ArgOk at hok
  simp only [hfind, hs, hh, hfill] at hok
  simp only [Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq, beq_iff_eq] at hok
  obtain ⟨⟨⟨⟨⟨⟨hA, hB⟩, hC⟩, hD⟩, hE⟩, hF⟩, hG⟩ := hok
  exact ⟨hA, hB, hC, hD, hE, hF, hG⟩

/-- **A root input node the record declares as a table's root or domain id is of a table the record lists.** -/
theorem nodesHeld_table {T : InferenceOnlyVerifier.Theta.Trusted} {P : Prog} {doc : Lean.Json} {ts : List Tmpl}
    {W : Spot.Wires.Wiring} {x : Inputs} {ns : List NodeDecl} (h : nodesHeld T P doc ts W x ns = true) {d : NodeDecl}
    (hd : d ∈ ns) {name : String} {root : Bool} (hf : d.fill = NodeFill.table name root) :
    ∃ tb, x.tables.find? (·.name == name) = some tb := by
  unfold nodesHeld at h
  simp only [Bool.and_eq_true] at h
  obtain ⟨⟨-, hall⟩, -⟩ := h
  have hok := List.all_eq_true.1 hall d hd
  split at hok
  · rename_i nd hnd
    simp only [Bool.and_eq_true] at hok
    obtain ⟨-, hm⟩ := hok
    rw [hf] at hm
    cases hh : Flock.Program.Widths.homo nd.out with
    | none => simp [hh] at hm
    | some w =>
      simp only [hh] at hm
      cases hfd : x.tables.find? (·.name == name) with
      | none => simp [hfd] at hm
      | some tb => exact ⟨tb, rfl⟩
  · cases hok

/-- An argument that reads a declared node has the node's width and a declaration of it. -/
theorem argOk_fill {P : Prog} {W : Spot.Wires.Wiring} {x : Inputs} {ns : List NodeDecl} {r : TRange} {u0 : Nat}
    {t : Tmpl} {a : Arg} {port : String} {pty : Ty} (hok : ArgOk P W x ns r u0 t a port pty = true) {part : Part}
    (hfind : a.seq.parts.toList.find? (overDeclared (ns.map (·.node))) = some part) {n : Nat} {b : Int} {q : Nat}
    {os : Int} {e : Nat} (hs : stretchOf ((P.defs[P.root]!).body.nodes[r.node]!) a = some (n, b, q, os, e)) :
    ∃ w, Flock.Program.Widths.homo ((P.defs[P.root]!).body.nodes[n]!).out = some w ∧
      ∃ fl, (ns.find? (·.node == n)).map (·.fill) = some fl := by
  unfold ArgOk at hok
  simp only [hfind, hs] at hok
  cases hh : Flock.Program.Widths.homo ((P.defs[P.root]!).body.nodes[n]!).out with
  | none => simp [hh] at hok
  | some w =>
    refine ⟨w, rfl, ?_⟩
    cases hf : (ns.find? (·.node == n)).map (·.fill) with
    | none => simp [hh, hf] at hok
    | some fl => exact ⟨fl, rfl⟩

/-- A registered read's check holds only of a registered read. -/
theorem readOk_fill {p : String} {q u0 : Nat} {so si A : Int} {wL : Nat} {d : Decl} {ua ub ra rb : Nat}
    (h : ReadOk p q u0 so si A wL d ua ub ra rb = true) : ∃ p', d.fill = Fill.read p' := by
  unfold ReadOk at h
  rcases hf : d.fill with _ | p' | _ | _ | _
  · simp [hf] at h
  · exact ⟨p', rfl⟩
  · simp [hf] at h
  · simp [hf] at h
  · simp [hf] at h

/-- A table read's check holds only of a table read of that table's root or domain id. -/
theorem tableCov_fill {d : Decl} {name : String} {root : Bool} (h : (match d.fill with
        | Fill.table name' root' => name' == name && root' == root
        | _ => false) = true) : d.fill = Fill.table name root := by
  rcases hf : d.fill with _ | _ | ⟨name', root'⟩ | _ | _
  · simp [hf] at h
  · simp [hf] at h
  · simp only [hf, Bool.and_eq_true, beq_iff_eq] at h
    rw [h.1, h.2]
  · simp [hf] at h
  · simp [hf] at h

/-! ## What the text declares of a row some declaration covers -/

/-- Among pairwise disjoint declarations, the first that covers a row is any that covers it. -/
theorem find?_decl : ∀ (ds : List Decl), (ds.map (·.w)).Pairwise (fun a b => Spot.Wires.disjoint a b = true) →
    ∀ (d : Decl) (r : Row), d ∈ ds → (d.w.indexOf r.tid r.unit r.port r.row).isSome = true →
      ds.find? (fun d => (d.w.indexOf r.tid r.unit r.port r.row).isSome) = some d
  | [], _, _, _, hd, _ => absurd hd List.not_mem_nil
  | e :: es, hp, d, r, hd, hc => by
    rw [List.map_cons, List.pairwise_cons] at hp
    rw [List.find?_cons]
    cases he : (e.w.indexOf r.tid r.unit r.port r.row).isSome with
    | true =>
      rcases List.mem_cons.1 hd with rfl | hd'
      · rfl
      · have hdis := hp.1 d.w (List.mem_map.2 ⟨d, hd', rfl⟩)
        rw [covers_not_disjoint he hc] at hdis
        cases hdis
    | false =>
      rcases List.mem_cons.1 hd with rfl | hd'
      · rw [he] at hc
        cases hc
      · exact find?_decl es hp.2 d r hd' hc

/-- A covered row's index in the declaration that covers it. -/
theorem indexOf_covered {w : Spot.Wires.Wire} {r : Row} (hc : (w.indexOf r.tid r.unit r.port r.row).isSome = true) :
    w.indexOf r.tid r.unit r.port r.row = some ((r.unit - w.start) * w.cuts + (r.row - w.cut0)) := by
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := (covers_iff w r).1 hc
  unfold Circuits.Wire.indexOf
  rw [if_pos]
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq]
  exact ⟨⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩, h6⟩

/-- **A row a registered read covers is declared that read's parameter at the bit its source is**, where the source is
at least 0. -/
theorem declaredAt_read {x : Inputs} {r : Row} {d : Decl}
    (hdis : Spot.Wires.pairwiseDisjoint (x.decls.map (·.w)) = true) (hd : d ∈ x.decls)
    (hc : (d.w.indexOf r.tid r.unit r.port r.row).isSome = true) {p : String} (hfill : d.fill = Fill.read p)
    (hnn : 0 ≤ d.w.writerAt ((r.unit - d.w.start) * d.w.cuts + (r.row - d.w.cut0))) :
    x.declaredAt r = some (.read p, 16 * (d.w.writerAt ((r.unit - d.w.start) * d.w.cuts + (r.row - d.w.cut0))).toNat) := by
  have hf := find?_decl x.decls ((Spot.Wires.pairwiseDisjoint_iff _).1 hdis) d r hd hc
  unfold Inputs.declaredAt
  rw [hf]
  simp only [Option.bind_some]
  rw [hfill, indexOf_covered hc]
  simp [Inputs.declares, hnn]

/-- **A row a table read covers is declared that table's root or domain id, at bit 0**, where the table is listed. -/
theorem declaredAt_table {x : Inputs} {r : Row} {d : Decl}
    (hdis : Spot.Wires.pairwiseDisjoint (x.decls.map (·.w)) = true) (hd : d ∈ x.decls)
    (hc : (d.w.indexOf r.tid r.unit r.port r.row).isSome = true) {name : String} {root : Bool}
    (hfill : d.fill = Fill.table name root) {tb : Spot.RecordText.Table}
    (htb : x.tables.find? (·.name == name) = some tb) :
    x.declaredAt r = some (.table name tb.param root, 0) := by
  have hf := find?_decl x.decls ((Spot.Wires.pairwiseDisjoint_iff _).1 hdis) d r hd hc
  unfold Inputs.declaredAt
  rw [hf]
  simp only [Option.bind_some]
  rw [hfill, indexOf_covered hc]
  simp [Inputs.declares, htb]

end InferenceOnlyVerifier.Properties.Record.SecurityProofs
