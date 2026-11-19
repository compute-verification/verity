module

public import Proofs.Properties.InferenceOnly.RecordShareStretch
public import Proofs.Properties.InferenceOnly.RecordShareArgs
public import Proofs.Properties.InferenceOnly.RecordShareOwn
public import Proofs.Properties.InferenceOnly.RecordTemplates
public import Proofs.Properties.InferenceOnly.RecordLeafWide
public import Proofs.Properties.InferenceOnly.RecordAligned
public import Mathlib.Tactic

/-!
# A leaf of a row whose argument reads a declared node

`decl_row`: where an argument of a unit's node reads a declared root input node, the gate a leaf of the unit's
parameter is is leaf `z` of that node, and the text declares the leaf's row at the bit of its parameter, or its table,
that leaf is at: `16 · o + w · z`, `o` the node's first 16-bit leaf of its parameter and `w` the node's leaf width.
The declaration that fills the row is one `covers` checked (`covers_cell`), its source at the row the corner's and the
slopes (`readOk_source`), and the leaf the node's stretch reads (`stretch_ref`).
-/

@[expose] public section

namespace InferenceOnlyVerifier.Properties.Record.SecurityProofs

open Lean (Json)
open Flock.Program (Prog Part Ty Arg)
open Flock.TemplateQuery (TRange)
open Spot.RecordText (Row Tmpl Decl Fill Inputs NodeDecl NodeFill templatesOf)
open InferenceOnlyVerifier.Record (covers withStarts stretchOf overDeclared nodesHeld)
open InferenceOnlyVerifier.Record.SecurityProofs (declared_inputs)
open InferenceOnlyVerifier.Properties.SecurityProofs

/-- A stretch's lowest leaf bounds every member's. -/
theorem lo_nonneg {b os : Int} {m Q : Nat} (hmQ : m ≤ Q) (h : 0 ≤ b + min 0 ((Q : Int) * os)) :
    0 ≤ b + (m : Int) * os := by
  have hmQ' : (m : Int) ≤ Q := by exact_mod_cast hmQ
  rcases le_or_gt 0 os with hos | hos
  · have h1 : (0 : Int) ≤ (m : Int) * os := by positivity
    have h2 : min 0 ((Q : Int) * os) ≤ 0 := min_le_left _ _
    linarith
  · have h1 : (Q : Int) * os ≤ (m : Int) * os := by nlinarith
    have h2 : min 0 ((Q : Int) * os) ≤ (Q : Int) * os := min_le_right _ _
    linarith

/-- **A read's source at a unit and row, in sixteenths of the node's leaf in the parameter**: the source `ReadOk` names,
sixteen times over, is the node's first bit in the parameter and `w` times the leaf the member's stretch reads. -/
theorem source_sixteen {w o e kc c m q : Nat} {b os : Int} (hA : (16 * (o : Int) + w * b) % 16 = 0)
    (hos : ((w : Int) * os) % 16 = 0) (hL : (w * (e / kc)) % 16 = 0) (hek : e % kc = 0) :
    16 * ((16 * (o : Int) + w * b) / 16 + ((m / q : Nat) : Int) * ((w : Int) * os / 16) +
      ((m % q : Nat) : Int) * ((w * e / 16 : Nat) : Int) + ((c * (w * (e / kc) / 16) : Nat) : Int)) =
      16 * (o : Int) + w * (b + ((m / q : Nat) : Int) * os + ((m % q : Nat) : Int) * e + ((c * (e / kc) : Nat) : Int)) := by
  have h1 : 16 * ((16 * (o : Int) + w * b) / 16) = 16 * (o : Int) + w * b :=
    Int.mul_ediv_cancel' (Int.dvd_of_emod_eq_zero hA)
  have h2 : 16 * ((w : Int) * os / 16) = (w : Int) * os := Int.mul_ediv_cancel' (Int.dvd_of_emod_eq_zero hos)
  have hwe : (w * e) % 16 = 0 := by
    apply Nat.mod_eq_zero_of_dvd
    have h16 : 16 ∣ w * (e / kc) := Nat.dvd_of_mod_eq_zero hL
    have : w * e = w * (e / kc) * kc := by
      rw [Nat.mul_assoc, Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hek)]
    rw [this]
    exact Dvd.dvd.mul_right h16 kc
  have h3 : 16 * (w * e / 16) = w * e := Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero hwe)
  have h4 : 16 * (w * (e / kc) / 16) = w * (e / kc) := Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero hL)
  have h3' : (16 : Int) * ((w * e / 16 : Nat) : Int) = (w : Int) * e := by exact_mod_cast h3
  have h4' : (16 : Int) * ((w * (e / kc) / 16 : Nat) : Int) = (w : Int) * ((e / kc : Nat) : Int) := by
    exact_mod_cast h4
  simp only [Nat.cast_mul, Nat.cast_add]
  linear_combination h1 + ((m / q : Nat) : Int) * h2 + ((m % q : Nat) : Int) * h3' + (c : Int) * h4'

/-- The first template of an id is the template of that id, among templates of distinct ids. -/
theorem find?_tid : ∀ (ts : List Tmpl), (ts.map (·.tid)).Nodup → ∀ t ∈ ts, ts.find? (·.tid == t.tid) = some t
  | [], _, t, ht => absurd ht List.not_mem_nil
  | e :: es, hnd, t, ht => by
    rw [List.map_cons, List.nodup_cons] at hnd
    rw [List.find?_cons]
    rcases List.mem_cons.1 ht with rfl | ht'
    · simp
    · by_cases he : e.tid = t.tid
      · exact absurd (he ▸ List.mem_map.2 ⟨t, ht', rfl⟩) hnd.1
      · have : (e.tid == t.tid) = false := by simpa using he
        rw [this]
        exact find?_tid es hnd.2 t ht'

theorem mem_zip_get {α β : Type} [Inhabited α] [Inhabited β] (l1 : Array α) (l2 : Array β) {i : Nat}
    (h1 : i < l1.size) (h2 : i < l2.size) : (l1[i]!, l2[i]!) ∈ l1.toList.zip l2.toList := by
  rw [getElem!_pos l1 i h1, getElem!_pos l2 i h2]
  have hlen : i < (l1.toList.zip l2.toList).length := by simp [h1, h2]
  have := List.getElem_mem hlen
  simpa [List.getElem_zip] using this

/-- **A parameter that a declared table node fills leaf for leaf has the node's width at every leaf**: the references of
the parameter's leaves are the node's leaves, and the decoder holds each to the width of the leaf that reads it. -/
theorem table_widths {P : Prog} (hP : ReadProg P) {k m i : Nat} (hk : k < (P.defs[P.root]!).body.nodes.size)
    (hform : ((P.defs[P.root]!).body.nodes[k]!).form = .call ∨ ((P.defs[P.root]!).body.nodes[k]!).form = .batch)
    (hm : m < ((P.defs[P.root]!).body.nodes[k]!).members) (hi : i < ((P.defs[P.root]!).body.nodes[k]!).args.size)
    (hip : i < (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).params.size) {n : Nat} {b : Int} {q : Nat} {os : Int}
    {e : Nat} (hs : stretchOf ((P.defs[P.root]!).body.nodes[k]!) (((P.defs[P.root]!).body.nodes[k]!).args[i]!) =
      some (n, b, q, os, e))
    (hno : BodiesOk.NodeOk (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!) ((P.defs[P.root]!).body.nodes[k]!))
    (hlo : 0 ≤ b + ((m / q : Nat) : Int) * os) {w : Nat}
    (hh : Flock.Program.Widths.homo ((P.defs[P.root]!).body.nodes[n]!).out = some w) :
    ∀ t, t < ((P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).params[i]!).2.leaves →
      ((P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).params[i]!).2.widths[t]! = w := by
  intro t ht
  have hrc := Walk.walkable_recut P hP.2.2.2.2.1
  obtain ⟨hroot, hB, hL, hD, hW⟩ := read_wide hP
  obtain ⟨⟨bytes, hdec⟩, -⟩ := hP
  have hgood : Resolve.Good P P.rootScope := ⟨rfl, rfl, rfl⟩
  obtain ⟨-, -, z, hget, -⟩ := stretch_ref hno hi hform hs hm ht hlo
  have hx := Resolve.argIdx_lt_of bytes P hdec hk hform hm hi ht
  have href0 := ((hB P.rootScope hgood).1 k hk).2.1 i _ hx
  have hget' : (((P.defs[P.rootScope.d]!).body.nodes[k]!).args[i]!).seq.get
      (Resolve.argIdx ((P.defs[P.root]!).body.nodes[k]!) i m t) = ⟨false, n, z⟩ := hget
  have hzlt : z < ((P.defs[P.root]!).body.nodes[n]!).out.leaves := by
    have h1 := (href0.2 (by rw [hget'])).2
    rw [hget'] at h1
    exact h1
  have hw := LeafWidths.arg_wide P hroot hB hD hW P.rootScope hgood k hk m hm i hip t ht
  have hwid : Flock.Program.Widths.wAt ((P.defs[P.root]!).body.nodes[n]!).out z =
      Flock.Program.Widths.wAt ((P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).params[i]!).2 t := by
    rcases hform with hf | hf
    · have h1 := hw.1 (Or.inr hf)
      have hg : (((P.defs[P.rootScope.d]!).body.nodes[k]!).args[i]!).seq.get t = ⟨false, n, z⟩ := by
        have h2 := hget'
        rw [Resolve.argIdx_call hf i m t] at h2
        exact h2
      rw [hg] at h1
      exact h1
    · have h1 := hw.2 hf
      have hg : (((P.defs[P.rootScope.d]!).body.nodes[k]!).args[i]!).seq.get
          (Resolve.memberIdx (((P.defs[P.rootScope.d]!).body.nodes[k]!).args[i]!) m t) = ⟨false, n, z⟩ := by
        have h2 := hget
        rw [Resolve.argIdx_batch hf i m t] at h2
        exact h2
      rw [hg] at h1
      exact h1
  rw [← LeafWidths.wAt_widths _ _ ht, ← hwid]
  exact LeafWidths.homo_wAt _ w hh z hzlt

/-- **Where an argument reads a declared node, the unit's leaf is a leaf of the node, and the text declares its row at
the leaf's place.** -/
theorem decl_row {T : InferenceOnlyVerifier.Theta.Trusted} {P : Prog} {doc : Json} {ts : List Tmpl}
    {W : Spot.Wires.Wiring} {x : Inputs} {ns : List NodeDecl} {rs : Array TRange} (hP : ReadProg P)
    (hts : templatesOf P doc = .ok ts) (hnodes : nodesHeld T P doc ts W x ns = true)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) (ts.map (·.tid)).toArray = .ok rs)
    (hdis : Spot.Wires.pairwiseDisjoint (x.decls.map (·.w)) = true)
    (hshape : ∀ d ∈ x.decls, ∀ p, d.fill = Fill.read p → ReadShaped d.w)
    {r : Row} {l g lo j m i t : Nat} (p : RowPos P rs W r l g lo j m i t) {part : Part}
    (hfind : ((((P.defs[P.root]!).body.nodes[rs[j]!.node]!).args[i]!).seq.parts).toList.find?
      (overDeclared (ns.map (·.node))) = some part) :
    ∃ n z w : Nat, n < (P.defs[P.root]!).body.nodes.size ∧ (ns.map (·.node)).contains n = true ∧
      z < ((P.defs[P.root]!).body.nodes[n]!).out.leaves ∧
      (P.defs[P.root]!).body.offsets[n + 1]! =
        (P.defs[P.root]!).body.offsets[n]! + ((P.defs[P.root]!).body.nodes[n]!).out.leaves ∧
      g = (P.defs[P.root]!).body.offsets[n]! + z ∧
      Flock.Program.Widths.homo ((P.defs[P.root]!).body.nodes[n]!).out = some w ∧
      ((∃ pp o bits, (ns.find? (·.node == n)).map (·.fill) = some (NodeFill.param pp o) ∧
          x.declaredAt r = some (.read pp, bits) ∧ (bits : Int) + lo = 16 * (o : Int) + (w : Int) * z) ∨
        (∃ name root tb, (ns.find? (·.node == n)).map (·.fill) = some (NodeFill.table name root) ∧
          x.tables.find? (·.name == name) = some tb ∧ x.declaredAt r = some (.table name tb.param root, 0) ∧
          lo = w * z)) := by
  have hrc := Walk.walkable_recut P hP.2.2.2.2.1
  obtain ⟨hov, hlay⟩ := read_walk hP
  have hP' := hP
  obtain ⟨⟨bytes, hdec⟩, -⟩ := hP'
  obtain ⟨hj, htj, huj, hk, hnode, hform, hm, hi, hip, hu, hname, hrow, hl, ht_eq, ht, hres, hlo⟩ := p
  have hmem := withStarts_mem rs hj
  rw [htj] at hmem
  obtain ⟨u0, hu0⟩ : ∃ u0, u0 = ((rs.extract 0 j).filter fun r' => r'.template == r.tid).foldl
      (fun a r' => a + r'.count) 0 := ⟨_, rfl⟩
  rw [← hu0] at hmem huj
  obtain ⟨t', ht'm, ht'id, -, ht'defn, -⟩ := unit_tmpl hP hts hrs hu
  have hnodup := (tmpls_named hts).1
  have hfindt : ts.find? (·.tid == rs[j]!.template) = some t' := by
    rw [htj, ← ht'id]
    exact find?_tid ts hnodup t' ht'm
  have hdefn : t'.defn = ((P.defs[P.root]!).body.nodes[rs[j]!.node]!).fn := ht'defn
  have hzip : (((P.defs[P.root]!).body.nodes[rs[j]!.node]!).args[i]!,
      (P.defs[t'.defn]!).params[i]!) ∈ ((P.defs[P.root]!).body.nodes[rs[j]!.node]!).args.toList.zip
        (P.defs[t'.defn]!).params.toList := by
    rw [hdefn]
    exact mem_zip_get _ _ hi hip
  have hok := nodesHeld_arg hnodes hrs hmem hfindt (a := ((P.defs[P.root]!).body.nodes[rs[j]!.node]!).args[i]!)
    (port := ((P.defs[t'.defn]!).params[i]!).1) (pty := ((P.defs[t'.defn]!).params[i]!).2) hzip
  rcases hsq : stretchOf ((P.defs[P.root]!).body.nodes[rs[j]!.node]!)
      (((P.defs[P.root]!).body.nodes[rs[j]!.node]!).args[i]!) with _ | ⟨n, b, q, os, e⟩
  · unfold ArgOk at hok
    simp [hfind, hsq] at hok
  have hdn : (ns.map (·.node)).contains n = true := stretch_declared hfind hsq
  obtain ⟨hn_lt, hn_in, hn_form⟩ := declared_inputs hnodes n hdn
  obtain ⟨w, hh, fl, hfl⟩ := argOk_fill hok hfind hsq
  obtain ⟨-, -, hno'⟩ := Resolve.decoded_root bytes P hdec hrc _ hk
  have hxlt := Resolve.argIdx_lt_of bytes P hdec hk hform hm hi ht
  have common : 0 ≤ b + ((m / q : Nat) : Int) * os →
      e = ((P.defs[((P.defs[P.root]!).body.nodes[rs[j]!.node]!).fn]!).params[i]!).2.leaves ∧ 0 < q ∧
        ∃ z : Nat, g = (P.defs[P.root]!).body.offsets[n]! + z ∧
          z < ((P.defs[P.root]!).body.nodes[n]!).out.leaves ∧
          (P.defs[P.root]!).body.offsets[n + 1]! =
            (P.defs[P.root]!).body.offsets[n]! + ((P.defs[P.root]!).body.nodes[n]!).out.leaves ∧
          (z : Int) = b + ((m / q : Nat) : Int) * os + ((m % q : Nat) : Int) * e + t := by
    intro hlo'
    obtain ⟨hel, hqpos, z, hget, hz⟩ := stretch_ref hno' hi hform hsq hm ht hlo'
    obtain ⟨hjlt, hol, hres'⟩ := Resolve.ref_leaf bytes P hdec hov hlay hk hi hxlt hget hn_lt hn_in hn_form
    refine ⟨hel, hqpos, z, ?_, hjlt, hol, hz⟩
    exact Resolve.run_two (Resolve.rs_unit P ((P.defs[P.root]!).body.nodes[rs[j]!.node]!).fn
      (P.child P.rootScope rs[j]!.node m).off rs[j]!.node m i t hform) hres' _ (Flock.Program.resolve_run hres)
  obtain ⟨kc, hkc⟩ : ∃ kc, kc = W.cutsOf r.tid r.port := ⟨_, rfl⟩
  rw [← hkc] at hrow hl ht_eq hlo
  have hkc' : W.cutsOf t'.tid ((P.defs[t'.defn]!).params[i]!).1 = kc := by
    rw [hkc, ht'id, hdefn, hname]
  have hport : ((P.defs[t'.defn]!).params[i]!).1 = r.port := by rw [hdefn]; exact hname
  cases fl with
  | param pp o =>
    obtain ⟨hA, hB, hC, hD, hE, hG, hH, hI, hcov⟩ := argOk_param hok hfind hsq hh hfl
    rw [hkc'] at hC hD hI hcov
    rw [hdefn] at hA
    have hmc : m < rs[j]!.count := hB ▸ hm
    have hmQ : m / q ≤ (rs[j]!.count - 1) / q := Nat.div_le_div_right (by omega)
    have hlo' := lo_nonneg hmQ hE
    obtain ⟨hel, hqpos, z, hgz, hzlt, hol, hzint⟩ := common hlo'
    obtain ⟨d, hd, hdr, hdp, h1, h2, h3, h4, hokd⟩ :=
      covers_cell (u := r.unit) (c := r.row) hdis hcov (by omega) (by omega) hrow
    have hokd' : ReadOk pp q u0 ((w : Int) * os / 16) ((w * e / 16 : Nat) : Int) (16 * (o : Int) + (w : Int) * b)
        (w * (e / kc) / 16) d (max u0 d.w.start) (min (u0 + rs[j]!.count) (d.w.start + d.w.readerUnits)) d.w.cut0
        (min kc (d.w.cut0 + d.w.cuts)) = true := hokd
    obtain ⟨p', hfill'⟩ := readOk_fill hokd'
    obtain ⟨hpp, hsrc⟩ := readOk_source hqpos hfill' (hshape d hd p' hfill') hokd' rfl rfl (Nat.min_le_right _ _)
      (Nat.min_le_right _ _) (by omega) (u := r.unit) (c := r.row) (by omega) (by omega) (by omega) (by omega)
    have hmu : r.unit - u0 = m := by omega
    rw [hmu] at hsrc
    have hcovr : (d.w.indexOf r.tid r.unit r.port r.row).isSome = true := by
      refine (covers_iff d.w r).2 ⟨?_, ?_, h1, h2, h3, h4⟩
      · rw [hdr, ht'id]
      · rw [hdp, hport]
    have h16 := source_sixteen (c := r.row) (m := m) (q := q) hG hH hI hD
    have hteq : t = r.row * (e / kc) + l := by rw [ht_eq, ← hel]
    have hz0 : (0 : Int) ≤ b + ((m / q : Nat) : Int) * os + ((m % q : Nat) : Int) * e +
        ((r.row * (e / kc) : Nat) : Int) := by
      have h5 : (0 : Int) ≤ ((m % q : Nat) : Int) * e := by positivity
      have h6 : (0 : Int) ≤ ((r.row * (e / kc) : Nat) : Int) := by positivity
      linarith
    rw [← hsrc] at h16
    have hV : 0 ≤ d.w.writerAt ((r.unit - d.w.start) * d.w.cuts + (r.row - d.w.cut0)) := by
      have h7 : (0 : Int) ≤ (w : Int) * (b + ((m / q : Nat) : Int) * os + ((m % q : Nat) : Int) * e +
        ((r.row * (e / kc) : Nat) : Int)) := mul_nonneg (by positivity) hz0
      have h8 : (0 : Int) ≤ 16 * (o : Int) := by positivity
      linarith
    have hfill'' : d.fill = Fill.read pp := by rw [hfill', hpp]
    have hdecl := declaredAt_read hdis hd hcovr hfill'' hV
    have hlo_eq : lo = l * w := by
      rw [hlo]
      have hconst := inRow_const (P.defs[((P.defs[P.root]!).body.nodes[rs[j]!.node]!).fn]!) i kc r.row w hrow
        hA
      exact bitAt_const _ w hconst l (by rw [inRow_size _ _ _ _ hrow]; omega)
    refine ⟨n, z, w, hn_lt, hdn, hzlt, hol, hgz, hh, Or.inl ⟨pp, o, _, hfl, hdecl, ?_⟩⟩
    rw [hlo_eq]
    have hbits : (((16 * (d.w.writerAt ((r.unit - d.w.start) * d.w.cuts + (r.row - d.w.cut0))).toNat : Nat)) : Int) =
        16 * d.w.writerAt ((r.unit - d.w.start) * d.w.cuts + (r.row - d.w.cut0)) := by
      push_cast
      rw [Int.toNat_of_nonneg hV]
    rw [hbits]
    have hzt : (z : Int) = b + ((m / q : Nat) : Int) * os + ((m % q : Nat) : Int) * e +
        ((r.row * (e / kc) : Nat) : Int) + l := by
      rw [hzint, hteq]
      push_cast
      ring
    rw [h16, hzt]
    push_cast
    ring
  | table name root =>
    obtain ⟨hB, hb0, hel0, hcm, hq1, hk1, hcov⟩ := argOk_table hok hfind hsq hh hfl
    rw [hkc'] at hk1 hcov
    have hmc : m < rs[j]!.count := hB ▸ hm
    have hos0 : (m : Int) * os = 0 := by
      rcases hcm with hc1 | hos
      · have : m = 0 := by omega
        rw [this]
        simp
      · rw [hos]
        simp
    have hlo0 : 0 ≤ b + ((m / q : Nat) : Int) * os := by
      rw [hq1, hb0]
      simp [Nat.div_one, hos0]
    obtain ⟨hel, hqpos, z, hgz, hzlt, hol, hzint⟩ := common hlo0
    obtain ⟨d, hd, hdr, hdp, h1, h2, h3, h4, hokd⟩ :=
      covers_cell (u := r.unit) (c := r.row) hdis hcov (by omega) (by omega) hrow
    have hfill' : d.fill = Fill.table name root := tableCov_fill hokd
    obtain ⟨nd, hnd, hndfill⟩ : ∃ nd : NodeDecl, ns.find? (·.node == n) = some nd ∧
        nd.fill = NodeFill.table name root := by
      cases hf : ns.find? (·.node == n) with
      | none => simp [hf] at hfl
      | some nd =>
        rw [hf] at hfl
        exact ⟨nd, rfl, by simpa using hfl⟩
    obtain ⟨tb, htb⟩ := nodesHeld_table hnodes (List.mem_of_find?_eq_some hnd) hndfill
    have hcovr : (d.w.indexOf r.tid r.unit r.port r.row).isSome = true := by
      refine (covers_iff d.w r).2 ⟨?_, ?_, h1, h2, h3, h4⟩
      · rw [hdr, ht'id]
      · rw [hdp, hport]
    have hdecl := declaredAt_table hdis hd hcovr hfill' htb
    have hrow0 : r.row = 0 := by omega
    have hteq : t = l := by rw [ht_eq, hrow0, Nat.zero_mul, Nat.zero_add]
    have hzl : (z : Int) = l := by
      rw [hzint, hq1, hb0, hteq]
      simp [Nat.div_one, Nat.mod_one, hos0]
    have hw := table_widths hP hk hform hm hi hip hsq hno' hlo0 hh
    have hlo_eq : lo = l * w := by
      rw [hlo]
      have hconst : ∀ l', l' < (inRow (P.defs[((P.defs[P.root]!).body.nodes[rs[j]!.node]!).fn]!) i kc r.row).2.size →
          (inRow (P.defs[((P.defs[P.root]!).body.nodes[rs[j]!.node]!).fn]!) i kc r.row).2[l']! = w := by
        intro l' hl'
        rw [inRow_size _ _ _ _ hrow] at hl'
        obtain ⟨h1', h2'⟩ := inRow_get (P.defs[((P.defs[P.root]!).body.nodes[rs[j]!.node]!).fn]!) i kc r.row l' hrow hl'
        rw [h2']
        exact hw _ h1'
      exact bitAt_const _ w hconst l (by rw [inRow_size _ _ _ _ hrow]; omega)
    refine ⟨n, z, w, hn_lt, hdn, hzlt, hol, hgz, hh, Or.inr ⟨name, root, tb, hfl, htb, hdecl, ?_⟩⟩
    have : z = l := by exact_mod_cast hzl
    rw [hlo_eq, this, Nat.mul_comm]

end InferenceOnlyVerifier.Properties.Record.SecurityProofs
