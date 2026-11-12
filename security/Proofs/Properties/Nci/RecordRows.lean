module

public import Proofs.Properties.Nci.RecordPrim
public import Proofs.Properties.Nci.RecordRowBits
public import Spot.WiresSound
public import Mathlib.Tactic

/-!
# A unit's rows

A unit reads through its parameters, and the record cuts each parameter into rows: parameter `i` of a unit, cut
`k` ways, is `k` rows of `leaves / k` leaves each.

* `rowLeaf_of_param`: a leaf of a unit's parameter is a leaf of one of the unit's rows. Leaf `t` is leaf
  `t % (leaves / k)` of row `t / (leaves / k)`. The cut divides the parameter's leaves: where the record declares
  none it is 1, and a declared cut divides the port's outer length (`Spot.Wires.cutsOk`, inside `Spot.Wires.check`),
  which is the parameter's (`TemplateOf`) and divides its leaves.
* `returnsOwn`: a leaf a unit returns is one of the unit's gates or a leaf of one of its rows. A unit of a
  primitive returns its one gate; a leaf any other unit returns resolves inside the unit or to a leaf of one of
  its parameters (`Resolve.ret_within`), and an output row's leaf is a leaf of the return (`outRow_lt`).
* `operandsOwn`: what a gate of a unit reads is an earlier gate of the unit or a leaf of one of its rows
  (`Resolve.gate_within`, and `prim_unit_operands` for a unit of a primitive).
* `ownRows_holds`: a leaf of a unit's row is an input gate, and no gate is a leaf of two rows or two leaves of
  one. A unit's parameter leaf resolves to a gate of an input node (`Resolve.unit_leaf`), no two to one gate
  (`Resolve.unit_leaf_inj`); the root node and the member say which unit, the parameter which port, and the leaf
  which row and which leaf of it.
* `ranked`: the wires `Spot.Wires.check` accepts rank the units, every row's writer below the row's unit.
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Flock.Program (Prog Scope Ref Ty Defn isInputId)
open Flock.TemplateQuery (TRange)
open NciCheck.RecordText (Row outerOf fieldsOf)
open NciCheck.Record.SecurityProofs (RootFirst ranges_spec)
open NciCheck.Properties.SecurityProofs

variable {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring} {S : Spot.Wires.Summary}

/-! ## The cut of a port divides its parameter's leaves -/

/-- A parameter's outer length divides its leaves. -/
private theorem outerOf_dvd (ty : Ty) : outerOf ty ∣ ty.leaves := by
  cases ty with
  | array n e =>
    rw [BodiesOk.leaves_array]
    show n ∣ n * e.leaves
    exact Nat.dvd_mul_right n _
  | value w => exact Nat.one_dvd _
  | tuple fs => exact Nat.one_dvd _
  | record fs => exact Nat.one_dvd _

/-- A wiring `Spot.Wires.check` accepts has sound cuts. -/
private theorem check_cuts (hS : Spot.Wires.check W = .ok S) : Spot.Wires.cutsOk W = true := by
  unfold Spot.Wires.check at hS
  split at hS
  · cases hS
  · rename_i hcut
    simpa using hcut

/-- A declared cut names a template of the wiring and one of its rows, is positive and divides the row's outer
length. -/
private theorem cut_dvd (hc : Spot.Wires.cutsOk W = true) {tid port : String} {kv : List (String × ℕ)} {k : ℕ}
    (hkv : W.cuts.lookup tid = some kv) (hk : kv.lookup port = some k) :
    ∃ t q, (tid, t) ∈ W.templates ∧ (port, q) ∈ t.rows ∧ 0 < k ∧ k ∣ q.outer := by
  unfold Spot.Wires.cutsOk at hc
  have h1 := List.all_eq_true.1 hc (tid, kv) (Spot.Wires.lookup_mem hkv)
  dsimp only at h1
  split at h1
  · rename_i t ht
    have h2 := List.all_eq_true.1 h1 (port, k) (Spot.Wires.lookup_mem hk)
    dsimp only at h2
    split at h2
    · rename_i q hq
      simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at h2
      exact ⟨t, q, Spot.Wires.lookup_mem ht, Spot.Wires.lookup_mem hq, h2.1.1, Nat.dvd_of_mod_eq_zero h2.1.2⟩
    · cases h2
  · cases h1

/-- **The cut of a unit's port is positive and divides the parameter's leaves.** -/
private theorem cutsOf_dvd (hT : Templates P rs W) (hS : Spot.Wires.check W = .ok S) {tid : String} {u : ℕ}
    {sc : Scope} (hu : UnitAt P rs tid u sc) {i : ℕ} (hi : i < (P.defs[sc.d]!).params.size) :
    0 < W.cutsOf tid ((P.defs[sc.d]!).params[i]!).1 ∧
      W.cutsOf tid ((P.defs[sc.d]!).params[i]!).1 ∣ ((P.defs[sc.d]!).params[i]!).2.leaves := by
  unfold Spot.Wires.Wiring.cutsOf
  cases hkv : W.cuts.lookup tid with
  | none => simp
  | some kv =>
    cases hk : kv.lookup ((P.defs[sc.d]!).params[i]!).1 with
    | none => simp [hk]
    | some k =>
      obtain ⟨t, q, ht, hq, hk0, hkq⟩ := cut_dvd (check_cuts hS) hkv hk
      obtain ⟨hinj, -, hrows, -⟩ := (hT tid t ht).2 u sc hu
      obtain ⟨i', hi', hname, -, hout⟩ := (hrows _ q).1 hq
      have hii : i' = i := hinj i' i hi' hi hname.symm
      rw [hii] at hout
      rw [hout] at hkq
      simp only [Option.getD_some, hk]
      exact ⟨hk0, Nat.dvd_trans hkq (outerOf_dvd _)⟩

/-! ## R1 -/

/-- **A leaf of a unit's parameter is a leaf of one of the unit's rows**: the row its cut puts it in. -/
theorem rowLeaf_of_param (hT : Templates P rs W) (hS : Spot.Wires.check W = .ok S) {tid : String} {u : ℕ}
    {sc : Scope} (hu : UnitAt P rs tid u sc) {i t g : ℕ} (hi : i < (P.defs[sc.d]!).params.size)
    (ht : t < ((P.defs[sc.d]!).params[i]!).2.leaves)
    (hg : Flock.Program.resolve P sc (⟨true, i, t⟩ : Ref) = .ok g) :
    ∃ r l lo hi, r.tid = tid ∧ r.unit = u ∧ RowLeaf P rs W r l g lo hi := by
  obtain ⟨hk0, q, hq⟩ := cutsOf_dvd hT hS hu hi
  obtain ⟨k, hkdef⟩ : ∃ k, W.cutsOf tid ((P.defs[sc.d]!).params[i]!).1 = k := ⟨_, rfl⟩
  rw [hkdef] at hk0 hq
  have ht' : t < k * q := by rw [← hq]; exact ht
  have hq0 : 0 < q := by
    rcases Nat.eq_zero_or_pos q with h | h
    · rw [h, Nat.mul_zero] at ht'
      exact absurd ht' (Nat.not_lt_zero _)
    · exact h
  have hdiv : ((P.defs[sc.d]!).params[i]!).2.leaves / k = q := by
    rw [hq]
    exact Nat.mul_div_cancel_left q hk0
  have hrow : t / q < k := (Nat.div_lt_iff_lt_mul hq0).2 ht'
  refine ⟨⟨tid, u, ((P.defs[sc.d]!).params[i]!).1, t / q⟩, t % q, _, _, rfl, rfl,
    sc, hu, i, hi, rfl, ?_, ?_, ?_, rfl, rfl⟩
  · show t / q < W.cutsOf tid ((P.defs[sc.d]!).params[i]!).1
    rw [hkdef]
    exact hrow
  · show t % q < (inRow (P.defs[sc.d]!) i (W.cutsOf tid ((P.defs[sc.d]!).params[i]!).1) (t / q)).2.size
    rw [hkdef, inRow_size _ _ _ _ hrow, hdiv]
    exact Nat.mod_lt _ hq0
  · show Flock.Program.resolve P sc (⟨true, i, t / q *
      (((P.defs[sc.d]!).params[i]!).2.leaves / W.cutsOf tid ((P.defs[sc.d]!).params[i]!).1) + t % q⟩ : Ref) = .ok g
    rw [hkdef, hdiv, Nat.div_add_mod' t q]
    exact hg

/-! ## R2 -/

/-- Fields' leaves, summed from `a`, are no fewer than `a`. -/
private theorem leaves_le_foldl (fs : List Ty) (a : ℕ) : a ≤ fs.foldl (fun a t => a + t.leaves) a := by
  induction fs generalizing a with
  | nil => exact Nat.le_refl _
  | cons x xs ih =>
    rw [List.foldl_cons]
    exact Nat.le_trans (Nat.le_add_right _ _) (ih _)

/-- The leaves of the fields before field `f`, and field `f`'s, are among the fields' leaves. -/
private theorem take_leaves (fs : List Ty) : ∀ (f a : ℕ), f < fs.length →
    (fs.take f).foldl (fun a t => a + t.leaves) a + fs[f]!.leaves ≤ fs.foldl (fun a t => a + t.leaves) a := by
  induction fs with
  | nil => intro f a hf; exact absurd hf (Nat.not_lt_zero _)
  | cons x xs ih =>
    intro f a hf
    cases f with
    | zero =>
      show a + x.leaves ≤ xs.foldl (fun a t => a + t.leaves) (a + x.leaves)
      exact leaves_le_foldl xs _
    | succ f =>
      show (xs.take f).foldl (fun a t => a + t.leaves) (a + x.leaves) + xs[f]!.leaves ≤
        xs.foldl (fun a t => a + t.leaves) (a + x.leaves)
      exact ih f _ (Nat.lt_of_succ_lt_succ hf)

/-- A type's leaves are its output fields' leaves. -/
private theorem fields_leaves (ty : Ty) : (fieldsOf ty).foldl (fun a t => a + t.leaves) 0 = ty.leaves := by
  cases ty with
  | value w => exact Nat.zero_add _
  | array n e => exact Nat.zero_add _
  | tuple fs =>
    have e : fieldsOf (.tuple fs) = if fs.isEmpty then [Ty.tuple fs] else fs.toList := rfl
    rw [e]
    split
    · exact Nat.zero_add _
    · rw [Ty.leaves, ← Array.foldl_toList]
  | record fs =>
    have e : fieldsOf (.record fs) = if fs.isEmpty then [Ty.record fs] else fs.toList.map (·.2) := rfl
    rw [e]
    split
    · exact Nat.zero_add _
    · rw [Ty.leaves, ← Array.foldl_toList, List.foldl_map]

/-- **A leaf of an output row is a leaf of the return.** -/
private theorem outRow_lt (d : Defn) (f l : ℕ) (hf : f < (fieldsOf d.ret).length)
    (hl : l < (outRow d f).2.size) : (outRow d f).1 + l < d.ret.leaves := by
  have h1 := take_leaves (fieldsOf d.ret) f 0 hf
  rw [fields_leaves] at h1
  have h2 : (outRow d f).2.size = ((fieldsOf d.ret)[f]!).leaves := widths_size _
  have h3 : (outRow d f).1 = ((fieldsOf d.ret).take f).foldl (fun a t => a + t.leaves) 0 := rfl
  rw [h2] at hl
  rw [h3]
  omega

/-- **What a unit returns is its own** (`ReturnsOwn`): a leaf of an output row of a unit is one of the unit's
gates or a leaf of one of the unit's reader rows. -/
theorem returnsOwn (hP : ReadProg P)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) (W.templates.map (·.1)).toArray = .ok rs)
    (hT : Templates P rs W) (hS : Spot.Wires.check W = .ok S) : ReturnsOwn P rs W := by
  intro tid u sc out l g lo hi' hu ho
  cases hp : (P.defs[sc.d]!).prim with
  | true => exact Or.inl (returnsOwn_prim hP hrs hu hp ho).2
  | false =>
    obtain ⟨hr, hrp, hlay, hB⟩ := read_bodies hP
    have hir := readProg_inputs hP
    obtain ⟨⟨bytes, hdec⟩, -, -, -, -, -, hfirst⟩ := hP
    obtain ⟨k, m, hk, -, -, hm, rfl⟩ := unit_node hr hfirst hir hrs hu
    have hc : (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).prim = false := hp
    have hgood : Resolve.Good P (P.child P.rootScope k m) :=
      Resolve.good_child P (parent := P.rootScope)
        (by show Resolve.Good P (.top P.root 0 none); unfold Resolve.Good; exact ⟨rfl, rfl, rfl⟩) hk rfl hc hm rfl
    obtain ⟨f, -, hf, hl, hres, -, -⟩ := ho
    obtain ⟨-, hcallee, -⟩ := Resolve.decoded_root bytes P hdec k hk
    obtain ⟨hbp, hcount, -⟩ := hcallee.2 hc
    have ht : (outRow (P.defs[(P.child P.rootScope k m).d]!) f).1 + l <
        (P.defs[(P.child P.rootScope k m).d]!).body.ret.count :=
      Nat.lt_of_lt_of_eq (outRow_lt _ f l hf hl) hcount.symm
    rcases Resolve.ret_within P ⟨hr, hrp⟩ hB hlay (P.child P.rootScope k m) hgood _ ht hres with
      hin | ⟨i, j, hi, hj, hij⟩
    · exact Or.inl hin
    · have hbp' : (P.defs[(P.child P.rootScope k m).d]!).body.params =
          (P.defs[(P.child P.rootScope k m).d]!).params := hbp
      rw [hbp'] at hi hj
      exact Or.inr (rowLeaf_of_param hT hS hu hi hj hij)

/-! ## R3 -/

/-- **What a gate of a unit reads is the unit's own** (`OperandsOwn`): an operand of a gate inside a unit is an
earlier gate of the unit or a leaf of one of the unit's reader rows. -/
theorem operandsOwn (hP : ReadProg P) (hO : NciCheck.Record.ownRows P = true)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) (W.templates.map (·.1)).toArray = .ok rs)
    (hT : Templates P rs W) (hS : Spot.Wires.check W = .ok S) : OperandsOwn P rs W := by
  intro tid u sc g p ops sc' hu hlo hhi hgate o ho
  cases hp : (P.defs[sc.d]!).prim with
  | true =>
    obtain ⟨-, -, -, -, -, -, -, hg1, -⟩ := prim_unit hP hrs hu hp
    have hg : g = sc.off := by
      rw [hg1] at hhi
      omega
    subst hg
    obtain ⟨i, t, hi, ht, hres⟩ := prim_unit_operands hP hO hrs hu hp hgate o ho
    exact Or.inr (rowLeaf_of_param hT hS hu hi ht hres)
  | false =>
    obtain ⟨hr, hrp, hlay, hB⟩ := read_bodies hP
    have hir := readProg_inputs hP
    obtain ⟨⟨bytes, hdec⟩, -, -, -, -, -, hfirst⟩ := hP
    obtain ⟨k, m, hk, -, -, hm, rfl⟩ := unit_node hr hfirst hir hrs hu
    have hc : (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).prim = false := hp
    rcases Resolve.gate_within P ⟨hr, hrp⟩ hB hlay hk hm hc hlo hhi hgate o ho with hin | ⟨i, j, hi, hj, hij⟩
    · exact Or.inl hin
    · obtain ⟨-, hcallee, -⟩ := Resolve.decoded_root bytes P hdec k hk
      have hbp' : (P.defs[(P.child P.rootScope k m).d]!).body.params =
          (P.defs[(P.child P.rootScope k m).d]!).params := (hcallee.2 hc).1
      rw [hbp'] at hi hj
      exact Or.inr (rowLeaf_of_param hT hS hu hi hj hij)

/-! ## R4 -/

/-- **A leaf of a row is a leaf of the row's parameter.** -/
private theorem inRow_lt (d : Defn) (i k row l : ℕ) (hrow : row < k) (hl : l < (inRow d i k row).2.size) :
    (inRow d i k row).1 + l < d.params[i]!.2.leaves := by
  rw [inRow_size d i k row hrow] at hl
  have h0 : (inRow d i k row).1 = row * (d.params[i]!.2.leaves / k) := rfl
  rw [h0]
  have h1 := Nat.mul_div_le d.params[i]!.2.leaves k
  have h2 := Nat.mul_le_mul_right (d.params[i]!.2.leaves / k) (show row + 1 ≤ k from hrow)
  rw [Nat.add_one_mul] at h2
  generalize d.params[i]!.2.leaves / k = q at hl h1 h2 ⊢
  generalize row * q = a at h2 ⊢
  generalize k * q = b at h1 h2
  omega

/-- **Where a unit is**: member `m` of the root node of a range `j`, a call or a batch that is no input node, with
as many arguments as the unit's Definition has parameters. -/
private theorem unit_place {ts : Array String} (hP : ReadProg P)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) {tid : String} {u : ℕ}
    {sc : Scope} (hu : UnitAt P rs tid u sc) :
    ∃ j m, j < rs.size ∧ rs[j]!.template = tid ∧ u = unitsBefore rs tid j + m ∧
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
  have hkey : isInputId (P.defs[((P.defs[P.root]!).body.nodes[rs[j]!.node]!).fn]!).key = false := by
    simpa using hno.2
  obtain ⟨-, -, hnode⟩ := Resolve.decoded_root bytes P hdec _ hlt
  refine ⟨j, m, hj, htj, huj, hsc, hlt, ?_, hform, ?_, hnode.1⟩
  · simp [Desc.isInputNode, hkey]
  · rw [← hcount]
    exact hm

/-- A row is its template, its unit, its port and its place in the port's cut. -/
private theorem row_ext (r r' : Row) (h1 : r.tid = r'.tid) (h2 : r.unit = r'.unit) (h3 : r.port = r'.port)
    (h4 : r.row = r'.row) : r = r' := by
  cases r
  cases r'
  simp_all

/-- **Every unit's rows are its own input gates** (`OwnRows`): a leaf of a reader row is an input gate, and a gate
is a leaf of one row at most, at one place. -/
theorem ownRows_holds (hP : ReadProg P) (hO : NciCheck.Record.ownRows P = true)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) (W.templates.map (·.1)).toArray = .ok rs) :
    OwnRows P rs W := by
  obtain ⟨hr, hrp, hL, -⟩ := read_bodies hP
  obtain ⟨hov, hlay⟩ := read_walk hP
  have hP' := hP
  have hir := readProg_inputs hP
  obtain ⟨⟨bytes, hdec⟩, -, hRL, hIR, -, -, hfirst⟩ := hP'
  constructor
  · rintro r l g lo hi' ⟨sc, hu, i, hip, -, hrow, hl, hres, -, -⟩
    obtain ⟨j, m, -, -, -, rfl, hk, hnode, hform, hm, hargs⟩ := unit_place hP hrs hu
    have ht := inRow_lt _ i _ r.row l hrow hl
    obtain ⟨n, x, hleaf, hn, hx, ho, hg⟩ := Resolve.unit_leaf bytes P hdec hov hlay hO hk hnode hform hm
      (Nat.lt_of_lt_of_eq hip hargs.symm) ht ((P.defs[P.root]!).body.nodes[rs[j]!.node]!).fn
      (P.child P.rootScope rs[j]!.node m).off
    have hgx := hg g hres
    obtain ⟨q, base, count, -, hrun, -, -, -⟩ := hleaf
    obtain ⟨-, -, hinp, -⟩ := NciCheck.Record.SecurityProofs.runOf_some hrun
    have hinp' : Desc.isInputNode P ((P.defs[P.root]!).body.nodes[n]!) = true := hinp
    have hmono := Resolve.offsets_mono P P.root (hL P.root hr) hrp (n + 1) (P.defs[P.root]!).body.nodes.size
      (by omega) (Nat.le_refl _)
    have hend : (P.defs[P.root]!).body.offsets[(P.defs[P.root]!).body.nodes.size]! = (P.defs[P.root]!).gates :=
      (hL P.root hr hrp).2.1
    have hlt : g < gates P := by
      show g < (P.defs[P.root]!).gates
      omega
    obtain ⟨p, ops, sc', _, hgate, -, -⟩ := gate_located hP hlt
    refine ⟨p, ops, sc', hgate, ?_⟩
    have hkn : n < DescAgree.nodeCount P := hn
    have hgk : (DescAgree.rootOffs P)[n]! ≤ g ∧ g < (DescAgree.rootOffs P)[n + 1]! := by
      show (P.defs[P.root]!).body.offsets[n]! ≤ g ∧ g < (P.defs[P.root]!).body.offsets[n + 1]!
      omega
    rw [DescAgree.gate_input P hRL hIR hgate hkn hgk, DescAgree.inputs_get P hkn]
    exact hinp'
  · rintro r l lo hi' r' l' lo' hi'' g ⟨sc, hu, i, hip, hname, hrow, hl, hres, -, -⟩
      ⟨sc', hu', i', hip', hname', hrow', hl', hres', -, -⟩
    obtain ⟨j, m, hj, htj, huj, rfl, hk, hnode, hform, hm, hargs⟩ := unit_place hP hrs hu
    obtain ⟨j', m', hj', htj', huj', rfl, hk', hnode', hform', hm', hargs'⟩ := unit_place hP hrs hu'
    have ht := inRow_lt _ i _ r.row l hrow hl
    have ht' := inRow_lt _ i' _ r'.row l' hrow' hl'
    obtain ⟨hkk, hmm, hii, htt⟩ := Resolve.unit_leaf_inj bytes P hdec hov hlay hO hk hk' hnode hnode' hform hform'
      hm hm' (Nat.lt_of_lt_of_eq hip hargs.symm) (Nat.lt_of_lt_of_eq hip' hargs'.symm) ht ht'
      ((P.defs[P.root]!).body.nodes[rs[j]!.node]!).fn (P.child P.rootScope rs[j]!.node m).off
      ((P.defs[P.root]!).body.nodes[rs[j']!.node]!).fn (P.child P.rootScope rs[j']!.node m').off hres hres'
    have hjj : j = j' := by
      rcases Nat.lt_trichotomy j j' with h | h | h
      · have := ranges_sorted hr hfirst hir hrs hj hj' h
        omega
      · exact h
      · have := ranges_sorted hr hfirst hir hrs hj' hj h
        omega
    subst hjj
    subst hmm
    subst hii
    have htid : r.tid = r'.tid := htj.symm.trans htj'
    have hport : r.port = r'.port := hname.symm.trans hname'
    have hunit : r.unit = r'.unit := by rw [huj, huj', htid]
    have hcut : W.cutsOf r'.tid r'.port = W.cutsOf r.tid r.port := by rw [htid, hport]
    rw [hcut] at hrow' hl' htt
    rw [inRow_size _ _ _ _ hrow] at hl
    rw [inRow_size _ _ _ _ hrow'] at hl'
    simp only [inRow] at htt
    obtain ⟨hrr, hll⟩ := Resolve.mul_add_inj hl hl' htt
    exact ⟨row_ext r r' htid hunit hport hrr, hll⟩

/-! ## R5 -/

/-- **The wires rank the units**: under a rank `Spot.Wires.check` gives, every row's writer ranks below the row's
unit. -/
theorem ranked (hS : Spot.Wires.check W = .ok S) : ∃ ρ : String × ℤ → ℕ, Ranked S ρ := by
  obtain ⟨ρ, hρ⟩ := Spot.Wires.check_ranked hS
  refine ⟨ρ, ?_⟩
  intro r wt wu out hw
  exact hρ _ _ (Spot.Wires.writerOf_reads hS hw)

end NciCheck.Properties.Record.SecurityProofs
