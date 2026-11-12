module

public import Proofs.Properties.Nci.LeafWidthsResolve
public import Proofs.Properties.Nci.RecordFields
public import Proofs.Properties.Nci.RecordLanding
public import Proofs.Properties.Nci.RecordLeafWide
public import Proofs.Properties.Nci.RecordPrim
public import Proofs.Properties.Nci.RecordRead
public import Proofs.Properties.Nci.RecordTemplates
public import Proofs.Properties.Nci.RecordWhole
public import Proofs.Properties.Nci.RecordWide
public import Mathlib.Tactic

/-!
# Wires land whole

A wire fills a reader row with an output row of its writer. `Record.unwhole` refuses a record with a wire whose two
rows do not line up, and this module states what a record it accepts has, in the terms of the record's circuit:

* `aligned` (`Aligned`): a leaf of the writer's row lies inside one leaf of the reader's row, or the reader's unit
  returns no leaf of the row.
* `filled` (`Filled`): every bit of a leaf of a wired row is a bit of a leaf of the writer's row.

Both read `wired_row`, what the check has tested of a row with a writer: the row and the writer's output row have
the same bits, and either the reader's parameter has leaves of one bit and is none of what its Definition passes on,
or each row's leaves are of one width and the writer's divides the reader's.

* Where the widths divide, the rows are leaves of one width each over the same bits (`bitAt_const`), so leaf `l'` of
  the writer lies inside leaf `l' / c` of the reader, `c` the quotient of the widths (`leaf_inside`, `leaf_count`).
* Where the parameter is not passed on, no leaf of it is a leaf the unit returns (`closed_not_returned`). A leaf a
  unit of a composite returns is a gate of the unit or a leaf of a parameter its Definition passes on
  (`Resolve.ret_passed_leaf`, which is `Resolve.ret_passed_on` with the leaf in its parameter); a leaf of a unit's
  parameter is a gate of an input node, so none of the unit's (`unit_param_out`), and one leaf of one parameter
  only (`Resolve.unit_leaf_inj`). A unit of a primitive returns its own gate.
* A bit of a row lies in one of its leaves (`leaf_of_bit`), and a leaf of a unit's output row resolves
  (`unit_ret_ok`), as a leaf of its reader row does (`unit_param_ok`): so the leaves the two statements name are
  leaves of the record's circuit (`outLeaf_intro`, `rowLeaf_intro`).
-/

@[expose] public section

namespace NciCheck.Properties.SecurityProofs.Resolve

open Flock.Program
open NciCheck.Record.SecurityProofs (Passing passOf passing_start passing_exit)

section
variable (P : Prog) (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
  (hB : ∀ sc, Good P sc → BodyOk P sc.d) (hL : ∀ d < P.defs.size, Walk.LaidOut P d)
  (hret : ∀ d < P.defs.size, (P.defs[d]!).prim = false →
    (P.defs[d]!).body.ret = Seq.ofParts (P.defs[d]!).body.ret.parts)
  (harg : ∀ d < P.defs.size, ∀ k < (P.defs[d]!).body.nodes.size,
    ∀ i < ((P.defs[d]!).body.nodes[k]!).args.size,
      (((P.defs[d]!).body.nodes[k]!).args[i]!).seq =
        Seq.ofParts (((P.defs[d]!).body.nodes[k]!).args[i]!).seq.parts)
include hroot hB hL hret harg

/-- **A walk of resolution that starts passing under `U` ends in `U` or leaves through a leaf of a parameter `U`'s
Definition passes on.** It is `run_passing` with the leaf in its parameter: the state the walk leaves `U` from is a
reference of `U`'s body. -/
theorem run_passing_leaf (U : Scope) (B : ℕ) :
    ∀ (n : ℕ), n ≤ Flock.Steps.FUEL → ∀ (s : Scope × Ref) (b : ℕ), Inv P B s → Under U s.1 → Passing P U s →
      Flock.Steps.run "scope resolution" (resolveStep P) n s = .ok b →
      (U.off ≤ b ∧ b < B) ∨ ∃ i j, i ∈ passOf P U.d ∧ i < (P.defs[U.d]!).body.params.size ∧
        j < ((P.defs[U.d]!).body.params[i]!).2.leaves ∧ resolve P U ⟨true, i, j⟩ = .ok b
  | 0, _, _, _, _, _, _, h => by simp [Flock.Steps.run, throw, throwThe, MonadExceptOf.throw] at h
  | n + 1, hn, s, b, hs, hu, hpass, h => by
    have h' := h
    rw [Flock.Steps.run.eq_2] at h'
    rcases hx : resolveStep P s with e | (a | s')
    · simp [hx, bind, Except.bind] at h'
    · simp only [hx, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h'
      rw [← h']
      exact Or.inl ⟨Nat.le_trans (under_off P U s.1 hs.1 hu) ((step_where P hB B s _ hs hx).1 a rfl).2,
        (step_ok P hroot hB hL B s _ hs hx).1 a rfl⟩
    · simp only [hx, bind, Except.bind] at h'
      have hs' := (step_ok P hroot hB hL B s _ hs hx).2 s' rfl
      rcases (step_where P hB B s _ hs hx).2 s' rfl with
        ⟨hp0, d, off, k, m, hc⟩ | ⟨hp, parent, d, off, k, m, hsc, hc⟩
      · have hpass' := step_passing P hroot hB hret harg U B s s' hs hx
          (fun hc' => by have h2 := hc'.2; rw [hp0] at h2; cases h2) hpass
        refine run_passing_leaf U B n (by omega) s' b hs' ?_ hpass' h'
        rw [hc]
        exact Or.inr hu
      · by_cases hU : s.1 = U
        · obtain ⟨sc, ⟨p, i, j⟩⟩ := s
          have hp' : p = true := hp
          have hU' : sc = U := hU
          rw [hp', hU'] at h hpass
          have hb : i < (P.defs[sc.d]!).body.params.size ∧ j < ((P.defs[sc.d]!).body.params[i]!).2.leaves :=
            hs.2.1.1 hp'
          rw [hU'] at hb
          exact Or.inr ⟨i, j, passing_exit P U i j hpass, hb.1, hb.2, resolve_of_run P hn h⟩
        · have hpass' := step_passing P hroot hB hret harg U B s s' hs hx (fun hc' => hU hc'.1) hpass
          have hup : Under U parent := by
            rw [hsc] at hu hU
            have hu' : U = .child parent d off k m ∨ Under U parent := hu
            rcases hu' with hu' | hu'
            · exact absurd hu'.symm hU
            · exact hu'
          rcases hc with hc | ⟨off', m', hc⟩
          · refine run_passing_leaf U B n (by omega) s' b hs' ?_ hpass' h'
            rw [hc]
            exact hup
          · refine run_passing_leaf U B n (by omega) s' b hs' ?_ hpass' h'
            rw [hc]
            exact Or.inr hup

/-- **A leaf an activation returns is one of its gates, or a leaf of a parameter its Definition passes on.** -/
theorem ret_passed_leaf (U : Scope) (hU : Good P U) (t : ℕ) (ht : t < (P.defs[U.d]!).body.ret.count) {b : ℕ}
    (h : resolve P U ((P.defs[U.d]!).body.ret.get t) = .ok b) :
    (U.off ≤ b ∧ b < U.off + (P.defs[U.d]!).gates) ∨
      ∃ i j, i ∈ passOf P U.d ∧ i < (P.defs[U.d]!).body.params.size ∧
        j < ((P.defs[U.d]!).body.params[i]!).2.leaves ∧ resolve P U ⟨true, i, j⟩ = .ok b := by
  obtain ⟨hd, hpc⟩ := good_d P hroot hB U hU
  obtain ⟨q, hq, t', ht', hget⟩ := seq_part _ (hret U.d hd hpc) t ht
  exact run_passing_leaf P hroot hB hL hret harg U _ Flock.Steps.FUEL (Nat.le_refl _) _ b
    (down_ok P hroot hB hL _ U hU (Nat.le_refl _) t ht) (under_refl U) (passing_start P U hd hpc hq ht' hget) h

end

end NciCheck.Properties.SecurityProofs.Resolve

namespace NciCheck.Properties.Record.SecurityProofs

open Flock.Program (Prog Scope Ref Ty Defn)
open Flock.TemplateQuery (TRange)
open NciCheck.RecordText (Row Tmpl fieldsOf outerOf templatesOf)
open NciCheck.Record (unwhole leavesOf passedOn)
open NciCheck.Record.SecurityProofs (runOf runOf_some mem_of_lookup shaped_of_tmplOf whole_spec bitsOf_eq passOf
  passedOn_get)
open NciCheck.Properties.SecurityProofs
open Spot.Bits (bitsOf)
open Lean (Json)

/-! ## Rows of one width -/

/-- Leaf 0 of a row starts at bit 0. -/
theorem bitAt_zero (ws : Array ℕ) : bitAt ws 0 = 0 := by
  unfold bitAt
  rw [Array.extract_eq_empty_of_le (Nat.zero_le _)]
  rfl

/-- In a row whose leaves are all `w` bits, leaf `l` starts at bit `l * w`. -/
theorem bitAt_const (ws : Array ℕ) (w : ℕ) (h : ∀ l, l < ws.size → ws[l]! = w) :
    ∀ l, l ≤ ws.size → bitAt ws l = l * w := by
  intro l
  induction l with
  | zero =>
    intro _
    rw [bitAt_zero, Nat.zero_mul]
  | succ l ih =>
    intro hl
    rw [bitAt_succ ws l hl, ih (Nat.le_of_lt hl), h l hl, Nat.succ_mul]

/-- A bit of a row lies in one of its leaves. -/
theorem leaf_of_bit (ws : Array ℕ) : ∀ n, n ≤ ws.size → ∀ b, b < bitAt ws n →
    ∃ l, l < n ∧ bitAt ws l ≤ b ∧ b < bitAt ws l + ws[l]! := by
  intro n
  induction n with
  | zero =>
    intro _ b hb
    rw [bitAt_zero] at hb
    exact absurd hb (Nat.not_lt_zero b)
  | succ n ih =>
    intro hn b hb
    rw [bitAt_succ ws n hn] at hb
    rcases Nat.lt_or_ge b (bitAt ws n) with h | h
    · obtain ⟨l, hl, h1, h2⟩ := ih (Nat.le_of_lt hn) b h
      exact ⟨l, Nat.lt_succ_of_lt hl, h1, h2⟩
    · exact ⟨n, Nat.lt_succ_self n, h, hb⟩

/-- Leaf `l` of a row of a parameter is a leaf of the parameter, with its width there. -/
theorem inRow_get (d : Defn) (i k row l : ℕ) (hrow : row < k) (hl : l < d.params[i]!.2.leaves / k) :
    (inRow d i k row).1 + l < d.params[i]!.2.leaves ∧
      (inRow d i k row).2[l]! = d.params[i]!.2.widths[(inRow d i k row).1 + l]! := by
  have hsz := inRow_size d i k row hrow
  have h1 := Nat.mul_div_le d.params[i]!.2.leaves k
  have h2 := Nat.mul_le_mul_right (d.params[i]!.2.leaves / k) (show row + 1 ≤ k from hrow)
  rw [Nat.add_one_mul] at h2
  have hlt : row * (d.params[i]!.2.leaves / k) + l < d.params[i]!.2.leaves := by omega
  exact ⟨hlt, extract_get _ _ _ l (Nat.lt_of_lt_of_eq hl hsz.symm)⟩

/-- A type of one width has it at every entry of its widths. -/
theorem homo_widths (t : Ty) (w : ℕ) (h : Flock.Program.Widths.homo t = some w) (j : ℕ) (hj : j < t.leaves) :
    t.widths[j]! = w := by
  rw [← LeafWidths.wAt_widths t j hj]
  exact LeafWidths.homo_wAt t w h j hj

/-- A row of a parameter of one width has leaves of that width. -/
theorem inRow_const (d : Defn) (i k row a : ℕ) (hrow : row < k)
    (h : Flock.Program.Widths.homo d.params[i]!.2 = some a) :
    ∀ l, l < (inRow d i k row).2.size → (inRow d i k row).2[l]! = a := by
  intro l hl
  rw [inRow_size d i k row hrow] at hl
  obtain ⟨h1, h2⟩ := inRow_get d i k row l hrow hl
  rw [h2]
  exact homo_widths _ a h _ h1

/-- An output row of a field of one width has leaves of that width. -/
theorem outRow_const (d : Defn) (f o : ℕ) (h : Flock.Program.Widths.homo (fieldsOf d.ret)[f]! = some o) :
    ∀ l, l < (outRow d f).2.size → (outRow d f).2[l]! = o := by
  intro l hl
  have hl' : l < (fieldsOf d.ret)[f]!.leaves := by
    rw [← widths_size]
    exact hl
  exact homo_widths _ o h l hl'

/-- A leaf of an output row is a leaf of the return. -/
theorem outRow_lt (d : Defn) (f l : ℕ) (hf : f < (fieldsOf d.ret).length) (hl : l < (outRow d f).2.size) :
    (outRow d f).1 + l < d.ret.leaves := by
  have hl' : l < (fieldsOf d.ret)[f]!.leaves := by
    rw [← widths_size]
    exact hl
  exact (field_leaf d.ret f l hf hl').1

/-- Leaf `l'` of a row of leaves of `o` bits lies inside leaf `l' / c` of a row of leaves of `o * c` bits. -/
theorem leaf_inside (o c l' : ℕ) (hc : 0 < c) :
    l' / c * (o * c) ≤ l' * o ∧ l' * o + o ≤ l' / c * (o * c) + o * c := by
  have h1 : l' / c * c ≤ l' := Nat.div_mul_le_self l' c
  have h2 : l' < l' / c * c + c := Nat.lt_div_mul_add hc
  have e1 : l' / c * (o * c) = l' / c * c * o := by
    rw [Nat.mul_comm o c, Nat.mul_assoc]
  have e2 : l' * o + o = (l' + 1) * o := by rw [Nat.add_one_mul]
  have e3 : l' / c * c * o + o * c = (l' / c * c + c) * o := by
    rw [Nat.add_mul, Nat.mul_comm c o]
  rw [e1]
  refine ⟨Nat.mul_le_mul_right o h1, ?_⟩
  rw [e2, e3]
  exact Nat.mul_le_mul_right o h2

/-- Of two rows, `nW` leaves of `o` bits no more than `nR` leaves of `o * c` bits: a leaf `l'` of the first has a
leaf `l' / c` of the second. -/
theorem leaf_count (o c nR nW l' : ℕ) (ho : o ≠ 0) (h : nW * o ≤ nR * (o * c)) (hl : l' < nW) :
    0 < c ∧ l' / c < nR := by
  have h1 : o * nW ≤ o * (c * nR) := by
    rw [Nat.mul_comm o nW, ← Nat.mul_assoc, Nat.mul_comm (o * c) nR]
    exact h
  have h2 : nW ≤ c * nR := Nat.le_of_mul_le_mul_left h1 (Nat.pos_of_ne_zero ho)
  have h3 : l' < c * nR := Nat.lt_of_lt_of_le hl h2
  refine ⟨?_, Nat.div_lt_of_lt_mul h3⟩
  rcases Nat.eq_zero_or_pos c with h0 | h0
  · rw [h0, Nat.zero_mul] at h3
    exact absurd h3 (Nat.not_lt_zero l')
  · exact h0

/-! ## A unit's place -/

/-- A unit of a composite is an activation resolution reaches. -/
theorem unit_good {P : Prog} {ts : Array String} {rs : Array TRange} (hP : ReadProg P)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) {tid : String} {u : ℕ}
    {sc : Scope} (hu : UnitAt P rs tid u sc) (hp : (P.defs[sc.d]!).prim = false) : Resolve.Good P sc := by
  obtain ⟨hr, -, -, -⟩ := read_bodies hP
  obtain ⟨k, m, hk, -, -, hm, rfl⟩ := unit_node hr hP.2.2.2.2.2.2 (readProg_inputs hP) hrs hu
  have hp' : (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).prim = false := hp
  exact Resolve.good_child P (parent := P.rootScope)
    (by show Resolve.Good P (.top P.root 0 none); unfold Resolve.Good; exact ⟨rfl, rfl, rfl⟩) hk rfl hp' hm rfl

/-- A unit's Definition is as the decoder leaves a Definition a node calls. -/
theorem unit_callee {P : Prog} {ts : Array String} {rs : Array TRange} (hP : ReadProg P)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) {tid : String} {u : ℕ}
    {sc : Scope} (hu : UnitAt P rs tid u sc) : BodiesOk.CalleeOk (P.defs[sc.d]!) := by
  obtain ⟨hr, -, -, -⟩ := read_bodies hP
  obtain ⟨k, m, hk, -, -, -, rfl⟩ := unit_node hr hP.2.2.2.2.2.2 (readProg_inputs hP) hrs hu
  obtain ⟨⟨bytes, hdec⟩, -⟩ := hP
  exact (Resolve.decoded_root bytes P hdec k hk).2.1

/-- **A leaf of a unit's parameter is no gate of the unit**: it is a gate of an input node of the root, and the
unit's gates are inside a root node that is no input node. -/
theorem unit_param_out {P : Prog} {ts : Array String} {rs : Array TRange} (hP : ReadProg P)
    (hO : NciCheck.Record.ownRows P = true)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) {tid : String} {u : ℕ}
    {sc : Scope} (hu : UnitAt P rs tid u sc) {i t g : ℕ} (hi : i < (P.defs[sc.d]!).params.size)
    (ht : t < ((P.defs[sc.d]!).params[i]!).2.leaves)
    (hg : Flock.Program.resolve P sc (⟨true, i, t⟩ : Ref) = .ok g) :
    g < sc.off ∨ sc.off + (P.defs[sc.d]!).gates ≤ g := by
  obtain ⟨hr, hrp, hlay, -⟩ := read_bodies hP
  obtain ⟨hov, hbl⟩ := read_walk hP
  obtain ⟨k, m, hk, hnode, hform, hm, hargs, rfl⟩ := unit_member hP hrs hu
  obtain ⟨⟨bytes, hdec⟩, -⟩ := hP
  obtain ⟨n, j, ⟨q, base, count, -, hrun, -, -, -⟩, hn, hj, ho, hres⟩ :=
    Resolve.unit_leaf bytes P hdec hov hbl hO hk hnode hform hm (hargs i hi) ht
      ((P.defs[P.root]!).body.nodes[k]!).fn (P.child P.rootScope k m).off
  have hgn := hres g hg
  obtain ⟨-, -, hin, -⟩ := runOf_some hrun
  have hin' : Desc.isInputNode P ((P.defs[P.root]!).body.nodes[n]!) = true := hin
  have hne : n ≠ k := by
    intro e
    rw [e, hnode] at hin'
    exact Bool.noConfusion hin'
  obtain ⟨h1, h2⟩ := unit_within P (hlay P.root hr) hrp hk hm
  have hmono := Resolve.offsets_mono P P.root (hlay P.root hr) hrp
  rcases Nat.lt_or_gt_of_ne hne with hlt | hgt
  · have h3 := hmono (n + 1) k hlt (Nat.le_of_lt hk)
    exact Or.inl (by omega)
  · have h3 := hmono (k + 1) n hgt (Nat.le_of_lt hn)
    exact Or.inr (by omega)

/-! ## A unit's leaves resolve -/

/-- **A leaf a unit returns resolves.** A unit of a primitive returns its own gate; a unit of a composite is an
activation resolution reaches, and a reference of its return resolves from there. -/
theorem unit_ret_ok {P : Prog} {ts : Array String} {rs : Array TRange} (hP : ReadProg P)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) {tid : String} {u : ℕ}
    {sc : Scope} (hu : UnitAt P rs tid u sc) {t : ℕ} (ht : t < (P.defs[sc.d]!).ret.leaves) :
    ∃ g, Flock.Program.resolve P sc ((P.defs[sc.d]!).body.ret.get t) = .ok g := by
  by_cases hp : (P.defs[sc.d]!).prim = true
  · obtain ⟨k, m, -, -, -, -, hbody, -, -⟩ := prim_unit hP hrs hu hp
    exact ⟨sc.off, Resolve.ret_default P sc t hbody⟩
  · have hp' : (P.defs[sc.d]!).prim = false := by simpa using hp
    obtain ⟨hr, hrp, hlay, hB⟩ := read_bodies hP
    have hgood := unit_good hP hrs hu hp'
    have hcount := ((unit_callee hP hrs hu).2 hp').2.1
    obtain ⟨-, -, -, -, hwk, -, -⟩ := hP
    unfold Desc.walkable at hwk
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hwk
    obtain ⟨⟨hc, -⟩, hres, hwb⟩ := hwk
    obtain ⟨hpar, hz⟩ := OperandsResolve.resolvable_spec P hres
    have hfuel : 4 * (Desc.activations P)[P.root]! ≤ Flock.Steps.FUEL := by
      unfold Desc.walkBound at hwb
      have : ((Desc.activations P)[P.root]! + 1) * 2 ≤
          ((Desc.activations P)[P.root]! + 1) * ((Desc.depths P)[P.root]! + 2) := Nat.mul_le_mul_left _ (by omega)
      omega
    exact OperandsResolve.resolve_ok P ⟨hr, hrp⟩ hB hlay hc hpar hz hfuel _ sc _
      (Resolve.down_ok P ⟨hr, hrp⟩ hB hlay _ sc hgood (Nat.le_refl _) t (by rw [hcount]; exact ht))

variable {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring} {S : Spot.Wires.Summary}

/-- A leaf of a unit's output row is a leaf of the record's circuit, at its bits in the row. -/
theorem outLeaf_intro {ts : Array String} (hP : ReadProg P)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) {tid : String} {u : ℕ}
    {sc : Scope} (hu : UnitAt P rs tid u sc) {f l : ℕ} (hf : f < (fieldsOf (P.defs[sc.d]!).ret).length)
    (hl : l < (outRow (P.defs[sc.d]!) f).2.size) :
    ∃ g, OutLeaf P sc ("o" ++ toString f) l g (bitAt (outRow (P.defs[sc.d]!) f).2 l)
      (bitAt (outRow (P.defs[sc.d]!) f).2 l + (outRow (P.defs[sc.d]!) f).2[l]!) := by
  obtain ⟨g, hg⟩ := unit_ret_ok hP hrs hu (outRow_lt _ f l hf hl)
  exact ⟨g, f, rfl, hf, hl, hg, rfl, rfl⟩

/-- A leaf of a unit's reader row is a leaf of the record's circuit, at its bits in the row. -/
theorem rowLeaf_intro {ts : Array String} (hP : ReadProg P) (hO : NciCheck.Record.ownRows P = true)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) {r : Row} {sc : Scope}
    (hu : UnitAt P rs r.tid r.unit sc) {i : ℕ} (hi : i < (P.defs[sc.d]!).params.size)
    (hp : (P.defs[sc.d]!).params[i]!.1 = r.port) (hrow : r.row < W.cutsOf r.tid r.port) {l : ℕ}
    (hl : l < (inRow (P.defs[sc.d]!) i (W.cutsOf r.tid r.port) r.row).2.size) :
    ∃ g, RowLeaf P rs W r l g (bitAt (inRow (P.defs[sc.d]!) i (W.cutsOf r.tid r.port) r.row).2 l)
      (bitAt (inRow (P.defs[sc.d]!) i (W.cutsOf r.tid r.port) r.row).2 l +
        (inRow (P.defs[sc.d]!) i (W.cutsOf r.tid r.port) r.row).2[l]!) := by
  have hl' := hl
  rw [inRow_size _ _ _ _ hrow] at hl'
  obtain ⟨g, hg⟩ := unit_param_ok hP hO hrs hu hi (inRow_get _ i _ _ l hrow hl').1
  exact ⟨g, sc, hu, i, hi, hp, hrow, hl, hg, rfl, rfl⟩

/-! ## A parameter that is not passed on -/

/-- **A leaf of a parameter the check holds closed is no leaf its unit returns.** The parameter is none of what the
unit's Definition passes on. A leaf a unit of a composite returns is a gate of the unit, which no leaf of a
parameter is, or a leaf of a parameter the Definition passes on, and a gate is one leaf of one parameter of the
unit at most. A unit of a primitive returns its own gate. -/
theorem closed_not_returned {ts : Array String} (hP : ReadProg P) (hO : NciCheck.Record.ownRows P = true)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) {tid : String} {u : ℕ}
    {sc : Scope} (hu : UnitAt P rs tid u sc) {i t g : ℕ} (hi : i < (P.defs[sc.d]!).params.size)
    (ht : t < ((P.defs[sc.d]!).params[i]!).2.leaves)
    (hg : Flock.Program.resolve P sc (⟨true, i, t⟩ : Ref) = .ok g)
    (hn : i ∉ (passedOn P)[sc.d]?.getD (List.range (P.defs[sc.d]!).params.size)) {out : String} {l lo up : ℕ}
    (ho : OutLeaf P sc out l g lo up) : False := by
  have hout := unit_param_out hP hO hrs hu hi ht hg
  by_cases hp : (P.defs[sc.d]!).prim = true
  · obtain ⟨-, h1, h2⟩ := returnsOwn_prim hP hrs hu hp ho
    omega
  · have hp' : (P.defs[sc.d]!).prim = false := by simpa using hp
    obtain ⟨f, -, hf, hl, hres, -, -⟩ := ho
    have hlt : (outRow (P.defs[sc.d]!) f).1 + l < (P.defs[sc.d]!).body.ret.count := by
      rw [((unit_callee hP hrs hu).2 hp').2.1]
      exact outRow_lt _ f l hf hl
    have hgood := unit_good hP hrs hu hp'
    obtain ⟨hov, hbl⟩ := read_walk hP
    obtain ⟨k, m, hk, hnode, hform, hm, hargs, rfl⟩ := unit_member hP hrs hu
    obtain ⟨⟨bytes, hdec⟩, -⟩ := hP
    obtain ⟨hroot, hB, hL, hprm, hret, harg⟩ := Resolve.decoded_walk bytes P hdec hov hbl
    rcases Resolve.ret_passed_leaf P hroot hB hL hret harg _ hgood _ hlt hres with
      hin | ⟨i', j', hmem, hi', hj', hres'⟩
    · omega
    · obtain ⟨hd, -⟩ := Resolve.good_d P hroot hB _ hgood
      rw [hprm _ hd hp'] at hi' hj'
      obtain ⟨-, -, hii, -⟩ := Resolve.unit_leaf_inj bytes P hdec hov hbl hO hk hk hnode hnode hform hform hm hm
        (hargs i hi) (hargs i' hi') ht hj'
        ((P.defs[P.root]!).body.nodes[k]!).fn (P.child P.rootScope k m).off
        ((P.defs[P.root]!).body.nodes[k]!).fn (P.child P.rootScope k m).off hg hres'
      subst hii
      apply hn
      unfold passOf at hmem
      rw [passedOn_get P hd, Option.getD_some] at hmem ⊢
      exact hmem

/-! ## The two rows of a wire -/

/-- **What the record check has tested of a row with a writer.** The row is a row of a parameter of its unit, and
its writer's row an output row of the writer's unit; the two rows have the same bits; and either the parameter has
leaves of one bit and is none of what the unit's Definition passes on, or each row's leaves are of one width and
the writer's divides the reader's. The check tests a wire on the text's templates `ts`, whose ids are apart
(`tmpls_named`), each the text's reading of its Definition (`templatesOf_tmplOf`), and a unit's Definition is its
template's (`unit_tmpl`). -/
theorem wired_row {tids : Array String} (hP : ReadProg P)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) tids = .ok rs)
    (hT : Templates P rs W) (hS : Spot.Wires.check W = .ok S) {doc : Json} {ts : List Tmpl}
    (hts : templatesOf P doc = .ok ts)
    (hwh : unwhole ts (ts.map (leavesOf P (passedOn P))) S.wires = none) {r : Row} {wt : String} {wu : ℤ}
    {out : String} (hw : S.writerOf r.tid r.unit r.port r.row = some (wt, wu, out)) :
    ∃ sc sc' i f, UnitAt P rs r.tid r.unit sc ∧ 0 ≤ wu ∧ UnitAt P rs wt wu.toNat sc' ∧
      i < (P.defs[sc.d]!).params.size ∧ (P.defs[sc.d]!).params[i]!.1 = r.port ∧
      f < (fieldsOf (P.defs[sc'.d]!).ret).length ∧ out = "o" ++ toString f ∧ r.row < W.cutsOf r.tid r.port ∧
      (inRow (P.defs[sc.d]!) i (W.cutsOf r.tid r.port) r.row).2.foldl (· + ·) 0 =
        (outRow (P.defs[sc'.d]!) f).2.foldl (· + ·) 0 ∧
      ((Flock.Program.Widths.homo (P.defs[sc.d]!).params[i]!.2 = some 1 ∧
          i ∉ (passedOn P)[sc.d]?.getD (List.range (P.defs[sc.d]!).params.size)) ∨
        ∃ a o, Flock.Program.Widths.homo (P.defs[sc.d]!).params[i]!.2 = some a ∧
          Flock.Program.Widths.homo (fieldsOf (P.defs[sc'.d]!).ret)[f]! = some o ∧ o ≠ 0 ∧ a % o = 0) := by
  obtain ⟨w, hwm, rt, wtT, p, ow, hrd, hpt, hwr, hwo, -, -, hlr, hlw, hlp, hlo, hrow, hwd, hk0, hko, hkw, hur, h0,
    huw, -⟩ := wire_of_writer hS hw
  obtain ⟨hur', hTr⟩ := hT r.tid rt (mem_of_lookup _ _ _ hlr)
  obtain ⟨sc, hsc⟩ := (hur' r.unit).1 hur
  obtain ⟨-, h16, hrows, -⟩ := hTr r.unit sc hsc
  obtain ⟨huw', hTw⟩ := hT wt wtT (mem_of_lookup _ _ _ hlw)
  obtain ⟨sc', hsc'⟩ := (huw' wu.toNat).1 huw
  obtain ⟨-, -, -, houts⟩ := hTw _ sc' hsc'
  obtain ⟨i, hi, hpi, hwords, houter⟩ := (hrows r.port p).1 (mem_of_lookup _ _ _ hlp)
  obtain ⟨f, hf, hof, hn⟩ := (houts out ow).1 (mem_of_lookup _ _ _ hlo)
  obtain ⟨tr, htr, htid, -, hdr, -⟩ := unit_tmpl hP hts hrs hsc
  obtain ⟨tw, htw, hwid, -, hdw, -⟩ := unit_tmpl hP hts hrs hsc'
  have hi' : i < (P.defs[tr.defn]!).params.size := by
    rw [hdr]
    exact hi
  have hf' : f < (fieldsOf (P.defs[tw.defn]!).ret).length := by
    rw [hdw]
    exact hf
  have e1 : (P.defs[tr.defn]!).params[i] = (P.defs[sc.d]!).params[i]! := by
    rw [← getElem!_pos (P.defs[tr.defn]!).params i hi', hdr]
  have e2 : (fieldsOf (P.defs[tw.defn]!).ret)[f] = (fieldsOf (P.defs[sc'.d]!).ret)[f]! := by
    rw [← getElem!_pos (fieldsOf (P.defs[tw.defn]!).ret) f hf', hdw]
  have hp' : (P.defs[tr.defn]!).params[i].1 = w.port := by
    rw [e1, hpt]
    exact hpi.symm
  have hs : ∀ t ∈ ts, NciCheck.Record.SecurityProofs.Shaped P t := fun t ht => by
    obtain ⟨x, hx⟩ := templatesOf_tmplOf hts t ht
    exact shaped_of_tmplOf hx
  have hspec := whole_spec (tmpls_named hts).1 hs hwh (by rw [(Spot.Wires.check_sound hS).1]; exact hwm) htr htw
    (htid.trans hrd.symm) (hwid.trans hwr.symm) hi' hp' hf' (hwo.trans hof)
  rw [e1, e2, hdr] at hspec
  refine ⟨sc, sc', i, f, hsc, h0, hsc', hi, hpi.symm, hf, hof, hrow, ?_, hspec.2⟩
  rw [outRow_bits, inRow_bits _ _ _ _ (by rw [← houter]; exact hko) hrow]
  obtain ⟨m, hm⟩ := h16 i hi
  have hA : bitsOf (P.defs[sc.d]!).params[i]!.2 = 16 * p.words := by omega
  have hB : p.words = W.cutsOf r.tid r.port * ow := by
    have h1 := Nat.div_add_mod p.words (W.cutsOf r.tid r.port)
    rw [hkw, hwd, Nat.add_zero] at h1
    exact h1.symm
  have hC : bitsOf (fieldsOf (P.defs[sc'.d]!).ret)[f]! = 16 * ow := by
    rw [hn, bitsOf_eq]
    exact hspec.1
  have hD : 16 * p.words = W.cutsOf r.tid r.port * (16 * ow) := by
    rw [hB, Nat.mul_left_comm]
  rw [hA, hD, Nat.mul_div_cancel_left _ hk0, hC]

/-! ## Aligned and Filled -/

/-- **Every wire lands whole** (`Aligned`), of a record whose wires pass the check's `unwhole`. -/
theorem aligned (hP : ReadProg P) (hO : NciCheck.Record.ownRows P = true)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) (W.templates.map (·.1)).toArray = .ok rs)
    (hT : Templates P rs W) (hS : Spot.Wires.check W = .ok S) {doc : Json} {ts : List Tmpl}
    (hts : templatesOf P doc = .ok ts)
    (hwh : unwhole ts (ts.map (leavesOf P (passedOn P))) S.wires = none) : Aligned P rs W S := by
  intro r l' g' lo' up' hwl
  obtain ⟨wt, wu, out, sc₁, hw, -, hu₁, hout₁⟩ := hwl
  obtain ⟨sc, sc', i, f, hsc, -, hsc', hi, hp, hf, rfl, hrow, hbits, hwhole⟩ := wired_row hP hrs hT hS hts hwh hw
  have hout : OutLeaf P sc' ("o" ++ toString f) l' g' lo' up' := by
    rw [← unitAt_fun P rs wt wu.toNat sc₁ sc' hu₁ hsc']
    exact hout₁
  obtain ⟨hl', -, rfl, rfl⟩ := outLeaf_data hout
  rcases hwhole with ⟨-, hnot⟩ | ⟨a, o, ha, ho, ho0, hdiv⟩
  · -- the parameter is not passed on: the unit returns no leaf of the row
    refine Or.inr fun l g lo up hrl _ _ hpass => ?_
    obtain ⟨sc₂, out₂, l₂, lo₂, up₂, hsc₂, hout₂⟩ := hpass
    have hret : OutLeaf P sc out₂ l₂ g lo₂ up₂ := by
      rw [← unitAt_fun P rs r.tid r.unit sc₂ sc hsc₂ hsc]
      exact hout₂
    obtain ⟨hl, hres, -, -⟩ := rowLeaf_data (portsNamed hrs hT) hsc hi hp hrl
    rw [inRow_size _ _ _ _ hrow] at hl
    exact closed_not_returned hP hO hrs hsc hi (inRow_get _ i _ _ l hrow hl).1 hres hnot hret
  · -- each row's leaves are of one width, and the writer's divides the reader's
    obtain ⟨c, rfl⟩ := Nat.dvd_of_mod_eq_zero hdiv
    have hR := inRow_const (P.defs[sc.d]!) i (W.cutsOf r.tid r.port) r.row (o * c) hrow ha
    have hWr := outRow_const (P.defs[sc'.d]!) f o ho
    have e1 : (outRow (P.defs[sc'.d]!) f).2.size * o = (outRow (P.defs[sc'.d]!) f).2.foldl (· + ·) 0 := by
      rw [← bitAt_size, bitAt_const _ o hWr _ (Nat.le_refl _)]
    have e2 : (inRow (P.defs[sc.d]!) i (W.cutsOf r.tid r.port) r.row).2.size * (o * c) =
        (inRow (P.defs[sc.d]!) i (W.cutsOf r.tid r.port) r.row).2.foldl (· + ·) 0 := by
      rw [← bitAt_size, bitAt_const _ (o * c) hR _ (Nat.le_refl _)]
    have hcnt : (outRow (P.defs[sc'.d]!) f).2.size * o ≤
        (inRow (P.defs[sc.d]!) i (W.cutsOf r.tid r.port) r.row).2.size * (o * c) :=
      Nat.le_of_eq (by rw [e1, e2, hbits])
    obtain ⟨hc, hlt⟩ := leaf_count o c _ _ l' ho0 hcnt hl'
    obtain ⟨hin1, hin2⟩ := leaf_inside o c l' hc
    obtain ⟨g, hrl⟩ := rowLeaf_intro hP hO hrs hsc hi hp hrow hlt
    refine Or.inl ⟨l' / c, g, _, _, hrl, ?_, ?_⟩
    · rw [bitAt_const _ (o * c) hR _ (Nat.le_of_lt hlt), bitAt_const _ o hWr _ (Nat.le_of_lt hl')]
      exact hin1
    · rw [bitAt_const _ (o * c) hR _ (Nat.le_of_lt hlt), bitAt_const _ o hWr _ (Nat.le_of_lt hl'), hR _ hlt,
        hWr _ hl']
      exact hin2

/-- **A wired row is its writer's bits** (`Filled`), of a record whose wires pass the check's `unwhole`. -/
theorem filled (hP : ReadProg P)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) (W.templates.map (·.1)).toArray = .ok rs)
    (hT : Templates P rs W) (hS : Spot.Wires.check W = .ok S) {doc : Json} {ts : List Tmpl}
    (hts : templatesOf P doc = .ok ts)
    (hwh : unwhole ts (ts.map (leavesOf P (passedOn P))) S.wires = none) : Filled P rs W S := by
  intro r l g lo up hsome hrl b _ hb2
  obtain ⟨⟨wt, wu, out⟩, hw⟩ := Option.isSome_iff_exists.1 hsome
  obtain ⟨sc, sc', i, f, hsc, h0, hsc', hi, hp, hf, rfl, hrow, hbits, -⟩ := wired_row hP hrs hT hS hts hwh hw
  obtain ⟨hl, -, rfl, rfl⟩ := rowLeaf_data (portsNamed hrs hT) hsc hi hp hrl
  have hb : b < bitAt (outRow (P.defs[sc'.d]!) f).2 (outRow (P.defs[sc'.d]!) f).2.size := by
    rw [bitAt_size, ← hbits, ← bitAt_size]
    exact Nat.lt_of_lt_of_le hb2 (bitAt_leaf_le _ l hl)
  obtain ⟨l', hl', h1, h2⟩ := leaf_of_bit _ _ (Nat.le_refl _) b hb
  obtain ⟨g', hg'⟩ := outLeaf_intro hP hrs hsc' hf hl'
  exact ⟨l', g', _, _, ⟨wt, wu, _, sc', hw, h0, hsc', hg'⟩, h1, h2⟩

end NciCheck.Properties.Record.SecurityProofs
