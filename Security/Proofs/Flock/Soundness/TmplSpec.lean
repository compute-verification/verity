module

public import Proofs.Flock.Soundness.Discharge.Hm.Gadget
public import Proofs.Flock.Soundness.ExecRows
public import Proofs.Flock.Soundness.TmplUnitNet
public import Proofs.Flock.Soundness.TmplValue

/-!
# Template unroll compositionality: a templated unit's rows meet each Definition's specification at every copy

A verifier that checks a templated unit once per Definition (`Flock.Tmpl`'s `Def`s and their calls), never on the rows
it unrolls to, needs what it checks of one Definition to hold at every copy the template places. A specification is any
predicate `Spec d v inp` of a copy of Definition `d`: its own columns' values `v` and its inputs' `inp`, the constant 1.
`Steps` asks each Definition to meet its specification wherever its own rows hold, every copy of each of its calls meets
its callee's, and its unwritten columns are 0: a fact about one Definition at a time. `holds_spec` gives it at every copy
that satisfies its Definition's equations (`Flock.Tmpl.Holds`), and `unroll_spec` at the unit: where the audit's `Rows` of
the net `Tmpl.unroll` builds hold with the constant 1, the root's rows past the input rows hold, and every copy of each of
the root's calls past the input rows meets its specification. `unrolled_spec` states it for the circuit a templated
descriptor `Desc.tmpl ct` unrolls to, `HmOut.unrolled ct`, with its constant read where the audit's `Rows` put it.

A Definition's own rows force the builder rows they realize (`Realizes`): `rowHolds_of_realizes`,
`stHolds_of_realizes` and `outs_of_realizes` are `Discharge.Hm.laid_rows` and `laid_outs` for a Definition, its
references in place of a laid-out net's columns (`refsOf`, as `HmNets.cols`). So a Definition with no calls whose rows
realize a builder program's rows meets that program's `stHolds`, and its committed outputs hold their forms. No verifier
path reads this file.
-/

@[expose] public section

namespace FlockSoundness.TmplSpec

open Flock Flock.Tmpl Flock.HmNets FlockSoundness.Discharge.Hm

/-! ## Specifications closed under a Definition's rows and calls -/

/-- **Each Definition meets its specification from its rows and its calls' copies** (`Spec d v inp`: a copy of `d`,
its own columns' values `v`, its inputs' `inp`, the constant 1): wherever it is bound to as many inputs as it has, its
own rows hold, every copy of each of its calls meets its callee's specification at the copy's columns and bound inputs,
and its columns no row or copy writes are 0, Definition `d` meets its own. -/
def Steps (defs : Array Def) (Spec : ℕ → (ℕ → Bool) → List Bool → Prop) : Prop :=
  ∀ d < defs.size, ∀ (v : ℕ → Bool) (inp : List Bool), inp.length = (defs.getD d default).ins →
    (∀ r ∈ (defs.getD d default).rows.toList, v r.1 = rowVal ((defs.getD d default).refV v inp true) r) →
    (∀ c ∈ (defs.getD d default).calls.toList, ∀ j < c.count,
      Spec c.def_ (fun y => v (c.base + j * c.stride + y)) (c.boundV ((defs.getD d default).refV v inp true) j)) →
    (∀ y < (defs.getD d default).cols, (∀ r ∈ (defs.getD d default).rows.toList, r.1 ≠ y) →
      (defs.getD d default).copyAt defs y = none → v y = false) →
    Spec d v inp

/-- **A copy that satisfies its Definition's equations meets its specification**, by induction over the copies inside
it: each callee is an earlier Definition, bound to as many inputs as it has. -/
theorem holds_spec {defs : Array Def} (hwf : ∀ d < defs.size, (defs.getD d default).wellFormed defs d = true)
    {Spec : ℕ → (ℕ → Bool) → List Bool → Prop} (hS : Steps defs Spec) :
    ∀ (fuel d : ℕ) (v : ℕ → Bool) (inp : List Bool), d < defs.size → inp.length = (defs.getD d default).ins →
      Holds defs true fuel d v inp → Spec d v inp := by
  intro fuel
  induction fuel with
  | zero => intro d v inp _ _ h; exact h.elim
  | succ fuel ih =>
    intro d v inp hd hlen h
    simp only [Holds] at h
    obtain ⟨hrows, hcalls, hgaps⟩ := h
    have hw := hwf d hd
    have hw' := hw
    unfold Def.wellFormed at hw'
    simp only [Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq, beq_iff_eq] at hw'
    refine hS d hd v inp hlen hrows (fun c hc j hj => ?_) hgaps
    have hcd : c.def_ < d := (hw'.2 c hc).1.1
    have e : c.boundV ((defs.getD d default).refV v inp true) j =
        (c.bound (fun x => x) j).map ((defs.getD d default).refV v inp true) :=
      (Flock.TmplValue.bound_val c (fun x => x) j _).symm
    refine ih c.def_ _ _ (by omega) ?_ (hcalls c hc j hj)
    rw [e, List.length_map]
    exact Flock.TmplValue.call_bound_length hw hc _ j

/-- **Template unroll compositionality**: where the audit's `Rows` of the net a template unrolls to hold at `z` with the
constant 1, the root's rows past the input rows hold at `z`, and every copy of each of the root's calls past the input
rows meets its Definition's specification at its columns and bound inputs, whenever each Definition meets its
specification from its rows and its calls' copies (`Steps`). -/
theorem unroll_spec {t : Tmpl} {n : Net} (hwf : t.wellFormed = true) (ha : t.apart = true)
    (h : Flock.Tmpl.unroll t = .ok n) (hu : n.unitLog ≤ 32) {Spec : ℕ → (ℕ → Bool) → List Bool → Prop}
    (hS : Steps t.defs Spec) {z : ℕ → Bool} (hs : (Rows.ofNet (TmplValue.unroll_built h)).Sat fun c => z c.val)
    (hone : z t.constPos = true) :
    (∀ r ∈ (t.defs.getD t.root default).rows.toList, t.inCols ≤ r.1 →
      z r.1 = rowVal ((t.defs.getD t.root default).refV z [] true) r) ∧
    ∀ c ∈ (t.defs.getD t.root default).calls.toList, t.inCols ≤ c.base → ∀ j < c.count,
      Spec c.def_ (fun y => z (c.base + j * c.stride + y)) (c.boundV ((t.defs.getD t.root default).refV z [] true) j) := by
  have h' := h
  unfold Flock.Tmpl.unroll at h'
  obtain ⟨acc, hr, hn⟩ := Flock.TmplOrder.bind_ok h'
  have hsat := TmplValue.rowHolds_of_rowsSat hn hu hs
  have hwfd := Flock.TmplValue.wf_getD hwf
  have hwf' := hwf
  unfold Tmpl.wellFormed at hwf'
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range, decide_eq_true_eq, beq_iff_eq] at hwf'
  obtain ⟨⟨⟨hroot, -⟩, hcols⟩, -⟩ := hwf'
  have hrootw := hwfd t.root
  unfold Def.wellFormed at hrootw
  simp only [Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq, beq_iff_eq] at hrootw
  refine ⟨fun r hrm hrin => ?_, fun c hc hb j hj => ?_⟩
  · have hrc : r.1 < (t.defs.getD t.root default).cols := hrootw.1.2 r hrm
    have hcp : t.constPos = (t.defs.getD t.root default).cols - 1 := rfl
    unfold Tmpl.rows at hr
    obtain ⟨acc1, h1, h2⟩ := Flock.TmplOrder.bind_ok hr
    have hdf : t.defs[t.root]? = some (t.defs.getD t.root default) := by
      simp [Array.getD_eq_getD_getElem?, hroot]
    generalize hdfe : t.defs.getD t.root default = df at hdf hrm hrc hcp h1 ⊢
    simp only [place, hdf] at h1
    obtain ⟨acc2, h3, h4⟩ := Flock.TmplOrder.bind_ok h1
    let A := Flock.Tmpl.canon (r.2.1.map (df.ref 0 [] t.constPos))
    let B := Flock.Tmpl.canon (r.2.2.map (df.ref 0 [] t.constPos))
    let fr := fun (acc : Flock.Tmpl.Acc) (r : ℕ × Array ℕ × Array ℕ) =>
      write acc (0 + r.1) (Flock.Tmpl.canon (r.2.1.map (df.ref 0 [] t.constPos)))
        (Flock.Tmpl.canon (r.2.2.map (df.ref 0 [] t.constPos)))
    have hk1 : Flock.TmplValue.Kept (0 + r.1) A B acc2 :=
      (Flock.TmplRows.foldlM_mem (f := fr) Flock.TmplRows.Sized
        (fun r acc => Flock.TmplValue.Kept (0 + r.1) (Flock.Tmpl.canon (r.2.1.map (df.ref 0 [] t.constPos)))
          (Flock.Tmpl.canon (r.2.2.map (df.ref 0 [] t.constPos))) acc)
        (fun _ _ _ hu hw => Flock.TmplRows.write_sized hw hu)
        (fun _ _ _ _ _ hw hq => Flock.TmplValue.write_kept hw hq)
        (fun _ _ _ hu hw => Flock.TmplValue.write_made hu hw) (Flock.TmplRows.sized_empty _) h3).2 r hrm
    have hk2 : Flock.TmplValue.Kept (0 + r.1) A B acc1 :=
      Flock.TmplOrder.foldlM_inv (fun a => Flock.TmplValue.Kept (0 + r.1) A B a)
        (fun _ _ u u' hu hw => Flock.TmplOrder.foldlM_inv (fun a => Flock.TmplValue.Kept (0 + r.1) A B a)
          (fun _ _ w w' hw0 hp => Flock.TmplValue.place_inv _ (fun hw kk => Flock.TmplValue.write_kept hw kk) _ hp hw0)
          hu hw) hk1 h4
    have hne : 0 + r.1 ≠ t.constPos := by
      intro e
      have h0 := (Flock.TmplRows.write_eq h2).1
      rw [← e, hk2.1] at h0
      cases h0
    have hk := Flock.TmplValue.write_kept h2 hk2
    have hx := hsat (0 + r.1) (by omega) (by omega)
    unfold Flock.TmplValue.RowHolds at hx
    rw [hk.2.1, hk.2.2, Flock.TmplValue.xorAt_canon, Flock.TmplValue.xorAt_canon, Array.toList_map,
      Array.toList_map, Flock.TmplValue.xorAt_map, Flock.TmplValue.xorAt_map] at hx
    simp only [Flock.TmplValue.ref_val, Nat.zero_add, List.map_nil, hone] at hx
    exact hx
  · have hH := Flock.TmplValue.rows_holds hwf ha hr z hsat c hc hb j hj
    rw [hone] at hH
    have hcd : c.def_ < t.root := (hrootw.2 c hc).1.1
    have e : (c.bound ((t.defs.getD t.root default).ref 0 [] t.constPos) j).map z =
        c.boundV ((t.defs.getD t.root default).refV z [] true) j := by
      rw [Flock.TmplValue.bound_val]
      congr 1
      funext x
      have hx := Flock.TmplValue.ref_val (t.defs.getD t.root default) 0 [] t.constPos z x
      simp only [Nat.zero_add, List.map_nil, hone] at hx
      exact hx
    rw [e] at hH
    refine holds_spec (fun d _ => hwfd d) hS _ c.def_ _ _ (by omega) ?_ hH
    rw [← e, List.length_map]
    exact Flock.TmplValue.call_bound_length (hwfd t.root) hc _ j

/-- The audit `Rows` of the net a template unrolls to have their constant at the template's constant column. -/
theorem unroll_one {t : Tmpl} {n : Net} (h : Flock.Tmpl.unroll t = .ok n) (hB : Built n) :
    (Rows.ofNet hB).one.val = t.constPos := by
  have h' := h
  unfold Flock.Tmpl.unroll at h'
  obtain ⟨_, -, hn⟩ := Flock.TmplOrder.bind_ok h'
  obtain ⟨-, -, -, -, hc, -⟩ := Net.ofRows_ok hn
  exact (Rows.ofNet_one hB).trans hc

/-- A templated circuit's statement circuit (`HmOut.unrolled`) has its template's unrolled net as its unit. -/
theorem unrolled_unit_of {ct : Circuit} {tu : TmplUnit} {n : Net} (ht : ct.tmpl = some tu)
    (hun : Flock.Tmpl.unroll tu.t = .ok n) (hsz : ct.unitNet < ct.nets.size) : (HmOut.unrolled ct).unit = n := by
  have he : HmOut.unrolled ct = { ct with nets := ct.nets.set! ct.unitNet ("unit", n), tmpl := none } := by
    unfold HmOut.unrolled
    rw [ht]
    simp only [hun]
  rw [he]
  show ((ct.nets.set! ct.unitNet ("unit", n)).getD ct.unitNet default).2 = n
  simp [Array.getD_eq_getD_getElem?, hsz]

/-- **`unroll_spec` at a templated circuit's statement circuit**: for a circuit `ct` carrying its unit's template `tu`
(`Desc.tmpl ct` unrolls to `HmOut.unrolled ct`), where the audit's `Rows` of `HmOut.unrolled ct`'s unit hold at `z` with
their constant 1, the template's root rows past its input rows hold at `z`, and every copy of each of its root's calls past
the input rows meets its specification. -/
theorem unrolled_spec {ct : Circuit} {tu : TmplUnit} (ht : ct.tmpl = some tu) (hsz : ct.unitNet < ct.nets.size)
    (hB : Built (HmOut.unrolled ct).unit) (hu : (HmOut.unrolled ct).unit.unitLog ≤ 32)
    {Spec : ℕ → (ℕ → Bool) → List Bool → Prop} (hS : Steps tu.t.defs Spec) {z : ℕ → Bool}
    (hs : (Rows.ofNet hB).Sat fun c => z c.val) (hone : z (Rows.ofNet hB).one.val = true) :
    (∀ r ∈ (tu.t.defs.getD tu.t.root default).rows.toList, tu.t.inCols ≤ r.1 →
      z r.1 = rowVal ((tu.t.defs.getD tu.t.root default).refV z [] true) r) ∧
    ∀ c ∈ (tu.t.defs.getD tu.t.root default).calls.toList, tu.t.inCols ≤ c.base → ∀ j < c.count,
      Spec c.def_ (fun y => z (c.base + j * c.stride + y))
        (c.boundV ((tu.t.defs.getD tu.t.root default).refV z [] true) j) := by
  obtain ⟨-, n, -, -, -, hun⟩ := TmplValue.unroll_of_unrollAt tu.unrolled
  have hU := unrolled_unit_of ht hun hsz
  generalize hm : (HmOut.unrolled ct).unit = m at hB hu hs hone
  obtain rfl : m = n := hm.symm.trans hU
  have hok := tu.ok
  unfold Tmpl.unrollOk at hok
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range] at hok
  obtain ⟨⟨⟨⟨⟨⟨⟨hwf, ha⟩, -⟩, -⟩, -⟩, -⟩, -⟩, -⟩ := hok
  rw [unroll_one hun hB] at hone
  exact unroll_spec hwf ha hun hu hS hs hone

/-! ## A Definition's rows force the builder rows they realize -/

/-- A form's references in Definition `df`, its variables through `φ`: its variables' references, then the constant's
(`df.cols + df.ins`) when its constant is set (`HmNets.cols`, a laid-out net's, with `φ` for `col`). -/
def refsOf (df : Def) (φ : Var → ℕ) (f : Lin) : Array ℕ :=
  (f.s.map φ).toArray ++ (if f.c then #[df.cols + df.ins] else #[])

theorem refV_const (df : Def) (v : ℕ → Bool) (inp : List Bool) (kv : Bool) : df.refV v inp kv (df.cols + df.ins) = kv := by
  simp [Def.refV]

/-- A form's references sum, at a copy with the constant 1, to the form's value under its variables' references. -/
theorem xorAt_refsOf (df : Def) (φ : Var → ℕ) (f : Lin) (v : ℕ → Bool) (inp : List Bool) :
    xorAt (refsOf df φ f).toList (df.refV v inp true) = ev (fun x => df.refV v inp true (φ x)) f := by
  unfold refsOf
  rw [Array.toList_append, Flock.TmplValue.xorAt_append, ev_eq, List.toList_toArray, Flock.TmplValue.xorAt_map]
  congr 1
  · simp only [xorAt, List.foldr_map]
  · cases f.c <;> simp [xorAt, refV_const]

theorem xorAt_canon_eq {xs ys : Array ℕ} (h : canon xs = canon ys) (z : ℕ → Bool) :
    xorAt xs.toList z = xorAt ys.toList z := by
  rw [← Flock.TmplValue.xorAt_canon xs, h, Flock.TmplValue.xorAt_canon]

/-- **Definition `df`'s own rows realize the builder rows `rows` under `φ`**: each one's output is at one of `df`'s own
columns, where `df` has a row whose `A` and `B` are its forms' references up to order and repeated pairs (`canon`). -/
def Realizes (df : Def) (φ : Var → ℕ) (rows : List Row) : Prop :=
  ∀ r ∈ rows, φ r.out < df.cols ∧ ∃ r' ∈ df.rows.toList, r'.1 = φ r.out ∧
    canon r'.2.1 = canon (refsOf df φ r.a) ∧ canon r'.2.2 = canon (refsOf df φ r.b)

/-- **A Definition's rows force the builder rows they realize** (`laid_rows` for a Definition): at a copy whose own rows
hold, the constant 1, every realized builder row holds under its references' values (`refV ∘ φ`). -/
theorem rowHolds_of_realizes {df : Def} {φ : Var → ℕ} {rows : List Row} (hre : Realizes df φ rows) {v : ℕ → Bool}
    {inp : List Bool} (hrows : ∀ r ∈ df.rows.toList, v r.1 = rowVal (df.refV v inp true) r) :
    ∀ r ∈ rows, rowHolds (fun x => df.refV v inp true (φ x)) r := by
  intro r hr
  obtain ⟨hout, r', hr', e1, eA, eB⟩ := hre r hr
  show df.refV v inp true (φ r.out) =
    (ev (fun x => df.refV v inp true (φ x)) r.a && ev (fun x => df.refV v inp true (φ x)) r.b)
  have href : df.refV v inp true (φ r.out) = v (φ r.out) := by simp [Def.refV, hout]
  rw [href, ← e1, hrows r' hr']
  unfold rowVal
  rw [xorAt_canon_eq eA, xorAt_canon_eq eB, xorAt_refsOf, xorAt_refsOf]

/-- `rowHolds_of_realizes` for a builder state's rows. -/
theorem stHolds_of_realizes {df : Def} {φ : Var → ℕ} {st : St} (hre : Realizes df φ st.rows) {v : ℕ → Bool}
    {inp : List Bool} (hrows : ∀ r ∈ df.rows.toList, v r.1 = rowVal (df.refV v inp true) r) :
    stHolds (fun x => df.refV v inp true (φ x)) st :=
  rowHolds_of_realizes hre hrows

/-- **A Definition's rows force the outputs they commit** (`laid_outs` for a Definition): output `j`, committed at own
column `oc j` by a row `form · 1`, holds its form under its references' values (`refV ∘ φ`). -/
theorem outs_of_realizes {df : Def} {φ : Var → ℕ} {outs : List Lin} {oc : ℕ → ℕ}
    (hre : ∀ j (hj : j < outs.length), oc j < df.cols ∧ ∃ r' ∈ df.rows.toList, r'.1 = oc j ∧
      canon r'.2.1 = canon (refsOf df φ outs[j]) ∧ canon r'.2.2 = #[df.cols + df.ins])
    {v : ℕ → Bool} {inp : List Bool} (hrows : ∀ r ∈ df.rows.toList, v r.1 = rowVal (df.refV v inp true) r) :
    ∀ j (hj : j < outs.length), v (oc j) = ev (fun x => df.refV v inp true (φ x)) outs[j] := by
  intro j hj
  obtain ⟨-, r', hr', e1, eA, eB⟩ := hre j hj
  rw [← e1, hrows r' hr']
  unfold rowVal
  rw [xorAt_canon_eq eA, xorAt_refsOf, ← Flock.TmplValue.xorAt_canon r'.2.2, eB]
  simp [xorAt, refV_const]

end FlockSoundness.TmplSpec
