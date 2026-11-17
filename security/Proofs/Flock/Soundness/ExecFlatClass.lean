module

public import Flock.HmRow
public import Proofs.Flock.Soundness.ComposePlace
public import Proofs.Flock.Soundness.ExecTemplateSetup
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Combinatorics.Matroid.Init
import Mathlib.Data.List.GetD
import Mathlib.Data.Nat.Totient
import Proofs.Flock.Soundness.SparseRows
import Proofs.Flock.Soundness.Types.Parts

/-! A typed flat class's unit, placed at each unit slot of the accepted statement (`setupH_flatPlacement`). -/

@[expose] public section

open Flock FlockLevel3

namespace FlockSoundness

open Compose Flock.DeriveAll Types.Dag

theorem netOfD_built {d : D} {name : String} {n : Net} (h : Typed.netOfD d name = .ok n) : Built n :=
  ⟨name, d.size, d.const, _, _, Typed.rowsA d, Typed.rowsB d, h⟩

def PastInputs (done : Array D) (u : D) (net : Net) : Prop :=
  ∀ c ∈ (order done u).drop u.inCols.size, net.inWords * Flock.WORD ≤ c

theorem read_flat' {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {k : Typed.Class} {net : Net}
    (h : Typed.read tags bytes tables = .ok (k, .inl net)) :
    Typed.isFlat (k.done.getD k.unit default) = true ∧ ∃ name, Typed.netOfD (k.done.getD k.unit default) name = .ok net ∧
      PastInputs k.done (k.done.getD k.unit default) net ∧
      (k.done.getD k.unit default).inCols.toList = (portCols net.inGroups).toList.flatMap fun p => List.range' p.1 p.2 := by
  refine ⟨?_, Typed.read_flat h⟩
  unfold Typed.read at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  split at h
  · simp at h
  split at h
  · simp at h
  split at h
  · rename_i hflat
    split at h
    · simp at h
    · simp only [Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, -⟩ := h
      exact hflat
  · split at h <;> simp at h

section unit

variable {types : String → Option CircuitType} {ls : Array Flock.BlockLayout} {info : String → Option TableInfo}
  {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D}

theorem netRows_flat (hd : deriveChecked types ls info words unit = .ok done)
    (hflat : Typed.isFlat (done.getD unit default) = true) {name : String} {net : Net}
    (hnet : Typed.netOfD (done.getD unit default) name = .ok net) (hlog : net.unitLog ≤ 32)
    (hpast : PastInputs done (done.getD unit default) net) :
    NetRows net (ofBlock words done (done.getD unit default)) := by
  have hdc := delta_const hd (unit_lt hd)
  have hcols := order_cols hd
  obtain ⟨-, hlt, -, -, -⟩ := order_sound hd 0
  obtain ⟨hul, ha, hb, -, hcst, -⟩ := Typed.netOfD_ok hnet
  have hcu := (ofRows_groups (show Net.ofRows _ _ _ _ _ _ _ = .ok net from hnet)).1
  generalize done.getD unit default = u at *
  simp only [Typed.isFlat, Bool.and_eq_true, Array.all_eq_true_iff_forall_mem, beq_iff_eq,
    Option.isNone_iff_eq_none] at hflat
  obtain ⟨⟨hparts, hreads⟩, hdelta⟩ := hflat
  have hpe : u.parts = #[] := Array.isEmpty_iff.1 hparts
  have hre : u.reads = #[] := Array.isEmpty_iff.1 hreads
  have hsize : ∀ c ∈ order done u, c < u.size := fun c hc => by
    rcases hcols c hc with h | ⟨p, hp, -⟩
    · exact h
    · rw [hpe] at hp; simp at hp
  have hw : 2 ^ net.unitLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by norm_num) hlog
  have hmem : ∀ k < (order done u).length - u.inCols.size,
      (order done u).getD (u.inCols.size + k) 0 ∈ (order done u).drop u.inCols.size := fun k hk => by
    rw [List.getD_eq_getElem _ _ (by omega)]
    exact List.mem_iff_getElem.2 ⟨k, by simp; omega, List.getElem_drop⟩
  have hmem' : ∀ k < (order done u).length - u.inCols.size,
      (order done u).getD (u.inCols.size + k) 0 ∈ order done u :=
    fun k hk => List.mem_of_mem_drop (hmem k hk)
  have hfind : ∀ c e, u.delta.find? (·.1 == c) = some e → c = u.const ∧ e = (u.const, none) := by
    intro c e he
    have h1 := Array.find?_some he
    have h2 := hdelta e (Array.mem_of_find?_eq_some he)
    simp only [beq_iff_eq] at h1
    exact ⟨h1 ▸ h2.1, Prod.ext h2.1 h2.2⟩
  refine ⟨fun c hc => ?_, fun k hk p hp => ?_, fun k hk hp => ?_⟩
  · have := hsize c hc
    omega
  · generalize hc : (order done u).getD (u.inCols.size + k) 0 = c at hp
    have hcm := hc ▸ hmem' k hk
    have hcs := hsize c hcm
    have hcr := hlt c hcm
    simp only at hp
    split at hp
    · cases hp
    · rename_i s hs
      exact absurd (hfind c _ hs).2 (by simp)
    · rename_i hnone
      have hne : c ≠ u.const := by
        rintro rfl
        rw [hdc] at hnone
        cases hnone
      have hfr : fullRow words u c = u.rows[c]? := by
        unfold fullRow
        rw [hre]
        rfl
      rw [hfr, Array.getElem?_eq_getElem hcr, Option.getD_some, Option.some.injEq] at hp
      subst hp
      have hrc : u.rows.getD c ([], []) = u.rows[c] := getD_of_getElem? (Array.getElem?_eq_getElem hcr)
      refine ⟨hc ▸ hpast _ (hmem k hk), by rw [hcst]; omega, ?_⟩
      rw [ofRows_row ha hw, ofRows_row hb hw]
      simp only [Typed.rowsA, Typed.rowsB, Array.getD_eq_getD_getElem?, Array.getElem?_map, Array.getElem?_range,
        show c < u.size from hcs, ↓reduceIte, Option.map_some, Option.getD_some, List.toList_toArray]
      rw [← Array.getD_eq_getD_getElem?, hrc]
  · generalize hc : (order done u).getD (u.inCols.size + k) 0 = c at hp
    simp only at hp
    split at hp
    · rename_i x hx
      rw [hcst]
      exact (hfind c _ hx).1
    · cases hp
    · cases hp

end unit

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}

theorem setupH_flatLayout (ht : tags.typed = true)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) {k : Typed.Class} {net : Net}
    (hr : Typed.read tags cf tables = .ok (k, .inl net)) : st.c.cls = some k ∧ st.c.unit = net ∧ Layout st := by
  obtain ⟨hp, hpin, hda, hdb, hm, hk⟩ := setupH_spec_typed ht h
  obtain ⟨k', r, hr', hcase⟩ := parseTyped_spec hp
  rw [hr] at hr'
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hr')
  rcases hcase with ⟨t, c', hrt, -, -⟩ | ⟨net', c', hrn, hpc, hc⟩
  · cases hrt
  · cases hrn
    obtain ⟨name, hnet, -⟩ := Typed.read_flat hr
    obtain ⟨P, ⟨more, hmore⟩, hunit, -⟩ := parse_facts_pre (fun x hx => by
      simp at hx
      subst hx
      exact netOfD_built hnet) hpc
    have P' : ParseFacts tags st.c := by
      rw [hc]
      exact ⟨P.nets, P.portWords, P.v1Segs, P.rowWires, P.unitNet, P.shaNet, P.hmNet, P.kinds, P.check,
        (by rw [← P.rowNets]; exact HmNets.check_congr rfl rfl rfl),
        (by rw [← P.recNet]; exact RecOpen.check_congr rfl rfl rfl rfl rfl rfl rfl),
        (by rw [← P.recResiduals]; exact RecResiduals.check_congr rfl rfl),
        (by rw [← P.recSparse]; exact RecSparse.check_congr rfl rfl),
        (by rw [← P.recOpening]; exact RecOpening.check_congr rfl rfl), P.typed, P.tmpl⟩
    refine ⟨by rw [hc], ?_, layout_of P' hpin hda hdb hm hk⟩
    have h0 : c'.nets[0]? = some ("unit", net) := by
      rw [← Array.getElem?_toList, hmore]
      simp
    have hu0 : c'.unitNet = 0 := by
      obtain ⟨hlt, -, hmin⟩ := Array.findIdx?_eq_some_iff_getElem.1 hunit
      by_contra hne
      have h1 := hmin 0 (Nat.pos_of_ne_zero hne)
      rw [Array.getElem?_eq_getElem (by omega)] at h0
      simp [Option.some.inj h0] at h1
    rw [hc]
    show (c'.nets.getD c'.unitNet default).2 = net
    rw [hu0, getD_of_getElem? h0]

theorem setupH_flatRealizes (ht : tags.typed = true)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) {k : Typed.Class} {net : Net}
    (hr : Typed.read tags cf tables = .ok (k, .inl net))
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {s₀ : ℕ} (hs : s₀ < st.c.unitRange.count) :
    Realizes ((setupH_flatLayout ht h hr).2.2.model regs)
      (ofBlock (Typed.wordsIn k.held) k.done (k.done.getD k.unit default)) (fun c => st.c.unitRange.slot s₀ + c) := by
  have L := (setupH_flatLayout ht h hr).2.2
  have hunit := (setupH_flatLayout ht h hr).2.1
  obtain ⟨hflat, name, hnet, hpast, -⟩ := read_flat' hr
  have hlog : net.unitLog ≤ 32 := hunit ▸ (setupH_flatLayout ht h hr).2.2.unitLog_le hs
  refine (setupH_flatLayout ht h hr).2.2.realizes regs hs (by rw [hunit]; exact netOfD_built hnet) ?_
  rw [hunit]
  exact netRows_flat k.hd hflat hnet hlog hpast

theorem setupH_flatPlacement (ht : tags.typed = true)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) {k : Typed.Class} {net : Net}
    (hr : Typed.read tags cf tables = .ok (k, .inl net))
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {s₀ : ℕ} (hs : s₀ < st.c.unitRange.count) :
    Placement ((setupH_flatLayout ht h hr).2.2.model regs)
      (Rows.compose (ofBlock (Typed.wordsIn k.held) k.done (k.done.getD k.unit default)) (ofBlock_ordered k.hd))
      (composePlace _ _ (ofBlock_ordered k.hd) (fun c => st.c.unitRange.slot s₀ + c)
        (setupH_flatRealizes ht h hr regs hs).lt) :=
  (setupH_flatRealizes ht h hr regs hs).placement (ofBlock_ordered k.hd)

end FlockSoundness
