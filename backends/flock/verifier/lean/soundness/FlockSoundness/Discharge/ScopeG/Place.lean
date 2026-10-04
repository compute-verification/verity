import FlockSoundness.Discharge.HiddenExec.Placed
import FlockSoundness.Discharge.ScopeG.Place.Flat
import FlockSoundness.Discharge.ScopeG.Place.Delta
import FlockSoundness.Discharge.ScopeG.Place.Stage
import FlockSoundness.Discharge.ScopeG.Place.Lookup
import FlockSoundness.Discharge.ScopeG.Place.Bounds
import FlockSoundness.Discharge.ScopeG.Place.Cols

set_option Elab.async false

/-!
# P1: the VU's rows placed at an accepted hidden statement

At an `AcceptedH` of a circuit at `scopeOkGT`, every VU `g` places the VU's rows (`vuRows`, `Rows.compose vuLogical`) at
`composePlace` of `vuPos c g`, its inputs on their unit slots (`vuInPos`): `vuPlace_at`. It is
`Compose.Realizes.placement` of `vuRealizes`, which assembles the block's rows at every column of `vuLogical`:

* the hidden statement reads its flat twin's rows there (`Place/Flat.lean`, `vuRowsFlat`);
* the flat twin's ranges' share is each net's row moved to its slot (`Place/Stage.lean`), a lookup slot's table share
  `lookupCols`' columns (`Place/Lookup.lean`);
* Δ adds the wired inputs' copies and nothing at computed rows (`Place/Delta.lean`);
* the slots' constants read the pin, every column is in the block (`Place/Bounds.lean`), the pin's row reads the pin
  (`Layout.placed_pin`);
* the order's columns decode into slots, and `vuPos` moves their reads to VU `g` (`Place/Cols.lean`).
-/

namespace FlockSoundness.Discharge.ScopeG

open Flock FlockLevel3 Compose
open FlockSoundness.Discharge.HiddenExec

/-- `slotRow` copies `one` only at the slot's constant. -/
theorem slotRow_none {c : Circuit} {k i x : ℕ} (h : slotRow c k i x = none) :
    x = (c.nets.getD k default).2.constPos := by
  by_contra hne
  revert h
  unfold slotRow
  simp only [if_neg hne]
  split_ifs
  · split <;> simp
  · simp

/-- `slotRowG` at an input row: `wireSrcG`'s column, or nothing. -/
theorem slotRowG_in {c : Circuit} {g k i x : ℕ} {p1 p2 : List ℕ} (hne : x ≠ (c.nets.getD k default).2.constPos)
    (hlt : x < netIn (c.nets.getD k default).2) (h : slotRowG c g k i x = some (p1, p2)) :
    p1 = (wireSrcG c g k i x).toList ∧ p2 = (wireSrcG c g k i x).toList := by
  unfold slotRowG at h
  simp only [if_neg hne, if_pos hlt] at h
  cases hw : wireSrcG c g k i x with
  | none => simp only [hw, Option.some.injEq, Prod.mk.injEq] at h; exact ⟨h.1.symm, h.2.symm⟩
  | some s => simp only [hw, Option.some.injEq, Prod.mk.injEq] at h; exact ⟨h.1.symm, h.2.symm⟩

/-- `slotRowG` at a computed row: the net's rows moved to the slot, a lookup slot's table columns on `B`. -/
theorem slotRowG_comp {c : Circuit} {g k i x : ℕ} {p1 p2 : List ℕ}
    (hne : x ≠ (c.nets.getD k default).2.constPos) (hlt : ¬x < netIn (c.nets.getD k default).2)
    (h : slotRowG c g k i x = some (p1, p2)) :
    p1 = ((c.nets.getD k default).2.a.row x).toList.map (c.slotCol k g i + ·) ∧
      p2 = (((c.nets.getD k default).2.b.row x).toList ++ lookupCols (c.nets.getD k default).2.lookup x).map
        (c.slotCol k g i + ·) := by
  unfold slotRowG at h
  simp only [if_neg hne, if_neg hlt, Option.some.injEq, Prod.mk.injEq] at h
  exact ⟨h.1.symm, h.2.symm⟩

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}

/-- **The block realizes the VU's rows at VU `g`**: `Compose.Realizes` of `vuLogical` at `vuPos st.c g`, in the
statement of an untyped hidden setup at `scopeOkGT`. -/
theorem vuRealizes (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) (L : Layout st)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) (hs : scopeOkGT st.c = true) {g : ℕ}
    (hg : g < st.c.g) : Realizes (L.model regs) (vuLogical st.c) (vuPos st.c g) := by
  have Lf : Layout (Hidden.flatTwin st) := flat_layout ht h
  have F : Aliased.UnitFacts tags (Hidden.flatTwin st) := flat_unitFacts ht h
  have hfl : VuRowsFlat st := vuRowsFlat ht h
  have hO := scopeOkGT_ordered hs
  have hpinv : ∀ j : Fin (2 ^ (L.model regs).kLog), (j = (L.model regs).pin) ↔ j.val = st.pin :=
    fun j => ⟨fun h => by rw [h]; rfl, fun h => Fin.ext h⟩
  have hmem : ∀ ℓ < (vuOrder st.c).length, (vuOrder st.c).getD ℓ 0 ∈ vuOrder st.c := fun ℓ hℓ => by
    rw [List.getD_eq_getElem _ _ hℓ]; exact List.getElem_mem hℓ
  have hcomp : ∀ k' < (vuOrder st.c).length - (vuInCols st.c).length,
      (vuOrder st.c).getD ((vuInCols st.c).length + k') 0 ∈ vuOrder st.c := fun k' hk' => hmem _ (by omega)
  have hdec : ∀ {p}, p ∈ vuOrder st.c → ∃ k ∈ vuNets st.c, ∃ i < st.c.perVu.getD k 0,
      ∃ x ≤ (st.c.nets.getD k default).2.constPos, p = st.c.slotCol k 0 i + x ∧ vuRow st.c p = slotRow st.c k i x ∧
        vuPos st.c g p = st.c.slotCol k g i + x ∧
        (p ∉ vuInCols st.c → k = st.c.unitNet → x < netIn (st.c.nets.getD k default).2 →
          (wireSrcG st.c g k i x).isSome) :=
    fun hp => vuOrder_decode Lf F hs hp g
  refine ⟨fun ℓ hℓ => ?_, fun k' hk' pr hpr r j hr => ?_, fun k' hk' pr hpr r j hr => ?_,
    fun k' hk' hpr r j hr => ?_, fun k' hk' hpr r j hr => ?_, fun j => ?_, fun j => ?_⟩
  · obtain ⟨k, hk, i, hi, x, hx, -, -, hpos, -⟩ := hdec (hmem ℓ hℓ)
    show vuPos st.c g ((vuOrder st.c).getD ℓ 0) < 2 ^ st.c.kLog
    rw [hpos]
    exact vuSlot_lt Lf F hk hg hi hx
  · obtain ⟨k, hk, i, hi, x, hx, -, hrow, hpos, hw⟩ := hdec (hcomp k' hk')
    have hpr' : slotRow st.c k i x = some pr := hrow.symm.trans hpr
    have hG : slotRowG st.c g k i x = some (pr.1.map (vuPos st.c g), pr.2.map (vuPos st.c g)) :=
      slotRow_vuPos Lf F hs hk hi hx hpr' (hO.mem k' hk' pr hpr) g
    have hr' : r.val = st.c.slotCol k g i + x := hr.trans hpos
    have hne : x ≠ (st.c.nets.getD k default).2.constPos := fun he => by
      rw [he] at hpr'; simp [slotRow] at hpr'
    have hS : rangesEntry (Hidden.flatTwin st) (slotEntryA (Hidden.flatTwin st)) r.val j.val =
        ((((st.c.nets.getD k default).2.a.row x).toList.map (st.c.slotCol k g i + ·)).count j.val : ZMod 2) :=
      stage_rowA Lf F hk hg hi (lt_of_le_of_ne hx hne) r j hr'
    show placedA st r j = _
    rw [(hfl k hk g hg i hi x hx r j hr').1]
    show rangesEntry (Hidden.flatTwin st) (slotEntryA (Hidden.flatTwin st)) r.val j.val +
      deltaEntry (Hidden.flatTwin st).da r.val j.val = (((pr.1.map (vuPos st.c g)).count j.val : ℕ) : ZMod 2)
    rw [hS]
    by_cases hlt : x < netIn (st.c.nets.getD k default).2
    · rw [(slotRowG_in hne hlt hG).1]
      exact delta_inA Lf F hk hg hi hlt (fun hu => hw (vuOrder_comp_notin hs hk') hu hlt) r j hr'
    · have h0 : deltaEntry (Hidden.flatTwin st).da r.val j.val = 0 :=
        (delta_comp Lf F hk hg hi (not_lt.1 hlt) (lt_of_le_of_ne hx hne) r j hr').1
      rw [h0, add_zero, (slotRowG_comp hne hlt hG).1]
  · obtain ⟨k, hk, i, hi, x, hx, -, hrow, hpos, hw⟩ := hdec (hcomp k' hk')
    have hpr' : slotRow st.c k i x = some pr := hrow.symm.trans hpr
    have hG : slotRowG st.c g k i x = some (pr.1.map (vuPos st.c g), pr.2.map (vuPos st.c g)) :=
      slotRow_vuPos Lf F hs hk hi hx hpr' (hO.mem k' hk' pr hpr) g
    have hr' : r.val = st.c.slotCol k g i + x := hr.trans hpos
    have hne : x ≠ (st.c.nets.getD k default).2.constPos := fun he => by
      rw [he] at hpr'; simp [slotRow] at hpr'
    have hS : rangesEntry (Hidden.flatTwin st) (slotEntryB (Hidden.flatTwin st)) r.val j.val =
        ((((st.c.nets.getD k default).2.b.row x).toList.map (st.c.slotCol k g i + ·)).count j.val : ZMod 2) +
          lookupAt st.c k g i x j.val :=
      stage_rowB Lf F hk hg hi (lt_of_le_of_ne hx hne) r j hr'
    show placedB st r j = _
    rw [(hfl k hk g hg i hi x hx r j hr').2]
    show rangesEntry (Hidden.flatTwin st) (slotEntryB (Hidden.flatTwin st)) r.val j.val +
      deltaEntry (Hidden.flatTwin st).db r.val j.val = (((pr.2.map (vuPos st.c g)).count j.val : ℕ) : ZMod 2)
    rw [hS]
    by_cases hlt : x < netIn (st.c.nets.getD k default).2
    · rw [(slotRowG_in hne hlt hG).2]
      exact delta_inB Lf F hk hg hi hlt (fun hu => hw (vuOrder_comp_notin hs hk') hu hlt) r j hr'
    · have h0 : deltaEntry (Hidden.flatTwin st).db r.val j.val = 0 :=
        (delta_comp Lf F hk hg hi (not_lt.1 hlt) (lt_of_le_of_ne hx hne) r j hr').2
      have hl : lookupAt st.c k g i x j.val =
          ((((lookupCols (st.c.nets.getD k default).2.lookup x).map (st.c.slotCol k g i + ·)).count j.val : ℕ) :
            ZMod 2) :=
        lookup_rowB Lf F hk hg hi (lt_of_le_of_ne hx hne) j.val
      rw [h0, add_zero, hl, (slotRowG_comp hne hlt hG).2, List.map_append, List.count_append, Nat.cast_add]
  · obtain ⟨k, hk, i, hi, x, hx, -, hrow, hpos, -⟩ := hdec (hcomp k' hk')
    have hx' := slotRow_none (hrow.symm.trans hpr)
    have hr' : r.val = st.c.slotCol k g i + (st.c.nets.getD k default).2.constPos := by
      rw [← hx']; exact hr.trans hpos
    show placedA st r j = _
    rw [(hfl k hk g hg i hi x hx r j (hr.trans hpos)).1, if_congr (hpinv j) rfl rfl]
    exact const_rowA Lf F hk hg hi r j hr'
  · obtain ⟨k, hk, i, hi, x, hx, -, hrow, hpos, -⟩ := hdec (hcomp k' hk')
    have hx' := slotRow_none (hrow.symm.trans hpr)
    have hr' : r.val = st.c.slotCol k g i + (st.c.nets.getD k default).2.constPos := by
      rw [← hx']; exact hr.trans hpos
    show placedB st r j = _
    rw [(hfl k hk g hg i hi x hx r j (hr.trans hpos)).2, if_congr (hpinv j) rfl rfl]
    exact const_rowB Lf F hk hg hi r j hr'
  · rw [if_congr (hpinv j) rfl rfl]
    exact (L.placed_pin j).1
  · rw [if_congr (hpinv j) rfl rfl]
    exact (L.placed_pin j).2

/-- **The VU's rows placed at an accepted hidden statement, at VU `g`, its inputs on their unit slots.** -/
theorem vuPlace_at {T : Model.Statement} (a : AcceptedH T) {c : Flock.Circuit} (hc : a.st.c = c)
    (hs : scopeOkGT c = true) {g : ℕ} (hg : g < c.g) :
    ∃ col : Fin (vuRows c hs).w → Fin (2 ^ T.kLog),
      (∀ x : Fin (vuRows c hs).w, x.val < vuB c hs → (col x).val = vuInPos c g x.val) ∧
        Placement T (vuRows c hs) col := by
  revert hc
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht, h, L, -, regs, hT⟩ := a
  intro hc
  subst hT
  subst hc
  have hre := vuRealizes ht h L regs hs hg
  refine ⟨composePlace _ _ (scopeOkGT_ordered hs) (vuPos st.c g) hre.lt, fun x hx => ?_,
    hre.placement (scopeOkGT_ordered hs)⟩
  have hx' : x.val < (vuInCols st.c).length := by rw [vuInCols, List.length_map, ← vuB_eq st.c hs]; exact hx
  rw [composePlace_val, if_pos (lt_of_lt_of_le hx' (scopeOkGT_ordered hs).le)]
  show vuPos st.c g ((vuOrder st.c).getD x.val 0) = _
  rw [vuOrder_getD_in st.c hx', vuPos_in (flat_layout ht h) (flat_unitFacts ht h) hs hx' g]

end FlockSoundness.Discharge.ScopeG
