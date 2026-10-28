import Proofs.Flock.Soundness.ComposeDag
import Proofs.Flock.Soundness.ExecSetup

/-!
# Composed rows, placed: what 1d and 1e hand to a table's class (`dp`)

`placement_of_realizes` (`Compose.lean`) proves `Placement` for `Rows.compose` from facts about the block matrices. Here
those facts are one proposition, and they are discharged for a flat statement `Stmt.setupH` accepted:
- **`Compose.Realizes S L pos`**: each logical column's position is in the block; at each computed row's position the
  matrices read exactly the positions of its columns, or the pin for a row that copies `one`; and the pin's row reads
  the pin. `Realizes.placement` is the `Placement` it gives, at `composePlace`.
- **`Compose.BlockFacts`**: 1e's form of the same facts, over the unit's block rows (`blockRow`, Δ applied), for the
  verifier's own `deriveChecked` run. `BlockFacts.realizes` turns them into `Realizes` for the unit's composed rows
  (`ofBlock`), and `BlockFacts.placement` is a table class's `placed` at a slot (the soundness lane's `TableClass`).
- **`Layout.realizes`**: in the model of a statement `setupH` accepted (`Layout`),
  at unit slot `s₀` with positions `slot s₀ + c`, `Realizes` holds for every `Logical` whose rows are the unit net's
  (`Compose.NetRows`). It reads Δ only as `Layout` does, for statements without a typed template.

`NetRows st.c.unit (ofBlock words done u)` is the link between the unit net `setupH` accepted and the rows the verifier
derives. For a typed flat class, whose unit net is built from those rows (`Typed.netOfD`), it is `netRows_flat`
(`ExecFlatClass.lean`).
-/

open Flock FlockLevel3

namespace FlockSoundness

open Audit

namespace Compose

/-- **The block matrices realize the composed rows at `pos`**: `placement_of_realizes`'s hypotheses, with the positions in
the block. -/
structure Realizes (S : Model.Statement) (L : Logical) (pos : ℕ → ℕ) : Prop where
  lt : ∀ ℓ < L.order.length, pos (L.order.getD ℓ 0) < 2 ^ S.kLog
  rowA : ∀ k < L.order.length - L.nIn, ∀ p, L.row (L.order.getD (L.nIn + k) 0) = some p →
    ∀ r j : Fin (2 ^ S.kLog), r.val = pos (L.order.getD (L.nIn + k) 0) →
      S.A₀ r j = (((p.1.map pos).count j.val : ℕ) : ZMod 2)
  rowB : ∀ k < L.order.length - L.nIn, ∀ p, L.row (L.order.getD (L.nIn + k) 0) = some p →
    ∀ r j : Fin (2 ^ S.kLog), r.val = pos (L.order.getD (L.nIn + k) 0) →
      S.B₀ r j = (((p.2.map pos).count j.val : ℕ) : ZMod 2)
  oneA : ∀ k < L.order.length - L.nIn, L.row (L.order.getD (L.nIn + k) 0) = none →
    ∀ r j : Fin (2 ^ S.kLog), r.val = pos (L.order.getD (L.nIn + k) 0) → S.A₀ r j = if j = S.pin then 1 else 0
  oneB : ∀ k < L.order.length - L.nIn, L.row (L.order.getD (L.nIn + k) 0) = none →
    ∀ r j : Fin (2 ^ S.kLog), r.val = pos (L.order.getD (L.nIn + k) 0) → S.B₀ r j = if j = S.pin then 1 else 0
  pinA : ∀ j, S.A₀ S.pin j = if j = S.pin then 1 else 0
  pinB : ∀ j, S.B₀ S.pin j = if j = S.pin then 1 else 0

theorem Realizes.placement {S : Model.Statement} {L : Logical} {pos : ℕ → ℕ} (hr : Realizes S L pos) (h : Ordered L) :
    Placement S (Rows.compose L h) (composePlace S L h pos hr.lt) :=
  placement_of_realizes hr.rowA hr.rowB hr.oneA hr.oneB hr.pinA hr.pinB

/-- **The composed rows are the net's**: every column of the order is at most the net's constant; each computed
row that doesn't copy `one` sits between the net's input rows and its constant and is the net's row there, in `A` and in
`B`; and a row that copies `one` is the net's constant. -/
structure NetRows (net : Net) (L : Logical) : Prop where
  le : ∀ c ∈ L.order, c ≤ net.constPos
  rows : ∀ k < L.order.length - L.nIn, ∀ p, L.row (L.order.getD (L.nIn + k) 0) = some p →
    net.inWords * WORD ≤ L.order.getD (L.nIn + k) 0 ∧ L.order.getD (L.nIn + k) 0 < net.constPos ∧
      p = ((net.a.row (L.order.getD (L.nIn + k) 0)).toList, (net.b.row (L.order.getD (L.nIn + k) 0)).toList)
  one : ∀ k < L.order.length - L.nIn, L.row (L.order.getD (L.nIn + k) 0) = none →
    L.order.getD (L.nIn + k) 0 = net.constPos

open Flock.DeriveAll Types.Dag

/-- **1e's facts at one slot of a table** (the plan's flock-verifier item 2): at each non-input column `c` of the unit's
order, the block matrices' rows at `pos c` read the unit's block row (`blockRow`: Δ's copy, else the row as the verifier
folds it), each column at `pos` and `one` at the pin; and the pin's row reads the pin. -/
structure BlockFacts (S : Model.Statement) (words : String → Option (Array ℕ)) (done : Array D) (u : D) (one : ℕ)
    (pos : ℕ → ℕ) : Prop where
  lt : ∀ c ∈ order done u, pos c < 2 ^ S.kLog
  rowA : ∀ c ∈ order done u, c ∉ u.inCols.toList → ∀ p, blockRow words u one c = some p →
    ∀ r j : Fin (2 ^ S.kLog), r.val = pos c →
      S.A₀ r j = (((p.1.map fun x => if x = one then S.pin.val else pos x).count j.val : ℕ) : ZMod 2)
  rowB : ∀ c ∈ order done u, c ∉ u.inCols.toList → ∀ p, blockRow words u one c = some p →
    ∀ r j : Fin (2 ^ S.kLog), r.val = pos c →
      S.B₀ r j = (((p.2.map fun x => if x = one then S.pin.val else pos x).count j.val : ℕ) : ZMod 2)
  pinA : ∀ j, S.A₀ S.pin j = if j = S.pin then 1 else 0
  pinB : ∀ j, S.B₀ S.pin j = if j = S.pin then 1 else 0

theorem blockRow_ne_none {words : String → Option (Array ℕ)} {d : D} {one c : ℕ} (hc : c < d.rows.size) :
    blockRow words d one c ≠ none := by
  have hr : d.rows[c]? = some d.rows[c] := Array.getElem?_eq_getElem hc
  unfold blockRow fullRow
  split
  · simp
  · simp
  · split <;> simp [hr]

theorem count_pin {S : Model.Statement} (j : Fin (2 ^ S.kLog)) :
    ((([S.pin.val].count j.val : ℕ) : ZMod 2)) = if j = S.pin then 1 else 0 := by
  by_cases hj : j = S.pin
  · subst hj; simp
  · rw [ite_eq_right hj, List.count_singleton]
    simp [Fin.val_inj, Ne.symm hj]

/-- **1e's facts give the class's placement facts.** For the verifier's own run `deriveChecked … = .ok done`, with `one`
past every row of the unit, `BlockFacts` at `pos` is `Realizes` for the unit's composed rows (`ofBlock`) at `pos`. -/
theorem BlockFacts.realizes {types : String → Option CircuitType} {ls : Array Flock.Layout} {info : String → Option TableInfo}
    {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D}
    (hd : deriveChecked types ls info words unit = .ok done) {S : Model.Statement} {one : ℕ} {pos : ℕ → ℕ}
    (hone : (done.getD unit default).rows.size ≤ one) (hf : BlockFacts S words done (done.getD unit default) one pos) :
    Realizes S (ofBlock words done (done.getD unit default)) pos := by
  have h := ofBlock_ordered hd
  obtain ⟨hn, hlt, htake, -, -⟩ := order_sound hd one
  generalize done.getD unit default = u at *
  have hlen : u.inCols.size ≤ (order done u).length := by
    have := congrArg List.length htake
    simp only [List.length_take, Array.length_toList] at this
    omega
  have hmem : ∀ ℓ < (order done u).length, (order done u).getD ℓ 0 ∈ order done u := fun ℓ hℓ => by
    rw [List.getD_eq_getElem _ _ hℓ]; exact List.getElem_mem hℓ
  have hnotin : ∀ k < (order done u).length - u.inCols.size,
      (order done u).getD (u.inCols.size + k) 0 ∉ u.inCols.toList := by
    intro k hk hin
    rw [← htake] at hin
    obtain ⟨i, hi, hie⟩ := List.getElem_of_mem hin
    have hi' : i < u.inCols.size := by simp at hi; omega
    rw [List.getElem_take] at hie
    have hℓ : u.inCols.size + k < (order done u).length := by omega
    rw [List.getD_eq_getElem _ _ hℓ] at hie
    have := (List.Nodup.getElem_inj_iff hn).1 hie
    omega
  have hne : ∀ c ∈ order done u, c ≠ one := fun c hc => by have := hlt c hc; omega
  have hcases := fun k => ofBlock_cases words done u ((order done u).getD (u.inCols.size + k) 0)
  have hnone : ∀ k < (order done u).length - u.inCols.size,
      blockRow words u one ((order done u).getD (u.inCols.size + k) 0) ≠ none := by
    intro k hk
    exact blockRow_ne_none (hlt _ (hmem (u.inCols.size + k) (by omega)))
  have hmap : ∀ k < (order done u).length - u.inCols.size, ∀ p,
      (ofBlock words done u).row ((order done u).getD (u.inCols.size + k) 0) = some p →
      (p.1.map fun x => if x = one then S.pin.val else pos x) = p.1.map pos ∧
        (p.2.map fun x => if x = one then S.pin.val else pos x) = p.2.map pos := by
    intro k hk p hp
    have hm := h.mem k hk p hp
    refine ⟨List.map_congr_left fun x hx => ?_, List.map_congr_left fun x hx => ?_⟩
    · rw [ite_eq_right (hne x (hm x (List.mem_append_left _ hx)))]
    · rw [ite_eq_right (hne x (hm x (List.mem_append_right _ hx)))]
  refine ⟨fun ℓ hℓ => hf.lt _ (hmem ℓ hℓ), fun k hk p hp r j hr => ?_, fun k hk p hp r j hr => ?_,
    fun k hk hp r j hr => ?_, fun k hk hp r j hr => ?_, hf.pinA, hf.pinB⟩
  · rcases hcases k with ⟨hr', -⟩ | ⟨p', hr', hb⟩ | ⟨-, hb⟩
    · rw [hr'] at hp; cases hp
    · rw [hr', Option.some.injEq] at hp
      subst hp
      rw [hf.rowA _ (hmem (u.inCols.size + k) (by have hk' : k < (order done u).length - u.inCols.size := hk; omega)) (hnotin k hk) p' (hb one) r j hr, (hmap k hk p' hr').1]
    · exact absurd (hb one) (hnone k hk)
  · rcases hcases k with ⟨hr', -⟩ | ⟨p', hr', hb⟩ | ⟨-, hb⟩
    · rw [hr'] at hp; cases hp
    · rw [hr', Option.some.injEq] at hp
      subst hp
      rw [hf.rowB _ (hmem (u.inCols.size + k) (by have hk' : k < (order done u).length - u.inCols.size := hk; omega)) (hnotin k hk) p' (hb one) r j hr, (hmap k hk p' hr').2]
    · exact absurd (hb one) (hnone k hk)
  · rcases hcases k with ⟨-, hb⟩ | ⟨p', hr', -⟩ | ⟨hr', -⟩
    · rw [hf.rowA _ (hmem (u.inCols.size + k) (by have hk' : k < (order done u).length - u.inCols.size := hk; omega)) (hnotin k hk) _ (hb one) r j hr]
      simp only [List.map_cons, List.map_nil, ite_true]
      exact count_pin j
    · rw [hr'] at hp; cases hp
    · rw [hr'] at hp; cases hp
  · rcases hcases k with ⟨-, hb⟩ | ⟨p', hr', -⟩ | ⟨hr', -⟩
    · rw [hf.rowB _ (hmem (u.inCols.size + k) (by have hk' : k < (order done u).length - u.inCols.size := hk; omega)) (hnotin k hk) _ (hb one) r j hr]
      simp only [List.map_cons, List.map_nil, ite_true]
      exact count_pin j
    · rw [hr'] at hp; cases hp
    · rw [hr'] at hp; cases hp

/-- **A table class's placement at one slot**, from 1e's facts there: the unit's composed rows, placed at
`composePlace`. -/
theorem BlockFacts.placement {types : String → Option CircuitType} {ls : Array Flock.Layout}
    {info : String → Option TableInfo} {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D}
    (hd : deriveChecked types ls info words unit = .ok done) {S : Model.Statement} {one : ℕ} {pos : ℕ → ℕ}
    (hone : (done.getD unit default).rows.size ≤ one) (hf : BlockFacts S words done (done.getD unit default) one pos) :
    Placement S (Rows.compose (ofBlock words done (done.getD unit default)) (ofBlock_ordered hd))
      (composePlace S _ (ofBlock_ordered hd) pos (hf.realizes hd hone).lt) :=
  (hf.realizes hd hone).placement (ofBlock_ordered hd)

end Compose

open Compose

/-! ## A flat statement's unit slots realize the rows of its unit net -/

/-- Shifting a list below `W` by `B·W`, counted at `j`: the count at `j mod W` when `j` is in block `B`, else 0. -/
theorem count_map_add_ite {W B : ℕ} (hW : 0 < W) (l : List ℕ) (hl : ∀ x ∈ l, x < W) (j : ℕ) :
    (((l.map (B * W + ·)).count j : ℕ) : ZMod 2) = if j / W = B then ((l.count (j % W) : ℕ) : ZMod 2) else 0 := by
  by_cases hq : j / W = B
  · rw [ite_eq_left hq]
    have hj : j = B * W + j % W := by
      have := Nat.div_add_mod j W
      rw [hq, Nat.mul_comm] at this
      exact this.symm
    have : (l.map (B * W + ·)).count j = l.count (j % W) := by
      conv_lhs => rw [hj]
      exact List.count_map_of_injective _ _ (fun a b hab => by simpa using hab) _
    rw [this]
  · rw [ite_eq_right hq, List.count_eq_zero.2]
    · rfl
    intro hm
    obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hm
    apply hq
    rw [show B * W + x = x + W * B by ring, Nat.add_mul_div_left _ _ hW, Nat.div_eq_of_lt (hl x hx), zero_add]

namespace Layout

variable {st : Stmt} (L : Layout st)
include L

/-- The unit net's computed row `c`'s columns are below its slot's size. -/
theorem unit_row_lt (hU : Built st.c.unit) {c : ℕ} (hlo : st.c.unit.inWords * WORD ≤ c) (hhi : c < st.c.unit.constPos)
    (hu : st.c.unit.unitLog ≤ 32) :
    (∀ x ∈ (st.c.unit.a.row c).toList, x < 2 ^ st.c.unit.unitLog) ∧
      ∀ x ∈ (st.c.unit.b.row c).toList, x < 2 ^ st.c.unit.unitLog := by
  have hn : (Rows.ofNet hU).nIn = st.c.unit.inWords * WORD := Nat.min_eq_left L.unit_in
  have hnc : (Rows.ofNet hU).nIn + (Rows.ofNet hU).nComp = st.c.unit.constPos := by
    show min _ _ + (_ - min _ _) = _
    have := Nat.min_le_right (st.c.unit.inWords * WORD) st.c.unit.constPos
    omega
  have hw : (Rows.ofNet hU).w ≤ 2 ^ st.c.unit.unitLog := by
    show _ + _ + 1 ≤ _
    rw [hnc]
    exact L.unit_const_lt
  let i : Fin (Rows.ofNet hU).nComp := ⟨c - (Rows.ofNet hU).nIn, by omega⟩
  have hi : (Rows.ofNet hU).nIn + i.val = c := by
    show (Rows.ofNet hU).nIn + (c - (Rows.ofNet hU).nIn) = c
    omega
  refine ⟨fun x hx => ?_, fun x hx => ?_⟩
  · rw [← hi, ← Rows.ofNet_a hU hu i] at hx
    obtain ⟨y, -, rfl⟩ := List.mem_map.1 hx
    exact lt_of_lt_of_le y.isLt hw
  · rw [← hi, ← Rows.ofNet_b hU hu i] at hx
    obtain ⟨y, -, rfl⟩ := List.mem_map.1 hx
    exact lt_of_lt_of_le y.isLt hw

theorem slot_eq {s₀ : ℕ} : st.c.unitRange.slot s₀ = (st.c.unitRange.pos0 + s₀) * 2 ^ st.c.unit.unitLog := by
  unfold Range.slot; rw [L.unit_slotLog]

/-- **The block's `A` at a unit slot's computed row**: the unit net's row, shifted to the slot. -/
theorem placedA_comp (hU : Built st.c.unit) {s₀ c : ℕ} (hs : s₀ < st.c.unitRange.count) (hlo : st.c.unit.inWords * WORD ≤ c)
    (hhi : c < st.c.unit.constPos) (r j : Fin (2 ^ st.c.kLog)) (hr : r.val = st.c.unitRange.slot s₀ + c) :
    placedA st r j = ((((st.c.unit.a.row c).toList.map (st.c.unitRange.slot s₀ + ·)).count j.val : ℕ) : ZMod 2) := by
  show rangesEntry st (slotEntryA st) r.val j.val + deltaEntry st.da r.val j.val = _
  rw [hr, L.rangesEntry_unit hs (lt_trans hhi L.unit_const_lt), L.delta_unit_comp hs hlo hhi, add_zero, L.slot_eq,
    count_map_add_ite (Nat.two_pow_pos _) _ (L.unit_row_lt hU hlo hhi (L.unitLog_le hs)).1]
  exact if_congr Iff.rfl (sparseEntry_built_a hU _ _) rfl

/-- **The block's `B` at a unit slot's computed row**: the unit net's row, shifted to the slot. -/
theorem placedB_comp (hU : Built st.c.unit) {s₀ c : ℕ} (hs : s₀ < st.c.unitRange.count) (hlo : st.c.unit.inWords * WORD ≤ c)
    (hhi : c < st.c.unit.constPos) (r j : Fin (2 ^ st.c.kLog)) (hr : r.val = st.c.unitRange.slot s₀ + c) :
    placedB st r j = ((((st.c.unit.b.row c).toList.map (st.c.unitRange.slot s₀ + ·)).count j.val : ℕ) : ZMod 2) := by
  have hlk : (st.c.nets.getD st.c.unitNet default).2.lookup = none := L.unit_lookup
  show rangesEntry st (slotEntryB st) r.val j.val + deltaEntry st.db r.val j.val = _
  rw [L.delta_b, hr, L.rangesEntry_unit hs (lt_trans hhi L.unit_const_lt), L.delta_unit_comp hs hlo hhi, add_zero,
    L.slot_eq, count_map_add_ite (Nat.two_pow_pos _) _ (L.unit_row_lt hU hlo hhi (L.unitLog_le hs)).2]
  refine if_congr Iff.rfl ?_ rfl
  show sparseEntry _ _ _ + lookupEntry (st.c.nets.getD st.c.unitNet default).2.lookup _ _ = _
  rw [hlk]
  show sparseEntry st.c.unit.b _ _ + 0 = _
  rw [add_zero, sparseEntry_built_b hU]

/-- **The block's `A` at a unit slot's constant reads the pin.** -/
theorem placedA_const {s₀ : ℕ} (hs : s₀ < st.c.unitRange.count) (r j : Fin (2 ^ st.c.kLog))
    (hr : r.val = st.c.unitRange.slot s₀ + st.c.unit.constPos) :
    placedA st r j = if j.val = st.pin then 1 else 0 := by
  have hW : 0 < 2 ^ st.c.unit.unitLog := Nat.two_pow_pos _
  show rangesEntry st (slotEntryA st) r.val j.val + deltaEntry st.da r.val j.val = _
  rw [hr, L.rangesEntry_unit hs L.unit_const_lt, L.delta_unit_const hs,
    ite_slot_eq (v := slotEntryA st (.net st.c.unitNet) st.c.unit.constPos) hW L.unit_const_lt
      (L.unit_const (L.unitLog_le hs)).2.1, ← L.slot_eq]
  exact const_row_sum _ _ _

/-- **The block's `B` at a unit slot's constant reads the pin.** -/
theorem placedB_const {s₀ : ℕ} (hs : s₀ < st.c.unitRange.count) (r j : Fin (2 ^ st.c.kLog))
    (hr : r.val = st.c.unitRange.slot s₀ + st.c.unit.constPos) :
    placedB st r j = if j.val = st.pin then 1 else 0 := by
  have hW : 0 < 2 ^ st.c.unit.unitLog := Nat.two_pow_pos _
  show rangesEntry st (slotEntryB st) r.val j.val + deltaEntry st.db r.val j.val = _
  rw [L.delta_b, hr, L.rangesEntry_unit hs L.unit_const_lt, L.delta_unit_const hs,
    ite_slot_eq (v := slotEntryB st (.net st.c.unitNet) st.c.unit.constPos) hW L.unit_const_lt
      (L.unit_const (L.unitLog_le hs)).2.2, ← L.slot_eq]
  exact const_row_sum _ _ _

/-- **The pin's row reads the pin**, in `A` and in `B`. -/
theorem placed_pin (j : Fin (2 ^ st.c.kLog)) :
    placedA st ⟨st.pin, L.pin_lt⟩ j = (if j.val = st.pin then 1 else 0) ∧
      placedB st ⟨st.pin, L.pin_lt⟩ j = if j.val = st.pin then 1 else 0 := by
  obtain ⟨rp, hrp, ip, hk, hc, hpin, hcr⟩ := L.pin
  have hoffW : (st.c.nets.getD ip default).2.constPos < 2 ^ rp.slotLog := by
    rw [L.slotLog rp hrp ip hk]; exact (L.fit ip).1
  have hp : st.pin = rp.slot 0 + (st.c.nets.getD ip default).2.constPos := by rw [hpin, start_eq_slot]
  constructor
  · show rangesEntry st (slotEntryA st) st.pin j.val + deltaEntry st.da st.pin j.val = _
    rw [L.delta_pin, add_zero]
    conv_lhs => rw [hp]
    rw [rangesEntry_of_covers L.disj hrp (L.const_covers hrp hk hc) _ _, (slot_div 0 hoffW).1, (slot_div 0 hoffW).2, hk,
      ite_slot_eq (v := slotEntryA st (.net ip) (st.c.nets.getD ip default).2.constPos) (Nat.two_pow_pos _) hoffW hcr.2.1,
      hp]
    rfl
  · show rangesEntry st (slotEntryB st) st.pin j.val + deltaEntry st.db st.pin j.val = _
    rw [L.delta_b, L.delta_pin, add_zero]
    conv_lhs => rw [hp]
    rw [rangesEntry_of_covers L.disj hrp (L.const_covers hrp hk hc) _ _, (slot_div 0 hoffW).1, (slot_div 0 hoffW).2, hk,
      ite_slot_eq (v := slotEntryB st (.net ip) (st.c.nets.getD ip default).2.constPos) (Nat.two_pow_pos _) hoffW hcr.2.2,
      hp]
    rfl

/-- **A flat accepted statement realizes, at unit slot `s₀`, every `Logical` whose rows are its unit net's.** The
positions are `slot s₀ + c`, and the statement is the model of the matrices the verifier folds (`Layout.model`). -/
theorem realizes (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {s₀ : ℕ} (hs : s₀ < st.c.unitRange.count)
    (hU : Built st.c.unit) {Lg : Logical} (hm : NetRows st.c.unit Lg) :
    Realizes (L.model regs) Lg (fun c => st.c.unitRange.slot s₀ + c) := by
  have hpinv : ∀ j : Fin (2 ^ (L.model regs).kLog), (j = (L.model regs).pin) ↔ j.val = st.pin :=
    fun j => ⟨fun h => by rw [h]; rfl, fun h => Fin.ext h⟩
  refine ⟨fun ℓ hℓ => ?_, fun k hk p hp r j hr => ?_, fun k hk p hp r j hr => ?_, fun k hk hp r j hr => ?_,
    fun k hk hp r j hr => ?_, fun j => ?_, fun j => ?_⟩
  · have hc := hm.le (Lg.order.getD ℓ 0) (by rw [List.getD_eq_getElem _ _ hℓ]; exact List.getElem_mem hℓ)
    exact L.unit_slot_lt hs (lt_of_le_of_lt hc L.unit_const_lt)
  · obtain ⟨hlo, hhi, rfl⟩ := hm.rows k hk p hp
    exact L.placedA_comp hU hs hlo hhi r j hr
  · obtain ⟨hlo, hhi, rfl⟩ := hm.rows k hk p hp
    exact L.placedB_comp hU hs hlo hhi r j hr
  · rw [hm.one k hk hp] at hr
    rw [if_congr (hpinv j) rfl rfl]
    exact L.placedA_const hs r j hr
  · rw [hm.one k hk hp] at hr
    rw [if_congr (hpinv j) rfl rfl]
    exact L.placedB_const hs r j hr
  · rw [if_congr (hpinv j) rfl rfl]
    exact (L.placed_pin j).1
  · rw [if_congr (hpinv j) rfl rfl]
    exact (L.placed_pin j).2

end Layout

end FlockSoundness
