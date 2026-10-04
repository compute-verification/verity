import FlockSoundness.Discharge.Aliased.Copies
import FlockSoundness.Discharge.ScopeG.Places
import FlockSoundness.Discharge.ScopeG.Place.Pos

set_option Elab.async false

/-!
# P1 bookkeeping: the VU's order, decoded into slots

What the placement of `vuLogical` (`Place.lean`) reads of `vuOrder` and `vuRowList`, apart from the block's entries:

* `vuOrder_getD_in`, `vuOrder_comp_notin`: the order begins with the VU's input columns (`kahn` appends to `done`), and
  a computed position holds no input column (`(vuOrder c).Nodup`, from `scopeOkGT`);
* `vuOrder_decode`: a column of the order is net `k`'s slot `i`, column `x ≤ constPos` at VU 0, its row is `slotRow`'s,
  `vuPos` moves it to `slotCol k g i + x`; a unit input column the VU computes is a wired one (`Aliased.srcKey`'s
  `.wire` is `wireSrc`'s);
* `slotRow_vuPos`: a row's reads, moved by `vuPos`, are `slotRowG`'s at VU `g`;
* `vuPos_in`: the input columns at VU `g` are `vuInPos`'s.

`vuPos` needs the VU's slot columns distinct (a range's slots are disjoint, `x ≤ constPos < 2 ^ slotLog`, and two nets'
ranges apart: `CheckFacts.disj`, `CheckFacts.one_range`, `Layout.fit`), hence `Layout` and `UnitFacts`.
-/

namespace FlockSoundness.Discharge.ScopeG

open Flock FlockLevel3

/-- **The VU's order begins with its input columns.** -/
theorem vuOrder_getD_in (c : Circuit) {ℓ : ℕ} (hℓ : ℓ < (vuInCols c).length) :
    (vuOrder c).getD ℓ 0 = (vuInCols c).getD ℓ 0 := sorry

/-- **A computed position of the VU's order holds no input column.** -/
theorem vuOrder_comp_notin {c : Circuit} (hs : scopeOkGT c = true) {k : ℕ}
    (hk : k < (vuOrder c).length - (vuInCols c).length) :
    (vuOrder c).getD ((vuInCols c).length + k) 0 ∉ vuInCols c := sorry

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags st) (hs : scopeOkGT st.c = true)

include L F hs

/-- **A column of the VU's order, decoded**: net `k`'s slot `i`, column `x` at VU 0, with `slotRow`'s row, at
`slotCol k g i + x` at VU `g`; a unit input the VU computes is wired. -/
theorem vuOrder_decode {p : ℕ} (hp : p ∈ vuOrder st.c) (g : ℕ) :
    ∃ k ∈ vuNets st.c, ∃ i < st.c.perVu.getD k 0, ∃ x ≤ (st.c.nets.getD k default).2.constPos,
      p = st.c.slotCol k 0 i + x ∧ vuRow st.c p = slotRow st.c k i x ∧ vuPos st.c g p = st.c.slotCol k g i + x ∧
      (p ∉ vuInCols st.c → k = st.c.unitNet → x < netIn (st.c.nets.getD k default).2 →
        (wireSrcG st.c g k i x).isSome) := sorry

/-- **A row's reads at VU `g`**: `slotRow`'s reads, moved by `vuPos`, are `slotRowG`'s. -/
theorem slotRow_vuPos {k i x : ℕ} (hk : k ∈ vuNets st.c) (hi : i < st.c.perVu.getD k 0)
    (hx : x ≤ (st.c.nets.getD k default).2.constPos) {pr : List ℕ × List ℕ} (hpr : slotRow st.c k i x = some pr)
    (hmem : ∀ y ∈ pr.1 ++ pr.2, y ∈ vuOrder st.c) (g : ℕ) :
    slotRowG st.c g k i x = some (pr.1.map (vuPos st.c g), pr.2.map (vuPos st.c g)) := sorry

/-- **The VU's input columns at VU `g` are `vuInPos`'s.** -/
theorem vuPos_in {ℓ : ℕ} (hℓ : ℓ < (vuInCols st.c).length) (g : ℕ) :
    vuPos st.c g ((vuInCols st.c).getD ℓ 0) = vuInPos st.c g ℓ := sorry

end FlockSoundness.Discharge.ScopeG
