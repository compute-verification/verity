import Proofs.Flock.Soundness.Discharge.Hidden.PortsP
import Proofs.Flock.Soundness.Discharge.Hidden.Layout

/-!
# A hidden-output statement's layout at public inputs (`hidden_layoutP`)

`hidden_layout'` at `SpecP`: Δ is `HmOut.delta`'s without the pairs into a public leaf's bits (`inDelta_toList`). No
constant pair is into one, since a public leaf's bit is a unit slot's input bit (`publicCol_at`), below the unit's
constant (`Layout.constPairs_not_below`), so the constant pairs come first as `Layout.delta` asks, and every other pair
is one of `HmOut.delta`'s, an input bit of a net slot.
-/

open Flock FlockLevel3
open Lean (Json)

namespace FlockSoundness.Discharge.Hidden

/-- `HmOut.delta`'s two sides are one, on an untyped circuit. -/
theorem outDelta_snd {c : Circuit} (ht : c.typed = none) (pin : ℕ) :
    (HmOut.delta c pin).2 = (HmOut.delta c pin).1 := by
  rw [hidden_delta', delta_snd _ _ ht]

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw : Option Json} {own : Option ByteArray} {pi : HmIn.Ports} {st : Stmt}

/-- **An untyped hidden-output statement accepted at public inputs has the layout the placement reads.** -/
theorem hidden_layoutP (ht : tags.typed = false) (s : SpecP tags cf pf coins tables draw own pi st)
    (hpin : HmRow.pinnedCount st.c.metaJ = 0) : Layout st := by
  obtain ⟨hp, hc⟩ := parse_untyped ht s.parse
  have P := parse_facts hp
  obtain ⟨ls, hcheck⟩ := P.check
  have C := check_facts P.typed hcheck
  have L : Layout { st with da := (HmRow.delta st.c st.pin).1, db := (HmRow.delta st.c st.pin).2 } :=
    layout_of P s.pin rfl rfl s.m s.kLog
  have hd := hidden_delta_unpinned hpin st.pin
  obtain ⟨rest, h1, h2⟩ := L.delta
  have h1' : (HmRow.delta st.c st.pin).1.toList = constPairs st.c st.pin ++ rest := h1
  have hkc : ∀ p ∈ constPairs st.c st.pin, keepCol st.c pi p.1 = true := by
    rintro ⟨a, b⟩ hab
    cases he : pi.ports.isEmpty
    · by_contra hk
      have hq : HmIn.publicCol st.c pi a = true := by simpa [keepCol, he] using hk
      obtain ⟨k, hk', jj, -, t, -, hq1, hlt, -⟩ := publicCol_at s.ports he C L.unit_built hq
      rw [hq1] at hab
      exact L.constPairs_not_below L.unit_mem L.unit_kind hk' (lt_of_lt_of_le hlt L.unit_in) b hab
    · exact keepCol_noPorts he _
  exact {
    disj := L.disj
    bound := L.bound
    slotLog := L.slotLog
    fit := L.fit
    unit_mem := L.unit_mem
    unit_kind := L.unit_kind
    unit_count := L.unit_count
    unit_built := L.unit_built
    unit_lookup := L.unit_lookup
    unit_in := L.unit_in
    unit_const := L.unit_const
    pin := L.pin
    delta_b := by
      rw [s.db, s.da]
      apply Array.ext'
      rw [(inDelta_toList _ _ _).1, (inDelta_toList _ _ _).2, outDelta_snd P.typed]
    delta := by
      refine ⟨(rest ++ outPairs st.c).filter (fun p => keepCol st.c pi p.1), ?_, fun x hx => ?_⟩
      · rw [s.da, (inDelta_toList _ _ _).1, hd, Array.toList_append, List.toList_toArray, h1', List.append_assoc,
          List.filter_append, List.filter_eq_self.2 hkc]
      · rcases List.mem_append.1 (List.mem_filter.1 hx).1 with hx | hx
        · exact h2 x hx
        · exact outPairs_inBit hp hc x hx
    kLog := L.kLog
    kLog6 := L.kLog6
    m := L.m }

end FlockSoundness.Discharge.Hidden
