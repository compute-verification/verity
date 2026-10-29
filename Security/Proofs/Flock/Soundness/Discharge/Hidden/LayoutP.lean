import Proofs.Flock.Soundness.Discharge.Hidden.PortsP
import Proofs.Flock.Soundness.Discharge.Hidden.Layout
import Proofs.Flock.Soundness.Discharge.Hidden.Commit

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

/-- The unit's input groups fit its input words, of any `Layout`: its logical net's (`built_spec`), which has the unit's
input groups and words (`UnitAt`). -/
theorem unit_groups_fit (L : Layout st) : ∀ g ∈ st.c.unit.inGroups.toList,
    g.bits.foldl (· + ·) 0 ≤ g.words * WORD ∧ g.col + g.words ≤ st.c.unit.inWords := by
  have h := (built_spec L.unit_built).2.2.2.2.1
  rwa [L.unit_at.groups, L.unit_at.in_eq] at h

/-- **An untyped hidden-output statement accepted at public inputs has the layout the placement reads**, on the
committed circuit: its flat twin's is `layout_of`'s for a flat file, `layout_of_tmpl`'s for a templated one
(`hidden_layoutC`), where it pins no output (`hidden_delta_unpinned`). -/
theorem hidden_layoutP (ht : tags.typed = false) (s : SpecP tags cf pf coins tables draw own pi st)
    (hpin : HmRow.pinnedCount st.c.metaJ = 0) : Layout st := by
  obtain ⟨L, hty, C, hout⟩ :
      Layout { st with da := (HmRow.delta st.c st.pin).1, db := (HmRow.delta st.c st.pin).2 } ∧
        st.c.typed = none ∧ CheckFacts st.c ∧ ∀ x ∈ outPairs st.c, InBit st.c x.1 := by
    cases hi : tags.isTmpl
    · have hP := s.parse
      rw [parseCommit_flat hi] at hP
      obtain ⟨hp, hc⟩ := parse_untyped ht hP
      have P := parse_facts hp
      obtain ⟨ls, hcheck⟩ := P.check
      exact ⟨layout_of P s.pin rfl rfl s.m s.kLog, P.typed, check_facts P.typed hcheck, outPairs_inBit hp hc⟩
    · have hT := s.parse
      rw [parseCommit_tmpl hi] at hT
      obtain ⟨tu, c₀, P, hck, he, -, -⟩ := TmplValue.parseTmpl_facts hT
      have hok := parseTmpl_nets_ok hT
      obtain ⟨ls, hcheck⟩ := P.check
      have hty : st.c.typed = none := by rw [he]; exact P.typed
      have C : CheckFacts st.c := check_facts (tags := tags) (ls := ls) hty (by rw [he]; exact hcheck)
      have hun : st.c.unitNet < st.c.nets.size := by rw [he]; exact P.unitNet
      have hsh : st.c.shaNet < st.c.nets.size := by rw [he]; exact P.shaNet
      have hhm : st.c.hmNet < st.c.nets.size := by rw [he]; exact P.hmNet
      have hc : HmOut.checkRows st.c = .ok () := by rw [he]; exact hck
      exact ⟨layout_of_tmpl hT s.pin rfl rfl s.m s.kLog, hty, C,
        outPairs_inBit_of (fun n => (hok n).1.2.2) hun hsh hhm C hc⟩
  have hd := hidden_delta_unpinned hpin st.pin
  obtain ⟨rest, h1, h2⟩ := L.delta
  have h1' : (HmRow.delta st.c st.pin).1.toList = constPairs st.c st.pin ++ rest := h1
  have hkc : ∀ p ∈ constPairs st.c st.pin, keepCol st.c pi p.1 = true := by
    rintro ⟨a, b⟩ hab
    cases he : pi.ports.isEmpty
    · by_contra hk
      have hq : HmIn.publicCol st.c pi a = true := by simpa [keepCol, he] using hk
      obtain ⟨k, hk', jj, -, t, -, hq1, hlt, -⟩ := publicCol_at s.ports he C (by have h := unit_groups_fit L; exact h) hq
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
    unit_at := L.unit_at
    unit_lookup := L.unit_lookup
    unit_in := L.unit_in
    unit_const := L.unit_const
    pin := L.pin
    delta_b := by
      rw [s.db, s.da]
      apply Array.ext'
      rw [(inDelta_toList _ _ _).1, (inDelta_toList _ _ _).2, outDelta_snd hty]
    delta := by
      refine ⟨(rest ++ outPairs st.c).filter (fun p => keepCol st.c pi p.1), ?_, fun x hx => ?_⟩
      · rw [s.da, (inDelta_toList _ _ _).1, hd, Array.toList_append, List.toList_toArray, h1', List.append_assoc,
          List.filter_append, List.filter_eq_self.2 hkc]
      · rcases List.mem_append.1 (List.mem_filter.1 hx).1 with hx | hx
        · exact h2 x hx
        · exact hout x hx
    kLog := L.kLog
    kLog6 := L.kLog6
    m := L.m }

end FlockSoundness.Discharge.Hidden
