import Proofs.Flock.Soundness.Discharge.Hidden.Rows
import Proofs.Flock.Soundness.Discharge.Hidden.Delta
import Proofs.Flock.Soundness.Discharge.Hm.Accept

/-!
# A hidden-output statement's layout (`HiddenLayout`)

An untyped statement `Stmt.setupHidden` accepts has `layout_of`'s layout on the statement with `HmRow.delta`'s Δ, and,
where it pins no output, `HmOut.delta` only appends `outPairs` to both of its sides (`hidden_delta_unpinned`). Every first column of `outPairs` is a
message bit of an output row's `sha512x3` slot or, past the row's compressions (an exact-length row's last block), of
its `hm96` slot's `pad` (`deltaMsgCol_rowBit`). Either is an input bit of a net slot (`deltaIn_inBit`), as
`Layout.delta` asks of every pair past the constant pairs.
-/

open Flock FlockLevel3
open Lean (Json)

namespace FlockSoundness.Discharge.Hidden

/-- **Row `p`'s message bit `t` of byte `byte`** is a message bit of the row's `sha512x3` slot for the byte's
compression, or, past the row's compressions, a `pad` bit of its `hm96` slot. -/
theorem deltaMsgCol_rowBit {c : Circuit} {g p byte t : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (ht : t < 8) :
    ShaRowBit c (deltaMsgCol c g p byte t) ∨ HmRowBit c (deltaMsgCol c g p byte t) := by
  have ho := Typed.byteAt_snd (c.ports.getD p default) byte
  by_cases hk : (HmRow.byteAt (c.ports.getD p default) byte).1 < ((HmRow.rowSlots c.ports).2[p]!).size
  · refine Or.inl ⟨g, hg, p, by rw [rowSlots_spec]; exact hp, (HmRow.byteAt (c.ports.getD p default) byte).1, hk, 1,
      by omega, 64 * (byte % 128 / 8) + 56 - 8 * (byte % 128 % 8) + t, by omega, ?_⟩
    simp only [deltaMsgCol, hk, ↓reduceIte, ho]
    omega
  · refine Or.inr ⟨g, hg, p, by rw [rowSlots_spec]; exact hp, 1, by omega,
      64 * (byte % 128 / 8) + 56 - 8 * (byte % 128 % 8) + t, by omega, ?_⟩
    simp only [deltaMsgCol, hk, ↓reduceIte, ho]
    omega

/-- **Every first column of `outPairs` is an input bit of a net slot**, for a circuit `HmRow.parse` accepted and
`HmOut.checkRows` passed. -/
theorem outPairs_inBit {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (hp : HmRow.parse tags cf tables = .ok c) (hc : HmOut.checkRows c = .ok ()) :
    ∀ x ∈ outPairs c, InBit c x.1 := by
  have P := parse_facts hp
  obtain ⟨ls, hcheck⟩ := P.check
  have C := check_facts P.typed hcheck
  intro x hx
  simp only [outPairs, List.mem_flatMap, List.mem_range, List.mem_cons, List.not_mem_nil, or_false] at hx
  obtain ⟨g, hg, j, hj, k, -, t, -, hx⟩ := hx
  have hlt := (checkRows_out hc hj).1
  have hs := deltaMsgCol_rowBit hg hlt (byte := 2 * k + t / 8) (Nat.mod_lt t (by norm_num))
  rcases hx with rfl | rfl <;> exact deltaIn_inBit P C (hs.imp id Or.inl)

/-- **An untyped hidden statement of one output row per output port pins no output** (`unpinned_of_rep`). -/
theorem setupHidden_unpinned {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Json} {program : Option ByteArray} {st : Stmt} (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st)
    (hr : st.c.outRowPorts = Array.replicate st.c.outPorts.size 1) : HmRow.pinnedCount st.c.metaJ = 0 :=
  have hpc := parse_untyped ht (setupHidden_spec' h).parse
  unpinned_of_rep hpc.1 hpc.2 hr

theorem hidden_layout' : HiddenLayout := by
  intro tags cf pf coins tables draw partition program st ht h hpin
  have s := setupHidden_spec' h
  obtain ⟨hp, hc⟩ := parse_untyped ht s.parse
  have P := parse_facts hp
  have L : Layout { st with da := (HmRow.delta st.c st.pin).1, db := (HmRow.delta st.c st.pin).2 } :=
    layout_of P s.pin rfl rfl s.m s.kLog
  have hd := hidden_delta_unpinned hpin st.pin
  obtain ⟨rest, h1, h2⟩ := L.delta
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
    delta_b := by rw [s.db, s.da, hd, delta_snd _ _ P.typed]
    delta := by
      refine ⟨rest ++ outPairs st.c, ?_, fun x hx => ?_⟩
      · rw [s.da, hd, Array.toList_append, List.toList_toArray]
        exact (congrArg (· ++ outPairs st.c) h1).trans (List.append_assoc _ _ _)
      · rcases List.mem_append.1 hx with hx | hx
        · exact h2 x hx
        · exact outPairs_inBit hp hc x hx
    kLog := L.kLog
    kLog6 := L.kLog6
    m := L.m }

end FlockSoundness.Discharge.Hidden
