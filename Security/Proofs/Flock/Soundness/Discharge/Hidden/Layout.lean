import Proofs.Flock.Soundness.Discharge.Hidden.Rows
import Proofs.Flock.Soundness.Discharge.Hidden.Delta
import Proofs.Flock.Soundness.Discharge.Hm.Accept

/-!
# A hidden-output statement's layout (`HiddenLayout`)

An untyped statement `Stmt.setupHidden` accepts has `layout_of`'s layout on the statement with `HmRow.delta`'s Δ, and,
where it pins no output, `HmOut.delta` only appends `outPairs` to both of its sides (`hidden_delta_unpinned`). Every first column of `outPairs` is a
message bit of an output row's `sha512x3` slot or, past the row's compressions (an exact-length row's last block), of
its `hm96` slot's `pad` (`deltaMsgCol_rowBit`). Either is an input bit of a net slot (`deltaIn_inBit`), as
`Layout.delta` asks of every pair past the constant pairs. `Hidden.Commit` proves `HiddenLayout` (`hidden_layout'`) for
a flat or a templated session.
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

end FlockSoundness.Discharge.Hidden
