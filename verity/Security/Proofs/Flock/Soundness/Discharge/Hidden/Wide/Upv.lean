import Proofs.Flock.Soundness.Discharge.Hidden.Facts
import Proofs.Flock.Soundness.Discharge.Layout.OutShape

/-!
# A unit per template instance, from acceptance

`HmRow.check` refuses an untyped circuit of `units_per_vu = 0` (PROTOCOL.md §16.10, D8), the one check that does when a
tail returns the outputs (`check_upv_pos`), and `Stmt.setupHidden` parses its circuit through it (`setupHidden_upv_pos`).
-/

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

open FlockSoundness.Discharge.Layout (ite_throw_cond)

/-- **`check_upv_pos`**: an untyped circuit `HmRow.check` accepts has a unit per template instance. -/
theorem check_upv_pos {c : Circuit} {tags : Tags} {ls : Lean.Json} (ht : c.typed = none)
    (h : HmRow.check c tags ls = .ok ()) : 0 < c.upv := by
  unfold HmRow.check at h
  obtain ⟨spans, -, h⟩ := except_bind_ok h
  obtain ⟨u, -, h⟩ := except_bind_ok h
  obtain ⟨-, h⟩ := ite_throw_cond h
  obtain ⟨-, h⟩ := ite_throw_cond h
  obtain ⟨-, h⟩ := ite_throw_cond h
  simp only [ht] at h
  obtain ⟨-, h⟩ := ite_throw_cond h
  obtain ⟨-, h⟩ := ite_throw_cond h
  replace h := ite_join h fun ha => (ite_throw_cond ha).2
  obtain ⟨u1, -, h⟩ := except_bind_ok h
  obtain ⟨u2, -, h⟩ := except_bind_ok h
  obtain ⟨wired, -, h⟩ := except_bind_ok h
  obtain ⟨wired2, -, h⟩ := except_bind_ok h
  obtain ⟨u3, -, h⟩ := except_bind_ok h
  obtain ⟨-, h⟩ := ite_throw_cond h
  obtain ⟨seen, -, h⟩ := except_bind_ok h
  obtain ⟨-, h⟩ := ite_throw_cond h
  obtain ⟨-, h⟩ := ite_throw_cond h
  obtain ⟨hupv, -⟩ := ite_throw_cond h
  simp only [beq_iff_eq] at hupv
  exact Nat.pos_of_ne_zero hupv

/-- **`setupHidden_upv_pos`**: a statement `Stmt.setupHidden` accepts with untyped tags has a unit per template
instance. -/
theorem setupHidden_upv_pos {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt} (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) : 0 < st.c.upv := by
  obtain ⟨hp, -⟩ := parse_untyped ht (setupHidden_spec' h).parse
  obtain ⟨ls, hls⟩ := (parse_facts hp).check
  exact check_upv_pos (parse_facts hp).typed hls

end FlockSoundness.Discharge.Hidden
