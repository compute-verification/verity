import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Upv
import Proofs.Flock.Soundness.ExecSetup

/-!
# A template instance per block, from acceptance

`HmRow.check` refuses an untyped circuit of `vus_per_block = 0` (PROTOCOL.md §16.10, D9), the one check that does when a
tail returns the outputs (`check_g_pos`), and `Stmt.setupH` parses its circuit through it (`setupH_g_pos`).
-/

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

open FlockSoundness.Discharge.Layout (ite_throw_cond)

/-- **`check_g_pos`**: an untyped circuit `HmRow.check` accepts has a template instance per block. -/
theorem check_g_pos {c : Circuit} {tags : Tags} {ls : Lean.Json} (ht : c.typed = none)
    (h : HmRow.check c tags ls = .ok ()) : 0 < c.g := by
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
  obtain ⟨-, h⟩ := ite_throw_cond h
  obtain ⟨hg, -⟩ := ite_throw_cond h
  simp only [beq_iff_eq] at hg
  exact Nat.pos_of_ne_zero hg

/-- **`setupH_g_pos`**: a statement `Stmt.setupH` accepts with untyped tags has a template instance per block. -/
theorem setupH_g_pos {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt} (ht : tags.typed = false)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) : 0 < st.c.g := by
  have hp := (setupH_spec ht h).1
  obtain ⟨ls, hls⟩ := (parse_facts hp).check
  exact check_g_pos (parse_facts hp).typed hls

end FlockSoundness.Discharge.Hidden
