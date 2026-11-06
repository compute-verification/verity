import Proofs.Flock.Soundness.Discharge.Hidden.Facts
import Proofs.Flock.Soundness.ParseStmt

/-!
# The committed circuit and the statement circuit

An untyped hidden-output session commits the circuit `HmOut.parseCommit` reads (a templated file's carries its
template), and its statement circuit (`HmOut.parseStmt`) is that circuit unrolled (`HmOut.unrolled`), which is the
committed circuit itself for a flat file. A session-side consumer takes `HmOut.parseCommit … = .ok st.c` where it took
`HmOut.parse … = .ok st.c`; a statement-side one reads `HmOut.unrolled st.c`.
-/

namespace FlockSoundness

open Flock

variable {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c cs : Circuit}

/-- An untyped `HmOut.parse` circuit carries no template. -/
theorem parse_tmpl_none (ht : tags.typed = false) (h : HmOut.parse tags cf tables = .ok c) : c.tmpl = none :=
  (parse_factsPre_nets (Discharge.Hidden.parse_untyped ht h).1).1.tmpl

theorem parseCommit_flat (ht : tags.isTmpl = false) :
    HmOut.parseCommit tags cf tables = HmOut.parse tags cf tables := by
  simp [HmOut.parseCommit, ht]

theorem parseCommit_tmpl (ht : tags.isTmpl = true) :
    HmOut.parseCommit tags cf tables = HmOut.parseTmpl tags cf tables := by
  simp [HmOut.parseCommit, ht]

/-- **The statement circuit is the committed circuit unrolled**, for untyped tags. -/
theorem parseStmt_of_commit (hty : tags.typed = false) (h : HmOut.parseCommit tags cf tables = .ok cs) :
    HmOut.parseStmt tags cf tables = .ok (HmOut.unrolled cs) := by
  cases ht : tags.isTmpl
  · rw [parseCommit_flat ht] at h
    rw [unrolled_flat (parse_tmpl_none hty h)]
    exact parseStmt_of_parse ht h
  · rw [parseCommit_tmpl ht] at h
    exact parseStmt_of_tmpl ht h

/-- **The committed circuit under a statement circuit**, for untyped tags: the one `HmOut.parseCommit` reads, which
unrolls to it. -/
theorem commit_of_parseStmt (hty : tags.typed = false) (h : HmOut.parseStmt tags cf tables = .ok c) :
    ∃ cs, HmOut.parseCommit tags cf tables = .ok cs ∧ c = HmOut.unrolled cs := by
  rcases parseStmt_cases h with ⟨ht, hp⟩ | ⟨ht, ct, hp, rfl⟩
  · exact ⟨c, by rw [parseCommit_flat ht]; exact hp, (unrolled_flat (parse_tmpl_none hty hp)).symm⟩
  · exact ⟨ct, by rw [parseCommit_tmpl ht]; exact hp, rfl⟩

/-- **The statement circuit's unit is the committed circuit's logical net**, and it is `Built`. -/
theorem unrolled_unit_commit (hty : tags.typed = false) (h : HmOut.parseCommit tags cf tables = .ok cs) :
    (HmOut.unrolled cs).unit = lunit cs ∧ (HmOut.unrolled cs).tmpl = none := by
  cases ht : tags.isTmpl
  · rw [parseCommit_flat ht] at h
    have hf := parse_tmpl_none hty h
    rw [unrolled_flat hf, lunit_flat hf]
    exact ⟨rfl, hf⟩
  · rw [parseCommit_tmpl ht] at h
    exact ⟨unrolled_unit h, unrolled_tmpl h⟩

end FlockSoundness
