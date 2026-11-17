module

public import Flock.HmRow
public import Proofs.Flock.Soundness.Discharge.Hidden.Facts
public import Proofs.Flock.Soundness.ExecPlacement
public import Proofs.Flock.Soundness.ParseStmt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Proofs.Flock.Soundness.ExecCircuit

/-! An untyped hidden-output session commits `HmOut.parseCommit`'s circuit, and its statement circuit (`HmOut.parseStmt`) is that circuit unrolled. -/

@[expose] public section

namespace FlockSoundness

open Flock

variable {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c cs : Circuit}

theorem parse_tmpl_none (ht : tags.typed = false) (h : HmOut.parse tags cf tables = .ok c) : c.tmpl = none :=
  (parse_factsPre_nets (Discharge.Hidden.parse_untyped ht h).1).1.tmpl

theorem parseCommit_flat (ht : tags.isTmpl = false) :
    HmOut.parseCommit tags cf tables = HmOut.parse tags cf tables := by
  simp [HmOut.parseCommit, ht]

theorem parseCommit_tmpl (ht : tags.isTmpl = true) :
    HmOut.parseCommit tags cf tables = HmOut.parseTmpl tags cf tables := by
  simp [HmOut.parseCommit, ht]

theorem parseStmt_of_commit (hty : tags.typed = false) (h : HmOut.parseCommit tags cf tables = .ok cs) :
    HmOut.parseStmt tags cf tables = .ok (HmOut.unrolled cs) := by
  cases ht : tags.isTmpl
  · rw [parseCommit_flat ht] at h
    rw [unrolled_flat (parse_tmpl_none hty h)]
    exact parseStmt_of_parse ht h
  · rw [parseCommit_tmpl ht] at h
    exact parseStmt_of_tmpl ht h

theorem commit_of_parseStmt (hty : tags.typed = false) (h : HmOut.parseStmt tags cf tables = .ok c) :
    ∃ cs, HmOut.parseCommit tags cf tables = .ok cs ∧ c = HmOut.unrolled cs := by
  rcases parseStmt_cases h with ⟨ht, hp⟩ | ⟨ht, ct, hp, rfl⟩
  · exact ⟨c, by rw [parseCommit_flat ht]; exact hp, (unrolled_flat (parse_tmpl_none hty hp)).symm⟩
  · exact ⟨ct, by rw [parseCommit_tmpl ht]; exact hp, rfl⟩

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
