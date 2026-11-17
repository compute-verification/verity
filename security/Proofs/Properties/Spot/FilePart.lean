module

public import Spot.FilePart
public import Spot.Partition
public import Spot.Program

/-! Where `FilePart.check` passes it pins the public file's units, which v2's other reads of the file read too. -/

@[expose] public section

open Lean (Json)

namespace Properties.Spot.FilePart

theorem ok_bind {ε α β : Type} (a : α) (f : α → Except ε β) : (Except.ok a >>= f) = f a := rfl

theorem error_bind {ε α β : Type} (e : ε) (f : α → Except ε β) : (Except.error e >>= f) = Except.error e := rfl

theorem why_none {strata : Array (String × Array (Nat × Nat))} {indices : Array Nat}
    (h : _root_.Spot.FilePart.why strata indices = none) :
    ∃ ns, _root_.Spot.FilePart.sizes strata = some ns ∧ indices = Array.range (ns.foldl (· + ·) 0) := by
  unfold _root_.Spot.FilePart.why at h
  cases hs : _root_.Spot.FilePart.sizes strata with
  | none => simp [hs] at h
  | some ns =>
    refine ⟨ns, rfl, ?_⟩
    by_cases hi : indices = Array.range (ns.foldl (· + ·) 0)
    · exact hi
    · simp [hs, hi] at h

theorem check_indices {program : ByteArray} {query : Json} {pubB : ByteArray}
    (h : _root_.Spot.FilePart.check program query pubB = .ok ()) :
    ∃ P strata ns hd, _root_.Spot.Program.parse program = .ok P ∧
      _root_.Spot.Partition.strata P query = .ok strata ∧
      _root_.Spot.FilePart.sizes strata = some ns ∧ _root_.Spot.FilePart.header pubB = .ok hd ∧
      _root_.Spot.FilePart.indicesOf hd = .ok (Array.range (ns.foldl (· + ·) 0)) := by
  unfold _root_.Spot.FilePart.check _root_.Spot.FilePart.unitIndices at h
  cases hP : _root_.Spot.Program.parse program with
  | error e => rw [hP, error_bind] at h; cases h
  | ok P =>
    rw [hP, ok_bind] at h
    cases hS : _root_.Spot.Partition.strata P query with
    | error e => rw [hS, error_bind] at h; cases h
    | ok strata =>
      rw [hS, ok_bind] at h
      cases hH : _root_.Spot.FilePart.header pubB with
      | error e => rw [hH, error_bind, error_bind] at h; cases h
      | ok hd =>
        rw [hH, ok_bind] at h
        cases hI : _root_.Spot.FilePart.indicesOf hd with
        | error e => rw [hI, error_bind] at h; cases h
        | ok idx =>
          rw [hI, ok_bind] at h
          cases hw : _root_.Spot.FilePart.why strata idx with
          | some w => rw [hw] at h; cases h
          | none =>
            obtain ⟨ns, hns, rfl⟩ := why_none hw
            exact ⟨P, strata, ns, hd, rfl, hS, hns, rfl, hI⟩

end Properties.Spot.FilePart
