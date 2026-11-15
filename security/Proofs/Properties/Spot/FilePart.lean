module

public import Spot.FilePart
public import Spot.Partition
public import Spot.Program

/-! # The population check pins the public file's units (`Spot.FilePart.check`)

`FilePart.check` reads the public file's `units.indices` off its header (`FilePart.header`: the header line through the
verifier's total parser, then `Flock.norm`), the value v2's other reads of the file read too (`OneStageZk.publicHeld`,
and `OneStageZk.provedUnits` through the same `FilePart.indicesOf`). Where it passes:

* `why_none`: the registration's strata, read stratum by stratum, are its units in order (`sizes` gives their sizes
  `ns`), and the file's units are `Array.range N`, `N` their sum;
* `check_indices`: the verifier's program parses, its strata under the query are those, and the file's header parses
  and names `Array.range N` as its `units.indices`.
-/

@[expose] public section

open Lean (Json)

namespace Properties.Spot.FilePart

theorem ok_bind {ε α β : Type} (a : α) (f : α → Except ε β) : (Except.ok a >>= f) = f a := rfl

theorem error_bind {ε α β : Type} (e : ε) (f : α → Except ε β) : (Except.error e >>= f) = Except.error e := rfl

/-- **`why` passes only on the units in order**: the strata read stratum by stratum are the units in order, of sizes
`ns`, and the file's units are `0, …, N − 1`, `N` the sum of `ns`. -/
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

/-- **The population check pins the file's units**: where `FilePart.check` passes on the verifier's program and query
and the public file `pubB`, the program parses to `P`, its strata under the query are `strata` of sizes `ns`, and the
file's header (`FilePart.header`) parses to `hd`, whose `units.indices` (`FilePart.indicesOf`) are `0, …, N − 1`, `N`
the sum of `ns`. -/
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
