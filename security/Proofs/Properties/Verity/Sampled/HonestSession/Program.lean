module

public import Proofs.Properties.Verity.Sampled.HonestSession.Data.Inputs
public import Proofs.Properties.Verity.Sampled.HonestSession.Hex

/-!
# The program, its population and its sampling design

The descriptor parses to `program` (`program_parse`), its query's population is the public file's one proof block
(`popMatches`), the sampling design the verifier derives from it is the subset of one (`design_eq`, `lawAt_of`), and
the partition object's digest is `partitionDigestB` (`partitionDigest_ok`). `PopMatches` is `popCheck`'s verdict, which
evaluates the partition object's validity (`isSha512Hex` of the program's spliced SHA-512, `Hex.isSha512Hex_hexOf`) and
the query over the descriptor; no step evaluates a hash. `Array.qsort` over the descriptor's keys is read by
`qsort_of_sorted`, the keys being ascending.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession.Program

open Flock

theorem program_parse : Spot.Program.parse ⟨desc.toArray⟩ = .ok program := by
  sorry -- piece P4 (program)

theorem program_sha : program.sha512 = programSha := by
  sorry -- piece P4 (program)

theorem popMatches : Flock.Bridge.PopMatches desc query nUnits := by
  sorry -- piece P4 (program)

theorem design_eq : Flock.Bridge.designAt desc query law none nUnits = .subset 1 := by
  sorry -- piece P4 (program)

theorem lawAt_of (v : Spot.Registration.Verifier) (hp : v.program = program) (hq : v.query = query)
    (hl : v.law = law) (ht : v.workTable = none) :
    lawAt v nUnits = .ok (Flock.Bridge.designAt desc query law none nUnits) := by
  sorry -- piece P4 (program)

theorem partitionDigest_ok : HmRow.partitionDigest partitionObj = .ok partitionDigestB := by
  sorry -- piece P4 (program)

end Verity.Sampled.Instance.HonestSession.Program
