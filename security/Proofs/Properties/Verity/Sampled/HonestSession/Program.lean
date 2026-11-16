module

public import Proofs.Properties.Verity.Sampled.HonestSession.Data.Inputs
public import Proofs.Properties.Verity.Sampled.HonestSession.Hex

/-!
# The program, its population and its sampling design

The descriptor parses to `program`, its query's population is the public file's one proof block, the sampling design
the verifier derives from it is the subset of one, and the partition object's digest is `partitionDigestB`: the named
proposition `ProgramFacts`, proved with no hypothesis by `ProgramPop.programFacts`. The pieces that use it take it as a
hypothesis. `PopMatches` is `popCheck`'s verdict, which
evaluates the partition object's validity (`isSha512Hex` of the program's spliced SHA-512, `Hex.isSha512Hex_hexOf`) and
the query over the descriptor; no step evaluates a hash. `Array.qsort` over the descriptor's keys is read by
`qsort_of_sorted`, the keys being ascending.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession

open Flock

/-- **Facts about the concrete program** (`ProgramPop.programFacts`): the descriptor parses to `program` with SHA-512 `programSha`, its
query's population matches the public file's, the sampling design is the subset of one, a verifier holding the session's
program, query, sampling design and work table derives that design, and the partition object's digest is `partitionDigestB`. -/
structure ProgramFacts : Prop where
  /-- The descriptor parses to `program`. -/
  program_parse : Spot.Program.parse ⟨desc.toArray⟩ = .ok program
  /-- `program`'s SHA-512 is `programSha`. -/
  program_sha : program.sha512 = programSha
  /-- The query's population matches the descriptor's. -/
  popMatches : Flock.Bridge.PopMatches desc query nUnits
  /-- The sampling design at the session's descriptor, query and sampling design is the subset of one. -/
  design_eq : Flock.Bridge.designAt desc query samplingDesign none nUnits = .subset 1
  /-- A verifier holding the session's program, query, sampling design and work table derives that design at `nUnits`. -/
  lawAt_of : ∀ (v : Spot.Docket.Verifier), v.program = program → v.query = query → v.samplingDesign = samplingDesign →
    v.workTable = none → samplingDesignAt v nUnits = .ok (Flock.Bridge.designAt desc query samplingDesign none nUnits)
  /-- The partition object's digest is `partitionDigestB`. -/
  partitionDigest_ok : HmRow.partitionDigest partitionObj = .ok partitionDigestB

end Verity.Sampled.Instance.HonestSession
