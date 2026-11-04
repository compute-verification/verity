/-
A Program's `Q_word` evaluation, as `Flock.Extract.evaluate` returns it: `partition_object.Partition`, which
`verity-evaluate`'s `partition` function holds to it.
-/
import Flock.Extract

namespace Flock.Extract

/-- `partition_object.Partition`: per Call, in root-body order, its root-body index, its Definition's index and that
Definition's cut (activations of one Definition share it); its units are numbered by Call, then within the Call's cut. -/
abbrev Partition := Array (Nat × Nat × Qword.DCut)

end Flock.Extract
