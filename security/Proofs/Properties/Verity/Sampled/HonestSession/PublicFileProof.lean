module

public import Proofs.Properties.Verity.Sampled.HonestSession.PublicFile

/-!
# Proof of `PublicFileFacts`

`PublicFileFacts` (`PublicFile.lean`) over `cH`, taking `CircuitFileFacts` as its one hypothesis. This file proves the
fields that need no fact of `cH` beyond `CircuitFileFacts`; the fields blocked on `CircuitFileFacts` not yet naming
`cH.upv` and `cH.dummyDigests` are reported in the lane note, not proved here.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession

open Flock

/-- `HmIn.checkOwn` over no public input port, no own recomputation: the match `(none, isEmpty = true)` returns, whatever
the population value. -/
theorem checkOwn_ok_pf : HmIn.checkOwn {} pubI none = .ok () := rfl

end Verity.Sampled.Instance.HonestSession
