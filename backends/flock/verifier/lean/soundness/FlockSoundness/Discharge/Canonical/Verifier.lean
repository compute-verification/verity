import FlockSoundness.Discharge.Canonical.Format
import FlockSoundness.Discharge.FailClosed.Scope
import FlockSoundness.Protocol.Discharge.Canonical.Verifier

set_option Elab.async false

/-!
# The verifier's obligation: the statement check accepts canonical circuits only

`parse_formatOk` is the one statement about the executable verifier the canonical headline (`Headline.lean`) adds: a
circuit the statement check passes (`FailClosed.Checks`) outside `Layout.scopeOk` is at a `--zk` statement with hidden
outputs, its unit returns the outputs, and it is in the canonical format (`formatOk`).

The check (`Flock.ProvedScope.check`, verifier change V1, `PROTOCOL.md` §16.16) accepts a circuit that `scopeOk` or
`canonicalOk` accepts; `canonicalOk` is `--zk`, hidden outputs, `outNet = unitNet` and the executable's
`ProvedScope.formatOk`. That is `formatOk` with its `readsRows` (Δ's key at every stacked unit input is not a wire or a
fixed column) as `cutsCovered`, a mask over the leaf cuts: every cut port of every copy has one
(`readsRows_of_cutsCovered`; at a cut port `srcAt` reads a wire or a fixed column only where no leaf cut names it), and
`portReadsB` as the port mask's (`FailClosed.portReadsB_eq`). `formatOk_of_provedScope` is the translation.

Every declaration here is proved.
-/

namespace FlockSoundness.Discharge.Canonical

open Flock Exec

end FlockSoundness.Discharge.Canonical

namespace FlockSoundness.Discharge.FailClosed

open Flock Exec ZkExec

/-- **A `--zk` statement whose circuit is outside `Layout.scopeOk` has hidden outputs.** -/
theorem ScopeZ.hidden {I : Inputs} (h : ScopeZ I) (hv : Layout.scopeOk (circZ I) = false) :
    I.tags.hiddenOutputs = true := by
  rw [← zkTags_hiddenOutputs]; exact (Canonical.parse_formatOk h hv).2.1

end FlockSoundness.Discharge.FailClosed
