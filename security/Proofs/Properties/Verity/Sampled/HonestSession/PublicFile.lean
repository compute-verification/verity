module

public import Proofs.Properties.Verity.Sampled.HonestSession.CircuitFile

/-!
# The public file loads, is drawn at the sample file and checks

Each side's reading of the public file over `cH` (`HmOut.loadPublic`, `HmIn.loadPublic` over no public input port), the
sampled statement's public file at `draw₀` (`HmIn.drawn`) and its check (`HmOut.checkPublic`), from the circuit's fields
`CircuitFileFacts` names. The header is `printJson hdrJ`, which parses back to `norm hdrJ` (`FirewallJson.parse_printJson`);
the walk is `Proofs.Flock.SessionLoop.Concrete.Public`'s, at the hidden-output header. These facts are the named
proposition `PublicFileFacts`, proved from `CircuitFileFacts` by `publicFileFacts` (`PublicFileProof.lean`): the pieces that
use it take it as a hypothesis, and `Witness` supplies the proof.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession

open Flock

/-- **Facts about the concrete public file** (proved: `publicFileFacts`): each side's load of it over `cH` returns the public value with
population `nUnits`, the verifier's own-port check passes, it is drawn at the sample file and passes the public check,
and its header publishes the one row's domain, the 64 bytes `domY`. -/
structure PublicFileFacts : Prop where
  /-- `HmOut.loadPublic` reads the file to `pubO`. -/
  hmOut_loadPublic : HmOut.loadPublic tagsZ cH publicFile = .ok pubO
  /-- `HmIn.loadPublic` over no public input port reads the file to `pubI`. -/
  hmIn_loadPublic : HmIn.loadPublic tagsZ cH {} publicFile = .ok pubI
  /-- `pubO`'s population is `nUnits`. -/
  pubO_n : pubO.n = nUnits
  /-- `pubI`'s population is `nUnits`. -/
  pubI_n : pubI.n = nUnits
  /-- The verifier's own public input check passes at `pubI`. -/
  checkOwn_ok : HmIn.checkOwn {} pubI none = .ok ()
  /-- The public file is drawn at the sample file `draw₀`, to `pubD`. -/
  drawn_ok : HmIn.drawn cH {} pubI draw₀ = .ok pubD
  /-- `pubD` passes the public check at `cH`. -/
  checkPublic_ok : HmOut.checkPublic cH pubD = .ok ()
  /-- `pubO`'s header publishes `domY` as the one port's domain id. -/
  pubO_domain : pubO.fv3Bytes "domain_ids" "y" = .ok domY

end Verity.Sampled.Instance.HonestSession
