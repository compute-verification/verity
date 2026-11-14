module

public import Proofs.Properties.Verity.Sampled.HonestSession.CircuitFile

/-!
# The public file loads, is drawn at the sample file and checks

Each side's reading of the public file over `cH` (`HmOut.loadPublic`, `HmIn.loadPublic` over no public input port), the
sampled statement's public file at `draw₀` (`HmIn.drawn`) and its check (`HmOut.checkPublic`), from the circuit's fields
`CircuitFile` names. The header is `printJson hdrJ`, which parses back to `norm hdrJ` (`FirewallJson.parse_printJson`);
the walk is `Proofs.Flock.SessionLoop.Concrete.Public`'s, at the hidden-output header.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession.PublicFile

open Flock

theorem hmOut_loadPublic : HmOut.loadPublic tagsZ cH publicFile = .ok pubO := by
  sorry -- piece P2 (public file)

theorem hmIn_loadPublic : HmIn.loadPublic tagsZ cH {} publicFile = .ok pubI := by
  sorry -- piece P2 (public file)

theorem pubO_n : pubO.n = nUnits := by
  sorry -- piece P2 (public file)

theorem pubI_n : pubI.n = nUnits := by
  sorry -- piece P2 (public file)

theorem checkOwn_ok : HmIn.checkOwn {} pubI none = .ok () := by
  sorry -- piece P2 (public file)

theorem drawn_ok : HmIn.drawn cH {} pubI draw₀ = .ok pubD := by
  sorry -- piece P2 (public file)

theorem checkPublic_ok : HmOut.checkPublic cH pubD = .ok () := by
  sorry -- piece P2 (public file)

end Verity.Sampled.Instance.HonestSession.PublicFile
