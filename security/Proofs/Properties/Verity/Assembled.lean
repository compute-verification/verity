module

public import Proofs.Properties.Verity.Soundness
public import Proofs.Properties.Verity.Sampled.Target

/-!
# Soundness at remote's verifier, from the targets still open

`Verity.SoundnessAt .remote` for a vocabulary whose sampled fields are the deployment's sessions (`Sampled.Deployed`,
`Sampled.Covers`), from the union bound over remote's three subprotocols (`remote_soundness_of_subprotocols`) with each
target discharged by its own subprotocol's theorem: SPOT's by `Sampled.spotSound_of_targets` (the coupling at each
session), PoUW's (`Vocab.PouwSound .remote`) and the inference-only policy's (`Vocab.InferenceOnlySound .remote`). Each
hypothesis is a target its goal proves; once each is a theorem, the conclusion holds by applying this one to them.
-/

@[expose] public section

namespace Verity

variable {ι α : Type} [Fintype α] {d : Accounting.Device ι}

/-- **Soundness at remote, from the targets still open**: remote's compiled verifier accepts files that passed the
commit phase's check while one of its three claims fails with probability at most their errors summed, for a vocabulary
whose sampled fields read the constructions `K` at the deployment's sessions `S`, wherever each session's coupling,
PoUW's target and the inference-only policy's target hold. -/
theorem remote_soundness_of_targets (V : Vocab d α) {K : Sampled.Constructions V}
    {S : ByteArray → Array ByteArray → Option Sampled.Live} (hD : Sampled.Deployed V K S)
    (hCov : Sampled.Covers V .remote S) (hC : ∀ θ F A, S θ F = some A → Sampled.CouplingTarget V .remote K A θ F)
    (hW : V.PouwSound .remote) (hI : V.InferenceOnlySound .remote) :
    V.SoundnessAt .remote :=
  remote_soundness_of_subprotocols ι α d V (Sampled.spotSound_of_targets V .remote hD hCov hC) hW hI

end Verity
