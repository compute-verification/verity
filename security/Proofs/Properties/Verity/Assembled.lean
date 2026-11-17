module

public import Proofs.Properties.Verity.InferenceOnly
public import Proofs.Properties.Verity.Sampled.Target

/-! `Verity.SoundnessAt .remote` from the union bound over remote's three subprotocols, at the targets still open. -/

@[expose] public section

namespace Verity

variable {ι α : Type} [Fintype α] {d : ResourceAccounting.Device ι}

theorem remote_soundness_of_targets (V : Vocab d α) {K : Sampled.Constructions V}
    {S : ByteArray → Array ByteArray → Option Sampled.Live} (hD : Sampled.Deployed V K S)
    (hCov : Sampled.Covers V .remote S) (hC : ∀ θ F A, S θ F = some A → Sampled.CouplingTarget V .remote K A θ F)
    (hW : V.PouwSound .remote) (hWi : V.WiresSound .remote) (hG : GateARow) :
    V.SoundnessAt .remote :=
  remote_soundness_of_subprotocols ι α d V (Sampled.spotSound_of_targets V .remote hD hCov hC) hW hWi hG

end Verity
