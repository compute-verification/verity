import Models.Pouw.Guarantees.Receipt
import Proofs.Pouw.Receipt.Duo
import Proofs.Pouw.Receipt.Li

/-!
# The receipt folds duo's and li's guarantees

Each statement of `Models/Pouw/Guarantees/Receipt.lean`, proved in `Receipt/Duo.lean` (duo) and `Receipt/Li.lean` (li).
-/

namespace Pouw.SecurityProofs.Receipt

theorem DuoWordInjective : Pouw.Guarantees.Receipt.DuoWordInjective :=
  fun _ k a r s => Pouw.ReceiptProofs.duoWord_injective k a r s

theorem DuoPairNoKernel : Pouw.Guarantees.Receipt.DuoPairNoKernel :=
  fun _ k₁ a₁ r₁ k₂ a₂ r₂ => Pouw.ReceiptProofs.duoPair_injective k₁ a₁ r₁ k₂ a₂ r₂

theorem DuoTripleNoKernel : Pouw.Guarantees.Receipt.DuoTripleNoKernel :=
  fun _ k₁ a₁ r₁ k₂ a₂ r₂ k₃ a₃ r₃ => Pouw.ReceiptProofs.duoTriple_injective k₁ a₁ r₁ k₂ a₂ r₂ k₃ a₃ r₃

theorem DuoQuadNoKernel : Pouw.Guarantees.Receipt.DuoQuadNoKernel :=
  fun _ k₁ a₁ r₁ k₂ a₂ r₂ k₃ a₃ r₃ k₄ a₄ r₄ =>
    Pouw.ReceiptProofs.duoQuad_injective k₁ a₁ r₁.1 r₁.2.1 r₁.2.2 k₂ ⟨a₂.1, a₂.2.1⟩ a₂.2.2 r₂ k₃ a₃ r₃ k₄ a₄ r₄

theorem LiHalfBijective : Pouw.Guarantees.Receipt.LiHalfBijective :=
  fun _ k a d => Pouw.ReceiptProofs.liHalf_bijective k a d

theorem LiCheckpointWordInjective : Pouw.Guarantees.Receipt.LiCheckpointWordInjective :=
  fun _ before after pre k a d post y => Pouw.ReceiptProofs.liCheckpoint_word_injective before after pre k a d post y

theorem LiCheckpointLowBit : Pouw.Guarantees.Receipt.LiCheckpointLowBit :=
  fun _ _ x before after pre k a d post y => Pouw.ReceiptProofs.liCheckpoint_lowBit x before after pre k a d post y

end Pouw.SecurityProofs.Receipt
