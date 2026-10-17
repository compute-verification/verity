import Models.Pouw.Guarantees.Receipt
import Proofs.Pouw.Receipt.Duo
import Proofs.Pouw.Receipt.DuoReceipt

/-!
# The receipt fold duo's guarantees

Each statement of `Models/Pouw/Guarantees/Receipt.lean`, proved in `Receipt/Duo.lean` (the whole state) and
`Receipt/DuoReceipt.lean` (the served receipt).
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

theorem DuoWordProjNoKernel : Pouw.Guarantees.Receipt.DuoWordProjNoKernel :=
  fun _ k a r h => Pouw.ReceiptProofs.duoWordProj_injective k a r h

end Pouw.SecurityProofs.Receipt
