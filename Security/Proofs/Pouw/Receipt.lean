import Models.Pouw.Guarantees.Receipt
import Proofs.Pouw.Receipt.Duo
import Proofs.Pouw.Receipt.Fx1
import Proofs.Pouw.Receipt.Run

/-!
# The receipt folds duo's and fx1's guarantees

Each statement of `Models/Pouw/Guarantees/Receipt.lean`: duo's words, pairs, triples and quadruples in
`Receipt/Duo.lean`, the state bijection and runs of any length in `Receipt/Run.lean`, and fx1's transport of each in
`Receipt/Fx1.lean`.
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

theorem DuoStepBijective : Pouw.Guarantees.Receipt.DuoStepBijective :=
  fun _ k r => ⟨fun a d => Pouw.ReceiptProofs.duoWord_state_bijective k a r d,
    fun a s => Pouw.ReceiptProofs.duoWord_injective k a r s⟩

theorem DuoRunWordInjective : Pouw.Guarantees.Receipt.DuoRunWordInjective :=
  fun _ _ k a r s w i => Pouw.ReceiptProofs.duoRun_update_injective k a r s w i

theorem Fx1WordInjective : Pouw.Guarantees.Receipt.Fx1WordInjective :=
  fun _ k a r m s => Pouw.ReceiptProofs.fx1Word_injective k a r m s

theorem Fx1PairNoKernel : Pouw.Guarantees.Receipt.Fx1PairNoKernel :=
  fun _ k₁ a₁ r₁ m₁ k₂ a₂ r₂ m₂ => Pouw.ReceiptProofs.fx1Pair_injective k₁ a₁ r₁ m₁ k₂ a₂ r₂ m₂

theorem Fx1TripleNoKernel : Pouw.Guarantees.Receipt.Fx1TripleNoKernel :=
  fun _ k₁ a₁ r₁ m₁ k₂ a₂ r₂ m₂ k₃ a₃ r₃ m₃ =>
    Pouw.ReceiptProofs.fx1Triple_injective k₁ a₁ r₁ m₁ k₂ a₂ r₂ m₂ k₃ a₃ r₃ m₃

theorem Fx1QuadNoKernel : Pouw.Guarantees.Receipt.Fx1QuadNoKernel :=
  fun _ k₁ a₁ r₁ m₁ k₂ a₂ r₂ m₂ k₃ a₃ r₃ m₃ k₄ a₄ r₄ m₄ =>
    Pouw.ReceiptProofs.fx1Quad_injective k₁ a₁ r₁ m₁ k₂ a₂ r₂ m₂ k₃ a₃ r₃ m₃ k₄ a₄ r₄ m₄

theorem Fx1StepBijective : Pouw.Guarantees.Receipt.Fx1StepBijective :=
  fun _ k r => ⟨fun a m d => Pouw.ReceiptProofs.fx1Word_state_bijective k a r m d,
    fun a m s => Pouw.ReceiptProofs.fx1Word_injective k a r m s⟩

theorem Fx1RunWordInjective : Pouw.Guarantees.Receipt.Fx1RunWordInjective :=
  fun _ _ k a r m s w i => Pouw.ReceiptProofs.fx1Run_update_injective k a r m s w i

end Pouw.SecurityProofs.Receipt
