import Definitions.Pouw.Guarantees.Receipt
import Proofs.Pouw.Receipt.Duo
import Proofs.Pouw.Receipt.Window
import Proofs.Pouw.Receipt.Li
import Proofs.Pouw.Receipt.Fx1
import Proofs.Pouw.Receipt.Run
import Proofs.Pouw.Receipt.DuoReceipt
import Proofs.Pouw.Receipt.AtState
import Proofs.Pouw.Receipt.Xor

/-!
# The receipt folds duo's, fx1's and li's guarantees

duo's four words anywhere in a run, its checkpoints every four words and the kernel of its long runs are in
`Receipt/Window.lean`.

Each statement of `Definitions/Pouw/Guarantees/Receipt.lean`: duo's words, pairs, triples and quadruples in
`Receipt/Duo.lean`, the state bijection and runs of any length in `Receipt/Run.lean`, the served receipt in
`Receipt/DuoReceipt.lean`, fx1's transport of each in `Receipt/Fx1.lean`, and li's half steps and checkpoints in
`Receipt/Li.lean`.

Runs at one start state, the counting bound past two words and the served chain's two-word counterexample, are in
`Receipt/AtState.lean`. Two words' all-state XOR kernel, the warp checkpoint's cross-chain cancellation, is in
`Receipt/Xor.lean`.
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

theorem DuoWindowNoKernel : Pouw.Guarantees.Receipt.DuoWindowNoKernel :=
  fun _ _ _ k a r w q k' a' r' w' => Pouw.ReceiptProofs.duoWindow_injective k a r w q k' a' r' w'

theorem DuoCheckpointNoKernel : Pouw.Guarantees.Receipt.DuoCheckpointNoKernel :=
  fun _ _ q => Pouw.ReceiptProofs.duoCheckpoints_injective q

theorem DuoLongRunKernel : Pouw.Guarantees.Receipt.DuoLongRunKernel :=
  fun n _ k a r => Pouw.ReceiptProofs.duoRun_kernel n.succ_pos (by omega) k a r

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

theorem LiHalfBijective : Pouw.Guarantees.Receipt.LiHalfBijective :=
  fun _ k a d => Pouw.ReceiptProofs.liHalf_bijective k a d

theorem LiCheckpointWordInjective : Pouw.Guarantees.Receipt.LiCheckpointWordInjective :=
  fun _ before after pre k a d post y => Pouw.ReceiptProofs.liCheckpoint_word_injective before after pre k a d post y

theorem LiCheckpointLowBit : Pouw.Guarantees.Receipt.LiCheckpointLowBit :=
  fun _ _ x before after pre k a d post y => Pouw.ReceiptProofs.liCheckpoint_lowBit x before after pre k a d post y

theorem DuoWordProjNoKernel : Pouw.Guarantees.Receipt.DuoWordProjNoKernel :=
  fun _ k a r h => Pouw.ReceiptProofs.duoWordProj_injective k a r h

theorem DuoReceiptNotInjective : Pouw.Guarantees.Receipt.DuoReceiptNotInjective :=
  fun hinj => (by decide : (0x1632c#32) ≠ 0x2237a#32) (hinj (a₁ := 0x1632c#32) (a₂ := 0x2237a#32) (by decide))

theorem DuoPairAtStateNotInjective : Pouw.Guarantees.Receipt.DuoPairAtStateNotInjective :=
  fun hinj => (by decide : ((0x40156b95#32, 0x0#32) : BitVec 32 × BitVec 32) ≠ (0xb4d85e5c#32, 0x375f7285#32))
    (hinj (a₁ := (0x40156b95#32, 0x0#32)) (a₂ := (0xb4d85e5c#32, 0x375f7285#32)) (by decide))

theorem DuoRunAtStateNotInjective : Pouw.Guarantees.Receipt.DuoRunAtStateNotInjective :=
  fun n L k a r s => Pouw.ReceiptProofs.duoRun_atState_not_injective k a r s (Nat.succ_pos n) (by omega : 2 < L + 3)

theorem DuoPairXorNoKernel : Pouw.Guarantees.Receipt.DuoPairXorNoKernel :=
  fun _ k₁ a₁ r₁ k₂ a₂ r₂ => Pouw.ReceiptProofs.duoPairXor_injective k₁ a₁ r₁ k₂ a₂ r₂

theorem Fx1PairXorNoKernel : Pouw.Guarantees.Receipt.Fx1PairXorNoKernel :=
  fun _ k₁ a₁ r₁ m₁ k₂ a₂ r₂ m₂ => Pouw.ReceiptProofs.fx1PairXor_injective k₁ a₁ r₁ m₁ k₂ a₂ r₂ m₂

end Pouw.SecurityProofs.Receipt
