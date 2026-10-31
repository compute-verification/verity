import Properties.NetworkCertifier.Period

/-!
# The named assumption of an honest period (`Properties.NetworkCertifier.PeriodAcceptMeans`' honest lemma)

That the honest period of `Properties.NetworkCertifier.Period` (`honestParams`, `honestWindow`) has a proof of its one
link-window that one-stage v2 accepts and that opens: N2's completeness on that link-window, which no kernel evaluation
makes, since the kernel runs no prover.
-/

open NetworkCheck.Period

namespace Properties.NetworkCertifier

/-- **Named assumption: an honest link-window is proved.** For some clock link under Σ_sync `honestSyncs` (N2's
declared ends and the auditor's one-stage v2 parameters for it) and N2's registered Program `prog` for it, some
registration of `prog` at window 0, the 64-byte digests both ends' certificates sign, and the salts, v2 evidence and
blobs of a proof of link-window 0 under those certificates pass one-stage v2 and open to the verifier's own public row:
N2's completeness on that link-window, which needs a proof v2 accepts, and the kernel runs no prover. -/
def HonestWindowProved : Prop :=
  ∃ (l : ClockLink) (prog : String) (reg : Registration) (de di : ByteArray) (s : Salts)
    (ev : SampledProofs.OneStageZk.Evidence) (bs : Core.Blobs),
    l.syncSet = honestSyncs ∧ de.size = 64 ∧ di.size = 64 ∧ PeriodWindowRegistered (honestParams l prog) 0 reg ∧
      PeriodWindowV2 (honestParams l prog) l #[] 0 reg (honestWindow de di s ev) bs ∧
      PeriodWindowOpened (honestParams l prog) l 0 reg (honestWindow de di s ev) bs

end Properties.NetworkCertifier
