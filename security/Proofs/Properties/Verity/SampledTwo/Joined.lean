module

public import Proofs.Properties.Verity.SampledTwo.ProjectionOf
public import Proofs.Properties.Verity.SampledTwo.JoinElems
public import Proofs.Properties.Verity.SampledTwo.MessagesShown

/-! The two-stage subprotocol's projection at the coin map's frame and the form's reading, over any
acceptance; it neither closes nor widens the earlier-period signature hole `ProjectionOf.lean` names. -/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open scoped ENNReal

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

noncomputable abbrev joinReading (A : Live₂) (C : CaseOf) (S : SessionReading) (L : CoinMap.Locate) : Reading A :=
  formReading A C S (CoinMap.dev L)

noncomputable abbrev joinPlace (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (L : CoinMap.Locate) :
    TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ :=
  fun ω r => CoinMap.place V θ F c σ L r ω

noncomputable abbrev joinFrame (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (C : CaseOf) (S : SessionReading) (L : CoinMap.Locate)
    (Dg : Verity.Exchange → ℕ → Option Msg) : Frame V A x θ F c σ :=
  Messages.frame V A x θ F c σ (joinPlace V A x θ F c σ L) (joinReading A C S L) Dg

theorem joinFrame_eq (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (C : CaseOf) (S : SessionReading) (L : CoinMap.Locate)
    (Dg : Verity.Exchange → ℕ → Option Msg) :
    joinFrame V A x θ F c σ C S L Dg =
      CoinMap.frame V A x θ F c σ L
        (Messages.prover V θ F c σ (joinPlace V A x θ F c σ L) (joinReading A C S L) Dg) :=
  rfl

noncomputable def joinMiss (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (C : CaseOf) (S : SessionReading) (L : CoinMap.Locate)
    (Dg : Verity.Exchange → ℕ → Option Msg) (q : ℕ) : ℝ≥0∞ :=
  ∑ i ∈ Finset.range (A.L.V.k x), ENNReal.ofReal (Real.sqrt
    ((Messages.msgFinder V θ F c σ (joinPlace V A x θ F c σ L) (joinReading A C S L) Dg i q).Pr Verity.sha512Msg))

theorem joined (V : Vocab d α) (k : Verity.Registration.Composite) (A : Live₂) (x : Msg) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (C : CaseOf)
    (S : SessionReading) (L : CoinMap.Locate) (Dg : Verity.Exchange → ℕ → Option Msg) (q : ℕ)
    (Acc Good : Verity.Exchange → Prop) (terms : ℝ≥0∞)
    (hAcc : ∀ o, o.params = θ → V.Verdict k .sampled o → Acc o)
    (hDistribution : CoinMapDistribution V A x θ F c σ
      (CoinMap.frame V A x θ F c σ L
        (Messages.prover V θ F c σ (joinPlace V A x θ F c σ L) (joinReading A C S L) Dg))
      (CoinMap.ElemAgrees L A x (readsAt A.h)) (CoinMap.gapOf V θ F c σ L (A.L.V.k x)))
    (hFit₁ : FirstFits C L Acc Good) (hFit₂ : SecondFits A C L Acc Good)
    (hSess : SessionsAgree A x S (CoinMap.dev L) Acc Good (CoinMap.ElemAgrees L A x (readsAt A.h)))
    (hFirst : CoinsAgree A (joinReading A C S L) Acc Good .first)
    (hSecond : CoinsAgree A (joinReading A C S L) Acc Good .second)
    (hSessions : CoinsAgree A (joinReading A C S L) Acc Good .sessions)
    (hStmt : StmtReadsNone A (joinReading A C S L))
    (hOpens : Messages.MsgOpens A x (joinReading A C S L) Acc Good Dg)
    (hDigest : ∀ i < A.L.V.k x,
      Messages.DigestBefore (joinFrame V A x θ F c σ C S L Dg) (joinReading A C S L) Acc Good Dg i)
    (hCut : CutBound V k θ F c σ Good terms)
    (hReach : ∀ o : Verity.Exchange, Acc o → Good o → (joinReading A C S L).stmt o = some x →
      ∀ (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (r : Coins A x),
        (∀ i : Fin (A.L.V.k x), (joinReading A C S L).reads i.val →
          ∀ c', (joinReading A C S L).coin o i.val = some c' → (joinReading A C S L).agree o i.val c' (r i)) →
        (∀ i < (Claims.run FlockDeploy.sha512 P A.L.V x ω r).msgs.length,
          (joinReading A C S L).msg o i = msgAt (Claims.run FlockDeploy.sha512 P A.L.V x ω r) i) →
        (Claims.run FlockDeploy.sha512 P A.L.V x ω r).out = .acc ∧
          A.L.V.k x ≤ (Claims.run FlockDeploy.sha512 P A.L.V x ω r).msgs.length)
    (hAt : StmtAt V A x θ F c σ (joinFrame V A x θ F c σ C S L Dg) (joinReading A C S L) Acc) :
    ProjectionBound V k A x θ F c σ (joinFrame V A x θ F c σ C S L Dg)
      (CoinMap.gapOf V θ F c σ L (A.L.V.k x) + joinMiss V A x θ F c σ C S L Dg q + terms) := by
  have hDistribution' : CoinMapDistribution V A x θ F c σ (joinFrame V A x θ F c σ C S L Dg)
      (CoinMap.ElemAgrees L A x (readsAt A.h)) (CoinMap.gapOf V θ F c σ L (A.L.V.k x)) := hDistribution
  have hDev : DevAgrees A x (joinReading A C S L) Acc Good (CoinMap.ElemAgrees L A x (readsAt A.h)) :=
    devAgrees_formReading A x C S (CoinMap.dev L) Acc Good (CoinMap.ElemAgrees L A x (readsAt A.h))
      (firstElems_of A x C L (readsAt A.h) Acc Good (readsAt_of_le A.h (Nat.zero_le _)) hFit₁)
      (secondElems_of A x C L (readsAt A.h) Acc Good (fun n _ hn => readsAt_of_le A.h hn) hFit₂) hSess
  have hRun : AcceptsRun A x (joinReading A C S L) (fun o => Acc o ∧ Good o) := by
    intro o hA hs Ω P ω r hc hm
    exact (hReach o hA.1 hA.2 hs Ω P ω r hc fun i hi =>
      hm i (lt_of_lt_of_le hi (Claims.Recursion.run_length_le FlockDeploy.sha512 P A.L.V x ω r))).1
  have hMiss : ∀ i < A.L.V.k x,
      ENNReal.ofReal (Claims.Pr
        (Messages.MissAt (joinFrame V A x θ F c σ C S L Dg) (joinReading A C S L) Acc Good i)) ≤
      ENNReal.ofReal (Real.sqrt ((Messages.msgFinder V θ F c σ (joinPlace V A x θ F c σ L) (joinReading A C S L)
        Dg i q).Pr Verity.sha512Msg)) :=
    fun i hi => Messages.missBound V A x θ F c σ (joinPlace V A x θ F c σ L) (joinReading A C S L) Acc Good Dg i q
      hOpens (hDigest i hi) (Messages.runShown A x)
  have hReached : Messages.MessagesReached V A x θ F c σ (joinFrame V A x θ F c σ C S L Dg) (joinReading A C S L)
      Acc Good (joinMiss V A x θ F c σ C S L Dg q) :=
    Messages.reachedOfRounds V A x θ F c σ (joinFrame V A x θ F c σ C S L Dg) (joinReading A C S L) Acc Good
      (fun i => ENNReal.ofReal (Real.sqrt ((Messages.msgFinder V θ F c σ (joinPlace V A x θ F c σ L)
        (joinReading A C S L) Dg i q).Pr Verity.sha512Msg))) hMiss
  have hMsg : MessagesFixedAgreed V A x θ F c σ (joinFrame V A x θ F c σ C S L Dg) (joinReading A C S L) Acc Good
      (joinMiss V A x θ F c σ C S L Dg q) :=
    messagesFixedAgreed_of_reached
      (fun ω r hA hG hc hm => (hReach _ hA hG (hAt ω r hA) _ (joinFrame V A x θ F c σ C S L Dg).P ω r hc hm).2)
      hReached
  exact projection_agreed V k A x θ F c σ (joinFrame V A x θ F c σ C S L Dg) (joinReading A C S L) Acc Good
    (CoinMap.ElemAgrees L A x (readsAt A.h)) (CoinMap.gapOf V θ F c σ L (A.L.V.k x))
    (joinMiss V A x θ F c σ C S L Dg q) terms hAcc hDistribution' hDev hFirst hSecond hSessions hStmt hMsg hCut hRun hAt

end Verity.SampledTwo
