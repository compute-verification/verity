module

public import Proofs.Properties.Verity.SampledTwo.ProjectionOf
public import Proofs.Properties.Verity.SampledTwo.JoinElems
public import Proofs.Properties.Verity.SampledTwo.MessagesShown

/-!
# The two-stage leg's projection, joined over any acceptance

`projection_agreed` (`ProjectionOf.lean`) gives the projection's bound from its pieces at any frame, reading and
agreement. This module fixes the three to the ones the pieces were proved at, and discharges the pieces that follow
from the others, with the acceptance `Acc` and the predicate `Good` still any:

* **The frame** (`joinFrame`): the coin map at a locator `L` (`CoinMap.place`), and the messages' prover at it
  (`Messages.prover`). It is the coin map's frame at that prover and the messages' frame at that coin map
  (`joinFrame_eq`).
* **The reading** (`joinReading`): the form's reading at the coin map's release (`CoinMap.dev L`).
* **The agreement**: the coin map's (`CoinMap.ElemAgrees`) at the rounds the run's verifier reads.
* **`hide`**: the coin trees' hiding term (`CoinMap.gapOf`). **`miss`** (`joinMiss`): the sum over the rounds of the
  square roots of their two-continuation finders' successes.

`joined` then takes each remaining piece in its owner's words:

* the wiring (`hAcc`), the coin map's law at the frame (`hLaw`), the cut (`hCut`);
* the stages' bytes within their rounds' releases that fit (`FirstFits`, `SecondFits`) and the sessions' releases'
  agreement (`SessionsAgree`), which give `DevAgrees` (`devAgrees_formReading`, `firstElems_of`, `secondElems_of`);
* the three groups' coins (`CoinsAgree`) and no coin read at the statements' round (`StmtReadsNone`);
* the messages opening their digests (`Messages.MsgOpens`) and each digest fixed by the coins shown before it
  (`Messages.DigestBefore`), which with the run's history showing its coins (`Messages.runShown`) bound each round's
  first difference (`Messages.missBound`, `Messages.reachedOfRounds`);
* the accept on the rounds a run reaches (`hReach`: the accept piece's `ReachedAccepts`, written out so that this
  module imports no file over the form's check), which gives both `AcceptsRun` and the run reaching every round where
  the coins read agree (`messagesFixedAgreed_of_reached`), and the statement read (`StmtAt`).

Nothing here reads the form's check. The file that sets `Acc` to the form's accept, `Good` to the cut's and `L` to the
run's locator is the one place the form comes in.

`terms` is taken as given. Where the cut is instantiated at the deployed composite it meets the open hole of the
device's history that `ProjectionOf.lean` names: a strategy whose evidence carries a genuine challenger signature from
an earlier period makes the forger's term `⊤`. The join neither closes nor widens it.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The join's reading**: the form's reading at the coin map's release of the locator `L`. -/
noncomputable abbrev joinReading (A : Live₂) (C : CaseOf) (S : SessionReading) (L : CoinMap.Locate) : Reading A :=
  formReading A C S (CoinMap.dev L)

/-- **The join's coin map**: the run's coins written into the tape at the releases the locator `L` puts on their
rounds. -/
noncomputable abbrev joinPlace (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (L : CoinMap.Locate) :
    TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ :=
  fun ω r => CoinMap.place V θ F c σ L r ω

/-- **The join's frame**: the coin map at the locator `L`, and the messages' prover at it, the join's reading and the
digests `Dg`. -/
noncomputable abbrev joinFrame (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (C : CaseOf) (S : SessionReading) (L : CoinMap.Locate)
    (Dg : Verity.Exchange → ℕ → Option Msg) : Frame V A x θ F c σ :=
  Messages.frame V A x θ F c σ (joinPlace V A x θ F c σ L) (joinReading A C S L) Dg

/-- **The join's frame is the coin map's**, at the messages' prover. -/
theorem joinFrame_eq (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (C : CaseOf) (S : SessionReading) (L : CoinMap.Locate)
    (Dg : Verity.Exchange → ℕ → Option Msg) :
    joinFrame V A x θ F c σ C S L Dg =
      CoinMap.frame V A x θ F c σ L
        (Messages.prover V θ F c σ (joinPlace V A x θ F c σ L) (joinReading A C S L) Dg) :=
  rfl

/-- **The join's `miss`**: the sum over the rounds of the square roots of their two-continuation finders' successes,
each finder at `q` SHA-512 calls. -/
noncomputable def joinMiss (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (C : CaseOf) (S : SessionReading) (L : CoinMap.Locate)
    (Dg : Verity.Exchange → ℕ → Option Msg) (q : ℕ) : ℝ≥0∞ :=
  ∑ i ∈ Finset.range (A.L.V.k x), ENNReal.ofReal (Real.sqrt
    ((Messages.msgFinder V θ F c σ (joinPlace V A x θ F c σ L) (joinReading A C S L) Dg i q).Pr Verity.sha512Msg))

/-- **The projection's bound, joined over any acceptance**: at the join's frame and reading, the wiring, the coin map's
law, the stages' bytes within their rounds' releases, the sessions' releases' agreement, the three groups' coins, the
messages opening their digests, each digest fixed by the coins shown before it, the cut, the accept on the rounds a run
reaches at the plays `Acc` and `Good` hold of, and the statement read give the projection's bound at the hiding term,
the join's `miss` and the cut's `terms`. -/
theorem joined (V : Vocab d α) (k : Verity.Registration.Composite) (A : Live₂) (x : Msg) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (C : CaseOf)
    (S : SessionReading) (L : CoinMap.Locate) (Dg : Verity.Exchange → ℕ → Option Msg) (q : ℕ)
    (Acc Good : Verity.Exchange → Prop) (terms : ℝ≥0∞)
    (hAcc : ∀ o, o.params = θ → V.Verdict k .sampled o → Acc o)
    (hLaw : CoinMapLaw V A x θ F c σ
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
  have hLaw' : CoinMapLaw V A x θ F c σ (joinFrame V A x θ F c σ C S L Dg)
      (CoinMap.ElemAgrees L A x (readsAt A.h)) (CoinMap.gapOf V θ F c σ L (A.L.V.k x)) := hLaw
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
    (joinMiss V A x θ F c σ C S L Dg q) terms hAcc hLaw' hDev hFirst hSecond hSessions hStmt hMsg hCut hRun hAt

end Verity.SampledTwo
