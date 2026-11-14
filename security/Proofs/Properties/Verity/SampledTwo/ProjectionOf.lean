module

public import Proofs.Properties.Verity.SampledTwo.Projection
public import Definitions.Core.Game.TwoRun
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.KS
import Proofs.Core.Game.InterleaveBridge
import Proofs.Properties.Verity.Sampled.CoinMap
import Proofs.Properties.Verity.Sampled.RecordDraw

/-!
# The two-stage leg's projection, assembled

`projection_of` proves `ProjectionOf`: the projection's named pieces give its bound. The argument is probability
bookkeeping over the pieces' `Prop`s, and reads nothing of the form's check but what they state.

1. On a play of the game after `θ`, which is at the parameters `θ` (`play_params`), the sampled verdict accepts only
   where the form's check does (`Acc`, the composite's wiring's), so a sampled accept is an accepted play whose log
   the check read is the device's (`Good`), or a sampled accept with `¬ Good`, which the cut bounds by `terms`
   (`CutBound`).
2. The first is moved to the frame's plays, at a uniform tape and uniform run coins where the coin map's agreement
   `Ag` holds, by the coin map's law (`CoinMapLaw`), at the cost `hide`.
3. On a frame play with `Acc ∧ Good` and `Ag`, each read round's coin read agrees with the run's (`ReadAgrees`;
   `projection_core`'s one premise about coins). There, either a message the check reads differs from the frame's
   prover's, which the messages' piece bounds by `miss`, or the run at the same tape and coins accepts: the check reads
   the frame's statement (`StmtAt`), and the accept piece (`AcceptsRun`) gives the run's accept.

Step 3's coins: every release the play's log holds for a read round agrees with the run's coin there (`DevAgrees`, at
`Ag`), and the coin the check reads there agrees with whatever that release agrees with, by the coins' piece of its
group (`CoinsAgree`; the statements' round reads none, `StmtReadsNone`).

Step 3's messages: the assembly meets a differing message only on plays where the coins read agree with the run's, so
it takes the messages' piece with that agreement in its event (`MessagesFixedAgreed`, which `MessagesFixed` gives:
`projection_agreed`, and `projection_of` from it). The difference matters to whoever proves the piece: on coins the
check did not read, the run's verifier may refuse before its last round, and the run then has no message where the
check read one, with no digest opened two ways. `messagesFixedAgreed_of_reached` gives the piece from the messages
fixed at the rounds a run reaches and the run reaching every round on agreeing coins.

`Good` is one predicate across the pieces. Each piece that takes it as a hypothesis of its plays holds at any stronger
one (`devAgrees_mono`, `coinsAgree_mono`, `messagesFixedAgreed_mono`), so the pieces may be proved at a weaker `Good`
than the cut's. A fact every play of the game has, which a piece needs of each exchange it speaks of, joins the cut's
`Good` at no cost (`cutBound_and`).

Where `CutBound` is instantiated at the deployed composite it meets an open hole of the device's history: a strategy
whose evidence carries a genuine challenger signature from an earlier period makes the forger's term `⊤`, because the
history holds one play's signatures. The assembly takes `terms` as given and neither closes nor widens that hole.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **A play of the game after `θ`, `F` and `c` is at the parameters `θ`.** -/
theorem play_params (V : Vocab d α) {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {σ : Verity.Game.Strategy (V.game θ F c)} {o : Verity.Exchange}
    (h : Verity.Game.PlayReaches (V.game θ F c) σ o) : o.params = θ := by
  obtain ⟨log, s, h⟩ := Verity.Game.PlayReaches.of_bind (f := finish (V.signer θ) θ F c σ.1)
    (releases (frames (V.families θ)) (V.families θ) []) σ.2 h
  exact (Verity.Sampled.reaches_finish ⟨s, h⟩).1

/-- **The coins the check read agree with the run's**: on the play `o`, at each round the run's verifier reads, the
coin the check reads agrees with the run's coin `r` there. It is `AcceptsRun`'s premise on the coins. -/
def ReadAgrees (A : Live₂) (x : Msg) (Rd : Reading A) (o : Verity.Exchange) (r : Coins A x) : Prop :=
  ∀ i : Fin (A.L.V.k x), Rd.reads i.val → ∀ c', Rd.coin o i.val = some c' → Rd.agree o i.val c' (r i)

/-- **The messages fixed where the coins agree** (the messages' piece as the assembly uses it): `MessagesFixed` with,
in its event, the coins the check read agreeing with the run's (`ReadAgrees`). `MessagesFixed` gives it
(`messagesFixedAgreed_of_fixed`). -/
def MessagesFixedAgreed (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ) (Rd : Reading A)
    (Acc Good : Verity.Exchange → Prop) (miss : ℝ≥0∞) : Prop :=
  ENNReal.ofReal (Claims.Pr fun s : TapeOf V θ F c σ × Coins A x =>
      Acc (Fr.out s.1 s.2) ∧ Good (Fr.out s.1 s.2) ∧ ReadAgrees A x Rd (Fr.out s.1 s.2) s.2 ∧
        ∃ i < A.L.V.k x, Rd.msg (Fr.out s.1 s.2) i ≠
          msgAt (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2) i) ≤ miss

/-- **`MessagesFixed` gives `MessagesFixedAgreed`**: the event is within `MessagesFixed`'s. -/
theorem messagesFixedAgreed_of_fixed {V : Vocab d α} {A : Live₂} {x : Msg} {θ : ByteArray} {F : Array ByteArray}
    {c : ByteArray} {σ : Verity.Game.Strategy (V.game θ F c)} {Fr : Frame V A x θ F c σ} {Rd : Reading A}
    {Acc Good : Verity.Exchange → Prop} {miss : ℝ≥0∞} (h : MessagesFixed V A x θ F c σ Fr Rd Acc Good miss) :
    MessagesFixedAgreed V A x θ F c σ Fr Rd Acc Good miss := by
  refine le_trans (ENNReal.ofReal_le_ofReal (Claims.Pr_mono fun s hs => ?_)) h
  obtain ⟨hA, hG, -, hm⟩ := hs
  exact ⟨hA, hG, hm⟩

/-- **The messages fixed where the coins agree, from the rounds a run reaches**: where on agreeing coins a run whose
messages are the check's at every round it reaches has all the verifier's messages, and a message the check reads
differs from the run's at a round the run reaches on plays at most `miss` likely, `MessagesFixedAgreed` holds at
`miss`. -/
theorem messagesFixedAgreed_of_reached {V : Vocab d α} {A : Live₂} {x : Msg} {θ : ByteArray} {F : Array ByteArray}
    {c : ByteArray} {σ : Verity.Game.Strategy (V.game θ F c)} {Fr : Frame V A x θ F c σ} {Rd : Reading A}
    {Acc Good : Verity.Exchange → Prop} {miss : ℝ≥0∞}
    (hR : ∀ (ω : TapeOf V θ F c σ) (r : Coins A x), Acc (Fr.out ω r) → Good (Fr.out ω r) →
      ReadAgrees A x Rd (Fr.out ω r) r →
      (∀ i < (Claims.run FlockDeploy.sha512 Fr.P A.L.V x ω r).msgs.length,
        Rd.msg (Fr.out ω r) i = msgAt (Claims.run FlockDeploy.sha512 Fr.P A.L.V x ω r) i) →
      A.L.V.k x ≤ (Claims.run FlockDeploy.sha512 Fr.P A.L.V x ω r).msgs.length)
    (h : ENNReal.ofReal (Claims.Pr fun s : TapeOf V θ F c σ × Coins A x =>
      Acc (Fr.out s.1 s.2) ∧ Good (Fr.out s.1 s.2) ∧
        ∃ i < (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2).msgs.length, Rd.msg (Fr.out s.1 s.2) i ≠
          msgAt (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2) i) ≤ miss) :
    MessagesFixedAgreed V A x θ F c σ Fr Rd Acc Good miss := by
  refine le_trans (ENNReal.ofReal_le_ofReal (Claims.Pr_mono fun s hs => ?_)) h
  obtain ⟨hA, hG, hC, i, hi, hne⟩ := hs
  refine ⟨hA, hG, ?_⟩
  by_contra hnone
  have hall : ∀ j < (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2).msgs.length,
      Rd.msg (Fr.out s.1 s.2) j = msgAt (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2) j :=
    fun j hj => Classical.not_not.1 fun hne' => hnone ⟨j, hj, hne'⟩
  exact hne (hall i (lt_of_lt_of_le hi (hR s.1 s.2 hA hG hC hall)))

/-- **The releases' agreement at a stronger `Good`**: `DevAgrees` at `Good` holds at any `Good'` that gives `Good`. -/
theorem devAgrees_mono {A : Live₂} {x : Msg} {Rd : Reading A} {Acc Good Good' : Verity.Exchange → Prop}
    {Ag : Verity.Exchange → Coins A x → Prop} (hG : ∀ o, Good' o → Good o) (h : DevAgrees A x Rd Acc Good Ag) :
    DevAgrees A x Rd Acc Good' Ag :=
  fun o hA hG' => h o hA (hG o hG')

/-- **A group's coins at a stronger `Good`**: `CoinsAgree` at `Good` holds at any `Good'` that gives `Good`. -/
theorem coinsAgree_mono {A : Live₂} {Rd : Reading A} {Acc Good Good' : Verity.Exchange → Prop} {g : Group}
    (hG : ∀ o, Good' o → Good o) (h : CoinsAgree A Rd Acc Good g) : CoinsAgree A Rd Acc Good' g :=
  fun o hA hG' => h o hA (hG o hG')

/-- **The messages at a stronger `Good`**: `MessagesFixedAgreed` at `Good` holds at any `Good'` that gives `Good`. -/
theorem messagesFixedAgreed_mono {V : Vocab d α} {A : Live₂} {x : Msg} {θ : ByteArray} {F : Array ByteArray}
    {c : ByteArray} {σ : Verity.Game.Strategy (V.game θ F c)} {Fr : Frame V A x θ F c σ} {Rd : Reading A}
    {Acc Good Good' : Verity.Exchange → Prop} {miss : ℝ≥0∞} (hG : ∀ o, Good' o → Good o)
    (h : MessagesFixedAgreed V A x θ F c σ Fr Rd Acc Good miss) :
    MessagesFixedAgreed V A x θ F c σ Fr Rd Acc Good' miss := by
  refine le_trans (ENNReal.ofReal_le_ofReal (Claims.Pr_mono fun s hs => ?_)) h
  obtain ⟨hA, hG', hrest⟩ := hs
  exact ⟨hA, hG _ hG', hrest⟩

/-- **The cut with what every play has**: a predicate `Good'` that holds of every play of the game after `θ`, `F` and
`c` joins the cut's `Good` at the same `terms`. -/
theorem cutBound_and {V : Vocab d α} {k : Verity.Registration.Composite} {θ : ByteArray} {F : Array ByteArray}
    {c : ByteArray} {σ : Verity.Game.Strategy (V.game θ F c)} {Good Good' : Verity.Exchange → Prop} {terms : ℝ≥0∞}
    (hR : ∀ o, Verity.Game.PlayReaches (V.game θ F c) σ o → Good' o) (h : CutBound V k θ F c σ Good terms) :
    CutBound V k θ F c σ (fun o => Good o ∧ Good' o) terms := by
  unfold CutBound at h ⊢
  refine le_trans (Verity.Game.prob_mono_reaches (p := fun o => V.Verdict k .sampled o ∧ ¬ (Good o ∧ Good' o))
    (q := fun o => V.Verdict k .sampled o ∧ ¬ Good o) (V.game θ F c) σ fun o hr ho => ?_) h
  exact ⟨ho.1, fun hg => ho.2 ⟨hg, hR o hr⟩⟩

/-- **The projection's bound from the coins of accepted plays**: the wiring, the coin map's law at an agreement `Ag`,
the messages fixed where the coins agree, the cut, the accept on the plays `Acc` and `Good` hold of, and the statement
read give the projection's bound at `hide + miss + terms`, where on every frame play the check accepts, whose log it
read is the device's and where `Ag` holds, each coin it reads at a read round agrees with the run's coin there. -/
theorem projection_core (V : Vocab d α) (k : Verity.Registration.Composite) (A : Live₂) (x : Msg) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ)
    (Rd : Reading A) (Acc Good : Verity.Exchange → Prop) (Ag : Verity.Exchange → Coins A x → Prop)
    (hide miss terms : ℝ≥0∞)
    (hAcc : ∀ o, o.params = θ → V.Verdict k .sampled o → Acc o)
    (hLaw : CoinMapLaw V A x θ F c σ Fr Ag hide)
    (hCoins : ∀ (ω : TapeOf V θ F c σ) (r : Coins A x), Acc (Fr.out ω r) → Good (Fr.out ω r) →
      Ag (Fr.out ω r) r → ∀ i : Fin (A.L.V.k x), Rd.reads i.val →
        ∀ c', Rd.coin (Fr.out ω r) i.val = some c' → Rd.agree (Fr.out ω r) i.val c' (r i))
    (hMsg : MessagesFixedAgreed V A x θ F c σ Fr Rd Acc Good miss) (hCut : CutBound V k θ F c σ Good terms)
    (hRun : AcceptsRun A x Rd (fun o => Acc o ∧ Good o)) (hAt : StmtAt V A x θ F c σ Fr Rd Acc) :
    ProjectionBound V k A x θ F c σ Fr (hide + miss + terms) := by
  have hsplit : Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ ≤
      Verity.Game.prob (fun o => Acc o ∧ Good o) (V.game θ F c) σ +
        Verity.Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ Good o) (V.game θ F c) σ := by
    refine (Verity.Game.prob_mono_reaches (p := fun o => V.Verdict k .sampled o)
      (q := fun o => (Acc o ∧ Good o) ∨ (V.Verdict k .sampled o ∧ ¬ Good o)) (V.game θ F c) σ
      (fun o hr ho => ?_)).trans
      (Verity.Game.prob_or_le (fun o => Acc o ∧ Good o) (fun o => V.Verdict k .sampled o ∧ ¬ Good o)
        (V.game θ F c) σ)
    by_cases hg : Good o
    · exact Or.inl ⟨hAcc o (play_params V hr) ho, hg⟩
    · exact Or.inr ⟨ho, hg⟩
  have hgood : Claims.Pr (fun s : TapeOf V θ F c σ × Coins A x =>
      (Acc (Fr.out s.1 s.2) ∧ Good (Fr.out s.1 s.2)) ∧ Ag (Fr.out s.1 s.2) s.2) ≤
      Claims.Pr (fun s : TapeOf V θ F c σ × Coins A x =>
          (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2).out = .acc) +
        Claims.Pr (fun s : TapeOf V θ F c σ × Coins A x =>
          Acc (Fr.out s.1 s.2) ∧ Good (Fr.out s.1 s.2) ∧ ReadAgrees A x Rd (Fr.out s.1 s.2) s.2 ∧
            ∃ i < A.L.V.k x, Rd.msg (Fr.out s.1 s.2) i ≠
              msgAt (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2) i) := by
    refine Claims.Sampling.Pr_or_le
      (q := fun s : TapeOf V θ F c σ × Coins A x =>
        (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2).out = .acc)
      (r := fun s : TapeOf V θ F c σ × Coins A x =>
        Acc (Fr.out s.1 s.2) ∧ Good (Fr.out s.1 s.2) ∧ ReadAgrees A x Rd (Fr.out s.1 s.2) s.2 ∧
          ∃ i < A.L.V.k x, Rd.msg (Fr.out s.1 s.2) i ≠
            msgAt (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2) i)
      (fun s hs => ?_) le_rfl le_rfl
    obtain ⟨⟨ha, hg⟩, hag⟩ := hs
    by_cases hm : ∃ i < A.L.V.k x, Rd.msg (Fr.out s.1 s.2) i ≠
        msgAt (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2) i
    · exact Or.inr ⟨ha, hg, hCoins s.1 s.2 ha hg hag, hm⟩
    exact Or.inl (hRun _ ⟨ha, hg⟩ (hAt s.1 s.2 ha) _ Fr.P s.1 s.2 (hCoins s.1 s.2 ha hg hag)
      (fun i hi => Classical.not_not.1 fun hne => hm ⟨i, hi, hne⟩))
  calc Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ
      ≤ Verity.Game.prob (fun o => Acc o ∧ Good o) (V.game θ F c) σ +
          Verity.Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ Good o) (V.game θ F c) σ := hsplit
    _ ≤ (ENNReal.ofReal (Claims.Pr fun s : TapeOf V θ F c σ × Coins A x =>
            (Acc (Fr.out s.1 s.2) ∧ Good (Fr.out s.1 s.2)) ∧ Ag (Fr.out s.1 s.2) s.2) + hide) + terms :=
          add_le_add (hLaw fun o => Acc o ∧ Good o) hCut
    _ ≤ (ENNReal.ofReal (Claims.Pr fun s : TapeOf V θ F c σ × Coins A x =>
            (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2).out = .acc) + miss + hide) + terms :=
          add_le_add (add_le_add ((ENNReal.ofReal_le_ofReal hgood).trans
            (ENNReal.ofReal_add_le.trans (add_le_add le_rfl hMsg))) le_rfl) le_rfl
    _ = ENNReal.ofReal (Claims.Pr fun s : TapeOf V θ F c σ × Coins A x =>
            (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2).out = .acc) + (hide + miss + terms) := by
          ring

/-- **The projection from its pieces, the messages fixed where the coins agree**: `ProjectionOf`'s premises, with
`MessagesFixedAgreed` in the place of `MessagesFixed`, give the projection's bound at `hide + miss + terms`. -/
theorem projection_agreed (V : Vocab d α) (k : Verity.Registration.Composite) (A : Live₂) (x : Msg) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ)
    (Rd : Reading A) (Acc Good : Verity.Exchange → Prop) (Ag : Verity.Exchange → Coins A x → Prop)
    (hide miss terms : ℝ≥0∞)
    (hAcc : ∀ o, o.params = θ → V.Verdict k .sampled o → Acc o)
    (hLaw : CoinMapLaw V A x θ F c σ Fr Ag hide) (hDev : DevAgrees A x Rd Acc Good Ag)
    (hFirst : CoinsAgree A Rd Acc Good .first) (hSecond : CoinsAgree A Rd Acc Good .second)
    (hSessions : CoinsAgree A Rd Acc Good .sessions) (hStmt : StmtReadsNone A Rd)
    (hMsg : MessagesFixedAgreed V A x θ F c σ Fr Rd Acc Good miss) (hCut : CutBound V k θ F c σ Good terms)
    (hRun : AcceptsRun A x Rd (fun o => Acc o ∧ Good o)) (hAt : StmtAt V A x θ F c σ Fr Rd Acc) :
    ProjectionBound V k A x θ F c σ Fr (hide + miss + terms) := by
  refine projection_core V k A x θ F c σ Fr Rd Acc Good Ag hide miss terms hAcc hLaw
    (fun ω r ha hg hag i hi c' hc => ?_) hMsg hCut hRun hAt
  have hd : ∀ e, Rd.dev (Fr.out ω r).log i.val = some e → Rd.agree (Fr.out ω r) i.val e (r i) :=
    fun e he => hDev _ ha hg r hag i hi e he
  cases hgi : groupOf A i.val with
  | first => exact hFirst _ ha hg _ hi hgi c' hc (r i) hd
  | second => exact hSecond _ ha hg _ hi hgi c' hc (r i) hd
  | stmt => exact absurd hgi (hStmt _ hi)
  | sessions => exact hSessions _ ha hg _ hi hgi c' hc (r i) hd

/-- **The projection from its pieces**: `ProjectionOf` holds at every vocabulary, composite, session of record,
statement, committed files, commitment and strategy. -/
theorem projection_of (V : Vocab d α) (k : Verity.Registration.Composite) (A : Live₂) (x : Msg) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) :
    ProjectionOf V k A x θ F c σ := by
  intro Fr Rd Acc Good Ag hide miss terms hAcc hLaw hDev hFirst hSecond hSessions hStmt hMsg hCut hRun hAt
  exact projection_agreed V k A x θ F c σ Fr Rd Acc Good Ag hide miss terms hAcc hLaw hDev hFirst hSecond hSessions
    hStmt (messagesFixedAgreed_of_fixed hMsg) hCut (fun o h => hRun o h.1) hAt

end Verity.SampledTwo
