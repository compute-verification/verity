module

public import Proofs.Properties.Verity.SampledTwo.Projection
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
   where the form's check does (`Acc`, the composite's wiring's), so a sampled accept is an accepted play whose log the check read is the device's (`Good`), or a sampled accept with `¬ Good`, which
   the cut bounds by `terms` (`CutBound`).
2. The first is moved to the frame's plays, at a uniform tape and uniform run coins, by the coin map's law's first
   clause, at the cost `hide`.
3. On a frame play with `Acc ∧ Good`, either a message the check reads differs from the frame's prover's, which the
   messages' piece bounds by `miss` (`MessagesFixed`), or the run at the same tape and coins accepts: each read
   round's coin read agrees with the run's (`projection_core`'s one premise about coins), the check reads the frame's
   statement (`StmtAt`), and the accept piece (`AcceptsRun`) gives the run's accept.

Step 3's coins: a coin the check reads at a read round is the device's release there, by the coins' piece of its group
(`CoinsRead`; the statements' round reads none, `StmtReadsNone`), and a release the play's log holds agrees with the
run's coin, by the coin map's second clause.

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

/-- **The projection's bound from the coins of accepted plays**: the wiring, the coin map's law's first clause, the
messages, the cut, the accept and the statement read give the projection's bound at `hide + miss + terms`, where on
every frame play the check accepts and whose log it read is the device's, each coin it reads at a read round agrees
with the run's coin there. -/
theorem projection_core (V : Vocab d α) (k : Verity.Registration.Composite) (A : Live₂) (x : Msg) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ)
    (Rd : Reading A) (Acc Good : Verity.Exchange → Prop) (hide miss terms : ℝ≥0∞)
    (hAcc : ∀ o, o.params = θ → V.Verdict k .sampled o → Acc o)
    (hLaw : ∀ E : Verity.Exchange → Prop, Verity.Game.prob E (V.game θ F c) σ ≤
      ENNReal.ofReal (Claims.Pr fun s : TapeOf V θ F c σ × Coins A x => E (Fr.out s.1 s.2)) + hide)
    (hCoins : ∀ (ω : TapeOf V θ F c σ) (r : Coins A x), Acc (Fr.out ω r) → Good (Fr.out ω r) →
      ∀ i : Fin (A.L.V.k x), Rd.reads i.val → ∀ c', Rd.coin (Fr.out ω r) i.val = some c' →
        Rd.agree (Fr.out ω r) i.val c' (r i))
    (hMsg : MessagesFixed V A x θ F c σ Fr Rd Acc Good miss) (hCut : CutBound V k θ F c σ Good terms)
    (hRun : AcceptsRun A x Rd Acc) (hAt : StmtAt V A x θ F c σ Fr Rd Acc) :
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
  have hgood : Claims.Pr (fun s : TapeOf V θ F c σ × Coins A x => Acc (Fr.out s.1 s.2) ∧ Good (Fr.out s.1 s.2)) ≤
      Claims.Pr (fun s : TapeOf V θ F c σ × Coins A x =>
          (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2).out = .acc) +
        Claims.Pr (fun s : TapeOf V θ F c σ × Coins A x =>
          Acc (Fr.out s.1 s.2) ∧ Good (Fr.out s.1 s.2) ∧
            ∃ i < A.L.V.k x, Rd.msg (Fr.out s.1 s.2) i ≠
              msgAt (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2) i) := by
    refine Claims.Sampling.Pr_or_le
      (q := fun s : TapeOf V θ F c σ × Coins A x =>
        (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2).out = .acc)
      (r := fun s : TapeOf V θ F c σ × Coins A x =>
        Acc (Fr.out s.1 s.2) ∧ Good (Fr.out s.1 s.2) ∧
          ∃ i < A.L.V.k x, Rd.msg (Fr.out s.1 s.2) i ≠
            msgAt (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2) i)
      (fun s hs => ?_) le_rfl le_rfl
    obtain ⟨ha, hg⟩ := hs
    by_cases hm : ∃ i < A.L.V.k x, Rd.msg (Fr.out s.1 s.2) i ≠
        msgAt (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2) i
    · exact Or.inr ⟨ha, hg, hm⟩
    exact Or.inl (hRun _ ha (hAt s.1 s.2 ha) _ Fr.P s.1 s.2 (hCoins s.1 s.2 ha hg)
      (fun i hi => Classical.not_not.1 fun hne => hm ⟨i, hi, hne⟩))
  calc Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ
      ≤ Verity.Game.prob (fun o => Acc o ∧ Good o) (V.game θ F c) σ +
          Verity.Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ Good o) (V.game θ F c) σ := hsplit
    _ ≤ (ENNReal.ofReal (Claims.Pr fun s : TapeOf V θ F c σ × Coins A x =>
            Acc (Fr.out s.1 s.2) ∧ Good (Fr.out s.1 s.2)) + hide) + terms :=
          add_le_add (hLaw fun o => Acc o ∧ Good o) hCut
    _ ≤ (ENNReal.ofReal (Claims.Pr fun s : TapeOf V θ F c σ × Coins A x =>
            (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2).out = .acc) + miss + hide) + terms :=
          add_le_add (add_le_add ((ENNReal.ofReal_le_ofReal hgood).trans
            (ENNReal.ofReal_add_le.trans (add_le_add le_rfl hMsg))) le_rfl) le_rfl
    _ = ENNReal.ofReal (Claims.Pr fun s : TapeOf V θ F c σ × Coins A x =>
            (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2).out = .acc) + (hide + miss + terms) := by
          ring

/-- **The projection from its pieces**: `ProjectionOf` holds at every vocabulary, composite, session of record,
statement, committed files, commitment and strategy. -/
theorem projection_of (V : Vocab d α) (k : Verity.Registration.Composite) (A : Live₂) (x : Msg) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) :
    ProjectionOf V k A x θ F c σ := by
  intro Fr Rd Acc Good hide miss terms hAcc hLaw hFirst hSecond hSessions hStmt hMsg hCut hRun hAt
  refine projection_core V k A x θ F c σ Fr Rd Acc Good hide miss terms hAcc hLaw.1
    (fun ω r ha hg i hi c' hc => hLaw.2 ω r i hi c' ?_) hMsg hCut hRun hAt
  rw [← hc]
  cases hgi : groupOf A i.val with
  | first => exact (hFirst _ ha hg _ hi hgi).symm
  | second => exact (hSecond _ ha hg _ hi hgi).symm
  | stmt => exact absurd hgi (hStmt _ hi)
  | sessions => exact (hSessions _ ha hg _ hi hgi).symm

end Verity.SampledTwo
