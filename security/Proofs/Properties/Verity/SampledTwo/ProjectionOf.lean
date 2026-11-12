module

public import Proofs.Properties.Verity.SampledTwo.Projection
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.KS

/-!
# The two-stage leg's projection, assembled

`projection_of` proves `ProjectionOf`: the projection's named pieces give its bound. The argument is probability
bookkeeping over the pieces' `Prop`s, and reads nothing of the form's check but what they state.

1. The sampled verdict accepts only where the form's check does (`Acc`, the composite's wiring's), so a sampled accept
   is an accepted play whose log the check read is the device's (`Good`), or a sampled accept with `¬ Good`, which
   the cut bounds by `terms` (`CutBound`).
2. The first is moved to the frame's plays, at a uniform tape and uniform run coins, by the coin map's law's first
   clause, at the cost `hide`.
3. On a frame play with `Acc ∧ Good`, either a message the check reads differs from the frame's prover's, which the
   messages' piece bounds by `miss` (`MessagesFixed`), or the run at the same tape and coins accepts: each read
   round's coin is the run's (`projection_core`'s one premise about coins), the check reads the frame's statement
   (`StmtAt`), and the accept piece (`AcceptsRun`) gives the run's accept.

Step 3's coins come from the coins' pieces of each group (`CoinsRead`; the statements' round reads none,
`StmtReadsNone`) and the coin map's second clause. `CoinMapLaw`'s second clause holds at every tape, so it asks a
release for every read round of every frame play, which a play that ends before one (a strategy that never opens the
second stage's session) does not have. `CoinMapLawReached` asks it only at the releases a play has, and
`CoinsReadSome` adds to `CoinsRead` that an accepted play's check read a coin at each read round; `projection_of_reached`
assembles the projection from those two in their place.

Where `CutBound` is instantiated at the deployed composite it meets an open hole of the device's history: a strategy
whose evidence carries a genuine challenger signature from an earlier period makes the forger's term `⊤`, because the
history holds one play's signatures. The assembly takes `terms` as given and neither closes nor widens that hole.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The projection's bound from the coins of accepted plays**: the wiring, the coin map's law's first clause, the
messages, the cut, the accept and the statement read give the projection's bound at `hide + miss + terms`, where on
every frame play the check accepts and whose log it read is the device's, the coin it reads at each read round is the
run's coin there. -/
theorem projection_core (V : Vocab d α) (k : Verity.Registration.Composite) (A : Live₂) (x : Msg) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ)
    (Rd : Reading A) (Acc Good : Verity.Exchange → Prop) (hide miss terms : ℝ≥0∞)
    (hAcc : ∀ o, V.Verdict k .sampled o → Acc o)
    (hLaw : ∀ E : Verity.Exchange → Prop, Verity.Game.prob E (V.game θ F c) σ ≤
      ENNReal.ofReal (Claims.Pr fun s : TapeOf V θ F c σ × Coins A x => E (Fr.out s.1 s.2)) + hide)
    (hCoins : ∀ (ω : TapeOf V θ F c σ) (r : Coins A x), Acc (Fr.out ω r) → Good (Fr.out ω r) →
      ∀ i : Fin (A.L.V.k x), Rd.reads i.val → Rd.coin (Fr.out ω r) i.val = some (r i))
    (hMsg : MessagesFixed V A x θ F c σ Fr Rd Acc Good miss) (hCut : CutBound V k θ F c σ Good terms)
    (hRun : AcceptsRun A x Rd Acc) (hAt : StmtAt V A x θ F c σ Fr Rd Acc) :
    ProjectionBound V k A x θ F c σ Fr (hide + miss + terms) := by
  have hsplit : Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ ≤
      Verity.Game.prob (fun o => Acc o ∧ Good o) (V.game θ F c) σ +
        Verity.Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ Good o) (V.game θ F c) σ := by
    refine (Verity.Game.prob_mono (p := fun o => V.Verdict k .sampled o)
      (q := fun o => (Acc o ∧ Good o) ∨ (V.Verdict k .sampled o ∧ ¬ Good o)) (fun o ho => ?_)
      (V.game θ F c) σ).trans
      (Verity.Game.prob_or_le (fun o => Acc o ∧ Good o) (fun o => V.Verdict k .sampled o ∧ ¬ Good o)
        (V.game θ F c) σ)
    by_cases hg : Good o
    · exact Or.inl ⟨hAcc o ho, hg⟩
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
    (fun ω r ha hg i hi => ?_) hMsg hCut hRun hAt
  rw [← hLaw.2 ω r i hi]
  cases hgi : groupOf A i.val with
  | first => exact hFirst _ ha hg _ hi hgi
  | second => exact hSecond _ ha hg _ hi hgi
  | stmt => exact absurd hgi (hStmt _ hi)
  | sessions => exact hSessions _ ha hg _ hi hgi

/-- **The coin map's law, at the releases a play has**: the period's play is distributed as the frame's play at a
uniform tape and uniform run coins, up to `hide`; and wherever the device released a read round's coin in a frame
play, that coin is the run's coin there. -/
def CoinMapLawReached (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ) (Rd : Reading A) (hide : ℝ≥0∞) : Prop :=
  (∀ E : Verity.Exchange → Prop, Verity.Game.prob E (V.game θ F c) σ ≤
      ENNReal.ofReal (Claims.Pr fun s : TapeOf V θ F c σ × Coins A x => E (Fr.out s.1 s.2)) + hide) ∧
    ∀ (ω : TapeOf V θ F c σ) (r : Coins A x) (i : Fin (A.L.V.k x)), Rd.reads i.val →
      ∀ v, Rd.dev (Fr.out ω r).log i.val = some v → v = r i

/-- **The coins the check reads at group `g`, read**: on a play the form's check accepts whose log the check read is the
device's, the check read a coin at each read round of group `g`, and it is the device's release for that round. -/
def CoinsReadSome (A : Live₂) (Rd : Reading A) (Acc Good : Verity.Exchange → Prop) (g : Group) : Prop :=
  ∀ o : Verity.Exchange, Acc o → Good o → ∀ i, Rd.reads i → groupOf A i = g →
    ∃ v, Rd.coin o i = some v ∧ Rd.dev o.log i = some v

/-- **The projection from its pieces, at the releases a play has**: `ProjectionOf` with `CoinMapLawReached` in place
of `CoinMapLaw` and `CoinsReadSome` in place of `CoinsRead`. -/
def ProjectionOfReached (V : Vocab d α) (k : Verity.Registration.Composite) (A : Live₂) (x : Msg) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) : Prop :=
  ∀ (Fr : Frame V A x θ F c σ) (Rd : Reading A) (Acc Good : Verity.Exchange → Prop) (hide miss terms : ℝ≥0∞),
    (∀ o, V.Verdict k .sampled o → Acc o) →
    CoinMapLawReached V A x θ F c σ Fr Rd hide → CoinsReadSome A Rd Acc Good .first →
      CoinsReadSome A Rd Acc Good .second → CoinsReadSome A Rd Acc Good .sessions → StmtReadsNone A Rd →
      MessagesFixed V A x θ F c σ Fr Rd Acc Good miss → CutBound V k θ F c σ Good terms → AcceptsRun A x Rd Acc →
      StmtAt V A x θ F c σ Fr Rd Acc → ProjectionBound V k A x θ F c σ Fr (hide + miss + terms)

/-- **The projection from its pieces, at the releases a play has**: `ProjectionOfReached` holds at every vocabulary,
composite, session of record, statement, committed files, commitment and strategy. -/
theorem projection_of_reached (V : Vocab d α) (k : Verity.Registration.Composite) (A : Live₂) (x : Msg)
    (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) :
    ProjectionOfReached V k A x θ F c σ := by
  intro Fr Rd Acc Good hide miss terms hAcc hLaw hFirst hSecond hSessions hStmt hMsg hCut hRun hAt
  refine projection_core V k A x θ F c σ Fr Rd Acc Good hide miss terms hAcc hLaw.1
    (fun ω r ha hg i hi => ?_) hMsg hCut hRun hAt
  have hread : ∃ v, Rd.coin (Fr.out ω r) i.val = some v ∧ Rd.dev (Fr.out ω r).log i.val = some v := by
    cases hgi : groupOf A i.val with
    | first => exact hFirst _ ha hg _ hi hgi
    | second => exact hSecond _ ha hg _ hi hgi
    | stmt => exact absurd hgi (hStmt _ hi)
    | sessions => exact hSessions _ ha hg _ hi hgi
  obtain ⟨v, hc, hd⟩ := hread
  rw [hc, hLaw.2 ω r i hi v hd]

end Verity.SampledTwo
