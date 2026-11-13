module

public import Proofs.Properties.Verity.SampledTwo.JoinsAt
public import Proofs.Properties.Verity.SampledTwo.Binding
public import Proofs.Properties.Verity.SampledTwo.CouplingOf

/-!
# The two-stage leg's coupling at the form's accept, from the coin map's law

`coupling_of` (`CouplingOf.lean`) gives `CouplingTargetTwo` from a projection's bound and a majority's at each
commitment and strategy. Here the projection's is the join's (`joinedOfLaw`, at the join's frame `joinFrame`), and the
majority's is MAJORITY's at that frame (`majorityBound_opens`), with the two bounds' sum as the coupling's terms
(`joinBound`, `bindBound`).

**One premise a commitment and strategy.** `joinedCouplingOfLaw` takes the coin map's law (`CoinMap.CoinMapHolds`) and,
at each commitment `c` and strategy `σ`, one structure, `CouplesAt`: the join's premise (`JoinsAt`), what the majority's
piece owes and the coupling's two clauses (the scope and the published `δ`). `joinedCoupling` (`JoinedForm.lean`) is
it at COINMAP's proof of the sampling design. The statement, the files, the stages' families, the deployment and the readings are
fixed across commitments and strategies except the statement `x`, which is the commitment's; the majority's statement
data (`s`, `Pt`, `dd`) and its witness's reading and openings (`readX`, `openOf`, `dColl`) are each commitment's and
strategy's.

**The openings are the witness's own** (`opens`, `OpensBy`). Some `openOf` meets `OpensBy` at every `readX` by choice
(`exists_opensBy`), and the bound under `ecr/sha-512` (`collisionResistant`) prices only an `openOf` an algorithm
computes. So the instance's `openOf` reads each witness's openings, `dColl` counts the SHA-512 calls that compute them
and `hashedAll`, and `exists_opensBy` discharges no `opens` field.

**Not established**: every field of `CouplesAt` is a premise. Those of `JoinsAt` are as `JoinsAt.lean` lists them; the
majority's (`extract`, `parse`, `partition`, `two`, `units`, `caps`, `opens`, `collisionResistant`) are MAJORITY's and the
instance's; `scope` and `delta` are the coupling's at the deployment. Every one is over the form's accept or the
period's game at the deployment, so the theorem says nothing about a run of record until the form accepts a play.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The projection's bound the join gives**: the coin map's hiding term, the messages' miss and the cut's terms at
the deployed two-session history. -/
noncomputable abbrev joinBound (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (C : CaseOf) (Ms : SessionReading) (tab : ℕ → String)
    (ℓ₁ ℓ₂ : String) (Dg : Verity.Exchange → ℕ → Option Msg) (q : ℕ) (P : Sessions.Deployment)
    (fs₁ fs₂ : ByteArray → List (String × ℕ)) (cut : List Verity.Release → ℕ) : ℝ≥0∞ :=
  CoinMap.gapOf V θ F c σ (joinLocate A tab ℓ₁ ℓ₂) (A.L.V.k x) +
    joinMiss V A x θ F c σ C (joinSessions A C Ms tab ℓ₁ ℓ₂) (joinLocate A tab ℓ₁ ℓ₂) Dg q +
    terms V (Sessions.devOf P fs₁ fs₂ cut) θ F c σ

/-- **The majority's bound at a frame** (`majorityBound_opens`): `√(min(W, 2n) · E[cost] / 2^256)` at the binding
finder that reads the openings `openOf` and spends `dColl`. -/
noncomputable abbrev bindBound (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ)
    (openOf : Claims.Sampling.Wit → Fin (V.circuitOf θ F).N → Option Hash.RowLeaf.Opening) (dColl : ℕ) : ℝ≥0∞ :=
  ENNReal.ofReal (Real.sqrt (min ((∑ g, (V.circuitOf θ F).w g : ℕ) : ℝ) (2 * V.units θ F) *
    (Ex (fun p => ((bindFinder V A x θ F c σ Fr (collOfOpen V θ F c openOf) dColl).cost p : ℝ)) /
      2 ^ (256 : ℝ))))

/-- **What the coupling takes at a commitment and strategy**: the join's premise, the majority's at the join's frame,
and the coupling's scope and `δ` clauses. -/
structure CouplesAt (V : Vocab d α) (k : Verity.Registration.Composite) (A : Live₂) (x : Msg) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (C : CaseOf)
    (Ms : SessionReading) (tab : ℕ → String) (ℓ₁ ℓ₂ : String) (Dg : Verity.Exchange → ℕ → Option Msg) (q : ℕ)
    (P : Sessions.Deployment) (fs₁ fs₂ : ByteArray → List (String × ℕ)) (cut : List Verity.Release → ℕ)
    (P' : Verity.Exchange → ℕ → Claims.Prover Unit) (s : Claims.Sampling.AuditStatement)
    (Pt : Claims.Sampling.Partition) (dd : Claims.Sampling.Design.Two)
    (readX : Claims.Sampling.Wit → Verity.Circuit.Trace (V.circuitOf θ F))
    (openOf : Claims.Sampling.Wit → Fin (V.circuitOf θ F).N → Option Hash.RowLeaf.Opening) (dColl : ℕ) : Prop where
  /-- The join's premise at the form's accept. -/
  joins : JoinsAt V k A x θ F c σ C Ms tab ℓ₁ ℓ₂ Dg P fs₁ fs₂ cut P'
  /-- The statement is in the session's scope, or the sampled verdict never accepts. -/
  scope : A.L.Scope x ∨ Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ = 0
  /-- The vocabulary's transcript is the majority at the join's frame. -/
  extract : V.extract θ F c σ =
    majority V A x θ F c σ (joinFrame V A x θ F c σ C (joinSessions A C Ms tab ℓ₁ ℓ₂) (joinLocate A tab ℓ₁ ℓ₂) Dg)
      readX
  /-- The statement's audit statement, partition and two-stage design. -/
  parse : A.L.parse x = some s
  partition : A.L.partition s = some Pt
  two : A.L.two s = some dd
  /-- The units are the replay blocks. -/
  units : UnitsAreReplay V θ F c FlockDeploy.sha512 A.L.commit s Pt dd readX
  /-- The design caps the claim's share `ε · n` at `δ`. -/
  caps : CapsAt dd Pt ((V.certOf θ).ε * V.units θ F) A.δ
  /-- The witness carries its openings; `openOf` reads them from the witness, and `exists_opensBy` discharges no
  instance of this field. -/
  opens : OpensBy V θ F c readX openOf
  /-- `ecr/sha-512` at the binding finder that reads `openOf` and spends `dColl`. -/
  collisionResistant : Claims.ExpectedCR FlockDeploy.sha512
    (bindFinder V A x θ F c σ (joinFrame V A x θ F c σ C (joinSessions A C Ms tab ℓ₁ ℓ₂) (joinLocate A tab ℓ₁ ℓ₂) Dg)
      (collOfOpen V θ F c openOf) dColl)
  /-- The published `δ` covers the frame's prover's error and both bounds. -/
  delta : V.finderCR .sampled θ F c σ → V.premises .sampled θ F c σ →
    ENNReal.ofReal (A.errAt x
        (joinFrame V A x θ F c σ C (joinSessions A C Ms tab ℓ₁ ℓ₂) (joinLocate A tab ℓ₁ ℓ₂) Dg).P) +
      (joinBound V A x θ F c σ C Ms tab ℓ₁ ℓ₂ Dg q P fs₁ fs₂ cut +
        bindBound V A x θ F c σ
          (joinFrame V A x θ F c σ C (joinSessions A C Ms tab ℓ₁ ℓ₂) (joinLocate A tab ℓ₁ ℓ₂) Dg) openOf dColl) ≤
      V.delta θ .sampled (V.cost θ F c σ)

/-- **The two-stage leg's coupling at the form's accept, from the coin map's law**: `coupling_of` at the join's frame,
with the join's projection bound (`joinedOfLaw`) and MAJORITY's majority bound there (`majorityBound_opens`), from the
coin map's law and one `CouplesAt` at each commitment and strategy. -/
theorem joinedCouplingOfLaw (V : Vocab d α) (k : Verity.Registration.Composite) (A : Live₂) (θ : ByteArray)
    (F : Array ByteArray) (x : (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) → Msg) (C : CaseOf)
    (Ms : SessionReading) (tab : ℕ → String) (ℓ₁ ℓ₂ : String) (Dg : Verity.Exchange → ℕ → Option Msg) (q : ℕ)
    (P : Sessions.Deployment) (fs₁ fs₂ : ByteArray → List (String × ℕ)) (cut : List Verity.Release → ℕ)
    (P' : Verity.Exchange → ℕ → Claims.Prover Unit)
    (s : (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) → Claims.Sampling.AuditStatement)
    (Pt : (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) → Claims.Sampling.Partition)
    (dd : (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) → Claims.Sampling.Design.Two)
    (readX : (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) →
      Claims.Sampling.Wit → Verity.Circuit.Trace (V.circuitOf θ F))
    (openOf : (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) →
      Claims.Sampling.Wit → Fin (V.circuitOf θ F).N → Option Hash.RowLeaf.Opening)
    (dColl : (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) → ℕ)
    (hHolds : CoinMap.CoinMapHolds (d := d) (α := α))
    (h : ∀ c σ, CouplesAt V k A (x c σ) θ F c σ C Ms tab ℓ₁ ℓ₂ Dg q P fs₁ fs₂ cut P' (s c σ) (Pt c σ) (dd c σ)
      (readX c σ) (openOf c σ) (dColl c σ)) :
    CouplingTargetTwo V k A θ F fun c σ =>
      joinBound V A (x c σ) θ F c σ C Ms tab ℓ₁ ℓ₂ Dg q P fs₁ fs₂ cut +
        bindBound V A (x c σ) θ F c σ
          (joinFrame V A (x c σ) θ F c σ C (joinSessions A C Ms tab ℓ₁ ℓ₂) (joinLocate A tab ℓ₁ ℓ₂) Dg)
          (openOf c σ) (dColl c σ) :=
  coupling_of V k A θ F x
    (fun c σ => joinFrame V A (x c σ) θ F c σ C (joinSessions A C Ms tab ℓ₁ ℓ₂) (joinLocate A tab ℓ₁ ℓ₂) Dg)
    (fun c σ => V.extract θ F c σ)
    (fun c σ => joinBound V A (x c σ) θ F c σ C Ms tab ℓ₁ ℓ₂ Dg q P fs₁ fs₂ cut)
    (fun c σ => bindBound V A (x c σ) θ F c σ
      (joinFrame V A (x c σ) θ F c σ C (joinSessions A C Ms tab ℓ₁ ℓ₂) (joinLocate A tab ℓ₁ ℓ₂) Dg)
      (openOf c σ) (dColl c σ))
    (fun c σ => (h c σ).scope)
    (fun c σ => joinedOfLaw V k A (x c σ) θ F c σ C Ms tab ℓ₁ ℓ₂ Dg q P fs₁ fs₂ cut P' hHolds (h c σ).joins)
    (fun c σ => majorityBound_opens V A (x c σ) θ F c σ
      (joinFrame V A (x c σ) θ F c σ C (joinSessions A C Ms tab ℓ₁ ℓ₂) (joinLocate A tab ℓ₁ ℓ₂) Dg)
      (s c σ) (Pt c σ) (dd c σ) (readX c σ) (openOf c σ) (dColl c σ) (h c σ).extract (h c σ).parse
      (h c σ).partition (h c σ).two (h c σ).units (h c σ).caps (h c σ).opens (h c σ).collisionResistant)
    (fun _ _ => rfl)
    (fun c σ => (h c σ).delta)

end Verity.SampledTwo
