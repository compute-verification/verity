module

public import Proofs.Properties.Verity.SampledTwo.WiredScope

/-!
# `CouplesAt` at the wired statement

`couplesAt_wired`: an instance of `CouplesAt` at a commitment and strategy, at the statement `JoinsAt` fixes (`stmtX`)
and at the audit statement, partition and design the session's verifier reads of it. Its `scope` is a lemma of
`JoinsAt` and `TwoHolds` (`scope_wired`), and `parse`, `partition` and `two` are what the verifier computes
(`WiredParse.lean`). The statement decodes at every commitment (`decode_of_stmtOf`), and the design takes one premise:

* `SpotFlock.RunsAs.Round2.TwoHolds A.h`: the held copies' design evaluates, a lemma wherever a play whose blobs don't
  collide with the registered files is accepted (`twoHolds_of_formAccepts` at `heldOfAt_goodCase`).

`units` is the premise `UnitsWired`: at the audit statement, partition and design the verifier reads, the vocabulary's
units are the replay blocks under `readX`. `joins`, `extract`, `caps`, `opens`, `collisionResistant`, `blobCR` and
`delta` are taken as they are; `caps`, like `units`, at what the verifier reads.

## What this does not show

`TwoHolds` where no play with no such collision is accepted, and `UnitsWired`: each is a hypothesis.
`JoinsAt` at a commitment whose bytes don't decode (`stmtOf c = none`), where its `stmtX` fails, so the instance is at
the commitments that decode.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The vocabulary's units are the replay blocks at what the verifier reads**: at the audit statement, partition
and design the session's verifier reads of `x`, `UnitsAreReplay` under the reading `readX`. -/
def UnitsWired (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (readX : Claims.Sampling.Wit → Verity.Circuit.Trace (V.circuitOf θ F)) : Prop :=
  ∀ s Pt dd, A.L.parse x = some s → A.L.partition s = some Pt → A.L.two s = some dd →
    UnitsAreReplay V θ F c FlockDeploy.sha512 A.L.commit s Pt dd readX

/-- **`CouplesAt` at the wired statement**: at the audit statement, partition and design the session's verifier reads
of it, from the join's premise, `TwoHolds` and `UnitsWired`, with the majority's and the coupling's other fields as
they are. -/
theorem couplesAt_wired {V : Vocab d α} {k : Verity.Registration.Composite} {A : Live₂} {x : Msg} {θ : ByteArray}
    {F : Array ByteArray} {c : ByteArray} {σ : Verity.Game.Strategy (V.game θ F c)} {C : CaseOf}
    {Ms : SessionReading} {tab : ℕ → String} {ℓ₁ ℓ₂ : String} {Dg : Verity.Exchange → ℕ → Option Msg} {q : ℕ}
    {P : Sessions.Deployment} {fs₁ fs₂ : ByteArray → List (String × ℕ)} {cut : List Verity.Release → ℕ}
    {P' : Verity.Exchange → ℕ → Claims.Prover Unit}
    {readX : Claims.Sampling.Wit → Verity.Circuit.Trace (V.circuitOf θ F)}
    {openOf : Claims.Sampling.Wit → Fin (V.circuitOf θ F).N → Option Hash.RowLeaf.Opening} {dColl : ℕ}
    (hJ : JoinsAt V k A x θ F c σ C Ms tab ℓ₁ ℓ₂ Dg P fs₁ fs₂ cut P')
    (hT : SpotFlock.RunsAs.Round2.TwoHolds A.h) (hU : UnitsWired V A x θ F c readX)
    (hX : V.extract θ F c σ =
      majority V A x θ F c σ (joinFrame V A x θ F c σ C (joinSessions A C Ms tab ℓ₁ ℓ₂) (joinLocate A tab ℓ₁ ℓ₂) Dg)
        readX)
    (hcaps : ∀ s Pt dd, A.L.parse x = some s → A.L.partition s = some Pt → A.L.two s = some dd →
      CapsAt dd Pt ((V.certOf θ).ε * V.units θ F) A.δ)
    (hopens : OpensBy V θ F c readX openOf)
    (hcr : Claims.ExpectedCR FlockDeploy.sha512
      (bindFinder V A x θ F c σ (joinFrame V A x θ F c σ C (joinSessions A C Ms tab ℓ₁ ℓ₂) (joinLocate A tab ℓ₁ ℓ₂) Dg)
        (collOfOpen V θ F c openOf) dColl))
    (hblob : Claims.ExpectedCR FlockDeploy.sha512 (blobFinder V θ F c σ (V.registered θ)))
    (hdelta : V.finderCR .sampled θ F c σ → V.premises .sampled θ F c σ →
      ENNReal.ofReal (A.errAt x
          (joinFrame V A x θ F c σ C (joinSessions A C Ms tab ℓ₁ ℓ₂) (joinLocate A tab ℓ₁ ℓ₂) Dg).P) +
        (joinCRBound V A x θ F c σ C Ms tab ℓ₁ ℓ₂ Dg q P fs₁ fs₂ cut +
          bindBound V A x θ F c σ
            (joinFrame V A x θ F c σ C (joinSessions A C Ms tab ℓ₁ ℓ₂) (joinLocate A tab ℓ₁ ℓ₂) Dg) openOf dColl) ≤
        V.delta θ .sampled (V.cost θ F c σ)) :
    ∃ s Pt dd, CouplesAt V k A x θ F c σ C Ms tab ℓ₁ ℓ₂ Dg q P fs₁ fs₂ cut P' s Pt dd readX openOf dColl := by
  obtain ⟨st, hst⟩ := decode_of_stmtOf hJ.stmtX
  obtain ⟨s, hs⟩ := parse_wired A hst
  obtain ⟨Pt, hPt⟩ := partition_wired A s
  obtain ⟨dd, hdd⟩ := two_wired hT s
  exact ⟨s, Pt, dd,
    { joins := hJ
      scope := scope_wired hJ hT
      extract := hX
      parse := hs
      partition := hPt
      two := hdd
      units := hU s Pt dd hs hPt hdd
      caps := hcaps s Pt dd hs hPt hdd
      opens := hopens
      collisionResistant := hcr
      blobCR := hblob
      delta := hdelta }⟩

end Verity.SampledTwo
