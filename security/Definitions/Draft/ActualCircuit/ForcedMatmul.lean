module

public import Definitions.Pouw.PouwWork
public import Definitions.Pouw.TS.Play
public import Definitions.Draft.ActualCircuit.Epoch

/-!
# Forced matrix multiplication's statement (draft)

Today's useful work (PoUW) on the actual circuit, over `ResourceAccounting.PouwWork`'s vocabulary: the verifier at
`P.V` holds the epoch's challenge, the words it heard and its verdict (`Verdict`); the node is the location `X`, the
epoch `[P.ts, P.td)`. The served protocol is Pearl+'s, whose deployed target is `Verity.Vocab.PouwSound` in a composite
that lists Pearl+'s id; that target is about the declared circuit's tiles at the credit its check published, and
these statements are about the actual circuit. Pearl+ is one choice of `Verdict`'s words and targets; nothing here is
specific to it.

Every premise is stated a challenge at a time (`Definitions.Draft.ActualCircuit.Epoch`): the challenge, the words and
the verdict are read from the verifier's view (`Readout`, `acceptsOn`, `goodOn`), the work from the one run
(`PouwWork.credits`, `PouwWork.help`), and every event is guarded by the view holding the challenge it is indexed by
(`HoldsChallenge`), since `Sound` ranges over strategies that ignore their index.

* `PouwDetectionAt`: A2 a challenge at a time.
* `PouwFloorAt`: a new form of A3, per challenge.
* `ForcedMatmulEpoch`: (a), one epoch, one bound, one error.
* `PouwDetectionBudgetAt`, `forcedGame`, `ForcedMatmulEpochPooled`: (b), one epoch with the pooling kept, as
  `ResourceAccounting.TS.RoundGame.BudgetAt` of a one-round game read on actual runs, which
  `Definitions.Draft.ActualCircuit.PooledEpochs` multiplies over a monitoring period.

The bound on ingress the help term needs is network accounting's own statement, met by intersection; it is not stated
here.
-/

@[expose] public section

namespace Draft.ActualCircuit

open ResourceAccounting ResourceAccounting.PouwWork ResourceAccounting.TS Pouw.RelativeSecurity Finset

variable {Ω : Type} {L : Type}

/-- **PoUW's verdict read on the verifier's view**: the challenge the verifier holds (`chal`: it is at `V`, so in its
view) and the words it heard (`heard`). New: `Verdict`'s rule and words read the strategy's challenge index and what
crossed into `V`; these read them off the one run's view. -/
structure Readout (Ω : Type) (L : Type) where
  chal : VerifierView L → Ω
  heard : VerifierView L → Inbox L

/-- The verifier accepts on its view `v`: `P.accept` at the challenge it holds, on the words it heard. -/
def acceptsOn (P : Verdict Ω L) (rd : Readout Ω L) (v : VerifierView L) : Prop :=
  P.accept (rd.chal v) (rd.heard v)

open Classical in
/-- The blocks whose heard word is right at the challenge the view holds. -/
noncomputable def goodOn (P : Verdict Ω L) (rd : Readout Ω L) (v : VerifierView L) : ℕ :=
  ((range P.N).filter fun k => P.word (rd.heard v) k = P.target (rd.chal v) k).card

/-- **The view holds the challenge `r`**: at index `r`, the challenge the verifier at `V` holds in the run `S r` is `r`.
Every per-challenge event is guarded by it: `Sound` also ranges over strategies whose runs ignore their index (a
strategy constant at one run), and those are excused at every index but the one their view holds. -/
def HoldsChallenge (rd : Readout Ω L) (V : Finset L) (S : Ω → Actual.Run L) (r : Ω) : Prop :=
  rd.chal ((S r).view V) = r

/-- **A2 a challenge at a time.** For every strategy `Sound` reads, the verifier accepts on a view holding the
challenge with fewer than `Nmin` heard words right at it with probability at most `δ`. New premise, open: the sample's
detection bound, words and challenge read off the view. -/
def PouwDetectionAt [Fintype Ω] [LinearOrder L] (𝓕 : Set (Actual.Run L)) (P : Verdict Ω L) (rd : Readout Ω L)
    (Nmin : ℕ) (δ : ℝ) : Prop :=
  Sound Ω 𝓕 P.V (acceptsOn P rd) (fun S r => HoldsChallenge rd P.V S r → Nmin ≤ goodOn P rd ((S r).view P.V)) δ

/-- **A3, per challenge.** A new form of A3; whether the conjecture survives it is the compute-accounting and Lean
leads' to say. For every strategy `Sound` reads, accepted or not, at all but a `δ` share of the challenges the view
holds, `Wb` times the words heard right at that challenge is at most the node's credits outside the marking `m` at
`π` during the epoch plus `ρ` a bit of help (`PouwWork.credits`, `PouwWork.help`). It is `PouwFloorOff` with the right
words counted at the challenge rather than at every challenge, and linear in them. A strategy whose inputs hold the
right words at every challenge makes it false unless the feasible set bounds the node's stored bits below that
table. -/
def PouwFloorAt [Fintype Ω] [LinearOrder L] (𝓕 : Set (Actual.Run L)) (P : Verdict Ω L) (rd : Readout Ω L)
    (X : Finset L) (π : (A : Placed L) → Fin A.N → ℕ) (m : (A : Placed L) → Finset (Fin A.N)) (Wb ρ : ℕ) (δ : ℝ) :
    Prop :=
  Sound Ω 𝓕 P.V (fun _ => True)
    (fun S r => HoldsChallenge rd P.V S r → Wb * goodOn P rd ((S r).view P.V) ≤
      credits S P X (fun r => π (S r).A) (fun r => m (S r).A) r + ρ * help S P X r) δ

/-- **(a) Forced matrix multiplication in one epoch.** If the verifier accepts in the epoch `[P.ts, P.td)` then,
except with probability `δdet + δskip` over its challenges, at a view holding the challenge, the node's credits outside
the marking plus `ρ` times the bits it took in during the epoch are at least `Wb · Nmin`. Premises: A2 and the new A3,
each a challenge at a time.

(a) fixes one tolerance, `Wb · (N − Nmin)`, before the epoch and pays its error in every epoch, so over a monitoring
period of `n` epochs either the errors add `n` times or each epoch's tolerance is taken separately. It drops the shape
of the shortfall's distribution, which (b) keeps so that the period's tolerance is paid once.

Not vacuous because its conclusion fails for the verifier that accepts every view against the strategy that holds the
challenge and computes and hears nothing (credits and ingress `0`, at `Wb · Nmin ≥ 1` and `δdet + δskip < 1`), where the
premise that fails is `PouwDetectionAt`; `PouwDetectionAt` holds at `δdet = 0`, `Nmin = N` for the verifier that accepts
only when every heard word is right at the challenge it holds, and I cannot name a verifier at which `PouwFloorAt` is
proved: it is the conjecture. -/
def ForcedMatmulEpoch : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (L : Type) [LinearOrder L] (P : Verdict Ω L) (𝓕 : Set (Actual.Run L))
    (rd : Readout Ω L) (X : Finset L) (π : (A : Placed L) → Fin A.N → ℕ)
    (m : (A : Placed L) → Finset (Fin A.N)) (Wb Nmin ρ : ℕ) (δdet δskip : ℝ),
    PouwDetectionAt 𝓕 P rd Nmin δdet → PouwFloorAt 𝓕 P rd X π m Wb ρ δskip →
    Sound Ω 𝓕 P.V (acceptsOn P rd)
      (fun S r => HoldsChallenge rd P.V S r →
        Wb * Nmin ≤ credits S P X (fun r => π (S r).A) (fun r => m (S r).A) r + ρ * help S P X r)
      (δdet + δskip)

/-- **The detection game** of a strategy: one round whose share is the words not heard right at the challenge the view
holds and whose check is that the view holds the challenge and the verifier accepts. -/
noncomputable def detectionGame (S : Ω → Actual.Run L) (P : Verdict Ω L) (rd : Readout Ω L) : RoundGame Ω 1 :=
  .round (fun r => ((P.N - goodOn P rd ((S r).view P.V) : ℕ) : ℝ))
    (fun r => HoldsChallenge rd P.V S r ∧ acceptsOn P rd ((S r).view P.V)) fun _ => .done

/-- **A2, pooled, a challenge at a time**: every strategy `Sound` reads has `Λ` bound its detection game
(`RoundGame.BudgetAt`: `E[exp((N − good)/Λ) · 1{holds the challenge ∧ accept}] ≤ 1`), stated through `Sound` at error
`0` with a guarantee that reads the strategy. New premise, open: the sample's detection bound at every number of wrong
words at once. -/
def PouwDetectionBudgetAt [Fintype Ω] [LinearOrder L] (𝓕 : Set (Actual.Run L)) (P : Verdict Ω L) (rd : Readout Ω L)
    (Λ : ℝ) : Prop :=
  Sound Ω 𝓕 P.V (fun _ => True) (fun S _ => RoundGame.BudgetAt Λ (detectionGame S P rd)) 0

/-- **The forced game** of a strategy: one round whose share is the node's shortfall below `Wb · N`, credits plus `ρ`
times help, and whose check is that the view holds the challenge, the verifier accepts and the new A3's inequality
holds there (off the skip event). -/
noncomputable def forcedGame (S : Ω → Actual.Run L) (P : Verdict Ω L) (rd : Readout Ω L) (X : Finset L)
    (π : (A : Placed L) → Fin A.N → ℕ) (m : (A : Placed L) → Finset (Fin A.N)) (Wb ρ : ℕ) : RoundGame Ω 1 :=
  .round
    (fun r => ((Wb * P.N - (credits S P X (fun r => π (S r).A) (fun r => m (S r).A) r + ρ * help S P X r) : ℕ) : ℝ))
    (fun r => HoldsChallenge rd P.V S r ∧ acceptsOn P rd ((S r).view P.V) ∧
      Wb * goodOn P rd ((S r).view P.V) ≤
        credits S P X (fun r => π (S r).A) (fun r => m (S r).A) r + ρ * help S P X r)
    fun _ => .done

/-- **(b) Forced matrix multiplication in one epoch, the pooling kept.** For every strategy `Sound` reads, `Wb · Λ`
bounds its forced game (`RoundGame.BudgetAt`): off the new A3's skip event, of probability at most `δskip` by
`PouwFloorAt`, `E[exp(shortfall/(Wb · Λ)) · 1{holds the challenge ∧ accept}] ≤ 1`. So at every tolerance at once,
`Wb · Λ · t`, the epoch accepts with a larger shortfall with probability at most `δskip + exp(−t)` (Theorem S's tail,
`ResourceAccounting.SecurityProofs.TS.budget_tail`, at the forced game), and over a monitoring period the epochs'
bounds multiply and the period's tolerance is counted once (`PooledEpochs`). Premises: `Λ > 0`, A2 pooled and the new
A3, each a challenge at a time.

Not vacuous because its conclusion fails for the verifier that accepts every view against the strategy that holds the
challenge and computes and hears nothing (shortfall `Wb · N ≥ 1` at every challenge, the new A3's inequality holding
there with no word right, so `E[exp(shortfall/(Wb · Λ)) · 1{check}] = exp(N/Λ) > 1`), where the premise that fails is
`PouwDetectionBudgetAt`; and I cannot name a verifier at which `PouwFloorAt` is proved. -/
def ForcedMatmulEpochPooled : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (L : Type) [LinearOrder L] (P : Verdict Ω L) (𝓕 : Set (Actual.Run L))
    (rd : Readout Ω L) (X : Finset L) (π : (A : Placed L) → Fin A.N → ℕ)
    (m : (A : Placed L) → Finset (Fin A.N)) (Wb ρ : ℕ) (Λ δskip : ℝ),
    0 < Λ → PouwDetectionBudgetAt 𝓕 P rd Λ → PouwFloorAt 𝓕 P rd X π m Wb ρ δskip →
    Sound Ω 𝓕 P.V (fun _ => True) (fun S _ => RoundGame.BudgetAt (Wb * Λ) (forcedGame S P rd X π m Wb ρ)) 0

end Draft.ActualCircuit
