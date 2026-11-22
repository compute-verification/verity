module

public import Definitions.Pouw.PouwWork
public import Definitions.Pouw.TS.Play
public import Definitions.Pouw.PearlC.Instance
public import Definitions.Draft.ActualCircuit.Epoch

/-!
# Forced matrix multiplication's statement (draft)

Today's useful work (PoUW) on the actual circuit, over `ResourceAccounting.PouwWork`'s vocabulary: the verifier at
`P.V` holds the epoch's challenge, the words it heard and its verdict (`Verdict`); the node is the location `X`, the
epoch `[P.ts, P.td)`. The served protocol is Pearl+'s, whose deployed target is `Verity.Vocab.PouwSound` in a composite
that lists Pearl+'s id; that target is about the declared circuit's tiles at the credit its check published, and
these statements are about the actual circuit. Pearl+ is not yet an instance of (a) and (b): its targets depend on
operands the prover commits during the epoch, its blocks are weighted by tickets, and its words reach `V` as a
commitment before the sample, not as words before `td`. Pearl+ has its own statement, `ForcedMatmulPearlPlusEpoch`.

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
* `CompiledImpliesAcceptsOn`: the lemma, about the verifier's own code, that its compiled answer implies `acceptsOn`.
* `PearlPlusDetectionAtTickets`, `PearlPlusTileCostAt`, `ForcedMatmulPearlPlusEpoch`: forced matrix multiplication as
  served, Pearl+, with the credited amount read from the transcript.

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

/-- **The compiled verifier's accept implies `acceptsOn`** (draft; belongs in Core; the Lean lead's to define; to be
proved, about the verifier's own code): the compiled verifier's accept on the transcript in the epoch `[P.ts, P.td)`
implies `acceptsOn` at the challenge the view holds. With it, (a) and (b) hold at `CompiledAccepts` in place of
`acceptsOn`. -/
def CompiledImpliesAcceptsOn (cv : CompiledVerifier L) (P : Verdict Ω L) (rd : Readout Ω L) : Prop :=
  ∀ v, CompiledAccepts cv P.ts P.td v → acceptsOn P rd v

/-- The priced work at `π` of the node `X`'s computed gates between the salt and the deadline, `[tₛ + ℓ, t_d − ℓ)`:
the span whose work `ResourceAccounting.Guarantees.B1` prices. -/
noncomputable def saltToDeadlineWork (X : Finset L) (π : (R : Actual.Run L) → Fin R.A.N → ℕ) (tₛ t_d ℓ : ℕ)
    (R : Actual.Run L) : ℕ :=
  R.A.work (π R) (R.A.region X (Set.Ico (tₛ + ℓ) (t_d - ℓ)))

/-- **Pearl+'s detection at ticket weights, a challenge at a time.** For every strategy `Sound` reads, Pearl+'s compiled
verifier accepts in the epoch `[tₛ, t_d)` on a view holding the challenge (the salt and the `t` tickets, `chal`) while
the ticket weight `cw` of the run's committed tiles that are right at that salt is below the credited reference work
`W v` read from the transcript, with probability at most `(1 − εs)^t + δbind`: the sample's term and SHA-512's
collision term, which binds the openings to the roots committed before the sample. New premise, open: a theorem only
at unit weights (`Pouw.Guarantees.WorkWeightedSampling`, and `PouwWork.detection_of_sampling` in the proofs tree). -/
def PearlPlusDetectionAtTickets [Fintype Ω] [LinearOrder L] (𝓕 : Set (Actual.Run L)) (V : Finset L)
    (cv : CompiledVerifier L) (tₛ t_d : ℕ) (chal : VerifierView L → Ω) (W : VerifierView L → ℕ)
    (cw : Actual.Run L → ℝ) (εs : ℝ) (t : ℕ) (δbind : ℝ) : Prop :=
  Sound Ω 𝓕 V (CompiledAccepts cv tₛ t_d) (fun S r => chal ((S r).view V) = r → (W ((S r).view V) : ℝ) ≤ cw (S r))
    ((1 - εs) ^ t + δbind)

/-- **The tile-cost conjecture at Pearl+'s tile, a challenge at a time.** For every strategy `Sound` reads whose oracle
calls are at most `q`, accepted or not, at all but a `δPearlC q N = (q + N)/2^128` share of the challenges its view
holds, the node's priced work between the salt and the deadline is at least `1 − 1/400` (Pearl+'s inert bound, the
debit cap of `Pouw.PearlC.pearlCProtocol`) times the ticket weight `cw` of its right tiles. New premise, open: it is to
follow from the conjecture at `pearlCProtocol` stated as `ResourceAccounting.Guarantees.B1` (whose row is
`pouwRow`); `B1` takes a `Pouw.Protocol` and `pearlCProtocol` is a `Pouw.PearlC.Protocol`, so that step can't be
stated on today's definitions. -/
def PearlPlusTileCostAt [Fintype Ω] [LinearOrder L] (𝓕 : Set (Actual.Run L)) (V : Finset L)
    (chal : VerifierView L → Ω) (X : Finset L) (π : (R : Actual.Run L) → Fin R.A.N → ℕ) (tₛ t_d ℓ : ℕ)
    (cw : Actual.Run L → ℝ) (q N : ℕ) : Prop :=
  Sound Ω 𝓕 V (fun _ => True)
    (fun S r => chal ((S r).view V) = r → (1 - 1 / 400 : ℝ) * cw (S r) ≤ saltToDeadlineWork X π tₛ t_d ℓ (S r))
    (Pouw.PearlC.δPearlC q N)

/-- **Forced matrix multiplication as served, Pearl+, in one epoch.** If Pearl+'s compiled verifier accepts in the
epoch `[tₛ, t_d)` then, except with probability `(1 − εs)^t + (q + N)/2^128 + δbind`, at a view holding the challenge,
the actual circuit is in the set of circuits whose priced work at the node between the salt and the deadline is at
least `(1 − 1/400) · W(τ)`, `W(τ)` read from the transcript (Pearl+'s credited reference work, computed by its verifier
from the openings). Premises: detection at ticket weights (`PearlPlusDetectionAtTickets`) and the tile-cost
conjecture at Pearl+'s tile (`PearlPlusTileCostAt`), each a challenge at a time.

It differs from (a): the credited amount is read from the transcript rather than fixed at `Wb · Nmin` before the
epoch; the acceptance is the compiled verifier's (`CompiledAccepts`) rather than a verdict on decoded words; the work
is all of the node's in `[tₛ + ℓ, t_d − ℓ)`, with no marking and no help term (a node fed its tiles from outside is the
tile-cost conjecture's to exclude, not priced); its `1/400` is Pearl+'s inert bound, not a price of help; and its error
carries the grinding term and SHA-512's binding term.

Not vacuous because its conclusion fails for the compiled verifier that accepts every transcript and reads
`W(τ) = 1` against the strategy that holds the challenge and computes nothing between the salt and the deadline (work
`0 < 399/400` at every challenge, with an error below `1`), where the premise that fails is
`PearlPlusDetectionAtTickets`. -/
def ForcedMatmulPearlPlusEpoch : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (L : Type) [LinearOrder L] (𝓕 : Set (Actual.Run L)) (V X : Finset L)
    (cv : CompiledVerifier L) (tₛ t_d ℓ : ℕ) (chal : VerifierView L → Ω) (W : VerifierView L → ℕ)
    (cw : Actual.Run L → ℝ) (π : (R : Actual.Run L) → Fin R.A.N → ℕ) (εs δbind : ℝ) (t q N : ℕ),
    PearlPlusDetectionAtTickets 𝓕 V cv tₛ t_d chal W cw εs t δbind →
    PearlPlusTileCostAt 𝓕 V chal X π tₛ t_d ℓ cw q N →
    Sound Ω 𝓕 V (CompiledAccepts cv tₛ t_d)
      (fun S r => chal ((S r).view V) = r →
        (1 - 1 / 400 : ℝ) * W ((S r).view V) ≤ saltToDeadlineWork X π tₛ t_d ℓ (S r))
      ((1 - εs) ^ t + Pouw.PearlC.δPearlC q N + δbind)

end Draft.ActualCircuit
