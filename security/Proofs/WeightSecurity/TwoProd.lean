module

public import Proofs.WeightSecurity.TwoReg

/-!
# A period of two-stage verification phases: one failure probability for the whole period

`PeriodTwoReg` weighs every incorrect replay block alike, at the least rate of any cell, and `two_undetected` pays a
failure probability in each verification phase and bounds the replay blocks each phase marks. Here the period pays one
failure probability `δ`: each verification phase's record carries the probability that its sample misses, as the
two-stage design gives it at the boundary trace the phase's strategy fixes, and those probabilities multiply across the
period's verification phases.

The boundary trace is a parameter `val`, one per verification phase and per release, statement and prover that the
phase's strategy sends before the phase's coins, so the statement reads the same at the committed boundary trace
(`Compiled.regVal` at a prover with no tape), at the boundary trace read at the statement (`Compiled.regValX`), at
either trace of a verifier whose extractor waits differently, and at a trace that reads the release. It takes each
verification phase's bound on the event (the run accepts with a sampled proof block incorrect at `val` and the
committed interiors) as a hypothesis, and no knowledge soundness of the argument.

* `Claims.Sampling.BestInteriorsAt`, `Design.Two.PaBestAt`: interiors at which each replay block escapes the second
  stage most often, and the design's `P̄_a` at the boundary values' footprint there.
* `CompiledTwo.FregTwoAt`: the incorrect set at boundary values `val` and the committed interiors, `FregTwo` with the
  boundary a parameter.
* `acceptedSampledAt`: the event's probability at a statement, a prover with no tape and boundary values.
* `regCountAt`, `escAt`: what a verification phase's strategy fixes before its coins, the replay blocks wrong at `val`
  and `P̄_a` at `val`'s footprint at best interiors (`0` and `1` out of the two-stage scope).
* `ServedEsc`, `twoAtRec`, `twoAtRun`: the record, a `Served` with that escape, and the period's run.
* `ServedEsc.leadEscape`, `ServedEsc.UndetectedBelow`: the escapes of the verification phases before the first whose
  sample holds an incorrect proof block, multiplied, and the event that an accepted phase's sample holds one or that
  product is below `δ`.
* `PeriodTwoProd`: every verification phase is accepted and the escapes multiply below `δ` with probability at most
  `δ + Σ_t ε_t`, each `ε_t` bounding verification phase `t`'s event at every strategy the run plays it with.
* `PeriodTwoLead`: the same bound on `ServedEsc.UndetectedBelow`, which a run with a rejected verification phase can
  be in. It gives `PeriodTwoProd`.
* `CompilerPeriodTwoProd`: both at every run of verifiers that meet the two-stage compiler's contract.

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.Sampling

/-- **Best interiors** at boundary values `val`: interiors, one per replay block, at which each replay block escapes
the second stage at least as often as at any other interior. -/
def BestInteriorsAt (s : AuditStatement) (Pt : Partition) (d : Design.Two) (val : ℕ → Option Bool)
    (ι : ℕ → ℕ → Bool) : Prop :=
  ∀ ρ < d.part.blocks.length, ∀ ι' : ℕ → ℕ → Bool,
    d.esc Pt (footprintAt s Pt d.part val ι') ρ ≤ d.esc Pt (footprintAt s Pt d.part val ι) ρ

/-- **`P̄_a` at best interiors**: the two-stage design's acceptance probability at the footprint of boundary values
`val` at best interiors (`BestInteriorsAt`). Every best interiors give one value: `P̄_a` reads a footprint through its
replay blocks' escapes only. -/
noncomputable def Design.Two.PaBestAt (d : Design.Two) (s : AuditStatement) (Pt : Partition)
    (val : ℕ → Option Bool) : ℝ :=
  d.Pa Pt (footprintAt s Pt d.part val (Classical.epsilon (BestInteriorsAt s Pt d val)))

end Claims.Sampling

namespace Claims.SamplingGeneric.CompiledTwo

open Claims
open Claims.Sampling (wrongAllAt)

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **The incorrect set at boundary values `val`**, at the prover's tape `ω` and the run's coins `r`: over the grid of
replay blocks and strata, the proof blocks incorrect at `val` and their replay block's committed interior (`iregVal`);
empty on a statement that doesn't parse or whose partitions don't evaluate. It is `FregTwo` with the boundary values a
parameter. -/
noncomputable def FregTwoAt (H : Msg → Msg) (L : CompiledTwo Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (val : ℕ → Option Bool) (r : Fin (L.V.k x) → Coin) : Finset ℕ :=
  match L.parse x with
  | none => ∅
  | some s =>
    match L.partition s, L.two s with
    | some Pt, some d => wrongAllAt s Pt d.part val (L.iregVal H P x ω r)
    | _, _ => ∅

end Claims.SamplingGeneric.CompiledTwo

namespace WeightSecurity.SecurityProofs

open Claims Claims.Sampling Claims.SamplingGeneric
open FlockSoundness.Game hiding Msg
open scoped ENNReal

/-- **What a run records of a verification phase, with its escape**: a `Served`, and the probability `escape` that the
phase's sample misses, which the strategy that plays the phase fixes before the phase's coins. -/
structure ServedEsc (O V Tick Adv : Type) extends Served O V Tick Adv where
  /-- The probability that the verification phase's sample misses, fixed before the phase's coins. -/
  escape : ℝ≥0∞

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin} {O V Tick Adv : Type}

namespace ServedEsc

open Classical in
/-- **The escapes before the first sampled incorrect proof block, multiplied**: those of each verification phase, up to
the first whose sample holds one of its incorrect proof blocks. It is `Served.lead` with the phases' escapes
multiplied in place of their counts added. -/
noncomputable def leadEscape : List (ServedEsc O V Tick Adv) → ℝ≥0∞
  | [] => 1
  | w :: l => if w.sampledIncorrect then 1 else w.escape * leadEscape l

/-- **The sampling leaves escapes that multiply below `δ` undetected**: an accepted verification phase's sample held
one of its incorrect proof blocks, or the escapes of the verification phases before the first sampled one multiply
below `δ`. Outside it, every verification phase accepted before the first rejection is one whose sample missed, and
the escapes of the verification phases before the first sampled one multiply to at least `δ`. -/
def UndetectedBelow (δ : ℝ) (l : List (ServedEsc O V Tick Adv)) : Prop :=
  (∃ w ∈ l, w.accepted ∧ w.sampledIncorrect) ∨ leadEscape l < ENNReal.ofReal δ

end ServedEsc

/-- **The event's probability** at a statement `x`, a prover `P` with no tape and boundary values `val`: over the
run's coins, the run accepts and its sample holds a proof block incorrect at `val` and the committed interiors
(`CompiledTwo.FregTwoAt`). -/
noncomputable def acceptedSampledAt (H : Msg → Msg) (L : CompiledTwo Coin A) (x : Msg) (P : Prover Unit)
    (val : ℕ → Option Bool) : ℝ :=
  Pr fun r : Fin (L.V.k x) → Coin =>
    (run H P L.V x () r).out = .acc ∧ ∃ u ∈ L.checked x r, u ∈ L.FregTwoAt H P x () val r

open Classical in
/-- **The incorrect replay blocks at boundary values `val`**: at the statement `x`, the replay blocks wrong at `val`
whatever their interior (`wrongAtBoundaryAt`); `0` on a statement out of the two-stage scope. -/
noncomputable def regCountAt (L : CompiledTwo Coin A) (x : Msg) (val : ℕ → Option Bool) : ℕ :=
  match L.parse x with
  | none => 0
  | some s =>
    match L.partition s, L.two s with
    | some Pt, some d => if InScopeTwo Pt d s then (wrongAtBoundaryAt s Pt d.part val).card else 0
    | _, _ => 0

open Classical in
/-- **The escape at boundary values `val`**: at the statement `x`, the two-stage design's `P̄_a` at `val`'s footprint at
best interiors (`Design.Two.PaBestAt`); `1` on a statement out of the two-stage scope, which no run accepts. It reads
the statement and `val` only, and no coin of the run. -/
noncomputable def escAt (L : CompiledTwo Coin A) (x : Msg) (val : ℕ → Option Bool) : ℝ≥0∞ :=
  match L.parse x with
  | none => 1
  | some s =>
    match L.partition s, L.two s with
    | some Pt, some d => if InScopeTwo Pt d s then ENNReal.ofReal (d.PaBestAt s Pt val) else 1
    | _, _ => 1

/-- **A two-stage verification phase's record at boundary values `val`**: accepted when the run accepts; sampled
incorrect when its sample holds a proof block incorrect at `val` and the committed interiors; incorrect, the count `n`,
and escape, the probability `e`, both of which its strategy fixes; and the release. -/
noncomputable def twoAtRec (H : Msg → Msg) (L : CompiledTwo Coin A) (val : ℕ → Option Bool) (n : ℕ) (e : ℝ≥0∞)
    (b : ((O → V) × Tick × Adv) × TwoPlay L) : ServedEsc O V Tick Adv where
  accepted := b.2.accepted H
  sampledIncorrect := ∃ u ∈ L.checked b.2.1 b.2.2.2, u ∈ L.FregTwoAt H b.2.2.1 b.2.1 () val b.2.2.2
  incorrect := n
  egress := b.1.1
  tick := b.1.2.1
  adv := b.1.2.2
  escape := e

/-- **The run over `T` two-stage verification phases** at the datacenter's strategy `s` and the boundary traces
`val`: verification phase `t` is recorded by `twoAtRec` at the boundary values `val t m x P` of the release `m`, the
statement `x` and the prover `P` its strategy sends, with the count and the escape there (`regCountAt`, `escAt`). -/
noncomputable def twoAtRun (H : Msg → Msg) (Lv : ℕ → CompiledTwo Coin A)
    (val : (t : ℕ) → (O → V) × Tick × Adv → Msg → Prover Unit → ℕ → Option Bool) (T : ℕ)
    (s : Strategy (periodG (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) 0 T)) :
    Verity.Game (List (ServedEsc O V Tick Adv)) :=
  periodRun (fun t => twoGame (Lv t) ((O → V) × Tick × Adv))
    (fun t τ => twoAtRec H (Lv t) (val t τ.1 τ.2.1.1 τ.2.1.2) (regCountAt (Lv t) τ.2.1.1 (val t τ.1 τ.2.1.1 τ.2.1.2))
      (escAt (Lv t) τ.2.1.1 (val t τ.1 τ.2.1.1 τ.2.1.2))) 0 T s

/-- **The period's one failure probability, at boundary traces `val`.** Over `T` two-stage verification phases played
by one adaptive strategy `s`, if at every strategy the run plays verification phase `t` with, on a statement in scope,
the phase's run accepts with a sampled proof block incorrect at `val` and the committed interiors with probability at
most `ε_t` (`acceptedSampledAt`), then for every `δ` the probability that every verification phase is accepted and the
phases' escapes (`escAt`: `P̄_a` at `val`'s footprint at best interiors) multiply below `δ` is at most
$$\delta + \sum_t \varepsilon_t.$$
Outside it, when every verification phase is accepted, the escapes multiply to at least `δ`. The sampling term is `δ`
however many verification phases the period has. -/
def PeriodTwoProd (H : Msg → Msg) (Lv : ℕ → CompiledTwo Coin A)
    (val : (t : ℕ) → (O → V) × Tick × Adv → Msg → Prover Unit → ℕ → Option Bool) : Prop :=
  ∀ (T : ℕ) (ε : ℕ → ℝ) (s : Strategy (periodG (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) 0 T)),
    (∀ t τ, PeriodPlays (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) 0 T s t τ → (Lv t).Scope τ.2.1.1 →
      acceptedSampledAt H (Lv t) τ.2.1.1 τ.2.1.2 (val t τ.1 τ.2.1.1 τ.2.1.2) ≤ ε t) →
    ∀ δ : ℝ, Verity.Game.value
        (fun l : List (ServedEsc O V Tick Adv) =>
          (∀ r ∈ l, r.accepted) ∧ (l.map ServedEsc.escape).prod < ENNReal.ofReal δ)
        (twoAtRun H Lv val T s) ≤
      ENNReal.ofReal δ + ∑ t ∈ Finset.range T, ENNReal.ofReal (ε t)

/-- **The period's one failure probability, through the first sampled incorrect proof block.** `PeriodTwoProd` with
the run's later verification phases free to be rejected: under the same hypothesis, for every `δ` the probability that
some accepted verification phase's sample holds a proof block incorrect at `val` and the committed interiors, or that
the escapes of the verification phases before the first whose sample holds one multiply below `δ`
(`ServedEsc.UndetectedBelow`), is at most
$$\delta + \sum_t \varepsilon_t.$$
It is `PeriodTwoReg` with the phases' escapes multiplied in place of one rate to the count of incorrect replay blocks,
and it gives `PeriodTwoProd`: a run whose verification phases are all accepted either has one whose sample holds an
incorrect proof block or multiplies every phase's escape. -/
def PeriodTwoLead (H : Msg → Msg) (Lv : ℕ → CompiledTwo Coin A)
    (val : (t : ℕ) → (O → V) × Tick × Adv → Msg → Prover Unit → ℕ → Option Bool) : Prop :=
  ∀ (T : ℕ) (ε : ℕ → ℝ) (s : Strategy (periodG (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) 0 T)),
    (∀ t τ, PeriodPlays (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) 0 T s t τ → (Lv t).Scope τ.2.1.1 →
      acceptedSampledAt H (Lv t) τ.2.1.1 τ.2.1.2 (val t τ.1 τ.2.1.1 τ.2.1.2) ≤ ε t) →
    ∀ δ : ℝ, Verity.Game.value (ServedEsc.UndetectedBelow δ) (twoAtRun H Lv val T s) ≤
      ENNReal.ofReal δ + ∑ t ∈ Finset.range T, ENNReal.ofReal (ε t)

/-- **The period's one failure probability from the verification phases' verifiers**: when every verification phase's
verifier meets the two-stage compiler's contract (`CompiledTwo.Sessions`: the two stages sample by the design, and what
is out of scope is refused), `PeriodTwoLead` and `PeriodTwoProd` hold at every boundary traces. It takes no knowledge
soundness of the argument: a caller meets their hypothesis with its own bound on the event at the trace it reads. -/
def CompilerPeriodTwoProd : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (O V Tick Adv : Type)
    (H : Msg → Msg) (Lv : ℕ → CompiledTwo Coin A)
    (val : (t : ℕ) → (O → V) × Tick × Adv → Msg → Prover Unit → ℕ → Option Bool),
    (∀ t, (Lv t).Sessions H) →
    PeriodTwoLead (O := O) (V := V) (Tick := Tick) (Adv := Adv) H Lv val ∧
      PeriodTwoProd (O := O) (V := V) (Tick := Tick) (Adv := Adv) H Lv val

end WeightSecurity.SecurityProofs
