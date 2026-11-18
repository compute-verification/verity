module

public import Definitions.Core.LevelsCredited

/-!
# A scheme: what a resource accounting protocol supplies, and what it gets back

A resource accounting protocol is to state its guarantee through these and define none of its own. No guarantee is proved
through an instance yet: each still stands in its own form.

* `Scheme`: what a protocol supplies to the proof layers. Its blocks, the units the audit samples; each block's
  cost, the resource it stands for; its charges, the resource it can't tie to a block; the runs the verifier
  accepts, with each run's two sides, `over` and `under`, in one unit; its conditions on the verifier's log; the
  blocks of a run that are incorrect; the event that breaks one of its own assumptions, with that event's probability
  bound; and its charge inequality: in an accepted run whose log meets its conditions and that breaks nothing, `over`
  exceeds `under` by at most the cost of the incorrect blocks plus its charges. The scheme says which of its
  quantities is which side. A scheme that bounds a resource used beyond what was credited (egress) sets `over` the
  resource used and `under` the resource credited; a scheme that bounds what is credited by what was spent (useful
  work) sets `over` the credit and `under` the spend. The sides are naturals: a guarantee with a factor rescales its
  own side into the common unit, and the structure carries no factor.
* `Scheme.missable`: the largest cost of a block set a sample misses with probability above a cutoff.
* `AuditBoundAt`: what the audit gives every scheme, in the form a scheme reads. At every cutoff `δ`, a run is
  accepted while its design's sample misses its incorrect blocks with probability at most `δ`, with probability at
  most `δ + ε`, `ε` the audit's own error. The design is the run's: a scheme whose design follows what the run
  recorded before the coin (a prover's declarations) reads it from the run, and whoever proves the bound shows the
  design was fixed before the coin. `AuditBound` is the same under one design for every run. The incorrect blocks are
  those of the run's committed values, so an accepted run's committed values are correct outside a set the sample
  would likely have hit. A scheme that bounds something other than a
  cost (the bits an accepted record can carry) reads this directly.
* `Scheme.GetsBack`: what a scheme gets back from the audit, a named `Prop`. At cutoff `δ`, an accepted run whose
  log meets the scheme's conditions has `over` above `under` by more than the missable cost under its own design plus
  the charges with probability at most `δ` plus the audit's error plus the scheme's own term.
* `Scheme.AccountingLayer`: the accounting layer's theorem, stated. For every scheme, `AuditBoundAt` at its accepted
  runs and incorrect blocks, with its break event bounded by its own term, gives `GetsBack` at every cutoff.
* `Scheme.GetsBackPeriod`, `Scheme.PeriodLayer`: the same over a period of execution phases, by the union bound over
  the phases: the probability is `T` times one phase's, and the phases' provers may adapt to one another.
* `Scheme.Placed`: a scheme's blocks placed in a circuit, each block its gate set. `Placed.credited` is the credited
  set the levels setting and the integrity profile take, and `Placed.Unsplit` says it is credited unsplit
  (`Levels.CreditedUnsplit`): each block is a block of the first level and of every later one.

A scheme names no level, no commitment and no duration: the audit decides which values its blocks open, and the
probability `pr` is the audit game's, over any prover. The code's interface is
`verity.protocols.resource_accounting.scheme` (`Scheme`, `Bound`), whose names these are.
-/

@[expose] public section

namespace Verity

open scoped ENNReal

/-- **What the audit gives every scheme**: under each run's sampling design `L r` over the blocks `U` and the audit
game's probability `pr` on runs, at every cutoff `δ` a run is accepted while its design's sample misses its incorrect
blocks with probability at most `δ`, with probability at most `δ + ε`. -/
def AuditBoundAt {U Run : Type} (L : Run → SamplingDesign U) (pr : (Run → Prop) → ℝ≥0∞) (accepts : Run → Prop)
    (incorrect : Run → Finset U) (ε : ℝ≥0∞) : Prop :=
  ∀ δ : ℝ≥0∞, pr (fun r => accepts r ∧ (L r).missProb (incorrect r) ≤ δ) ≤ δ + ε

/-- **The audit's bound under one design**: `AuditBoundAt` where every run's design is `L`. -/
def AuditBound {U Run : Type} (L : SamplingDesign U) (pr : (Run → Prop) → ℝ≥0∞) (accepts : Run → Prop)
    (incorrect : Run → Finset U) (ε : ℝ≥0∞) : Prop :=
  AuditBoundAt (fun _ => L) pr accepts incorrect ε

/-- **A scheme**: what a resource accounting protocol supplies to the proof layers. -/
structure Scheme where
  /-- The blocks: the units the audit samples. -/
  Block : Type
  [fintype : Fintype Block]
  /-- The resource a block stands for. -/
  cost : Block → ℕ
  /-- The charges: resource the scheme can't tie to a block, the sum of its charges' bounds. -/
  charges : ℕ
  /-- A run of the audit, as far as the scheme reads it. -/
  Run : Type
  /-- The verifier accepts the run. -/
  accepts : Run → Prop
  /-- The run's larger side: the resource used (egress), or the credit (useful work). -/
  over : Run → ℕ
  /-- The run's smaller side: the resource credited (egress), or the spend (useful work). -/
  under : Run → ℕ
  /-- The run's log meets the scheme's conditions: each entry it names is logged before, or after, the coin or other
  logged event it names. -/
  logOK : Run → Prop
  /-- The run's incorrect blocks. -/
  incorrect : Run → Finset Block
  /-- The run breaks one of the scheme's own assumptions (a key guessed, a hash collision). -/
  breaks : Run → Prop
  /-- The scheme's own term: the bound on the probability of `breaks`. -/
  δscheme : ℝ≥0∞
  /-- The charge inequality: in an accepted run whose log meets the conditions and that breaks nothing, `over`
  exceeds `under` by at most the cost of the incorrect blocks plus the charges. -/
  charge : ∀ r, accepts r → logOK r → ¬ breaks r → over r ≤ under r + ∑ b ∈ incorrect r, cost b + charges

attribute [instance] Scheme.fintype

namespace Scheme

open Classical in
/-- The missable cost at cutoff `δ`: the largest cost of a block set the sample misses with probability above `δ`. -/
noncomputable def missable (S : Scheme) (L : SamplingDesign S.Block) (δ : ℝ≥0∞) : ℕ :=
  (Finset.univ.filter fun T : Finset S.Block => δ < L.missProb T).sup fun T => ∑ b ∈ T, S.cost b

/-- **What a scheme gets back**: under each run's sampling design `L r` and the audit game's probability `pr`, at
cutoff `δ` an accepted run whose log meets the conditions has `over` above `under` by more than the missable cost under
its own design plus the charges with probability at most `δ + ε + δscheme`, `ε` the audit's error. A scheme with one
design takes `fun _ => L`. -/
def GetsBack (S : Scheme) (L : S.Run → SamplingDesign S.Block) (pr : (S.Run → Prop) → ℝ≥0∞) (δ ε : ℝ≥0∞) : Prop :=
  pr (fun r => S.accepts r ∧ S.logOK r ∧ S.under r + S.missable (L r) δ + S.charges < S.over r) ≤ δ + ε + S.δscheme

/-- **The accounting layer's theorem** (stated). For every scheme, sampling design of each run and probability on its
runs that is monotone and subadditive: the audit's bound at the scheme's accepted runs and incorrect blocks, and the
scheme's break event bounded by its own term, give what the scheme gets back at every cutoff. -/
def AccountingLayer : Prop :=
  ∀ (S : Scheme) (L : S.Run → SamplingDesign S.Block) (pr : (S.Run → Prop) → ℝ≥0∞) (ε : ℝ≥0∞),
    (∀ p q : S.Run → Prop, (∀ r, p r → q r) → pr p ≤ pr q) →
    (∀ p q : S.Run → Prop, pr (fun r => p r ∨ q r) ≤ pr p + pr q) →
    AuditBoundAt L pr S.accepts S.incorrect ε → pr S.breaks ≤ S.δscheme → ∀ δ, S.GetsBack L pr δ ε

/-- **What a scheme gets back over a period** of `T` execution phases, each a run of the scheme read from the period's
run `r` by `phase t`: at cutoff `δ`, some phase is accepted with a log that meets the conditions and has `over` above
`under` by more than the missable cost plus the charges with probability at most `T · (δ + ε + δscheme)`. Outside that
event, summed over the accepted phases whose logs meet the conditions, `over` exceeds `under` by at most `T` times the
missable cost plus the charges. -/
def GetsBackPeriod (S : Scheme) (L : S.Run → SamplingDesign S.Block) {R : Type} (T : ℕ) (phase : Fin T → R → S.Run)
    (pr : (R → Prop) → ℝ≥0∞) (δ ε : ℝ≥0∞) : Prop :=
  pr (fun r => ∃ t, S.accepts (phase t r) ∧ S.logOK (phase t r) ∧
      S.under (phase t r) + S.missable (L (phase t r)) δ + S.charges < S.over (phase t r)) ≤ T * (δ + ε + S.δscheme)

/-- **The accounting layer over a period** (stated). For every scheme and probability on a period's runs that is
monotone, subadditive and zero on the empty event: what the scheme gets back at each execution phase, the probability
read through that phase, gives what it gets back over the period. The phases' provers may adapt to one another: each
phase's bound is over every prover, and the period's is their union bound, with no product of the phases' designs. -/
def PeriodLayer : Prop :=
  ∀ (S : Scheme) (L : S.Run → SamplingDesign S.Block) (R : Type) (T : ℕ) (phase : Fin T → R → S.Run)
    (pr : (R → Prop) → ℝ≥0∞) (δ ε : ℝ≥0∞),
    (∀ p q : R → Prop, (∀ r, p r → q r) → pr p ≤ pr q) →
    (∀ p q : R → Prop, pr (fun r => p r ∨ q r) ≤ pr p + pr q) →
    pr (fun _ => False) = 0 →
    (∀ t, S.GetsBack L (fun p => pr fun r => p (phase t r)) δ ε) → S.GetsBackPeriod L T phase pr δ ε

/-- **A scheme placed in a circuit** with gates `G`: each block's gate set, two blocks never the same set. -/
structure Placed (S : Scheme) (G : Type) where
  /-- A block's gates. -/
  gates : S.Block → Finset G
  /-- Two blocks are never the same gate set. -/
  inj : Function.Injective gates

variable {S : Scheme} {G : Type}

/-- The credited blocks of a placed scheme: its blocks' gate sets, the credited set the levels setting and the
integrity profile take. -/
def Placed.credited [DecidableEq G] (P : S.Placed G) : Finset (Finset G) :=
  Finset.univ.image P.gates

/-- **A placed scheme's blocks are credited unsplit** under the levels `L`: each is a block of the first level and of
every later one, so its values are committed before the sample. -/
def Placed.Unsplit [Fintype G] [DecidableEq G] (P : S.Placed G) (L : Levels G) : Prop :=
  L.CreditedUnsplit P.credited

end Scheme

end Verity
