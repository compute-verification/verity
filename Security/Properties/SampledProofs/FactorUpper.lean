module

public import SampledProofs.FactorUpper
public import Definitions.Claims.SamplingTwoRankin

/-!
# Rankin's factor, bounded above: what the bound bounds (`SampledProofs.FactorUpper`)

`SampledProofs.FactorUpper` bounds Rankin's factor of a replay block above, in exact arithmetic, from the block's cells
(`Cell`: the second stage's `k` and the proof blocks by committed bits). These properties say the bound is above the
factor `Claims.Sampling.AdmissibleTwoRankin` and its patterns' form multiply, at every `q`, `θ`, breakpoints and
precision it accepts:

* `FactorUpperSound`: when `lnFactorUpper cs P` returns `v`, `G_ρ(θ) ≤ e^v` (`Claims.Sampling.rankinFactor`) for every
  replay block `ρ` whose cells `cs` describe (`Describes`);
* `FactorUpperPatternsSound`: when `lnFactorUpperPatterns cs S P` returns `v`, the patterns' factor is at most `e^v`
  (`Claims.Sampling.rankinFactorPatterns`) for every such `ρ` whose proof blocks are read by at most one request, which
  serves `S` bits.

How tight the bound is (`optimize.two_stage_exfil`'s figure, at the points the module's `#guard`s check) is not stated.
-/

@[expose] public section

namespace Properties.SampledProofs

open Claims.Sampling
open _root_.SampledProofs.FactorUpper (Cell Params lnFactorUpper lnFactorUpperPatterns)

/-- **`cs` describes replay block `ρ`'s cells**: one cell per stratum of the proof-block partition, each with the second
stage's `k` there and, for every width `b`, as many proof blocks committing `b` bits as cell `(ρ, t)` has. -/
def Describes (s : AuditStatement) (Pt : Partition) (d : Design.Two) (ρ : ℕ) (cs : List Cell) : Prop :=
  cs.length = Pt.strata.length ∧
    ∀ (t : ℕ) (c : Cell), cs[t]? = some c →
      d.k.getD t 0 = c.k ∧
        ∀ b, ((cell Pt d.part ρ t).filter fun v => committed s.prog (Pt.blocks.getD v []) = b).card = c.count b

/-- **The bound is above Rankin's factor**: when `lnFactorUpper cs P` returns `v`, every replay block `ρ` that `cs`
describes has `G_ρ(θ) ≤ e^v` at the first stage's rate `P.q` and Rankin's `P.θ`. -/
def FactorUpperSound : Prop :=
  ∀ (cs : List Cell) (P : Params) (v : ℚ), lnFactorUpper cs P = some v →
    ∀ (s : AuditStatement) (Pt : Partition) (d : Design.Two) (ρ : ℕ), Describes s Pt d ρ cs →
      rankinFactor s Pt d (P.q : ℝ) (P.θ : ℝ) ρ ≤ Real.exp (v : ℝ)

/-- **The bound is above the patterns' factor**: when `lnFactorUpperPatterns cs S P` returns `v`, every replay block `ρ`
that `cs` describes, whose proof blocks at most one request reads and that request serving `S` bits, has its patterns'
factor at most `e^v`. -/
def FactorUpperPatternsSound : Prop :=
  ∀ (cs : List Cell) (S : ℕ) (P : Params) (v : ℚ), lnFactorUpperPatterns cs S P = some v →
    ∀ (s : AuditStatement) (Pt : Partition) (d : Design.Two) (R : Type) [Fintype R] [DecidableEq R]
      (req : Fin s.prog.outs.length → R) (ρ : ℕ), Describes s Pt d ρ cs →
      (∀ r r', (blocksTo s.prog Pt req (blocksIn Pt d.part ρ) r).Nonempty →
        (blocksTo s.prog Pt req (blocksIn Pt d.part ρ) r').Nonempty → r = r') →
      (∀ r, (blocksTo s.prog Pt req (blocksIn Pt d.part ρ) r).Nonempty → servedBits req r = S) →
      rankinFactorPatterns s Pt d req (P.q : ℝ) (P.θ : ℝ) ρ ≤ Real.exp (v : ℝ)

end Properties.SampledProofs
