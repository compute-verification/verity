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
  replay block `ρ` whose cells `cs` covers (`Covers`: as many proof blocks of each width or more, a `k` or less);
* `FactorUpperPatternsSound`: when `lnFactorUpperPatterns cs reads N S P` returns `v`, the patterns' factor is at most
  `e^v` (`Claims.Sampling.rankinFactorPatterns`) for every such `ρ` whose proof blocks are read by at most `N` requests,
  each serving `S` bits and reading at most `reads` (`ReadsCover`), no two reading one proof block.

How tight the bound is (`optimize.two_stage_exfil`'s figure, at the points the module's `#guard`s check) is not stated.
-/

@[expose] public section

namespace Properties.SampledProofs

open Claims.Sampling
open _root_.SampledProofs.FactorUpper (Cell Params countIn lnFactorUpper lnFactorUpperPatterns)

/-- **`cs` covers replay block `ρ`'s cells**: one cell per stratum of the proof-block partition, each with, for every
width `b`, at least as many proof blocks committing `b` bits as cell `(ρ, t)` has, and a second stage's `k` at most the
design's there. One list covers every replay block whose cells it bounds this way. -/
def Covers (s : AuditStatement) (Pt : Partition) (d : Design.Two) (ρ : ℕ) (cs : List Cell) : Prop :=
  cs.length = Pt.strata.length ∧
    ∀ (t : ℕ) (c : Cell), cs[t]? = some c →
      c.k ≤ d.k.getD t 0 ∧
        ∀ b, ((cell Pt d.part ρ t).filter fun v => committed s.prog (Pt.blocks.getD v []) = b).card ≤ c.count b

/-- **The bound is above Rankin's factor**: when `lnFactorUpper cs P` returns `v`, every replay block `ρ` that `cs`
covers has `G_ρ(θ) ≤ e^v` at the first stage's rate `P.q` and Rankin's `P.θ`. -/
def FactorUpperSound : Prop :=
  ∀ (cs : List Cell) (P : Params) (v : ℚ), lnFactorUpper cs P = some v →
    ∀ (s : AuditStatement) (Pt : Partition) (d : Design.Two) (ρ : ℕ), Covers s Pt d ρ cs →
      rankinFactor s Pt d (P.q : ℝ) (P.θ : ℝ) ρ ≤ Real.exp (v : ℝ)

/-- **Request `r` reads at most `reads` of replay block `ρ`**: for every stratum `t` and width `b`, at most `reads[t]`'s
count of the proof blocks committing `b` bits in cell `(ρ, t)`. -/
def ReadsCover (s : AuditStatement) (Pt : Partition) (d : Design.Two) {R : Type} (req : Fin s.prog.outs.length → R)
    (ρ : ℕ) (reads : List (List (ℕ × ℕ))) (r : R) : Prop :=
  ∀ t b, ((blocksTo s.prog Pt req (cell Pt d.part ρ t) r).filter fun v =>
    committed s.prog (Pt.blocks.getD v []) = b).card ≤ countIn (reads.getD t []) b

/-- **The bound is above the patterns' factor**: when `lnFactorUpperPatterns cs reads N S P` returns `v`, every replay
block `ρ` that `cs` covers has its patterns' factor at most `e^v`, when at most `N` requests read its proof blocks, each
serving `S` bits and reading at most `reads`, and no two read one proof block. -/
def FactorUpperPatternsSound : Prop :=
  ∀ (cs : List Cell) (reads : List (List (ℕ × ℕ))) (N S : ℕ) (P : Params) (v : ℚ),
    lnFactorUpperPatterns cs reads N S P = some v →
    ∀ (s : AuditStatement) (Pt : Partition) (d : Design.Two) (R : Type) [Fintype R] [DecidableEq R]
      (req : Fin s.prog.outs.length → R) (ρ : ℕ), Covers s Pt d ρ cs →
      (Finset.univ.filter fun r => (blocksTo s.prog Pt req (blocksIn Pt d.part ρ) r).Nonempty).card ≤ N →
      (∀ r, (blocksTo s.prog Pt req (blocksIn Pt d.part ρ) r).Nonempty →
        servedBits req r = S ∧ ReadsCover s Pt d req ρ reads r) →
      (∀ r r', r ≠ r' →
        Disjoint (blocksTo s.prog Pt req (blocksIn Pt d.part ρ) r) (blocksTo s.prog Pt req (blocksIn Pt d.part ρ) r')) →
      rankinFactorPatterns s Pt d req (P.q : ℝ) (P.θ : ℝ) ρ ≤ Real.exp (v : ℝ)

end Properties.SampledProofs
