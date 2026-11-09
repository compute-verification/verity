module

public import Definitions.Claims.SamplingServed

/-!
# The two-stage sampled audit's guarantees

`InfluenceTwoPatterns`, **the patterns' form** of what the incorrect proof blocks of a two-stage sampled audit can
serve: on fixed input values, the served outputs of the boundary traces the two-stage relation takes at `δ` number at
most `Σ_p 2^{Σ_r bits(p r)}`, `p` over the admissible sets' patterns `r ↦ lightOf |r| T_r` at any split `req` of the
served outputs into requests.

`CapTwo`, **the per-window cap**: at a first stage that catches each replay block at rate `r₁` and a second stage that
samples each cell at rate `r₂`, the acceptance probability at a footprint is at most `(1 − r₁·r₂)^m`, `m` the replay
blocks it marks incorrect. So a footprint in `M̄_δ` marks fewer than every `K` with `(1 − r₁·r₂)^K ≤ δ`.
-/

@[expose] public section

namespace Claims.Guarantees

open Claims.Sampling

/-- **The patterns' form of the two-stage influence set**: in the two-stage scope (`InScopeTwo`), at every `δ`, input
values `a` and split `req` of the served outputs into requests, the influence set has at most
`Σ_p 2^{Σ_r lightBits |r| (p r)}` members, `p` over the admissible sets' patterns `r ↦ lightOf |r| T_r`. A request `T`
doesn't touch costs nothing; one whose `T_r` commits at least `|r|` bits costs `|r|`, with its blocks forgotten; any
other costs `b(T_r)`. -/
def InfluenceTwoPatterns : Prop :=
  ∀ (H : Msg → Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement) (Pt : Partition) (d : Design.Two),
    InScopeTwo Pt d s → ∀ (δ : ℝ) (a : ℕ → Option Bool) (R : Type) [Fintype R] [DecidableEq R]
    (req : Fin s.prog.outs.length → R),
    (influenceTwo H Cm s Pt d δ a).card ≤
      ∑ p ∈ (admissibleTwo Pt d δ).image (fun T r => lightOf s Pt (servedBits req r) (blocksTo s.prog Pt req T r)),
        2 ^ ∑ r, lightBits s Pt (servedBits req r) (p r)

/-- **The per-window cap**: at a first stage that catches each replay block at rate `r₁` (`Design.Two.FirstRate`) and
a second stage that samples each cell at rate `r₂` (`Design.Two.SecondRate`), for every boundary trace `w` and
interiors `ι`, the acceptance probability at their footprint is at most `(1 − r₁·r₂)^m`, `m` the replay blocks the
footprint marks incorrect (`Design.Two.wrongBlocks`). -/
def CapTwo : Prop :=
  ∀ (H : Msg → Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement) (Pt : Partition) (d : Design.Two) (r₁ r₂ : ℝ),
    d.FirstRate r₁ → d.SecondRate Pt r₂ → 0 ≤ r₁ → r₁ ≤ 1 → 0 ≤ r₂ → ∀ (w : Wit) (ι : ℕ → ℕ → Bool),
      d.Pa Pt (footprint H Cm s Pt d.part w ι) ≤
        (1 - r₁ * r₂) ^ (d.wrongBlocks Pt (footprint H Cm s Pt d.part w ι)).card

end Claims.Guarantees
