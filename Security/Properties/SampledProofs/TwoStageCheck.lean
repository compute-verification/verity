module

public import SampledProofs.TwoStageCheck
public import Properties.SampledProofs.FactorUpper

/-!
# A two-stage point, checked: what its bits bound (`SampledProofs.TwoStageCheck`)

`SampledProofs.TwoStageCheck.checkTwoStage` returns the bits a two-stage point certifies, from its first stage's rate,
Rankin's `θ`, `δ` and its replay blocks' classes. These properties say that number bounds the count
`Claims.Sampling.CompiledTwo.Exfil` bounds the influence set by, at every deployment the point describes:

* `FirstStageAt d q`: the registered first stage samples each replay block at a rate at least `q`: `bernoulli:NUM/DEN`
  with `q ≤ NUM/DEN`, or the deployed uniform `Q`-subset of the population's `N` replay blocks with every input replay
  block (`[subset Q, all]`) with `q N ≤ Q`;
* `ClassesCover`: each replay block `ρ` is in a class `cls ρ` whose cells cover it (`Covers`), and class `i` holds at
  most its `count` replay blocks;
* `RequestsCover`, the patterns' form: each request reads in one replay block, no two read one proof block, and at most
  a class's `N` requests read in each of its replay blocks, each serving the point's `S` bits and reading at most the
  class's `reads` (`ReadsCover`);
* `TwoStageCheckSound`: a sets' point's bits `b` bound `Σ_{T admissible} 2^{b(T)} ≤ 2^b`;
* `TwoStageCheckPatternsSound`: a patterns' point's bits bound the admissible sets' patterns' sum
  (`Claims.Sampling.AdmissibleTwoRankinPatterns`' left side) by `2^b`.

How tight the bits are (the figure's driver measures them against `optimize.two_stage_exfil`) is not stated.
-/

@[expose] public section

namespace Properties.SampledProofs

open Claims.Sampling
open _root_.SampledProofs.TwoStageCheck (Point Form checkTwoStage)

/-- **The registered first stage samples each replay block at a rate at least `q`**: `bernoulli:NUM/DEN` with
`q ≤ NUM/DEN`, or `[subset Q, all]` over the replay partition's two classes, the population `c` and the inputs, with
`Q ≤ |c|` and `q |c| ≤ Q`. -/
def FirstStageAt (d : Design.Two) (q : ℝ) : Prop :=
  (∃ num den : ℕ, d.replay.rules = [.bernoulli num den] ∧ q ≤ (num : ℝ) / den) ∨
    ∃ (Q : ℕ) (c ic : List (List (List ℕ))), d.replay.rules = [.subset Q, .all] ∧ d.part.classes = [c, ic] ∧
      Q ≤ c.flatten.length ∧ q * c.flatten.length ≤ Q

/-- **The point's classes cover the replay blocks**: replay block `ρ` is in class `cls ρ`, whose cells cover it, and
class `i` holds at most its `count` replay blocks. -/
def ClassesCover (s : AuditStatement) (Pt : Partition) (d : Design.Two) (D : Point) (cls : ℕ → ℕ) : Prop :=
  (∀ ρ < d.part.blocks.length, ∃ c, D.classes[cls ρ]? = some c ∧ Covers s Pt d ρ c.cells) ∧
    ∀ i c, D.classes[i]? = some c → ((Finset.range d.part.blocks.length).filter (cls · = i)).card ≤ c.count

/-- **The requests read as the point's classes say**: each request reads in one replay block, no two read one proof
block, and in each replay block `ρ` at most its class's `N` requests read, each serving `D.S` bits and reading at most
the class's `reads`. -/
def RequestsCover (s : AuditStatement) (Pt : Partition) (d : Design.Two) {R : Type} [Fintype R] [DecidableEq R]
    (req : Fin s.prog.outs.length → R) (D : Point) (cls : ℕ → ℕ) : Prop :=
  (∀ r, ∃ ρ, ∀ v ∈ blocksTo s.prog Pt req (Finset.range Pt.blocks.length) r, replayOf Pt d.part v = some ρ) ∧
    (∀ r r', r ≠ r' → Disjoint (blocksTo s.prog Pt req (Finset.range Pt.blocks.length) r)
      (blocksTo s.prog Pt req (Finset.range Pt.blocks.length) r')) ∧
    ∀ ρ < d.part.blocks.length, ∀ c, D.classes[cls ρ]? = some c →
      (Finset.univ.filter fun r => (blocksTo s.prog Pt req (blocksIn Pt d.part ρ) r).Nonempty).card ≤ c.N ∧
        ∀ r, (blocksTo s.prog Pt req (blocksIn Pt d.part ρ) r).Nonempty →
          servedBits req r = D.S ∧ ReadsCover s Pt d req ρ c.reads r

/-- **A sets' point's bits bound the count**: when `checkTwoStage D` returns `b`, every statement in scope whose first
stage samples at `D.q` or more and whose replay blocks the classes cover has `Σ_{T admissible at δ} 2^{b(T)} ≤ 2^b`. -/
def TwoStageCheckSound : Prop :=
  ∀ (D : Point) (b : ℚ), checkTwoStage D = some b → D.form = .sets →
    ∀ (s : AuditStatement) (Pt : Partition) (d : Design.Two) (cls : ℕ → ℕ), InScopeTwo Pt d s →
      FirstStageAt d D.q → ClassesCover s Pt d D cls →
        ((∑ T ∈ admissibleTwo Pt d D.δ, 2 ^ bitsOf s Pt T : ℕ) : ℝ) ≤ (2 : ℝ) ^ (b : ℝ)

/-- **A patterns' point's bits bound the patterns' count**: when `checkTwoStage D` returns `b`, every statement in
scope whose first stage samples at `D.q` or more, whose replay blocks the classes cover and whose requests read as
they say has the admissible sets' patterns' sum at most `2^b`. -/
def TwoStageCheckPatternsSound : Prop :=
  ∀ (D : Point) (b : ℚ), checkTwoStage D = some b → D.form = .patterns →
    ∀ (s : AuditStatement) (Pt : Partition) (d : Design.Two) (R : Type) [Fintype R] [DecidableEq R]
      (req : Fin s.prog.outs.length → R) (cls : ℕ → ℕ), InScopeTwo Pt d s → FirstStageAt d D.q →
      ClassesCover s Pt d D cls → RequestsCover s Pt d req D cls →
        ((∑ p ∈ (admissibleTwo Pt d D.δ).image
              (fun T r => lightOf s Pt (servedBits req r) (blocksTo s.prog Pt req T r)),
            2 ^ ∑ r, lightBits s Pt (servedBits req r) (p r) : ℕ) : ℝ) ≤ (2 : ℝ) ^ (b : ℝ)

end Properties.SampledProofs
