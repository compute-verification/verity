module

public import Proofs.Properties.Spot.TwoStageCheck
public import Proofs.Flock.SamplingGeneric.TwoRankinRequests

/-!
# A patterns' point's bits bound its count, at requests that read several replay blocks

`Properties.Spot.TwoStageCheckPatternsSound` takes `RequestsCover`, whose first clause has each request read in one
replay block. Its conclusion holds without that clause, from the other two: no two requests read one proof block, and
in each replay block at most its class's `N` requests read, each serving the point's `S` bits and reading at most the
class's `reads`.

* `twoStageCheckPatternsSound_form_requests`: the conclusion at a point whose form is the patterns', from Rankin's
  bound at every split (`Claims.Sampling.admissibleTwo_rankin_patterns_requests`) and each replay block's factor bound
  at its class (`factorUpperPatternsSound`), which reads only the requests that read in that replay block.
* `twoStageCheckPatternsSound_requests`: the same in `TwoStageCheckPatternsSound`'s shape, `RequestsCover`'s last two
  clauses in its place.
* `twoStageCheckPatternsSound_of_requests`: `TwoStageCheckPatternsSound` follows, `RequestsCover`'s first clause
  unread.
-/

@[expose] public section

namespace Properties.Spot.SecurityProofs

open Finset
open Claims.Sampling
open _root_.Spot.TwoStageCheck
open TwoStageCheckProof

/-- **A patterns' point's bits bound the patterns' count, at requests that read several replay blocks**, at a point
whose form is the patterns': `RequestsCover`'s clauses but the first, that each request reads in one replay block. -/
theorem twoStageCheckPatternsSound_form_requests (D : Point) (b : ℚ) (hb : checkTwoStage D = some b)
    (hform : D.form = .patterns) (s : AuditStatement) (Pt : Partition) (d : Design.Two) (R : Type) [Fintype R]
    [DecidableEq R] (req : Fin s.prog.outs.length → R) (cls : ℕ → ℕ) (hs : InScopeTwo Pt d s)
    (hfirst : FirstStageAt d D.q) (hcov : ClassesCover s Pt d D cls)
    (hdisj : ∀ r r', r ≠ r' → Disjoint (blocksTo s.prog Pt req (Finset.range Pt.blocks.length) r)
      (blocksTo s.prog Pt req (Finset.range Pt.blocks.length) r'))
    (hreads : ∀ ρ < d.part.blocks.length, ∀ c, D.classes[cls ρ]? = some c →
      (Finset.univ.filter fun r => (blocksTo s.prog Pt req (blocksIn Pt d.part ρ) r).Nonempty).card ≤ c.N ∧
        ∀ r, (blocksTo s.prog Pt req (blocksIn Pt d.part ρ) r).Nonempty →
          servedBits req r = D.S ∧ ReadsCover s Pt d req ρ c.reads r) :
    ((∑ p ∈ (admissibleTwo Pt d D.δ).image (fun T r => lightOf s Pt (servedBits req r) (blocksTo s.prog Pt req T r)),
        2 ^ ∑ r, lightBits s Pt (servedBits req r) (p r) : ℕ) : ℝ) ≤ (2 : ℝ) ^ (b : ℝ) := by
  obtain ⟨-, hq1, hθ, hδ, -⟩ := valid_of_check hb
  have hq1' : ((D.q : ℚ) : ℝ) ≤ 1 := by exact_mod_cast hq1
  have hR := admissibleTwo_rankin_patterns_requests s Pt d (fun _ => (D.q : ℝ)) D.δ D.θ R req hs
    (firstStage_le hs hq1' hfirst) (fun _ => hq1') (by exact_mod_cast hδ) (by exact_mod_cast hθ)
  refine bits_of_rankin hb cls hR (fun ρ _ => rankinFactorPatterns_nonneg s Pt d req hq1' _ ρ)
    (fun ρ hρ c v hc hv => ?_) (fun ρ hρ => (hcov.1 ρ hρ).imp fun _ h => h.1) hcov.2
  obtain ⟨c', hc', hcv⟩ := hcov.1 ρ hρ
  rw [hc, Option.some.injEq] at hc'
  subst hc'
  have hv' : _root_.Spot.FactorUpper.lnFactorUpperPatterns c.cells c.reads c.N D.S D.params = some v := by
    simpa [classFactor, hform] using hv
  obtain ⟨hN, hserve⟩ := hreads ρ hρ c hc
  exact factorUpperPatternsSound c.cells c.reads c.N D.S D.params v hv' s Pt d R req ρ hcv hN hserve
    fun r r' hne => (hdisj r r' hne).mono (blocksTo_mono req (blocksIn_subset Pt d.part ρ) r)
      (blocksTo_mono req (blocksIn_subset Pt d.part ρ) r')

/-- **A patterns' point's bits bound the patterns' count, at requests that read several replay blocks**:
`TwoStageCheckPatternsSound`'s statement with `RequestsCover`'s last two clauses in its place. -/
theorem twoStageCheckPatternsSound_requests (D : Point) (b : ℚ)
    (hb : checkTwoStage { D with form := .patterns } = some b) (s : AuditStatement) (Pt : Partition) (d : Design.Two)
    (R : Type) [Fintype R] [DecidableEq R] (req : Fin s.prog.outs.length → R) (cls : ℕ → ℕ) (hs : InScopeTwo Pt d s)
    (hfirst : FirstStageAt d D.q) (hcov : ClassesCover s Pt d D cls)
    (hdisj : ∀ r r', r ≠ r' → Disjoint (blocksTo s.prog Pt req (Finset.range Pt.blocks.length) r)
      (blocksTo s.prog Pt req (Finset.range Pt.blocks.length) r'))
    (hreads : ∀ ρ < d.part.blocks.length, ∀ c, D.classes[cls ρ]? = some c →
      (Finset.univ.filter fun r => (blocksTo s.prog Pt req (blocksIn Pt d.part ρ) r).Nonempty).card ≤ c.N ∧
        ∀ r, (blocksTo s.prog Pt req (blocksIn Pt d.part ρ) r).Nonempty →
          servedBits req r = D.S ∧ ReadsCover s Pt d req ρ c.reads r) :
    ((∑ p ∈ (admissibleTwo Pt d D.δ).image (fun T r => lightOf s Pt (servedBits req r) (blocksTo s.prog Pt req T r)),
        2 ^ ∑ r, lightBits s Pt (servedBits req r) (p r) : ℕ) : ℝ) ≤ (2 : ℝ) ^ (b : ℝ) :=
  twoStageCheckPatternsSound_form_requests { D with form := .patterns } b hb rfl s Pt d R req cls hs hfirst hcov hdisj
    hreads

/-- **`TwoStageCheckPatternsSound`** follows: `RequestsCover`'s first clause is not read. -/
theorem twoStageCheckPatternsSound_of_requests : TwoStageCheckPatternsSound :=
  fun D b hb s Pt d R _ _ req cls hs hfirst hcov hreq =>
    twoStageCheckPatternsSound_requests D b hb s Pt d R req cls hs hfirst hcov hreq.2.1 hreq.2.2

end Properties.Spot.SecurityProofs
