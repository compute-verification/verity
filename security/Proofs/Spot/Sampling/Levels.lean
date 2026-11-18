module

public import Proofs.Core.LevelsKinds
public import Proofs.Core.LevelsProofBlocks
public import Proofs.Spot.Sampling.Scheme

/-!
# The sampling layer's audit bound, read on levels

`Spot.Sampling.givesAuditBound` gives `Verity.AuditBound` under a design `D` over one partition's units. A level draws
blocks as gate sets. Where one level's design at the whole circuit misses the blocks `blk` gives a set of units with
`D`'s miss of that set, the same bound holds under the levels' sample, with the incorrect set read through `blk`:

* `Levels.auditBound_of_misses`: `AuditBound` under `D` gives `AuditBound` under `L.design` at any number of levels,
  over any units, where `L.design` misses each set's blocks with `D`'s miss of the set. That hypothesis is the content:
  `Levels.misses_map_one` proves it at one level that is `D` read through an injective `blk`
  (`SamplingDesign.missProb_map_image`); at two levels it is shown for the design at hand.
* `Levels.auditBound_one`: the same at one level, with the hypothesis on the level's design (`Levels.missProb_one`).
* `Levels.accept_le_of_sound`: `Spot.Sampling.Sound` gives that bound at one level, `Pr[accept ∧ L.design misses the
  committed transcript's incorrect blocks with probability ≤ δ] ≤ δ + εks + εbind`: `givesAuditBound` read through
  `auditBound_one`.
* `Levels.SlotsAreBlocks.injective_fin`: a claims partition's proof blocks placed as the level's blocks give an
  injective `blk` on its slots.
-/

@[expose] public section

namespace Verity

namespace Levels

open Game
open scoped ENNReal

variable {G : Type} [Fintype G] [DecidableEq G]

/-- **The audit bound under a design, read on levels** that miss each set's blocks with the design's miss, at any
number of levels. -/
theorem auditBound_of_misses (L : Levels G) {U Run : Type} (D : SamplingDesign U) (blk : U → Finset G)
    (hD : ∀ S, L.design.missProb (S.image blk) = D.missProb S)
    {pr : (Run → Prop) → ℝ≥0∞} {accepts : Run → Prop} {incorrect : Run → Finset U} {ε : ℝ≥0∞}
    (h : AuditBound D pr accepts incorrect ε) :
    AuditBound L.design pr accepts (fun r => (incorrect r).image blk) ε := by
  intro δ
  have he : (fun r => accepts r ∧ L.design.missProb ((incorrect r).image blk) ≤ δ) =
      fun r => accepts r ∧ D.missProb (incorrect r) ≤ δ := by
    funext r
    rw [hD]
  rw [he]
  exact h δ

/-- **The audit bound under a design, read on one level** that misses each set's blocks with the design's miss. -/
theorem auditBound_one (L : Levels G) {l : Level G} (hL : L.levels = [l]) {U Run : Type} (D : SamplingDesign U)
    (blk : U → Finset G) (hD : ∀ S, (l.design Finset.univ).missProb (S.image blk) = D.missProb S)
    {pr : (Run → Prop) → ℝ≥0∞} {accepts : Run → Prop} {incorrect : Run → Finset U} {ε : ℝ≥0∞}
    (h : AuditBound D pr accepts incorrect ε) :
    AuditBound L.design pr accepts (fun r => (incorrect r).image blk) ε :=
  L.auditBound_of_misses D blk (fun S => (L.missProb_one hL _).trans (hD S)) h

/-- **One level that is a design read through an injective `blk`** misses each set's blocks with the design's miss. -/
theorem misses_map_one (L : Levels G) {l : Level G} (hL : L.levels = [l]) {U : Type} (D : SamplingDesign U)
    {blk : U → Finset G} (hblk : Function.Injective blk) (hl : l.design Finset.univ = D.map blk) (S : Finset U) :
    L.design.missProb (S.image blk) = D.missProb S := by
  rw [L.missProb_one hL, hl, SamplingDesign.missProb_map_image hblk]

/-- The placement read on `P`'s slots as `Fin |P.blocks|` is injective. -/
theorem SlotsAreBlocks.injective_fin {L : Levels G} {P : Claims.Sampling.Partition} {blk : ℕ → Finset G}
    (h : L.SlotsAreBlocks P blk) : Function.Injective fun v : Fin P.blocks.length => blk v :=
  fun a b hab => Fin.ext (h.1 (Finset.mem_coe.2 (Finset.mem_range.2 a.2))
    (Finset.mem_coe.2 (Finset.mem_range.2 b.2)) hab)

/-- **`Sound` gives the audit bound at one level**: `Spot.Sampling.givesAuditBound` at the design `D`, read on one
level that misses each set's blocks with `D`'s miss (`auditBound_one`). Every audit strategy is accepted while the
levels' sample misses its committed transcript's incorrect blocks with probability at most `δ` with probability at most
`δ` plus `Sound`'s knowledge and binding errors at its registration and state. -/
theorem accept_le_of_sound (hS : Spot.Sampling.Sound) (L : Levels G) {l : Level G} (hL : L.levels = [l]) {U : Type}
    [DecidableEq U] (D : SamplingDesign U) (blk : U → Finset G)
    (hD : ∀ S, (l.design Finset.univ).missProb (S.image blk) = D.missProb S)
    (Reg : Type) (session : Finset U → Reg → Game Bool) (A : Spot.Sampling.ExtractionAnalysis D Reg session)
    {εks εbind : (R : Reg) → Spot.Sampling.Cont D Reg session R → ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.BindSound εbind) (σ : Strategy (Spot.Sampling.audit D Reg session)) :
    AuditBound L.design (fun p => prob p (Spot.Sampling.audit D Reg session) σ) (fun o => o.1 = true)
      (fun _ => (A.incorrect (A.committedOf σ)).image blk)
      (εks (Spot.Sampling.reg σ) (Spot.Sampling.cont σ) + εbind (Spot.Sampling.reg σ) (Spot.Sampling.cont σ)) :=
  L.auditBound_one hL D blk hD (Spot.Sampling.givesAuditBound hS D Reg session A εks εbind hks hlink σ)

end Levels

end Verity
