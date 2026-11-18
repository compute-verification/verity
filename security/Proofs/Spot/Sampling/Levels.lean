module

public import Proofs.Core.LevelsProofBlocks
public import Proofs.Spot.Sampling.Scheme

/-!
# The sampling layer's audit bound, read on levels

`Spot.Sampling.givesAuditBound` gives `Verity.AuditBound` under a design `D` over one partition's units. A level draws
blocks as gate sets. Where one level's design at the whole circuit misses the blocks `blk` gives a set of units with
`D`'s miss of that set, the same bound holds under the levels' sample, with the incorrect set read through `blk`:

* `Levels.auditBound_one`: `AuditBound` under `D` gives `AuditBound` under `L.design` at one level, over any units
  (`Levels.design_one`). `SamplingDesign.missProb_map_image` gives its hypothesis where the level's design is `D` read
  through an injective `blk`.
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

/-- **The audit bound under a design, read on one level** that misses each set's blocks with the design's miss. -/
theorem auditBound_one (L : Levels G) {l : Level G} (hL : L.levels = [l]) {U Run : Type} (D : SamplingDesign U)
    (blk : U → Finset G) (hD : ∀ S, (l.design Finset.univ).missProb (S.image blk) = D.missProb S)
    {pr : (Run → Prop) → ℝ≥0∞} {accepts : Run → Prop} {incorrect : Run → Finset U} {ε : ℝ≥0∞}
    (h : AuditBound D pr accepts incorrect ε) :
    AuditBound L.design pr accepts (fun r => (incorrect r).image blk) ε := by
  intro δ
  have he : (fun r => accepts r ∧ L.design.missProb ((incorrect r).image blk) ≤ δ) =
      fun r => accepts r ∧ D.missProb (incorrect r) ≤ δ := by
    funext r
    rw [L.missProb_one hL, hD]
  rw [he]
  exact h δ

/-- The placement read on `P`'s slots as `Fin |P.blocks|` is injective. -/
theorem SlotsAreBlocks.injective_fin {L : Levels G} {P : Claims.Sampling.Partition} {blk : ℕ → Finset G}
    (h : L.SlotsAreBlocks P blk) : Function.Injective fun v : Fin P.blocks.length => blk v :=
  fun a b hab => Fin.ext (h.1 (Finset.mem_coe.2 (Finset.mem_range.2 a.2))
    (Finset.mem_coe.2 (Finset.mem_range.2 b.2)) hab)

/-- **`Sound` gives the audit bound at one level**: `Spot.Sampling.givesAuditBound` at the design `D`, read on one
level that misses each set's blocks with `D`'s miss (`auditBound_one`). Every audit strategy is accepted while the
levels' sample misses its committed transcript's incorrect blocks with probability at most `δ` with probability at most
`δ` plus `Sound`'s knowledge and binding errors at its registration and state. -/
theorem accept_le_of_sound (hS : Spot.Sampling.Sound) (L : Levels G) {l : Level G} (hL : L.levels = [l]) {n : ℕ}
    (D : SamplingDesign (Fin n)) (blk : Fin n → Finset G)
    (hD : ∀ S, (l.design Finset.univ).missProb (S.image blk) = D.missProb S)
    (Reg : Type) (session : Finset (Fin n) → Reg → Game Bool) (Tr : Type) (incorrect : Tr → Finset (Fin n))
    (committed : (R : Reg) → ((ω : D.Ω) → Strategy (session (D.draw ω) R)) → Tr)
    (ks : (S : Finset (Fin n)) → (R : Reg) → Strategy (session S R) → Fin n → ℝ≥0∞)
    (link : (S : Finset (Fin n)) → (R : Reg) → Strategy (session S R) → Fin n → Tr → ℝ≥0∞)
    (hcover : ∀ S R (τ : Strategy (session S R)) u X, u ∈ S → u ∈ incorrect X →
      prob (· = true) (session S R) τ ≤ ks S R τ u + link S R τ u X)
    (εks εbind : (R : Reg) → ((ω : D.Ω) → Strategy (session (D.draw ω) R)) → ℝ≥0∞)
    (hks : ∀ R (τ : (ω : D.Ω) → Strategy (session (D.draw ω) R)) (tgt : D.Ω → Option (Fin n)),
      avg (fun ω => (tgt ω).elim 0 fun u => if u ∈ D.draw ω then ks (D.draw ω) R (τ ω) u else 0) ≤ εks R τ)
    (hlink : ∀ R (τ : (ω : D.Ω) → Strategy (session (D.draw ω) R)) (tgt : D.Ω → Option (Fin n)),
      avg (fun ω => (tgt ω).elim 0 fun u =>
        if u ∈ D.draw ω then link (D.draw ω) R (τ ω) u (committed R τ) else 0) ≤ εbind R τ)
    (σ : Strategy (Game.send Reg fun R => Game.coin D.Ω fun ω =>
        (session (D.draw ω) R).map fun ok => (ok, D.draw ω))) :
    let cont : (ω : D.Ω) → Strategy (session (D.draw ω) σ.1) := fun ω =>
      Strategy.ofMap (fun ok => (ok, D.draw ω)) (session (D.draw ω) σ.1) (σ.2 ω)
    let X := committed σ.1 cont
    AuditBound L.design
      (fun p => prob p
        (Game.send Reg fun R => Game.coin D.Ω fun ω => (session (D.draw ω) R).map fun ok => (ok, D.draw ω)) σ)
      (fun o => o.1 = true) (fun _ => (incorrect X).image blk) (εks σ.1 cont + εbind σ.1 cont) :=
  L.auditBound_one hL D blk hD
    (Spot.Sampling.givesAuditBound hS n D Reg session Tr incorrect committed ks link hcover εks εbind hks hlink σ)

end Levels

end Verity
