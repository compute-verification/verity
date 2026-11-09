module

public import Definitions.Claims.SamplingTwoReg
import Proofs.Flock.Claims.Sampling.Extract
import Proofs.Flock.SamplingGeneric.TwoFoot

/-!
# The two-stage committed form's sets

The sets the committed form at two stages reads, at boundary values `val`:

* `wrongAllAt`: over the grid of replay blocks and strata, each cell's incorrect proof blocks at its replay block's
  interior values. On the grid, its share of a cell is that cell's `wrongInAt` (`inter_cell_wrongAllAt`).
* `CompiledTwo.FregTwo_eq`: on a statement that parses and whose partitions evaluate, the committed incorrect set is
  `wrongAllAt` at the committed boundary trace and interiors.
* `CompiledTwo.FregTwo_forkOn`: the committed incorrect set reads no second-stage coin, since the committed interior
  doesn't (`CompiledTwo.iregVal_forkOn`).
-/

@[expose] public section

namespace Claims.Sampling

open Finset

theorem wrongInAt_subset_cell (s : AuditStatement) (Pt Pρ : Partition) (val : ℕ → Option Bool) (ρ t : ℕ)
    (ι : ℕ → Bool) : wrongInAt s Pt Pρ val ρ t ι ⊆ cell Pt Pρ ρ t :=
  filter_subset _ _

theorem footprintAt_le_card (s : AuditStatement) (Pt Pρ : Partition) (val : ℕ → Option Bool) (ι : ℕ → ℕ → Bool)
    (ρ t : ℕ) : footprintAt s Pt Pρ val ι ρ t ≤ (cell Pt Pρ ρ t).card :=
  card_le_card (wrongInAt_subset_cell s Pt Pρ val ρ t (ι ρ))

/-- **The incorrect proof blocks at boundary values `val` and interior values `ι`** over the grid of replay blocks and
strata: each cell's at its replay block's interior. -/
noncomputable def wrongAllAt (s : AuditStatement) (Pt Pρ : Partition) (val : ℕ → Option Bool) (ι : ℕ → ℕ → Bool) :
    Finset ℕ :=
  (range Pρ.blocks.length ×ˢ range Pt.strata.length).biUnion fun p => wrongInAt s Pt Pρ val p.1 p.2 (ι p.1)

theorem wrongAllAt_subset (s : AuditStatement) (Pt Pρ : Partition) (val : ℕ → Option Bool) (ι : ℕ → ℕ → Bool) :
    wrongAllAt s Pt Pρ val ι ⊆ range Pt.blocks.length := by
  intro v hv
  obtain ⟨p, _, hvp⟩ := mem_biUnion.1 hv
  exact cell_subset Pt Pρ p.1 p.2 (wrongInAt_subset_cell s Pt Pρ val p.1 p.2 _ hvp)

/-- On the grid, a cell's share of `wrongAllAt` is its incorrect proof blocks at its replay block's interior. -/
theorem inter_cell_wrongAllAt (s : AuditStatement) (Pt Pρ : Partition) (val : ℕ → Option Bool) (ι : ℕ → ℕ → Bool)
    {ρ t : ℕ} (hρ : ρ < Pρ.blocks.length) (ht : t < Pt.strata.length) :
    wrongAllAt s Pt Pρ val ι ∩ cell Pt Pρ ρ t = wrongInAt s Pt Pρ val ρ t (ι ρ) := by
  ext v
  rw [mem_inter]
  constructor
  · rintro ⟨hv, hvc⟩
    obtain ⟨p, _, hvp⟩ := mem_biUnion.1 hv
    obtain ⟨h1, h2⟩ := cell_disj (wrongInAt_subset_cell s Pt Pρ val p.1 p.2 _ hvp) hvc
    subst h1
    subst h2
    exact hvp
  · intro hv
    exact ⟨mem_biUnion.2 ⟨(ρ, t), mem_product.2 ⟨mem_range.2 hρ, mem_range.2 ht⟩, hv⟩,
      wrongInAt_subset_cell s Pt Pρ val ρ t _ hv⟩

end Claims.Sampling

namespace Claims.SamplingGeneric

open Claims
open Claims.Sampling (AuditStatement Partition InScopeTwo forkOn forkOn_forkOn wrongAllAt)

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- On a statement that parses and whose partitions evaluate, the committed incorrect set is `wrongAllAt` at the
committed boundary trace and interiors. -/
theorem FregTwo_eq (H : Msg → Msg) (L : CompiledTwo Coin A) {Ω : Type} (P : Prover Ω) {x : Msg}
    {s : AuditStatement} {Pt : Partition} {d : Sampling.Design.Two} (hs : L.parse x = some s)
    (hPt : L.partition s = some Pt) (hd : L.two s = some d) (ω : Ω) (r : Fin (L.V.k x) → Coin) :
    L.FregTwo H P x ω r = wrongAllAt s Pt d.part (L.regVal H P x ω) (L.iregVal H P x ω r) := by
  simp only [FregTwo, hs, hPt, hd, wrongAllAt]

/-- **The committed interior reads no second-stage coin.** -/
theorem iregVal_forkOn (H : Msg → Msg) (L : CompiledTwo Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r e : Fin (L.V.k x) → Coin) : L.iregVal H P x ω (forkOn (L.stage2 x) r e) = L.iregVal H P x ω r := by
  funext ρ g
  simp only [iregVal, forkOn_forkOn]

/-- **The committed incorrect set reads no second-stage coin.** -/
theorem FregTwo_forkOn (H : Msg → Msg) (L : CompiledTwo Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r e : Fin (L.V.k x) → Coin) : L.FregTwo H P x ω (forkOn (L.stage2 x) r e) = L.FregTwo H P x ω r := by
  simp only [FregTwo, iregVal_forkOn]

theorem inScopeTwo_of (L : CompiledTwo Coin A) {x : Msg} {s : AuditStatement} {Pt : Partition}
    {d : Sampling.Design.Two} (hx : L.Scope x) (hs : L.parse x = some s) (hPt : L.partition s = some Pt)
    (hd : L.two s = some d) : InScopeTwo Pt d s := by
  obtain ⟨s', Pt', d', hs', hPt', hd', hin⟩ := hx
  rw [hs] at hs'
  cases hs'
  rw [hPt] at hPt'
  cases hPt'
  rw [hd] at hd'
  cases hd'
  exact hin

end CompiledTwo

end Claims.SamplingGeneric
