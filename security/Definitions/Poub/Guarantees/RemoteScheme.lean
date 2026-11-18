module

public import Definitions.Poub.Remote.Scheme
public import Definitions.Poub.Guarantees.RemoteSketch

/-!
# The bits a PoUB session reads, as a scheme (trusted)

PoUB's claim of bits read on a session's trace (`ResourceAccounting.Remote.Held.Trace`, the claim `Trace.reads` that
`Properties.Poub.Line.ReadAt` credits), stated through `Verity.Scheme` (`Trace.scheme`):

* the blocks are the credited passes' receipts, a pass and a registered block, each costing the block's `b` bits;
* `over` is the bits of the blocks the credited passes plan (`Trace.planned`), the credit, and `under` the bits they
  read on the node (`Trace.reads`), the spend; there are no charges;
* the incorrect blocks are the wrong receipts (`Trace.wrong`);
* the break event is an accepted session whose receipts' sketch fails (`Trace.Sketched`), which `SketchBound` bounds,
  and the scheme's own term is that bound;
* the log has no condition of its own: the order the claim needs (each pass's seed after its window's coin and the root
  before it, which places its ticks) is inside the sketch's premises (`Receipted`, `Present`), so inside the break term;
* the charge inequality is `Trace.planned_le`: where the sketch holds, every planned block is read on the node or a
  wrong receipt.

What it gets back, through the accounting layer (`Verity.Scheme.AccountingLayer`):

* **ReadsGetsBack**: at any sketch bound `δs`, the audit's bound at the session's accepted outcomes and wrong receipts
  gives `GetsBack` at every cutoff, the audit's probability the uniform one on the session's outcomes.
* **ReadsGetsBackDeployed**: the same at the deployed receipts, `δs = δSketch 216 513 Q P N ε`, under
  `SketchBoundDeployed`'s premises.
-/

@[expose] public section

namespace ResourceAccounting.Remote.Held.Trace

open Finset
open scoped ENNReal

variable {Ω : Type} (T : Trace Ω)

/-- The receipts' sketch holds at `ω`: every correct receipt of a registered block in a credited pass was computed while
the block was in node memory or on a chip during the pass (`SketchBound`'s event). -/
def Sketched (ω : Ω) : Prop :=
  ∀ p < T.passes, ∀ j ∈ T.correct ω p, ∃ t, T.tick ω p ≤ t ∧ t < T.tick ω (p + 1) ∧ j ∈ T.present ω t

/-- The credited passes' wrong receipts of registered blocks, each a pass and a block. -/
def wrong (ω : Ω) : Finset (Fin T.passes × Fin T.blocks) :=
  univ.filter fun x => x.2 ∈ T.plan ω x.1 \ T.correct ω x.1

theorem card_wrong (ω : Ω) : (T.wrong ω).card = T.incorrect ω := by
  unfold wrong incorrect
  rw [card_filter, Fintype.sum_prod_type]
  simp only [sum_boole, Nat.cast_id, filter_mem_eq_inter, univ_inter]
  exact Fin.sum_univ_eq_sum_range (fun p => (T.plan ω p \ T.correct ω p).card) T.passes

/-- **The charge**: where the receipts' sketch holds, each planned block of a credited pass is read on the node or a
wrong receipt. -/
theorem planned_le {ω : Ω} (h : T.Sketched ω) : T.planned ω ≤ T.reads ω T.passes + (T.wrong ω).card := by
  rw [card_wrong]
  unfold planned reads incorrect
  rw [← sum_add_distrib]
  refine sum_le_sum fun p hp => ?_
  have hsub : T.plan ω p ∩ T.correct ω p ⊆ T.read ω p := fun j hj => by
    obtain ⟨t, h1, h2, ht⟩ := h p (mem_range.1 hp) j (mem_inter.1 hj).2
    exact mem_inter.2 ⟨hj, mem_biUnion.2 ⟨t, mem_Ico.2 ⟨h1, h2⟩, ht⟩⟩
  have := card_sdiff_add_card_inter (T.plan ω p) (T.correct ω p)
  have := card_le_card hsub
  omega

/-- **The bits read, as a scheme**, `b` bits a block, at the sketch's term `δs`: the blocks the credited passes'
receipts, each costing `b`; `over` the planned blocks' bits and `under` the bits read on the node; no charges; the
incorrect blocks the wrong receipts; the break event an accepted outcome whose sketch fails; no log condition. -/
noncomputable def scheme (b : ℕ) (δs : ℝ≥0∞) : Verity.Scheme where
  Block := Fin T.passes × Fin T.blocks
  cost _ := b
  charges := 0
  Run := Ω
  accepts := T.accepts
  over ω := b * T.planned ω
  under ω := b * T.reads ω T.passes
  logOK _ := True
  incorrect := T.wrong
  breaks ω := T.accepts ω ∧ ¬ T.Sketched ω
  δscheme := δs
  charge ω ha _ hb := by
    have hk : T.Sketched ω := by
      by_contra hn
      exact hb ⟨ha, hn⟩
    have := Nat.mul_le_mul_left b (T.planned_le hk)
    show b * T.planned ω ≤ b * T.reads ω T.passes + ∑ _x ∈ T.wrong ω, b + 0
    rw [sum_const, smul_eq_mul, add_zero]
    rw [mul_add, mul_comm b (T.wrong ω).card] at this
    exact this

end ResourceAccounting.Remote.Held.Trace

namespace ResourceAccounting.Guarantees.Remote

open ResourceAccounting.Remote.Held ResourceAccounting.Remote.Held.Assumptions ResourceAccounting.Remote.Sketch
  ResourceAccounting.Remote.Sketch.Assumptions
open scoped ENNReal

/-- **The bits read, through the scheme, at any sketch bound.** On any trace whose receipts' sketch fails on an
accepted outcome with probability at most `δs` (`SketchBound`), and any sampling design `D` on its receipts at whose
accepted outcomes and wrong receipts the audit's bound holds at error `ε` (`AuditBound`, the uniform probability on the
outcomes, `Run.prE`), the scheme `T.scheme b δs` gets back at every cutoff `δ`: an accepted outcome credits its planned
blocks' bits beyond the bits read on the node and the missable cost with probability at most `δ + ε + δs`. -/
def ReadsGetsBack : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (T : Trace Ω) (b : ℕ) (δs : ℝ) (D : Verity.SamplingDesign (Fin T.passes × Fin T.blocks))
    (ε : ℝ≥0∞),
    Verity.AuditBound (fun _ => D) (ResourceAccounting.Remote.Run.prE (Ω := Ω)) T.accepts T.wrong ε →
    SketchBound T δs →
    ∀ δ, (T.scheme b (ENNReal.ofReal δs)).GetsBack (fun _ => D) (ResourceAccounting.Remote.Run.prE (Ω := Ω)) δ ε

/-- **The bits read, through the scheme, at the deployed receipts.** `ReadsGetsBack` on a trace over a remote
session's sketch outcomes under `SketchBoundDeployed`'s premises, at its term `δSketch 216 513 Q P N εrank`. -/
def ReadsGetsBackDeployed : Prop :=
  ∀ (β Hs : Type) [Fintype β] [Fintype Hs] (P N S L q Q : ℕ) (G qtot : Fin P → ℕ) (εrank : ℝ)
    (A : Remote.Sketch.Run β Hs P N 128 512 S L q 64 G) (T : Trace (Outcome β Hs N 128)) (b : ℕ)
    (D : Verity.SamplingDesign (Fin T.passes × Fin T.blocks)) (ε : ℝ≥0∞),
    0 ≤ εrank → T.passes ≤ P →
    ResourceAccounting.Assumptions.SketchRank 64 128 27 (ENNReal.ofReal εrank) →
    ∑ j, G j ≤ Q + P → ∑ j, qtot j ≤ Q →
    ResourceAccounting.Assumptions.SketchHashRandom A (((Q : ℝ) + P) ^ 2 / 2 ^ 513) → KeyBudget A qtot → Receipted A T →
    Present A T →
    Verity.AuditBound (fun _ => D) (ResourceAccounting.Remote.Run.prE (Ω := Outcome β Hs N 128)) T.accepts T.wrong
      ε →
    ∀ δ, (T.scheme b (ENNReal.ofReal (δSketch 216 513 Q P N εrank))).GetsBack (fun _ => D)
      (ResourceAccounting.Remote.Run.prE (Ω := Outcome β Hs N 128)) δ ε

end ResourceAccounting.Guarantees.Remote
