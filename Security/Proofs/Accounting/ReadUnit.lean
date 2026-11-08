module

public import Definitions.Accounting.ReadUnit
public import Proofs.Flock.Claims.Prob
public import Proofs.Hash.EmbeddingRead
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-!
# Read units bind to their tables' rows

The binding of table reads at a picked index (`Definitions.Accounting.ReadUnit`), each read's gate refinement
(`Accounting.ReadUnit.GatesRefine`) a hypothesis:

* `readUnitBinds`: at a transcript where a read unit is correct and its registered rows hold, it returns its table's
  row at its index, or SHA-512 collides on what the two checks hashed (`Hash.SecurityProofs.EmbeddingReadBinding`).
* `readUnitDrawn`: over a run's coins, the run accepts with a drawn read that returns another row than its table's at
  its index, where nothing its check and the table's check hashed collides, with probability at most the layer below's
  bound: that the run accepts with a drawn read wrong, or its registered rows failing, in the extractor's transcript.
* `readUnitDrawnCR`: the same with the collision priced through `Claims.ExpectedCR` at the reads' collision finder
  (`Accounting.ReadUnit.Reads.finder`): the layer below's bound plus the finder's expected SHA-512 calls over `2^256`.
-/

@[expose] public section

namespace Accounting.ReadUnit

/-- A collision among some messages is one among any list holding them. -/
theorem collisionIn_mono {H : Hash.Msg → Verity.Digest} {xs ys : List Hash.Msg} (h : Hash.CollisionIn H xs)
    (hs : ∀ m ∈ xs, m ∈ ys) : Hash.CollisionIn H ys := by
  obtain ⟨x, hx, y, hy, hxy⟩ := h
  exact ⟨x, hs x hx, y, hs y hy, hxy⟩

/-- A union bound for uniform coins. -/
theorem pr_or_le {α : Type} [Fintype α] (p q : α → Prop) :
    Claims.Pr (fun a => p a ∨ q a) ≤ Claims.Pr p + Claims.Pr q := by
  classical
  unfold Claims.Pr Claims.prUniform
  rw [← add_div]
  refine div_le_div_of_nonneg_right ?_ (by positivity)
  exact_mod_cast (Finset.card_le_card fun a ha => by
    simp only [Finset.mem_filter, Finset.mem_union, Finset.mem_univ, true_and] at ha ⊢; exact ha).trans
    (Finset.card_union_le _ _)

end Accounting.ReadUnit

namespace Accounting.SecurityProofs

open Accounting.ReadUnit

variable {C : Verity.Circuit}

/-- **A read unit returns its table's row at its index**: at a transcript `X` where unit `u` is correct and its
registered rows hold for the table under the root `r` and the domain id `d`, the read's gate refinement (`GatesRefine`,
a hypothesis) makes the row it returns the table's row at the index its index gates hold, for every opening of that
leaf the root accepts, or `H` or `G` collides on what the read's check and the opening's hashed. -/
theorem readUnitBinds (H G : Hash.Msg → Verity.Digest) (key : Hash.Msg) (consts : Fin C.N → Option Bool) {n : ℕ}
    (P : Verity.Partition C n) (u : Fin n) (q : Ports C) (r d : Verity.Digest) (X : C.Transcript)
    (hgates : GatesRefine H G key consts P u q) (hcorrect : P.Correct X u) (hrows : RegisteredRows consts P u q r d X) :
    Binds H G key q r d X := by
  intro o₀ h₀
  obtain ⟨hh, hok⟩ := hgates X hcorrect hrows.consts_hold
  have hok' := hok hrows.ok_eq
  rw [hrows.root_eq, hrows.domain_eq] at hok'
  have hlt : q.tokOf X < q.count := by
    obtain ⟨-, -, hlt, -⟩ := h₀
    exact hlt
  rw [hh]
  exact Hash.SecurityProofs.EmbeddingReadBinding H G (q.domainAt d) key r q.count q.words (q.tokOf X) (q.opening X)
    hok' o₀ hlt h₀

/-- **At an accepted run's drawn reads, each returns its table's row at its index, or SHA-512 collides on what its
check and the table's hashed**, but with probability at most the layer below's bound. Over the run's coins `Ω`, with its
acceptance `acc`, its drawn units `drawn` and the extractor's transcript `X`: where each read unit's gates compute its
reference (`GatesRefine`, a hypothesis), each read's table is committed under its root, and the run accepts with a drawn
read wrong or its registered rows failing in `X` with probability at most `ε`, it accepts with a drawn read that returns
another row than its table's at its index, where no two messages the read's check and the table's check hashed collide
under `H`, with probability at most `ε`. -/
theorem readUnitDrawn {Ω : Type} [Fintype Ω] (H : Hash.Msg → Verity.Digest) (key : Hash.Msg)
    (consts : Fin C.N → Option Bool) {n : ℕ} (P : Verity.Partition C n) (R : Reads C n) (acc : Ω → Prop)
    (drawn : Ω → Finset (Fin n)) (X : Ω → C.Transcript) (ε : ℝ)
    (hgates : ∀ u ∈ R.units, GatesRefine H H key consts P u (R.port u)) (hcommitted : R.Committed H H key)
    (below : Claims.Pr (fun ω => acc ω ∧ ∃ u ∈ drawn ω, u ∈ R.units ∧
      ¬ (P.Correct (X ω) u ∧ RegisteredRows consts P u (R.port u) (R.root u) (R.domain u) (X ω))) ≤ ε) :
    Claims.Pr (fun ω => acc ω ∧ ∃ u ∈ drawn ω, u ∈ R.units ∧ R.index u (X ω) < (R.port u).count ∧
      (R.port u).hOf (X ω) ≠ (R.table u (R.index u (X ω))).row ∧
        ¬ Hash.CollisionIn H (R.hashedAt H key u (X ω))) ≤ ε := by
  apply le_trans ?_ below
  apply Claims.Pr_mono
  intro ω hω
  obtain ⟨hacc, u, hu, hr, hlt, hne, hnc⟩ := hω
  refine ⟨hacc, u, hu, hr, fun hb => ?_⟩
  rcases readUnitBinds H H key consts P u (R.port u) (R.root u) (R.domain u) (X ω) (hgates u hr) hb.1 hb.2 _
      (hcommitted u hr _ hlt) with h | h | h
  · exact hne h
  · exact hnc (collisionIn_mono h fun m hm => List.mem_append_left _ hm)
  · exact hnc (collisionIn_mono h fun m hm => List.mem_append_right _ hm)

/-- **At an accepted run's drawn reads, each returns its table's row at its index**, but with probability at most the
layer below's bound plus the reads' collision finder's (`Reads.finder`) expected SHA-512 calls over `2^256`, at a finder
that meets `Claims.ExpectedCR` (`ecr/sha-512`). The finder outputs two messages the drawn reads hashed that collide
under `H`, if there are two; its cost `cost` is the reader's to state from the adversary it runs. -/
theorem readUnitDrawnCR {Ω : Type} [Fintype Ω] [Nonempty Ω] (H : Hash.Msg → Verity.Digest) (key : Hash.Msg)
    (consts : Fin C.N → Option Bool) {n : ℕ} (P : Verity.Partition C n) (R : Reads C n) (acc : Ω → Prop)
    (drawn : Ω → Finset (Fin n)) (X : Ω → C.Transcript) (cost : Ω → ℕ) (ε : ℝ)
    (hgates : ∀ u ∈ R.units, GatesRefine H H key consts P u (R.port u)) (hcommitted : R.Committed H H key)
    (below : Claims.Pr (fun ω => acc ω ∧ ∃ u ∈ drawn ω, u ∈ R.units ∧
      ¬ (P.Correct (X ω) u ∧ RegisteredRows consts P u (R.port u) (R.root u) (R.domain u) (X ω))) ≤ ε)
    (hcr : Claims.ExpectedCR (fun m => (H m).bytes) (R.finder H key drawn X cost)) :
    Claims.Pr (fun ω => acc ω ∧ ∃ u ∈ drawn ω, u ∈ R.units ∧ R.index u (X ω) < (R.port u).count ∧
      (R.port u).hOf (X ω) ≠ (R.table u (R.index u (X ω))).row) ≤
      ε + Claims.Ex (fun ω => (cost ω : ℝ)) / 2 ^ (256 : ℝ) := by
  classical
  have hsplit : ∀ ω, (acc ω ∧ ∃ u ∈ drawn ω, u ∈ R.units ∧ R.index u (X ω) < (R.port u).count ∧
      (R.port u).hOf (X ω) ≠ (R.table u (R.index u (X ω))).row) →
      (acc ω ∧ ∃ u ∈ drawn ω, u ∈ R.units ∧ R.index u (X ω) < (R.port u).count ∧
        (R.port u).hOf (X ω) ≠ (R.table u (R.index u (X ω))).row ∧
          ¬ Hash.CollisionIn H (R.hashedAt H key u (X ω))) ∨
      Hash.CollisionIn H (R.drawnHashed H key drawn X ω) := by
    rintro ω ⟨hacc, u, hu, hr, hlt, hne⟩
    by_cases hc : Hash.CollisionIn H (R.hashedAt H key u (X ω))
    · exact Or.inr (collisionIn_mono hc fun m hm =>
        List.mem_flatMap.2 ⟨u, Finset.mem_toList.2 (Finset.mem_filter.2 ⟨hu, hr⟩), hm⟩)
    · exact Or.inl ⟨hacc, u, hu, hr, hlt, hne, hc⟩
  have hfind : Claims.Pr (fun ω => Hash.CollisionIn H (R.drawnHashed H key drawn X ω)) ≤
      (R.finder H key drawn X cost).Pr (fun m => (H m).bytes) := by
    unfold Claims.Finder.Pr
    apply Claims.Pr_mono
    intro ω h
    obtain ⟨x, hx, y, hy, hxy⟩ := h
    have h' : ∃ x ∈ R.drawnHashed H key drawn X ω, ∃ y ∈ R.drawnHashed H key drawn X ω,
        Hash.Collision (fun m => (H m).bytes) x y := ⟨x, hx, y, hy, hxy⟩
    refine ⟨h'.choose, h'.choose_spec.2.choose, ?_, h'.choose_spec.2.choose_spec.2⟩
    simp [Reads.finder, h']
  exact (Claims.Pr_mono hsplit).trans ((pr_or_le _ _).trans
    (add_le_add (readUnitDrawn H key consts P R acc drawn X ε hgates hcommitted below) (hfind.trans hcr)))

end Accounting.SecurityProofs
