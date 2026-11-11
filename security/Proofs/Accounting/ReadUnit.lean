module

public import Definitions.Accounting.ReadUnit
public import Proofs.Core.Circuit
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
  row at its index in its local evaluation, or SHA-512 collides on what the two checks hashed
  (`Hash.SecurityProofs.EmbeddingReadBinding`). The local evaluation is the row the unit's own gates consume, and the
  transcript's where the row's gates are committed (`localEval_of_committed`).
* `readUnitDrawn`: over a run's coins, the run accepts with a drawn read that returns another row than its table's at
  its index, where nothing its check and the table's check hashed collides, with probability at most the layer below's
  bound: that the run accepts with a drawn read's unit wrong, or the read's registered rows failing, in the extractor's
  transcript. A unit may hold several reads, and a read is drawn when its unit is.
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

/-- A correct unit's local evaluation is the transcript at a gate outside the unit, and at a committed gate in it. -/
theorem localEval_of_committed {C : Verity.Circuit} {n : ℕ} {P : Verity.Partition C n} {X : C.Trace} {u : Fin n}
    (hcorrect : P.Correct X u) {g : Fin C.N} (hg : P.unit g = some u → g ∈ P.committed) :
    P.localEval X u g = X g := by
  by_cases hu : P.unit g = some u
  · exact (hcorrect g (hg hu) hu).symm
  · exact P.localEval_of_not_mem X hu

/-- The bits gates hold in a correct unit's local evaluation are the transcript's, where those in the unit are
committed. -/
theorem bits_localEval {C : Verity.Circuit} {n : ℕ} {P : Verity.Partition C n} {X : C.Trace} {u : Fin n}
    (hcorrect : P.Correct X u) {k : ℕ} {gs : Fin k → Fin C.N}
    (hg : ∀ i, P.unit (gs i) = some u → gs i ∈ P.committed) : bits (P.localEval X u) gs = bits X gs := by
  unfold bits
  exact congrArg List.ofFn (funext fun i =>
    congrArg (fun b => BitVec.getLsbD b 0) (localEval_of_committed hcorrect (hg i)))

end Accounting.ReadUnit

namespace Accounting.SecurityProofs

open Accounting.ReadUnit

variable {C : Verity.Circuit}

/-- **A read unit returns its table's row at its index**: at a transcript `X` where unit `u` is correct and its
registered rows hold for the table under the root `r` and the domain id `d`, the read's gate refinement (`GatesRefine`,
a hypothesis) makes the row it returns in `u`'s local evaluation the table's row at the index its index gates hold
there, for every opening of that leaf the root accepts, or `H` or `G` collides on what the read's check and the
opening's hashed. -/
theorem readUnitBinds (H G : Hash.Msg → Verity.Digest) (key : Hash.Msg) (consts : Fin C.N → Option Bool) {n : ℕ}
    (P : Verity.Partition C n) (u : Fin n) (q : Ports C) (r d : Verity.Digest) (X : C.Trace)
    (hgates : GatesRefine H G key consts P u q) (hcorrect : P.Correct X u) (hrows : RegisteredRows consts P u q r d X) :
    Binds H G key q r d (P.localEval X u) := by
  intro o₀ h₀
  obtain ⟨hh, hok⟩ := hgates.computes X hcorrect hrows.consts_hold
  have hroot : q.rootOf (P.localEval X u) = r :=
    (congrArg digestOf (bits_localEval hcorrect hgates.root_committed)).trans hrows.root_eq
  have hdomain : q.domainOf (P.localEval X u) = d :=
    (congrArg digestOf (bits_localEval hcorrect hgates.domain_committed)).trans hrows.domain_eq
  have hokL : q.okOf (P.localEval X u) = true :=
    (congrArg (fun b => BitVec.getLsbD b 0) (localEval_of_committed hcorrect hgates.ok_committed)).trans hrows.ok_eq
  have hok' := hok hokL
  rw [hroot, hdomain] at hok'
  have hlt : q.tokOf (P.localEval X u) < q.count := by
    obtain ⟨-, -, hlt, -⟩ := h₀
    exact hlt
  rw [hh]
  exact Hash.SecurityProofs.EmbeddingReadBinding H G (q.domainAt d) key r q.count q.words (q.tokOf (P.localEval X u))
    (q.opening (P.localEval X u)) hok' o₀ hlt h₀

/-- **At an accepted run's drawn reads, each returns its table's row at its index, or SHA-512 collides on what its
check and the table's hashed**, but with probability at most the layer below's bound. Over the run's coins `Ω`, with its
acceptance `acc`, its drawn units `drawn` and the extractor's transcript `X`: where each read's gates compute its
reference in its unit (`GatesRefine`, a hypothesis), each read's table is committed under its root, and the run accepts
with a drawn read's unit wrong or the read's registered rows failing in `X` with probability at most `ε`, it accepts
with a drawn read that returns, in its unit's local evaluation of `X`, another row than its table's at its index, where
no two messages the read's check and the table's check hashed collide under `H`, with probability at most `ε`. A read
is drawn when its unit is. -/
theorem readUnitDrawn {Ω : Type} [Fintype Ω] (H : Hash.Msg → Verity.Digest) (key : Hash.Msg)
    (consts : Fin C.N → Option Bool) {n : ℕ} {ι : Type} (P : Verity.Partition C n) (R : Reads C n ι)
    (acc : Ω → Prop) (drawn : Ω → Finset (Fin n)) (X : Ω → C.Trace) (ε : ℝ)
    (hgates : ∀ i ∈ R.reads, GatesRefine H H key consts P (R.unit i) (R.port i)) (hcommitted : R.Committed H H key)
    (below : Claims.Pr (fun ω => acc ω ∧ ∃ i ∈ R.reads, R.unit i ∈ drawn ω ∧
      ¬ (P.Correct (X ω) (R.unit i) ∧
        RegisteredRows consts P (R.unit i) (R.port i) (R.root i) (R.domain i) (X ω))) ≤ ε) :
    Claims.Pr (fun ω => acc ω ∧ ∃ i ∈ R.reads, R.unit i ∈ drawn ω ∧
      R.index i (P.localEval (X ω) (R.unit i)) < (R.port i).count ∧
      (R.port i).hOf (P.localEval (X ω) (R.unit i)) ≠
        (R.table i (R.index i (P.localEval (X ω) (R.unit i)))).row ∧
        ¬ Hash.CollisionIn H (R.hashedAt H key i (P.localEval (X ω) (R.unit i)))) ≤ ε := by
  apply le_trans ?_ below
  apply Claims.Pr_mono
  intro ω hω
  obtain ⟨hacc, i, hi, hd, hlt, hne, hnc⟩ := hω
  refine ⟨hacc, i, hi, hd, fun hb => ?_⟩
  rcases readUnitBinds H H key consts P (R.unit i) (R.port i) (R.root i) (R.domain i) (X ω) (hgates i hi) hb.1 hb.2 _
      (hcommitted i hi _ hlt) with h | h | h
  · exact hne h
  · exact hnc (collisionIn_mono h fun m hm => List.mem_append_left _ hm)
  · exact hnc (collisionIn_mono h fun m hm => List.mem_append_right _ hm)

/-- **At an accepted run's drawn reads, each returns its table's row at its index**, in its unit's local evaluation of
the extractor's transcript, but with probability at most the layer below's bound plus the reads' collision finder's
(`Reads.finder`) expected SHA-512 calls over `2^256`, at a finder that meets `Claims.ExpectedCR` (`ecr/sha-512`). The
finder outputs two messages the drawn reads hashed that collide under `H`, if there are two; its cost `cost` is the
reader's to state from the adversary it runs. -/
theorem readUnitDrawnCR {Ω : Type} [Fintype Ω] [Nonempty Ω] (H : Hash.Msg → Verity.Digest) (key : Hash.Msg)
    (consts : Fin C.N → Option Bool) {n : ℕ} {ι : Type} (P : Verity.Partition C n) (R : Reads C n ι)
    (acc : Ω → Prop) (drawn : Ω → Finset (Fin n)) (X : Ω → C.Trace) (cost : Ω → ℕ) (ε : ℝ)
    (hgates : ∀ i ∈ R.reads, GatesRefine H H key consts P (R.unit i) (R.port i)) (hcommitted : R.Committed H H key)
    (below : Claims.Pr (fun ω => acc ω ∧ ∃ i ∈ R.reads, R.unit i ∈ drawn ω ∧
      ¬ (P.Correct (X ω) (R.unit i) ∧
        RegisteredRows consts P (R.unit i) (R.port i) (R.root i) (R.domain i) (X ω))) ≤ ε)
    (hcr : Claims.ExpectedCR (fun m => (H m).bytes) (R.finder P H key drawn X cost)) :
    Claims.Pr (fun ω => acc ω ∧ ∃ i ∈ R.reads, R.unit i ∈ drawn ω ∧
      R.index i (P.localEval (X ω) (R.unit i)) < (R.port i).count ∧
      (R.port i).hOf (P.localEval (X ω) (R.unit i)) ≠
        (R.table i (R.index i (P.localEval (X ω) (R.unit i)))).row) ≤
      ε + Claims.Ex (fun ω => (cost ω : ℝ)) / 2 ^ (256 : ℝ) := by
  classical
  have hsplit : ∀ ω, (acc ω ∧ ∃ i ∈ R.reads, R.unit i ∈ drawn ω ∧
      R.index i (P.localEval (X ω) (R.unit i)) < (R.port i).count ∧
      (R.port i).hOf (P.localEval (X ω) (R.unit i)) ≠
        (R.table i (R.index i (P.localEval (X ω) (R.unit i)))).row) →
      (acc ω ∧ ∃ i ∈ R.reads, R.unit i ∈ drawn ω ∧
        R.index i (P.localEval (X ω) (R.unit i)) < (R.port i).count ∧
        (R.port i).hOf (P.localEval (X ω) (R.unit i)) ≠
          (R.table i (R.index i (P.localEval (X ω) (R.unit i)))).row ∧
          ¬ Hash.CollisionIn H (R.hashedAt H key i (P.localEval (X ω) (R.unit i)))) ∨
      Hash.CollisionIn H (R.drawnHashed P H key drawn X ω) := by
    rintro ω ⟨hacc, i, hi, hd, hlt, hne⟩
    by_cases hc : Hash.CollisionIn H (R.hashedAt H key i (P.localEval (X ω) (R.unit i)))
    · exact Or.inr (collisionIn_mono hc fun m hm =>
        List.mem_flatMap.2 ⟨i, Finset.mem_toList.2 (Finset.mem_filter.2 ⟨hi, hd⟩), hm⟩)
    · exact Or.inl ⟨hacc, i, hi, hd, hlt, hne, hc⟩
  have hfind : Claims.Pr (fun ω => Hash.CollisionIn H (R.drawnHashed P H key drawn X ω)) ≤
      (R.finder P H key drawn X cost).Pr (fun m => (H m).bytes) := by
    unfold Claims.Finder.Pr
    apply Claims.Pr_mono
    intro ω h
    obtain ⟨x, hx, y, hy, hxy⟩ := h
    have h' : ∃ x ∈ R.drawnHashed P H key drawn X ω, ∃ y ∈ R.drawnHashed P H key drawn X ω,
        Hash.Collision (fun m => (H m).bytes) x y := ⟨x, hx, y, hy, hxy⟩
    refine ⟨h'.choose, h'.choose_spec.2.choose, ?_, h'.choose_spec.2.choose_spec.2⟩
    simp [Reads.finder, h']
  exact (Claims.Pr_mono hsplit).trans ((pr_or_le _ _).trans
    (add_le_add (readUnitDrawn H key consts P R acc drawn X ε hgates hcommitted below) (hfind.trans hcr)))

end Accounting.SecurityProofs
