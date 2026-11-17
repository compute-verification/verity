module

public import Definitions.ResourceAccounting.ReadUnit
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

/-! Read units bind to their tables' rows, each read's gate refinement (`ResourceAccounting.ReadUnit.GatesRefine`) a
hypothesis. -/

@[expose] public section

namespace ResourceAccounting.ReadUnit

theorem collisionIn_mono {H : Hash.Msg → Verity.Digest} {xs ys : List Hash.Msg} (h : Hash.CollisionIn H xs)
    (hs : ∀ m ∈ xs, m ∈ ys) : Hash.CollisionIn H ys := by
  obtain ⟨x, hx, y, hy, hxy⟩ := h
  exact ⟨x, hs x hx, y, hs y hy, hxy⟩

theorem pr_or_le {α : Type} [Fintype α] (p q : α → Prop) :
    Claims.Pr (fun a => p a ∨ q a) ≤ Claims.Pr p + Claims.Pr q := by
  classical
  unfold Claims.Pr Claims.prUniform
  rw [← add_div]
  refine div_le_div_of_nonneg_right ?_ (by positivity)
  exact_mod_cast (Finset.card_le_card fun a ha => by
    simp only [Finset.mem_filter, Finset.mem_union, Finset.mem_univ, true_and] at ha ⊢; exact ha).trans
    (Finset.card_union_le _ _)

theorem localEval_of_committed {C : Verity.Circuit} {n : ℕ} {P : Verity.Partition C n} {X : C.Trace} {u : Fin n}
    (hcorrect : P.Correct X u) {g : Fin C.N} (hg : P.unit g = some u → g ∈ P.committed) :
    P.localEval X u g = X g := by
  by_cases hu : P.unit g = some u
  · exact (hcorrect g (hg hu) hu).symm
  · exact P.localEval_of_not_mem X hu

theorem bits_localEval {C : Verity.Circuit} {n : ℕ} {P : Verity.Partition C n} {X : C.Trace} {u : Fin n}
    (hcorrect : P.Correct X u) {k : ℕ} {gs : Fin k → Fin C.N}
    (hg : ∀ i, P.unit (gs i) = some u → gs i ∈ P.committed) : bits (P.localEval X u) gs = bits X gs := by
  unfold bits
  exact congrArg List.ofFn (funext fun i =>
    congrArg (fun b => BitVec.getLsbD b 0) (localEval_of_committed hcorrect (hg i)))

end ResourceAccounting.ReadUnit

namespace ResourceAccounting.SecurityProofs

open ResourceAccounting.ReadUnit

variable {C : Verity.Circuit}

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

theorem readUnitSampled {Ω : Type} [Fintype Ω] (H : Hash.Msg → Verity.Digest) (key : Hash.Msg)
    (consts : Fin C.N → Option Bool) {n : ℕ} {ι : Type} (P : Verity.Partition C n) (R : Reads C n ι)
    (acc : Ω → Prop) (sampled : Ω → Finset (Fin n)) (X : Ω → C.Trace) (ε : ℝ)
    (hgates : ∀ i ∈ R.reads, GatesRefine H H key consts P (R.unit i) (R.port i)) (hcommitted : R.Committed H H key)
    (below : Claims.Pr (fun ω => acc ω ∧ ∃ i ∈ R.reads, R.unit i ∈ sampled ω ∧
      ¬ (P.Correct (X ω) (R.unit i) ∧
        RegisteredRows consts P (R.unit i) (R.port i) (R.root i) (R.domain i) (X ω))) ≤ ε) :
    Claims.Pr (fun ω => acc ω ∧ ∃ i ∈ R.reads, R.unit i ∈ sampled ω ∧
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

/-- The finder's cost `cost` is the reader's to state from the adversary it runs. -/
theorem readUnitSampledCR {Ω : Type} [Fintype Ω] [Nonempty Ω] (H : Hash.Msg → Verity.Digest) (key : Hash.Msg)
    (consts : Fin C.N → Option Bool) {n : ℕ} {ι : Type} (P : Verity.Partition C n) (R : Reads C n ι)
    (acc : Ω → Prop) (sampled : Ω → Finset (Fin n)) (X : Ω → C.Trace) (cost : Ω → ℕ) (ε : ℝ)
    (hgates : ∀ i ∈ R.reads, GatesRefine H H key consts P (R.unit i) (R.port i)) (hcommitted : R.Committed H H key)
    (below : Claims.Pr (fun ω => acc ω ∧ ∃ i ∈ R.reads, R.unit i ∈ sampled ω ∧
      ¬ (P.Correct (X ω) (R.unit i) ∧
        RegisteredRows consts P (R.unit i) (R.port i) (R.root i) (R.domain i) (X ω))) ≤ ε)
    (hcr : Claims.ExpectedCR (fun m => (H m).bytes) (R.finder P H key sampled X cost)) :
    Claims.Pr (fun ω => acc ω ∧ ∃ i ∈ R.reads, R.unit i ∈ sampled ω ∧
      R.index i (P.localEval (X ω) (R.unit i)) < (R.port i).count ∧
      (R.port i).hOf (P.localEval (X ω) (R.unit i)) ≠
        (R.table i (R.index i (P.localEval (X ω) (R.unit i)))).row) ≤
      ε + Claims.Ex (fun ω => (cost ω : ℝ)) / 2 ^ (256 : ℝ) := by
  classical
  have hsplit : ∀ ω, (acc ω ∧ ∃ i ∈ R.reads, R.unit i ∈ sampled ω ∧
      R.index i (P.localEval (X ω) (R.unit i)) < (R.port i).count ∧
      (R.port i).hOf (P.localEval (X ω) (R.unit i)) ≠
        (R.table i (R.index i (P.localEval (X ω) (R.unit i)))).row) →
      (acc ω ∧ ∃ i ∈ R.reads, R.unit i ∈ sampled ω ∧
        R.index i (P.localEval (X ω) (R.unit i)) < (R.port i).count ∧
        (R.port i).hOf (P.localEval (X ω) (R.unit i)) ≠
          (R.table i (R.index i (P.localEval (X ω) (R.unit i)))).row ∧
          ¬ Hash.CollisionIn H (R.hashedAt H key i (P.localEval (X ω) (R.unit i)))) ∨
      Hash.CollisionIn H (R.sampledHashed P H key sampled X ω) := by
    rintro ω ⟨hacc, i, hi, hd, hlt, hne⟩
    by_cases hc : Hash.CollisionIn H (R.hashedAt H key i (P.localEval (X ω) (R.unit i)))
    · exact Or.inr (collisionIn_mono hc fun m hm =>
        List.mem_flatMap.2 ⟨i, Finset.mem_toList.2 (Finset.mem_filter.2 ⟨hi, hd⟩), hm⟩)
    · exact Or.inl ⟨hacc, i, hi, hd, hlt, hne, hc⟩
  have hfind : Claims.Pr (fun ω => Hash.CollisionIn H (R.sampledHashed P H key sampled X ω)) ≤
      (R.finder P H key sampled X cost).Pr (fun m => (H m).bytes) := by
    unfold Claims.Finder.Pr
    apply Claims.Pr_mono
    intro ω h
    obtain ⟨x, hx, y, hy, hxy⟩ := h
    have h' : ∃ x ∈ R.sampledHashed P H key sampled X ω, ∃ y ∈ R.sampledHashed P H key sampled X ω,
        Hash.Collision (fun m => (H m).bytes) x y := ⟨x, hx, y, hy, hxy⟩
    refine ⟨h'.choose, h'.choose_spec.2.choose, ?_, h'.choose_spec.2.choose_spec.2⟩
    simp [Reads.finder, h']
  exact (Claims.Pr_mono hsplit).trans ((pr_or_le _ _).trans
    (add_le_add (readUnitSampled H key consts P R acc sampled X ε hgates hcommitted below) (hfind.trans hcr)))

end ResourceAccounting.SecurityProofs
