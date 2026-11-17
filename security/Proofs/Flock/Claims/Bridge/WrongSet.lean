module

public import Proofs.Flock.Soundness.Discharge.ZkLink.Public
public import Definitions.Claims.Sampling
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
import Std.Tactic.BVDecide.Normalize.Prop

/-! The claims' incorrect set inside the game's wrong set. -/

@[expose] public section

namespace FlockSoundness.Discharge.ZkLink.WrongSet

open Game Model Accounting Audit ZkSession

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] {mPts : ℕ}
variable (A : Arith F K) {C : Verity.Boolean.Circuit} {n : ℕ} (P : Audit.Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)
  {decode : DecoderZK (C := C) planZ tab} {Dh : Type} {Hc : List UInt8 → Dh}
  (vb : ValueBindingZ A P planZ tab decode Hc)

def need (Cp : Claims.Sampling.Prog) (b : List ℕ) : List ℕ :=
  b ++ b.flatMap fun g => (Cp.gates[g]?.map Claims.Sampling.BGate.reads).getD []

def Lowers (Cp : Claims.Sampling.Prog) (b : List ℕ) (φ : ℕ → Fin C.N) (Xv : Fin C.N → Bool) : Prop :=
  ∀ ref : ℕ → Option Bool, (∀ g ∈ need Cp b, (∀ c, Cp.gates[g]? ≠ some (.const c)) → ref g = some (C.eval Xv (φ g))) →
    Cp.Correct ref b

def Binds (H : Claims.Msg → Claims.Msg) (Cm : Claims.Sampling.CommitmentScheme Bool) (s : Claims.Sampling.AuditStatement)
    (w : Claims.Sampling.Wit) (b : List ℕ)
    (φ : ℕ → Fin C.N) (R : Reg) (u : Fin n) : Prop :=
  ∀ op : vb.Pos → vb.Opening, (∀ g ∈ P.io u, vb.com (op (vb.pos g)) = vb.cs R (vb.pos g)) →
    ∀ g ∈ need s.prog b, (∀ c, s.prog.gates[g]? ≠ some (.const c)) →
      (g ∉ b ∨ s.prog.Boundary b g = true) →
      φ g ∈ P.io u ∧ w.val H Cm s g = some (vb.bit (vb.val (op (vb.pos (φ g)))) (φ g))

variable {A P planZ tab vb}

theorem not_faulty_of_rightRegZ {H : Claims.Msg → Claims.Msg} {Cm : Claims.Sampling.CommitmentScheme Bool}
    {s : Claims.Sampling.AuditStatement}
    {Pc : Claims.Sampling.Partition} {w : Claims.Sampling.Wit} {R : Reg} {Xv : Fin C.N → Bool} {u : Fin n} {v : ℕ}
    {b : List ℕ} {φ : ℕ → Fin C.N}
    (hr : Pc.blocks[v]? = some b) (hlow : Lowers s.prog b φ Xv) (hbind : Binds A P planZ tab vb H Cm s w b φ R u)
    (hu : RightRegZ A P planZ tab vb R Xv u) : ¬ Claims.Sampling.Faulty H Cm s Pc w v := by
  obtain ⟨-, op, hop⟩ := hu
  intro hf
  refine hf ⟨b, hr, fun g => C.eval Xv (φ g), hlow _ fun g hg hc => ?_⟩
  unfold Claims.Sampling.Prog.refIn
  split_ifs with hin
  · rfl
  · have hB : g ∉ b ∨ s.prog.Boundary b g = true := by
      by_cases h1 : g ∈ b
      · exact Or.inr (by simpa [h1] using hin)
      · exact Or.inl h1
    obtain ⟨hio, hw⟩ := hbind op (fun g hg => (hop g hg).1) g hg hc hB
    rw [hw, (hop (φ g) hio).2]

/-- `Lowers` and `Binds` are hypotheses here: `Lowers` reads no prover, and `Binds` is the event that the
witness's openings and the registration's agree, which a collision breaks. -/
theorem faulty_card_le_incorrectRegZ (H : Claims.Msg → Claims.Msg) (Cm : Claims.Sampling.CommitmentScheme Bool)
    (s : Claims.Sampling.AuditStatement)
    (Pc : Claims.Sampling.Partition) (w : Claims.Sampling.Wit) (R : Reg) (Xv : Fin C.N → Bool) (slot : Fin n → ℕ)
    (b : Fin n → List ℕ) (φ : ℕ → Fin C.N)
    (hcov : ∀ v < Pc.blocks.length, ∃ u, slot u = v) (hr : ∀ u, Pc.blocks[slot u]? = some (b u))
    (hlow : ∀ u, Lowers s.prog (b u) φ Xv) (hbind : ∀ u, Binds A P planZ tab vb H Cm s w (b u) φ R u) :
    (Claims.Sampling.faulty H Cm s Pc w).card ≤ (incorrectRegZ A P planZ tab vb R Xv).card := by
  classical
  calc (Claims.Sampling.faulty H Cm s Pc w).card ≤ ((incorrectRegZ A P planZ tab vb R Xv).image slot).card := by
        apply Finset.card_le_card
        intro v hv
        simp only [Claims.Sampling.faulty, Finset.mem_filter, Finset.mem_range] at hv
        obtain ⟨u, rfl⟩ := hcov v hv.1
        refine Finset.mem_image.2 ⟨u, (mem_incorrectRegZ A P planZ tab vb).2 fun h => ?_, rfl⟩
        exact not_faulty_of_rightRegZ (hr u) (hlow u) (hbind u) h hv.2
    _ ≤ _ := Finset.card_image_le

end FlockSoundness.Discharge.ZkLink.WrongSet
