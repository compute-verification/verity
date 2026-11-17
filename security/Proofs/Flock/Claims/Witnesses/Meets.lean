module

public import Definitions.Claims.Sampling
public import Proofs.Flock.Claims.DeployCode
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
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.DeployEfficient
import Std.Tactic.BVDecide.Normalize.Prop

/-! What the claims ask of a record holds at the deployed ones; that some bytes parse to the readable scope's statement is beyond Lean, since `HmRow.parse` regenerates the `hm96` nets beyond what the kernel evaluates. -/

@[expose] public section

open FlockDeploy

local notation "𝔽" => FlockLevel3.GF128

namespace Claims

/-- **The plain record's decoder meets the list-decoding spec** at the record's code, length and radius. -/
theorem Flock.decoder_meets_plain (rel : Flock.Readable.Parsed → Flock.Readable.Relation) :
    Claims.Flock.ListWithin (deployPlain rel).len (deployPlain rel).enc (deployPlain rel).θ
      (deployPlain rel).decode := by
  rw [show (deployPlain rel).decode = Claims.GS.plainDecodeRD from decode_plain rel]
  exact listWithin_plain rel

/-- **The masked record's decoder meets the list-decoding spec** at the record's code, length and radius. -/
theorem Flock.decoder_meets_zk (rel : Flock.Readable.Parsed → Flock.Readable.Relation) :
    Claims.Flock.ListWithin (deployZk rel).len (deployZk rel).enc (deployZk rel).θ (deployZk rel).decode := by
  rw [show (deployZk rel).decode = Claims.GS.zkDecodeRD from decode_zk rel]
  exact listWithin_zk rel

/-- **The masked record is the masked verifier.** -/
theorem Flock.masked_meets_zk (rel : Flock.Readable.Parsed → Flock.Readable.Relation) : (deployZk rel).zk = true :=
  rfl

def Flock.scopeStmt (m : ℕ) : Claims.Flock.Stmt :=
  ⟨m, 6, ⟨0, fun g => g.elim0, fun g => g.elim0, ∅⟩, fun _ _ => .pub 0, fun _ _ => .pub 0, [],
    ⟨true, false, true, "hm96-sha512/v1", 1, false, false, false, false, 1, false, 0, true, true, false, false, false⟩⟩

/-- **The plain record's readable scope holds a statement with a witness in its relation.** -/
theorem Flock.scope_meets_plain (rel : Flock.Readable.Parsed → Flock.Readable.Relation) :
    ∃ (s : Claims.Flock.Stmt) (w : Claims.Flock.Wit), Claims.Flock.InScope (deployPlain rel).zk s ∧
      Claims.Flock.RelS s w :=
  ⟨Flock.scopeStmt 22, ⟨[]⟩, by
    show Claims.Flock.InScope false _
    unfold Claims.Flock.InScope
    decide, fun _ _ o => o.elim0⟩

/-- **The masked record's readable scope holds a statement with a witness in its relation**, at `2^25` trace bits. -/
theorem Flock.scope_meets_zk (rel : Flock.Readable.Parsed → Flock.Readable.Relation) :
    ∃ (s : Claims.Flock.Stmt) (w : Claims.Flock.Wit), Claims.Flock.InScope (deployZk rel).zk s ∧
      Claims.Flock.RelS s w :=
  ⟨Flock.scopeStmt 25, ⟨[]⟩, by
    show Claims.Flock.InScope true _
    unfold Claims.Flock.InScope
    decide, fun _ _ o => o.elim0⟩

def Sampling.scopeStmt : Claims.Sampling.AuditStatement := ⟨[], ⟨[.input], [0]⟩, ⟨"Q_word", 1, []⟩, ⟨[.all]⟩, []⟩

def Sampling.scopePartition : Claims.Sampling.Partition := ⟨[[[[0]]]]⟩

/-- **The sampling scope holds a statement**: its readable form reads no field of the record. -/
theorem Sampling.scope_meets :
    ∃ (P : Claims.Sampling.Partition) (s : Claims.Sampling.AuditStatement), Claims.Sampling.InScope P s := by
  refine ⟨Sampling.scopePartition, Sampling.scopeStmt, ⟨?_, ?_⟩,
    ⟨by decide, by decide, by decide, by decide, by decide⟩, rfl, ?_⟩
  · intro g x hg i hi
    cases g with
    | zero =>
      simp only [Sampling.scopeStmt, List.getElem?_cons_zero, Option.some.injEq] at hg
      subst hg
      simp [Claims.Sampling.BGate.reads] at hi
    | succ n => simp [Sampling.scopeStmt] at hg
  · intro o ho
    simp [Sampling.scopeStmt] at ho ⊢
    exact ho
  · simp [Sampling.scopeStmt, Sampling.scopePartition, Claims.Sampling.OneStage.WellFormed]

/-- **A private circuit of a size class**: the class of one gate over two inputs and one output, and a private circuit
of it, the AND of the two inputs, on inputs of its width. -/
theorem Sampling.universal_meets :
    ∃ (c : Claims.Sampling.SizeClass) (n : Claims.Sampling.PrivateCircuit) (x : List Bool), n.Fits c ∧ x.length = c.nIn :=
  ⟨⟨1, 2, 1⟩, ⟨2, [(1, 0, 1)], [2]⟩, [true, false], by unfold Claims.Sampling.PrivateCircuit.Fits; decide, rfl⟩

/-- **At a sampling record over the plain record**, its Flock record's decoder meets the list-decoding spec. -/
theorem Sampling.decoder_meets_plain (rel : Flock.Readable.Parsed → Flock.Readable.Relation)
    (D : Claims.Sampling.Deploy Coin (List 𝔽) Claims.Flock.PlainTrace) (hD : D.flock = deployPlain rel) :
    Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ D.flock.decode := by
  rw [hD]
  exact Claims.Flock.decoder_meets_plain rel

/-- **At a sampling record over the masked record**, its Flock record's decoder meets the list-decoding spec. -/
theorem Sampling.decoder_meets_zk (rel : Flock.Readable.Parsed → Flock.Readable.Relation)
    (D : Claims.Sampling.Deploy Coin (List 𝔽) Claims.Flock.ZkTrace) (hD : D.flock = deployZk rel) :
    Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ D.flock.decode := by
  rw [hD]
  exact Claims.Flock.decoder_meets_zk rel

/-- **At a sampling record over the masked record**, its Flock record is the masked verifier. -/
theorem Sampling.masked_meets_zk (rel : Flock.Readable.Parsed → Flock.Readable.Relation)
    (D : Claims.Sampling.Deploy Coin (List 𝔽) Claims.Flock.ZkTrace) (hD : D.flock = deployZk rel) :
    D.flock.zk = true := by
  rw [hD]
  rfl

end Claims
