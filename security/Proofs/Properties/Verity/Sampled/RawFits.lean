module

public import Proofs.Flock.Claims.Complete.Encode
import all Init.Data.String.Defs
import all Init.Data.String.TakeDrop
import all Init.Data.String.Slice
import all Init.Data.String.Pattern.String
import all Init.Data.String.Pattern.Basic
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

/-! # Flock's statement reads back as itself where its lengths and numbers fit their 8-byte words -/

@[expose] public section

namespace Verity.Recursive.Part

open FlockDeploy (Raw encNat encBytes encStr encBool encOpt encList Rd rdTake rdNat rdBytes rdStr rdBool rdOpt rdList)

theorem rdBool_encBool (b : Bool) (rest : List UInt8) : rdBool.run (encBool b ++ rest) = some (b, rest) := by
  cases b <;> rfl

theorem rdOpt_encOpt {α : Type} {r : Rd α} {e : α → Claims.Msg} {a : Option α}
    (hr : ∀ v ∈ a, ∀ rest, r.run (e v ++ rest) = some (v, rest)) (rest : List UInt8) :
    (rdOpt r).run (encOpt e a ++ rest) = some (a, rest) := by
  cases a with
  | none => rfl
  | some v =>
    have h := hr v rfl rest
    show (rdOpt r).run (encBool true ++ (e v ++ rest)) = some (some v, rest)
    simp only [rdOpt, StateT.run_bind, rdBool_encBool, Option.bind_eq_bind, Option.bind_some, ite_true,
      StateT.run_map, h]
    rfl

theorem format_fits : FlockDeploy.FORMAT.toUTF8.size < 2 ^ 64 := by decide

/-- **A statement's lengths and numbers fit their 8-byte words** (`encNat`). -/
def RawFits (w : Raw) : Prop :=
  w.statement.toUTF8.size < 2 ^ 64 ∧ w.circuit.size < 2 ^ 64 ∧ w.publicFile.size < 2 ^ 64 ∧
    w.tables.length < 2 ^ 64 ∧ (∀ p ∈ w.tables, p.1.toUTF8.size < 2 ^ 64 ∧ p.2.size < 2 ^ 64) ∧ w.count < 2 ^ 64 ∧
    (∀ b ∈ w.draw, b.size < 2 ^ 64) ∧ (∀ b ∈ w.partition, b.size < 2 ^ 64) ∧ (∀ b ∈ w.program, b.size < 2 ^ 64) ∧
    (∀ n ∈ w.work, n < 2 ^ 64) ∧ (∀ b ∈ w.workTable, b.size < 2 ^ 64) ∧ (∀ n ∈ w.stratified, n < 2 ^ 64) ∧
    (∀ b ∈ w.closure, b.size < 2 ^ 64) ∧ (∀ b ∈ w.registered, b.size < 2 ^ 64) ∧ (∀ b ∈ w.publicInputs, b.size < 2 ^ 64)

theorem rdOptBytes {a : Option ByteArray} (h : ∀ b ∈ a, b.size < 2 ^ 64) (rest : List UInt8) :
    (rdOpt rdBytes).run (encOpt encBytes a ++ rest) = some (a, rest) :=
  rdOpt_encOpt (fun b hb rest => FlockComplete.rdBytes_encBytes (h b hb) rest) rest

theorem rdOptNat {a : Option Nat} (h : ∀ n ∈ a, n < 2 ^ 64) (rest : List UInt8) :
    (rdOpt rdNat).run (encOpt encNat a ++ rest) = some (a, rest) :=
  rdOpt_encOpt (fun n hn rest => FlockComplete.rdNat_encNat (h n hn) rest) rest

theorem run_bind_of {α β : Type} {x : Rd α} {f : α → Rd β} {s s' : Claims.Msg} {a : α}
    (h : x.run s = some (a, s')) : (x >>= f).run s = (f a).run s' := by
  rw [StateT.run_bind, h]
  rfl

theorem raw_read_encode {w : Raw} (h : RawFits w) (rest : List UInt8) :
    Raw.read.run (w.encode ++ rest) = some (w, rest) := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15⟩ := h
  have ht : ∀ rest, (rdList (do
      let n ← rdStr
      let b ← rdBytes
      pure (n, b))).run (encList (fun p => encStr p.1 ++ encBytes p.2) w.tables ++ rest) = some (w.tables, rest) :=
    fun rest => FlockComplete.rdList_encList (fun p hp rest => by
      rw [List.append_assoc, run_bind_of (FlockComplete.rdStr_encStr (h5 p hp).1 _),
        run_bind_of (FlockComplete.rdBytes_encBytes (h5 p hp).2 _)]
      rfl) h4 rest
  simp only [Raw.encode, List.append_assoc]
  unfold Raw.read
  rw [run_bind_of (FlockComplete.rdStr_encStr format_fits _)]
  dsimp only
  rw [if_neg (fun h => h rfl)]
  rw [run_bind_of (FlockComplete.rdStr_encStr h1 _), run_bind_of (FlockComplete.rdBytes_encBytes h2 _),
    run_bind_of (FlockComplete.rdBytes_encBytes h3 _), run_bind_of (ht _), run_bind_of (rdBool_encBool _ _),
    run_bind_of (rdBool_encBool _ _), run_bind_of (FlockComplete.rdNat_encNat h6 _), run_bind_of (rdOptBytes h7 _),
    run_bind_of (rdOptBytes h8 _), run_bind_of (rdOptBytes h9 _), run_bind_of (rdOptNat h10 _),
    run_bind_of (rdOptBytes h11 _), run_bind_of (rdOptNat h12 _), run_bind_of (rdOptBytes h13 _),
    run_bind_of (rdOptBytes h14 _), run_bind_of (rdOptBytes h15 _)]
  rfl

/-- **Flock's statement reads back as itself** where it fits. -/
theorem decode_encode_fits {w : Raw} (h : RawFits w) : FlockDeploy.decode w.encode = some w := by
  have hr := raw_read_encode h []
  rw [List.append_nil] at hr
  unfold FlockDeploy.decode
  rw [hr]
  simp

end Verity.Recursive.Part
