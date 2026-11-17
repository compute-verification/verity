module

public import Proofs.Flock.Claims.Recursion.NextRRows
public import Proofs.Flock.Claims.Recursion.VerdictRows
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
import Proofs.Flock.Claims.Recursion.VerdictRowG

/-! The facts of a decision itself that `DecAt.runs_assembled` takes, which every decision makes hold. -/

@[expose] public section

namespace RecDeploy.Rows

open Flock
open FlockDeploy (Coin FinalP pointsOf)

namespace DecAt

variable {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin}
  (D : DecAt rd x ms cs)

theorem points : D.sess.points = pointsOf D.spec D.d.c₁ :=
  (FlockDeploy.decodeJWith_sub _ D.decode).1

theorem ps_size : D.d.f.ps.size = 2 * D.sts.size := by
  obtain ⟨-, -, -, -, h, -⟩ := FlockDeploy.decR_fits D.dec
  rw [D.setup_eq.2]
  exact h

theorem rep_lt {i : ℕ} (hi : i < D.sts.size) (r : Fin 2) : 2 * i + r < D.d.f.ps.size := by
  rw [D.ps_size]
  omega

theorem tables_size : D.spec.tables.size = D.sts.size := by
  obtain ⟨⟨s, spec, sts⟩, c₁, r₀, rss, fl, link, pf, hv, -, -, -, -, -, hd⟩ := FlockDeploy.decR_some D.dec
  have h := (FlockDeploy.setup_tables hv).2
  obtain ⟨hs, ht⟩ := D.setup_eq
  rw [hs, ht, hd]
  exact h

theorem lt_tables {i : ℕ} (hi : i < D.sts.size) : i < D.spec.tables.size := by
  rw [D.tables_size]
  exact hi

theorem sts_dense {xo : Claims.Msg} {sto : Claims.Recursion.CompiledStatement} (hst : parse xo = some sto)
    (hf : Claims.Recursion.Form.Dense sto.form) (hx : x = xIOf xo) : D.sts.size = 1 := by
  obtain ⟨s, spec, st, -, hset, -, -, -⟩ := dense_inner hst hf
  obtain ⟨⟨s', spec', sts'⟩, c₁, r₀, rss, fl, link, pf, hv, -, -, -, -, -, hd⟩ := FlockDeploy.decR_some D.dec
  rw [hx, hset] at hv
  cases hv
  rw [D.setup_eq.2, hd]
  rfl

theorem zero_lt {xo : Claims.Msg} {sto : Claims.Recursion.CompiledStatement} (hst : parse xo = some sto)
    (hf : Claims.Recursion.Form.Dense sto.form) (hx : x = xIOf xo) : 0 < D.sts.size := by
  rw [D.sts_dense hst hf hx]
  exact Nat.one_pos

theorem zero_lt_tables {xo : Claims.Msg} {sto : Claims.Recursion.CompiledStatement} (hst : parse xo = some sto)
    (hf : Claims.Recursion.Form.Dense sto.form) (hx : x = xIOf xo) : 0 < D.spec.tables.size :=
  D.lt_tables (D.zero_lt hst hf hx)

end DecAt

end RecDeploy.Rows
