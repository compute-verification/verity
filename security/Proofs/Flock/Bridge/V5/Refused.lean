module

public import Proofs.Flock.Bridge.LeafMap
public import Flock.KsScope
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

/-! `SampledKSLiveV5R`'s roots guard (`RootFixesDomainL`), read off `verify --zk`'s refusal of colliding
published roots. -/

@[expose] public section

namespace Flock.Bridge

open Properties.Certifier.Bytes (bs bs_inj)

theorem rootsOfL_ofPublic_getD (c : Flock.Circuit) (pub : Flock.Public) {q : ℕ} (hq : q < c.ports.size) :
    (rootsOfL c (LeafMap.ofPublic c pub pub.n)).getD q [] = bs (Flock.PortRoots.pubRoot c pub q) := by
  have hg : (rootsOfL c (LeafMap.ofPublic c pub pub.n)).getD q [] = bs (Flock.FV3.root
      ((LeafMap.ofPublic c pub pub.n).dom q) (leavesL c (LeafMap.ofPublic c pub pub.n) q) Sha512.hash) := by
    simp [rootsOfL, List.getD_eq_getElem?_getD, hq]
  rw [hg]
  unfold Flock.PortRoots.pubRoot Flock.PortRoots.pubRootWith Flock.PortRoots.countOf Flock.PortRoots.digOf
    LeafMap.ofPublic leavesL
  cases pub.tables <;> rfl

theorem fixesL_of_collidesPub_false {c : Flock.Circuit} {pub : Flock.Public}
    (h : Flock.PortRoots.collidesPub c pub = false) : RootFixesDomainL c (LeafMap.ofPublic c pub pub.n) := by
  intro q q' hlt hq he
  rw [rootsOfL_ofPublic_getD c pub (hlt.trans hq), rootsOfL_ofPublic_getD c pub hq] at he
  rw [LeafMap.ofPublic_dom]
  exact Flock.PortRoots.idOf_eq_of_collidesPub_false h hlt hq (bs_inj he)

end Flock.Bridge
