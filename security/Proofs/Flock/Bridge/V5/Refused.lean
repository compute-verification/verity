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

/-!
# `SampledKSLiveV5R`'s roots guard, read off `verify --zk`'s acceptance

`verify --zk` refuses a public file two of whose ports' published roots are one root under different published domains
(`Flock.PortRoots.collidesPub`), with or without shared rows. Each port's published root (`PortRoots.pubRoot`) is over
its published domain and its table's rows, or one row per unit, at its schema: the root `rootsOfL` gives it at the
public file's leaf map. So the refusal is `RootFixesDomainL` at that leaf map, the guard of V5's bound
`SampledKSLiveV5R` (V5 itself pays for it with `ecr/sha-512` at the roots finder).

* `rootsOfL_ofPublic_getD`: `rootsOfL` at `LeafMap.ofPublic` over the file's population is `pubRoot`'s.
* `fixesL_of_collidesPub_false`: `collidesPub = false` is `RootFixesDomainL` there.
-/

@[expose] public section

namespace Flock.Bridge

open Properties.NetworkCertifier.Bytes (bs bs_inj)

/-- **The verifier's published roots are `rootsOfL`'s at the public file's leaf map**: port `q`'s published root
(`Flock.PortRoots.pubRoot`) is `rootsOfL`'s at `LeafMap.ofPublic` over the file's population. -/
theorem rootsOfL_ofPublic_getD (c : Flock.Circuit) (pub : Flock.Public) {q : ℕ} (hq : q < c.ports.size) :
    (rootsOfL c (LeafMap.ofPublic c pub pub.n)).getD q [] = bs (Flock.PortRoots.pubRoot c pub q) := by
  have hg : (rootsOfL c (LeafMap.ofPublic c pub pub.n)).getD q [] = bs (Flock.FV3.root
      ((LeafMap.ofPublic c pub pub.n).dom q) (leavesL c (LeafMap.ofPublic c pub pub.n) q) Sha512.hash) := by
    simp [rootsOfL, List.getD_eq_getElem?_getD, hq]
  rw [hg]
  unfold Flock.PortRoots.pubRoot Flock.PortRoots.pubRootWith Flock.PortRoots.countOf Flock.PortRoots.digOf
    LeafMap.ofPublic leavesL
  cases pub.tables <;> rfl

/-- **No colliding pair in the verifier's published roots fixes each root's domain**: `RootFixesDomainL` at the public
file's leaf map over its population, with or without shared rows. -/
theorem fixesL_of_collidesPub_false {c : Flock.Circuit} {pub : Flock.Public}
    (h : Flock.PortRoots.collidesPub c pub = false) : RootFixesDomainL c (LeafMap.ofPublic c pub pub.n) := by
  intro q q' hlt hq he
  rw [rootsOfL_ofPublic_getD c pub (hlt.trans hq), rootsOfL_ofPublic_getD c pub hq] at he
  rw [LeafMap.ofPublic_dom]
  exact Flock.PortRoots.idOf_eq_of_collidesPub_false h hlt hq (bs_inj he)

end Flock.Bridge
