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
# `SampledKSLiveV5`'s shared-row file hypotheses, read off `verify --zk`'s acceptance

`verify --zk` admits a shared-row public file only in the form `SampledKSLiveV5` covers (`Flock.KsScope.sharedWhy`):
the circuit's public input ports parse (`HmIn.parse`), the file loads over them (`HmIn.loadPublic`) with its tables,
and no two ports' roots over those tables are one root under different domains (`KsScope.collidesT`). Those are V5's
hypotheses `HmIn.parse` and `HmIn.loadPublic`, and `RootFixesDomainL` at the file's leaf map, the guard of V5's bound
`SampledKSLiveV5R` (V5 itself pays for it with `ecr/sha-512` at the roots finder).

* `fixesL_of_collidesT_false`: `collidesT = false` is `RootFixesDomainL` at the tables' leaf map (`LeafMap.ofTables`):
  the roots `collidesT` computes, each port's at its schema over its table's rows, are `rootsOfL`'s.
* `v5Files_of_sharedOk`: a file `KsScope.SharedOk` admits gives those three file hypotheses, at every `n`.
* `v5Files_of_check`: `KsScope.checkOf` passing at a shared-row file gives them, at the circuit it checked.
-/

@[expose] public section

namespace Flock.Bridge

open Properties.NetworkCertifier.Bytes (bs bs_inj)

theorem rootsOfL_ofTables_getD (c : Flock.Circuit) (pub : Flock.Public) (t : Flock.Tables) {q : ℕ}
    (hq : q < c.ports.size) :
    (rootsOfL c (LeafMap.ofTables c pub t)).getD q [] = bs (Flock.KsScope.rootT c pub t q) := by
  simp [rootsOfL, List.getD_eq_getElem?_getD, hq]
  rfl

/-- **No colliding pair over a file's tables fixes each root's domain**: `RootFixesDomainL` at the tables' leaf map. -/
theorem fixesL_of_collidesT_false {c : Flock.Circuit} {pub : Flock.Public} {t : Flock.Tables}
    (h : Flock.KsScope.collidesT c pub t = false) : RootFixesDomainL c (LeafMap.ofTables c pub t) := by
  intro q q' hlt hq he
  rw [rootsOfL_ofTables_getD c pub t (hlt.trans hq), rootsOfL_ofTables_getD c pub t hq] at he
  exact Flock.KsScope.domOf_eq_of_collidesT_false h hlt hq (bs_inj he)

/-- **`KsScope`'s shared-row form is V5's**: the public input ports parse, the file loads over them with its tables,
and `RootFixesDomainL` holds at its leaf map, at every `n`. -/
theorem v5Files_of_sharedOk {tags : Flock.Tags} {c : Flock.Circuit} {pf : ByteArray}
    (h : Flock.KsScope.SharedOk tags c pf) :
    ∃ pi pub t, Flock.HmIn.parse c = .ok pi ∧ Flock.HmIn.loadPublic tags c pi pf = .ok pub ∧ pub.tables = some t ∧
      ∀ n, RootFixesDomainL c (LeafMap.ofPublic c pub n) := by
  obtain ⟨pi, pub, t, hp, hl, ht, hc⟩ := h
  exact ⟨pi, pub, t, hp, hl, ht, fun n => by rw [LeafMap.ofPublic_some n ht]; exact fixesL_of_collidesT_false hc⟩

/-- **`verify --zk`'s form check at a shared-row file gives V5's file hypotheses**: when `KsScope.checkOf` passes at a
circuit (a templated statement's carrying its template) and the public file as `HmOut.loadPublic` loads it has tables,
V5's `HmIn.parse`, `HmIn.loadPublic` and `RootFixesDomainL` hold at that circuit. -/
theorem v5Files_of_check {tags : Flock.Tags} {pf : ByteArray} {count : ℕ} {draw : Option Lean.Json}
    {c : Flock.Circuit} {pub : Flock.Public} {t : Flock.Tables}
    (h : Flock.KsScope.checkOf (.ok c) tags pf count draw = .ok ()) (hpub : Flock.HmOut.loadPublic tags c pf = .ok pub)
    (ht : pub.tables = some t) :
    ∃ pi pub' t', Flock.HmIn.parse c = .ok pi ∧ Flock.HmIn.loadPublic tags c pi pf = .ok pub' ∧
      pub'.tables = some t' ∧ ∀ n, RootFixesDomainL c (LeafMap.ofPublic c pub' n) := by
  obtain ⟨-, -, hf, -⟩ := Flock.KsScope.checkOf_ok h rfl hpub
  rcases hf with hn | hok
  · rw [ht] at hn; cases hn
  · exact v5Files_of_sharedOk hok

end Flock.Bridge
