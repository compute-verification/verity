module

public import Proofs.Flock.Claims.Complete.Json
public import Proofs.Flock.Claims.Complete.Json.Goals
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
# Custody's round count at the honest run

`custodyCount_goal`: the goal `CustodyCount`, under the hypotheses of the record's goals (`Claims.Out64`,
`ProverOk`). The round messages are the sends after the link, each with its coin (`Run.mids_eq`), and those sends are
the live session's rounds, one each (`Run.json`, `Recs.length_eq`).
-/

@[expose] public section

namespace FlockComplete.JsonRT

open Flock FlockComplete FlockDeploy

/-- **The round messages and their coins**: the sends after the first, the `j`th with coin `1 + j`. -/
theorem Run.mids_eq {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation} (o : Run H rel)
    {b : Claims.Msg} {bs : List Claims.Msg} (hbs : o.bodies.toList = b :: bs) :
    o.mids = bs.zip ((List.range bs.length).map fun j => Claims.coinSeq o.r (1 + j)) := by
  have hn : o.bodies.size = bs.length + 1 := by
    rw [← Array.length_toList, hbs, List.length_cons]
  unfold Run.mids Run.sends
  rw [hbs, hn, List.range_succ_eq_map]
  simp only [List.map_cons, List.cons_append, midsOf, List.map_map]
  rw [List.dropLast_concat]
  have : ((List.range (bs.length + 1)).map (Claims.coinSeq o.r ∘ Nat.succ)).dropLast =
      (List.range bs.length).map fun j => Claims.coinSeq o.r (1 + j) := by
    rw [range_succ_dropLast]
    congr 1
    funext j
    simp [Nat.add_comm]
  rw [this]
  congr 1
  simp [Function.comp_def]

/-- **Custody: the record has one round per round message**: the goal `CustodyCount`. -/
theorem custodyCount_goal {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (h64 : Claims.Out64 H) (hP : ProverOk H rel) : CustodyCount H rel := by
  intro o
  obtain ⟨st, rb, streams, hst, ht, htabs, hss, hpf, hex, hlink, ⟨bs, hbs, hrecs⟩, hrec, hdec, hsess, hlv, hlive⟩ :=
    Run.json h64 hP o
  rw [Run.mids_eq o hbs, hlive]
  have := Recs.length_eq hrecs
  simp only [Array.toList_map, List.flatMap_map, List.length_zip, List.length_map, List.length_range, Nat.min_self,
    beq_iff_eq]
  rw [← this, List.length_flatMap, List.length_flatMap]
  simp

end FlockComplete.JsonRT
