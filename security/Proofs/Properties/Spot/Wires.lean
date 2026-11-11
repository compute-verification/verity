module

public import Properties.Spot.Wires
public import Spot.WiresSound

/-! # Proof that the dataflow a wiring check accepts is acyclic (`Properties.Spot.Wires`)

A dependency of the dataflow (`Population.Dataflow.Dep`) is an output row `(template, unit, output)` one unit writes and
another reads through an entry of a wire, so it is that entry's edge of `Wiring.Reads` (`Wires.dep_reads`); a chain of
dependencies is a chain of those edges (`Wires.transGen_reads`), and `check_acyclic` refuses one from a unit back to
itself.

The honest witness is `Wires.honest`, a template whose second unit reads its first under a stated order, the shape of a
window's circuit; the nontrivial one is `Wires.selfRead`, a unit wired to read its own output, whose dataflow has a cycle.
-/

@[expose] public section

open Spot.Wires

namespace Properties.Spot

namespace Wires

theorem dep_reads {c : Wiring} {x y : String × Int} (h : (wiresDataflow c).Dep x y) : c.Reads x y := by
  unfold Population.Dataflow.Dep at h
  obtain ⟨s, hw, hr⟩ := h
  simp only [wiresDataflow, List.mem_toFinset, List.mem_map] at hw
  obtain ⟨o, -, rfl⟩ := hw
  simp only [wiresDataflow, List.mem_toFinset, List.mem_filterMap, List.mem_flatMap] at hr
  obtain ⟨e, ⟨w, hwc, he⟩, hey⟩ := hr
  split at hey
  · rename_i hy
    refine ⟨w, hwc, e, he, ?_, hy.symm⟩
    rw [Option.some.inj hey]
  · cases hey

theorem transGen_reads {c : Wiring} {x y : String × Int} (h : Relation.TransGen (wiresDataflow c).Dep x y) :
    Relation.TransGen c.Reads x y := by
  induction h with
  | single h => exact .single (dep_reads h)
  | tail _ h ih => exact .tail ih (dep_reads h)

end Wires

open Wires

theorem wiresAcyclic : WiresAcyclic := fun _ _ h u hu => check_acyclic h u (transGen_reads hu)

theorem wiresAcyclic_honest : ∃ S, check honest = .ok S := by
  have hok : orderOk honest [("g", 0, 1), ("g", 1, 1)] = true := by
    simp (disch := decide) only [orderOk, honest, List.all_cons, List.all_nil, Bool.and_true, List.filter_cons,
      List.filter_nil, List.mergeSort_of_pairwise]
    decide
  have hof : orderOf honest = .ok [("g", 0, 1), ("g", 1, 1)] := by
    have h : honest.order = some [("g", 0, 1), ("g", 1, 1)] := rfl
    simp only [orderOf, statedOrder, h, hok, ite_true]
  have hc : cutsOk honest = true := by decide
  have hf : (firstWire honest (wireCode honest)).isNone = true := by decide
  have hd : pairwiseDisjoint honest.wires = true := by decide
  have ho : (orderCode honest orderedWire).isNone = true := by
    simp only [orderCode, hof]; decide
  exact ⟨_, by simp only [check, hc, Option.isNone_iff_eq_none.1 hf, hd, Option.isNone_iff_eq_none.1 ho]; rfl⟩

theorem wiresAcyclic_nontrivial : ¬ (wiresDataflow selfRead).Acyclic := fun h =>
  h ("g", 0) (.single ⟨("g", 0, "o"), by simp only [wiresDataflow, List.mem_toFinset]; decide,
    by simp only [wiresDataflow, List.mem_toFinset]; decide⟩)

end Properties.Spot
