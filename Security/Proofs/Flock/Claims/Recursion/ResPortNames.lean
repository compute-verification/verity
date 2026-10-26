import Proofs.Flock.Recursive.VStarCheck
import Proofs.Flock.VBridge.AlgHolds
import Proofs.Flock.C3.Reps.ReadWhole
import Proofs.Flock.VBridge.Family.PartKeys

/-!
# V\*'s part's row ports are the verifier's group's, by name and size (`checkPart_ports`)

At a part's statement `.inner j nm pt` whose circuit passes `RecVStar.checkPart`, at the shape, holo flag and part `jj`
the unit's name and META give, V\*'s part's row ports `pt.S.ports` are the verifier's group `jj`'s row ports `P`
(`RecResiduals.groups (structureOf sh) (portSizes sh)`), each named as the whole S names it (`RecVStar.portNames`:
round `r`'s `m<r>`, then `acc0` and `acc1`) and of its size there (`RecResiduals.portSizes`). `checkPart` compares
both: the sizes against `RecResiduals.part`'s, and the names at `RecVStar.partPorts`, which is the group's `P`.

* `checkPart_ports`: the ports as a list over the group's `P`.
* `checkPart_port`: port `i`, for `i` below the group's `P`'s size, is `(portNames[P[i]]!, portSizes[P[i]]!)`.
* `portNames_round`: a round's port is named `m<r>`.
-/

namespace RecDeploy.ResLower

open Flock

theorem map_pair_of {α β γ : Type} {f : α → β} {g : α → γ} {l : List (β × γ)} {A : List α}
    (h1 : l.map (·.1) = A.map f) (h2 : l.map (·.2) = A.map g) : l = A.map fun a => (f a, g a) := by
  induction l generalizing A with
  | nil =>
    cases A with
    | nil => rfl
    | cons _ _ => simp at h1
  | cons x l ih =>
    cases A with
    | nil => simp at h1
    | cons a A =>
      simp only [List.map_cons, List.cons.injEq] at h1 h2
      rw [List.map_cons, ih (A := A) h1.2 h2.2, show x = (f a, g a) from Prod.ext h1.1 h2.1]

/-- **V\*'s part's row ports are the verifier's group's, by name and size**: at a part's statement whose circuit passes
the part check, at the unit name's part `jj`, the shape and the META holo flag, S and its groups return, and V\*'s
part's row ports are group `jj`'s row ports, each named and sized as the whole S's. -/
theorem checkPart_ports {j : ℕ} {nm : String} {pt : VStar.Part} {c : Circuit}
    (h : RecVStar.checkPart (.inner j nm pt) c = .ok ()) {m k cl jj : ℕ} {sh : Shape} {holo : Bool}
    {S : RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (h1 : RecResiduals.params c.unit.name = some (m, k, cl, jj)) (h2 : RecResiduals.shape? m k cl = some sh)
    (h3 : RecResiduals.holoOf c = .ok holo) (h4 : RecResiduals.part { sh with holo } jj = .ok (S, ns, nv)) :
    ∃ GS gs, RecAlgebra.structureOf { sh with holo } = .ok GS ∧
      RecResiduals.groups GS (RecResiduals.portSizes { sh with holo }) = .ok gs ∧ ∃ hj : jj < gs.size,
        pt.S.ports = gs[jj].2.1.toList.map fun p =>
          ((RecVStar.portNames { sh with holo })[p]!, (RecResiduals.portSizes { sh with holo })[p]!) := by
  obtain ⟨GS, gs, hS, hg⟩ := FlockC3.Reps.ReadWhole.part_ok h4
  have hj := FlockVBridge.OfVStar.part_lt hS hg h4
  obtain ⟨-, hnsE, -⟩ := FlockVBridge.OfVStar.part_eq hS hg hj h4
  obtain ⟨hns, -, hnm⟩ := FlockSoundness.Discharge.Recursion.checkPart_ok h rfl h1 h2 h3 h4
  have hnm' := hnm _ (FlockVBridge.OfVStar.partPorts_eq hS hg hj)
  exact ⟨GS, gs, hS, hg, hj, map_pair_of hnm' (hns.trans hnsE)⟩

/-- **V\*'s part's port `i`** is group `jj`'s port `P[i]`, named and sized as the whole S's. -/
theorem checkPart_port {j : ℕ} {nm : String} {pt : VStar.Part} {c : Circuit}
    (h : RecVStar.checkPart (.inner j nm pt) c = .ok ()) {m k cl jj : ℕ} {sh : Shape} {holo : Bool}
    {S : RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (h1 : RecResiduals.params c.unit.name = some (m, k, cl, jj)) (h2 : RecResiduals.shape? m k cl = some sh)
    (h3 : RecResiduals.holoOf c = .ok holo) (h4 : RecResiduals.part { sh with holo } jj = .ok (S, ns, nv)) :
    ∃ GS gs, RecAlgebra.structureOf { sh with holo } = .ok GS ∧
      RecResiduals.groups GS (RecResiduals.portSizes { sh with holo }) = .ok gs ∧ ∃ hj : jj < gs.size,
        pt.S.ports.length = gs[jj].2.1.size ∧ ∀ (i : ℕ) (hi : i < gs[jj].2.1.size),
          pt.S.ports[i]? = some ((RecVStar.portNames { sh with holo })[gs[jj].2.1[i]]!,
            (RecResiduals.portSizes { sh with holo })[gs[jj].2.1[i]]!) := by
  obtain ⟨GS, gs, hS, hg, hj, hp⟩ := checkPart_ports h h1 h2 h3 h4
  refine ⟨GS, gs, hS, hg, hj, by rw [hp]; simp, fun i hi => ?_⟩
  rw [hp]
  simp [hi]

/-- **A round's row port is named `m<r>`** in the whole S (`portNames`). -/
theorem portNames_round {sh : Shape} {r : ℕ} (hr : r < (RecAlgebra.rounds sh).size) :
    (RecVStar.portNames sh)[r]! = s!"m{r}" := by
  rw [RecVStar.portNames, getElem!_pos _ r (by simp only [Array.size_append, Array.size_map, Array.size_range,
      List.size_toArray, List.length_cons, List.length_nil]; omega),
    Array.getElem_append_left (by simp only [Array.size_map, Array.size_range]; omega), Array.getElem_map,
    Array.getElem_range]

end RecDeploy.ResLower
