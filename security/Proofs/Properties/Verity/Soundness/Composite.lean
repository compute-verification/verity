module

public import Properties.Verity.Game

/-!
# A composite read off its list

A composite is the list of its subprotocols' ids (`Verity.Registration.Composite`), and what the game reads of one is
a function of that list (`Properties.Verity.Game`): its subprotocols, each one's compiled check, the ties between them
and its verifier. These are the facts about those functions, at every list and at the two that have names.

* **At every list.** A subprotocol is one of a composite's exactly when the composite lists its id
  (`mem_subprotocols`), and its compiled check there is its own (`component_eq_some`), the same in every composite, one
  of the composite's components (`component_mem`, `mem_componentsOf`). The subprotocols' ids are distinct
  (`Subprotocol.id_injective`), so a composite names each of its components' ids once (`ids_nodup`) and a component is
  the check of the one subprotocol with its id (`component_of_id`, `component_of_onBlobs` on the blobs sent).
* **At onsite and remote.** The verifier is the code's: `Core.Onsite.protocol` and `Core.Remote.protocol`
  (`verifier_onsite`, `verifier_remote`), the composite the code builds from the list (`ofIds_onsite`, `ofIds_remote`)
  and lists beside it (`verifier_mem`), with the code's components (`componentsOf_onsite`, `componentsOf_remote`) and
  ties (`ties_onsite`, `ties_remote`). The spec's two lists are the code's components' ids (`ids_onsite`,
  `ids_remote`).
-/

@[expose] public section

namespace Verity

open Core

namespace Subprotocol

theorem mem_all (a : Subprotocol) : a ∈ Subprotocol.all := by cases a <;> decide

/-- The subprotocols' ids are distinct. -/
theorem ids_nodup : (Subprotocol.all.map (·.id)).Nodup := by decide

/-- A protocol id names at most one subprotocol. -/
theorem id_injective {a b : Subprotocol} (h : a.id = b.id) : a = b := by
  revert h
  cases a <;> cases b <;> decide

end Subprotocol

variable {k : Verity.Registration.Composite} {a : Subprotocol} {P : BlobProtocol}

/-- **Composite `k`'s compiled check for subprotocol `a` is `a`'s own**, where `k` lists `a`'s id. -/
theorem component_eq_some : component k a = some P ↔ a.id ∈ k ∧ P = a.protocol := by
  unfold component
  split
  next h => exact ⟨fun e => ⟨h, (Option.some.inj e).symm⟩, fun e => congrArg some e.2.symm⟩
  next h => exact ⟨fun e => (by cases e), fun e => absurd e.1 h⟩

/-- **A composite's subprotocols are the ones whose ids it lists.** -/
theorem mem_subprotocols : a ∈ subprotocols k ↔ a.id ∈ k := by
  unfold subprotocols component
  by_cases h : a.id ∈ k <;> simp [h]

theorem component_of_mem (h : a ∈ subprotocols k) : component k a = some a.protocol :=
  component_eq_some.2 ⟨mem_subprotocols.1 h, rfl⟩

theorem component_mem (h : component k a = some P) : P ∈ componentsOf k := by
  obtain ⟨hm, rfl⟩ := component_eq_some.1 h
  exact List.mem_map.2 ⟨a, List.mem_filter.2 ⟨a.mem_all, decide_eq_true hm⟩, rfl⟩

/-- Each of a composite's components is the compiled check of one of its subprotocols. -/
theorem mem_componentsOf (h : P ∈ componentsOf k) : ∃ a, component k a = some P := by
  obtain ⟨a, ha, rfl⟩ := List.mem_map.1 h
  exact ⟨a, component_eq_some.2 ⟨of_decide_eq_true (List.mem_filter.1 ha).2, rfl⟩⟩

/-- Each composite names each of its components' ids once. -/
theorem ids_nodup (k : Verity.Registration.Composite) : ((componentsOf k).map (·.id)).Nodup := by
  unfold componentsOf
  rw [List.map_map]
  exact Subprotocol.ids_nodup.sublist (List.filter_sublist.map _)

/-- **A component with subprotocol `a`'s id is `a`'s compiled check.** -/
theorem component_of_id (hP : P ∈ componentsOf k) (hid : P.id = a.id) : component k a = some P := by
  obtain ⟨b, hb⟩ := mem_componentsOf hP
  obtain ⟨-, rfl⟩ := component_eq_some.1 hb
  obtain rfl : b = a := Subprotocol.id_injective hid
  exact hb

/-- **A case at subprotocol `a`'s compiled check is `a`'s**: where one of composite `k`'s components on the blobs sent
is `a`'s compiled check on them, `a` is one of `k`'s subprotocols. -/
theorem component_of_onBlobs {s : Sent} {Q : Protocol} (hQ : Q ∈ s.components k)
    (h : Q = a.protocol.onBlobs (Blobs.ofArray s.blobs)) : component k a = some a.protocol := by
  obtain ⟨P, hP, rfl⟩ := List.mem_map.1 hQ
  have hc : component k a = some P := component_of_id hP (congrArg (·.id) h)
  exact hc.trans (congrArg some (component_eq_some.1 hc).2)

/-! ## Onsite and remote -/

theorem componentsOf_onsite : componentsOf .onsite = Onsite.components := rfl

theorem componentsOf_remote : componentsOf .remote = Remote.components := rfl

/-- Onsite's list is its components' ids, in the code's order. -/
theorem ids_onsite : Onsite.components.map (·.id) = Verity.Registration.Composite.onsite := by decide

/-- Remote's list is its components' ids, in the code's order. -/
theorem ids_remote : Remote.components.map (·.id) = Verity.Registration.Composite.remote := by decide

theorem tied_onsite : tied .onsite = Onsite.tied := rfl

theorem tied_remote : tied .remote = Remote.tied := rfl

/-- Onsite's ties are the code's: PoUW's windows on the policy's record, and PoUS and PoUB-P2 on one store. -/
theorem ties_onsite : ties .onsite = Onsite.ties := rfl

/-- Remote's tie is the code's: PoUW's windows on the policy's record. -/
theorem ties_remote : ties .remote = [Remote.windows] := rfl

/-- **Onsite's verifier is the code's**, `Core.Onsite.protocol`. -/
theorem verifier_onsite : verifier .onsite = Onsite.protocol := rfl

/-- **Remote's verifier is the code's**, `Core.Remote.protocol`. -/
theorem verifier_remote : verifier .remote = Remote.protocol := rfl

/-- **The code builds onsite's verifier from onsite's list.** -/
theorem ofIds_onsite : Composites.ofIds Verity.Registration.Composite.onsite = .ok (verifier .onsite) := rfl

/-- **The code builds remote's verifier from remote's list.** -/
theorem ofIds_remote : Composites.ofIds Verity.Registration.Composite.remote = .ok (verifier .remote) := rfl

/-- **Onsite's and remote's verifiers are the ones the composites' table lists**, each beside its list. -/
theorem verifier_mem (hk : k = .onsite ∨ k = .remote) : (verifier k, k) ∈ Composites.composites := by
  rcases hk with rfl | rfl
  · exact List.mem_cons.2 (Or.inl rfl)
  · exact List.mem_cons.2 (Or.inr (List.mem_cons.2 (Or.inl rfl)))

end Verity
