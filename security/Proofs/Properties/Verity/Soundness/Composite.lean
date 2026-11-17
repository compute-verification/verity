module

public import Properties.Verity.Game

/-! What the game reads of a composite as functions of its list of subprotocol ids, at every list and at
onsite, remote and remote's plain sibling. -/

@[expose] public section

namespace Verity

open Core

namespace Subprotocol

theorem mem_all (a : Subprotocol) : a ∈ Subprotocol.all := by cases a <;> decide

theorem ids_nodup (k : Verity.Registration.Composite) : (Subprotocol.all.map (·.idAt k)).Nodup := by
  by_cases h : PouwCheck.PearlPlus.Protocol.protocolId ∈ k <;> by_cases hr : rowsId ∈ k <;>
    simp only [Subprotocol.all, List.map_cons, List.map_nil, idAt, protocolAt, h, hr, ↓reduceIte, rowsCheck] <;>
    decide

theorem id_injective {k : Verity.Registration.Composite} {a b : Subprotocol} (h : a.idAt k = b.idAt k) : a = b := by
  exact List.inj_on_of_nodup_map (ids_nodup k) (mem_all a) (mem_all b) h

end Subprotocol

variable {k : Verity.Registration.Composite} {a : Subprotocol} {P : BlobProtocol}

theorem component_eq_some : component k a = some P ↔ a.idAt k ∈ k ∧ P = a.protocolAt k := by
  unfold component
  split
  next h => exact ⟨fun e => ⟨h, (Option.some.inj e).symm⟩, fun e => congrArg some e.2.symm⟩
  next h => exact ⟨fun e => (by cases e), fun e => absurd e.1 h⟩

theorem mem_subprotocols : a ∈ subprotocols k ↔ a.idAt k ∈ k := by
  unfold subprotocols component
  by_cases h : a.idAt k ∈ k <;> simp [h]

theorem component_of_mem (h : a ∈ subprotocols k) : component k a = some (a.protocolAt k) :=
  component_eq_some.2 ⟨mem_subprotocols.1 h, rfl⟩

theorem component_mem (h : component k a = some P) : P ∈ componentsOf k := by
  obtain ⟨hm, rfl⟩ := component_eq_some.1 h
  exact List.mem_map.2 ⟨a, List.mem_filter.2 ⟨a.mem_all, decide_eq_true hm⟩, rfl⟩

theorem mem_componentsOf (h : P ∈ componentsOf k) : ∃ a, component k a = some P := by
  obtain ⟨a, ha, rfl⟩ := List.mem_map.1 h
  exact ⟨a, component_eq_some.2 ⟨of_decide_eq_true (List.mem_filter.1 ha).2, rfl⟩⟩

theorem ids_nodup (k : Verity.Registration.Composite) : ((componentsOf k).map (·.id)).Nodup := by
  unfold componentsOf
  rw [List.map_map]
  exact (Subprotocol.ids_nodup k).sublist (List.filter_sublist.map _)

theorem component_of_id (hP : P ∈ componentsOf k) (hid : P.id = a.idAt k) : component k a = some P := by
  obtain ⟨b, hb⟩ := mem_componentsOf hP
  obtain ⟨-, rfl⟩ := component_eq_some.1 hb
  obtain rfl : b = a := Subprotocol.id_injective hid
  exact hb

theorem component_of_onBlobs {s : Sent} {Q : Protocol} (hQ : Q ∈ s.components k)
    (h : Q = (a.protocolAt k).onBlobs (Blobs.ofArray s.blobs)) : component k a = some (a.protocolAt k) := by
  obtain ⟨P, hP, rfl⟩ := List.mem_map.1 hQ
  have hc : component k a = some P := component_of_id hP (congrArg (·.id) h)
  exact hc.trans (congrArg some (component_eq_some.1 hc).2)

theorem componentsOf_onsite : componentsOf .onsite = Onsite.components := rfl

theorem componentsOf_remote : componentsOf .remote = Remote.V1.components := rfl

theorem ids_onsite : Onsite.components.map (·.id) = Verity.Registration.Composite.onsite := by decide

theorem ids_remote : Remote.V1.components.map (·.id) = Verity.Registration.Composite.remote := by decide

theorem tied_onsite : tied .onsite = Onsite.tied := rfl

theorem tied_remote : tied .remote = Remote.V1.tied := rfl

theorem ties_onsite : ties .onsite = Onsite.ties := rfl

theorem ties_remote : ties .remote = Remote.V1.ties := rfl

theorem verifier_onsite : verifier .onsite = Onsite.protocol := rfl

theorem verifier_remote : verifier .remote = Remote.V1.protocol := rfl

theorem ofIds_onsite : Composites.ofIds Verity.Registration.Composite.onsite = .ok (verifier .onsite) := rfl

theorem ofIds_remote : Composites.ofIds Verity.Registration.Composite.remote = .ok (verifier .remote) := rfl

theorem verifier_mem (hk : k = .onsite ∨ k = .remote) : (verifier k, k) ∈ Composites.composites := by
  rcases hk with rfl | rfl
  · exact List.mem_cons.2 (Or.inl rfl)
  · exact List.mem_cons.2 (Or.inr (List.mem_cons.2 (Or.inl rfl)))

theorem componentsOf_plain :
    componentsOf .plain = [rowsCheck, PouwCheck.PearlPlus.Protocol.protocol, VStar.Recursive.Sampled.protocol] := rfl

theorem ofIds_plain : Composites.ofIds Verity.Registration.Composite.plain = .error Composites.plainRefusal := rfl

end Verity
