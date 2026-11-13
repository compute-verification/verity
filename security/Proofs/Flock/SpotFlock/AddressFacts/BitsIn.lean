module

public import Proofs.Flock.SpotFlock.AddressFacts.WidthOf

/-!
# Each position's bit inside its port's row

`bitsInPorts_of`: `BitsInPorts`, which the width and the copy at the address map ask, from two conditions:

* `PortsOnce` (`WiringOk`): each member circuit names each of its ports once. The address map's port is the first
  port of a row's name (`portNamed`), so only then is it the port the position's row is.
* `BitsOwn`: each source bit's and each written bit's row is a port of its member's circuit and its bit is inside
  that port's `16 w` bits, a fact of the members' programs alone.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held)

/-- **Each commitment position's bit is inside a port of its row's name**: each source bit's and each written bit's
row names a port of its member's circuit, and its bit is inside that port's `16 w` bits. -/
def BitsOwn : Prop :=
  (∀ s ∈ sourceBits h.ms, ∃ q, q < (memberCircuit h s.2.1).ports.size ∧
      (memberCircuit h s.2.1).ports[q]!.name = s.2.2.row ∧
      s.2.2.bit < 16 * (memberCircuit h s.2.1).ports[q]!.words) ∧
    ∀ y ∈ writtenBits h.ms, ∃ q, q < (memberCircuit h y.2.1).ports.size ∧
      (memberCircuit h y.2.1).ports[q]!.name = y.2.2.row ∧
      y.2.2.bit < 16 * (memberCircuit h y.2.1).ports[q]!.words

variable {h}

/-- **Where a member names each port once, the address map's port of a port's name is that port.** -/
theorem portNamed_eq {t q : ℕ} {name : String} (hO : ((memberCircuit h t).ports.toList.map (·.name)).Nodup)
    (hq : q < (memberCircuit h t).ports.size) (hname : (memberCircuit h t).ports[q]!.name = name) :
    portNamed h t name = q := by
  obtain ⟨-, hlt, hn⟩ := portNamed_spec (h := h) ⟨q, hq, hname⟩
  rw [getElem!_pos (memberCircuit h t).ports _ hlt] at hn
  rw [getElem!_pos (memberCircuit h t).ports q hq] at hname
  have e1 : ((memberCircuit h t).ports.toList.map (·.name))[portNamed h t name]'(by simp [hlt]) =
      ((memberCircuit h t).ports.toList.map (·.name))[q]'(by simp [hq]) := by
    simp only [List.getElem_map, Array.getElem_toList]
    rw [hn, hname]
  exact (List.Nodup.getElem_inj_iff hO).1 e1

/-- A position's bit inside a port of its row's name is inside the address map's port of that name. -/
theorem bit_lt_portNamed {t : ℕ} {r : Spot.Bits.RowPos} (hO : ((memberCircuit h t).ports.toList.map (·.name)).Nodup)
    (hq : ∃ q, q < (memberCircuit h t).ports.size ∧ (memberCircuit h t).ports[q]!.name = r.row ∧
      r.bit < 16 * (memberCircuit h t).ports[q]!.words) :
    r.bit < 16 * ((memberCircuit h t).ports[portNamed h t r.row]!).words := by
  obtain ⟨q, hlt, hname, hbit⟩ := hq
  rw [portNamed_eq hO hlt hname]
  exact hbit

/-- **`BitsInPorts` from each member naming its ports once and each position's bit inside its own port.** -/
theorem bitsInPorts_of (hO : PortsOnce h) (hB : BitsOwn h) : BitsInPorts h := by
  refine ⟨fun s hs => ?_, fun y hy => ?_⟩
  · obtain ⟨x, t, r⟩ := s
    obtain ⟨ht, -⟩ := mem_sourceBits hs
    exact bit_lt_portNamed (hO t ht) (hB.1 (x, t, r) hs)
  · obtain ⟨x, t, r⟩ := y
    obtain ⟨ht, -⟩ := mem_writtenBits hy
    exact bit_lt_portNamed (hO t ht) (hB.2 (x, t, r) hy)

end SpotFlock
