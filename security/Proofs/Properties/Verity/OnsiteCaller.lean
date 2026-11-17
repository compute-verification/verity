module

public import Properties.Verity.Game
public import Proofs.Properties.Remote
public import Proofs.Properties.Verity.Soundness

/-! SPOT's case in an accepted composite keeps the caller's facts, read off the caller's check's emptiness. -/

@[expose] public section

namespace Verity

open ResourceAccounting Core

theorem stageHeld_of_accepts (hm : Properties.Remote.CallerCheckSound) {k : Verity.Registration.Composite} {s : Sent}
    (h : Accepts k s) (ha : Subprotocol.sampled ∈ subprotocols k) :
    ∃ ps c ev, Decoded s ps c ev ∧ ∃ x ∈ Properties.Composition.triples (s.components k) ps ev,
      x.1 = VStar.Recursive.Sampled.protocol.onBlobs (Blobs.ofArray s.blobs) ∧ Properties.Remote.StageHeld x.2.2 := by
  obtain ⟨ps, c, ev, hdec, pub, prof, ⟨⟨⟨-, -, hfst, hpar⟩, hcases, -⟩, -⟩, -⟩ := accepts_same h
  have hm' : VStar.Recursive.Sampled.protocol.onBlobs (Blobs.ofArray s.blobs) ∈
      (Properties.Composition.triples (s.components k) ps ev).map (·.1) := by
    rw [hfst]
    exact List.mem_map.2 ⟨_, component_mem (component_of_mem ha), rfl⟩
  obtain ⟨x, hx, hx1⟩ := List.mem_map.mp hm'
  have hacc := hcases x hx
  rw [hx1] at hacc
  exact ⟨ps, c, ev, hdec, x, hx, hx1, Properties.Remote.SecurityProofs.stage_held hm (hpar x hx) hacc⟩

theorem onsite_stageHeld {s : Sent} (h : Accepts .onsite s) :
    ∃ ps c ev, Decoded s ps c ev ∧ ∃ x ∈ Properties.Composition.triples (s.components .onsite) ps ev,
      x.1 = VStar.Recursive.Sampled.protocol.onBlobs (Blobs.ofArray s.blobs) ∧ Properties.Remote.StageHeld x.2.2 :=
  stageHeld_of_accepts Properties.Remote.SecurityProofs.callerCheckSound h (mem_subprotocols.2 (by decide))

end Verity
