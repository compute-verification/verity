module

public import Properties.Verity.Game
public import Proofs.Properties.Remote
public import Proofs.Properties.Verity.Soundness

/-!
# SPOT's case in an accepted composite keeps the caller's facts

SPOT's check holds none of the caller's facts (`Properties.Spot.CallerHeld`); SPOT's component reads SPOT's acceptance
through the caller's check (`VStar.Recursive.Caller.gate`), and `Properties.Remote.SecurityProofs.stage_held` reads the
facts off that check's emptiness (`Properties.Remote.CallerCheckSound`). A composite's acceptance holds each of its cases
to its component's check, at the parameters the composition names for it (`accepts_same`, whose fit carries each case's
parameters), so at every composite that lists SPOT the case SPOT's component accepted keeps the caller's facts at the
form its parameters name (`stageHeld_of_accepts`); at onsite on the proved caller's check, with no premise
(`onsite_stageHeld`).
-/

@[expose] public section

namespace Verity

open ResourceAccounting Core

/-- **A composite's acceptance holds SPOT's case to the caller's facts**: where composite `k`, which lists SPOT, accepts
what the prover sent, SPOT's case in it, at the parameters `k` names for it, keeps the caller's facts at the form those
parameters name (`Properties.Remote.StageHeld`), where the caller's check finds nothing only where the record keeps them
(`Properties.Remote.CallerCheckSound`). -/
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

/-- **Onsite's acceptance holds SPOT's case to the caller's facts**: `stageHeld_of_accepts` at onsite, which lists
SPOT, on the proved caller's check (`Properties.Remote.SecurityProofs.callerCheckSound`). -/
theorem onsite_stageHeld {s : Sent} (h : Accepts .onsite s) :
    ∃ ps c ev, Decoded s ps c ev ∧ ∃ x ∈ Properties.Composition.triples (s.components .onsite) ps ev,
      x.1 = VStar.Recursive.Sampled.protocol.onBlobs (Blobs.ofArray s.blobs) ∧ Properties.Remote.StageHeld x.2.2 :=
  stageHeld_of_accepts Properties.Remote.SecurityProofs.callerCheckSound h (mem_subprotocols.2 (by decide))

end Verity
