module

public import Proofs.Flock.Claims.Complete.Json.RoundTrip
public import Proofs.Flock.Claims.DeployInputs
public import Proofs.Flock.Recursive.FirewallWire
public import Proofs.Properties.Verity.Sampled.Record.PartSame

/-!
# A printed partition object reads back alike

A session statement carries its partition object as the bytes of its text (`Flock.RecordText.printT`). The verifier's
parser reads the text of a printable object (`Claims.Json.Good`: every object a well-formed tree, every number an
integer) back as an object alike it (`FlockComplete.JsonRT.parse_print`). So `flock-verify`'s reading of the
statement's partition (`FlockDeploy.jsonOf`) is that object up to `PartSame` (`partition_readsBack`), which is what a
session's two readings of the partition may differ by (`Verity.Sampled.Record.SessionMatch`).
-/

@[expose] public section

namespace Flock.PartCong

open Lean (Json)

/-- **A printed partition object reads back alike**: the bytes of a printable object's text read, as `flock-verify`
reads a statement's partition, as an object alike it. -/
theorem partition_readsBack {o : Json} (h : Claims.Json.Good o) :
    ∃ P, (some (RecordText.printT o).toUTF8).mapM (FlockDeploy.jsonOf "partition") = .ok P ∧
      PartSame P (some o) := by
  obtain ⟨j', hp, hs⟩ := FlockComplete.JsonRT.parse_print o h
  rw [FlockComplete.printJ_eq] at hp
  refine ⟨some j', ?_, .inr ⟨j', o, rfl, rfl, hs.symm⟩⟩
  rw [Option.mapM_some]
  unfold FlockDeploy.jsonOf
  rw [FlockSoundness.Discharge.Recursion.FirewallWire.fromUTF8?_toUTF8]
  simp only [hp]
  rfl

end Flock.PartCong
