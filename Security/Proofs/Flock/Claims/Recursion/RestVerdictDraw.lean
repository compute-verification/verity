import Proofs.Flock.Claims.Recursion.RestVerdictCustody
import Proofs.Flock.Claims.Complete.ZkTables.Stmt
import Proofs.Flock.Claims.JsonNorm

/-!
# `hrest`'s base checks with the draw tied at several tables

`DrawTied` holds at every statement of several tables (`drawTied_of_one`): there the setup sets the session's
`unitDraw` to the statement's draw (`FlockComplete.ZkTables.setupTables_spec`), which `Draw.UnitDraw.ofJson` reads, so
it is not `null` (`FlockComplete.ZkTables.ofJson_obj`), and which the verifier rebuilt (`Raw.ins_read`), so rebuilding it
again changes nothing (`norm_norm`). At one table the setup reads `unitDraw` back from the public
file's parsed header, and the draw is tied only where that header carries the statement's draw (`DrawTiedOne`).

`basesHold_of` is `basesHold` with custody proved (`custodyRecJ`) and the draw tied off one table.
-/

namespace RecDeploy

open Flock
open Lean (Json)
open Claims (Msg)
open FlockDeploy (FinalP setupOf drawEq)

/-- **At one table, the session's draw is the statement's**: `DrawTied` at the statements of one table. -/
def DrawTiedOne : Prop :=
  ∀ (x : Msg) (s : FlockDeploy.Ins) (spec : SessionSpec) (sts : Array Stmt), setupOf false x = some (s, spec, sts) →
    s.read.check = .ok () → sts.size = 1 → drawEq (drawOfU spec.unitDraw) s.read.draw = true

/-- **The session's draw is the statement's at several tables**, so `DrawTied` is `DrawTiedOne`. -/
theorem drawTied_of_one (h1 : DrawTiedOne) : DrawTied := by
  intro x s spec sts hx hq
  have hs := (FlockDeploy.setupOf_some hx).2
  have hpi := Claims.Bridge.Custody.publicInputs_none_of_check hq
  have h := FlockDeploy.Ins.setup_read hs
  unfold Readable.Read.setup at h
  rw [hpi] at h
  have hsz := (FlockComplete.ZkTables.setupTables_size h).1
  rcases Nat.lt_or_ge 1 sts.size with hgt | hle
  · obtain ⟨hud, -, hd, -⟩ := FlockComplete.ZkTables.setupTables_spec h hgt
    rw [hud]
    cases hdr : s.read.draw with
    | none => rfl
    | some d =>
      obtain ⟨D, hD, -⟩ := hd d hdr
      have hn := FlockComplete.ZkTables.ofJson_obj hD
      obtain ⟨w, -, hw⟩ := FlockComplete.JsonRT.insOf_false (FlockDeploy.setupOf_some hx).1
      obtain ⟨dr, -, hdr'⟩ := (FlockComplete.JsonRT.Raw.ins_read hw).2.2
      rw [hdr] at hdr'
      obtain ⟨d₀, -, rfl⟩ := Option.map_eq_some_iff.1 hdr'.symm
      have hnn := FlockDeploy.norm_norm d₀
      have e : drawOfU (some (Flock.norm d₀)) = some (Flock.norm d₀) := by
        cases h : Flock.norm d₀ with
        | null => exact absurd h hn
        | _ =>
          rw [h] at hnn
          exact congrArg some hnn
      rw [e]
      exact FlockDeploy.drawEq_refl _
  · exact h1 x s spec sts hx hq (by omega)

/-- **The base checks hold** where the reading fixes the last message's statement and refuses unshaped proofs, and the
session's draw is the statement's at one table. -/
theorem basesHold_of (rd : Msg → Option FinalP) (hstmt : DecStmt rd) (hsh : DecShaped rd) (h1 : DrawTiedOne) :
    BasesHold rd :=
  basesHold rd hstmt hsh (drawTied_of_one h1) (custodyRecJ rd)

end RecDeploy
