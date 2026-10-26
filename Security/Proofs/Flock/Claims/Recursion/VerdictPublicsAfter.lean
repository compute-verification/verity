import Proofs.Flock.Claims.Recursion.VerdictTargetL
import Proofs.Flock.Claims.Recursion.LastCallsR

/-!
# `NonVacuous`'s public-words target, from the `after` check

`Top.targetPublics`: wherever the `after` check passes at a decision, custody makes the record's public words the link's
(`custody_ok`), which are empty at every decision (`decR_publics`), and the decoder reads the record's public words as
`liveOfJ` does (`FlockDeploy.KS.decodeJWithK_publics`). So `TargetPublics` holds at every `P`.
-/

namespace RecDeploy.Top

open Claims (Msg)
open FlockDeploy (Coin FinalP)

/-- **`TargetPublics` holds**: the `after` check's custody makes the decoded public words the link's, which are empty
at a decision. -/
theorem targetPublics (rd : Msg → Option FinalP) (P : Msg → List Msg → List Coin → Prop) : TargetPublics rd P := by
  intro x ms cs hA D
  obtain ⟨-, hr, hb, -⟩ := hA
  obtain ⟨-, lv, hlv, hcu⟩ := Claims.Bridge.Custody.checksAtJ_ok (after_of_base hr hb D D.spec D.sts D.setup)
  have hd := D.decode
  rw [Flock.Record.decodeJWith_eq_decodeJWithK] at hd
  have e := (Claims.Bridge.Custody.custody_ok hcu).2.1
  rw [FlockDeploy.decR_publics D.dec] at e
  rw [FlockDeploy.KS.decodeJWithK_publics id _ hd hlv, ← Array.all_toList, e, List.all_eq_true]
  intro p hp
  obtain ⟨_, _, rfl⟩ := List.mem_map.1 hp
  rfl

end RecDeploy.Top
