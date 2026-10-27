import Proofs.Flock.Claims.Complete.Json.Stmt

/-!
# The session's draw at one table

At one table the session's `unitDraw` is the public file's header's (`Stmt.spec`). In scope, Flock's plain verifier sets
a drawn statement up through `setupOne`, which writes the draw into that header (`HmRow.drawn`), so the session's draw
is the statement's, and not `null` (`unitDraw_one`): what `stmtOk` shows at a good case, from the scope and the setup
alone.
-/

namespace FlockComplete.JsonRT

open Flock FlockDeploy
open Lean (Json JsonNumber)
open FlockSoundness FlockSoundness.Refine

/-- **In scope, at one table, the session's draw is the statement's.** -/
theorem unitDraw_one {rel : Readable.Parsed → Readable.Relation}
    {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt}
    (hsc : Claims.Flock.Scope (deployPlain rel) x) (hset : setupOf false x = some (s, spec, sts))
    (h1 : sts.size = 1) (hd : s.read.draw.isSome = true) :
    ∃ d, spec.unitDraw = some d ∧ s.read.draw = some d ∧ d ≠ Json.null := by
  obtain ⟨p, hp, hsc'⟩ := (Readable.scope_iff (deployPlain rel) rel (readOf false) rfl
    (fun _ _ h => readOf_zk h) x).1 hsc
  have hi := (setupOf_some hset).1
  unfold Readable.parseBytes readOf at hp
  rw [hi] at hp
  have ht := Readable.toRead_of_parse (r := s.read) hp
  obtain ⟨hc, -⟩ := (Readable.scope_ok_iff p).1 hsc'
  have hR := ((Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hc).1
  have hpi : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hR.2.2.2.2.2])
  have hs := Ins.setup_read (setupOf_some hset).2
  rw [← ht] at hs
  unfold Readable.Read.setup at hs
  rw [hpi] at hs
  obtain ⟨st, hone, hspec⟩ : ∃ st, Discharge.HiddenExec.setupOne s.read.tags s.read.I.circuitFile s.read.I.publicFile
      s.read.coins s.read.I.tables s.read.draw s.read.I.partition s.read.I.program = .ok st ∧ spec = st.spec := by
    rcases Discharge.HiddenExec.setupTables_okAny hs with ⟨-, st, hst, hsp, -⟩ | ⟨hc1, hsz, -⟩
    · exact ⟨st, by rw [← ht]; exact hst, hsp⟩
    · omega
  obtain ⟨⟨c', pop, hl, hdr⟩, hinst, -, -, -⟩ := setupOne_stmt hone
  obtain ⟨d, hdd⟩ := Option.isSome_iff_exists.1 hd
  have hu := unitDraw_of hl (hdr d hdd)
  have hU : spec.unitDraw = some d := by
    rw [hspec]
    simp [Stmt.spec, hu, Except.toOption]
  obtain ⟨D, hD, -⟩ := hinst d hu
  refine ⟨d, hU, hdd, fun hn => ?_⟩
  rw [hn] at hD
  cases hD

end FlockComplete.JsonRT
