import Proofs.Flock.Claims.Complete.ZkTables.StmtOne
import Proofs.Flock.Claims.Recursion.RestVerdictCustody

/-!
# `hrest`'s base checks, at every reader

`basesHold_spec`: `BasesHold rd` at every reader `rd`, with no JSON premise. It
reads no witness and no prover: every fact is about `decR` and the statement's setup, so it holds at every decision of
`nextR`, adversarial files included.

* **The statement and the shapes** are `decR`'s refusals: it keeps the last message only at the statement
  (`FlockDeploy.decR_stmt`) and the proofs only where they are shaped for their files (`FlockDeploy.decR_shaped`).
* **Custody** at the reader's record is `custodyRecJ`.
* **The draw** (`drawTiedS`): at a statement that carries a draw (`parse_innerOk` gives one), the session's `unitDraw`
  is the statement's draw, which is not `null`. At several tables the setup sets it (`setupTables_spec`). At one table
  it is the public file's parsed header's `unit_draw`, which the setup put the draw into (`HmRow.drawn`), which the
  loaded header keeps as an object (`loadPublic_header`, at `unitDraw_of`); the header's draw decodes
  (`setupOne_stmt`), so it is not `null`.

`DrawTied` asks the tie at statements without a draw too, where at one table the header may carry any `unit_draw`;
`.after` reads it only where the statement carries one (`holdsR_afterS`, `DrawTiedS`).
-/

namespace FlockComplete.ZkTables

open Flock
open Lean (Json)
open FlockDeploy (setupOf drawEq)

open private setupOne_stmt unitDraw_of from Proofs.Flock.Claims.Complete.ZkTables.StmtOne

/-- **At a statement that carries a draw, the session's draw is it**, not `null`, at any number of tables. -/
theorem unitDraw_of_setup {x : Claims.Msg} {s : FlockDeploy.Ins}
    {spec : SessionSpec} {sts : Array Flock.Stmt} (hx : setupOf false x = some (s, spec, sts))
    (hq : s.read.check = .ok ()) {d : Json} (hdr : s.read.draw = some d) :
    spec.unitDraw = some d ∧ d ≠ Json.null := by
  have hs := (FlockDeploy.setupOf_some hx).2
  have hpi := Claims.Bridge.Custody.publicInputs_none_of_check hq
  have h := FlockDeploy.Ins.setup_read hs
  unfold Readable.Read.setup at h
  rw [hpi] at h
  have hsz := (setupTables_size h).1
  rcases Nat.lt_or_ge 1 sts.size with hgt | hle
  · obtain ⟨hud, -, hd, -⟩ := setupTables_spec h hgt
    obtain ⟨D, hD, -⟩ := hd d hdr
    exact ⟨hud.trans hdr, ofJson_obj hD⟩
  · have h1 : sts.size = 1 := by omega
    rcases FlockSoundness.Discharge.HiddenExec.setupTables_okAny h with ⟨-, st, hst, hspec, hsts⟩ | ⟨hc, hsz', -⟩
    · obtain ⟨⟨c, pop, hl, hd⟩, hinst, -, -⟩ := setupOne_stmt hst
      have hu := unitDraw_of hl (hd d hdr)
      obtain ⟨u, hu', -⟩ := hinst d hu
      refine ⟨?_, ofJson_obj hu'⟩
      rw [hspec]
      simp [Flock.Stmt.spec, hu, Except.toOption]
    · omega

end FlockComplete.ZkTables

namespace RecDeploy

open Flock
open Lean (Json)
open Claims (Msg)
open FlockDeploy (Coin FinalP VerdictR holdsR decR DecR actR checksAtJ drawOfJ drawEq setupOf)
open Claims.Recursion.VStarB

noncomputable section

/-- **The session's draw is the statement's, where the statement carries one**: `DrawTied` at the statements with a
draw. -/
def DrawTiedS : Prop :=
  ∀ (x : Msg) (s : FlockDeploy.Ins) (spec : SessionSpec) (sts : Array Stmt), setupOf false x = some (s, spec, sts) →
    s.read.check = .ok () → s.read.draw.isSome = true → drawEq (drawOfU spec.unitDraw) s.read.draw = true

/-- **`DrawTiedS` holds.** -/
theorem drawTiedS : DrawTiedS := by
  intro x s spec sts hx hq hn
  obtain ⟨d, hdr⟩ := Option.isSome_iff_exists.1 hn
  obtain ⟨hU, hnull⟩ := FlockComplete.ZkTables.unitDraw_of_setup hx hq hdr
  have hnn := (FlockDeploy.insOf_draw_norm (FlockDeploy.setupOf_some hx).1 hdr).1
  have e : drawOfU (some d) = some d := by
    cases d with
    | null => exact absurd rfl hnull
    | _ => exact congrArg some hnn
  rw [hU, hdr, e]
  exact FlockDeploy.drawEq_refl _

/-- **`.after` holds at every decision at SHA-512** on as many coins as messages, of a statement whose scope check
passes and which carries a draw, where the session's draw is the statement's there and custody passes at the reader's
record: `holdsR_after` at `DrawTiedS`. -/
theorem holdsR_afterS (rd : Msg → Option FinalP) (x : Msg) (as : List Msg) (cs : List Coin)
    (hlen : as.length = cs.length)
    (hck : ∀ s spec sts, setupOf false x = some (s, spec, sts) → s.read.check = .ok () ∧ s.read.draw.isSome = true)
    (hud : DrawTiedS) (hcu : CustodyRecJ rd) :
    holdsR FlockDeploy.sha512 rd .after x as cs := by
  unfold holdsR
  split
  · next s spec sts link c₁ mids f h =>
    obtain ⟨hx, hd⟩ := actR_decide_setup h
    intro spec' sts' hs'
    rw [setupWith_sha512_of hx] at hs'
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hs')
    obtain ⟨hq, hn⟩ := hck _ _ _ hx
    obtain ⟨lv, hL, hC⟩ := hcu x as cs _ hlen hd hq
    have hD : drawEq (drawOfJ f.j) s.read.draw = true := by
      obtain ⟨v, c₁', r₀, rss, fl, link', pf, -, -, -, -, -, -, he⟩ := FlockDeploy.decR_some hd
      obtain ⟨-, hspec, -, -, -, -, hf⟩ := DecR.mk.inj he
      rw [hf]
      dsimp only
      rw [FlockDeploy.recJ, drawOfJ_recordObj, ← hspec]
      exact drawEq_map_norm (fun e he => (FlockDeploy.insOf_draw_norm (FlockDeploy.setupOf_some hx).1 he).1)
        (hud _ _ _ _ hx hq hn)
    exact checksAtJ_ok_of (f := ⟨f.stmt, f.j, #[]⟩) (FlockDeploy.decR_stmt hd) hD hL hC hq hn
  · trivial

/-- **The reading fixes the last message's statement** (`decR_stmt`). -/
theorem decStmt (rd : Msg → Option FinalP) : DecStmt rd := fun _ _ _ _ h => FlockDeploy.decR_stmt h

/-- **The reading refuses proofs not shaped for their files** (`decR_shaped`). -/
theorem decShaped (rd : Msg → Option FinalP) : DecShaped rd := fun _ _ _ _ h => FlockDeploy.decR_shaped h

/-- **`hrest`'s base checks hold at every reader**: at the inner statement of
every compiled statement the outer reader parses, on as many coins as messages, each of the verdict's other checks
holds, whatever the messages. -/
theorem basesHold_spec (rd : Msg → Option FinalP) :
    BasesHold rd := by
  intro x st hx b hb as cs hlen
  have hck : ∀ s spec sts, setupOf false (xIOf x) = some (s, spec, sts) →
      s.read.check = .ok () ∧ s.read.draw.isSome = true := by
    intro s spec sts h
    obtain ⟨s', spec', sts', h', hq, hn⟩ := parse_innerOk hx
    rw [h] at h'
    obtain ⟨rfl, -⟩ := Prod.mk.inj (Option.some.inj h')
    exact ⟨hq, hn⟩
  simp only [VerdictP.bases, List.mem_cons, List.not_mem_nil, or_false] at hb
  rcases hb with rfl | rfl | rfl | rfl
  · exact holdsR_setup rd _ as cs
  · exact holdsR_shaped rd _ as cs (decShaped rd _ as cs)
  · exact holdsR_names rd _ as cs fun s spec sts h => (hck s spec sts h).1
  · exact holdsR_afterS rd _ as cs hlen hck drawTiedS (custodyRecJ rd)

end

end RecDeploy
