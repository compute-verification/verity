import FlockSoundness.Discharge.Placed
import FlockSoundness.Discharge.ZkExec
import FlockSoundness.Discharge.ZkSession.Session
import FlockSoundness.Protocol.Discharge.ZkBind.Place

/-!
# `--zk` unit placements: each drawn unit's `UnitPlace` from setupH's accept

The compiled `--zk` count curve (`ZkLink.zk_flock_countC_execOS_cr`) takes each drawn unit's place in its `--zk`
table's statement (`place : ∀ S R u, u ∈ S → UnitPlace P (tabOfZ planZ tab S R u).S u`). This is the `--zk` form of
e2e-placed's `Placed.Sites.progPlaces_keyProg`, for plans of `TabZK` tables:

* `acceptsZK_accepted`: flock-verify `--zk`'s acceptance (`ZkExec.AcceptsZK`), on the untyped path without hidden
  outputs, gives for every table of the session an `Placed.Accepted` statement: `Stmt.setupH`'s run under the table's
  `--zk` tags (`tagsAt (zkTags I) I.count j`), on the session's circuit file and tables, at any regions.
* `SitesZ`: each drawn unit's site, an acceptance whose statement is its `--zk` table's, with its VU and block.
* `SitesZ.placeKey`: the place of each drawn unit of the circuit's program on Δ's source keys (`Placed.keyProg`),
  from the sites and the parse alone: its rows the program's (W6 from the parse), its instance the program's own
  (`Const.Prog.instC`), its block the site's (`SitesZ.placeKey_spec`), aliased as Δ copies.
-/

open Flock FlockLevel3

namespace FlockSoundness.Discharge.ZkBind

open Model Audit Placed ZkSession Exec ZkExec

/-! ## Every table of a `--zk` acceptance is an accepted untyped statement -/

/-- **flock-verify `--zk`'s acceptance gives an `Accepted` statement per table**: on the untyped path without hidden
outputs, every table `j` of the session passes `Zk.shapeOk`, and at any regions there is an acceptance of `Stmt.setupH`
under the table's `--zk` tags on the session's circuit file and tables, whose run returned the session's `j`-th
statement (at the record's draw, for one table). -/
theorem acceptsZK_accepted {I : Inputs} {t : Exec.Transcript} (hh : I.tags.hiddenOutputs = false)
    (ht : I.tags.typed = false) (h : AcceptsZK I t) :
    ∃ spec sts, setupOf (zkInputs I) t.record = .ok (spec, sts) ∧
      ∀ (j : ℕ) (hj : j < sts.size), Flock.Zk.shapeOk sts[j] = .ok () ∧
        ∀ regs : List (Model.Region sts[j].c.kLog (sts[j].m - sts[j].c.kLog)),
          ∃ (T : Model.Statement) (a : Accepted T), a.st = sts[j] ∧ HEq a.regs regs ∧
            a.tags = tagsAt (zkTags I) I.count j ∧ a.cf = I.circuitFile ∧ a.tables = I.tables ∧
            (I.count ≤ 1 → a.draw = recordDraw t.record) := by
  obtain ⟨spec, sts, hs, -, -, hall⟩ := acceptsZK_setupH I t hh h
  refine ⟨spec, sts, hs, fun j hj => ⟨(hall j hj).1, fun regs => ?_⟩⟩
  obtain ⟨d, hd, hdr⟩ := (hall j hj).2
  have ht' : (tagsAt (zkTags I) I.count j).typed = false :=
    (tagsAt_typed _ _ _).trans ((zkTags_typed I).trans ht)
  exact ⟨_, ⟨tagsAt (zkTags I) I.count j, I.circuitFile, I.publicFile, zkCoins I, I.tables, d, I.partition,
    I.program, sts[j], ht', hd, regs, rfl⟩, rfl, HEq.rfl, rfl, rfl, rfl, hdr⟩

/-! ## Each drawn unit's site in its `--zk` table -/

section Sites

variable {F : Type} [Field F] [Fintype F] [DecidableEq F] {n : ℕ} {Reg : Type} {mPts : ℕ}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)

variable {planZ tab} {c : Flock.Circuit} (hU : Built c.unit) (si : SitesZ planZ tab)
  (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
  (outs : Finset (Fin (gates (inGates n (circuitRows c hU).nIn) (circuitRows c hU) n)))

/-- **`placeKey` is the program's unit at its site**: its rows are the program's, its instance the program's own
(`Const.Prog.instC`), its block the site's. -/
theorem SitesZ.placeKey_spec (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    (si.placeKey hU hparse outs S R u h).R = (keyProg c hU n).rows u ∧
      HEq (si.placeKey hU hparse outs S R u h).inst (Const.Prog.instC (keyProg c hU n) outs u) ∧
      (si.placeKey hU hparse outs S R u h).o = si.o S R u h := by
  exact ⟨rfl, HEq.rfl, rfl⟩

end Sites

end FlockSoundness.Discharge.ZkBind
