module

public import Proofs.Flock.SpotFlock.RunsAs.Outcome
import all Spot.Run
import all Spot.Record
import Proofs.Flock.Claims.Sampling.Select.Code

/-!
# The steps of an accepted two-stage run (part 5)

An accepted two-stage run of `Run.run` passed its first stage on the challenger's bytes with no closure map, its second
stage over the first stage's proved units, and round 2's statement check with no code (`accepted_steps`). The run's
answer names `accept` only where the boundary audit and round 2's attached record both read accepted; round 2's record
reads accepted only where its check has no code (`codes_of_attach`), and that check at the form's switch has no code
only where the check with the form allowed has none (`check_true_of_codes`).
-/

@[expose] public section

namespace SpotFlock.RunsAs

open Lean (Json)

/-- **What the verifier holds of the run as `Run.run` builds it** for round 2's check (`interiorAudit`), over the replay
sample `sample`, the interior's strata and the second stage's draw. -/
def runOwn (i : Spot.Run.Input) (it : Spot.Run.Interior) (sample : Array ℕ)
    (strata : Array (String × Array (ℕ × ℕ))) (draw : Json) : Spot.SecondStage.Own :=
  { sample, parts := Spot.SecondStage.parts sample.size strata, interior := it.record, boundary := i.record,
    drawn := (Spot.Draw.proved draw).map Int.toNat,
    links := Spot.SecondStage.replayLinks i.params.verifier.program.prog it.replay.prog.programFn i.record,
    reported := Spot.Record.sessionRegistered i.verdicts, verified := Spot.SecondStage.verifiedOf i.verdicts,
    refs := it.circuits.map fun (t, _, h, _) => (t, h), refReads := it.circuits.map fun (t, _, _, r) => (t, r),
    wires := Spot.SecondStage.replayWires i.params.verifier.program.prog it.replay.prog.programFn,
    rows := Spot.SecondStage.replayRows i.params.verifier.program.prog it.replay.prog.programFn,
    returns := Spot.SecondStage.replayReturns i.params.verifier.program.prog it.replay.prog.programFn i.record,
    supplied := it.params.verifier.supplied,
    fixed := it.params.verifier.pinned.map (·.1) ++ it.params.verifier.outputs.map (·.1) }

/-- An object `Json.mkObj` builds is a well-formed map. -/
theorem mkObj_wf (l : List (String × Json)) : ∃ m : Std.TreeMap.Raw String Json compare,
    Json.mkObj l = .obj m ∧ m.WF :=
  ⟨_, rfl, Std.TreeMap.Raw.WF.ofList⟩

/-- `getD` of `Json.mkObj l` at a key no entry of `l` has is `null`. -/
theorem getD_mkObj_none {l : List (String × Json)} {k : String} (hk : (l.map Prod.fst).contains k = false) :
    Spot.getD (Json.mkObj l) k = .null := by
  rw [Json.mkObj, getD_obj, Std.TreeMap.Raw.getElem?_ofList_of_contains_eq_false hk]
  all_goals rfl

/-- `getD` of `Json.mkObj l` at a key of `l`'s, its keys distinct, is its value. -/
theorem getD_mkObj_mem {l : List (String × Json)} {k : String} {v : Json}
    (hd : l.Pairwise fun a b => ¬compare a.1 b.1 = .eq) (hm : (k, v) ∈ l) : Spot.getD (Json.mkObj l) k = v := by
  rw [Json.mkObj, getD_obj, Std.TreeMap.Raw.getElem?_ofList_of_mem (Std.LawfulEqCmp.compare_eq_iff_eq.2 rfl) hd hm]
  all_goals rfl

/-- **A refused run names no symbol.** -/
theorem symbol_refused (step : String) (codes : Array String) :
    Spot.getD (Json.mkObj [("step", Json.str step), ("codes", Json.arr (codes.map Json.str))]) "symbol" = .null :=
  getD_mkObj_none (by simp)

/-- **An answer's symbol** is the one it lists, its other keys and the staged interior entry's distinct from it. -/
theorem symbol_answer (receipt draw read symbol pub audit : Json) (staged : List (String × Json))
    (hs : staged = [] ∨ ∃ e, staged = [("interior", e)]) :
    Spot.getD (Json.mkObj ([("receipt", receipt), ("draw", draw), ("read", read), ("symbol", symbol),
      ("public", pub), ("audit", audit)] ++ staged)) "symbol" = symbol := by
  apply getD_mkObj_mem
  · rcases hs with rfl | ⟨e, rfl⟩ <;>
      simp [List.pairwise_cons, Std.compare_eq_iff_eq]
  · simp

-- `run` unfolds to about 50 decided leaves under one `match` or `if` per step, hence the heartbeat budget.
set_option maxHeartbeats 8000000 in
/-- **A run's answer**: refused at a step, or past every step, its first stage on the challenger's bytes, its second
stage over the first stage's proved units, round 2's audit, and the answer naming `accept` where both audits read
accepted. -/
private theorem run_path {i : Spot.Run.Input} {j : Json} (h : Spot.Run.run i = .ok j) :
    (∃ (step : String) (codes : Array String),
      j = Json.mkObj [("step", Json.str step), ("codes", Json.arr (codes.map Json.str))]) ∨
    ∃ (workTable : Option Json) (resolved : Spot.Select.Rule) (draw : Json) (rest : Array ℕ) (read : ℕ),
      i.rule.tableOver i.params.verifier.program i.params.verifier.query i.params.verifier.workTable =
        .ok workTable ∧
      ¬i.closure.isSome = true ∧
      Spot.Select.resolve i.rule i.params.verifier.program i.params.verifier.query = .ok resolved ∧
      Spot.Select.select resolved i.params.verifier.program i.params.verifier.query workTable i.closure i.source =
        .ok (draw, rest, read) ∧
      ¬(!Spot.Run.coinsHold i.coins i.record i.params.verifier.windowRoot (i.source.extract 0 read)) = true ∧
      ∃ (inner : Option (Spot.Run.Interior × Json × Json × Json × ℕ)) (rec_ : Json) (acc₂ : Bool)
        (staged : List (String × Json)) (receipt pub audit : Json),
        Spot.Run.secondStage i ((Spot.Draw.proved draw).map Int.toNat) = .ok (.ok inner) ∧
        Spot.Run.interiorAudit i ((Spot.Draw.proved draw).map Int.toNat) inner = .ok (acc₂, staged) ∧
        j = Json.mkObj ([("receipt", receipt), ("draw", draw), ("read", Lean.toJson read),
          ("symbol", Json.str (if (Spot.jeq (Spot.getD rec_ "accepted") (.bool true) && acc₂) = true then "accept"
            else "reject")), ("public", pub), ("audit", audit)] ++ staged) := by
  unfold Spot.Run.run at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat'
    first
    | split at h
    | focus (cases h; done)
    | (obtain ⟨_, h⟩ | ⟨_, h⟩ := Spot.Run.ite_eq_ok h; exact Or.inl ⟨_, _, h.symm⟩)
    | exact Or.inl ⟨_, _, (Except.ok.inj h).symm⟩
    | exact Or.inr ⟨_, _, _, _, _, by assumption, by assumption, by assumption, by assumption, by assumption,
        _, _, _, _, _, _, _, by assumption, by assumption, (Except.ok.inj h).symm⟩

/-- **The second stage under a two-stage rule with an interior** is the interior's steps. -/
private theorem secondStage_two {i : Spot.Run.Input} {sample : Array ℕ} {grain : Json} {replay : String}
    {k : Spot.Select.Counts} {it : Spot.Run.Interior} {inner : Option (Spot.Run.Interior × Json × Json × Json × ℕ)}
    (hr : i.rule = .twoStage grain replay k) (hi : i.interior = some it)
    (h : Spot.Run.secondStage i sample = .ok (.ok inner)) :
    ∃ x, Spot.Run.interiorSteps i it k sample = .ok (.ok x) ∧ inner = some x := by
  unfold Spot.Run.secondStage at h
  rw [hr, hi] at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  cases hs : Spot.Run.interiorSteps i it k sample with
  | error e => simp [hs] at h
  | ok r =>
    cases r with
    | error e => simp [hs, Except.map] at h
    | ok x =>
      simp [hs, Except.map] at h
      exact ⟨x, rfl, by rw [← h]⟩

set_option maxHeartbeats 4000000 in
/-- **The interior's steps, accepted**: the interior is the run's, its cells and counts are the replay block's and the
rule's, its second stage is `selectSecond` on its bytes over the replay sample, and its coins hold. -/
private theorem interiorSteps_ok {i : Spot.Run.Input} {it : Spot.Run.Interior} {k : Spot.Select.Counts}
    {sample : Array ℕ} {x : Spot.Run.Interior × Json × Json × Json × ℕ}
    (h : Spot.Run.interiorSteps i it k sample = .ok (.ok x)) :
    x.1 = it ∧ ∃ ns ids ks, Spot.Select.replayCells it.replay it.params.verifier.query = .ok ns ∧
      Spot.Select.templateIds it.params.verifier.query = .ok ids ∧ k.over ids ns = .ok ks ∧
      Spot.Select.selectSecond sample ns ks it.source = .ok (x.2.2.1, x.2.2.2.2) ∧
      ¬(!Spot.Run.coinsHold it.coins it.record it.params.verifier.windowRoot
        (it.source.extract 0 x.2.2.2.2)) = true := by
  unfold Spot.Run.interiorSteps at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat'
    first
    | split at h
    | focus (cases h; done)
    | (cases h; exact ⟨rfl, _, _, _, by assumption, by assumption, by assumption, by assumption, by assumption⟩)

/-- **Round 2's audit of an interior**: the run's `Own` (`runOwn`) at the interior's strata, the audit record of the
second stage's draw, the answer whether round 2's attached record reads accepted, and one staged entry. -/
private theorem interiorAudit_some {i : Spot.Run.Input} {sample : Array ℕ} {it : Spot.Run.Interior}
    {receipt draw law : Json} {read : ℕ} {acc : Bool} {staged : List (String × Json)}
    (h : Spot.Run.interiorAudit i sample (some (it, receipt, draw, law, read)) = .ok (acc, staged)) :
    (∃ e, staged = [("interior", e)]) ∧ ∃ rec_ strata,
      Spot.Run.audit i it.record receipt draw law (it.provedUnits.getD (Spot.Draw.proved draw))
        (it.instances.getD (it.provedUnits.getD (Spot.Draw.proved draw)).size) it.served it.sessionRoots =
          .ok rec_ ∧
      Spot.Partition.strata it.params.verifier.program it.params.verifier.query = .ok strata ∧
      acc = Spot.jeq (Spot.getD (Spot.SecondStage.attach rec_
        (Spot.SecondStage.check Spot.SecondStage.FORM (runOwn i it sample strata draw) it.statements)
        (Spot.SecondStage.interiorReads (runOwn i it sample strata draw) ((it.statements.getD #[]).toList)))
        "accepted") (.bool true) := by
  unfold Spot.Run.interiorAudit at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat'
    first
    | split at h
    | focus (cases h; done)
    | (simp only [Except.ok.injEq, Prod.mk.injEq] at h
       exact ⟨⟨_, h.2.symm⟩, _, _, by assumption, by assumption, h.1.symm⟩)

set_option maxHeartbeats 4000000 in
/-- **A boundary record is a well-formed object**: `Record.record` answers with one. -/
private theorem record_wf {x : Spot.Record.Input} {r : Json} (h : Spot.Record.record x = .ok r) :
    ∃ m : Std.TreeMap.Raw String Json compare, r = .obj m ∧ m.WF := by
  unfold Spot.Record.record at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat'
    first
    | split at h
    | focus (cases h; done)
    | (cases h; exact ⟨_, rfl, Std.TreeMap.Raw.WF.ofList⟩)

/-- **An audit record is a well-formed object.** -/
private theorem audit_wf {i : Spot.Run.Input} {record receipt draw law : Json} {proved : Array Int} {instances : Int}
    {served sessionRoots r : Json}
    (h : Spot.Run.audit i record receipt draw law proved instances served sessionRoots = .ok r) :
    ∃ m : Std.TreeMap.Raw String Json compare, r = .obj m ∧ m.WF := by
  unfold Spot.Run.audit at h
  obtain ⟨x, _, h⟩ := Properties.Spot.SecurityProofs.bind_eq_ok.1 h
  exact record_wf h

/-- **Part 5: the steps of an accepted two-stage run.** Its first stage ran with no closure map on the challenger's
bytes under `Select.resolve`'s rule and the work table `Rule.tableOver` expands, with coins that hold; its second stage
ran on the interior's bytes over the units the first stage proves, at the replay block's cells and the rule's counts,
with coins that hold; and round 2's statement check with the form allowed reports no code at the `Own` the run builds
(`runOwn`). -/
theorem accepted_steps {i : Spot.Run.Input} {it : Spot.Run.Interior} {j : Json} (ha : AcceptedTwo i it j) :
    i.closure = none ∧
    ∃ (workTable : Option Json) (resolved : Spot.Select.Rule) (draw : Json) (rest : Array ℕ) (read : ℕ),
      i.rule.tableOver i.params.verifier.program i.params.verifier.query i.params.verifier.workTable =
        .ok workTable ∧
      Spot.Select.resolve i.rule i.params.verifier.program i.params.verifier.query = .ok resolved ∧
      Spot.Select.select resolved i.params.verifier.program i.params.verifier.query workTable i.closure i.source =
        .ok (draw, rest, read) ∧
      ¬(!Spot.Run.coinsHold i.coins i.record i.params.verifier.windowRoot (i.source.extract 0 read)) = true ∧
      ∃ (grain : Json) (replay : String) (k : Spot.Select.Counts) (ns ids ks : Array ℕ) (draw₂ : Json) (read₂ : ℕ)
        (strata : Array (String × Array (ℕ × ℕ))),
        i.rule = .twoStage grain replay k ∧
        Spot.Select.replayCells it.replay it.params.verifier.query = .ok ns ∧
        Spot.Select.templateIds it.params.verifier.query = .ok ids ∧ k.over ids ns = .ok ks ∧
        Spot.Select.selectSecond ((Spot.Draw.proved draw).map Int.toNat) ns ks it.source = .ok (draw₂, read₂) ∧
        ¬(!Spot.Run.coinsHold it.coins it.record it.params.verifier.windowRoot (it.source.extract 0 read₂)) = true ∧
        Spot.Partition.strata it.params.verifier.program it.params.verifier.query = .ok strata ∧
        (Spot.SecondStage.check true (runOwn i it ((Spot.Draw.proved draw).map Int.toNat) strata draw₂)
          it.statements).codes = #[] := by
  obtain ⟨grain, replay, k, hr⟩ := ha.two
  rcases run_path ha.run with ⟨step, codes, hj⟩ | ⟨workTable, resolved, draw, rest, read, ht, hc, hres, hsel, hcoins,
      inner, rec_, acc₂, staged, receipt, pub, audit, hsec, hia, hj⟩
  · have hs := ha.accept
    rw [hj, symbol_refused] at hs
    cases hs
  · obtain ⟨x, hx, rfl⟩ := secondStage_two hr ha.interior hsec
    obtain ⟨hx1, ns, ids, ks, hns, hids, hks, hss, hcoins₂⟩ := interiorSteps_ok hx
    obtain ⟨it', receipt₂, draw₂, law₂, read₂⟩ := x
    simp only at hx1
    subst hx1
    obtain ⟨⟨e, he⟩, rec₂, strata, haud, hstrata, hacc⟩ := interiorAudit_some hia
    have hsym := ha.accept
    rw [hj, symbol_answer _ _ _ _ _ _ _ (Or.inr ⟨e, he⟩)] at hsym
    have hboth : (Spot.jeq (Spot.getD rec_ "accepted") (.bool true) && acc₂) = true := by
      by_contra hn
      rw [if_neg hn] at hsym
      simp at hsym
    have hacc₂ : acc₂ = true := (Bool.and_eq_true _ _).1 hboth |>.2
    obtain ⟨m, hm, hwf⟩ := audit_wf haud
    rw [hacc₂, hm] at hacc
    refine ⟨Option.not_isSome_iff_eq_none.1 hc, workTable, resolved, draw, rest, read, ht, hres, hsel, hcoins,
      grain, replay, k, ns, ids, ks, draw₂, read₂, strata, hr, hns, hids, hks, hss, hcoins₂, hstrata, ?_⟩
    exact check_true_of_codes (codes_of_attach hwf hacc.symm)

end SpotFlock.RunsAs
