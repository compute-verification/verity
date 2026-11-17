module

public import Proofs.Flock.SpotFlock.RunsAs.Outcome
import all Lean.Data.Json.Basic
import Std.Data.TreeMap.Raw.Lemmas
import Std.Data.TreeMap.Raw.WF
public import Spot.Run
public import Spot.Record
import Proofs.Flock.Claims.Sampling.Select.Code

/-! An accepted two-stage run of `Run.run` passed its first stage on the challenger's bytes with no closure map, its
second stage over the first stage's proved units, and round 2's statement check with no code (`accepted_steps`). -/

@[expose] public section

namespace SpotFlock.RunsAs

open Lean (Json)

theorem mkObj_wf (l : List (String × Json)) : ∃ m : Std.TreeMap.Raw String Json compare,
    Json.mkObj l = .obj m ∧ m.WF :=
  ⟨_, rfl, Std.TreeMap.Raw.WF.ofList⟩

theorem getD_mkObj_none {l : List (String × Json)} {k : String} (hk : (l.map Prod.fst).contains k = false) :
    Spot.getD (Json.mkObj l) k = .null := by
  rw [Json.mkObj, getD_obj, Std.TreeMap.Raw.getElem?_ofList_of_contains_eq_false hk]
  all_goals rfl

theorem getD_mkObj_mem {l : List (String × Json)} {k : String} {v : Json}
    (hd : l.Pairwise fun a b => ¬compare a.1 b.1 = .eq) (hm : (k, v) ∈ l) : Spot.getD (Json.mkObj l) k = v := by
  rw [Json.mkObj, getD_obj, Std.TreeMap.Raw.getElem?_ofList_of_mem (Std.LawfulEqCmp.compare_eq_iff_eq.2 rfl) hd hm]
  all_goals rfl

theorem symbol_refused (step : String) (codes : Array String) :
    Spot.getD (Json.mkObj [("step", Json.str step), ("codes", Json.arr (codes.map Json.str))]) "symbol" = .null :=
  getD_mkObj_none (by simp)

theorem symbol_answer (receipt draw read symbol pub audit : Json) (staged : List (String × Json))
    (hs : staged = [] ∨ ∃ e, staged = [("interior", e)]) :
    Spot.getD (Json.mkObj ([("receipt", receipt), ("draw", draw), ("read", read), ("symbol", symbol),
      ("public", pub), ("audit", audit)] ++ staged)) "symbol" = symbol := by
  apply getD_mkObj_mem
  · rcases hs with rfl | ⟨e, rfl⟩ <;>
      simp [List.pairwise_cons, Std.compare_eq_iff_eq]
  · simp

private theorem of_not_not {b : Bool} (h : ¬(!b) = true) : b = true := by
  cases b
  · exact absurd rfl h
  · rfl

private theorem eq_of_not_bne {α : Type} [BEq α] [LawfulBEq α] {a b : α} (h : ¬(a != b) = true) : a = b := by
  cases hab : a == b
  · exact absurd (by simp [bne, hab]) h
  · exact eq_of_beq hab

private theorem none_of_forall {α : Type} {o : Option α} (h : ∀ c, o = some c → False) : o = none := by
  cases o with
  | none => rfl
  | some c => exact (h c rfl).elim

-- `run` unfolds to about 50 decided leaves under one `match` or `if` per step, hence the heartbeat budget.
set_option maxHeartbeats 8000000 in
theorem run_path {i : Spot.Run.Input} {form : Bool} {j : Json} (h : Spot.Run.run i form = .ok j) :
    (∃ (step : String) (codes : Array String),
      j = Json.mkObj [("step", Json.str step), ("codes", Json.arr (codes.map Json.str))]) ∨
    ∃ (n : ℕ) (rule : Spot.Select.Rule) (samplingDesign : Json) (workTable : Option Json) (resolved : Spot.Select.Rule)
        (draw : Json) (rest : Array ℕ) (read : ℕ),
      Spot.nat? (Spot.getD i.record "population") = some n ∧
      i.rule.over n (Spot.Select.strataSizes i.params.verifier.program i.params.verifier.query) = some rule ∧
      rule.samplingDesignOver n = .ok samplingDesign ∧
      i.rule.tableOver i.params.verifier.program i.params.verifier.query i.params.verifier.workTable =
        .ok workTable ∧
      ¬(!(Spot.Docket.check i.record { i.params.verifier with samplingDesign, workTable }).isEmpty) = true ∧
      ¬i.closure.isSome = true ∧
      Spot.Select.resolve i.rule i.params.verifier.program i.params.verifier.query = .ok resolved ∧
      Spot.Select.select resolved i.params.verifier.program i.params.verifier.query workTable i.closure i.source =
        .ok (draw, rest, read) ∧
      (∀ code, Spot.Run.coinsCode i.key i.main i.coins i.record i.params.verifier.windowRoot i.family
        (i.source.extract 0 read) = some code → False) ∧
      ∃ (inner : Option (Spot.Run.Interior × Json × Json × Json × ℕ)) (acc₂ : Bool)
        (staged : List (String × Json)) (receipt : Json) (sym : String) (pub audit : Json),
        Spot.Run.secondStage i ((Spot.Draw.proved draw).map Int.toNat) = .ok (.ok inner) ∧
        Spot.Run.interiorAudit form i ((Spot.Draw.proved draw).map Int.toNat) inner = .ok (acc₂, staged) ∧
        j = Json.mkObj ([("receipt", receipt), ("draw", draw), ("read", Lean.toJson read),
          ("symbol", Json.str sym), ("public", pub), ("audit", audit)] ++ staged) ∧
        (sym = "accept" → ∃ rec_, (Spot.jeq (Spot.getD rec_ "accepted") (.bool true) && acc₂) = true) := by
  unfold Spot.Run.run at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat'
    first
    | split at h
    | focus (cases h; done)
    | (obtain ⟨_, h⟩ | ⟨_, h⟩ := Spot.Run.ite_eq_ok h; exact Or.inl ⟨_, _, h.symm⟩)
    | exact Or.inl ⟨_, _, (Except.ok.inj h).symm⟩
    | exact Or.inr ⟨_, _, _, _, _, _, _, _, by assumption, by assumption, by assumption, by assumption,
        by assumption, by assumption, by assumption, by assumption, by assumption,
        _, _, _, _, _, _, _, by assumption, by assumption, (Except.ok.inj h).symm, fun _ => ⟨_, by assumption⟩⟩
    | exact Or.inr ⟨_, _, _, _, _, _, _, _, by assumption, by assumption, by assumption, by assumption,
        by assumption, by assumption, by assumption, by assumption, by assumption,
        _, _, _, _, _, _, _, by assumption, by assumption, (Except.ok.inj h).symm, fun hs => by simp at hs⟩

theorem secondStage_two {i : Spot.Run.Input} {sample : Array ℕ} {grain : Json} {replay : String}
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
theorem interiorSteps_ok {i : Spot.Run.Input} {it : Spot.Run.Interior} {k : Spot.Select.Counts}
    {sample : Array ℕ} {x : Spot.Run.Interior × Json × Json × Json × ℕ}
    (h : Spot.Run.interiorSteps i it k sample = .ok (.ok x)) :
    x.1 = it ∧ ∃ ns ids ks, Spot.Select.replayCells it.replay it.params.verifier.query = .ok ns ∧
      Spot.Select.templateIds it.params.verifier.query = .ok ids ∧ k.over ids ns = .ok ks ∧
      ¬(Spot.nat? (Spot.getD it.record "population") != some (sample.size * ns.toList.sum)) = true ∧
      ¬(!(Spot.Docket.check it.record { it.params.verifier with
        samplingDesign := Json.mkObj [("law", Json.str "subset"), ("k", Lean.toJson (sample.size * ns.toList.sum))] }).isEmpty) =
          true ∧
      Spot.Select.selectSecond sample ns ks it.source = .ok (x.2.2.1, x.2.2.2.2) ∧
      (∀ code, Spot.Run.coinsCode i.key i.main it.coins it.record it.params.verifier.windowRoot it.family
        (it.source.extract 0 x.2.2.2.2) = some code → False) := by
  unfold Spot.Run.interiorSteps at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat'
    first
    | split at h
    | focus (cases h; done)
    | (cases h; exact ⟨rfl, _, _, _, by assumption, by assumption, by assumption, by assumption, by assumption,
        by assumption, by assumption⟩)

private theorem interiorAudit_some {form : Bool} {i : Spot.Run.Input} {sample : Array ℕ} {it : Spot.Run.Interior}
    {receipt draw samplingDesign : Json} {read : ℕ} {acc : Bool} {staged : List (String × Json)}
    (h : Spot.Run.interiorAudit form i sample (some (it, receipt, draw, samplingDesign, read)) = .ok (acc, staged)) :
    (∃ e, staged = [("interior", e)]) ∧ ∃ rec_ strata,
      Spot.Run.audit i it.params.verifier it.record receipt draw samplingDesign
        (it.provedUnits.getD (Spot.Draw.proved draw))
        (it.instances.getD (it.provedUnits.getD (Spot.Draw.proved draw)).size) it.served it.sessionRoots
          it.params.windowValues = .ok rec_ ∧
      Spot.Partition.strata it.params.verifier.program it.params.verifier.query = .ok strata ∧
      acc = Spot.jeq (Spot.getD (Spot.SecondStage.attach rec_
        (Spot.SecondStage.checkAny form (runOwn i it sample strata draw) it.statements)
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
private theorem record_wf {x : Spot.Record.Input} {r : Json} (h : Spot.Record.record x = .ok r) :
    ∃ m : Std.TreeMap.Raw String Json compare, r = .obj m ∧ m.WF := by
  unfold Spot.Record.record at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat'
    first
    | split at h
    | focus (cases h; done)
    | (cases h; exact ⟨_, rfl, Std.TreeMap.Raw.WF.ofList⟩)

private theorem audit_wf {i : Spot.Run.Input} {v : Spot.Docket.Verifier} {record receipt draw samplingDesign : Json}
    {proved : Array Int} {instances : Int} {served sessionRoots r : Json} {named : Array String}
    (h : Spot.Run.audit i v record receipt draw samplingDesign proved instances served sessionRoots named = .ok r) :
    ∃ m : Std.TreeMap.Raw String Json compare, r = .obj m ∧ m.WF := by
  unfold Spot.Run.audit at h
  obtain ⟨x, _, h⟩ := Properties.Spot.SecurityProofs.bind_eq_ok.1 h
  exact record_wf h

theorem accepted_steps {i : Spot.Run.Input} {it : Spot.Run.Interior} {j : Json} (ha : AcceptedTwo i it j) :
    i.closure = none ∧
    ∃ (workTable : Option Json) (resolved : Spot.Select.Rule) (draw : Json) (rest : Array ℕ) (read : ℕ),
      i.rule.tableOver i.params.verifier.program i.params.verifier.query i.params.verifier.workTable =
        .ok workTable ∧
      Spot.Select.resolve i.rule i.params.verifier.program i.params.verifier.query = .ok resolved ∧
      Spot.Select.select resolved i.params.verifier.program i.params.verifier.query workTable i.closure i.source =
        .ok (draw, rest, read) ∧
      Spot.Run.coinsCode i.key i.main i.coins i.record i.params.verifier.windowRoot i.family
        (i.source.extract 0 read) = none ∧
      ∃ (grain : Json) (replay : String) (k : Spot.Select.Counts) (ns : Array ℕ) (ids : Array String) (ks : Array ℕ)
        (draw₂ : Json) (read₂ : ℕ) (strata : Array (String × Array (ℕ × ℕ))),
        i.rule = .twoStage grain replay k ∧
        Spot.Select.replayCells it.replay it.params.verifier.query = .ok ns ∧
        Spot.Select.templateIds it.params.verifier.query = .ok ids ∧ k.over ids ns = .ok ks ∧
        Spot.Select.selectSecond ((Spot.Draw.proved draw).map Int.toNat) ns ks it.source = .ok (draw₂, read₂) ∧
        Spot.Run.coinsCode i.key i.main it.coins it.record it.params.verifier.windowRoot it.family
          (it.source.extract 0 read₂) = none ∧
        Spot.Partition.strata it.params.verifier.program it.params.verifier.query = .ok strata ∧
        (Spot.SecondStage.check true (runOwn i it ((Spot.Draw.proved draw).map Int.toNat) strata draw₂)
          it.statements).codes = #[] := by
  obtain ⟨grain, replay, k, hr⟩ := ha.two
  rcases run_path ha.run with ⟨step, codes, hj⟩ | ⟨_, _, _, workTable, resolved, draw, rest, read, -, -, -, ht,
      -, hc, hres, hsel, hcoins, inner, acc₂, staged, receipt, sym, pub, audit, hsec, hia, hj, himp⟩
  · have hs := ha.accept
    rw [hj, symbol_refused] at hs
    cases hs
  · obtain ⟨x, hx, rfl⟩ := secondStage_two hr ha.interior hsec
    obtain ⟨hx1, ns, ids, ks, hns, hids, hks, -, -, hss, hcoins₂⟩ := interiorSteps_ok hx
    obtain ⟨it', receipt₂, draw₂, samplingDesign₂, read₂⟩ := x
    simp only at hx1
    subst hx1
    obtain ⟨⟨e, he⟩, rec₂, strata, haud, hstrata, hacc⟩ := interiorAudit_some hia
    have hsym := ha.accept
    rw [hj, symbol_answer _ _ _ _ _ _ _ (Or.inr ⟨e, he⟩)] at hsym
    obtain ⟨rec_, hboth⟩ := himp (Json.str.inj hsym)
    have hacc₂ : acc₂ = true := (Bool.and_eq_true_iff.1 hboth).2
    obtain ⟨m, hm, hwf⟩ := audit_wf haud
    rw [hacc₂, hm] at hacc
    rw [Spot.SecondStage.checkAny_eq_check (ha.sampledNone _ _ _)] at hacc
    refine ⟨Option.not_isSome_iff_eq_none.1 hc, workTable, resolved, draw, rest, read, ht, hres, hsel,
      none_of_forall hcoins, grain, replay, k, ns, ids, ks, draw₂, read₂, strata, hr, hns, hids, hks, hss,
      none_of_forall hcoins₂, hstrata, ?_⟩
    exact check_true_of_codes (codes_of_attach hwf hacc.symm)

theorem accepted_registered {i : Spot.Run.Input} {it : Spot.Run.Interior} {j : Json} (ha : AcceptedTwo i it j) :
    ∃ (n : ℕ) (rule : Spot.Select.Rule) (samplingDesign : Json) (workTable : Option Json) (resolved : Spot.Select.Rule)
        (draw : Json) (rest : Array ℕ) (read : ℕ) (ns : Array ℕ),
      Spot.nat? (Spot.getD i.record "population") = some n ∧
      i.rule.over n (Spot.Select.strataSizes i.params.verifier.program i.params.verifier.query) = some rule ∧
      rule.samplingDesignOver n = .ok samplingDesign ∧
      i.rule.tableOver i.params.verifier.program i.params.verifier.query i.params.verifier.workTable =
        .ok workTable ∧
      (Spot.Docket.check i.record { i.params.verifier with samplingDesign, workTable }).isEmpty = true ∧
      Spot.Select.resolve i.rule i.params.verifier.program i.params.verifier.query = .ok resolved ∧
      Spot.Select.select resolved i.params.verifier.program i.params.verifier.query workTable i.closure i.source =
        .ok (draw, rest, read) ∧
      Spot.Select.replayCells it.replay it.params.verifier.query = .ok ns ∧
      Spot.nat? (Spot.getD it.record "population") =
        some (((Spot.Draw.proved draw).map Int.toNat).size * ns.toList.sum) ∧
      (Spot.Docket.check it.record { it.params.verifier with
        samplingDesign := Json.mkObj [("law", Json.str "subset"),
          ("k", Lean.toJson (((Spot.Draw.proved draw).map Int.toNat).size * ns.toList.sum))] }).isEmpty = true := by
  obtain ⟨grain, replay, k, hr⟩ := ha.two
  rcases run_path ha.run with ⟨step, codes, hj⟩ | ⟨n, rule, samplingDesign, workTable, resolved, draw, rest, read, hn, hrule,
      hlaw, ht, hchk, -, hres, hsel, -, inner, acc₂, staged, receipt, sym, pub, audit, hsec, -, -, -⟩
  · have hs := ha.accept
    rw [hj, symbol_refused] at hs
    cases hs
  · obtain ⟨x, hx, rfl⟩ := secondStage_two hr ha.interior hsec
    obtain ⟨-, ns, ids, ks, hns, -, -, hpop, hchk₂, -, -⟩ := interiorSteps_ok hx
    exact ⟨n, rule, samplingDesign, workTable, resolved, draw, rest, read, ns, hn, hrule, hlaw, ht, of_not_not hchk, hres, hsel,
      hns, eq_of_not_bne hpop, of_not_not hchk₂⟩

theorem answer_sample {i : Spot.Run.Input} {it : Spot.Run.Interior} {j : Json} (ha : AcceptedTwo i it j) :
    ∃ (workTable : Option Json) (resolved : Spot.Select.Rule) (rest : Array ℕ) (read : ℕ),
      i.rule.tableOver i.params.verifier.program i.params.verifier.query i.params.verifier.workTable =
        .ok workTable ∧
      Spot.Select.resolve i.rule i.params.verifier.program i.params.verifier.query = .ok resolved ∧
      Spot.Select.select resolved i.params.verifier.program i.params.verifier.query workTable i.closure i.source =
        .ok (Spot.getD j "draw", rest, read) := by
  obtain ⟨grain, replay, k, hr⟩ := ha.two
  rcases run_path ha.run with ⟨step, codes, hj⟩ | ⟨_, _, _, workTable, resolved, draw, rest, read, -, -, -, ht,
      -, -, hres, hsel, -, inner, acc₂, staged, receipt, sym, pub, audit, hsec, hia, hj, -⟩
  · have hs := ha.accept
    rw [hj, symbol_refused] at hs
    cases hs
  · obtain ⟨x, -, rfl⟩ := secondStage_two hr ha.interior hsec
    obtain ⟨it', receipt₂, draw₂, samplingDesign₂, read₂⟩ := x
    obtain ⟨⟨e, he⟩, -⟩ := interiorAudit_some hia
    have hd : Spot.getD j "draw" = draw := by
      rw [hj, he]
      apply getD_mkObj_mem
      · simp [List.pairwise_cons, Std.compare_eq_iff_eq]
      · simp
    exact ⟨workTable, resolved, rest, read, ht, hres, by rw [hd]; exact hsel⟩

end SpotFlock.RunsAs
